"""Compare stock AR, stock DFlash2 and AX Engine on frozen Qwen3.8-27B inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from math import ceil
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request


DEFAULT_LANES = ("ar", "stock8", "tree32")
PARITY_LANES = ("ar", "tree16", "chain8")
VERIFIED_LANES = ("ar", "chain8")      # the routes AX_QUALITY_PROFILE=verified admits
AX_DRAFT_PARITY_LANES = ("ar", "replay8", "axdraft8")
KV_GRAPH_LANE = "replay8-kv-graph"
KV_DIRECT_LANE = "replay8-kv-direct"
PLAN_REUSE_LANE = "replay8-plan-reuse"
HOTPATHS = {
    "replay8": ("gdn_replay", None),
    "axdraft8": ("gdn_replay", None),
    "replay16": ("gdn_replay", None),     # experimental: replay with a 16-token draft block
    KV_GRAPH_LANE: ("kv_graph", "AX_DFLASH_KV_GRAPH"),
    KV_DIRECT_LANE: ("kv_direct", "AX_DFLASH_KV_DIRECT"),
    PLAN_REUSE_LANE: ("plan_reuse", "AX_DFLASH_PLAN_REUSE"),
}
REPLAY_LANES = tuple(HOTPATHS)
# Tree lanes draft a 16-token block and verify a tree of N nodes (FC1_TREE_N); tree32 is the default route.
# The fused tree kernels (FC1_FUSED=1) take 16, 32 or 64 nodes.
TREE_LANES = ("tree16", "tree32", "tree64")
LANES = (*DEFAULT_LANES, *REPLAY_LANES, "stock16", *(lane for lane in TREE_LANES if lane != "tree32"), "chain8")
PROMPT_SHA256 = "e2b5edfeb7a18b4b01c2a67501a6659fe06eeb3ff2f98e95ef3cd957e17a7fe0"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verified_launch(args, lane):
    """AX_QUALITY_PROFILE=verified: only the lane's route is given; the launcher derives the rest."""
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", str(args.target.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port), "--max-running-requests", str(args.slots),
        "--cuda-graph-max-bs-decode", str(args.slots), "--context-length", str(args.context_length),
        "--max-total-tokens", str(args.kv_tokens), "--mem-fraction-static", str(args.mem_fraction_static),
        "--skip-server-warmup", "--random-seed", "42",
    ]
    flags = {"SGLANG_PLUGINS": "ax-engine", "AX_QUALITY_PROFILE": "verified",
             "SGLANG_FLASHINFER_WORKSPACE_SIZE": str(64 * 1024 * 1024)}
    if lane == "ar":
        return command, flags, None
    flags["AX_TREE"] = "0"
    command += ["--speculative-algorithm", "FC1_DFLASH_TREE", "--speculative-draft-model-path",
                str(args.draft.resolve()), "--speculative-num-draft-tokens", "8"]
    return command, flags, "FC1_DFLASH_TREE"


def launch_config(args, lane):
    if args.verified_profile:
        return verified_launch(args, lane)
    # Fit the entire frozen prompt in one prefill pass for the parity comparison.
    ax_parity = getattr(args, "ax_draft_parity", False)
    prefill_size = "4096" if args.verify_parity or ax_parity else "512"
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", str(args.target.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port), "--attention-backend", "flashinfer",
        "--fp4-gemm-backend", "flashinfer_cutlass", "--max-running-requests", str(args.slots),
        "--cuda-graph-max-bs-decode", str(args.slots), "--context-length", str(args.context_length),
        "--max-total-tokens", str(args.kv_tokens), "--mem-fraction-static", str(args.mem_fraction_static),
        "--chunked-prefill-size", prefill_size, "--skip-server-warmup",
        "--disable-radix-cache", "--disable-flashinfer-autotune", "--random-seed", "42",
    ]
    # The parity lanes were matched with eager prefills. Speed lanes keep SGLang's prefill CUDA graphs:
    # eager prefills cost the 27B routes 7-16% at one client and 16-34% at eight on an RTX PRO 6000
    # (results/rtx_pro_6000_prefill_graph); reports before it ran with the flag.
    if args.verify_parity or ax_parity or os.environ.get("AX_BENCH_EAGER_PREFILL") == "1":
        command.append("--disable-prefill-cuda-graph")
    if ax_parity:
        command += ["--mamba-ssm-dtype", "float32", "--kv-cache-dtype", "fp8_e4m3"]
        if not args.overlap:
            command.append("--disable-overlap-schedule")
        if args.eager_decode:
            command.append("--disable-cuda-graph")
    flags = {"SGLANG_PLUGINS": "none", "SGLANG_FLASHINFER_WORKSPACE_SIZE": str(64 * 1024 * 1024)}
    if args.verify_parity:
        flags.update(SGLANG_PLUGINS="ax-engine", AX_NVFP4_VERIFY_PARITY="1", AX_NVFP4_PARITY_ROWS="0")
    if lane == "ar":
        return command, flags, None
    tree = lane in TREE_LANES
    custom_dflash = tree or lane == "chain8"
    algorithm = "FC1_DFLASH_TREE" if custom_dflash else "DFLASH"
    draft_block = "16" if tree or lane in ("stock16", "replay16") else "8"
    draft = args.ax_draft if lane == "axdraft8" else args.draft
    command += ["--speculative-algorithm", algorithm, "--speculative-draft-model-path",
                str(draft.resolve()), "--speculative-num-draft-tokens", draft_block]
    if not custom_dflash:
        if not ax_parity:
            command += ["--mamba-ssm-dtype", "float32"]
        command += ["--linear-attn-verify-backend", "triton"]
    if lane in REPLAY_LANES:
        flags.update(SGLANG_PLUGINS="ax-engine", AX_GDN_CHAIN_REPLAY="1", AX_FI_OFFSET_FIX="0",
                     AX_DFLASH_BENCH_ACTIVITY="1")
        _, flag = HOTPATHS[lane]
        if flag:
            flags[flag] = "1"
        if lane == "axdraft8":
            flags["AX_DRAFT_MODEL"] = str(draft.resolve())
    if custom_dflash:
        flags.update(
            SGLANG_PLUGINS="ax-engine", AX_NVFP4_VERIFY_PARITY="1" if args.verify_parity else "0",
            AX_FAST="1", AX_LATTICE="1", AX_GDN_LAYERS="48", AX_GDN_HEADS="48", FC1_TREE64="1",
            FC1_TREE_N=lane[4:] if tree else "8", FC1_DRAFT_BLOCK=draft_block,
            FC1_BRANCH_PILOT="1", FC1_OURS="1", FC1_FUSED="0" if args.verify_parity else "1",
            FC1_COMPILE_BUILD="1", AX_TREE="1" if tree else "0", AX_CHAIN_WIDTH="8",
        )
        if tree:
            flags.update(AX_CHAIN_GRAPH_WIDTHS="8", AX_CHAIN_DRAFT_WIDTHS="8", AX_SSM_POSITIONS="8")
        if args.verify_parity:
            flags["AX_GDN_RECURRENT_VERIFY"] = "1"
    return command, flags, algorithm


def wait_for_server(process, base_url, model_path, algorithm):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/get_server_info", timeout=2) as response:
                info = json.load(response)
        except OSError:
            time.sleep(1)
            continue
        if (info.get("model_path"), info.get("speculative_algorithm")) != (str(model_path.resolve()), algorithm):
            raise RuntimeError(f"Unexpected server at {base_url}")
        return info
    raise TimeoutError("SGLang startup timed out")


def request_generation(base_url, input_ids, length):
    body = {"input_ids": input_ids, "sampling_params": {
        "temperature": 0, "max_new_tokens": length, "ignore_eos": True,
    }}
    request = urllib.request.Request(f"{base_url}/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240 + length) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def generate(base_url, input_ids, length, speculative):
    result, elapsed = request_generation(base_url, input_ids, length)
    return generation_row(result, elapsed, length, speculative)


def generation_row(result, wall_s, length, speculative):
    output_ids = result["output_ids"]
    meta = result["meta_info"]
    if len(output_ids) != length or meta["finish_reason"]["type"] != "length":
        raise ValueError(f"Expected {length} output tokens, got {len(output_ids)} and {meta['finish_reason']}")
    verify_count = meta.get("spec_verify_ct")
    if speculative and not verify_count:
        raise ValueError("Speculative route reported no verification rounds")
    return {"wall_s": wall_s, "output_ids": output_ids, "spec_verify_ct": verify_count}


def run_pass(base_url, prompts, concurrency, length, speculative):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        start = time.perf_counter()
        # map preserves the frozen prompt order even when requests finish out of order.
        rows = list(pool.map(lambda ids: generate(base_url, ids, length, speculative), prompts))
        wall_s = time.perf_counter() - start
    return summarize_pass(rows, wall_s, concurrency, length, speculative)


def run_fixed_pass(base_url, prompts, concurrency, length, speculative):
    """Submit fixed request groups in order through SGLang's batch API."""
    start = time.perf_counter()
    rows = []
    for offset in range(0, len(prompts), concurrency):
        batch = prompts[offset:offset + concurrency]
        result, elapsed = request_generation(base_url, batch, length)
        if not isinstance(result, list) or len(result) != len(batch):
            raise ValueError(f"SGLang returned the wrong shape for {len(batch)} requests")
        rows.extend(generation_row(item, elapsed, length, speculative) for item in result)
    summary = summarize_pass(rows, time.perf_counter() - start, concurrency, length, speculative)
    summary["median_batch_s"] = summary.pop("median_request_s")
    summary["p95_batch_s"] = summary.pop("p95_request_s")
    return summary


def summarize_pass(rows, wall_s, concurrency, length, speculative):
    request_times = sorted(row["wall_s"] for row in rows)
    output_tokens = length * len(rows)
    return {"concurrency": concurrency, "wall_s": wall_s, "output_tokens": output_tokens,
            "median_request_s": statistics.median(request_times),
            "p95_request_s": request_times[ceil(0.95 * len(rows)) - 1],
            "tokens_per_verify": output_tokens / sum(row["spec_verify_ct"] for row in rows)
            if speculative else None, "rows": rows}


def stop_server(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def hotpath_snapshot(base_url):
    with urllib.request.urlopen(f"{base_url}/server_info", timeout=10) as response:
        states = json.load(response)["internal_states"]
    if len(states) != 1 or "ax_dflash_activity" not in states[0]:
        raise RuntimeError("DFlash hot-path accounting requires one instrumented scheduler")
    return states[0]["ax_dflash_activity"]


def hotpath_pass(base_url, path, concurrency, measure):
    while True:
        before = hotpath_snapshot(base_url)
        result = measure()
        after = hotpath_snapshot(base_url)
        if after["capture_attempts"] == before["capture_attempts"]:
            break  # discard passes affected by a first-time graph capture

    optimized_before = before["optimized"].get(path, {})
    optimized_after = after["optimized"].get(path, {})
    counts = {}
    for size, total in sorted(after["rounds"].items(), key=lambda item: int(item[0])):
        rounds = total - before["rounds"].get(size, 0)
        if rounds:
            optimized = optimized_after.get(size, 0) - optimized_before.get(size, 0)
            counts[size] = {"optimized": optimized, "fallback": rounds - optimized}
    requested = counts.get(str(concurrency), {})
    if not requested.get("optimized"):
        raise RuntimeError(f"DFlash {path} ran no optimized rounds at live batch size {concurrency} "
                           f"during the timed pass. Measured rounds: {counts}")
    if requested["fallback"]:
        raise RuntimeError(f"DFlash {path} fell back at live batch size {concurrency} during the timed pass. "
                           f"Measured rounds: {counts}")
    result["hotpath"] = {"path": path, "rounds_by_live_batch": counts}
    return result


def report_matches(label, reference_ids, rows):
    matches = 0
    earliest = None
    for request, (reference, row) in enumerate(zip(reference_ids, rows, strict=True), start=1):
        candidate = row["output_ids"]
        if reference == candidate:
            matches += 1
            continue
        difference = next((position, expected, actual) for position, (expected, actual) in enumerate(
            zip(reference, candidate, strict=True), start=1,
        ) if expected != actual)
        if earliest is None or difference[0] < earliest[0]:
            earliest = (*difference, request)

    detail = ""
    if earliest is not None:
        position, expected, actual, request = earliest
        detail = f", earliest difference at output token {position} of request {request} ({expected} vs {actual})"
    print(f"{label}: {matches}/{len(rows)} exact{detail}")
    return matches


def run_lane(args, lane, prompts):
    out = args.out / lane
    out.mkdir(parents=True, exist_ok=True)
    (out / "passes.json").unlink(missing_ok=True)
    command, flags, algorithm = launch_config(args, lane)
    write_json(out / "launch.json", {"command": command, "flags": flags,
                                    "prompt_sha256": getattr(args, "prompt_sha256", PROMPT_SHA256)})
    env = {key: value for key, value in os.environ.items() if not key.startswith(("FC1_", "AX_", "SGLANG_"))}
    env.update(flags, HF_HUB_OFFLINE="1")
    base_url = f"http://127.0.0.1:{args.port}"
    speculative = algorithm is not None
    measure = run_fixed_pass if getattr(args, "ax_draft_parity", False) else run_pass
    log_path = out / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            info = wait_for_server(process, base_url, args.target, algorithm)
            if bool(info.get("enable_linear_replayssm_spec")) != (lane in REPLAY_LANES):
                raise RuntimeError(f"Unexpected ReplaySSM setting for {lane}")
            measure(base_url, [max(prompts, key=len)], 1, 32, speculative)
            measure(base_url, prompts[:args.slots], args.slots, 32, speculative)
            passes = []
            for concurrency, repeat in ((1, 1), *args.concurrent_passes, (1, 1)):
                if lane in HOTPATHS:
                    result = hotpath_pass(base_url, HOTPATHS[lane][0], concurrency, lambda: measure(
                        base_url, prompts * repeat, concurrency, args.output_tokens, speculative,
                    ))
                else:
                    result = measure(base_url, prompts * repeat, concurrency, args.output_tokens, speculative)
                passes.append(result)
                print(f"{lane} C{concurrency}: {result['output_tokens'] / result['wall_s']:.2f} output tok/s",
                      flush=True)
            write_json(out / "passes.json", passes)
            return passes
        finally:
            stop_server(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--ax-draft", type=Path, help="AX Draft depth-seven checkpoint for axdraft8")
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lanes", nargs="+", choices=LANES)
    parser.add_argument("--verify-parity", action="store_true",
                        help="enable AX NVFP4 parity for AR, tree, and chain lanes")
    parser.add_argument("--ax-draft-parity", action="store_true",
                        help="compare fixed request groups against target-only greedy output in each pass")
    parser.add_argument("--overlap", action="store_true",
                        help="enable overlap in the AX Draft parity comparison")
    parser.add_argument("--eager-decode", action="store_true",
                        help="disable decode CUDA graphs in the AX Draft parity comparison")
    parser.add_argument("--verified-profile", action="store_true",
                        help="launch the ar and chain8 lanes with AX_QUALITY_PROFILE=verified alone")
    parser.add_argument("--slots", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--kv-tokens", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=128, help="tokens generated per request")
    parser.add_argument("--concurrency", type=int, nargs="+",
                        help="concurrency of the passes between the two serialized passes (default: --slots twice)")
    parser.add_argument("--repeat", type=int, default=4,
                        help="times each concurrent pass sends the prompt list")
    parser.add_argument("--mem-fraction-static", type=float, default=0.94)
    parser.add_argument("--port", type=int, default=30517)
    args = parser.parse_args()
    if any(not 1 <= value <= args.slots for value in args.concurrency or ()):
        parser.error("--concurrency values must be between 1 and --slots")
    args.concurrent_passes = [(value, args.repeat) for value in args.concurrency or (args.slots, args.slots)]
    if args.verify_parity and args.verified_profile:
        parser.error("--verify-parity and --verified-profile are separate runs")
    if args.ax_draft_parity and (args.verify_parity or args.verified_profile):
        parser.error("--ax-draft-parity is a separate comparison")
    if (args.overlap or args.eager_decode) and not args.ax_draft_parity:
        parser.error("--overlap and --eager-decode require --ax-draft-parity")
    args.verify_parity = args.verify_parity or args.verified_profile   # the same lanes and comparison
    if args.lanes is None:
        args.lanes = (VERIFIED_LANES if args.verified_profile else PARITY_LANES if args.verify_parity
                      else AX_DRAFT_PARITY_LANES if args.ax_draft_parity else DEFAULT_LANES)
    if args.ax_draft_parity and (set(args.lanes) != set(AX_DRAFT_PARITY_LANES)
                                or len(args.lanes) != len(AX_DRAFT_PARITY_LANES)):
        parser.error("--ax-draft-parity requires ar, replay8, and axdraft8")
    if args.verified_profile and any(lane not in VERIFIED_LANES for lane in args.lanes):
        parser.error("--verified-profile supports the ar and chain8 lanes: the profile refuses the tree route")
    if args.verified_profile and "ar" not in args.lanes:
        parser.error("--verified-profile requires the ar lane as the serialized reference")
    if args.verify_parity and any(lane not in PARITY_LANES for lane in args.lanes):
        parser.error("--verify-parity supports the ar, tree16, and chain8 lanes")
    if not args.verify_parity and "chain8" in args.lanes:
        parser.error("chain8 requires --verify-parity or --verified-profile")
    if "axdraft8" in args.lanes and args.ax_draft is None:
        parser.error("axdraft8 requires --ax-draft")
    raw_prompts = args.input_ids.read_bytes()
    args.prompt_sha256 = hashlib.sha256(raw_prompts).hexdigest()
    if not args.ax_draft_parity and args.prompt_sha256 != PROMPT_SHA256:
        parser.error("--input-ids must be the published parity-expanded-ids.json asset")
    prompts = json.loads(raw_prompts)
    if not isinstance(prompts, list) or not prompts or any(
        not isinstance(prompt, list) or not prompt
        or any(type(token) is not int or token < 0 for token in prompt) for prompt in prompts
    ):
        parser.error("--input-ids must contain a nonempty JSON array of nonempty token arrays")
    if args.ax_draft_parity and (
        args.repeat < 1 or any(len(prompts) * repeat % size for size, repeat in args.concurrent_passes)
    ):
        parser.error("AX Draft parity requires full fixed-size groups in each concurrent pass")
    lengths = sorted((len(prompt) for prompt in prompts), reverse=True)
    if lengths[0] + args.output_tokens > args.context_length:
        parser.error(f"--context-length must cover the longest prompt and its output "
                     f"({lengths[0] + args.output_tokens} tokens)")
    widest = max(concurrency for concurrency, _ in args.concurrent_passes)
    needed = sum(lengths[:widest]) + widest * args.output_tokens
    if needed > args.kv_tokens:
        # A smaller pool queues requests, and the pass would run below its stated concurrency.
        parser.error(f"--kv-tokens must be at least {needed} for {widest} concurrent requests "
                     f"of {args.output_tokens} output tokens")
    runs = {lane: run_lane(args, lane, prompts) for lane in args.lanes}
    if args.ax_draft_parity:
        failed = False
        # First check whether the batched target agrees with the serial target.
        serial = [row["output_ids"] for row in runs["ar"][0]["rows"]]
        for target, replay, ax_draft in zip(*(runs[lane] for lane in AX_DRAFT_PARITY_LANES), strict=True):
            concurrency = target["concurrency"]
            canonical = [row["output_ids"] for row in target["rows"]]
            failed |= report_matches(f"ar C{concurrency} against ar C1",
                                     (serial[i % len(serial)] for i in range(len(canonical))),
                                     target["rows"]) != len(canonical)
            for lane, result in (("replay8", replay), ("axdraft8", ax_draft)):
                failed |= report_matches(f"{lane} C{concurrency} against ar C{concurrency}",
                                         canonical, result["rows"]) != len(canonical)
        if failed:
            raise SystemExit("Greedy token sequences differ; see the saved passes for each lane")
        return
    comparisons = (("stock8", "replay8"), ("stock16", "replay16"),
                   *(("replay8", lane) for lane in REPLAY_LANES if lane not in ("replay8", "replay16")))
    for control, candidate in comparisons:
        if control not in runs or candidate not in runs:
            continue
        for reference, result in zip(runs[control], runs[candidate], strict=True):
            report_matches(f"{candidate} C{result['concurrency']} against {control}",
                           (row["output_ids"] for row in reference["rows"]), result["rows"])
    if args.verify_parity:
        reference_lane = "ar" if "ar" in runs else args.lanes[0]
        reference = [row["output_ids"] for row in runs[reference_lane][0]["rows"]]
        failed = False
        for lane, passes in runs.items():
            for result in passes:
                rows = result["rows"]
                matches = report_matches(f"{lane} C{result['concurrency']} against {reference_lane} C1",
                                         (reference[i % len(reference)] for i in range(len(rows))), rows)
                failed |= matches != len(rows)
        if failed:
            raise SystemExit("Greedy token sequences differ; see the saved passes for each lane")


if __name__ == "__main__":
    main()
