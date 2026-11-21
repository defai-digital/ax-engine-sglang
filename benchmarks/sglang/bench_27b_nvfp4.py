"""Compare stock AR, stock DFlash2 and AX Engine on frozen Qwen3.8-27B inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
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
LANES = (*DEFAULT_LANES, "replay8", "tree16", "chain8")
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
    prefill_size = "4096" if args.verify_parity else "512"
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", str(args.target.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port), "--attention-backend", "flashinfer",
        "--fp4-gemm-backend", "flashinfer_cutlass", "--max-running-requests", str(args.slots),
        "--cuda-graph-max-bs-decode", str(args.slots), "--context-length", str(args.context_length),
        "--max-total-tokens", str(args.kv_tokens), "--mem-fraction-static", str(args.mem_fraction_static),
        "--chunked-prefill-size", prefill_size, "--skip-server-warmup",
        "--disable-radix-cache", "--disable-flashinfer-autotune", "--disable-prefill-cuda-graph",
        "--random-seed", "42",
    ]
    flags = {"SGLANG_PLUGINS": "none", "SGLANG_FLASHINFER_WORKSPACE_SIZE": str(64 * 1024 * 1024)}
    if args.verify_parity:
        flags.update(SGLANG_PLUGINS="ax-engine", AX_NVFP4_VERIFY_PARITY="1", AX_NVFP4_PARITY_ROWS="0")
    if lane == "ar":
        return command, flags, None
    tree = lane in ("tree16", "tree32")
    custom_dflash = tree or lane == "chain8"
    algorithm = "FC1_DFLASH_TREE" if custom_dflash else "DFLASH"
    draft_block = "16" if tree else "8"
    command += ["--speculative-algorithm", algorithm, "--speculative-draft-model-path",
                str(args.draft.resolve()), "--speculative-num-draft-tokens", draft_block]
    if not custom_dflash:
        command += ["--mamba-ssm-dtype", "float32", "--linear-attn-verify-backend", "triton"]
    if lane == "replay8":
        flags.update(SGLANG_PLUGINS="ax-engine", AX_GDN_CHAIN_REPLAY="1", AX_FI_OFFSET_FIX="0")
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


def generate(base_url, input_ids, length, speculative):
    body = {"input_ids": input_ids, "sampling_params": {
        "temperature": 0, "max_new_tokens": length, "ignore_eos": True,
    }}
    request = urllib.request.Request(f"{base_url}/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240 + length) as response:
        result = json.load(response)
    wall_s = time.perf_counter() - started
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
    return {"concurrency": concurrency, "wall_s": wall_s, "output_tokens": length * len(rows),
            "median_request_s": statistics.median(row["wall_s"] for row in rows), "rows": rows}


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


def run_lane(args, lane, prompts):
    out = args.out / lane
    out.mkdir(parents=True, exist_ok=True)
    (out / "passes.json").unlink(missing_ok=True)
    command, flags, algorithm = launch_config(args, lane)
    write_json(out / "launch.json", {"command": command, "flags": flags, "prompt_sha256": PROMPT_SHA256})
    env = {key: value for key, value in os.environ.items() if not key.startswith(("FC1_", "AX_", "SGLANG_"))}
    env.update(flags, HF_HUB_OFFLINE="1")
    base_url = f"http://127.0.0.1:{args.port}"
    speculative = algorithm is not None
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            info = wait_for_server(process, base_url, args.target, algorithm)
            if bool(info.get("enable_linear_replayssm_spec")) != (lane == "replay8"):
                raise RuntimeError(f"Unexpected ReplaySSM setting for {lane}")
            run_pass(base_url, [max(prompts, key=len)], 1, 32, speculative)
            run_pass(base_url, prompts[:args.slots], args.slots, 32, speculative)
            passes = []
            for concurrency, repeat in ((1, 1), *args.concurrent_passes, (1, 1)):
                result = run_pass(base_url, prompts * repeat, concurrency, args.output_tokens, speculative)
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
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lanes", nargs="+", choices=LANES)
    parser.add_argument("--verify-parity", action="store_true",
                        help="enable AX NVFP4 parity for AR, tree, and chain lanes")
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
    args.verify_parity = args.verify_parity or args.verified_profile   # the same lanes and comparison
    if args.lanes is None:
        args.lanes = (VERIFIED_LANES if args.verified_profile else PARITY_LANES if args.verify_parity
                      else DEFAULT_LANES)
    if args.verified_profile and any(lane not in VERIFIED_LANES for lane in args.lanes):
        parser.error("--verified-profile supports the ar and chain8 lanes: the profile refuses the tree route")
    if args.verify_parity and any(lane not in PARITY_LANES for lane in args.lanes):
        parser.error("--verify-parity supports the ar, tree16, and chain8 lanes")
    if not args.verify_parity and "chain8" in args.lanes:
        parser.error("chain8 requires --verify-parity or --verified-profile")
    raw_prompts = args.input_ids.read_bytes()
    if hashlib.sha256(raw_prompts).hexdigest() != PROMPT_SHA256:
        parser.error("--input-ids must be the published parity-expanded-ids.json asset")
    prompts = json.loads(raw_prompts)
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
    if args.verify_parity:
        reference_lane = "ar" if "ar" in runs else args.lanes[0]
        reference = [row["output_ids"] for row in runs[reference_lane][0]["rows"]]
        failed = False
        for lane, passes in runs.items():
            for result in passes:
                rows = result["rows"]
                matches = sum(row["output_ids"] == reference[i % len(reference)] for i, row in enumerate(rows))
                print(f"{lane} C{result['concurrency']} against {reference_lane} C1: {matches}/{len(rows)} exact")
                failed |= matches != len(rows)
        if failed:
            raise SystemExit("Greedy token sequences differ; see the saved passes for each lane")


if __name__ == "__main__":
    main()
