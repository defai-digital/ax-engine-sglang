"""Compare two plugin checkouts on one CUDA host with the same local checkpoints.

This is a bounded development correctness check, not a throughput qualification.
It owns and terminates only the server process groups it starts.
Use `compare-greedy --help` for offline comparison of saved lanes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


PROMPTS = (
    "Write a Python function that returns the largest value in a nonempty list.",
    "Explain why a Python dictionary cannot use a list as a key.",
    "What is 17 times 23? Show a short calculation.",
    "A train travels 120 km in 90 minutes. What is its average speed in km/h?",
    "Explain the difference between weather and climate in two sentences.",
    "Give three practical ways to reduce food waste at home.",
)

DFLASH_ROUTES = ("tree", "chain")
ROUTES = (*DFLASH_ROUTES, "ar", "mtp3", "mtp7", "mtp12")


def post(base, body):
    request = urllib.request.Request(base + "/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read())
    finish = (result.get("meta_info") or {}).get("finish_reason") or {}
    if not isinstance(result.get("output_ids"), list) or finish.get("type") not in ("stop", "length"):
        raise RuntimeError(f"incomplete generation: {finish!r}")
    return result["output_ids"]


def validate_token_sequences(label, sequences):
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"{label} must contain at least one token sequence")
    if any(not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids)
           for ids in sequences):
        raise ValueError(f"{label} must contain lists of non-negative integer token IDs")


def validate_prompts(prompts):
    validate_token_sequences("prompts", prompts)
    if any(not ids for ids in prompts):
        raise ValueError("prompt token sequences must be nonempty")


def greedy_comparison(prompts, reference, candidate):
    """Describe each first divergence on the conditioning prefix shared by both lanes."""
    validate_prompts(prompts)
    validate_token_sequences("reference", reference)
    validate_token_sequences("candidate", candidate)
    if len(prompts) != len(reference) or len(prompts) != len(candidate):
        raise ValueError("both lanes must contain an output for every input prompt")

    differences = []
    for index, (prompt, left, right) in enumerate(zip(prompts, reference, candidate)):
        if left == right:
            continue
        pos = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y),
                   min(len(left), len(right)))
        differences.append({
            "prompt_index": index,
            "output_position": pos,
            "prefix_ids": prompt + left[:pos],
            "reference_token_id": left[pos] if pos < len(left) else None,
            "candidate_token_id": right[pos] if pos < len(right) else None,
        })
    return {"passed": not differences, "prompts": len(prompts),
            "identical": len(prompts) - len(differences), "differences": differences}


def compare_saved_greedy(argv):
    parser = argparse.ArgumentParser(description="Compare saved greedy lanes without launching a server.")
    parser.add_argument("--reference", type=Path, required=True, help="reference lane directory")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate lane directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        ref_prompts = json.loads((args.reference / "input_ids.json").read_text())
        candidate_prompts = json.loads((args.candidate / "input_ids.json").read_text())
        # JSON booleans and floats can compare equal to integer token IDs in Python.
        if json.dumps(ref_prompts) != json.dumps(candidate_prompts):
            raise ValueError("the saved lanes have different input token IDs")
        reference, candidate = (
            json.loads((lane / "tokens.json").read_text()) for lane in (args.reference, args.candidate)
        )
        report = greedy_comparison(ref_prompts, reference, candidate)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Greedy token identity: {report['identical']}/{report['prompts']} prompts")
    return 0 if report["passed"] else 1


def source_digest(repo):
    digest = hashlib.sha256()
    for path in sorted((repo / "ax-engine-sglang/ax_engine_sglang").glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def sampling_mode(args):
    """Distribution sampling mode; server RNG cohorts need independent server seeds."""
    if not args.sampling_n:
        return "none"
    return "request" if args.sampling_seed >= 0 else "server_rng"


def lane_server_seed(args, name):
    """Server RNG seed for one lane; sampled server-RNG cohorts must not share a seed."""
    if sampling_mode(args) == "server_rng" and not args.candidate_only and name == "candidate":
        return args.server_seed + 1
    return args.server_seed


def lane_sampling(args, name):
    """Recorded sampling controls for one lane."""
    return {"seed_mode": sampling_mode(args), "requested_seed_base": args.sampling_seed,
            "server_random_seed": lane_server_seed(args, name)}


def lane_route(args, name):
    return args.baseline_route if name == "baseline" and args.baseline_route else args.route


def launch_command(args, name):
    server_seed = lane_server_seed(args, name)
    route = lane_route(args, name)
    command = [
        str(args.python), "-m", "sglang.launch_server", "--model-path", str(args.model),
        "--served-model-name", "qwen", "--attention-backend", args.attention_backend,
        "--max-running-requests", str(args.slots),
        "--cuda-graph-max-bs-decode", str(args.slots), "--context-length", str(args.context_limit),
        "--max-total-tokens", "8192", "--mem-fraction-static", str(args.mem_fraction),
        "--disable-radix-cache", "--disable-flashinfer-autotune", "--disable-prefill-cuda-graph",
        "--random-seed", str(server_seed),
        "--port", str(args.port),
    ]
    if route in DFLASH_ROUTES:
        command += ["--speculative-algorithm", "FC1_DFLASH_TREE",
                    "--speculative-draft-model-path", str(args.draft), "--speculative-num-draft-tokens", "16"]
    elif route.startswith("mtp"):
        steps = int(route[3:])
        command += ["--speculative-algorithm", "EAGLE", "--speculative-num-steps", str(steps),
                    "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(steps + 1)]
        if args.mtp_draft:
            command += ["--speculative-draft-model-path", str(args.mtp_draft)]
    if args.model_override:
        command += ["--json-model-override-args", args.model_override]
    if args.kv_cache_dtype:
        command += ["--kv-cache-dtype", args.kv_cache_dtype]
    if args.fp4_gemm_backend:
        command += ["--fp4-gemm-backend", args.fp4_gemm_backend]
    if args.gdn_recurrent_verify and route in DFLASH_ROUTES:
        command += ["--linear-attn-verify-backend", "triton"]
    if args.moe_runner_backend:
        command += ["--moe-runner-backend", args.moe_runner_backend]
    if args.speculative_moe_runner_backend:
        command += ["--speculative-moe-runner-backend", args.speculative_moe_runner_backend]
    if args.eager:
        command += ["--disable-cuda-graph"]
    if args.enable_deterministic_inference:
        command += ["--enable-deterministic-inference"]
    if args.prefetch_checkpoints:
        command += ["--weight-loader-prefetch-checkpoints", "--weight-loader-prefetch-num-threads", "2"]
    return command


def benchmark_script(repo, filename):
    """Resolve a benchmark helper in current and pre-reorganization checkouts."""
    current = repo / "benchmarks" / "sglang" / filename
    return current if current.is_file() else repo / "track_l" / filename


def sampling_collect_command(args, base, lane):
    return [
        str(args.python), str(benchmark_script(args.candidate, "sampling_dist_check.py")),
        "collect", "--base-url", base, "--model", str(args.model),
        "--n", str(args.sampling_n), "--tokens", "6", "--seed0", str(args.sampling_seed),
        "--out", str(lane / "sampled.jsonl"),
    ]


def sampling_compare_command(args):
    if sampling_mode(args) == "request":
        raise ValueError("paired request-seed cohorts are correlated; use independent server RNG cohorts")
    command = [
        str(args.python), str(benchmark_script(args.candidate, "sampling_dist_check.py")),
        "compare", str(args.out / "baseline/sampled.jsonl"),
        str(args.out / "candidate/sampled.jsonl"),
    ]
    if sampling_mode(args) == "server_rng":
        command.append("--require-independent-seeds")
    return command


def compare_sampling_cohorts(args):
    """Recorded sampling comparison status; 'passed' only after the dual-lane compare succeeds."""
    if not args.sampling_n:
        return "not_requested"
    if args.candidate_only:
        return "not_run_single_lane"
    subprocess.run(sampling_compare_command(args), check=True)
    return "passed"


def run_lane(args, name, repo, encoded):
    lane = args.out / name
    lane.mkdir()
    inputs = lane / "input_ids.json"
    inputs.write_text(json.dumps(encoded) + "\n")
    source_sha256 = source_digest(repo)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AX_", "FC1_"))}
    env.update({
        "PATH": str(args.python.parent) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": str(repo / "ax-engine-sglang"), "SGLANG_PLUGINS": "ax-engine",
        "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
    })
    route = lane_route(args, name)
    if route in DFLASH_ROUTES:
        narrower_width = str(args.chain_width) if args.chain_width < 16 else ""
        env.update({
            "AX_FAST": "1", "AX_LATTICE": "1",
            "AX_GDN_LAYERS": "48", "AX_GDN_HEADS": "48", "FC1_TREE64": "1",
            "FC1_TREE_N": "16", "FC1_DRAFT_BLOCK": "16",
            "FC1_FUSED": "0" if args.gdn_recurrent_verify else "1",
            "FC1_BRANCH_PILOT": "1", "FC1_OURS": "1", "AX_CHAIN_WIDTH": str(args.chain_width),
            "AX_CHAIN_GRAPH_WIDTHS": f"16 {narrower_width}".strip(), "AX_CHAIN_DRAFT_WIDTHS": narrower_width,
        })
        if route == "chain":
            env["AX_TREE"] = "0"
        if args.dflash_joint_sampling:
            env["AX_DFLASH_JOINT_SAMPLING"] = "1"
        if args.gdn_recurrent_verify:
            env["AX_GDN_RECURRENT_VERIFY"] = "1"
    if args.qsa_ring:
        env["AX_QSA_RING"] = str(args.qsa_ring)
    if args.qsa_check:
        env["AX_QSA_CHECK"] = "1"
    if args.qsa_stable_topk:
        env["AX_QSA_STABLE_TOPK"] = "1"
    if getattr(args, "nvfp4_verify_parity", False):
        env.update(AX_NVFP4_VERIFY_PARITY="1", SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV="0",
                   SGLANG_DISABLE_SILU_FP4_QUANT_FUSION="1", SGLANG_FLASHINFER_USE_PAGED="1")
        env["AX_NVFP4_PARITY_ROWS"] = str(args.nvfp4_parity_rows)
    if args.gdn_unpacked_decode:
        env["AX_GDN_UNPACKED_DECODE"] = "1"
    command = launch_command(args, name)
    sampling = lane_sampling(args, name)
    (lane / "launch.json").write_text(json.dumps({"command": command,
        "source_sha256": source_sha256, "route": route, "sampling": sampling, "plugin_env": {
        k: v for k, v in env.items() if k.startswith(("AX_", "FC1_")) or k in {
            "SGLANG_PLUGINS", "SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV",
            "SGLANG_DISABLE_SILU_FP4_QUANT_FUSION", "SGLANG_FLASHINFER_USE_PAGED"}}}, indent=2) + "\n")
    base = f"http://127.0.0.1:{args.port}"
    with (lane / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, cwd=repo, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        succeeded = False
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} server exited {process.returncode}; see {lane / 'server.log'}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{name} server startup timed out")
                time.sleep(1)
            def generate(ids):
                return post(base, {"input_ids": ids, "sampling_params": {
                    "temperature": 0.0, "max_new_tokens": args.max_tokens,
                }})
            generate(encoded[0])  # warm up before recording the reference
            started = time.perf_counter()
            before = [generate(ids) for ids in encoded]
            if args.nvfp4_verify_parity:
                seconds = time.perf_counter() - started
                count = sum(map(len, before))
                (lane / "greedy_timing.json").write_text(json.dumps({
                    "seconds": seconds, "output_tokens": count, "output_tokens_per_second": count / seconds,
                    "requests": len(encoded), "concurrency": 1, "includes_prefill": True,
                }, indent=2) + "\n")
            (lane / "tokens.json").write_text(json.dumps(before, indent=2) + "\n")
            subprocess.run([
                str(args.python), str(benchmark_script(repo, "mixed_requests_check.py")),
                "--base-url", base, "--max-tokens", "96", "--rounds", "1",
                "--out", str(lane / "mixed.json"),
            ], env=env, check=True)
            if args.stress_requests:
                stress_command = [
                    str(args.python), str(benchmark_script(args.candidate, "stress_multiuser.py")),
                    "--base-url", base, "--server-log", str(lane / "server.log"),
                    "--input-ids", str(inputs), "--n-ref", str(len(encoded)),
                    "--ref-tokens", "32", "--n-stress", str(args.stress_requests),
                    "--workers", "8", "--abort", "0.2", "--out", str(lane / "stress.json"),
                ]
                if args.repeat_reference:
                    stress_command.append("--repeat-reference")
                subprocess.run(stress_command, env=env, check=True)
            if args.sampling_n:
                subprocess.run(sampling_collect_command(args, base, lane), env=env, check=True)
            after = [generate(ids) for ids in encoded]
            if before != after:
                raise RuntimeError(f"{name}: sequential greedy output changed after mixed-request slot reuse")
            if source_digest(repo) != source_sha256:
                raise RuntimeError(f"{name}: plugin source changed during the run; repeat on a fixed checkout")
            succeeded = True
            return {"source_sha256": source_sha256, "tokens": before,
                    "mixed_requests": "passed", "post_mixed_identity": "passed",
                    "sampling": sampling}
        finally:
            if process.poll() is None and args.keep_server_on_failure and not succeeded:
                (lane / "retained-server.json").write_text(json.dumps({
                    "pid": process.pid, "base_url": base, "source_sha256": source_sha256,
                    "stop_command": f"kill -TERM -- -{process.pid}"}, indent=2) + "\n")
                print(f"Retained failed candidate server process group {process.pid} for debugging", flush=True)
            elif process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def main():
    if sys.argv[1:2] == ["compare-greedy"]:
        return compare_saved_greedy(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "baseline", "candidate", "model", "draft", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30095)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--stress-requests", type=int, default=0)
    parser.add_argument("--repeat-reference", action="store_true")
    parser.add_argument("--keep-server-on-failure", action="store_true",
                        help="retain an owned candidate server for interactive failure diagnosis")
    parser.add_argument("--sampling-n", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=-1,
                        help="per-request seed base; requires --enable-deterministic-inference")
    parser.add_argument("--server-seed", type=int, default=0,
                        help="baseline --random-seed; the candidate uses the next seed for server RNG cohorts")
    parser.add_argument("--nvfp4-verify-parity", action="store_true",
                        help="use the RTX dense NVFP4 matched projection and attention profile")
    parser.add_argument("--nvfp4-parity-rows", type=int, choices=range(0, 65), metavar="0..64",
                        help="0 (default) uses the row-count-invariant FP8 path; 1..64 keeps SGLang's "
                             "kernels and pads AR decode and verification to groups of that many rows")
    parser.add_argument("--enable-deterministic-inference", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--route", choices=ROUTES, default="tree")
    parser.add_argument("--baseline-route", choices=ROUTES,
                        help="override the baseline route for a cross-route comparison")
    parser.add_argument("--chain-width", type=int, choices=range(2, 17), default=8)
    parser.add_argument("--mtp-draft", type=Path, help="optional compact checkpoint for built-in MTP")
    parser.add_argument("--qsa-ring", type=int, default=0)
    parser.add_argument("--qsa-check", action="store_true")
    parser.add_argument("--qsa-stable-topk", action="store_true")
    parser.add_argument("--dflash-joint-sampling", action="store_true")
    parser.add_argument("--gdn-unpacked-decode", action="store_true")
    parser.add_argument("--gdn-recurrent-verify", action="store_true",
                        help="run the unpacked GDN decode kernel on each verification step")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--prefetch-checkpoints", action="store_true")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--long-context", type=int, default=0, help="add a prompt with this many background tokens")
    parser.add_argument("--input-ids", type=Path, help="use an existing JSON array of tokenized prompts")
    parser.add_argument("--model-override", help="JSON model config overrides passed to SGLang")
    parser.add_argument("--attention-backend")
    parser.add_argument("--kv-cache-dtype",
                        help="pass an explicit SGLang KV cache dtype instead of using auto")
    parser.add_argument("--fp4-gemm-backend", help="explicit SGLang NVFP4 GEMM backend")
    parser.add_argument("--moe-runner-backend", help="explicit SGLang target MoE backend")
    parser.add_argument("--speculative-moe-runner-backend", help="explicit SGLang draft MoE backend")
    args = parser.parse_args()
    args.attention_backend = args.attention_backend or ("flashinfer" if args.nvfp4_verify_parity else "triton")
    # Server subprocesses run from their lane's checkout. Keep paths relative
    # to the caller, and preserve virtualenv interpreter symlinks.
    for name in ("python", "baseline", "candidate", "model", "draft", "mtp_draft", "out"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.absolute())
    if not 1 <= args.port <= 65535 or args.startup_timeout <= 0:
        parser.error("port must be in 1..65535 and startup timeout must be positive")
    if min(args.slots, args.context_limit, args.max_tokens) <= 0 or min(args.stress_requests, args.sampling_n, args.long_context) < 0:
        parser.error("slots/context/output limits must be positive; test counts must be non-negative")
    if not 0 < args.mem_fraction < 1:
        parser.error("--mem-fraction must be finite and between 0 and 1")
    if args.sampling_n and args.sampling_seed >= 0 and not args.enable_deterministic_inference:
        parser.error("request sampling seeds require --enable-deterministic-inference; "
                     "use --server-seed with --sampling-seed -1 for server RNG cohorts")
    if args.sampling_n and args.sampling_seed < 0 and args.enable_deterministic_inference:
        parser.error("server RNG sampling cohorts require --enable-deterministic-inference to be off; "
                     "use explicit --sampling-seed values for request-seed cohorts")
    if args.sampling_n and args.sampling_seed >= 0 and not args.candidate_only:
        parser.error("paired request-seed cohorts are correlated; use independent server RNG cohorts "
                     "for a two-sample comparison, or --candidate-only for request-seed collection")
    if args.keep_server_on_failure and not args.candidate_only:
        parser.error("--keep-server-on-failure requires --candidate-only")
    if args.baseline_route and args.candidate_only:
        parser.error("--baseline-route requires a baseline lane")
    lanes = ("candidate",) if args.candidate_only else ("baseline", "candidate")
    uses_dflash = any(lane_route(args, name) in DFLASH_ROUTES for name in lanes)
    model_config = json.loads((args.model / "config.json").read_text())
    if args.nvfp4_parity_rows is not None and not args.nvfp4_verify_parity:
        parser.error("--nvfp4-parity-rows requires --nvfp4-verify-parity")
    if args.nvfp4_parity_rows is None:
        args.nvfp4_parity_rows = 0
    if args.nvfp4_verify_parity:
        if any(lane_route(args, name) not in (*DFLASH_ROUTES, "ar") for name in lanes):
            parser.error("--nvfp4-verify-parity supports the dense AR, tree and chain routes")
        if (model_config.get("model_type") != "qwen3_5"
                or model_config.get("quantization_config", {}).get("quant_algo") != "MIXED_PRECISION"):
            parser.error("--nvfp4-verify-parity needs the Qwen3.8 dense mixed NVFP4/FP8 checkpoint")
        if args.attention_backend != "flashinfer" or args.fp4_gemm_backend not in (None, "flashinfer_cutlass"):
            parser.error("--nvfp4-verify-parity uses FlashInfer attention and flashinfer_cutlass FP4 GEMM")
        args.fp4_gemm_backend = "flashinfer_cutlass"
        args.gdn_unpacked_decode = True
        args.gdn_recurrent_verify |= uses_dflash
        args.dflash_joint_sampling |= uses_dflash
    if uses_dflash and model_config.get("model_type") in ("qwen4_exp", "qwen4_exp_text"):
        parser.error("Flash Next DFlash2 verification needs PLE/QSA support; use AR or MTP")
    if args.qsa_check and not args.eager:
        parser.error("--qsa-check requires --eager")
    if args.dflash_joint_sampling and not uses_dflash:
        parser.error("--dflash-joint-sampling requires a DFlash2 lane")
    if args.gdn_recurrent_verify and not uses_dflash:
        parser.error("--gdn-recurrent-verify requires a DFlash2 lane")
    if args.input_ids and args.long_context:
        parser.error("--input-ids and --long-context are mutually exclusive")
    if args.long_context + args.max_tokens + 128 > args.context_limit:
        parser.error("--context-limit must leave room for the background, prompt and output")
    if args.model_override:
        json.loads(args.model_override)
    try:
        encoded = (json.loads(args.input_ids.read_text()) if args.input_ids else tokenize_prompts(args))
        validate_prompts(encoded)
        if max(map(len, encoded)) + args.max_tokens > args.context_limit:
            raise ValueError("tokenized prompts and output budget exceed the context limit")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "harness.py").write_bytes(Path(__file__).read_bytes())
    # Refuse an occupied port before starting any server or writing a success record.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    results = {}
    for name in lanes:
        results[name] = run_lane(args, name, getattr(args, name), encoded)
        print(f"{name}: mixed-request and state-reuse checks passed", flush=True)
    greedy_status = "not_run_single_lane"
    if not args.candidate_only:
        report = greedy_comparison(encoded, results["baseline"]["tokens"], results["candidate"]["tokens"])
        path = args.out / "greedy_comparison.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise SystemExit(f"baseline and candidate greedy tokens differ; see {path}")
        greedy_status = "passed"
    sampling_comparison = compare_sampling_cohorts(args)
    results.update(passed=True, greedy_comparison=greedy_status, sampling_comparison=sampling_comparison,
                   prompts=len(encoded), max_tokens=args.max_tokens,
                   model=str(args.model), draft=str(args.draft),
                   route=args.route,
                   baseline_route=None if args.candidate_only else lane_route(args, "baseline"),
                   input_lengths=[len(ids) for ids in encoded],
                   controls={"slots": args.slots, "context_limit": args.context_limit,
                             "stress_requests": args.stress_requests, "sampling_n": args.sampling_n,
                             "sampling_seed": args.sampling_seed, "sampling_mode": sampling_mode(args),
                             "qsa_ring": args.qsa_ring,
                             "server_seed": args.server_seed,
                             "enable_deterministic_inference": args.enable_deterministic_inference,
                             "qsa_check": args.qsa_check, "eager": args.eager},
                   evidence="development correctness; not a performance qualification")
    (args.out / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"PASS: {len(encoded)} prompts on {', '.join(lanes)}; mixed-request/state checks completed")


def tokenize_prompts(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = []
    prompts = list(PROMPTS)
    if args.long_context:
        background = tokenizer("This is neutral background information. ", add_special_tokens=False).input_ids
        notes = tokenizer.decode((background * (1 + args.long_context // len(background)))[:args.long_context])
        prompts.append("Reference notes:\n" + notes + "\nAnswer this task:\n" + PROMPTS[2])
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            enable_thinking=False, tokenize=False,
        )
        encoded.append(list(tokenizer(text, add_special_tokens=False).input_ids))
    return encoded


if __name__ == "__main__":
    sys.exit(main())
