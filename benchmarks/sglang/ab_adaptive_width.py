#!/usr/bin/env python3
"""Throughput of the batch-size width table and the adaptive policy against each fixed chain width.

Each lane starts one server with the launch defaults (no exact-output claim) and the same captured
chain graphs; only the width policy differs. The concurrency sweep runs twice per lane: the second
sweep shows the run-to-run spread, and for the adaptive lane the rate after its warm-up.
The mixed workload repeats a fixed sequence of request waves in each sweep.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import cycle
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_27b_nvfp4 as bench  # noqa: E402


CAPPED = re.compile(r"max_running_requests is capped to (\d+)")


def launch(args, lane):
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", str(args.target.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port), "--max-running-requests", str(args.slots),
        "--cuda-graph-max-bs-decode", str(args.slots), "--context-length", "8192",
        "--max-total-tokens", str(args.kv_tokens), "--mem-fraction-static", str(args.mem_fraction_static),
        "--skip-server-warmup", "--random-seed", "42",
        "--speculative-algorithm", "FC1_DFLASH_TREE", "--speculative-draft-model-path",
        str(args.draft.resolve()), "--speculative-num-draft-tokens", "16", *args.server_args,
    ]
    flags = {"SGLANG_PLUGINS": "ax-engine", "AX_CHAIN_GRAPH_WIDTHS": " ".join(map(str, args.widths)),
             "AX_TIME": "1"}
    if lane == "adaptive":
        flags["AX_POLICY_MODE"] = "adaptive"
        flags.update(args.adaptive_env)
    elif lane == "table":
        flags["AX_CHAIN_WIDTH_BY_BATCH"] = args.table
    elif lane == "schedule":
        flags["AX_CHAIN_SCHEDULE"] = args.schedule
    else:
        flags["AX_CHAIN_WIDTH"] = lane[1:]
    flags.update(args.env)
    return command, flags


def run_mixed_pass(base_url, prompts, concurrency, length):
    """Send the same ordered waves to every lane, draining each before the next."""
    rows = []
    wave_sizes = []
    with ThreadPoolExecutor(max_workers=max(concurrency)) as pool:
        start = time.perf_counter()
        for requested in cycle(concurrency):
            offset = len(rows)
            wave = prompts[offset:offset + requested]
            if not wave:
                break
            rows.extend(pool.map(lambda ids: bench.generate(base_url, ids, length, True), wave))
            wave_sizes.append(len(wave))
        wall_s = time.perf_counter() - start
    result = bench.summarize_pass(rows, wall_s, max(wave_sizes), length, True)
    result["wave_sizes"] = wave_sizes
    return result


def run_lane(args, lane, prompts):
    out = args.out / lane
    out.mkdir(parents=True, exist_ok=True)
    command, flags = launch(args, lane)
    bench.write_json(out / "launch.json", {"command": command, "flags": flags,
                                           "prompt_sha256": args.prompt_sha256,
                                           "mixed_concurrency": args.mixed_concurrency})
    env = {key: value for key, value in os.environ.items() if not key.startswith(("FC1_", "AX_", "SGLANG_"))}
    env.update(flags, HF_HUB_OFFLINE="1")
    base_url = f"http://127.0.0.1:{args.port}"
    timed_prompts = prompts * args.repeat
    requested = args.mixed_concurrency or args.concurrency
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            bench.wait_for_server(process, base_url, args.target, "FC1_DFLASH_TREE")
            # SGLang lowers the request limit when the GDN state cache is too small for it, and
            # says so in its log; the concurrency passes would then run fewer requests than named.
            capped = CAPPED.findall((out / "server.log").read_text(errors="replace"))
            if capped and int(capped[-1]) < max(requested):
                raise RuntimeError(f"the server runs {capped[-1]} requests at once, fewer than "
                                   f"the requested concurrency {max(requested)}: {out / 'server.log'}")
            bench.run_pass(base_url, timed_prompts[:args.slots], args.slots, 32, True)
            rows = []
            for sweep in range(args.sweeps):
                if args.mixed_concurrency:
                    results = (run_mixed_pass(base_url, timed_prompts, requested, args.output_tokens),)
                else:
                    results = (bench.run_pass(base_url, timed_prompts, concurrency, args.output_tokens, True)
                               for concurrency in requested)
                for result in results:
                    rate = result["output_tokens"] / result["wall_s"]
                    row = {"sweep": sweep, "concurrency": result["concurrency"], "output_tok_s": rate,
                           "wall_s": result["wall_s"], "median_request_s": result["median_request_s"],
                           "p95_request_s": result["p95_request_s"], "tokens_per_verify": result["tokens_per_verify"]}
                    label = f"C{result['concurrency']}"
                    if "wave_sizes" in result:
                        row["wave_sizes"] = result["wave_sizes"]
                        label = f"waves {'/'.join(map(str, requested))}"
                    rows.append(row)
                    print(f"{lane} sweep {sweep} {label}: {rate:.1f} output tok/s", flush=True)
            bench.write_json(out / "passes.json", rows)
            return rows
        finally:
            bench.stop_server(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--lanes", nargs="+",
                        help="w<width>, table, schedule or adaptive; default: every width, then table and adaptive")
    parser.add_argument("--table", default="1:16,4:8", help="AX_CHAIN_WIDTH_BY_BATCH of the table lane")
    parser.add_argument("--schedule", default="16x31,8x1",
                        help="AX_CHAIN_SCHEDULE of the schedule lane: width changes without a policy")
    parser.add_argument("--adaptive-env", nargs="*", default=[], metavar="NAME=VALUE")
    parser.add_argument("--env", nargs="*", default=[], metavar="NAME=VALUE",
                        help="more plugin settings for every lane, such as AX_ROUND_DUMP=/path")
    parser.add_argument("--server-args", default="", type=str.split,
                        help="more server arguments for every lane, one quoted string")
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    workload.add_argument("--mixed-concurrency", type=int, nargs="+",
                          help="repeat these wave sizes in one pass, waiting for each wave to finish")
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--kv-tokens", type=int, default=32768)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    args.adaptive_env = dict(item.split("=", 1) for item in args.adaptive_env)
    args.env = dict(item.split("=", 1) for item in args.env)
    lanes = args.lanes or [f"w{width}" for width in args.widths] + ["table", "adaptive"]
    concurrency = args.mixed_concurrency or args.concurrency
    if any(not 1 <= size <= args.slots for size in concurrency):
        parser.error("each concurrency must be between 1 and --slots")
    if "table" in lanes and not {int(e.split(":")[1]) for e in args.table.split(",")} <= set(args.widths):
        parser.error("--table uses a width that --widths does not capture")
    if "adaptive" in lanes and len(args.widths) < 2:
        parser.error("the adaptive lane needs two --widths")
    raw_prompts = args.input_ids.read_bytes()
    args.prompt_sha256 = hashlib.sha256(raw_prompts).hexdigest()
    prompts = json.loads(raw_prompts)
    if not prompts or args.repeat < 1:
        parser.error("provide at least one prompt and a positive --repeat")
    if args.mixed_concurrency:
        first_peak_end = sum(concurrency[:concurrency.index(max(concurrency)) + 1])
        if len(prompts) * args.repeat < first_peak_end:
            parser.error("the repeated prompt set must cover the mixed schedule through its first peak wave")
    results = {lane: run_lane(args, lane, prompts) for lane in lanes}
    bench.write_json(args.out / "summary.json", results)


if __name__ == "__main__":
    main()
