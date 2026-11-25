"""Measure DFlash2's draft vocabulary subset against the native full draft head."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

import bench_27b_nvfp4 as bench


PATH = "draft_vocab"
FEATURE = "AX_DFLASH_DRAFT_VOCAB"


@contextmanager
def server(args, dest, lane):
    dest.mkdir(parents=True)
    command, env_flags, algorithm = bench.launch_config(args, "replay8")
    command += ["--watchdog-timeout", "1200"]
    env_flags["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = "python"
    if lane == "subset":
        env_flags[FEATURE] = "1"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("AX_", "FC1_", "SGLANG_"))}
    env.update(env_flags, HF_HUB_OFFLINE="1")
    url = f"http://127.0.0.1:{args.port}"
    bench.write_json(dest / "launch.json", {"command": command, "flags": env_flags,
                                            "prompt_sha256": args.prompt_sha256})
    with (dest / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            info = bench.wait_for_server(process, url, args.target, algorithm)
            if not info.get("enable_linear_replayssm_spec"):
                raise RuntimeError("Expected native DFlash2 with accepted-prefix replay")
            yield url
        finally:
            bench.stop_server(process)


def tally(before, after):
    subset_before = before["optimized"].get(PATH, {})
    subset_after = after["optimized"].get(PATH, {})
    return {size: {"total": total - before["rounds"].get(size, 0),
                   "subset": subset_after.get(size, 0) - subset_before.get(size, 0)}
            for size, total in sorted(after["rounds"].items(), key=lambda pair: int(pair[0]))
            if total > before["rounds"].get(size, 0)}


def replay_shapes(before, after):
    shapes = {}
    for phase, counts in after["graphs"].items():
        earlier = before["graphs"].get(phase, {})
        shapes[phase] = []
        for shape, total in sorted(counts.items()):
            if total > earlier.get(shape, 0):
                live, slots, width = map(int, shape.split(":"))
                shapes[phase].append({"live_requests": live, "graph_slots": slots,
                                      "live_rows": live * width, "graph_rows": slots * width,
                                      "replays": total - earlier.get(shape, 0)})
    return shapes


def measure(url, prompts, concurrency, length):
    before = bench.hotpath_snapshot(url)
    result = bench.run_pass(url, prompts, concurrency, length, True)
    after = bench.hotpath_snapshot(url)
    result["draft_graphs"] = tally(before, after)
    result["replay_shapes"] = replay_shapes(before, after)
    return result


def sampled_fallback(url, input_ids):
    body = {"input_ids": input_ids, "sampling_params": {
        "temperature": 0.8, "top_k": 20, "top_p": 0.9, "max_new_tokens": 32, "ignore_eos": True,
    }}
    request = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    before = bench.hotpath_snapshot(url)
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.load(response)
    graphs = tally(before, bench.hotpath_snapshot(url))
    if len(result["output_ids"]) != 32 or not graphs or any(row["subset"] for row in graphs.values()):
        raise RuntimeError(f"Sampled request did not complete through the full draft head: {graphs}")
    return graphs


def compare(args, prompts, fallback_prompts):
    summary = {}
    for order in (("full", "subset"), ("subset", "full")):
        label = order[0] + "-first"
        results = {}
        for lane in order:
            dest = args.out / label / lane
            results[lane] = {}
            with server(args, dest, lane) as url:
                bench.run_pass(url, prompts[:args.slots], args.slots, 32, True)
                for clients in args.concurrency:
                    result = measure(url, prompts, clients, args.output_tokens)
                    graphs = result["draft_graphs"]
                    if lane == "subset" and not graphs.get(str(clients), {}).get("subset"):
                        raise RuntimeError(f"No subset graph replay at C{clients}. Recorded rounds: {graphs}")
                    bench.write_json(dest / f"c{clients}.json", result)
                    results[lane][clients] = result
                    print(f"{label} {lane} C{clients}: {result['output_tokens'] / result['wall_s']:.2f} tok/s",
                          flush=True)
                if fallback_prompts:
                    fallback = measure(url, fallback_prompts, 1, 32)
                    if not fallback["draft_graphs"] or any(x["subset"] for x in fallback["draft_graphs"].values()):
                        raise RuntimeError("Out-of-subset prompts did not use the full draft head")
                    bench.write_json(dest / "multilingual-fallback.json", fallback)
                if args.check_sampling:
                    bench.write_json(dest / "sampled-fallback.json", sampled_fallback(url, prompts[0]))
        summary[label] = {}
        for clients in args.concurrency:
            full, subset = (results[lane][clients] for lane in ("full", "subset"))
            matches = bench.report_matches(f"{label} C{clients}",
                                           (row["output_ids"] for row in full["rows"]), subset["rows"])
            summary[label][clients] = {
                "matches": matches, "prompts": len(prompts),
                "throughput_change_pct": 100 * (full["wall_s"] / subset["wall_s"] - 1),
                **{lane: {"output_tok_s": run["output_tokens"] / run["wall_s"],
                          **{key: run[key] for key in ("p95_request_s", "tokens_per_verify", "draft_graphs", "replay_shapes")}}
                   for lane, run in (("full", full), ("subset", subset))},
            }
        bench.write_json(args.out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--fallback-ids", type=Path, help="Tokenized prompts with an excluded ID in the last 32 tokens")
    parser.add_argument("--check-sampling", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--kv-tokens", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--port", type=int, default=30517)
    parser.set_defaults(verify_parity=False, verified_profile=False)
    args = parser.parse_args()
    args.slots = max(args.concurrency)
    data = args.input_ids.read_bytes()
    prompts = json.loads(data)
    fallback = json.loads(args.fallback_ids.read_bytes()) if args.fallback_ids else []
    if (min(args.concurrency) < 1 or args.output_tokens < 1
            or not isinstance(prompts, list) or not isinstance(fallback, list)
            or len(prompts) < args.slots or any(
                not isinstance(prompt, list) or not prompt or any(type(i) is not int or i < 0 for i in prompt)
                for prompt in prompts + fallback
            )):
        parser.error("Provide valid token-ID lists, one prompt per client, and a positive output length")
    lengths = sorted(map(len, prompts + fallback), reverse=True)
    length = max(args.output_tokens, 32)
    if lengths[0] + length > args.context_length or sum(lengths[:args.slots]) + args.slots * length > args.kv_tokens:
        parser.error("Increase context and KV limits to fit the requested workload")
    args.prompt_sha256 = hashlib.sha256(data).hexdigest()
    args.out.mkdir(parents=True)
    compare(args, prompts, fallback)


if __name__ == "__main__":
    main()
