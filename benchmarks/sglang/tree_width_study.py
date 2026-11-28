"""Driver of the 2026-11-25 RTX PRO 6000 tree and width study (results/rtx_pro_6000_tree_width/REPORT.md).

    python benchmarks/sglang/tree_width_study.py trace  --source SRC --models M --ids ids200.json --out OUT
    python benchmarks/sglang/tree_width_study.py replay --source SRC --models M --ids ids200.json --out OUT --lanes replay8 replay16 stock16
    python benchmarks/sglang/tree_width_study.py lanes  --source SRC --models M --ids ids200.json --out OUT --lanes ar tree16 btree4

`trace` serves the lattice tree (one request) and the width-16 chain with AX_LATTICE_TRACE and sends
24 prompts per family, renaming the trace file after each family pass. `replay` serves each lane of
benchmarks/sglang/bench_27b_nvfp4.py and runs every family at C1, C2, C4 and C8 with 256 greedy
tokens per request, saving the output ids. `lanes` serves AR, the one-request tree (`tree16`) and the
batched tree (`btreeK`: AX_BTREE=K) with the same adaptive-width launch and saves every output for
exact-output comparisons with AR. Families come from the prompt text: HumanEval prompts ask
for a Python function, GSM8K prompts for a step-by-step solution, the rest are UltraChat.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def families(ids, target):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    out = {"code": [], "math": [], "chat": []}
    for prompt in ids:
        text = tokenizer.decode(prompt)
        family = ("code" if "Complete the following Python function" in text
                  else "math" if "Solve the problem step by step" in text else "chat")
        out[family].append(prompt)
    return out


def serve(bench, command, flags, algorithm, target, port, log_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("FC1_", "AX_", "SGLANG_"))}
    env.update(flags, HF_HUB_OFFLINE="1")
    log = log_path.open("w")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    info = bench.wait_for_server(process, f"http://127.0.0.1:{port}", target, algorithm)
    return process, info


def strip(result):
    """Keep output ids and per-request times; drop nothing needed for exact-output comparisons."""
    return {**result, "rows": [{"wall_s": r["wall_s"], "spec_verify_ct": r["spec_verify_ct"],
                                "output_ids": r["output_ids"]} for r in result["rows"]]}


def trace(args, bench, ab, groups):
    url = f"http://127.0.0.1:{args.port}"
    lanes = (("tree16", "w16", {"FC1_TREE_N": "16"}, (1,)),
             ("chain16", "w16", {"AX_TREE": "0"}, (1, 4)))
    for name, lane, env, concurrencies in lanes:
        out = args.out / f"trace-{name}"
        out.mkdir(parents=True, exist_ok=True)
        live = out / "live.jsonl"
        options = SimpleNamespace(
            target=args.models / "27b-nvfp4", draft=args.models / "27b-dflash2", port=args.port, slots=args.slots,
            kv_tokens=args.kv_tokens, mem_fraction_static=args.mem_fraction or 0.82, widths=[16, 8], table="1:16,4:8", schedule="16x31,8x1",
            adaptive_env={}, server_args=args.server_args.split(),
            env={**args.server_env, **env, "AX_LATTICE": "1", "AX_LATTICE_TRACE": str(live)})
        command, flags = ab.launch(options, lane)
        bench.write_json(out / "launch.json", {"command": command, "flags": flags})
        process, _ = serve(bench, command, flags, "FC1_DFLASH_TREE", options.target, args.port, out / "server.log")
        try:
            bench.run_pass(url, groups["chat"][-8:], 8, 32, True)
            live.unlink(missing_ok=True)
            for concurrency in concurrencies:
                for family, prompts in groups.items():
                    result = bench.run_pass(url, prompts[:args.trace_prompts], concurrency, args.output_tokens, True)
                    live.rename(out / f"c{concurrency}-{family}.jsonl")
                    rate = result["output_tokens"] / result["wall_s"]
                    print(f"trace {name} C{concurrency} {family}: {rate:.1f} tok/s (diagnostic, synchronized)",
                          flush=True)
        finally:
            bench.stop_server(process)


def replay(args, bench, groups):
    url = f"http://127.0.0.1:{args.port}"
    for lane in args.lanes:
        out = args.out / f"speed-{lane}"
        out.mkdir(parents=True, exist_ok=True)
        options = SimpleNamespace(target=args.models / "27b-nvfp4", draft=args.models / "27b-dflash2",
                                  port=args.port, slots=args.slots, context_length=4096, kv_tokens=args.kv_tokens,
                                  mem_fraction_static=args.mem_fraction or 0.90, verified_profile=False, verify_parity=False)
        command, flags, algorithm = bench.launch_config(options, lane)
        bench.write_json(out / "launch.json", {"command": command, "flags": flags})
        process, info = serve(bench, command, flags, algorithm, options.target, args.port, out / "server.log")
        try:
            if bool(info.get("enable_linear_replayssm_spec")) != lane.startswith("replay"):
                raise RuntimeError(f"unexpected ReplaySSM setting for {lane}")
            bench.run_pass(url, groups["chat"][-8:], 8, 32, True)
            passes = []
            for concurrency in args.concurrency:
                for family, prompts in groups.items():
                    def measure():
                        return bench.run_pass(url, prompts, concurrency, args.output_tokens, True)
                    result = (bench.hotpath_pass(url, "gdn_replay", concurrency, measure)
                              if lane.startswith("replay") else measure())
                    result = {**strip(result), "family": family}
                    passes.append(result)
                    bench.write_json(out / "passes.json", passes)
                    print(f"{lane} C{concurrency} {family}: {result['output_tokens'] / result['wall_s']:.1f} tok/s, "
                          f"{result['tokens_per_verify']:.2f} tokens/verify", flush=True)
        finally:
            bench.stop_server(process)


# The tree16 lane of bench_27b_nvfp4.py (fused GDN tree kernels for Qwen3.8-27B's 48 GDN layers and
# heads), without its per-node SSM positions, which AX_BTREE refuses.
BTREE_ENV = {"AX_TREE": "1", "AX_LATTICE": "1", "AX_FAST": "1", "AX_GDN_LAYERS": "48", "AX_GDN_HEADS": "48",
             "FC1_TREE64": "1", "FC1_BRANCH_PILOT": "1", "FC1_OURS": "1", "FC1_FUSED": "1",
             "FC1_COMPILE_BUILD": "1", "FC1_DRAFT_BLOCK": "16", "FC1_TREE_N": "16"}


def lanes(args, bench, ab, groups):
    url = f"http://127.0.0.1:{args.port}"
    prompts = [p for family in ("code", "math", "chat") for p in groups[family]][:args.limit]
    for lane in args.lanes:
        out = args.out / f"lane-{lane}"
        out.mkdir(parents=True, exist_ok=True)
        options = SimpleNamespace(
            target=args.models / "27b-nvfp4", draft=args.models / "27b-dflash2", port=args.port, slots=args.slots,
            context_length=4096, kv_tokens=args.kv_tokens, verified_profile=False, verify_parity=False,
            mem_fraction_static=args.mem_fraction or 0.82, widths=[16, 8], table="1:16,4:8", schedule="16x31,8x1", adaptive_env={},
            server_args=args.server_args.split(), env={**BTREE_ENV, **args.server_env})
        if lane == "ar":
            options.mem_fraction_static = args.mem_fraction or 0.90
            command, flags, algorithm = bench.launch_config(options, "ar")
        else:
            name = lane
            if name.endswith("s"):          # ...s: the tree shape of --shape-env (tune_tree_optuna.py)
                options.env.update(args.shape_env)
                name = name[:-1]
            if name.startswith("btree"):    # btreeK: AX_BTREE=K; btreeKn: SGLang's native GDN kernels
                options.env["AX_BTREE"] = name[5:].rstrip("n")
                if name.endswith("n"):
                    options.env["AX_BTREE_NATIVE"] = "1"
            elif name.startswith("tree") and name[4:] in ("16", "32", "64"):   # one-request tree of N nodes
                options.env["FC1_TREE_N"] = name[4:]
            else:
                raise SystemExit(f"unknown lane {lane!r}: ar, treeN (16/32/64), btreeK, btreeKn, each optionally +s")
            command, flags = ab.launch(options, "w16")
            algorithm = "FC1_DFLASH_TREE"
        bench.write_json(out / "launch.json", {"command": command, "flags": flags})
        process, _ = serve(bench, command, flags, algorithm, options.target, args.port, out / "server.log")
        try:
            bench.run_pass(url, groups["chat"][-8:], 8, 32, algorithm is not None)
            passes = []
            for concurrency in args.concurrency:
                result = strip(bench.run_pass(url, prompts, concurrency, args.output_tokens, algorithm is not None))
                passes.append(result)
                bench.write_json(out / "passes.json", passes)
                verify = f", {result['tokens_per_verify']:.2f} tokens/verify" if result["tokens_per_verify"] else ""
                print(f"{lane} C{concurrency}: {result['output_tokens'] / result['wall_s']:.1f} tok/s{verify}",
                      flush=True)
        finally:
            bench.stop_server(process)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("trace", "replay", "lanes"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30530)
    parser.add_argument("--lanes", nargs="+", default=["replay8", "replay16"])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--trace-prompts", type=int, default=24)
    parser.add_argument("--limit", type=int, default=200, help="lanes: prompts per pass")
    parser.add_argument("--server-env", nargs="*", default=[], metavar="KEY=VALUE",
                        help="trace/lanes: extra server environment, e.g. SGLANG_FLASHINFER_WORKSPACE_SIZE=67108864")
    parser.add_argument("--slots", type=int, default=8, help="server request slots (2 on a 32 GB RTX 5090)")
    parser.add_argument("--kv-tokens", type=int, default=32768)
    parser.add_argument("--mem-fraction", type=float, help="static memory fraction (default 0.82, AR and replay 0.90)")
    parser.add_argument("--shape-env", nargs="*", default=[], metavar="KEY=VALUE",
                        help="lanes: tree-shape settings of the lanes whose name ends in s (e.g. tree16s)")
    # Before results/rtx_pro_6000_prefill_graph the default also passed --disable-prefill-cuda-graph.
    parser.add_argument("--server-args", default="--disable-radix-cache "
                        "--attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass")
    args = parser.parse_args()
    try:
        args.shape_env = dict(item.split("=", 1) for item in args.shape_env)
        args.server_env = dict(item.split("=", 1) for item in args.server_env)
    except ValueError:
        parser.error("--shape-env and --server-env take KEY=VALUE items")
    sys.path.insert(0, str(args.source / "benchmarks/sglang"))
    import bench_27b_nvfp4 as bench
    import ab_adaptive_width as ab
    groups = families(json.loads(args.ids.read_text()), args.models / "27b-nvfp4")
    args.out.mkdir(parents=True, exist_ok=True)
    bench.write_json(args.out / f"families-{args.mode}.json", {k: len(v) for k, v in groups.items()})
    if args.mode == "trace":
        trace(args, bench, ab, groups)
    elif args.mode == "lanes":
        lanes(args, bench, ab, groups)
    else:
        replay(args, bench, groups)


if __name__ == "__main__":
    main()
