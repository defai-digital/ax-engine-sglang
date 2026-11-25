#!/usr/bin/env python3
"""Capture, qualify and compare FP8 DFlash2 verification tactics on SM120."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace
import warnings

import ax_fp8_verify as fp8
import bench_27b_nvfp4 as bench


def graph_time(call, torch):
    graph = torch.cuda.CUDAGraph()
    call()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(12):
            call()
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(4):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / 48


def qualify_projection(name, snapshots, rows, factories, torch):
    workspaces = {n: torch.empty(n, device="cuda", dtype=torch.uint8)
                  for n in {sample["workspace_bytes"] for sample in snapshots}}
    operands = []
    expected = []
    for sample in snapshots:
        a, b, scale_a, scale_b, result = sample["tensors"]
        args = [a, b, scale_a, scale_b, torch.empty_like(result), workspaces[sample["workspace_bytes"]]]
        if fp8.projection_for(args, torch, rows) != name:
            raise ValueError(f"Invalid recorded operands for {name}")
        operands.append(args)
        expected.append(result.view(torch.int16))
    native = factories[fp8.NATIVE]()
    native_tactics = [fp8.tactic_from_json(sample["native"]) for sample in snapshots]

    def run_native():
        for args, tactic in zip(operands, native_tactics, strict=True):
            native(inputs=args, tactic=tactic)

    def exact():
        return all(torch.equal(args[4].view(torch.int16), result)
                   for args, result in zip(operands, expected, strict=True))

    run_native()
    if not exact():
        raise RuntimeError(f"Native eager replay changed BF16 outputs for {name}")
    native_us = graph_time(run_native, torch) / len(operands)
    if not exact():
        raise RuntimeError(f"Native CUDA graph replay changed BF16 outputs for {name}")

    results = []
    for backend, factory in factories.items():
        try:
            runner = factory()
            valid = set.intersection(*(set(runner.get_valid_tactics(args, None)) for args in operands))
        except (ImportError, RuntimeError, ValueError) as error:
            torch.cuda.synchronize()
            results.append({"backend": backend, "status": "unavailable", "error": str(error)})
            continue
        for tactic in sorted(valid | {-1}, key=repr):
            if backend == fp8.NATIVE and all(value == tactic for value in native_tactics):
                continue

            def run_candidate():
                for args in operands:
                    runner(inputs=args, tactic=tactic)

            item = {"backend": backend, "tactic": tactic}
            results.append(item)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", message="cuDNN fp8 GEMM tactic failed.*")
                    run_candidate()
                    if not exact():
                        item["status"] = "eager_mismatch"
                        continue
                    pairs = []
                    for trial in range(3):
                        calls = (run_native, run_candidate) if trial % 2 == 0 else (run_candidate, run_native)
                        timings = {}
                        for call in calls:
                            timings[call] = graph_time(call, torch) / len(operands)
                            if not exact():
                                if call is run_native:
                                    raise AssertionError(f"Native graph changed BF16 outputs for {name}")
                                break
                        else:
                            pairs.append((timings[run_native], timings[run_candidate]))
                            continue
                        break
                    if len(pairs) != 3:
                        item["status"] = "graph_mismatch"
                        continue
            except (RuntimeError, ValueError, UserWarning) as error:
                torch.cuda.synchronize()
                item.update(status="unavailable", error=str(error))
                continue
            item.update(status="exact", native_us=statistics.median(a for a, _ in pairs),
                        candidate_us=statistics.median(b for _, b in pairs),
                        ratio=statistics.median(b / a for a, b in pairs))
    return {"native_us": native_us, "native": native_tactics, "samples": len(operands),
            "workspace_bytes": sorted(workspaces), "candidates": results}


def probe(args):
    from flashinfer.gemm import gemm_base as gemm
    import torch

    manifest = json.loads((args.operands / "manifest.json").read_text())
    if manifest["identity"]["runtime"] != fp8.require_runtime(torch) or manifest["rows"] not in fp8.ROWS:
        raise ValueError("FP8 captures must match the probe runtime and use a supported row count")
    reports = {}
    profile = {**manifest, "projections": {}}
    for name in fp8.PROJECTIONS.values():
        files = sorted(args.operands.glob(f"{name}-*.pt"))
        if len(files) < 2:
            raise ValueError(f"Expected at least two distinct captured layers for {name}")
        samples = [torch.load(path, map_location="cuda", weights_only=True) for path in files]
        result = qualify_projection(name, samples, manifest["rows"], fp8.runner_factories(gemm), torch)
        reports[name] = result
        eligible = [entry for entry in result["candidates"] if entry["status"] == "exact" and entry["ratio"] < 1]
        if eligible:
            best = min(eligible, key=lambda entry: entry["ratio"])
            profile["projections"][name] = {key: best[key] for key in ("backend", "tactic")}
            profile["projections"][name].update({key: result[key] for key in ("native", "samples", "workspace_bytes")})
            print(f"{name}: {best['native_us']:.2f} → {best['candidate_us']:.2f} μs ({best['backend']})")
        else:
            print(f"{name}: no faster tactic matched the recorded BF16 outputs")
    bench.write_json(args.out / "probe.json", {**manifest, "projections": reports})
    if profile["projections"]:
        bench.write_json(args.out / "profile.json", profile)


def plugin_path(directory):
    dist = directory / "ax_fp8_verify-0.0.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ax-fp8-verify\nVersion: 0.0.0\n")
    (dist / "entry_points.txt").write_text("[sglang.srt.plugins]\nax-fp8-verify = ax_fp8_verify:register\n")
    return os.pathsep.join((str(directory), str(Path(__file__).resolve().parent)))


@contextmanager
def server(args, dest, mode, plugin=None):
    dest.mkdir(parents=True)
    settings = SimpleNamespace(**vars(args), slots=args.concurrency, verify_parity=False, verified_profile=False)
    command, flags, algorithm = bench.launch_config(settings, "replay8")
    if mode == "capture":
        command.append("--disable-decode-cuda-graph")
        flags.update(AX_FP8_CAPTURE=str(args.out / "operands"), AX_FP8_SAMPLES=str(args.samples))
    else:
        graph_batches = (1, 2, 4, *(rows // 8 for rows in fp8.ROWS))
        command += ["--cuda-graph-bs-decode", *(str(size) for size in graph_batches if size <= args.concurrency)]
    if mode == "candidate":
        flags["AX_FP8_PROFILE"] = str(args.profile.resolve())
    env = {key: value for key, value in os.environ.items() if not key.startswith(("AX_", "FC1_", "SGLANG_"))}
    if plugin:
        flags.update(SGLANG_PLUGINS="ax-engine,ax-fp8-verify", AX_FP8_TARGET=str(args.target.resolve()),
                     AX_FP8_ROWS=str(args.concurrency * 8), AX_FP8_MODE=mode)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (plugin, env.get("PYTHONPATH"))))
    env.update(flags, HF_HUB_OFFLINE="1")
    url = f"http://127.0.0.1:{args.port}"
    bench.write_json(dest / "launch.json", {"command": command, "flags": flags,
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


def capture(args, prompts, plugin):
    target = args.out / "operands"
    target.mkdir()
    with server(args, args.out / "capture", "capture", plugin) as url:
        bench.run_pass(url, prompts, args.concurrency, args.output_tokens, True)
    for name in fp8.PROJECTIONS.values():
        if len(list(target.glob(f"{name}-*.pt"))) < args.samples:
            raise RuntimeError(f"Captured fewer than {args.samples} {name} layers. See capture/server.log.")


def compare(args, prompts, plugin):
    summary = {}
    for order in (("native", "candidate"), ("candidate", "native")):
        order_dir = args.out / f"{order[0]}-first"
        results = {}
        for mode in order:
            dest = order_dir / mode
            with server(args, dest, mode, plugin if mode == "candidate" else None) as url:
                c = args.concurrency
                bench.run_pass(url, prompts[:c], c, 32, True)
                path = "gdn_replay" if mode == "native" else fp8.BENCH_PATH
                result = bench.hotpath_pass(url, path, c, lambda: bench.run_pass(
                    url, prompts, c, args.output_tokens, True,
                ))
            bench.write_json(dest / "pass.json", result)
            results[mode] = result
        native, candidate = results["native"], results["candidate"]
        matches = bench.report_matches(order_dir.name, (row["output_ids"] for row in native["rows"]), candidate["rows"])
        summary[order_dir.name] = {"matches": matches, "prompts": len(prompts),
                                   "throughput_change_pct": 100 * (native["wall_s"] / candidate["wall_s"] - 1), **{
            mode: {"output_tok_s": run["output_tokens"] / run["wall_s"], "p95_request_s": run["p95_request_s"],
                   "tokens_per_verify": run["tokens_per_verify"], "hotpath": run["hotpath"]}
            for mode, run in results.items()
        }}
        bench.write_json(args.out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    audit = commands.add_parser("capture", help="save live eager verify operands and native BF16 outputs")
    audit.add_argument("--samples", type=int, default=4)
    search = commands.add_parser("probe", help="test each tactic on the saved operands, eager and graphed")
    search.add_argument("--operands", type=Path, required=True)
    ab = commands.add_parser("compare", help="measure a qualified profile in both launch orders")
    ab.add_argument("--profile", type=Path, required=True)
    for command in (audit, ab, search):
        command.add_argument("--out", type=Path, required=True)
    for command in (audit, ab):
        command.add_argument("--target", type=Path, required=True)
        command.add_argument("--draft", type=Path, required=True)
        command.add_argument("--input-ids", type=Path, required=True)
        command.add_argument("--concurrency", type=int, choices=[rows // 8 for rows in fp8.ROWS], default=8)
        command.add_argument("--output-tokens", type=int, default=512 if command is ab else 64)
        command.add_argument("--context-length", type=int, default=4096)
        command.add_argument("--kv-tokens", type=int, default=32_768)
        command.add_argument("--mem-fraction-static", type=float, default=0.90)
        command.add_argument("--port", type=int, default=30517)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.mode == "probe":
        args.out.mkdir(parents=True)
        probe(args)
        return
    data = args.input_ids.read_bytes()
    prompts = json.loads(data)
    if (not isinstance(prompts, list) or len(prompts) < args.concurrency
            or any(not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids) for ids in prompts)
            or args.output_tokens < 1 or (args.mode == "capture" and args.samples < 2)):
        parser.error("Provide at least one valid input per client, a positive token count and at least two capture samples")
    lengths = sorted(map(len, prompts), reverse=True)
    if (lengths[0] + max(args.output_tokens, 32) > args.context_length
            or sum(lengths[:args.concurrency]) + args.concurrency * max(args.output_tokens, 32) > args.kv_tokens):
        parser.error("Increase context and KV limits to fit the full requested batch")
    import torch

    identity = {"checkpoint": fp8.checkpoint_id(args.target), "runtime": fp8.require_runtime(torch)}
    if args.mode == "compare":
        fp8.read_profile(args.profile, identity, args.concurrency * 8)
    args.prompt_sha256 = hashlib.sha256(data).hexdigest()
    args.out.mkdir(parents=True)
    plugin = plugin_path(args.out / "plugin")
    if args.mode == "capture":
        capture(args, prompts, plugin)
    else:
        compare(args, prompts, plugin)


if __name__ == "__main__":
    main()
