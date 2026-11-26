"""Time FlashInfer's NVFP4 GEMM backends at verify-row counts on the 27B MLP shapes (Blackwell).

Review item: target verification at C8 / C16 replay spends most of its kernel time in the NVFP4 and FP8
GEMMs (results/rtx_pro_6000_hotpath_validation). This times `flashinfer.mm_fp4` with every backend
that runs on this GPU, at the padded row counts a verify step uses (C1 tree 32 nodes, C8 width-8 replay
64 rows, C16 128 rows), on the NVFP4 projections of Qwen3.8-27B NVFP4 (MLP gate+up 34816x5120 and down
5120x17408). Each backend is checked for bit-identical eager and CUDA-graph outputs; outputs of
different backends are compared with the max abs difference only (their reduction orders differ).

    python benchmarks/sglang/nvfp4_gemm_backends.py --rows 8 16 32 64 128 --out nvfp4-backends.json

Weights and activations are random (GEMM time does not depend on the values). The GEMMs cycle through
weight copies totalling at least four times the L2 cache (128 MB on the RTX PRO 6000), so the weights
stream from device memory as in a forward pass over 64 layers.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408)}   # (N, K)
BACKENDS = ("cutlass", "cudnn", "b12x", "cute-dsl", "cutedsl_low_latency", "trtllm", "cake")
FP4_MAX, FP8_MAX = 6.0, 448.0


def quantize(x):
    import flashinfer
    global_sf = (FP8_MAX * FP4_MAX / x.float().abs().max()).to(torch.float32)
    packed, sf = flashinfer.fp4_quantize(x, global_sf, 16, False, True)
    return packed, sf, global_sf


def time_graph(run, reps, trials):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(reps):
            run(i)
    samples = []
    for _ in range(trials):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop) * 1000 / reps)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    import flashinfer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS))
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "flashinfer", flashinfer.__version__,
          flush=True)
    results = []
    for layer, (n, k) in SHAPES.items():
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        w_fp4, w_sf, w_gs = quantize(w)
        size = w_fp4.numel() * w_fp4.element_size() + w_sf.numel() * w_sf.element_size()
        count = max(2, -(-4 * torch.cuda.get_device_properties(0).L2_cache_size // size))
        weights = [(w_fp4, w_sf)] + [(w_fp4.clone(), w_sf.clone()) for _ in range(count - 1)]
        for rows in args.rows:
            x = torch.randn(rows, k, device="cuda").to(torch.bfloat16)
            x_fp4, x_sf, x_gs = quantize(x)
            alpha = (1.0 / (x_gs * w_gs)).to(torch.float32)
            row = {"layer": layer, "rows": rows, "n": n, "k": k, "weight_copies": len(weights), "backends": {}}
            outputs = {}
            for backend in args.backends:
                out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")

                def run(i=0, backend=backend, out=out):
                    b_fp4, b_sf = weights[i % len(weights)]
                    return flashinfer.mm_fp4(x_fp4, b_fp4.T, x_sf, b_sf.T, alpha, torch.bfloat16, out,
                                             block_size=16, backend=backend)
                try:
                    for _ in range(3):   # autotune / JIT outside the graph
                        run()
                    torch.cuda.synchronize()
                    eager = out.clone()
                    us = time_graph(run, args.reps, args.trials)
                    graph_out = out.clone()
                except Exception as error:
                    row["backends"][backend] = {"error": f"{type(error).__name__}: {str(error).splitlines()[0][:150]}"}
                    continue
                outputs[backend] = eager
                row["backends"][backend] = {"us": us, "eager_equals_graph": bool(torch.equal(eager, graph_out))}
            ref = next(iter(outputs.values()), None)
            for backend, value in outputs.items():
                row["backends"][backend]["max_abs_diff_vs_first"] = float((value.float() - ref.float()).abs().max())
            timed = {b: v["us"] for b, v in row["backends"].items() if "us" in v}
            if timed:
                best = min(timed, key=timed.get)
                line = "  ".join(f"{b} {u:7.2f}" for b, u in timed.items())
                print(f"{layer:7s} M={rows:4d}  {line}   best {best}", flush=True)
            failed = {b: v["error"] for b, v in row["backends"].items() if "error" in v}
            if failed and rows == args.rows[0]:
                for b, e in failed.items():
                    print(f"   {b}: {e}", flush=True)
            results.append(row)
    if args.out:
        args.out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "results": results}, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
