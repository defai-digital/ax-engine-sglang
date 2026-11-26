"""Time SGLang's NVFP4 Marlin W4A16 GEMM at verify-row counts with each tile configuration (Hopper).

On a GPU without FP4 tensor cores (SM90) SGLang serves NVFP4 checkpoints through Marlin W4A16
(`--fp4-gemm-backend marlin`). Marlin picks its tile from a fixed priority list without timing the
alternatives (`determine_exec_config` in sglang/kernels/jit/csrc/gemm/marlin/gptq_marlin.cuh). This
script compiles a copy of that source whose tile (thread_k, thread_n) and parallelism (SM count) can be
forced with AX_MARLIN_TK / AX_MARLIN_TN / AX_MARLIN_SMS, loads one decoder layer's NVFP4 MLP weights
from the checkpoint, prepares them as SGLang does, and for each row count times every valid
configuration against the automatic one. It reports whether each candidate's output equals the
automatic choice bit for bit (a different K split changes the reduction order).

    python benchmarks/sglang/marlin_fp4_tune.py --model /root/m/27b-nvfp4 --rows 1 8 16 32 64 128 \\
        --out marlin-tune.json

Timing uses CUDA graphs of `--reps` back-to-back GEMMs, so launch overhead is excluded as in serving.
The GEMMs cycle through copies of the weights totalling at least four times the L2 cache, so every
GEMM streams its weights from device memory as in a forward pass over 64 layers (one copy would stay in
the H200's 50 MB L2 for the 45 MB down projection).
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
from types import SimpleNamespace

import torch

TILES = ((128, 128), (64, 128), (128, 64), (64, 256))   # (thread_k, thread_n) of Marlin's config tables
OVERRIDE = """  int thread_k_init = -1;
  int thread_n_init = -1;
"""
PATCHED = """  int thread_k_init = std::getenv("AX_MARLIN_TK") ? std::atoi(std::getenv("AX_MARLIN_TK")) : -1;
  int thread_n_init = std::getenv("AX_MARLIN_TN") ? std::atoi(std::getenv("AX_MARLIN_TN")) : -1;
  if (std::getenv("AX_MARLIN_SMS")) sms = std::min(sms, std::atoi(std::getenv("AX_MARLIN_SMS")));
"""


def tunable_module(workdir: Path):
    """Compile gptq_marlin.cuh with the config override (a separate JIT module, same kernels)."""
    import sglang.kernels.jit as jit
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    source = Path(jit.__file__).parent / "csrc/gemm/marlin"
    target = workdir / "marlin"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    entry = target / "gptq_marlin.cuh"
    text = entry.read_text()
    if text.count(OVERRIDE) != 1:
        raise SystemExit("gptq_marlin.cuh changed: the thread_k_init / thread_n_init defaults were not found")
    entry.write_text("#include <cstdlib>\n#include <algorithm>\n" + text.replace(OVERRIDE, PATCHED))
    args = make_cpp_args(torch.bfloat16)
    return load_jit("ax_gptq_marlin_tune", *args, cuda_files=[str(entry)],
                    cuda_wrappers=[("gptq_marlin_gemm", f"gptq_marlin_gemm<{args}>")])


def read_tensors(model: Path, names):
    """Named tensors from the checkpoint's safetensors shards (no full load)."""
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    dtypes = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16}
    out = {}
    for name in names:
        path = model / index[name]
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
            meta = header[name]
            start, end = meta["data_offsets"]
            handle.seek(8 + size + start)
            raw = bytearray(handle.read(end - start))
        out[name] = torch.frombuffer(raw, dtype=dtypes[meta["dtype"]]).reshape(meta["shape"]).clone()
    return out


def prepared_layer(tensors, prefix, parts):
    """One Marlin layer from NVFP4 parts (gate + up are fused along N as SGLang's MergedColumnParallelLinear)."""
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_nvfp4_layer_for_marlin

    weight = torch.cat([tensors[f"{prefix}.{p}.weight"] for p in parts]).cuda()
    scale = torch.cat([tensors[f"{prefix}.{p}.weight_scale"] for p in parts]).cuda()
    # One global scale per fused layer, as SGLang's ModelOpt NVFP4 loader keeps (the largest part's)
    global_scale = torch.stack([tensors[f"{prefix}.{p}.weight_scale_2"].reshape(()) for p in parts]).max()
    layer = SimpleNamespace(output_size_per_partition=weight.shape[0], input_size_per_partition=weight.shape[1] * 2,
                            params_dtype=torch.bfloat16, weight=weight, weight_scale=scale,
                            weight_global_scale=global_scale.cuda().to(torch.bfloat16), bias=None)
    prepare_nvfp4_layer_for_marlin(layer)
    return layer


def copies_of(layer, count):
    """`count` layers with distinct weight memory (the workspace is shared: the GEMMs run in order)."""
    return [layer] + [SimpleNamespace(weight=layer.weight.clone(), weight_scale=layer.weight_scale.clone(),
                                      weight_global_scale=layer.weight_global_scale.clone(),
                                      workspace=layer.workspace) for _ in range(count - 1)]


def copy_count(layer):
    size = sum(t.numel() * t.element_size() for t in (layer.weight, layer.weight_scale))
    return max(2, -(-4 * torch.cuda.get_device_properties(0).L2_cache_size // size))


def gemm(module, layers, x, fp32_reduce):
    from sglang.kernels.ops.gemm.gptq_marlin import _MAX_THREAD_N
    from sglang.srt.layers.quantization.utils import get_scalar_types

    _, scalar_types = get_scalar_types()
    m, k = x.shape
    n = layers[0].weight.size(1) * 8 // 16
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    c_tmp = (torch.empty(sms * min((m + 15) // 16 * 16, 64) * _MAX_THREAD_N, dtype=torch.float32, device=x.device)
             if fp32_reduce else torch.empty(0, dtype=torch.float32, device=x.device))
    empty_i = torch.empty(0, dtype=torch.int32, device=x.device)

    a_tmp = torch.empty(0, dtype=x.dtype, device=x.device)

    def run(i=0):
        layer = layers[i % len(layers)]
        module.gptq_marlin_gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, empty_i, empty_i,
                                empty_i, out, c_tmp, a_tmp, layer.workspace, scalar_types.float4_e2m1f.id, True,
                                False, fp32_reduce, False)
        return out
    return run


def time_graph(run, reps, trials):
    run()
    torch.cuda.synchronize()
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128])
    parser.add_argument("--sms", type=int, nargs="*", default=[], help="also try these SM counts (fewer blocks)")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/ax-marlin-tune"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)
    args.workdir.mkdir(parents=True, exist_ok=True)
    module = tunable_module(args.workdir)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = next(re.sub(r"\.gate_proj\.weight$", "", name) for name in index
                  if re.search(rf"layers\.{args.layer}\.mlp\.gate_proj\.weight$", name))
    names = [f"{prefix}.{p}.{s}" for p in ("gate_proj", "up_proj", "down_proj")
             for s in ("weight", "weight_scale", "weight_scale_2")]
    tensors = read_tensors(args.model, names)
    layers = {"gate_up": prepared_layer(tensors, prefix, ("gate_proj", "up_proj")),
              "down": prepared_layer(tensors, prefix, ("down_proj",))}
    copies = {name: copies_of(layer, copy_count(layer)) for name, layer in layers.items()}
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{torch.cuda.get_device_name()} SMs {sms}; layer {args.layer}: "
          + ", ".join(f"{k} N={l.output_size_per_partition} K={l.input_size_per_partition} ({len(copies[k])} copies)"
                      for k, l in layers.items()),
          flush=True)

    results = []
    for name, layer in layers.items():
        k = layer.input_size_per_partition
        for rows in args.rows:
            x = (torch.randn(rows, k, device="cuda") * 0.5).to(torch.bfloat16)
            for key in ("AX_MARLIN_TK", "AX_MARLIN_TN", "AX_MARLIN_SMS"):
                os.environ.pop(key, None)
            reference = gemm(module, copies[name], x, True)().clone()
            base = time_graph(gemm(module, copies[name], x, True), args.reps, args.trials)
            row = {"layer": name, "rows": rows, "auto_us": base, "candidates": []}
            candidates = [(tk, tn, None, fp32) for tk, tn in TILES for fp32 in (True, False)]
            candidates += [(None, None, s, True) for s in args.sms] + [(None, None, None, False)]
            for tk, tn, sm, fp32 in candidates:
                env = {"AX_MARLIN_TK": tk, "AX_MARLIN_TN": tn, "AX_MARLIN_SMS": sm}
                for key, value in env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = str(value)
                try:
                    run = gemm(module, copies[name], x, fp32)
                    out = run().clone()
                    us = time_graph(run, args.reps, args.trials)
                except Exception as error:   # an invalid tile for this shape
                    row["candidates"].append({"tk": tk, "tn": tn, "sms": sm, "fp32_reduce": fp32,
                                              "error": str(error).splitlines()[0][:160]})
                    continue
                row["candidates"].append({"tk": tk, "tn": tn, "sms": sm, "fp32_reduce": fp32, "us": us,
                                          "speedup": base / us, "bit_identical": bool(torch.equal(out, reference)),
                                          "max_abs_diff": float((out.float() - reference.float()).abs().max())})
            for key in ("AX_MARLIN_TK", "AX_MARLIN_TN", "AX_MARLIN_SMS"):
                os.environ.pop(key, None)
            best = max((c for c in row["candidates"] if "us" in c), key=lambda c: c["speedup"])
            print(f"{name:7s} M={rows:4d} auto {base:8.2f} us  best {best['us']:8.2f} us "
                  f"({best['speedup']:.3f}x tk={best['tk']} tn={best['tn']} sms={best['sms']} "
                  f"fp32={best['fp32_reduce']} identical={best['bit_identical']})", flush=True)
            results.append(row)
    if args.out:
        args.out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "sms": sms, "results": results},
                                       indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
