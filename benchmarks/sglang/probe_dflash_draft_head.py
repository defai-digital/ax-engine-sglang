"""Compare graph-replayed draft head projection and top-k on checkpoint weights."""

import argparse
import json
from pathlib import Path
from statistics import median


def capture(torch, call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def time_graph(torch, graph):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(16):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from sglang.srt.layers.quantization import fp4_utils
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config, ModelOptFp4LinearMethod
    from sglang.srt.models.dflash import _radix_topk
    from ax_engine_sglang.ax_dflash_draft_vocab import (
        VOCAB_SIZE, SUBSET_SIZE, pack_nvfp4_head, pack_weight, subset_topk,
    )

    fp4_utils.FP4_GEMM_RUNNER_BACKEND = fp4_utils.Fp4GemmRunnerBackend.FLASHINFER_CUTLASS
    index = json.loads((args.target / "model.safetensors.index.json").read_text())["weight_map"]
    head = torch.nn.Module()
    for name in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
        key = "lm_head." + name
        with safe_open(args.target / index[key], framework="pt", device="cpu") as archive:
            tensor = archive.get_tensor(key).cuda()
            head.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    if head.weight.shape[0] != VOCAB_SIZE or head.weight.dtype != torch.uint8:
        raise ValueError("Expected the Qwen NVFP4 checkpoint head")
    head.quant_method = ModelOptFp4LinearMethod(
        ModelOptFp4Config.from_config({"quant_algo": "NVFP4", "group_size": 16}),
    )

    report = {"gpu": torch.cuda.get_device_name(), "seed": 42, "vocab": VOCAB_SIZE, "subset": SUBSET_SIZE,
              "input": "synthetic BF16 activations and checkpoint NVFP4 weights", "measurements": []}
    with torch.inference_mode():
        head.quant_method.process_weights_after_loading(head)
        reduced = pack_nvfp4_head(head)
        torch.manual_seed(report["seed"])
        for rows in (7, 14, 56):
            hidden = torch.randn((rows, head.weight.shape[1] * 2), device="cuda", dtype=torch.bfloat16)
            full_logits = head.quant_method.apply(head, hidden, None)
            reference = pack_weight(full_logits.T).T.contiguous()
            actual = reduced.quant_method.apply(reduced, hidden, None)
            ref_bits, actual_bits = reference.view(torch.int16), actual.view(torch.int16)
            values, _ = _radix_topk(reference, 16)
            full, _ = capture(torch, lambda: _radix_topk(head.quant_method.apply(head, hidden, None), 16))
            subset, (ids, chosen) = capture(torch, lambda: subset_topk(hidden, reduced, 16, _radix_topk))
            gathered = full_logits.gather(1, ids).view(torch.int16)
            chosen_bits = chosen.view(torch.int16)
            pairs = []
            for i in range(7):
                first, second = (full, subset) if i % 2 == 0 else (subset, full)
                pair = (time_graph(torch, first), time_graph(torch, second))
                pairs.append(pair if i % 2 == 0 else pair[::-1])
            report["measurements"].append({
                "rows": rows, "full_us": median(a for a, _ in pairs), "subset_us": median(b for _, b in pairs),
                "reduction_pct": 100 * (1 - median(b / a for a, b in pairs)), "pairs_us": pairs,
                "matching_logits": int((ref_bits == actual_bits).sum().item()), "total_logits": actual.numel(),
                "matching_mapped_scores": int((gathered == chosen_bits).sum().item()),
                "matching_topk_values": int((values.view(torch.int16) == chosen_bits).sum().item()),
                "total_topk": ids.numel(),
            })
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
