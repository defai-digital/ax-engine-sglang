"""Opt-in SGLang forward hooks and matched-prefix analysis for NVFP4 diagnostics.

Use SGLang's --forward-hooks with target_modules=["logits_processor"] and
hook_factory="nvfp4_forward_probe:logits_hook", config={"run_dir": ...}.
Add only this script's directory to PYTHONPATH; the plugin can remain installed
from its wheel. Hooks record only while run_dir/active.json exists and skip CUDA
graph capture. Run with --disable-cuda-graph: graph replay does not execute Python
hooks. These synchronous probes are for correctness investigation, not timing.

Each run needs requests.json mapping labels to input_ids/output_ids. active.json
contains {"label": ...} for the one request in flight. Compare never aligns logits
whose speculative input history differs from the committed output prefix.
"""
import argparse
import json
from pathlib import Path
from uuid import uuid4


def _active(root):
    path = root / "active.json"
    return json.loads(path.read_text()) if path.exists() else None


def logits_hook(config):
    import os
    import torch

    root = Path(config["run_dir"])
    destination = root / "passes"
    destination.mkdir(parents=True, exist_ok=True)
    capture_id = uuid4().hex
    counter = 0

    def hook(module, inputs, output):
        nonlocal counter
        if torch.cuda.is_current_stream_capturing():
            return
        active = _active(root)
        if active is None or output.next_token_logits is None:
            return
        if len(inputs) < 4 or not hasattr(inputs[3], "positions"):
            raise RuntimeError("logits probe needs positional ForwardBatch metadata")
        batch = inputs[3]
        # SGLang also attaches these hooks to the MTP drafter. Its draft
        # prefill can share the target's EXTEND mode and process ID.
        spec_info = getattr(batch, "spec_info", None)
        if getattr(spec_info, "is_draft_input", lambda: False)():
            return
        if batch.batch_size != 1:
            raise RuntimeError("matched-prefix probe requires one request in flight")
        record = {
            "label": active["label"], "mode": batch.forward_mode.name,
            "input_ids": inputs[0].detach().cpu().clone(),
            "positions": batch.positions.detach().cpu().clone(),
            "logits": output.next_token_logits.detach().cpu().clone(),
        }
        torch.save(record, destination / f"pass-{os.getpid()}-{capture_id}-{counter:06d}.pt")
        counter += 1

    return hook


def linear_hook(config):
    """Replay a real quantized linear with the exact same input row at several M.

    The original output is returned unchanged. module.forward bypasses this
    module's registered hooks; linear projections have no KV/recurrent state.
    """
    import torch

    root = Path(config["run_dir"])
    destination = root / "linear"
    destination.mkdir(parents=True, exist_ok=True)
    counter = 0

    def hook(module, inputs, output):
        nonlocal counter
        if torch.cuda.is_current_stream_capturing() or counter >= config.get("trials", 3):
            return
        active = _active(root)
        if active is None:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.shape[0]:
            raise RuntimeError("linear probe needs a nonempty unquantized input matrix")
        row = x[-1:].detach().clone()
        original = output[0] if isinstance(output, tuple) else output
        original_snapshot = original.detach().clone()
        original_row = original_snapshot[-1:].float()

        def run(value):
            result = module.forward(value)
            return result[0] if isinstance(result, tuple) else result

        with torch.inference_mode():
            reference = run(row).float()
            repeat = run(row).float()
            measurements = []
            for rows in sorted({1, 2, 4, 8, 16, 32, 64, 128, x.shape[0]}):
                actual = run(row.expand(rows, -1).contiguous())[:1].float()
                delta = actual - reference
                measurements.append({"rows": rows, "max_abs": float(delta.abs().max()),
                                     "rms": float(delta.square().mean().sqrt()),
                                     "equal": bool(torch.equal(actual, reference))})
            if not torch.equal(original, original_snapshot):
                raise RuntimeError("probe replays overwrote the original output; diagnostic run is invalid")
            report = {"label": active["label"], "name": config["name"],
                      "quant_method": type(module.quant_method).__name__,
                      "original_rows": x.shape[0], "input_dtype": str(x.dtype),
                      "repeat_equal": bool(torch.equal(reference, repeat)),
                      "original_equal_to_single": bool(torch.equal(original_row, reference)),
                      "original_max_abs_vs_single": float((original_row - reference).abs().max()),
                      "original_output_preserved": True,
                      "measurements": measurements}
        (destination / f"{config['name']}-{counter:03d}.json").write_text(json.dumps(report, indent=2) + "\n")
        counter += 1

    return hook


def mlp_hook(config):
    """Apply the same row-count control to gate, activation/quantization and down.

    A proxy exposes the dense MLP's stateless forward and projection quantizer
    without adding attributes or hooks to the live model's child modules.
    """
    from types import SimpleNamespace

    callback = linear_hook(config)

    def hook(module, inputs, output):
        proxy = SimpleNamespace(forward=module.forward, quant_method=module.gate_up_proj.quant_method)
        return callback(proxy, inputs, output)

    return hook


def aligned_rows(input_ids, positions, logits_rows, prompt, output):
    """Return (logit row, predicted output offset, full conditioning prefix)."""
    if len(input_ids) != len(positions):
        raise ValueError("input IDs and logical positions have different lengths")
    if any(b != a + 1 for a, b in zip(positions, positions[1:])):
        raise ValueError("probe only supports contiguous chain positions, not a tree")
    if logits_rows == len(input_ids):
        indices = list(range(logits_rows))
    elif logits_rows == 1 and input_ids:
        indices = [len(input_ids) - 1]  # prefill returns only the final logits
    else:
        raise ValueError("cannot align logits rows with the recorded inputs")
    context = prompt + output
    result = []
    for row, index in enumerate(indices):
        position = positions[index]
        generated = position + 1 - len(prompt)
        if position < 0 or position >= len(context) or not 0 <= generated < len(output):
            continue
        # A rejected earlier speculative token invalidates every later row,
        # even when a later token happens to equal the committed output again.
        if any(p < 0 or p >= len(context) or token != context[p]
               for token, p in zip(input_ids[:index + 1], positions[:index + 1])):
            continue
        result.append((row, generated, tuple(context[:position + 1])))
    return result


def load_run(root):
    import torch

    requests = json.loads((root / "requests.json").read_text())
    rows = {}
    for path in sorted((root / "passes").glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        label = record["label"]
        request = requests[label]
        scores = record["logits"]
        if scores.ndim != 2 or not bool(torch.isfinite(scores).all()):
            raise ValueError(f"invalid logits: {path}")
        for index, generated, prefix in aligned_rows(
                record["input_ids"].tolist(), record["positions"].tolist(), len(scores),
                request["input_ids"], request["output_ids"]):
            key = (label, prefix)
            if key in rows:
                raise ValueError(f"duplicate conditioning prefix: {path}, {label}/{generated}")
            rows[key] = {"generated": generated, "scores": scores[index].clone(),
                         "mode": record["mode"], "rows": len(scores)}
    if not rows:
        raise ValueError(f"no aligned rows found: {root}")
    first_rows = {label for (label, _), row in rows.items() if row["generated"] == 0}
    if first_rows != set(requests):
        raise ValueError("each request needs its captured first prediction")
    return rows


def compare(reference, candidate):
    import torch

    a, b = load_run(reference), load_run(candidate)
    first_a = {label: prefix for (label, prefix), row in a.items() if row["generated"] == 0}
    first_b = {label: prefix for (label, prefix), row in b.items() if row["generated"] == 0}
    if first_a != first_b:
        raise ValueError("runs must contain the same labeled prompts")
    matched = sorted(set(a) & set(b))
    if not matched:
        raise ValueError("runs have no identical conditioning prefixes")
    records = []
    for key in matched:
        x, y = a[key]["scores"].float(), b[key]["scores"].float()
        if x.shape != y.shape:
            raise ValueError("logits have different vocabulary sizes")
        ta, tb = int(x.argmax()), int(y.argmax())
        delta = y - x
        records.append({"label": key[0], "generated": a[key]["generated"],
                        "reference_mode": a[key]["mode"], "candidate_mode": b[key]["mode"],
                        "reference_rows": a[key]["rows"], "candidate_rows": b[key]["rows"],
                        "reference_top": ta, "candidate_top": tb,
                        "max_abs": float(delta.abs().max()),
                        "centered_rms": float((delta - delta.mean()).square().mean().sqrt()),
                        "softmax_total_variation": float((x.softmax(0) - y.softmax(0)).abs().sum() / 2),
                        "candidate_regret_in_reference": float(x.max() - x[tb]),
                        "reference_regret_in_candidate": float(y.max() - y[ta])})
    return {"completed": True, "scope": "eager logits at identical committed prefixes; observational only",
            "matched_prefixes": len(records),
            "argmax_mismatches": sum(r["reference_top"] != r["candidate_top"] for r in records),
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
