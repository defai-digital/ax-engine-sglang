"""Derive a byte-preserving Qwen4/Flash Next MTP snapshot from local safetensors.

Keep checkpoints outside the repository. The source must already be a complete,
pinned snapshot. This selects the same MTP/embedding names as SGLang's MTP loader;
it does not quantize weights or change model configuration.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil


METADATA = (
    "config.json", "generation_config.json", "hf_quant_config.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
    "special_tokens_map.json", "chat_template.jinja", "README.md", "LICENSE", "LICENSE.txt",
)
EMBEDDINGS = {"model.embed_tokens.weight", "model.language_model.embed_tokens.weight"}


def tensor_digest(tensor):
    import torch
    return hashlib.sha256(memoryview(tensor.reshape(-1).view(torch.uint8).numpy())).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-id", required=True, help="declared Hub repository identity")
    parser.add_argument("--revision", required=True, help="pinned 40-digit commit SHA")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be a pinned 40-digit commit SHA")
    config = json.loads((args.source / "config.json").read_text())
    if config.get("model_type") not in ("qwen4_exp", "qwen4_exp_text"):
        parser.error("this derivation supports only Qwen4/Flash Next MTP checkpoints")
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    groups = defaultdict(list)
    for name, shard in index["weight_map"].items():
        if "mtp" in name or name in EMBEDDINGS:
            if Path(shard).name != shard or not shard.endswith(".safetensors"):
                parser.error(f"unexpected source shard path: {shard}")
            groups[shard].append(name)
    names = {name for values in groups.values() for name in values}
    if not any("mtp" in name for name in names) or not names.intersection(EMBEDDINGS):
        parser.error("source must contain MTP weights and a token embedding")
    args.out.mkdir(parents=True, exist_ok=False)
    from safetensors import safe_open
    from safetensors.torch import save_file
    weights, manifest = {}, {}
    total = 0
    for shard, selected in sorted(groups.items()):
        with safe_open(str(args.source / shard), framework="pt", device="cpu") as source:
            for name in sorted(selected):
                value = source.get_tensor(name)
                size = value.numel() * value.element_size()
                weights[name] = value
                manifest[name] = {"source_shard": shard, "shape": list(value.shape),
                                  "dtype": str(value.dtype), "bytes": size,
                                  "sha256": tensor_digest(value)}
                total += size
        print(f"Selected {len(selected)} tensors from {shard}", flush=True)
    for name in METADATA:
        source = args.source / name
        if source.is_file():
            shutil.copyfile(source, args.out / name)
    output = args.out / "model.safetensors"
    save_file(weights, str(output), metadata={"format": "pt"})
    with safe_open(str(output), framework="pt", device="cpu") as restored:
        if set(restored.keys()) != names:
            raise RuntimeError("serialized tensor names do not match the selected source")
        for name in restored.keys():
            if tensor_digest(restored.get_tensor(name)) != manifest[name]["sha256"]:
                raise RuntimeError(f"serialized tensor hash differs: {name}")
    (args.out / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total},
        "weight_map": {name: output.name for name in sorted(names)},
    }, indent=2) + "\n")
    # This completion record is written only after every serialized tensor passes.
    (args.out / "derivation.json").write_text(json.dumps({
        "source": args.source_id, "revision": args.revision.lower(),
        "tensors": len(weights), "bytes": total, "weights": manifest,
    }, indent=2) + "\n")
    print(f"Verified {len(weights)} tensors, {total} bytes: {args.out}", flush=True)


if __name__ == "__main__":
    main()
