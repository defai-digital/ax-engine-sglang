"""Build a draft-vocabulary token map for SGLang's --speculative-token-map.

The map lists the most frequent token ids in assistant text from training splits that the speed
benchmark does not use (UltraChat train_sft replies, GSM8K train answers), plus every added or
special token. SGLang slices the draft lm_head to these rows and maps the draft's choices back to
full-vocabulary ids before verification, so the target still verifies against the full
vocabulary; a token outside the map can only be produced by the target, never proposed.

    python benchmarks/sglang/build_token_map.py --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
        --sizes 16384 32768 --out-dir /w/token_map --coverage results/<run>/ar/ar/ids.jsonl

--coverage reports, for each map size, the fraction of generated tokens in the given ids.jsonl
files that fall inside the map (an upper bound on what the reduced draft can propose).
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def corpus(ultrachat_rows, gsm8k_rows):
    from datasets import load_dataset

    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    for ex in uc.select(range(min(ultrachat_rows, len(uc)))):
        for msg in ex["messages"]:
            if msg["role"] == "assistant":
                yield "chat", msg["content"]
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for ex in gsm.select(range(min(gsm8k_rows, len(gsm)))):
        yield "math", ex["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="tokenizer source")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32768])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ultrachat-rows", type=int, default=40000)
    parser.add_argument("--gsm8k-rows", type=int, default=7473)
    parser.add_argument("--coverage", nargs="*", default=[], help="ids.jsonl files to measure")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    counts = Counter()
    sources = Counter()
    for source, text in corpus(args.ultrachat_rows, args.gsm8k_rows):
        counts.update(tok.encode(text, add_special_tokens=False))
        sources[source] += 1
    forced = set(tok.all_special_ids) | set(tok.added_tokens_decoder)
    ranked = [t for t, _ in counts.most_common() if t not in forced]
    print(f"corpus: {dict(sources)} texts, {sum(counts.values())} tokens, "
          f"{len(counts)} distinct, {len(forced)} forced special/added ids")

    rows = []
    for path in args.coverage:
        rows += [json.loads(line) for line in open(path)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        keep = sorted(forced | set(ranked[: max(0, size - len(forced))]))
        out = args.out_dir / f"hot_{size}.pt"
        torch.save(keep, out)
        line = f"size {size}: {len(keep)} ids -> {out}"
        if rows:
            keep_set = set(keep)
            by_set = {}
            for r in rows:
                hit, total = by_set.get(r["set"], (0, 0))
                ids = r["output_ids"]
                by_set[r["set"]] = (hit + sum(i in keep_set for i in ids), total + len(ids))
            line += "  coverage " + " ".join(f"{s}={h / t:.4f}" if t else f"{s}=n/a"
                                             for s, (h, t) in sorted(by_set.items()))
        print(line, flush=True)


if __name__ == "__main__":
    main()
