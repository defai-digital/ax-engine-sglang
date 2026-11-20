"""Build a markdown report from a results directory.

  python benchmarks/baselines/summarize.py results/run1 > results/run1/REPORT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main(root: Path):
    modes = sorted([d.name for d in root.iterdir() if d.is_dir()],
                   key=lambda m: (m != "ar", m))
    print(f"# Speed and quality report: `{root}`\n")
    for envf in sorted(root.glob("env*.json")):
        print(f"## Environment ({envf.name})\n\n```\n" + json.dumps(load(envf), indent=2) + "\n```\n")

    # ---- speed
    for preset in ("greedy", "instruct", "thinking", "qwen38"):
        speed = {m: load(root / m / f"speed_{preset}.json") for m in modes}
        speed = {m: s for m, s in speed.items() if s}
        if not speed:
            continue
        ar = speed.get("ar", {})
        base = {(r["prompt_set"], r["concurrency"]): r
                for r in ar.get("results", []) if not ar.get("diagnostic")}
        best_mtp: dict = {}   # best built-in MTP mode per (set, concurrency)
        for m, s in speed.items():
            if m.startswith("mtp") and not s.get("diagnostic"):
                for r in s["results"]:
                    k = (r["prompt_set"], r["concurrency"])
                    best_mtp[k] = max(best_mtp.get(k, 0.0), r["output_tok_s"])
        print(f"## Speed ({preset})\n")
        print("| mode | set | conc | out tok/s | per-req tok/s p50 | TTFT p50 (s) | accept len "
              "| vs ar | vs best MTP |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m, s in speed.items():
            diagnostic = s.get("diagnostic")
            label = f"{m} (diagnostic)" if diagnostic else m
            for r in s["results"]:
                b = None if diagnostic else base.get((r["prompt_set"], r["concurrency"]))
                sp = f"{r['output_tok_s'] / b['output_tok_s']:.2f}x" if b and b["output_tok_s"] else "-"
                acc = r.get("spec", {}).get("mean_acceptance_length")
                bm = None if diagnostic else best_mtp.get((r["prompt_set"], r["concurrency"]))
                vm = f"{r['output_tok_s'] / bm:.2f}x" if bm else "-"
                dec, ttft = r["per_req_decode_tok_s_p50"], r["ttft_s_p50"]
                print(f"| {label} | {r['prompt_set']} | {r['concurrency']} | {r['output_tok_s']:.1f} | "
                      f"{f'{dec:.1f}' if dec is not None else '-'} | {f'{ttft:.3f}' if ttft is not None else '-'} | "
                      f"{f'{acc:.2f}' if acc else '-'} | {sp} | {vm} |")
        print()

    # ---- quality + greedy parity
    for preset in ("greedy", "instruct", "thinking"):
        q = {m: load(root / m / f"quality_{preset}.json") for m in modes}
        q = {m: v for m, v in q.items() if v}
        if not q:
            continue
        tasks = sorted({t for v in q.values() for t in v["scores"]})
        print(f"## Quality ({preset})\n")
        print("| mode | " + " | ".join(tasks) + " |")
        print("|---|" + "---|" * len(tasks))
        for m, v in q.items():
            cells = [f"{v['scores'][t]['accuracy']:.3f}" if t in v["scores"] else "-" for t in tasks]
            print(f"| {m} | " + " | ".join(cells) + " |")
        print()
        if preset == "greedy" and "ar" in q:
            print("### Greedy output parity vs `ar` (fraction of byte-identical outputs)\n")
            print("| mode | " + " | ".join(tasks) + " |")
            print("|---|" + "---|" * len(tasks))
            for m in q:
                if m == "ar":
                    continue
                cells = []
                for t in tasks:
                    a, b = root / "ar" / f"samples_{t}_greedy.jsonl", root / m / f"samples_{t}_greedy.jsonl"
                    if not (a.exists() and b.exists()):
                        cells.append("-")
                        continue
                    ra = {j["id"]: j["output"] for j in map(json.loads, a.open())}
                    rb = {j["id"]: j["output"] for j in map(json.loads, b.open())}
                    common = ra.keys() & rb.keys()
                    same = sum(ra[k] == rb[k] for k in common)
                    cells.append(f"{same / len(common):.3f}" if common else "-")
                print(f"| {m} | " + " | ".join(cells) + " |")
            print("\nNote: speculative decoding is lossless in exact arithmetic, but batched GPU "
                  "numerics can flip near-tied greedy tokens. Compare accuracy too, not only byte parity.\n")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/latest"))
