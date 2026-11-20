"""Who is right at the first differing token? An fp32 reference for greedy identity (Track L, GL1).

bf16 speculative decoding is lossless only up to floating-point differences, so "the tree lane
differs from AR at token 46" alone does not say which one left the exact greedy path. For every
prompt where two lanes differ, this script runs the target once in fp32 (Hugging Face, no
speculation, no batching) on prompt + the common prefix and reports, at the first differing
position:

  regret   per lane, fp32 log-prob of the fp32 argmax minus that of the lane's token (0 = the lane
           chose the fp32 winner)
  margin   |log-prob(a's token) - log-prob(b's token)| in fp32
  winner   which lane's token equals the fp32 argmax (a, b, or neither)
  tie      both lanes' tokens within TIE (default 0.05) of the fp32 winner: the expected kind of
           bf16 flip. A lane with a large regret left the greedy path for real (a bug, not noise).

    python benchmarks/sglang/adjudicate.py --a R/ar/ar/ids.jsonl --b R/tree32cf/dftree/ids.jsonl \
        --out R/adjudicate_ar_tree.json [--model Qwen/Qwen3.5-9B] [--limit 200]

Prompts are rebuilt exactly as token_identity.py collect builds them (benchmarks/baselines/prompts.py, chat
template, thinking off) and checked against the saved prompt_sha1 when the file has one.
A lane that flips only at near-ties, and wins about as often as AR, is lossless in the bf16 sense;
a lane that loses at large gaps is not.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "benchmarks/baselines"))
from token_identity import first_diff, prompt_hash                   # noqa: E402


def rows(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["set"], r["i"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="ids.jsonl of lane a (usually AR)")
    ap.add_argument("--b", required=True, help="ids.jsonl of lane b")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--tie", type=float, default=0.05, help="regret below which a token counts as tied")
    ap.add_argument("--limit", type=int, default=10 ** 9, help="max differing prompts to adjudicate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import load_prompt_sets

    A, B = rows(args.a), rows(args.b)
    if set(A) != set(B):
        sys.exit("the two files cover different prompts")
    n = max(i for _, i in A) + 1
    sets = load_prompt_sets(n)
    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        return tok(text, add_special_tokens=False).input_ids

    todo = []
    for k in sorted(A):
        d = first_diff(A[k]["output_ids"], B[k]["output_ids"])
        if d is not None:
            todo.append((k, d))
    print(f"{len(todo)} of {len(A)} prompts differ; adjudicating {min(len(todo), args.limit)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).cuda().eval()
    results = []
    for (fam, i), d in todo[:args.limit]:
        ids = encode(sets[fam][i])
        for r in (A[(fam, i)], B[(fam, i)]):
            if r.get("prompt_sha1") and r["prompt_sha1"] != prompt_hash(ids):
                sys.exit(f"prompt {(fam, i)} does not match the saved prompt hash; prompt sets changed")
        a_ids, b_ids = A[(fam, i)]["output_ids"], B[(fam, i)]["output_ids"]
        x = torch.tensor([ids + a_ids[:d]], device="cuda")
        with torch.no_grad():
            lp = model(x).logits[0, -1].float().log_softmax(-1)
        top = lp.topk(2)
        ta = a_ids[d] if d < len(a_ids) else None                     # None: that lane stopped here
        tb = b_ids[d] if d < len(b_ids) else None
        arg = int(top.indices[0])
        best = float(top.values[0])
        ra = best - float(lp[ta]) if ta is not None else None        # None: the lane ended here
        rb = best - float(lp[tb]) if tb is not None else None
        results.append({
            "set": fam, "i": i, "pos": d, "top2_gap": float(top.values[0] - top.values[1]),
            "argmax": arg, "a_tok": ta, "b_tok": tb, "a_regret": ra, "b_regret": rb,
            "margin": abs(ra - rb) if ra is not None and rb is not None else None,
            "winner": "a" if ta == arg else "b" if tb == arg else "neither",
            "tie": ra is not None and rb is not None and ra < args.tie and rb < args.tie,
        })
        print(json.dumps(results[-1]), flush=True)
    def regrets(key):
        return [r[key] for r in results if r[key] is not None]
    ra_all, rb_all = regrets("a_regret"), regrets("b_regret")
    summary = {
        "a": args.a, "b": args.b, "model": args.model, "dtype": args.dtype, "prompts": len(A),
        "differing": len(todo), "adjudicated": len(results),
        "a_wins": sum(r["winner"] == "a" for r in results), "b_wins": sum(r["winner"] == "b" for r in results),
        "neither": sum(r["winner"] == "neither" for r in results),
        "ties": sum(r["tie"] for r in results), "tie_threshold": args.tie,
        "a_regret_median": statistics.median(ra_all) if ra_all else None, "a_regret_max": max(ra_all, default=None),
        "b_regret_median": statistics.median(rb_all) if rb_all else None, "b_regret_max": max(rb_all, default=None),
        "a_regret_gt_tie": sum(x >= args.tie for x in ra_all), "b_regret_gt_tie": sum(x >= args.tie for x in rb_all),
        "rows": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
