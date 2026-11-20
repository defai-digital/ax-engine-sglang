"""Is every emitted token the target's greedy choice on the lane's own prefix? (Track L, GL1)

`adjudicate.py` looks at the first token where two lanes differ. After that point the lanes follow
different texts, so a verifier or state bug that shows up later (a wrong recurrent state after a
commit, a bad rollback, a cache slot reused across rounds) would not be seen. This script checks
every token of every answer instead: for each prompt it runs the target once (Hugging Face, fp32 by
default, no speculation) on prompt + the lane's whole output, teacher-forced, and at each output
position compares the lane's token with the reference argmax on that same prefix.

  mismatch  the lane's token is not the reference argmax
  regret    reference log-prob of the argmax minus that of the lane's token (0 on a match)

A lossless lane has mismatches only where the reference itself is nearly undecided (small regret),
at the rate plain decoding (AR, bf16) shows against the same fp32 reference. A lane whose verifier
or committed state is wrong emits tokens with large regret. Run it on AR and the stock chain too:
they are the controls that say what bf16 alone does.

    python benchmarks/sglang/prefix_audit.py --model Qwen/Qwen3.8-27B --out R/prefix_audit.json \
        ar=R/ar/ids.jsonl chain8n=R/chain8n/dflash/ids.jsonl tree=R/tree32cfl_b16/dftree/ids.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "benchmarks/baselines"))
from token_identity import prompt_hash                                # noqa: E402

BINS = (0.05, 0.1, 0.25, 0.5, 1.0)


def rows(path):
    return [json.loads(line) for line in open(path)]


def summarise(per_prompt):
    regrets = [r for p in per_prompt for r in p["regrets"]]
    tokens = sum(p["tokens"] for p in per_prompt)
    worst = max(per_prompt, key=lambda p: p["max_regret"], default=None)
    return {
        "prompts": len(per_prompt), "tokens": tokens, "mismatches": len(regrets),
        "mismatch_rate": len(regrets) / tokens if tokens else None,
        "prompts_all_match": sum(not p["regrets"] for p in per_prompt),
        "max_regret": max(regrets, default=0.0),
        **{f"regret_ge_{b}": sum(r >= b for r in regrets) for b in BINS},
        "worst": {k: worst[k] for k in ("set", "i", "max_regret", "max_regret_pos")} if worst else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lanes", nargs="+", help="name=ids.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--limit", type=int, default=10 ** 9, help="prompts per family")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import load_prompt_sets

    lanes = dict(spec.split("=", 1) for spec in args.lanes)
    data = {name: rows(path) for name, path in lanes.items()}
    n = max(r["i"] for rs in data.values() for r in rs) + 1
    sets = load_prompt_sets(n)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).cuda().eval()

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        return tok(text, add_special_tokens=False).input_ids

    result = {"model": args.model, "dtype": args.dtype, "lanes": {}}
    for name, rs in data.items():
        per_prompt = []
        for r in rs:
            if r["i"] >= args.limit or not r["output_ids"]:
                continue
            ids = encode(sets[r["set"]][r["i"]])
            if r.get("prompt_sha1") and r["prompt_sha1"] != prompt_hash(ids):
                sys.exit(f"{name}: prompt {(r['set'], r['i'])} does not match the saved prompt hash")
            out = r["output_ids"]
            x = torch.tensor([ids + out], device="cuda")
            with torch.no_grad():
                # row t predicts the token after position t: rows len(ids)-1 .. predict out[0..]
                lp = model(x).logits[0, len(ids) - 1:len(ids) - 1 + len(out)].float().log_softmax(-1)
            got = torch.tensor(out, device="cuda")
            regret = lp.max(-1).values - lp.gather(-1, got[:, None])[:, 0]
            miss = (lp.argmax(-1) != got).nonzero()[:, 0]
            reg = regret[miss].tolist()
            per_prompt.append({
                "set": r["set"], "i": r["i"], "tokens": len(out), "regrets": reg, "positions": miss.tolist(),
                "max_regret": max(reg, default=0.0),
                "max_regret_pos": int(miss[regret[miss].argmax()]) if reg else None,
            })
        s = summarise(per_prompt)
        s["by_family"] = {f: summarise([p for p in per_prompt if p["set"] == f])
                          for f in sorted({p["set"] for p in per_prompt})}
        s["file"] = lanes[name]
        s["rows"] = per_prompt
        result["lanes"][name] = s
        print(name, json.dumps({k: v for k, v in s.items() if k not in ("rows", "by_family")}), flush=True)
    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
