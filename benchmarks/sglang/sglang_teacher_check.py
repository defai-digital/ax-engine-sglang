"""Teacher-forced check of committed text through a stock SGLang server (fresh prefill / NVFP4 oracle).

docs/decisions/2026-11-16-dual-attention-backend.md, section 5 items 2 and 5. Every output saved by
`token_identity.py collect` (ids*.jsonl: prompt index, output token ids, prompt hash) is sent back
as one prefill, prompt + committed text, to a *stock* server that serves the same checkpoint
without radix cache. The server returns, per committed position, the log-probability of the
committed token and its top-2 (`return_logprob`, `logprob_start_len`, `top_logprobs_num`). At every
position the committed token must be the reference's greedy choice; when it is not, the regret
(reference best minus committed log-probability, nats) decides: under `--tie` (0.5) a near-tie,
otherwise a failure. The run fails on any mismatch over the tie, on a mismatch rate over
`--max-rate` (0.5%, the Stage 0 lanes measured 0.36-0.48% against Hugging Face), or when the
reference server's log does not show radix cache off and zero cached tokens on every prefill.
SGLang returns the input log-probabilities with an unscored leading row (the first position of the
requested span), so the span starts at the last prompt token and that row is dropped. This needs no Hugging Face model, so it works for NVFP4 as well as bf16, and it
recomputes the whole prefix in one prefill: the reference server's log must show `#cached-token: 0`
on every prefill (checked with --server-log).

    python benchmarks/sglang/sglang_teacher_check.py --base-url http://127.0.0.1:30001 --server-log REF/server.log \
        --ids LANE/ids.jsonl LANE/ids_c4.jsonl --model Qwen/Qwen3.8-27B --out LANE/teacher.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "benchmarks/baselines"))
from token_identity import post, prompt_hash   # noqa: E402

# one log line per prefill batch, which can hold several requests
CACHED = re.compile(r"Prefill batch, #new-seq: (\d+),.*?#cached-token: (\d+)")


def score_response(meta, output, prompt_length, label):
    finish = meta.get("finish_reason")
    finish = finish.get("type") if isinstance(finish, dict) else finish
    if finish not in ("stop", "length") or meta.get("prompt_tokens") != prompt_length:
        raise ValueError(f"{label}: teacher did not finish scoring the full prompt")
    # The requested span starts at the last prompt token. Its first row is unscored.
    lps, tops = meta["input_token_logprobs"][1:], meta["input_top_logprobs"][1:]
    if len(lps) != len(output) or len(tops) != len(output):
        raise ValueError(f"{label}: teacher logprob length does not match the committed output")
    mismatches = []
    for pos, (token, lp, top) in enumerate(zip(output, lps, tops)):
        if not lp or lp[0] is None or lp[1] != token or not top:
            raise ValueError(f"{label}: teacher logprob row {pos} does not match token {token}")
        committed_lp = float(lp[0])
        top_scores = [(float(entry[0]), entry[1]) for entry in top]
        if (not math.isfinite(committed_lp) or committed_lp > 0
                or any(not math.isfinite(score) or score > 0 for score, _ in top_scores)):
            raise ValueError(f"{label}: non-finite or positive teacher logprob at token {pos}")
        # Preserve the teacher's tie order when multiple entries have the same score.
        best_lp, best_tok = max(top_scores, key=lambda entry: entry[0])
        if best_tok != token:
            regret = best_lp - committed_lp
            if regret < 0:
                raise ValueError(f"{label}: committed score exceeds the teacher's top score at token {pos}")
            mismatches.append({"pos": pos, "committed": token, "best": best_tok, "regret": regret})
    return mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="stock server, same checkpoint, radix cache off")
    ap.add_argument("--server-log", required=True, help="its log, to check that no prefill reused cached tokens")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B", help="tokenizer for the prompts")
    ap.add_argument("--tie", type=float, default=0.5)
    ap.add_argument("--max-rate", type=float, default=0.005, help="largest allowed near-tie mismatch rate")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.workers < 1 or not math.isfinite(args.tie) or args.tie <= 0 or not 0 <= args.max_rate <= 1:
        ap.error("workers and tie threshold must be positive; --max-rate must be in [0, 1]")
    args.out.unlink(missing_ok=True)

    from transformers import AutoTokenizer
    from prompts import load_prompt_sets
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = []
    for f in args.ids:
        with open(f) as source:
            for line in source:
                rows.append((f, json.loads(line)))
    sets = load_prompt_sets(max(r["i"] for _, r in rows) + 1)

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        return tok(text, add_special_tokens=False).input_ids

    head = Path(args.server_log).read_text(errors="replace")
    # server_args are logged as a dict (0.5.23: 'disable_radix_cache': True) or as keywords (=True)
    if not re.search(r"disable_radix_cache'?\s*[:=]\s*True", head):
        raise SystemExit(f"{args.server_log}: the reference server does not run with --disable-radix-cache")
    if not re.search(r"speculative_algorithm'?\s*[:=]\s*None", head):
        raise SystemExit(f"{args.server_log}: the reference server is not a plain (non-speculative) server")
    log_off = Path(args.server_log).stat().st_size

    def one(item):
        f, r = item
        ids = encode(sets[r["set"]][r["i"]])
        if r.get("prompt_sha1") and r["prompt_sha1"] != prompt_hash(ids):
            raise SystemExit(f"{f}: prompt {r['set']}/{r['i']} does not match its saved hash")
        out = r["output_ids"]
        if not out:
            raise ValueError(f"{f}: empty output for {r['set']}/{r['i']}")
        body = {"input_ids": ids + out, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                "return_logprob": True, "logprob_start_len": len(ids) - 1, "top_logprobs_num": 2}
        meta = post(f"{args.base_url}/generate", body, retries=1)["meta_info"]
        return f, r, score_response(meta, out, len(ids) + len(out), f"{f}: {r['set']}/{r['i']}")

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, rows))

    text = Path(args.server_log).read_bytes()[log_off:].decode(errors="replace")
    batches = [(int(n), int(c)) for n, c in CACHED.findall(text)]
    cached = [c for _, c in batches]
    prefilled = sum(n for n, _ in batches)
    mism = [dict(file=f, set=r["set"], i=r["i"], **b) for f, r, bads in results for b in bads]
    tokens = sum(len(r["output_ids"]) for _, r, _ in results)
    over = [m for m in mism if m["regret"] >= args.tie]
    summary = {
        "outputs": len(results), "tokens": tokens, "mismatches": len(mism),
        "mismatch_rate": len(mism) / max(1, tokens), "over_tie": len(over), "tie": args.tie,
        "worst_regret": max((m["regret"] for m in mism), default=0.0),
        "max_rate": args.max_rate,
        "reference_prefill_batches": len(cached),
        "reference_prefilled_requests": prefilled,
        "reference_cached_tokens_max": max(cached) if cached else None,
        "passed": (not over and len(mism) / max(1, tokens) <= args.max_rate and tokens > 0
                   and prefilled >= len(results) and max(cached) == 0),
        "rows": mism,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
