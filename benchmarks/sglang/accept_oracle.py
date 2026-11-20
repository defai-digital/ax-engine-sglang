"""Independent accept oracle for the multi-user Stage 0 checks (docs/decisions/2026-11-15-multiuser-width-policy.md).

The ax-engine-sglang server, run with AX_ROUND_DUMP=<file>, writes one line per request and decode
round: request id, round kind (tree | chain), batch size, the verified tokens (anchor first; for a
chain round the W drafted columns, for a tree round the accepted path) and the accepted length.
This script recomputes every round's acceptance from the target alone:

  1. Per request, the committed text is the concatenation over its rounds of tokens[0 .. accept]
     (each round's anchor is the previous round's bonus token; the first anchor is the prefill
     token). The request is matched to its prompt through the saved outputs (ids*.jsonl of
     token_identity.py collect). Each saved response needs a matching dump; extra requests
     such as warmup are skipped.
  2. One teacher-forced Hugging Face pass (bf16, the serving precision) over prompt + committed
     text gives the target's greedy choice at every position.
  3. Check the accepted drafts and the bonus against the greedy prediction at each committed
     position. In a chain round, also check the first rejected draft against the prediction
     for the bonus. Later rejected drafts would need a different teacher-forced prefix.
     Positions clipped by the output budget or stop condition are excluded.

A mismatch where the server and this bf16 reference disagree on a near-tie is floating-point noise;
the summary reports each mismatch's regret (log-prob of the reference's argmax minus that of the
server's token at that position) so the pass line (regret under 0.5 nats) can be applied.

    python benchmarks/sglang/accept_oracle.py --rounds R/rounds.jsonl --ids R/ids.jsonl R/ids_c4.jsonl \
        --model Qwen/Qwen3.8-27B --out R/oracle.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "benchmarks/baselines"))
from token_identity import prompt_hash   # noqa: E402


def load_rounds(path):
    by_rid, last = {}, {}
    with open(path) as source:
        for number, line in enumerate(source, start=1):
            row = json.loads(line)
            rid = row["rid"]
            index = row.get("round")
            if index is not None:
                if type(index) is not int or index <= last.get(rid, -1):
                    raise ValueError(f"{path}:{number}: non-increasing round number for {rid}")
                last[rid] = index
            by_rid.setdefault(rid, []).append(row)
    return by_rid


def committed_text(rounds):
    out, starts = [], []
    for r in rounds:
        starts.append(len(out))
        out.extend(r["tokens"][: r["accept"] + 1])
    return out, starts


def match_responses(by_rid, saved):
    """Match each saved response to a dumped request, including repeated passes."""
    expected = collections.Counter((key, tuple(out), sha) for key, out, sha in saved)
    matched = []
    for rid, rounds in by_rid.items():
        committed, starts = committed_text(rounds)
        prefix = tuple(committed)
        # The dump can include drafts clipped at EOS or the token budget.
        # Its only possible missing committed token is the final bonus.
        hits = [item for item, remaining in expected.items() if remaining and item[1] and prefix
                and (prefix[:len(item[1])] == item[1] or item[1][:-1] == prefix)]
        if len(hits) > 1:
            raise ValueError(f"{rid}: round dump matches more than one saved response")
        if hits:
            item = hits[0]
            expected[item] -= 1
            matched.append((rid, rounds, starts, item))
    if missing := expected.total():
        raise ValueError(f"{missing} saved responses have no matching round dump")
    return matched


def audit_round(round_data, start, output, greedy):
    """Return coverage and all token errors on this committed prefix."""
    tokens, accepted = round_data["tokens"], round_data["accept"]
    if not 0 <= accepted < len(tokens):
        raise ValueError(f"invalid accepted length: {accepted} for {len(tokens)} tokens")
    stop = min(start + accepted + 1, len(output))
    if start >= stop:
        return 0, [], None
    accept_errors = []
    for pos in range(start + 1, stop):
        if output[pos] != greedy[pos]:
            error = {"pos": pos, "server_token": output[pos]}
            if not accept_errors:
                error["oracle_accept"] = pos - start - 1
            accept_errors.append(error)
    bonus_error = None
    if stop < len(output):
        bonus, target = output[stop], greedy[stop]
        if bonus != target:
            bonus_error = {"pos": stop, "server_token": bonus, "bonus": True}
        if (not accept_errors and round_data["kind"] == "chain"
                and accepted + 1 < len(tokens) and tokens[accepted + 1] == target):
            accept_errors.append({
                "pos": stop, "server_token": bonus, "oracle_accept": accepted + 1,
                "oracle_accept_is_lower_bound": True, "structural": bonus == target,
            })
    return stop - start - 1 + int(stop < len(output)), accept_errors, bonus_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True)
    ap.add_argument("--ids", nargs="+", required=True, help="ids*.jsonl files the server produced")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--tie", type=float, default=0.5, help="regret (nats) below which a mismatch counts as a near-tie")
    ap.add_argument("--max-rate", type=float, default=0.005, help="largest allowed near-tie mismatch rate (exit 1 above)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import load_prompt_sets

    saved = []
    for f in args.ids:
        for line in open(f):
            r = json.loads(line)
            saved.append(((r["set"], r["i"]), r["output_ids"], r.get("prompt_sha1")))
    n = max(k[1] for k, _, _ in saved) + 1
    sets = load_prompt_sets(n)
    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        return tok(text, add_special_tokens=False).input_ids

    by_rid = load_rounds(args.rounds)
    matches = match_responses(by_rid, saved)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    stats = collections.defaultdict(lambda: {"rounds": 0, "accept_wrong": 0, "bonus_wrong": 0, "checked_tokens": 0})
    mismatches = []
    for rid, rounds, starts, ((fam, i), out, sha) in matches:
        ids = encode(sets[fam][i])
        if sha and sha != prompt_hash(ids):
            sys.exit(f"prompt {(fam, i)} does not match the saved prompt hash")
        x = torch.tensor([[*ids, *out]], device="cuda")
        with torch.no_grad():
            logits = model(x).logits[0, len(ids) - 1:-1].float()
        best, greedy_ids = logits.max(-1)
        if not bool(torch.isfinite(best).all()):
            raise ValueError(f"{rid}: invalid teacher logits")
        greedy = greedy_ids.tolist()

        for r, s in zip(rounds, starts):
            kind = r["kind"]
            toks, acc = r["tokens"], r["accept"]
            key = (kind, len(toks) if kind == "chain" else "tree", r["bs"] > 1, r.get("backend"), r.get("graph"))
            st = stats[key]
            st["rounds"] += 1
            checked, accept_errors, bonus_error = audit_round(r, s, out, greedy)
            st["checked_tokens"] += checked
            st["accept_wrong"] += int(bool(accept_errors))
            st["bonus_wrong"] += int(bonus_error is not None)
            errors = {e["pos"]: e for e in accept_errors}
            if bonus_error:
                errors.setdefault(bonus_error["pos"], bonus_error)["bonus"] = True
            for pos, error in errors.items():
                mismatches.append({"rid": rid, "set": fam, "i": i, "kind": kind, "bs": r["bs"],
                                   "server_accept": acc, **error,
                                   "regret": float(best[pos] - logits[pos, error["server_token"]])})
        del logits
    table = [{"kind": k[0], "width": k[1], "several_requests": k[2], "backend": k[3], "graph": k[4], **v}
             for k, v in sorted(stats.items(), key=str)]
    worst = max((m["regret"] for m in mismatches), default=0.0)
    summary = {"rounds_file": args.rounds, "requests_matched": len(matches), "requests_skipped": len(by_rid) - len(matches),
               "by_kind": table, "mismatches": len(mismatches),
               "mismatches_over_tie": sum(m["regret"] >= args.tie for m in mismatches),
               "structural_mismatches": sum(m.get("structural", False) for m in mismatches),
               "tie_threshold": args.tie, "worst_regret": worst, "rows": mismatches}
    tokens = sum(v["checked_tokens"] for v in stats.values())
    summary["checked_tokens"] = tokens
    summary["passed"] = (tokens > 0 and summary["mismatches_over_tie"] == 0
                         and summary["structural_mismatches"] == 0
                         and len(mismatches) / tokens <= args.max_rate)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
