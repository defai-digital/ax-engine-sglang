"""Pair two greedy speed runs prompt by prompt (benchmarks/baselines/bench_speed.py speed_greedy.json).

  python benchmarks/sglang/paired_speed.py OURS_JSON STOCK_JSON [--sets code math chat] [--conc 16]

Aggregate tok/s compares two servers fairly only if they generate the same text. With NVFP4 (and
any kernel-order change) greedy near-ties can resolve differently, so the two sides write different
answers; one extra answer that runs to max_tokens lengthens the tail of a small batch and moves the
aggregate by several percent. This splits each (set, concurrency) point into prompts whose outputs
have the same length on both sides (a proxy for identical text; token ids are not in the file) and
the rest, and reports for each group the median per-request decode tok/s ratio and the
decode-time-weighted ratio, next to the aggregate ratio.

Request record: [start_s, ttft_s, total_s, completion_tokens] (bench_speed.py).
"""
import argparse
import json
from pathlib import Path
import statistics as st


def points(path):
    d = json.loads(Path(path).read_text())
    if d.get("diagnostic"):
        raise SystemExit(f"{path} contains diagnostic timings")
    indexed = {(r["prompt_set"], r["concurrency"]): r for r in d["results"]}
    if len(indexed) != len(d["results"]):
        raise SystemExit(f"{path} contains duplicate prompt set and concurrency levels")
    return indexed


def decode_seconds(point):
    if "request_decode_seconds" in point:
        return point["request_decode_seconds"]
    # Older fixtures only record time to the final stream event.
    return [total - ttft if ttft is not None else None for _, ttft, total, _ in point["requests"]]


def group(ra, rb, ta, tb, idx):
    rows = [i for i in idx if ta[i] is not None and ta[i] > 0 and ra[i][3] > 1
            and tb[i] is not None and tb[i] > 0 and rb[i][3] > 1]
    if not rows:
        return None
    med = st.median(((ra[i][3] - 1) / ta[i]) / ((rb[i][3] - 1) / tb[i]) for i in rows)
    rate_a = sum(ra[i][3] - 1 for i in rows) / sum(ta[i] for i in rows)
    rate_b = sum(rb[i][3] - 1 for i in rows) / sum(tb[i] for i in rows)
    return med, rate_a / rate_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("stock")
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--conc", nargs="*", type=int)
    args = ap.parse_args()
    a, b = points(args.ours), points(args.stock)
    print("| set | c | aggregate | same-length prompts | median ratio | time-weighted | other prompts | median | time-weighted |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(set(a) & set(b), key=lambda k: (k[0], k[1])):
        s, c = key
        if (args.sets and s not in args.sets) or (args.conc and c not in args.conc):
            continue
        if a[key].get("prompt_sha256") != b[key].get("prompt_sha256"):
            raise SystemExit(f"{s} c={c}: ordered prompt digests must match")
        stock_rate = b[key]["output_tok_s"]
        agg = f"{a[key]['output_tok_s'] / stock_rate:.3f}" if stock_rate else "-"
        if "requests" not in a[key] or "requests" not in b[key]:
            print(f"| {s} | {c} | {agg} | no per-request records |")
            continue
        ra, rb = a[key]["requests"], b[key]["requests"]
        ta, tb = decode_seconds(a[key]), decode_seconds(b[key])
        if not len(ra) == len(rb) == len(ta) == len(tb):
            raise SystemExit(f"{s} c={c}: request and timing counts must match")
        n = len(ra)
        same = [i for i in range(n) if ra[i][3] == rb[i][3]]
        other = [i for i in range(n) if ra[i][3] != rb[i][3]]
        g1, g2 = group(ra, rb, ta, tb, same), group(ra, rb, ta, tb, other)
        f = lambda g: f"{g[0]:.3f} | {g[1]:.3f}" if g else "- | -"
        print(f"| {s} | {c} | {agg} | {len(same)}/{n} | {f(g1)} | {len(other)} | {f(g2)} |")


if __name__ == "__main__":
    main()
