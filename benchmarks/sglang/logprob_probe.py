"""Next-token probabilities a server reports at output position 1, for a fixed first output token.

  python benchmarks/sglang/logprob_probe.py --base-url URL --model MODEL --out probe.json [--n 200]

Sampled speculative decoding is lossless only if the verify forward's probabilities equal the plain
decode step's. Position 0 of an answer comes from the prefill; position 1 is the first token the
speculative verify decides. This sends each prompt of sampling_dist_check.PROMPTS N times (sampled,
three tokens, top-20 logprobs), keeps the answers whose first token is the most common first
token (or the one given with --first), and records the mean over requests of the top-20
probabilities at position 1 as SGLang reports them, plus the spread of the top token's value across
requests. Requests with return_logprob run chain verify on this plugin (ax_chain.tree_round), so
the probe measures stock DFlash verify, never the tree. Comparing the file from a plain decoding server with the file from
a speculative server shows whether the verify's distribution differs from the decode step's.
"""
import argparse
import json
import math
import os
import sys
import urllib.request
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampling_dist_check import PROMPTS  # noqa: E402


def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--first", nargs="*", help="first-token text per prompt to condition on (default: most common)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    result = {}
    for p, prompt in enumerate(PROMPTS):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      tokenize=True, enable_thinking=False)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = [int(x) for x in ids]
        rows = []
        for _ in range(a.n):
            r = post(a.base_url + "/generate", {
                "input_ids": ids, "return_logprob": True, "top_logprobs_num": 20,
                "sampling_params": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_new_tokens": 3}})
            m = r["meta_info"]
            top = m.get("output_top_logprobs") or []
            if len(r["output_ids"]) >= 2 and len(top) >= 2:
                rows.append((r["output_ids"][0], {int(t[1]): float(t[0]) for t in top[1] if t[0] is not None}))
        # condition on one first token; --first pins it (token text) so two servers are compared on the
        # same context even if their most common first token differs
        first = (tok.encode(a.first[p], add_special_tokens=False)[0] if a.first and a.first[p]
                 else Counter(f for f, _ in rows).most_common(1)[0][0])
        acc, cnt = defaultdict(float), 0
        for f, d in rows:
            if f != first:
                continue
            cnt += 1
            for t, lp in d.items():
                acc[t] += math.exp(lp)
        probs = sorted(((v / cnt, t) for t, v in acc.items()), reverse=True)[:12]
        result[p] = {"first": first, "first_text": tok.decode([first]), "n": cnt,
                     "top": [(tok.decode([t]), round(v, 4)) for v, t in probs]}
        # spread across requests of the top token's probability: one context, so a plain decode step
        # reports one value; a verify whose rows see the draft would report several
        t0 = probs[0][1]
        vals = sorted(math.exp(d[t0]) for f, d in rows if f == first and t0 in d)
        result[p]["top_token_spread"] = [round(vals[0], 4), round(vals[len(vals) // 2], 4), round(vals[-1], 4)] if vals else None
        print(p, repr(tok.decode([first])), cnt, result[p]["top"][:6], "spread(min,med,max)", result[p]["top_token_spread"], flush=True)
    json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
