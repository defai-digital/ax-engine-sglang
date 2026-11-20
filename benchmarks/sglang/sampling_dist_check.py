"""Is sampled speculative decoding lossless in distribution? Two-sample test against a reference server.

  python benchmarks/sglang/sampling_dist_check.py collect --base-url URL --model MODEL --out a.jsonl [--n 500]
  python benchmarks/sglang/sampling_dist_check.py compare a.jsonl b.jsonl [--alpha 1e-3]

Greedy runs can be checked token by token; sampled runs cannot. A lossless sampled verify leaves
the distribution of every generated token unchanged, so samples from our server and from a stock
server on the same prompts must come from the same distribution at each position. collect sends
each prompt N times, one request in flight (so a tree server takes tree rounds), with the model
card's sampling (temperature 1.0, top-p 0.95, top-k 20), and keeps
the first --tokens output ids. compare runs a chi-square test of homogeneity per (prompt, position)
on the token counts (tokens with fewer than 5 expected samples pooled into one cell) and on the
first two tokens jointly (a pooled cell still below 5 expected is merged into the smallest cell),
and reports every test with p < alpha and the number of tests below 0.01 next to the number expected
by chance. Ended responses count as a separate outcome at later positions. With the default
alpha 1e-3 and 28 tests the family-wise false-alarm rate is about 3%;
later positions are marginal (mixtures over earlier tokens), so they detect a difference but do not
localise it. Position 0 comes from the prefill, so it checks the test
itself; later positions come from the speculative verify.

SGLang only honors --seed0 request seeds with deterministic inference enabled.
Otherwise use --seed0 -1 and start reference/candidate servers with different
--random-seed values for independent distribution cohorts. Collection records
the effective seed mode and server settings in a metadata sidecar.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import urllib.error
import urllib.request
from collections import Counter

PROMPTS = [
    "Write the opening sentence of a short story about a lighthouse keeper.",
    "List a few words that describe autumn.",
    "Tell me something interesting about octopuses.",
    "Suggest a name for a small coffee shop and say why.",
]


def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def collect(a):
    path = Path(a.out)
    sidecar = metadata_path(path)
    sidecar.unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    with urllib.request.urlopen(a.base_url + "/server_info", timeout=30) as response:
        info = json.loads(response.read())
    server = info.get("server_args", info)
    deterministic = server.get("enable_deterministic_inference")
    if a.seed0 >= 0 and deterministic is not True:
        sys.exit("--seed0 requires SGLang --enable-deterministic-inference; request sampling_seed "
                 "is otherwise ignored. Use --seed0 -1 with distinct server --random-seed values.")
    if a.seed0 < 0 and deterministic is not False:
        sys.exit("server RNG collection requires deterministic inference disabled; "
                 "otherwise choose explicit --seed0 request seeds")
    metadata = {"prompts": len(PROMPTS), "samples_per_prompt": a.n, "tokens": a.tokens,
                "requested_seed_base": a.seed0,
                "seed_mode": "request" if a.seed0 >= 0 else "server_rng",
                "server_random_seed": server.get("random_seed"),
                "enable_deterministic_inference": deterministic,
                "sampling_backend": server.get("sampling_backend"),
                "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = []
    for prompt in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      tokenize=True, enable_thinking=False)
        if hasattr(ids, "keys"):                 # transformers 5 returns a BatchEncoding
            ids = ids["input_ids"]
        prompts.append([int(token) for token in ids])
    metadata["prompt_sha256"] = [hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                                 for ids in prompts]
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging:
        samples = Path(staging) / "samples"
        with samples.open("w") as output:
            for row in sample_rows(a, prompts):
                output.write(json.dumps(row) + "\n")
        metadata["samples_sha256"] = hashlib.sha256(samples.read_bytes()).hexdigest()
        pending = Path(staging) / "metadata"
        pending.write_text(json.dumps(metadata, indent=2) + "\n")
        samples.replace(path)
        pending.replace(sidecar)
    print(f"wrote {a.out}: {len(PROMPTS)} prompts x {a.n}")


def sample_rows(a, prompts):
    for p, ids in enumerate(prompts):
        for i in range(a.n):
            params = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_new_tokens": a.tokens}
            if a.seed0 >= 0:
                params["sampling_seed"] = a.seed0 + p * max(100000, a.n) + i
            try:
                r = post(a.base_url + "/generate", {"input_ids": ids, "sampling_params": params})
            except urllib.error.HTTPError as e:
                sys.exit(f"HTTP {e.code}: {e.read()[:400]!r}")
            out = r.get("output_ids")
            finish = (r.get("meta_info") or {}).get("finish_reason")
            finish = finish.get("type") if isinstance(finish, dict) else finish
            if (not isinstance(out, list) or not out or len(out) > a.tokens
                    or any(type(token) is not int or token < 0 for token in out)
                    or finish not in ("stop", "length") or (finish == "length" and len(out) != a.tokens)):
                sys.exit(f"incomplete sample {p}/{i}: finish {finish!r}")
            yield {"p": p, "i": i, "ids": out}


def chi2_homogeneity(ca, cb):
    """Return a p-value, or None if pooling erased observed differences."""
    from scipy.stats import chi2_contingency
    na, nb = sum(ca.values()), sum(cb.values())
    keys = sorted(set(ca) | set(cb))
    big, rest = [], [0, 0]
    for k in keys:
        x, y = ca.get(k, 0), cb.get(k, 0)
        if (x + y) * min(na, nb) / (na + nb) >= 5:
            big.append([x, y])
        else:
            rest[0] += x
            rest[1] += y
    if sum(rest):
        if (rest[0] + rest[1]) * min(na, nb) / (na + nb) >= 5 or not big:
            big.append(rest)
        else:                                   # pooled cell still below 5 expected: merge into the smallest
            j = min(range(len(big)), key=lambda i: big[i][0] + big[i][1])
            big[j] = [big[j][0] + rest[0], big[j][1] + rest[1]]
    if len(big) < 2:
        same = all(ca.get(k, 0) * nb == cb.get(k, 0) * na for k in keys)
        return (1.0 if same else None), 0
    _, p, dof, _ = chi2_contingency(list(zip(*big)))
    return p, dof


def load_samples(path, digest=None):
    rows, seen = {}, set()
    sha = hashlib.sha256()
    with open(path, "rb") as source:
        for line in source:
            sha.update(line)
            item = json.loads(line)
            prompt, sample, ids = item["p"], item["i"], item["ids"]
            if (type(prompt) is not int or prompt < 0 or type(sample) is not int or sample < 0
                    or not isinstance(ids, list) or not ids
                    or any(type(token) is not int or token < 0 for token in ids)):
                raise ValueError("samples need non-negative prompt/sample IDs and nonempty token arrays")
            key = (prompt, sample)
            if key in seen:
                raise ValueError(f"duplicate prompt/sample ID: {key}")
            seen.add(key)
            rows.setdefault(prompt, []).append(ids)
    if not rows:
        raise ValueError("sample file is empty")
    if digest is not None and sha.hexdigest() != digest:
        raise ValueError(f"{path}: sample file hash differs from the collection metadata")
    return rows


def metadata_path(path):
    return Path(str(path) + ".metadata.json")


def load_metadata(path, required):
    try:
        return json.loads(metadata_path(path).read_text())
    except FileNotFoundError:
        if required:
            raise
        return {}


def check_metadata_counts(rows, meta):
    if type(meta["prompts"]) is not int or meta["prompts"] <= 0:
        raise ValueError("metadata prompts must be a positive integer")
    if type(meta["samples_per_prompt"]) is not int or meta["samples_per_prompt"] <= 0:
        raise ValueError("metadata samples_per_prompt must be a positive integer")
    if (set(rows) != set(range(meta["prompts"]))
            or any(len(samples) != meta["samples_per_prompt"] for samples in rows.values())):
        raise ValueError("sample counts do not match the collection metadata")
    tokens = meta["tokens"]
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("metadata tokens must be a positive integer")
    if any(len(sample) > tokens for samples in rows.values() for sample in samples):
        raise ValueError("sample length exceeds the recorded token budget")


def compare(a):
    token_budget = None
    try:
        ma, mb = (load_metadata(path, a.require_independent_seeds) for path in (a.a, a.b))
        A, B = load_samples(a.a, ma.get("samples_sha256")), load_samples(a.b, mb.get("samples_sha256"))
        if set(A) != set(B):
            raise ValueError("sample files contain different prompt sets")
        if any(len(A[p]) != len(B[p]) for p in A):
            raise ValueError("sample files contain different per-prompt sample counts")
        if ma and mb and ma.get("prompt_sha256") != mb.get("prompt_sha256"):
            raise ValueError("cohorts recorded different tokenized prompts")
        if a.require_independent_seeds:
            if ma["seed_mode"] != "server_rng" or mb["seed_mode"] != "server_rng":
                raise ValueError("independent-seed gate currently requires server RNG cohorts")
            sa, sb = ma["server_random_seed"], mb["server_random_seed"]
            if type(sa) is not int or type(sb) is not int or sa == sb:
                raise ValueError("independent cohorts require different recorded server RNG seeds")
            if any(meta.get("enable_deterministic_inference") is not False for meta in (ma, mb)):
                raise ValueError("independent server RNG cohorts require deterministic inference disabled")
            check_metadata_counts(A, ma)
            check_metadata_counts(B, mb)
            for key in ("tokens", "temperature", "top_p", "top_k"):
                if ma.get(key) != mb.get(key):
                    raise ValueError(f"cohorts recorded different {key}")
            token_budget = ma["tokens"]
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"DIST_FAIL invalid samples: {error}")
        sys.exit(2)
    if any(len(samples) < 10 for samples in A.values()):
        print("DIST_INCONCLUSIVE: need at least 10 samples per prompt in each cohort")
        sys.exit(2)
    tests, low, bad, inconclusive = 0, 0, [], []
    for p in sorted(set(A) & set(B)):
        L = token_budget or max(max(map(len, A[p])), max(map(len, B[p])))
        counts = []
        for pos in range(L):
            # IDs are nonnegative, so -1 marks an already ended response.
            counts.append((pos, Counter(x[pos] if len(x) > pos else -1 for x in A[p]),
                           Counter(x[pos] if len(x) > pos else -1 for x in B[p])))
        if L >= 2:
            counts.append(("0-1", Counter(tuple(x[:2]) for x in A[p]), Counter(tuple(x[:2]) for x in B[p])))
        for pos, ca, cb in counts:
            pv, dof = chi2_homogeneity(ca, cb)
            if pv is None:
                inconclusive.append((p, pos))
                continue
            tests += 1
            low += pv < 0.01
            if pv < a.alpha:
                bad.append((p, pos, pv, dof))
    print(f"DIST tests={tests} p<0.01: {low} (chance {tests * 0.01:.1f}) p<{a.alpha}: {len(bad)}")
    for b in bad:
        print("   prompt %s position %s p=%.2e dof=%d" % b)
    for p, pos in inconclusive:
        print(f"   prompt {p} position {pos}: low expected counts leave no comparison after pooling")
    if bad:
        print("DIST_FAIL")
        sys.exit(1)
    if inconclusive:
        print("DIST_INCONCLUSIVE")
        sys.exit(2)
    print("DIST_PASS")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--base-url", default="http://127.0.0.1:30000")
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, default=500)
    c.add_argument("--tokens", type=int, default=6)
    c.add_argument("--seed0", type=int, default=-1,
                   help="per-request seed base (requires deterministic inference); -1 = server RNG")
    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--alpha", type=float, default=1e-3)
    d.add_argument("--require-independent-seeds", action="store_true",
                   help="require metadata sidecars with different server RNG seeds")
    a = ap.parse_args()
    if a.cmd == "collect" and min(a.n, a.tokens) <= 0:
        c.error("sample count and token budget must be positive")
    if a.cmd == "compare" and not 0 < a.alpha < 1:
        d.error("alpha must be between 0 and 1")
    collect(a) if a.cmd == "collect" else compare(a)


if __name__ == "__main__":
    main()
