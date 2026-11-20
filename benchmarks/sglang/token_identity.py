"""Token-identity check for lossless speculative decoding (Track L, gate GL1 precursor).

Collect: send the benchmark prompts (benchmarks/baselines/prompts.py, same sets and order as bench_speed.py) to a
running SGLang server one at a time, greedy, thinking off, and save the exact output token ids.

    python benchmarks/sglang/token_identity.py collect --out runs/tree32c_ids.jsonl --n 64

Each row also records the request's wall time and output length, so the same file feeds the paired
speed statistics in benchmarks/sglang/paired_ci.py (requests run one at a time, after WARMUP unrecorded
requests on prompts outside the measured list).

Compare: per prompt set, the share of answers that are token-identical to a reference run, and for
the rest the position of the first differing token.

    python benchmarks/sglang/token_identity.py compare ar_ids.jsonl tree32c_ids.jsonl [chain24_ids.jsonl ...]

Greedy speculative decoding in bf16 is lossless only up to floating-point differences between the
verify pass (many tokens at once) and plain decoding (one token), so a chain lane is the yardstick:
a tree lane should not diverge from AR more often or earlier than the chain does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ax-engine-sglang"))
from ax_engine_sglang.contracts import benchmark_record
from common import DIAGNOSTIC_ENV_KEYS, launch_diagnostic


def post(url, body, timeout=600, retries=3):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"HTTP {e.code} from {url}: {e.read()[:500]!r}")   # a rejected request will not heal
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries - 1:
                sys.exit(f"{url} failed {retries} times: {e}")
            print(f"retrying {url} after {e}", file=sys.stderr, flush=True)
            time.sleep(5)


SERVER_KEYS = ("model_path", "port", "version", "speculative_algorithm", "speculative_num_draft_tokens",
               "speculative_draft_model_path", "attention_backend", "disable_overlap_schedule",
               "disable_radix_cache", "max_running_requests", "random_seed", "revision", "quantization", "dtype")
ENV_KEYS = (*DIAGNOSTIC_ENV_KEYS, "AX_SITE_SHA", "AX_FAST", "AX_LATTICE", "AX_GDN_LAYERS", "AX_GDN_HEADS", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
            "FC1_TREE64", "FC1_TREE_N", "FC1_DRAFT_BLOCK", "FC1_COMPILE_BUILD", "FC1_FUSED", "FC1_BRANCH_PILOT",
            "FC1_OURS", "FC1_NGRAM_PATH", "MODES", "BLOCK", "DRAFT", "EXTRA_ARGS")


FINISH_OK = ("stop", "length")   # SGLang finish types of a complete answer (EOS / stop token, or max tokens)


def server_info(base_url):
    try:
        with urllib.request.urlopen(f"{base_url}/get_server_info", timeout=10) as r:
            d = json.loads(r.read())
    except Exception as e:                                             # noqa: BLE001
        return {"error": str(e)}
    return {k: d.get(k) for k in SERVER_KEYS}


def prompt_hash(ids):
    return hashlib.sha1(json.dumps(list(ids)).encode()).hexdigest()[:16]


def collect(args):
    out = Path(args.out)
    for stale in (out, out.with_suffix(".meta.json")):     # a failed rerun must not leave a valid old pass
        stale.unlink(missing_ok=True)
    from transformers import AutoTokenizer
    from prompts import load_prompt_sets
    info = server_info(args.base_url)
    if "error" in info or not info.get("model_path"):
        sys.exit(f"cannot identify the server on {args.base_url}: {info}")
    if args.expect_algorithm is not None and info.get("speculative_algorithm") != (args.expect_algorithm or None):
        sys.exit(f"server on {args.base_url} runs {info.get('speculative_algorithm')!r}, expected "
                 f"{args.expect_algorithm!r}: another server may own the port")
    sets = load_prompt_sets(args.n + args.warmup)   # warmup prompts come after the measured ones
    selected = {name: prompts for name, prompts in sets.items() if not args.sets or name in args.sets}
    for name, prompts in selected.items():
        if len(prompts) < args.n:
            sys.exit(f"{name} has {len(prompts)} prompts, requested {args.n}")
    tok = AutoTokenizer.from_pretrained(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")

    # --context-tokens N: put about N tokens of reference text (the other benchmark prompts, in a fixed
    # order) ahead of each prompt, for checks that need a long context (e.g. sparse attention that
    # selects among blocks only once the context exceeds its budget)
    filler = ""
    if args.context_tokens:
        pool = [q for prompts in sets.values() for q in prompts]
        parts, n = [], 0
        while n < args.context_tokens:
            q = pool[len(parts) % len(pool)]
            parts.append(q)
            n += len(tok(q, add_special_tokens=False).input_ids)
        filler = "Reference material (do not answer it):\n\n" + "\n\n---\n\n".join(parts) + "\n\n=== Task ===\n\n"

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": filler + p}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        return tok(text, add_special_tokens=False).input_ids

    params = {"temperature": 0.0, "max_new_tokens": args.max_tokens}
    first = next(iter(selected.values()))
    for k in range(args.warmup):   # compile / graph capture / allocator warmup, not recorded
        post(f"{args.base_url}/generate", {"input_ids": encode(first[min(args.n + k, len(first) - 1)]), "sampling_params": params})
    jobs = [(name, i, encode(p)) for name, prompts in selected.items()
            for i, p in enumerate(prompts[:args.n])]

    def one(job):
        name, i, ids = job
        t0 = time.perf_counter()
        # a measured request is never retried: a retry would hide a timeout inside its wall time
        r = post(f"{args.base_url}/generate", {"input_ids": list(ids), "sampling_params": params}, retries=1)
        wall = time.perf_counter() - t0
        out_ids = r.get("output_ids")
        if out_ids is None:
            sys.exit("server response has no output_ids; cannot check token identity")
        meta = r.get("meta_info") or {}
        fin = meta.get("finish_reason")
        fin_type = fin.get("type") if isinstance(fin, dict) else fin
        if fin_type not in FINISH_OK:
            sys.exit(f"{name}[{i}]: finish reason {fin!r}; an aborted or partial answer is not a result")
        return {"set": name, "i": i, "prompt_sha1": prompt_hash(ids), "output_ids": out_ids,
                "n_out": len(out_ids), "wall_s": wall, "prompt_tokens": len(ids), "finish": fin_type}

    workload_start = time.perf_counter()
    if args.concurrency == 1:
        results = [one(j) for j in jobs]
    else:
        # several requests in flight (multi-user identity checks); rows keep the prompt order, and
        # wall_s is then the request's latency under load, not a one-request speed
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(args.concurrency) as pool:
            results = list(pool.map(one, jobs))
    workload_wall = time.perf_counter() - workload_start
    with open(tmp, "w") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")
    rows_written = len(results)
    env = {k: os.environ.get(k) for k in ENV_KEYS}
    # diagnostic switches act in the server, so take them from the server's launch record
    # (benchmarks/sglang/run_baselines.sh writes it before starting the server); unknown without one
    launch = json.loads(Path(args.server_env).read_text()) if args.server_env else None
    diagnostic = launch_diagnostic(launch)
    meta = {"complete": True, "rows": rows_written, "diagnostic": diagnostic, "server_launch": launch,
            "ids_sha1": hashlib.sha1(tmp.read_bytes()).hexdigest(),
            "server": info, "env": env,
            "tokenizer": args.model, "n": args.n, "max_tokens": args.max_tokens, "warmup": args.warmup,
            "sets": args.sets, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "concurrency": args.concurrency,
            "context_tokens": args.context_tokens,
            "benchmark": benchmark_record(
                rows=results, wall_seconds=workload_wall, server=info, model=args.model,
                concurrency=args.concurrency, max_tokens=args.max_tokens, diagnostic=diagnostic,
            )}
    # ids first, then the sidecar: the sidecar is the completion marker paired_ci.py requires
    os.replace(tmp, out)
    mtmp = out.with_suffix(".meta.tmp")
    mtmp.write_text(json.dumps(meta, indent=2))
    os.replace(mtmp, out.with_suffix(".meta.json"))
    print(f"wrote {out}")


FINISH_OK_COMPARE = ("stop", "length")


def load(path, legacy=False):
    side = Path(path).with_suffix(".meta.json")
    if side.exists():
        m = json.loads(side.read_text())
        if m.get("ids_sha1") and hashlib.sha1(Path(path).read_bytes()).hexdigest() != m["ids_sha1"]:
            sys.exit(f"{path} does not match the hash in its sidecar")
    elif not legacy:
        sys.exit(f"{path} has no completion sidecar (incomplete pass?); --legacy for old files")
    rows, hashes = {}, {}
    for line in open(path):
        r = json.loads(line)
        k = (r["set"], r["i"])
        if k in rows:
            sys.exit(f"{path}: prompt {k} appears twice")
        fin = r.get("finish")
        fin = fin.get("type") if isinstance(fin, dict) else fin
        if fin is not None and fin not in FINISH_OK_COMPARE:
            sys.exit(f"{path}: prompt {k} finished with {fin!r} (aborted answer)")
        rows[k] = r["output_ids"]
        hashes[k] = r.get("prompt_sha1")
    return rows, hashes


def first_diff(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(args):
    ref, ref_h = load(args.files[0], args.legacy)
    print(f"reference: {args.files[0]}")
    for path in args.files[1:]:
        other, oth_h = load(path, args.legacy)
        print(f"== {path}")
        if set(other) != set(ref):
            # a shorter file must not report "20/20 identical" over an easy subset
            sys.exit(f"{path} covers {len(other)} prompts, the reference {len(ref)}: not comparable")
        bad_hash = [k for k in ref if ref_h[k] and oth_h[k] and ref_h[k] != oth_h[k]]
        if bad_hash:
            sys.exit(f"{path}: {len(bad_hash)} prompts differ from the reference prompt text, e.g. {bad_hash[0]}")
        for name in sorted({k[0] for k in ref}):
            keys = [k for k in ref if k[0] == name]
            diffs = [first_diff(ref[k], other[k]) for k in keys]
            bad = [d for d in diffs if d is not None]
            line = f"  {name:5s} identical {len(keys) - len(bad)}/{len(keys)}"
            if bad:
                line += (f"  first difference at token: median {statistics.median(bad):.0f}, "
                         f"min {min(bad)}, max {max(bad)}")
            print(line)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:30000"))
    c.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3.5-9B"))
    c.add_argument("--n", type=int, default=64)
    c.add_argument("--max-tokens", type=int, default=1024)
    c.add_argument("--sets", nargs="*", choices=["code", "math", "chat"])
    c.add_argument("--warmup", type=int, default=4)
    c.add_argument("--context-tokens", type=int, default=0,
                   help="prepend about this many tokens of reference text to every prompt (long-context checks)")
    c.add_argument("--concurrency", type=int, default=1,
                   help="requests in flight (1 = one at a time, the speed setting; >1 for multi-user identity checks)")
    c.add_argument("--expect-algorithm", help="speculative algorithm the server must report ('' = none, AR)")
    c.add_argument("--server-env", help="the server's launch record (env_gpu<N>.json of run_baselines.sh)")
    m = sub.add_parser("compare")
    m.add_argument("files", nargs="+")
    m.add_argument("--legacy", action="store_true", help="accept files without a completion sidecar")
    args = ap.parse_args()
    if (args.cmd == "collect"
            and (min(args.n, args.max_tokens, args.concurrency) < 1 or min(args.warmup, args.context_tokens) < 0)):
        c.error("request, token and concurrency counts must be positive; warmup and context counts must be nonnegative")
    collect(args) if args.cmd == "collect" else compare(args)


if __name__ == "__main__":
    main()
