"""Multi-user stress check for ax-engine-sglang (Stage 0 edge cases, docs/decisions/2026-11-15-multiuser-width-policy.md).

Against a running server:
  1. reference: N_REF prompts one at a time (greedy, REF_TOKENS tokens), saved;
  2. stress: N_STRESS requests from up to WORKERS client threads (the server caps the running
     batch), each with a random max_new_tokens in {1, 2, 7, 16, 64, 256, 512} (EOS and max-token
     cuts across width changes), a share ABORT of them aborted after a random delay through
     /abort_request (cancellation mid-round), prompts drawn with repetition (slot reuse);
  3. reference again, one at a time: every output must be token-identical to step 1 (a request alone
     is deterministic, so any difference is state leaking from the stress phase).
Pool accounting: the server log's token and recurrent-state ("mamba") usage on a one-request
prefill after each phase must return to the idle value seen before the stress phase. With
cancellation enabled, at least one response must report an abort.

    python benchmarks/sglang/stress_multiuser.py --base-url http://127.0.0.1:30000 --server-log R/server.log --out R/stress.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/baselines"))


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


FIELD = {k: re.compile(rf"{k}: ([0-9.]+)") for k in ("#running-req", "full token usage", "mamba usage")}


def last_usage(log, after):
    """(running, token usage, mamba usage) of the last prefill line written after byte offset `after`
    (SGLang prints the fields in its own order; each is read by name)."""
    data = Path(log).read_bytes()[after:].decode(errors="replace")
    lines = [l for l in data.splitlines() if "Prefill batch" in l]
    if not lines:
        return None
    vals = [FIELD[k].search(lines[-1]) for k in ("#running-req", "full token usage", "mamba usage")]
    return tuple(float(v.group(1)) for v in vals) if all(vals) else None


def generation_result(result, budget, *, allow_abort=False):
    finish = (result.get("meta_info") or {}).get("finish_reason")
    finish = finish.get("type") if isinstance(finish, dict) else finish
    output = result.get("output_ids")
    if (not isinstance(output, list) or len(output) > budget
            or any(type(token) is not int or token < 0 for token in output)):
        raise ValueError("malformed generation output_ids")
    if allow_abort and finish == "abort":
        return output, finish
    if not output or finish not in ("stop", "length") or (finish == "length" and len(output) != budget):
        raise ValueError(f"incomplete generation: {finish!r}, {len(output)}/{budget} tokens")
    return output, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--n-ref", type=int, default=24)
    ap.add_argument("--ref-tokens", type=int, default=256)
    ap.add_argument("--n-stress", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--abort", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input-ids", type=Path, help="JSON array of token-id prompts for offline controls")
    ap.add_argument("--repeat-reference", action="store_true",
                    help="repeat the reference before stress to detect pre-existing instability")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if min(args.n_ref, args.ref_tokens, args.n_stress, args.workers) < 1 or not 0 <= args.abort <= 1:
        ap.error("request, token and worker counts must be positive; --abort must be in [0, 1]")
    args.out.unlink(missing_ok=True)

    if args.input_ids:
        enc = json.loads(args.input_ids.read_text())
        if (not isinstance(enc, list) or not enc or any(
                not isinstance(row, list) or not row or any(type(token) is not int or token < 0 for token in row)
                for row in enc)):
            raise SystemExit("--input-ids must contain nonempty arrays of non-negative integer token IDs")
    else:
        from transformers import AutoTokenizer
        from prompts import load_prompt_sets
        tok = AutoTokenizer.from_pretrained(args.model)
        sets = load_prompt_sets(max(args.n_ref, 64))
        prompts = [p for name in sets for p in sets[name]]

        def encode(p):
            text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
            return tok(text, add_special_tokens=False).input_ids

        enc = [encode(p) for p in prompts]
    ref_idx = list(range(0, len(enc), max(1, len(enc) // args.n_ref)))[: args.n_ref]

    def generate(ids, max_tokens, rid=None):
        body = {"input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens}}
        if rid:
            body["rid"] = rid
        return post(f"{args.base_url}/generate", body)

    def reference():
        return {i: generation_result(generate(enc[i], args.ref_tokens), args.ref_tokens)[0] for i in ref_idx}

    def usage_probe():
        off = Path(args.server_log).stat().st_size
        generation_result(generate(enc[0], 1), 1)
        time.sleep(2)
        return last_usage(args.server_log, off)

    t0 = time.monotonic()
    usage0 = usage_probe()
    ref1 = reference()
    ref_repeat = reference() if args.repeat_reference else None
    lengths = [1, 2, 7, 16, 64, 256, 512]
    stats = {"ok": 0, "aborted_sent": 0, "aborted_seen": 0, "errors": 0, "finish": {}}
    lock = threading.Lock()

    def one(k):
        r = random.Random(args.seed * 100003 + k)
        ids = enc[r.randrange(len(enc))]
        rid = uuid.uuid4().hex
        selected = r.random() < args.abort
        budget = r.choice(lengths)
        done, sent = threading.Event(), threading.Event()
        cancellation = None
        if selected:
            delay = r.uniform(0.05, 2.0)

            def cancel():
                if done.wait(delay):
                    return
                sent.set()
                with lock:
                    stats["aborted_sent"] += 1
                try:
                    post(f"{args.base_url}/abort_request", {"rid": rid}, timeout=30)
                except Exception as error:
                    with lock:
                        stats["errors"] += 1
                        stats.setdefault("error_samples", []).append(f"abort request: {error!r}"[:200])

            cancellation = threading.Thread(target=cancel)
            cancellation.start()
        try:
            out = generate(ids, budget, rid=rid)
            _, fin = generation_result(out, budget, allow_abort=sent.is_set())
            with lock:
                stats["finish"][fin] = stats["finish"].get(fin, 0) + 1
                stats["ok"] += 1
                if fin == "abort":
                    stats["aborted_seen"] += 1
        except urllib.error.HTTPError as e:
            with lock:
                stats["errors"] += 1
                stats.setdefault("error_samples", []).append(f"HTTP {e.code}: {e.read()[:200]!r}")
        except Exception as e:   # noqa: BLE001 - every failure is a result here
            with lock:
                stats["errors"] += 1
                stats.setdefault("error_samples", []).append(repr(e)[:200])
        finally:
            done.set()
            if cancellation is not None:
                cancellation.join()

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, range(args.n_stress)))
    time.sleep(5)
    usage1 = usage_probe()
    ref2 = reference()
    differ = [i for i in ref_idx if ref1[i] != ref2[i]]
    repeat_differ = [i for i in ref_idx if ref_repeat is not None and ref1[i] != ref_repeat[i]]
    usage_returned = usage0 is not None and usage0 == usage1 and usage0[0] == 0
    cancellation_exercised = stats["aborted_seen"] > 0
    result = {"stress": stats, "usage_before": usage0, "usage_after": usage1,
              "usage_returned": usage_returned, "cancellation_exercised": cancellation_exercised,
              "reference_prompts": len(ref_idx), "reference_differ": differ,
              "reference_before": ref1, "reference_after": ref2,
              "reference_repeat": ref_repeat, "reference_repeat_differ": repeat_differ,
              "passed": not differ and not repeat_differ and stats["errors"] == 0
                        and usage_returned and (args.abort == 0 or cancellation_exercised),
              "wall_s": time.monotonic() - t0}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
