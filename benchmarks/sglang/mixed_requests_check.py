"""Mixed request kinds against a running server: greedy, sampled, JSON-schema constrained, logprobs.

  python benchmarks/sglang/mixed_requests_check.py --base-url http://127.0.0.1:30000 --out mixed.json

The tree serves one plain greedy request; every other kind must run SGLang's chain verify without
stopping the server (ax_chain.tree_round). Each kind is sent alone (one request in flight, so the
round decision sees exactly that request) and then all kinds at once, several rounds. Checks: every
request returns 200; JSON-schema outputs parse and match the schema's required keys; logprob
requests return one logprob per output token; the server still answers /health at the end. Whether
the greedy output sent with the others equals the greedy output alone is recorded, not gated
(batch composition changes greedy near-ties).
"""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import sys
import urllib.request

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},
          "required": ["name", "year"]}
PROMPT = "Name one programming language and the year it first appeared. Answer briefly."


def post(base, body, timeout=300):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def kinds(model, max_tokens):
    base = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    return {
        "greedy": dict(base, temperature=0.0),
        "sampled": dict(base, temperature=1.0, top_p=0.95, top_k=20, seed=7),
        "default_params": dict(base),      # server applies the model's generation config (temperature 1.0)
        "json_schema": dict(base, temperature=0.0,
                            response_format={"type": "json_schema", "json_schema": {"name": "lang", "schema": SCHEMA}}),
        "logprobs": dict(base, temperature=0.0, logprobs=True, top_logprobs=2),
    }


def check(kind, status, out):
    if status != 200:
        return f"{kind}: HTTP {status}"
    choices = out.get("choices") if isinstance(out, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return f"{kind}: missing or invalid choice"
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish not in ("stop", "length"):
        return f"{kind}: incomplete generation: {finish!r}"
    message = choice.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        return f"{kind}: empty or invalid message content"
    if kind == "json_schema":
        try:
            obj = json.loads(text)
        except ValueError:
            return f"json_schema: not JSON: {text[:80]!r}"
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or type(obj.get("year")) is not int:
            return f"json_schema: expected an object with string name and integer year: {obj!r}"
    if kind == "logprobs":
        logprobs, usage = choice.get("logprobs"), out.get("usage")
        lp = logprobs.get("content") if isinstance(logprobs, dict) else None
        n = usage.get("completion_tokens") if isinstance(usage, dict) else None
        count = len(lp) if isinstance(lp, list) else 0
        if not count or type(n) is not int or count != n:
            return f"logprobs: {count} entries for {n} tokens"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if min(a.max_tokens, a.rounds) < 1:
        ap.error("token budget and concurrent rounds must be positive")
    a.out.unlink(missing_ok=True)
    ks = kinds(a.model, a.max_tokens)
    errors, result = [], {"alone": {}, "together": []}
    for kind, body in ks.items():
        try:
            status, out = post(a.base_url, body)
        except OSError as exc:
            errors.append(f"alone {kind}: {exc!r}")
            continue
        e = check(kind, status, out)
        errors += [f"alone {e}"] if e else []
        result["alone"][kind] = out["choices"][0]["message"]["content"] if e is None else None
    for r in range(a.rounds):
        with cf.ThreadPoolExecutor(len(ks) * 2) as ex:
            futs = {ex.submit(post, a.base_url, body): kind for kind in ks for body in (ks[kind], ks[kind])}
            got = {}
            for f in cf.as_completed(futs):
                kind = futs[f]
                try:
                    status, out = f.result()
                except Exception as exc:          # connection reset = server stopped
                    errors.append(f"together r{r} {kind}: {exc!r}")
                    continue
                e = check(kind, status, out)
                errors += [f"together r{r} {e}"] if e else []
                if e is None:
                    got.setdefault(kind, []).append(out["choices"][0]["message"]["content"])
        result["together"].append(got)
        # recorded, not gated: batch composition changes greedy near-ties (NVFP4 especially)
        result.setdefault("greedy_same_as_alone", []).extend(g == result["alone"].get("greedy") for g in got.get("greedy", []))
    try:
        with urllib.request.urlopen(a.base_url + "/health", timeout=10) as r:
            alive = r.status == 200
    except Exception:
        alive = False
    if not alive:
        errors.append("server does not answer /health")
    result.update(errors=errors, passed=not errors)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"MIXED passed={not errors} errors={len(errors)}")
    for e in errors[:10]:
        print("  ", e)
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
