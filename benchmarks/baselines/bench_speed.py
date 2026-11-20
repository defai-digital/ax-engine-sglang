"""Decode-speed benchmark against a running vLLM server.

Measures per-request time to first token (TTFT) and decode tokens/s, aggregate output tokens/s,
and (for MTP modes) speculative acceptance from vLLM's /metrics, per prompt set x concurrency.

  python benchmarks/baselines/bench_speed.py --mode ar --out results/run1
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

from common import (DEFAULT_BASE_URL, launch_diagnostic, request_kwargs, scrape_spec_metrics, spec_summary,
                    write_json)
from prompts import load_prompt_sets


async def one_request(client, prompt, kw, seed=None):
    kw = dict(kw)
    if seed is not None:
        kw["seed"] = seed
    t0 = time.perf_counter()
    ttft = None
    t_last = None
    n_tokens = None
    finish = None
    async for chunk in await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        **kw,
    ):
        if chunk.usage is not None:
            n_tokens = chunk.usage.completion_tokens
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if (delta.content or getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)):
                t_last = time.perf_counter() - t0
                if ttft is None:
                    ttft = t_last
    total = time.perf_counter() - t0
    if n_tokens is None:
        raise RuntimeError("stream ended without a usage chunk")
    if finish not in {"stop", "length"}:
        raise RuntimeError(f"generation did not complete: {finish!r}")
    # A single content chunk gives no observed decode interval. Exclude the
    # trailing protocol/usage chunks, and never divide by a fabricated duration.
    decode_s = t_last - ttft if ttft is not None and t_last > ttft and n_tokens > 1 else None
    return {
        "ttft_s": ttft,
        "total_s": total,
        "completion_tokens": n_tokens,
        "finish_reason": finish,
        "decode_s": decode_s,
        "decode_tok_s": (n_tokens - 1) / decode_s if decode_s is not None else None,
    }


async def run_level(client, prompts, concurrency, n_requests, kw):
    """Run the same request list at every concurrency level: request i uses prompt i and seed i."""
    if not 0 < n_requests <= len(prompts) or concurrency < 1:
        raise ValueError("a benchmark level needs positive concurrency and enough distinct prompts")
    sem = asyncio.Semaphore(concurrency)

    async def guarded(i, p):
        async with sem:
            start = time.perf_counter() - t0
            r = await one_request(client, p, kw, seed=i)
            r["start_s"] = start
            return r

    t0 = time.perf_counter()
    rows = await asyncio.gather(*(guarded(i, p) for i, p in enumerate(prompts[:n_requests])))
    wall = time.perf_counter() - t0
    toks = sum(r["completion_tokens"] for r in rows)

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None

    dec = [r["decode_tok_s"] for r in rows if r["decode_tok_s"] is not None]
    ttft = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "wall_s": wall,
        "output_tokens": toks,
        "output_tok_s": toks / wall,
        "per_req_decode_tok_s_p50": statistics.median(dec) if dec else None,
        "per_req_decode_tok_s_p10": pct(dec, 0.10),
        "ttft_s_p50": statistics.median(ttft) if ttft else None,
        "ttft_s_p90": pct(ttft, 0.90),
        "mean_completion_tokens": toks / len(rows),
        "truncated_frac": sum(r["finish_reason"] == "length" for r in rows) / len(rows),
        # per request, in submission order: start (s after the level began), ttft, total, tokens
        "requests": [[round(r["start_s"], 4), round(r["ttft_s"], 4) if r["ttft_s"] is not None else None,
                      round(r["total_s"], 4), r["completion_tokens"]]
                     for r in rows],
        "request_decode_seconds": [r["decode_s"] for r in rows],
    }


async def main_async(args):
    launch = json.loads(args.server_env.read_text()) if args.server_env else None
    diagnostic = launch_diagnostic(launch)
    sets = (json.loads(args.prompt_file.read_text()) if args.prompt_file
            else load_prompt_sets(args.n_requests + 4))
    if not isinstance(sets, dict) or not sets:
        raise SystemExit("prompt file must contain named prompt lists")
    if args.sets:
        missing = set(args.sets) - sets.keys()
        if missing:
            raise SystemExit(f"missing prompt sets: {', '.join(sorted(missing))}")
        sets = {k: v for k, v in sets.items() if k in args.sets}
    for name, prompts in sets.items():
        if not isinstance(prompts, list) or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise SystemExit(f"{name} must contain nonempty prompt strings")
        if len(prompts) < args.n_requests:
            raise ValueError(f"{name} has {len(prompts)} prompts, requested {args.n_requests}")
        if len(set(prompts[:args.n_requests])) != args.n_requests:
            raise SystemExit(f"{name} contains duplicate measured prompts")
    client = AsyncOpenAI(base_url=f"{args.base_url}/v1", api_key="EMPTY", timeout=3600)
    kw = request_kwargs(args.preset, args.thinking, args.max_tokens, seed=None)

    # warmup (compile/cuda graphs); uses prompts outside the measured request list where possible
    first = next(iter(sets.values()))
    warmup = first[args.n_requests:args.n_requests + 4] or first[:4]
    await asyncio.gather(*(one_request(client, p, kw, seed=10_000 + i) for i, p in enumerate(warmup)))

    results = []
    for name, prompts in sets.items():
        measured = json.dumps(prompts[:args.n_requests], ensure_ascii=False, separators=(",", ":"))
        prompt_sha256 = hashlib.sha256(measured.encode()).hexdigest()
        for c in args.concurrency:
            before = scrape_spec_metrics(args.base_url)
            row = await run_level(client, prompts, c, args.n_requests, kw)
            row.update(prompt_set=name, prompt_sha256=prompt_sha256,
                       spec=spec_summary(before, scrape_spec_metrics(args.base_url)))
            results.append(row)
            acc = row["spec"].get("mean_acceptance_length")
            dec, ttft = row["per_req_decode_tok_s_p50"], row["ttft_s_p50"]
            print(f"[{args.mode}] {name:5s} c={c:<3d} out={row['output_tok_s']:8.1f} tok/s  "
                  f"p50/req={f'{dec:.1f}' if dec is not None else '-'} tok/s  "
                  f"ttft50={f'{ttft:.3f}s' if ttft is not None else '-'} len={row['mean_completion_tokens']:.0f}"
                  f"  trunc={row['truncated_frac']:.2f}"
                  + (f"  accept_len={acc:.2f}" if acc else "")
                  + (f"  METRICS_ERROR {row['spec']['error']}" if "error" in row["spec"] else ""),
                  flush=True)

    write_json(Path(args.out) / args.mode / f"speed_{args.preset}.json", {
        "mode": args.mode, "preset": args.preset, "thinking": args.thinking,
        "max_tokens": args.max_tokens, "n_requests": args.n_requests, "results": results,
        "diagnostic": diagnostic,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, help="label for this server config, e.g. ar / mtp2")
    ap.add_argument("--out", default="results/latest")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--server-env", type=Path, help="server launch record containing diagnostic flags")
    ap.add_argument("--prompt-file", type=Path, help="JSON object mapping prompt set names to ordered prompt lists")
    ap.add_argument("--preset", default="greedy", choices=["greedy", "instruct", "thinking", "qwen38"])
    ap.add_argument("--thinking", action="store_true", help="enable Qwen thinking mode")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--n-requests", type=int, default=64,
                    help="distinct prompts per set; the same list runs at every concurrency level")
    ap.add_argument("--sets", nargs="+", choices=["code", "math", "chat"])
    args = ap.parse_args()
    if min(args.max_tokens, args.n_requests, *args.concurrency) <= 0:
        ap.error("token count, request count and concurrency must be positive")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
