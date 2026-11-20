# Code QA request: Track A baseline benchmark (Qwen3.8-27B on SGLang)

You are a code reviewer. Do not edit files. Answer in English, under 600 words. List concrete bugs
first, ordered by how much GPU time or result validity they would cost; give file:line and a fix.
Skip style remarks.

These files are running now on a pod with 2x B200 and will take ~2 GPU-hours; a bug means wrong
baseline numbers that every later decision depends on:
- track_a/serve_sglang.sh      SGLang launcher: ar, mtpN (built-in MTP head through the EAGLE
                               algorithm), dspark, dflash
- track_a/run_baselines.sh     per-GPU lane: start server, run the speed benchmark twice, stop it
- phase0/bench_speed.py        streaming OpenAI-client benchmark (TTFT, tok/s, concurrency levels)
- phase0/common.py             sampling presets, /metrics scraping, acceptance length
- phase0/summarize.py          report table, speedup vs AR and vs best MTP
- phase0/prompts.py            prompt sets

Facts you can rely on:
- SGLang 0.5.23. `/metrics` has counters `sglang:spec_verify_calls_total` and
  `sglang:generation_tokens_total`, and gauges `sglang:spec_accept_length`, `sglang:spec_accept_rate`.
- A smoke test of mode mtp3 worked: 83 tokens in 0.33 s; `chat_template_kwargs.enable_thinking`
  toggles reasoning; `--reasoning-parser qwen3` returns `reasoning_content`.
- Two lanes share one output directory and run concurrently: GPU 0 with MODES="ar mtp3 mtp2 mtp5
  mtp7", GPU 1 with MODES="dflash dspark" and EXTRA_ARGS="--disable-flashinfer-autotune" (the
  autotune step hung for the DFlash draft model on this GPU).
- The public numbers we want to reproduce: temperature 1.0, top-p 0.95, top-k 20, thinking on,
  concurrency 1/8/32, tok/s = total output tokens / wall time.

Check in particular:
1. Is the acceptance length (generated tokens / verify calls, per benchmark level, scraped as deltas
   around each level) right, including with warm-up requests, with two servers on different ports,
   and for the AR mode?
2. Throughput accounting: `output_tok_s`, per-request decode tok/s, TTFT when the first streamed
   chunk is reasoning, usage chunk handling, requests that end early (EOS before max_tokens), and
   whether a seed of 0 on every request makes sampled runs identical across concurrency slots.
3. Fairness between modes: same prompts and order, same max_tokens, warm-up, prefix/radix cache
   reuse between repeated prompts (the chat set repeats 16 prompts to fill 32; requests_per_slot),
   autotune on for MTP but off for the draft models, default KV dtype.
4. Shell robustness: `set -euo pipefail` with the cleanup trap, killing GPU processes by
   `nvidia-smi -i $GPU` when CUDA_VISIBLE_DEVICES is exported (index remapping), the two lanes
   writing `env.json`, a server that fails to start, `wait_for_server` timeout.
5. `--speculative-num-draft-tokens $((n + 1))` for MTP with `--speculative-num-steps n` and
   topk 1: is that the right relation?
6. Anything else that would make the AR / MTP / DSpark / DFlash2 comparison invalid.
