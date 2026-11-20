Verification passed (`py_compile`, `bash -n`, and a focused repro). The seven findings are recorded as structured QA artifacts with citable evidence. `review_complete` is the closure tool for the *review* lane and rejects *qa*-workflow findings; since this is a QA request, the registered QA findings + verification envelopes are the appropriate evidence, so I'll deliver the requested answer rather than duplicate artifacts under a different lane.

---

**Bugs, ordered by cost to result validity / GPU time**

1. **Radix cache + seed 0 + repeated prompts bias every mode.** `serve_sglang.sh:15-21` never sets `--disable-radix-cache`; `common.py:24` defaults `seed=0`; `prompts.py:39` pads 16 chat prompts to 32; `bench_speed.py:98` gives n_req=128 at c=32 (each prompt 4×). Identical repeated prompts hit cached prefill and share KV in-flight, deflating TTFT and inflating tok/s. The boost grows with concurrency (c=1: 16 distinct prompts; c=32: each reused 4×), distorting the scaling curve and absolute numbers vs public. **Fix:** `--disable-radix-cache` for all modes; `seed=None` or a unique per-request seed for sampled presets.

2. **Acceptance length is per-forward, wrong at c8/c32.** `common.py:88` = `generation_tokens_total / spec_verify_calls_total`. One verify forward covers the whole running batch, so this ≈ batch_size × per-request acceptance. Repro: per-request 3.0 → reported 3.0@c1 but 24.0@c8. The `spec_accept_length` gauge is never scraped (`common.py:42-43`). **Fix:** report the `sglang:spec_accept_length` gauge, or divide by mean batch size. Cross-check: the c8 value should be ~8× the gauge. (Premise: verify_calls counts forwards — confirm against the gauge.)

3. **Autotune asymmetry biases MTP vs DFlash/DSpark.** GPU1 runs `--disable-flashinfer-autotune` (`serve_sglang.sh:44` via EXTRA_ARGS); GPU0 (ar/mtp) keeps it on → draft models untuned, AR/MTP tuned. **Fix:** disable autotune for all modes (or fix the DFlash hang and enable everywhere); log EXTRA_ARGS per lane.

4. **env.json race can crash the report.** `run_baselines.sh:22-35`: both lanes write `$OUT/env.json` non-atomically; `summarize.py:13` does unguarded `json.loads`. Interleaving → invalid JSON → crash; last-writer-wins also drops GPU/MODES/EXTRA_ARGS provenance. **Fix:** per-lane `env_gpu$GPU.json`, atomic tmp+rename, record lane config; guard `load()`.

5. **Silent /metrics failure corrupts a level.** `common.py:53-54` returns `{}` on error; `spec_summary` (line 70) then deltas against 0 → cumulative counters (warmup + prior levels) reported as that level's acceptance. **Fix:** retry or flag the level invalid; never delta against a missing baseline.

6. **Thinking truncation invisible.** `bench_speed.py:44-49` omits `finish_reason`; `MAX_TOKENS_THINK=2048` (`run_baselines.sh:18`) likely truncates gsm8k/code reasoning. Throughput math is fine, but comparability to the public budget is uncheckable. **Fix:** capture finish_reason + truncation rate; match public max_tokens.

7. **Datasets fetched at bench time.** `prompts.py:25-37` calls `load_dataset` after server start, 14×; a cold HF cache/blocked egress fails every bench post-startup → wasted run. **Fix:** pre-cache/vendor prompts; load before starting servers.

**Checked, correct — don't "fix":** `$((n+1))` draft tokens with topk=1 is the right EAGLE relation (`serve_sglang.sh:28`). AR is safe, no div-by-zero (`common.py:86`). TTFT fires on the first reasoning chunk (`bench_speed.py:37`). Tokens come only from `usage.completion_tokens` — no double count, reasoning included. Early-EOS uses actual tokens/time. qwen38 = temp1.0/top_p0.95/top_k20, no presence_penalty. `nvidia-smi -i $GPU` is correct here: nvidia-smi uses physical indices (ignores CUDA_VISIBLE_DEVICES) and GPU is that same index.

*Evidence: py_compile + bash -n pass; repro confirms acceptance 3.0@c1 vs 24.0@c8, seed=0, chat 16/32.*