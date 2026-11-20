# Phase 0: Qwen3.6-35B-A3B baseline

Goal: measure the bar that any NCM design must clear. We compare plain autoregressive (AR)
decoding with Qwen's built-in MTP speculative decoding, using vLLM on one GPU.

## What it measures

| Script | Output |
|---|---|
| `bench_speed.py` | Per prompt set (code / math / chat) and concurrency (1, 4, 16, 64): aggregate output tok/s, per-request decode tok/s, TTFT, and MTP mean acceptance length (from vLLM `/metrics`) |
| `eval_quality.py` | HumanEval pass@1, MBPP-sanitized pass@1, GSM8K accuracy (greedy, non-thinking); per-sample outputs |
| `summarize.py` | `REPORT.md`: speed tables with speed-up vs `ar`, quality, greedy output parity of each MTP mode vs `ar` |

Modes: `ar`, `mtp1`, `mtp2`, `mtp3` (number of MTP draft tokens). The model card recommends `mtp2`.

## Run

```bash
bash benchmarks/baselines/setup_pod.sh
export HF_HOME=/workspace/hf && source /workspace/venv-benchmarks/baselines/bin/activate

# smoke test (~15 min)
MODES="ar mtp2" LIMIT=20 CONCURRENCY="1 8" SAMPLED=0 benchmarks/baselines/run_all.sh

# full sweep (several hours: full HumanEval 164 + MBPP 257 + GSM8K 1319 per mode)
benchmarks/baselines/run_all.sh
```

Each sweep needs a new output directory. The runner checks that its port is free
and confirms the server advertises the sweep's model name before benchmarking.

Running pieces by hand:

```bash
benchmarks/baselines/serve.sh mtp2 &                       # in one shell
python benchmarks/baselines/bench_speed.py --mode mtp2 --out results/manual --concurrency 1 16
python benchmarks/baselines/eval_quality.py --mode mtp2 --out results/manual --tasks humaneval --limit 50
python benchmarks/baselines/summarize.py results/manual
```

Knobs (env): `MODEL`, `TP` (default 1), `MAX_LEN` (32768), `GPU_UTIL`, `MTP_METHOD`
(default `qwen3_next_mtp` per the model card; try `mtp` if your vLLM rejects it), `EXTRA_ARGS`
(passed through to `vllm serve`, e.g. `--no-enable-prefix-caching`).

## Defaults and caveats

- **Non-thinking mode** by default (`enable_thinking: false`) so evals finish quickly. Add
  `--thinking --preset thinking --max-tokens 16384` for thinking-mode numbers. Expect much longer runs.
- Speed benchmarks use **natural stopping** (no `ignore_eos`), because MTP acceptance depends on real
  content. Compare modes by tok/s, not by wall time per request.
- The decode rate is estimated from the interval between the first and last content chunks.
  Requests with a single content chunk contribute to aggregate throughput and TTFT. Their
  per-request decode duration is recorded as `null`.
- Pass `--server-env` with the launch record when benchmarking instrumented SGLang runs.
  Reports label their timings as diagnostic and exclude them from speed ratios.
- Pass `--prompt-file` to reuse a JSON mapping of workload names to ordered prompt lists.
  Each speed record carries the measured list's SHA-256 so the paired comparison can
  verify prompt identity and order. Track A runs accept the same file through `PROMPT_FILE`
  and record the precision label supplied in `BENCHMARK_PRECISION`.
- Speed prompts repeat across concurrency levels, so vLLM prefix caching can lower TTFT. Decode tok/s is
  unaffected. Use `EXTRA_ARGS=--no-enable-prefix-caching` for cold-prefill TTFT.
- Greedy parity is "byte-identical output". Speculative decoding is lossless in exact arithmetic, but
  batch-dependent GPU numerics can flip near-tied tokens, so read parity together with accuracy.
- `eval_quality.py` **executes model-written code** (subprocess, 15 s timeout, isolated `python -I`).
  Run it on the pod.

## Gate to Phase 1

Record from `REPORT.md`:
1. Batch-1 decode tok/s for `ar` and the best MTP mode, per prompt set (code matters most).
2. Mean acceptance length of the best MTP mode.
3. HumanEval / MBPP / GSM8K for `ar` and MTP (they should match within noise).

These numbers become the bar for Phase 2: an NCM drafter must reach ≥1.5× the best MTP mode's
end-to-end tok/s on code and chat at batch 1, with outputs identical to the verifier's.
