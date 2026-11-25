# AX Engine for SGLang benchmarks

Run these tools from the repository root on a Linux NVIDIA CUDA host.
Install the plugin and SGLang version specified by the benchmark.
[Platform and model support](../../docs/README.md)

The shared prompt sets, speed harness and quality evaluator live in
the baseline suite. [Shared baselines](../baselines/)

## Stock and plugin baselines
```bash
source /root/venv-sgl/bin/activate          # uv pip install "sglang[all]"  (0.5.23)
GPU=0 PORT=30000 MODES="ar mtp3 mtp7" bash benchmarks/sglang/run_baselines.sh
GPU=1 PORT=30001 MODES="dflash dspark mtp5 mtp2" bash benchmarks/sglang/run_baselines.sh
python benchmarks/baselines/summarize.py results/track_a_baselines > results/track_a_baselines/REPORT.md
```
Each mode runs greedy without thinking and the model card sampling (temperature 1.0, top-p 0.95,
top-k 20) with thinking, on code (HumanEval), math (GSM8K) and chat prompts at concurrency 1, 8
and 32. Every mode sees the same 64 requests per set (prompt i, seed i). All modes run with
FlashInfer autotune and the radix prefix cache off, so the comparison uses one kernel policy
(autotune hangs for the DFlash draft graph on B200). DSpine is the exception: its plugin requires
the radix cache, so a sweep that includes `dspine` must set `RADIX_CACHE=1` for every mode
(`run_baselines.sh` refuses a mixed sweep otherwise). DSpine's plugin also accepts only the `fa3`
draft attention backend, which SGLang 0.5.22 supports on SM80..SM90 GPUs: on B200 the server does
not start (hang in draft graph capture; `NotImplementedError` from sgl-kernel without CUDA graphs),
so measured DSpine numbers need an H100/H200.

DSpine needs its own environment because the plugin pins SGLang 0.5.22:
```bash
git clone https://github.com/Tencent/DSpine && uv venv /root/venv-dspine && source /root/venv-dspine/bin/activate
uv pip install -e DSpine
GPU=0 PORT=30000 MODES="dspine" OUT=results/track_a_stage0 bash benchmarks/sglang/run_baselines.sh
```
Numbers from different SGLang versions are not compared with each other; Stage 0 re-measures every
baseline that enters a comparison on one version.

## Speculative decoding checks

| Entry point | Purpose |
|---|---|
| `bench_27b_nvfp4.py` | Fixed-output Qwen3.8-27B NVFP4 benchmark. Use `--verify-parity` to compare AX AR, tree 16 and chain 8 bit for bit, or `--verified-profile` to launch AR and chain 8 with `AX_QUALITY_PROFILE=verified` alone. Use `--lanes stock8 replay8 tree16 tree32` for the GDN replay and tree-width comparison. Compare `replay8` with `replay8-kv-graph`, `replay8-kv-direct`, or `replay8-plan-reuse` to measure the experimental KV append and planning paths. Each timed replay pass saves optimized and fallback counts by live batch size in `passes.json`. |
| `ab_adaptive_width.py` | Throughput of chain-width policies on the same captured graphs, one server per lane: `w<W>` (fixed width), `table` (`AX_CHAIN_WIDTH_BY_BATCH`, `--table`, default `1:16,4:8`), `schedule` (`AX_CHAIN_SCHEDULE`, `--schedule`, default `16x31,8x1`) and `adaptive` (`--adaptive-env` for its settings). `--mixed-concurrency 1 2 4 8` repeats those request waves in one pass. `--env NAME=VALUE` applies to every lane. Without the verified profile the generated text differs between lanes, so use many distinct prompts (`--input-ids`, about 200) and `--repeat 1`. Check the server log for `max_running_requests is capped` when the GDN state cache lowers the limit. |
| `run_stage1a.sh` | Qwen3.8-27B comparison with greedy, sampling and multi-user checks |
| `run_tree_plugin.sh` | Select and run SGLang stock and plugin lanes. Use `LANES="chain8n axreplay8"` for stock width-eight DFlash2 and AX accepted-prefix GDN replay. See the [profiling procedure](../../docs/decisions/2026-11-21-dflash-hotpath.md#stability-and-profiling-follow-up). |
| `run_paired.sh` | Repeat matched benchmark lanes for paired statistics |
| `run_trtllm_dflash2.sh` | TensorRT-LLM DFlash2 comparison |
| `token_identity.py`, `sampling_dist_check.py` | Collect and compare greedy IDs and sampled distributions |
| `mixed_requests_check.py`, `stress_multiuser.py` | Check mixed serving, cancellation and slot reuse |

The regression runner can compare a current checkout with an older checkout
that still stores its validation scripts under `track_l/`.
[Regression runner](../../scripts/check_model_regression.py)

`--verified-profile` requires `ar` when selecting lanes explicitly. A
timed replay pass requires at least one optimized round and zero fallbacks
at the requested live batch size. The results also record smaller batches.
[RTX PRO 6000 validation and workload settings](../../results/rtx_pro_6000_hotpath_validation/REPORT.md)

## DFlash2 FP8 verification tactics

`ab_fp8_verify.py` runs on one RTX PRO 6000 or RTX 5090 with SGLang 0.5.23,
FlashInfer 0.7.2, and the AX plugin installed. Use local Qwen3.8-27B NVFP4
and DFlash2 checkpoints. Capture and qualify profiles separately for each
GPU model. Set `TUNE_IDS` and `EVAL_IDS` to JSON files
containing arrays of token-ID lists. Use distinct held-out evaluation prompts
and keep both files fixed across comparable runs.

```bash
c=8
run="results/fp8_verify_tactics/raw/rtx-c${c}"
python benchmarks/sglang/ab_fp8_verify.py capture \
  --target "$TARGET" --draft "$DRAFT" --input-ids "$TUNE_IDS" \
  --concurrency "$c" --out "$run/capture"
python benchmarks/sglang/ab_fp8_verify.py probe \
  --operands "$run/capture/operands" --out "$run/probe"
python benchmarks/sglang/ab_fp8_verify.py compare \
  --target "$TARGET" --draft "$DRAFT" --input-ids "$EVAL_IDS" \
  --concurrency "$c" --profile "$run/probe/profile.json" --out "$run/compare"
```

The RTX 5090 C8 run used `--context-length 2048 --kv-tokens 8192
--mem-fraction-static 0.94` for both server commands. The four supported
concurrencies are C6, C8, C12, and C16, which use 48, 64, 96, and 128 rows
when all clients are live. Capture defaults to four distinct layers per
projection. Check `probe.json` for
rejected kernels and paired timings. Run `compare` when the probe emits
`profile.json`. It starts a fresh server for each lane in both launch
orders and requires verified candidate graph replay at the requested live
batch size. `summary.json` records throughput, p95 request latency,
tokens per verify, exact output sequences, and graph coverage. Use
`--output-tokens 128` in a separate output directory for a short-output
comparison.
