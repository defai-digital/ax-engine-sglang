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
| `bench_27b_nvfp4.py` | Fixed-output Qwen3.8-27B NVFP4 benchmark. Use `--verify-parity` to compare AX AR, tree 16 and chain 8 bit for bit, or `--verified-profile` to launch AR and chain 8 with `AX_QUALITY_PROFILE=verified` alone and compare them the same way; `--slots`, `--concurrency`, `--repeat` and `--output-tokens` size the passes. Use `--lanes stock8 replay8 tree16 tree32` for the GDN replay and tree-width comparison. |
| `ab_adaptive_width.py` | Throughput of chain-width policies on the same captured graphs, one server per lane: `w<W>` (fixed width), `table` (`AX_CHAIN_WIDTH_BY_BATCH`, `--table`, default `1:16,4:8`), `schedule` (`AX_CHAIN_SCHEDULE`, `--schedule`, default `16x31,8x1`) and `adaptive` (`--adaptive-env` for its settings). `--env NAME=VALUE` applies to every lane. Without the verified profile the generated text differs between lanes, so use many distinct prompts (`--input-ids`, about 200) and `--repeat 1`; ten prompts moved lanes by several percent. It compares the request limit the server reports with the sweep's concurrency; also check the server log for `max_running_requests is capped`, which SGLang prints when the GDN state cache lowers the limit |
| `run_stage1a.sh` | Qwen3.8-27B comparison with greedy, sampling and multi-user checks |
| `run_tree_plugin.sh` | Select and run SGLang stock and plugin lanes |
| `run_paired.sh` | Repeat matched benchmark lanes for paired statistics |
| `run_trtllm_dflash2.sh` | TensorRT-LLM DFlash2 comparison |
| `token_identity.py`, `sampling_dist_check.py` | Collect and compare greedy IDs and sampled distributions |
| `mixed_requests_check.py`, `stress_multiuser.py` | Check mixed serving, cancellation and slot reuse |

The regression runner can compare a current checkout with an older checkout
that still stores its validation scripts under `track_l/`.
[Regression runner](../../scripts/check_model_regression.py)
