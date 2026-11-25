# DFlash2 FP8 verification tactics

## RTX PRO 6000 serving

The assigned Runpod host had no free GPU, so this change has no new
RTX PRO 6000 serving or SM120 CUTLASS measurements. The reference below
is the earlier Qwen3.8-27B NVFP4 width-eight DFlash2 run with accepted-prefix
GDN replay on 64 held-out UltraChat prompts and 512 output tokens per request.
The ranges cover both launch orders in that run.
[Prior RTX measurements](../rtx_pro_6000_hotpath_validation/REPORT.md#warmed-chat-throughput)

| Live clients | Replay output tok/s | p95 request latency |
|---|---:|---:|
| 8 | 771.72–782.87 | 6.32–6.46 s |
| 16 | 1,200.63–1,202.68 | 7.74–8.43 s |

## Thor-01 synthetic CUDA measurements

The probe ran on NVIDIA Thor (SM110) using SGLang 0.5.23, FlashInfer 0.7.2,
PyTorch 2.14.1+cu130, and cuDNN 9.24. Each row count used two seeded
synthetic E4M3 activation and weight pairs per projection, with BF16
outputs and FP32 scales of 0.0625 and 0.125. The seed was 42 and the
workspace was 40 MiB.

Times are medians across three paired graph trials, in microseconds per
projection. Each timed trial replayed the graph four times. The graph
contained 12 iterations over the two weight sets. The reduction is the
median of the three paired relative changes. The table reports the fastest
bit-exact tactic by that measure, compared with the cuBLAS default.

| Rows | Projection | Fastest exact tactic | Native (µs) | Candidate (µs) | Time reduction |
|---:|---|---|---:|---:|---:|
| 48 | GDN QKVZ | cuBLAS 6 | 340.45 | 324.62 | 4.65% |
| 48 | Attention QKV | cuDNN 9, knob 12=228 | 298.29 | 284.72 | 4.55% |
| 64 | GDN QKVZ | cuDNN 9, knob 12=228 | 350.07 | 334.34 | 4.49% |
| 64 | Attention QKV | cuDNN 9, knob 12=228 | 312.32 | 294.06 | 5.88% |
| 96 | GDN QKVZ | cuBLAS 5 | 334.06 | 331.89 | 0.62% |
| 96 | Attention QKV | cuDNN 9, knob 12=132 | 302.93 | 298.91 | 1.70% |
| 128 | GDN QKVZ | cuBLAS 5 | 339.12 | 336.51 | 0.84% |
| 128 | Attention QKV | cuBLAS 3 | 309.77 | 306.19 | 1.07% |

Of 182 candidate and shape combinations, 150 matched in eager and graph
execution. The other 32 changed BF16 bits in eager execution and were
excluded. A separate synthetic hook test used FlashInfer 0.7.2 and CUDA
graph replay to confirm the selected 64-row attention tactic matched BF16
bits and reported coverage. These tests cover isolated projections.

## Reproduce on RTX PRO 6000

Run `capture` and `probe` at each target concurrency. Run both launch
orders with `compare` on held-out prompts if the probe emits a profile.
Record the paired serving results from `summary.json` alongside the
projection audit.
[Benchmark commands](../../benchmarks/sglang/README.md#dflash2-fp8-verification-tactics)
