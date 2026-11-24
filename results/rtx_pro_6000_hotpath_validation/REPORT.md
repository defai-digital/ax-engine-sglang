# DFlash2 hot-path validation on RTX PRO 6000

Run on 2026-11-24 on one NVIDIA RTX PRO 6000 Blackwell Server Edition
(SM120, 97,887 MiB), driver 595.112.07, with SGLang 0.5.23, PyTorch
2.14.1+cu130, FlashInfer 0.7.2, and the AX Engine worktree at `9b6028d`.
The target is Qwen3.8-27B NVFP4 and the draft checkpoint is its DFlash2
drafter. Raw requests, launch configurations, counters, and server logs
are under `raw/` on the machine that collected the results.

## Verified width-eight chain

Each pass used the same ten frozen prompts, greedy decoding, and exactly
1,024 generated tokens per prompt. Both lanes used `AX_QUALITY_PROFILE=verified`,
eight request slots, and a 16,384-token KV pool. The first serialized AR
pass supplied the reference. All 40 AR outputs and all 40 chain outputs
matched that reference token for token. The second C1 pass repeated the
serialized check after concurrency reached eight.

| Pass | AR tok/s | Chain tok/s | Speedup | AR p95 (s) | Chain p95 (s) |
|---|---:|---:|---:|---:|---:|
| C1 | 53.89 | 110.59 | 2.05× | 19.76 | 11.63 |
| C2 | 108.88 | 197.32 | 1.81× | 19.51 | 12.27 |
| C8 | 258.74 | 469.25 | 1.81× | 20.98 | 15.34 |
| C1 repeat | 54.13 | 111.58 | 2.06× | 19.75 | 11.64 |

The chain emitted 2.83–2.84 tokens per verification. The ten-request C8
pass includes the tail after the active batch drops below eight.

## Native DFlash2 and accepted-prefix replay

The native width-eight lanes used the same ten frozen prompts, eight
request slots, FP8 target KV, and BF16 draft KV. Each lane ran both token
budgets after one serial and one eight-request warmup pass. The two launch
orders used fresh servers. All measured replay batches, including tails below the
requested concurrency, recorded zero fallbacks. Each C8 replay pass
observed 22 full batches at 128 tokens and 181 at 1,024 tokens.

| First lane | Tokens | Clients | Stock tok/s | Replay tok/s | Change | Exact sequences |
|---|---:|---:|---:|---:|---:|---:|
| Stock | 128 | 1 | 115.81 | 119.05 | +2.8% | 10/10 |
| Stock | 128 | 8 | 439.05 | 460.86 | +5.0% | 5/10 |
| Stock | 1,024 | 1 | 104.57 | 107.66 | +3.0% | 10/10 |
| Stock | 1,024 | 8 | 501.64 | 564.70 | +12.6% | 5/10 |
| Replay | 128 | 1 | 116.03 | 118.50 | +2.1% | 10/10 |
| Replay | 128 | 8 | 549.70 | 469.50 | −14.6% | 4/10 |
| Replay | 1,024 | 1 | 104.57 | 107.42 | +2.7% | 10/10 |
| Replay | 1,024 | 8 | 500.01 | 557.82 | +11.6% | 5/10 |

The first C8 differences appeared at token 3 of prompt 7 in both 128-token
runs (`220` versus `393`). At 1,024 tokens the first differences were
token 8 of prompt 1 (`1590` versus `25`) with stock first and token 1 of
prompt 2 (`32` versus `760`) with replay first. Stock itself matched only
5/10 C8 outputs across its two launches at each length, so the paired
comparisons include run-to-run variation. Replay eliminated the 10.12 GB
intermediate SSM snapshot allocation at eight slots.

The replay stress pass used 96 requests with 12 clients and requested a
20% cancellation share. It received all 12 issued cancellations, saw zero
request errors, reproduced all ten reference outputs across both the
pre-stress repeat and the post-stress pass, and returned token and Mamba
pool usage to the baseline probe.

## Warmed chat throughput

Each pass sent the same 64 held-out UltraChat prompts and generated 512
tokens per request. Each lane warmed at the requested concurrency on 16
other prompts. Servers used 16 request slots and a 32,768-token KV pool.
Profiling and NVTX were disabled for these passes.

| First lane | Clients | Stock tok/s | Replay tok/s | Change | Stock p95 (s) | Replay p95 (s) | Exact sequences |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock | 8 | 664.75 | 771.72 | +16.1% | 7.45 | 6.46 | 48/64 |
| Stock | 16 | 943.90 | 1,202.68 | +27.4% | 9.98 | 7.74 | 64/64 |
| Replay | 8 | 664.34 | 782.87 | +17.8% | 7.47 | 6.32 | 48/64 |
| Replay | 16 | 952.14 | 1,200.63 | +26.1% | 9.91 | 8.43 | 35/64 |

Acceptance stayed within 3.11–3.15 output tokens per verification. Replay
ran 1,142–1,161 full batches at C8 and 84–93 at C16 with zero fallbacks
across all measured sizes.

## Nsight attribution

Nsight Systems 2026.1.0 captured CUDA graph nodes and NVTX ranges on the
first 24 held-out chat prompts at the same 512-token budget. CUDA profiling
ran separately from the throughput passes. The values below are median
projected GPU spans over each captured pass, including its batch tail.

| Range | C8 stock (ms) | C8 replay (ms) | C16 stock (ms) | C16 replay (ms) |
|---|---:|---:|---:|---:|
| Draft forward | 4.07 | 4.07 | 4.32 | 4.34 |
| Target verify | 26.48 | 20.98 | 33.48 | 23.36 |
| Recurrent and convolution commit | 1.73 | 1.73 | 3.01 | 2.82 |
| Draft KV append | 0.34 | 0.35 | 0.36 | 0.36 |
| GPU idle between rounds | 0.15 | 0.15 | 0.12 | 0.14 |

Target verification accounts for the largest change. At C16 the GDN
update kernel inside the projected verify window went from 3.58 to 1.31
seconds of summed kernel time across 337 verify rounds. The CUTLASS
block-scaled and cuBLAS FP8 GEMMs made up 65% of the remaining C16 replay
verify kernel time. Prioritize target-verify GEMM tuning and GDN kernel
profiling on the replay lane. Typical gaps within a round were below
0.014 ms, and draft KV append stayed below 0.36 ms at the median.

## Mixed request waves

Each sweep sent 200 held-out prompts (60 code, 60 math, 80 chat), with 256
output tokens per request. All lanes drained the same sequence of waves
of 1, 2, 4, and 8 clients, including the partial final wave. The two
sweeps per lane used one warmed server with radix caching disabled.

| Lane | Output tok/s, mean | Request p95 (s), two sweeps | Tokens per verify |
|---|---:|---:|---:|
| Fixed width 16 | 411.91 | 3.22 / 3.26 | 4.78 |
| Fixed width 8 | 445.62 | 2.89 / 2.88 | 4.22 / 4.21 |
| Table `1:16,4:8` | 433.95 | 3.01 / 3.02 | 4.42 / 4.41 |

The table's runtime logs showed width 16 replay at live batches 1–3 and
width 8 replay at live batches 4–8. The table gained 5.4% throughput over
fixed width 16. Fixed width 8 led both throughput and p95 for this wave
schedule.

## Serialized tree divergence

The 16-node tree matched 8/10 serialized AR outputs at 1,024 tokens. The
AR score probe reproduced all ten original AR outputs. The tree probe
collected scores directly from accepted verify rows and checked their
argmaxes against the emitted tokens. Both divergences reproduced the
[prior run](../../docs/reports/2026-11-21-verified-profile-gate.md) on an
identical decoded prefix. Prompt and token positions below are one-based.

| Prompt | Token | AR choice | Tree choice | AR logprobs, AR / tree token | Tree logprobs, AR / tree token |
|---|---:|---:|---:|---:|---:|
| 2 | 330 | 5816 | 258 | −0.772 / −1.147 | −0.970 / −0.845 |
| 4 | 493 | 4338 | 2579 | −0.643 / −0.768 | −0.980 / −0.480 |

The target verification logits reverse the ranking of both candidate pairs
relative to AR. Width-eight chain decoding passed the verified gate. Native
replay and tree results support separate performance experiments.

## Software checks

The local CPU suite passed 284 tests with three skips. The profiler now
registers the AX hooks before wrapping the native worker when both plugins
are selected. A runtime import check forced the profiler to load first
and confirmed its `target_commit` span wrapped the accepted-prefix fold.
Grok and AGY reviewed the replay and mixed-width harnesses. Muse returned
a subscription quota error during this follow-up.
