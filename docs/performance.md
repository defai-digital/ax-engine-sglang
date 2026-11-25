# Serving performance

## How to read the comparisons

- **Stock SGLang AR** uses standard autoregressive decoding without speculation.
- **Stock SGLang + DFlash2** uses SGLang's built-in DFlash2 chain verifier and
  the indicated draft block size.
- **SGLang + ax-engine-sglang** enables the AX Engine add-on and uses the
  verifier and width shown for that run.

Each row compares output tokens per second within one workload and platform.
The reports record checkpoint revisions, request limits and kernel settings.
[Model routes and precision support](README.md#model-routes)

## B200 · Qwen3.8-27B

The 64-slot NVFP4 run used 64 prompts per family, greedy decoding and a
1,024-token output limit. Each result is the mean of two passes. Cells
show code / math / chat output tok/s.

![B200 NVFP4 throughput for stock DFlash2 block 8 and the add-on at concurrency 1 and 16.](figures/b200-27b-nvfp4-throughput.svg)

| Concurrency | Stock SGLang + DFlash2 block 8 | SGLang + ax-engine-sglang |
|---|---:|---:|
| 1 | 816 / 733 / 401 | 1,160 / 971 / 458 (tree 16) |
| 16 | 3,628 / 4,670 / 2,980 | 4,379.5 / 4,989 / 3,110 (chain 8) |

The add-on measured **1.14×–1.42×** stock DFlash2 at concurrency one
and **1.04×–1.21×** at concurrency 16. The 64-slot NVFP4 campaign
paired the stock and add-on speculative routes. The study also records
other concurrency levels, BF16 and TensorRT-LLM results.
[B200 dense-model study](decisions/2026-11-16-dual-attention-backend.md)

### Experimental recurrent GDN replay

On one B200, the benchmark overlay kept SGLang's recurrent DFlash2
verification and replayed each accepted prefix into the FP32 state. The
reverse-order run measured width-8 DFlash2 at concurrency 16 with 32
request slots, 64 greedy NVFP4 chat prompts and a 1,024-token output limit.

![B200 NVFP4 chat at concurrency 16. Stock DFlash2 reached 2,685.6 output tok/s and the recurrent GDN replay overlay reached 2,957.3 output tok/s.](figures/b200-dflash2-recurrent-replay-throughput.svg)

| DFlash2 configuration | Output tok/s | Median decode tok/s per request | Median time to first token |
|---|---:|---:|---:|
| Stock GDN snapshots | 2,685.6 | 190.2 | 130 ms |
| Recurrent GDN replay overlay | 2,957.3 | 210.5 | 129 ms |

Replay raised aggregate throughput by **10.1%**. A separate 256-token
check matched all **64 of 64** stock greedy sequences.
[Benchmark data and reproduction](decisions/2026-11-16-dual-attention-backend.md#recurrent-verification-with-accepted-prefix-replay-2026-11-20)

### B200 DFlash2 serving retest

The 128-token retest pooled two streamed passes per route and used the
same greedy prompts for all lanes. Values are code / math / chat output
tok/s.

![B200 Qwen3.8-27B NVFP4 throughput at one and eight clients for stock AR, stock DFlash2, AX tree/chain, and AX GDN replay.](figures/b200-27b-nvfp4-serving-retest.svg)

| Clients | Stock AR | Stock DFlash2 | AX tree/chain | AX GDN replay |
|---:|---:|---:|---:|---:|
| 1 | 136.0 / 136.0 / 136.4 | 448.5 / 408.1 / 281.3 | **632.5** / **516.5** / **333.8** | 456.6 / 415.3 / 286.6 |
| 8 | 870.4 / 851.6 / 833.6 | 1,438.4 / 1,172.9 / 903.2 | 1,503.0 / 1,206.6 / 790.1 | **1,599.5** / **1,428.3** / **949.6** |

AX tree/chain led all three single-client workloads. Width-eight GDN
replay led all three at C8 and lowered chat p95 request latency from 1.41
to 1.34 s compared with stock DFlash2. An AR confirmation after checkpoint
staging reached 953.0 tok/s on C8 chat, close to the replay result.
[Per-pass throughput and output agreement](../results/blackwell_nvfp4_serving_retest/REPORT.md#b200-qwen38-27b-nvfp4)

## RTX PRO 6000 · Qwen3.8-27B NVFP4

All lanes used SGLang 0.5.23, a fixed 128-token greedy output, FlashInfer
attention and FlashInfer CUTLASS NVFP4 GEMM. The tests used the same frozen
prompts and include prefill and local HTTP overhead. The table pools two
passes by dividing total output tokens by total wall time. NVFP4 describes
the target checkpoint. The DFlash2 drafter was BF16.

![RTX PRO 6000 throughput by serving route and concurrency. Both panels use the same output token rate scale.](figures/rtx-27b-nvfp4-throughput.svg)

| Concurrency and requests per pass | Stock SGLang AR | Stock SGLang + DFlash2 | SGLang + ax-engine-sglang |
|---|---:|---:|---:|
| 1, 10 requests | 54.41 | 124.50 (block 8) | 156.87 (tree 64) |
| 8, 40 requests | 388.29 | 489.86 (block 8) | 484.78 (tree 32, chain width 8) |

Tree 64 delivered **2.88×** AR and **1.26×** stock DFlash2 at concurrency
one. Median request latency was 2.379 s for AR, 1.003 s for stock DFlash2
and 0.818 s for tree 64. At concurrency eight, stock DFlash2 and the add-on
exchanged the lead across the two passes. Their pooled rates were within
about 1%. [Full RTX sweep, per-pass rates and launch commands](../results/rtx_27b_nvfp4_speed/REPORT.md)

### RTX speed and parity profiles

The speed sweep ran with verification parity disabled. Tree 64 matched
packed AR on one of ten complete 128-token sequences in each pass. A
separate verification run used a 32-token output budget and AR configured
with the same parity numerics as the speculative lanes. Tree and chain
each matched that reference on all ten prompts. The independent AR and
DFlash2 cohorts passed all 28 checks in the recorded sampling comparison.
Parity AR reached 47.93 tok/s, parity chain 21.68 tok/s and parity tree
14.65 tok/s on that shorter workload. [Parity protocol and results](reports/2026-11-19-rtx-nvfp4-parity.md)

That first profile ran each target projection once per candidate. The current
profile pads AR decode and verification to groups of 16 projection rows. On
the same ten inputs, tree and chain again matched the AR reference on all ten
prompts and the sampling comparison passed all 28 checks.

| RTX PRO 6000 parity profile, 32-token budget, concurrency one | Output tok/s | Relative to AR |
|---|---:|---:|
| Parity AR | 46.34 | 1.00× |
| Parity DFlash2 tree, 16 nodes | 86.77 | 1.87× |
| Parity DFlash2 chain, width 8 | 88.24 | 1.90× |

With a 128-token budget, tree and chain matched AR on all ten complete
sequences. Tree reached 123.64 tok/s and chain 122.88 tok/s against 49.09
tok/s for parity AR.
[Row group measurements and limits](reports/2026-11-20-rtx-nvfp4-parity-row-groups.md)

Row groups matched only with one request in flight: at concurrency eight, 20
or 21 of 30 outputs equalled the same request alone. The profile now runs FP8
projections on a path that gave a row the same bits at every row count.

| RTX PRO 6000 parity profile, eight slots, 128-token budget | Equal to alone at C8 | Alone tok/s | C8 tok/s |
|---|---:|---:|---:|
| Parity AR | 30/30 | 48.84 | 237.80 |
| Parity DFlash2 tree, 16 nodes | 30/30 | 112.33 | 384.84 |
| Parity DFlash2 chain, width 8 | 30/30 | 118.02 | 401.66 |
| Parity DFlash2 tree, 16 nodes, former 16-row groups | 21/30 | 111.44 | 240.68 |

Tree 64 also matched AR on all ten inputs, at 71.16 tok/s alone with two slots.
[Invariant GEMM measurements and limits](reports/2026-11-20-rtx-nvfp4-parity-invariant-gemm.md)

GDN verification then moved from one decode call per candidate to one launch
per layer, with the same bits. On a second pod, with prefill graphs and
FlashInfer autotune enabled, every lane below equalled the reference AR on all
ten inputs and its own alone outputs on 30/30 requests at concurrency eight.

| RTX PRO 6000 parity profile, 128-token budget | Alone tok/s | C8 tok/s |
|---|---:|---:|
| Parity AR | 51.19 | 268.70 |
| Parity DFlash2 tree, 16 nodes | 160.70 | 570.58 |
| Parity DFlash2 tree, 32 nodes | 160.21 | 559.08 |
| Parity DFlash2 chain, width 8 | 140.74 | 575.09 |
| Parity DFlash2 tree, 64 nodes, two slots | 140.44 | 248.27 |

[One-launch GDN measurements and limits](reports/2026-11-20-rtx-nvfp4-parity-gdn-one-launch.md)

Those runs stopped at end-of-sequence. With exactly 128 tokens per request,
FlashInfer autotune changed one or two of ten outputs at concurrency eight, so
the parity defaults now disable it and keep prefill graphs on. Every lane below
equalled parity AR on all ten inputs and its own alone outputs at concurrency
eight. Tree 64 runs with eight slots through the capped snapshot buffer.

| RTX PRO 6000, parity defaults, exactly 128 tokens | C1 tok/s | C8 tok/s |
|---|---:|---:|
| Parity AR | 51.79 | 226.78 |
| Parity DFlash2 tree, 16 nodes | 118.09 | 493.03 |
| Parity DFlash2 tree, 32 nodes | 120.05 | 520.60 |
| Parity DFlash2 chain, width 8 | 109.76 | 507.35 |
| Parity DFlash2 tree, 64 nodes, `AX_SSM_POSITIONS=8` | 102.34 | 508.50 |

On the same pod with the speed sweep's flags, speed-profile tree 16 reached
126.59 and 449.86 tok/s and parity tree 16 112.70 and 387.57 tok/s.
[Fixed-length comparison and profile](reports/2026-11-20-rtx-nvfp4-parity-forced-128.md)

A narrower GDN value tile then gave GDN verification four times the programs
with the same bits. Parity tree 16 reached 128.53 tok/s at C1, tree 32 131.58,
chain 8 115.43 and tree 64 114.10; AR stayed at 51.82 with the same outputs.
[Value tile measurements](reports/2026-11-20-rtx-nvfp4-parity-forced-128.md#narrower-gdn-value-tile)

### Verified profile gate

`AX_QUALITY_PROFILE=verified` was then launched by itself on an RTX PRO 6000
with eight slots, so the launcher derived the parity settings (prefill chunk
4,096, no prefill graphs, radix cache and autotune off). Each lane sent the ten
frozen prompts repeatedly; every sequence was compared with AR alone.

| RTX PRO 6000, verified profile, eight slots | Equal to AR, 128 tokens | Equal to AR, 1,024 tokens | C1 tok/s | C8 tok/s, 128 / 1,024 tokens |
|---|---:|---:|---:|---:|
| AR | 1,000 of 1,000 | 120 of 120 | 52–54 | 362 / 356 |
| DFlash2 chain, width 8 | 1,000 of 1,000 | 120 of 120 | 109–113 | 489 / 561 |
| DFlash2 tree, 16 nodes | 1,000 of 1,000 | 116 of 120 | 126–132 | 499 / 592 |

The tree lane's four unequal sequences came from two prompts in two serialized
passes. The first differences were at output tokens 330 and 493, counted from
one. Chain rounds with two or more requests matched. The sequences are ten
distinct prompts repeated at concurrency 1, 2 and 8.

### DFlash2 hot-path follow-up

The verified width-eight chain matched all 40 1,024-token sequences in
the follow-up and delivered 2.05× AR throughput at C1 and 1.81× at C2 and
C8. The serialized tree probe reproduced both divergences and measured
reversed token rankings on the shared decoded prefixes.

Native accepted-prefix replay improved warmed chat throughput by 16–18%
at C8 and 26–27% at C16 in both launch orders, on 64 held-out prompts and
512 output tokens per request. P95 latency fell from 9.91–9.98 s to
7.74–8.43 s at C16. The stress test recorded 12 confirmed cancellations
and zero errors. The median C16 target-verify GPU span fell from 33.5 ms
to 23.4 ms, with draft KV append near 0.36 ms in both lanes.
[Throughput, token scores, and Nsight attribution](../results/rtx_pro_6000_hotpath_validation/REPORT.md)

### FP8 tactic serving comparison

The RTX PRO 6000 ran both launch orders at concurrency eight. Each timed
pass used 64 held-out UltraChat prompts and 512 greedy output tokens per
request. Both lanes used width-eight DFlash2 and GDN replay.

![RTX PRO 6000 Qwen3.8-27B NVFP4 throughput with both FP8 tactic policies at concurrency eight.](figures/rtx-pro-6000-27b-fp8-verify-throughput.svg)

| First lane | Default (output tok/s) | Candidate (output tok/s) | Change | Default → candidate p95 (s) | Exact outputs |
|---|---:|---:|---:|---:|---:|
| Default | 699.74 | 718.01 | +2.61% | 7.16 → 7.13 | 57/64 |
| Candidate | 710.20 | 724.04 | +1.95% | 6.96 → 7.09 | 57/64 |

The 64-row GDN QKVZ tactic reduced its projection time by 4.55% on
captured operands. The attention QKV tactic was within 0.1% of the native
tactic. In serving, both modes repeated their own 64 outputs across
launch orders. Seven outputs differed between modes in each order.
[RTX PRO 6000 projection audit and graph coverage](../results/fp8_verify_tactics/REPORT.md#rtx-pro-6000-live-qwen38-27b-nvfp4) · [RTX 5090 serving measurements](#fp8-verification-tactics-at-c8)

### Chain width by batch size

On the speed profile (no exact-output claim) the better fixed chain width
depends on how many requests run. The first comparison used the ten frozen
prompts; because the generated text differs between lanes without the verified
profile, it was repeated on 200 distinct prompts (256 output tokens, radix cache
off, two sweeps; concurrency two uses the second sweep, after a slow first pass).
Output tok/s against fixed width 16, RTX PRO 6000:

| Lane | C2 | C3 | C4 | C8 |
|---|---:|---:|---:|---:|
| width 8 | −3.9% | −0.8% | +3.1% | +9.0% |
| `AX_CHAIN_WIDTH_BY_BATCH="1:16,4:8"` | +0.1% | +0.7% | +3.0% | +9.3% |
| `AX_CHAIN_WIDTH_BY_BATCH="1:16,3:8"` | +0.1% | −1.8% | +3.9% | +8.9% |

Width 16 reached 347.9, 470.7, 570.9 and 792.6 tok/s. `1:16,4:8` was within 1%
of the better fixed width at every concurrency in this fixed-concurrency
sweep. Switching widths twice every 32 rounds cost nothing measurable.
The adaptive policy
(`AX_POLICY_MODE=adaptive`) was 2.3% below width 16 at concurrency two and 0.6%
below width 8 at eight on another pod, so it stays experimental and off by
default.
[Gate, width comparisons and limits](reports/2026-11-21-verified-profile-gate.md)

On 200 held-out prompts submitted in waves of 1, 2, 4, and 8 clients,
fixed width 8 led at 445.6 tok/s, followed by `1:16,4:8` at 434.0 and fixed
width 16 at 411.9. The table switched widths at four live requests.
[Mixed-wave throughput and latency](../results/rtx_pro_6000_hotpath_validation/REPORT.md#mixed-request-waves)

## GeForce RTX 5090 · Qwen3.8-27B NVFP4

Stock AR, stock DFlash2 and AX Engine ran on one 32 GB RTX 5090 with
SGLang 0.5.23 and two GPU request slots. The text workload used the RTX
PRO 6000 suite's frozen inputs and fixed 128-token greedy outputs. Rates
pool two passes and include prefill and loopback HTTP.

![GeForce RTX 5090 throughput by serving route at concurrency one and two. Both panels use the same output token rate scale.](figures/rtx-5090-27b-nvfp4-throughput.svg)

| Concurrency and requests per pass | Stock SGLang AR | Stock SGLang + DFlash2, block 8 | SGLang + ax-engine-sglang |
|---|---:|---:|---:|
| 1, 10 requests | 57.64 | 125.01 | **135.18** (tree 32) |
| 2, 40 requests | 101.77 | **182.38** | 181.36 (tree 32, chain width 8) |

Tree 32 delivered **2.35×** stock AR and **1.08×** stock DFlash2 at
concurrency one. AX Engine delivered **1.78×** stock AR at concurrency
two and finished within **0.6%** of stock DFlash2. Verification parity
was disabled for the timed comparison. Both speculative routes matched
packed AR on 0 of 10 complete outputs per pass at concurrency one.
[Per-pass timings, token matches and reproduction](../results/rtx_5090_27b_nvfp4/REPORT.md)

### Experimental recurrent GDN replay

The width-eight DFlash2 follow-up reused the frozen prompts and 128-token
greedy outputs. The two-slot rates pool four passes and the eight-slot
rate pools two. Rates include prefill and loopback HTTP.

![Stock DFlash2 and AX GDN replay throughput on the RTX 5090. Only replay completed the eight-slot run.](figures/rtx-5090-27b-gdn-replay-throughput.svg)

| Concurrency | Stock DFlash2 output tok/s | AX GDN replay output tok/s |
|---:|---:|---:|
| 1 | 123.21 | 124.39 |
| 2 | 178.67 | 181.19 |
| 8 | GPU memory limit | 389.63 |

Replay matched all 200 fixed-length stock DFlash2 sequences at one and
two concurrent requests. It replaced 3.38 GiB of speculative SSM
snapshots with about 0.017 GiB of replay buffers in the two-slot runs.
[Replay evidence and test settings](../results/rtx_5090_27b_nvfp4/REPORT.md#accepted-prefix-gdn-replay-follow-up)

### FP8 verification tactics at C8

Both lanes used width-eight DFlash2 with accepted-prefix GDN replay.
Each timed pass ran 64 held-out UltraChat prompts to 512 greedy output
tokens per request. The candidate used cuDNN for GDN QKVZ and attention
QKV at 64 verification rows.

![RTX 5090 end-to-end throughput for default and candidate FP8 tactics in both launch orders at C8.](figures/rtx-5090-27b-fp8-verify-throughput.svg)

| First lane | Default (output tok/s) | Candidate (output tok/s) | Change | Default → candidate p95 (s) | Exact outputs |
|---|---:|---:|---:|---:|---:|
| Default | 802.33 | 824.01 | +2.70% | 6.31 → 6.10 | 10/64 |
| Candidate | 807.25 | 824.02 | +2.08% | 6.30 → 6.10 | 10/64 |

Each lane reproduced all 64 of its own outputs across launch orders.
The candidate replayed 83 full eight-request tactic graphs per pass,
with zero fallbacks at that size. The isolated projections differed
by about 0.1% in paired timings. During serving, the candidate generated
3.17 tokens per verification versus 3.09 for the default lane.
[Full projection audit and serving results](../results/fp8_verify_tactics/REPORT.md#rtx-5090-live-qwen38-27b-nvfp4)

## Jetson Thor · Qwen3.8-27B

All rows use the single-request FlashInfer lanes on a Linux arm64 Jetson
Thor with SGLang 0.5.23. The 27B NVFP4 run used 12 prompts per family with
a 512-token output limit. The BF16 run used eight prompts per family with
a 384-token limit. Each cell gives code / math / chat output tok/s for
that checkpoint's workload.

![On Jetson Thor, the AX Engine tree has the highest measured throughput for code, math and chat in each BF16 and NVFP4 panel.](figures/jetson-thor-27b-throughput.svg)

| Target precision | Stock SGLang AR | Stock SGLang + DFlash2 chain 16 | SGLang + ax-engine-sglang tree 32 |
|---|---:|---:|---:|
| NVFP4 | 13.4 / 13.5 / 13.5 | 93.0 / 73.5 / 30.6 | 106.1 / 92.1 / 43.1 |
| BF16 | 4.6 / 4.6 / 4.6 | 43.1 / 29.6 / 12.5 | 47.5 / 37.6 / 18.3 |

The Thor report also records Qwen3.5-9B BF16 with the original DFlash
drafter. Its FlashInfer follow-up changed host load, and the chain lane
overlapped a download. See the report for the conditions and the separate
Triton lanes. [Jetson Thor results](reports/2026-11-19-jetson-thor.md)

### FP8 verification projections

The Thor-01 (SM110) probe timed the default cuBLAS tactic and the fastest
bit-exact candidate on synthetic E4M3 inputs. Each pair below shows
default → candidate median CUDA graph time per projection. Percentages
are medians of three paired time reductions.

![Thor-01 time reductions for bit-exact GDN QKVZ and attention QKV candidates against default cuBLAS, measured on synthetic CUDA inputs.](figures/thor-01-fp8-projection-time.svg)

| Verification rows | GDN QKVZ (µs) | Time reduction | Attention QKV (µs) | Time reduction |
|---:|---:|---:|---:|---:|
| 48 | 340.45 → 324.62 | 4.65% | 298.29 → 284.72 | 4.55% |
| 64 | 350.07 → 334.34 | **4.49%** | 312.32 → 294.06 | **5.88%** |
| 96 | 334.06 → 331.89 | 0.62% | 302.93 → 298.91 | 1.70% |
| 128 | 339.12 → 336.51 | 0.84% | 309.77 → 306.19 | 1.07% |

The probe excluded 32 of 182 candidate and shape combinations after
they changed BF16 output bits. The timing comparison covers individual
projections on Thor-01.
[Tactics, software versions, and workload](../results/fp8_verify_tactics/REPORT.md#thor-01-synthetic-cuda-measurements) · [Benchmark commands](../benchmarks/sglang/README.md#dflash2-fp8-verification-tactics)

## Qwen3.8-Flash-Next NVFP4 · MTP

Flash Next uses SGLang MTP with the add-on's QSA fix. Its comparison uses
stock MTP3 as the speculative baseline.

### RTX PRO 6000 Flash Next serving retest

Both passes used the same 128-token greedy workloads and pinned-host FP8
PLE. Values are code / math / chat output tok/s.

![RTX PRO 6000 Flash Next NVFP4 throughput by serving route and workload at one and eight clients.](figures/rtx-pro-6000-flash-next-nvfp4-throughput.svg)

| Clients | Stock AR | Stock MTP3 | AX MTP3 | AX MTP7 |
|---:|---:|---:|---:|---:|
| 1 | 83.1 / 83.8 / 83.7 | 183.5 / 164.1 / 134.0 | 182.1 / 164.8 / **135.0** | **218.9** / **174.0** / 114.7 |
| 8 | 414.1 / 421.5 / 399.3 | 679.0 / **582.6** / **455.4** | 594.2 / 572.4 / 405.5 | **728.0** / 542.7 / 356.5 |

At C8, AX MTP7 was 7.2% faster than stock MTP3 on code. Stock MTP3 led
math and chat. Both MTP3 routes reached 458.22 tok/s in their second
chat pass.
[RTX PRO 6000 setup and complete pass results](../results/blackwell_nvfp4_serving_retest/REPORT.md#rtx-pro-6000-qwen38-flash-next-nvfp4)

### B200 Flash Next serving retest

The 128-token retest used two streamed passes per lane. Values are
code / math / chat output tok/s.

![B200 Flash Next NVFP4 throughput for AR, MTP3, AX MTP3, and AX MTP7 at one and eight clients.](figures/b200-flash-next-nvfp4-serving-retest.svg)

| Clients | Stock AR | Stock MTP3 | AX MTP3 | AX MTP7 |
|---:|---:|---:|---:|---:|
| 1 | 136.9 / 136.4 / 130.6 | 304.9 / 273.3 / **223.4** | 302.8 / 270.9 / 213.3 | **389.7** / **297.1** / 208.9 |
| 8 | 614.7 / 717.4 / 594.2 | 720.7 / 947.0 / 684.5 | 745.1 / 976.8 / **715.4** | **1,126.9** / **978.8** / 607.5 |

AX MTP7 led code at both concurrencies. AX MTP3 led C8 chat with
715.4 tok/s and 1.94 s p95 request latency.
[B200 setup, pass rates, and exact-output counts](../results/blackwell_nvfp4_serving_retest/REPORT.md#b200-qwen38-flash-next-nvfp4)

### One B200

In the second B200 pass at
concurrency one, each lane used greedy decoding with a 256-token output
limit. Rates are in output tok/s.

![B200 Flash Next NVFP4 throughput for stock SGLang AR, stock MTP3 and add-on MTP7, with code, math and chat shown separately.](figures/b200-flash-next-mtp-throughput.svg)

| Workload | Stock SGLang AR | Stock SGLang MTP3 | SGLang MTP7 + ax-engine-sglang QSA fix |
|---|---:|---:|---:|
| Code | 204.9 | 435.2 | 547.9 |
| Math | 212.5 | 432.3 | 487.7 |
| Chat | 211.7 | 334.4 | 296.8 |

The wider MTP window improved code and math in this pass, while stock MTP3
had higher chat throughput. The report includes both passes, time to first
token and multi-client results. [B200 Flash Next release benchmark](reports/2026-11-18-b200-v0.1.1.md)

### Two Jetson Thor hosts

SGLang 0.5.23 served the RadixArk NVFP4 checkpoint with two pipeline stages,
one on each Linux arm64 host. The single-request runs used FlashInfer CUTLASS
for NVFP4 MoE and a 2.5 GbE link between hosts. Rates are in output tok/s.

![Flash Next throughput across two Jetson Thor hosts, including stock AR, stock MTP3, and the QSA fix with MTP3 and MTP7.](figures/jetson-thor-flash-next-mtp-throughput.svg)

| Workload | Stock SGLang AR | Stock SGLang MTP3 | MTP3 + ax-engine-sglang QSA fix | MTP7 + ax-engine-sglang QSA fix |
|---|---:|---:|---:|---:|
| Code | 18.3 | 32.0 | 33.3 | **43.1** |
| Math | 19.0 | 32.4 | 32.5 | **37.5** |
| Chat | 19.2 | 24.8 | **25.7** | 22.8 |

The two add-on routes used a 16-slot QSA pending ring. MTP7 gave the highest
code and math throughput. The MTP3 add-on route led the four measured chat
lanes. [Thor Flash Next configuration and accepted tokens](reports/2026-11-19-jetson-thor.md#5-qwen38-flash-next-nvfp4-over-two-thors)

## Reproduce

- [Run the GeForce RTX 5090 comparison](../results/rtx_5090_27b_nvfp4/REPORT.md#reproduce-the-comparison).
- [Run the RTX three-route comparison](../results/rtx_27b_nvfp4_speed/REPORT.md#launch-the-measured-configurations).
- [Run SGLang benchmarks and serving checks](../benchmarks/sglang/).
- [Validate benchmark records](../contracts/benchmark-result-v1.schema.json).
