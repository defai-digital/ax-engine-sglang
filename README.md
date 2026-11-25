# AX Engine for SGLang

**AX Engine's Linux CUDA edition for NVIDIA GPUs on amd64 and arm64.**

`ax-engine-sglang` brings AX Engine's inference acceleration approach to
SGLang as a Python add-on. SGLang runs the server, schedules requests and
executes the model on CUDA. The add-on extends speculative decoding, state
verification and model-specific attention behavior.

The Apple Silicon edition runs on MLX. The two editions share speculative
acceptance rules and benchmark evidence contracts, with serving paths and
checkpoint formats specific to each runtime.
[Documentation](docs/README.md) · [Benchmarks](docs/performance.md) · [AX Engine](https://github.com/defai-digital/ax-engine)

## How it speeds up decoding

AX aims to emit more tokens per target-model pass while keeping the target in
charge of what gets emitted. On dense Qwen models, it can verify multiple
candidate continuations as a tree for one active request and use linear
chains when requests are batched. On Flash Next, it fixes the sparse
attention state so SGLang's multi-token prediction (MTP) can verify longer
drafts. Each route pays off when it saves more time in target passes than it
adds in drafting and verification.
[How it works and why we chose this design](docs/design.md)

## Quick start

Use Python 3.12 on Linux with an NVIDIA GPU and a CUDA 13-compatible driver.
Install the v0.5.1 GitHub release wheel in a fresh virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install https://github.com/defai-digital/ax-engine-sglang/releases/download/v0.5.1/ax_engine_sglang-0.5.1-py3-none-any.whl
```

Use `python -m pip install ./ax-engine-sglang` from the repository root for a
source checkout. If you already have a working SGLang 0.5.23 environment, add
`--no-deps` to install only the add-on.
[Installation and release status](docs/README.md#installation-and-release-status)

The `serve` command activates the AX plugin automatically and accepts SGLang
server flags. Use a GPU with enough memory for the target and the drafter.

```bash
ax-engine-sglang serve RadixArk/Qwen3.8-27B-NVFP4 \
  --speculative-algorithm FC1_DFLASH_TREE \
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 16
```

The 0.2.1 wheel served this target and drafter on Jetson Thor. A single greedy
request returned 32 tokens and the trace contained six DFlash2 tree rounds.
[Thor smoke test and flags](docs/README.md#021-installed-wheel-smoke-on-jetson-thor) · [Launch defaults and multi-user settings](ax-engine-sglang/README.md#launch-defaults)

## Changes in v0.5.1

Document the RTX PRO 6000 verification priorities for Qwen3.8-27B NVFP4,
including live FP8 tactic validation and verify-row sweeps informed by the
sibling AX engines. [Performance review](docs/decisions/2026-11-21-dflash-hotpath.md#rtx-pro-6000-target-verification-priorities)

## Changes in v0.5.0

- Add `ax-engine-sglang doctor` and a verified launch profile for AR and
  width-eight DFlash2 chain decoding.
- Add experimental chain width selection by live batch size and validate
  recurrent state capacity before loading weights.
- Measure accepted-prefix GDN replay on RTX PRO 6000, with live batch
  counters, cancellation checks, and Nsight phase timings.

[Release notes](https://github.com/defai-digital/ax-engine-sglang/releases/tag/v0.5.0) · [RTX PRO 6000 hot-path validation](results/rtx_pro_6000_hotpath_validation/REPORT.md)

## Supported platforms

| Host | NVIDIA configurations with recorded runs |
|---|---|
| Linux amd64 (x86_64) | B300, B200, RTX PRO 6000, and GeForce RTX 5090 |
| Linux arm64 (aarch64) | Jetson Thor |

SGLang 0.5.23 is the reference runtime. Use a CUDA-enabled SGLang environment
with NVIDIA drivers and wheels or a container built for your host
architecture. Model, precision, memory and kernel requirements vary by
configuration. [Platform and model records](docs/README.md#platforms)

## SGLang compatibility and accelerated models

Use the standard SGLang server, client API and model loader. Stock AR, MTP
and DFlash are available for the models and configurations supported by
SGLang. The table lists the routes accelerated by this add-on.

| Model family | Measured target precision | What the add-on accelerates |
|---|---|---|
| Qwen3.5-9B | BF16 | DFlash v1 tree verification on eligible requests |
| Qwen3.8-27B dense | BF16 and NVFP4 | DFlash2 tree verification for eligible single requests and chain verification for grouped requests |
| Qwen3.8-Flash-Next (Qwen4-Exp) | BF16 and NVFP4 | SGLang MTP with the QSA pending-ring fix. DFlash2 tree serving remains unavailable |

For NVFP4, load an existing SGLang-compatible NVFP4 checkpoint with the
GPU kernels required by that checkpoint. Dense 27B and Flash Next were
verified on separate serving routes. [Model routes and precision support](docs/README.md#model-routes)

## Measured throughput

[Qwen3.8-27B](#qwen38-27b) · [Qwen3.8-Flash-Next](#qwen38-flash-next)

### Qwen3.8-27B

#### B200 · Qwen3.8-27B NVFP4

Greedy runs used 64 prompts per family, a 1,024-token output limit and 64
server slots. Each rate averages two passes. Values are output tok/s in
code / math / chat order.

![B200 NVFP4 throughput for stock DFlash2 and the add-on at concurrency 1 and 16.](docs/figures/b200-27b-nvfp4-throughput.svg)

| Concurrency | Stock SGLang + DFlash2 block 8 | SGLang + ax-engine-sglang |
|---|---:|---:|
| 1 | 816 / 733 / 401 | **1,160 / 971 / 458** (tree 16) |
| 16 | 3,628 / 4,670 / 2,980 | **4,379.5 / 4,989 / 3,110** (chain 8) |

[Full B200 27B benchmark and launch settings](docs/performance.md#b200--qwen38-27b)

##### Experimental recurrent GDN replay

The benchmark overlay used width-8 stock DFlash2 on one B200 with SGLang
0.5.23 and 32 request slots. Both timed runs used the same 64 greedy NVFP4
chat prompts at concurrency 16 and a 1,024-token output limit.

![B200 NVFP4 chat at concurrency 16. Stock DFlash2 reached 2,685.6 output tok/s and the recurrent GDN replay overlay reached 2,957.3 output tok/s.](docs/figures/b200-dflash2-recurrent-replay-throughput.svg)

| DFlash2 configuration | Output tok/s | Median decode tok/s per request | Median time to first token |
|---|---:|---:|---:|
| Stock GDN snapshots | 2,685.6 | 190.2 | 130 ms |
| Recurrent GDN replay overlay | **2,957.3** | **210.5** | 129 ms |

Replay raised aggregate throughput by **10.1%**. A separate 256-token
check matched all **64 of 64** stock greedy sequences.
[Benchmark data and reproduction](docs/decisions/2026-11-16-dual-attention-backend.md#recurrent-verification-with-accepted-prefix-replay-2026-11-20)

##### B200 DFlash2 serving retest

Each route ran two streamed passes with exactly 128 greedy output tokens
per request. Values are code / math / chat output tok/s.

![B200 Qwen3.8-27B NVFP4 serving retest at one and eight clients, with stock AR, stock DFlash2, AX tree/chain, and AX GDN replay.](docs/figures/b200-27b-nvfp4-serving-retest.svg)

| Clients | Stock AR | Stock DFlash2 | AX tree/chain | AX GDN replay |
|---:|---:|---:|---:|---:|
| 1 | 136.0 / 136.0 / 136.4 | 448.5 / 408.1 / 281.3 | **632.5** / **516.5** / **333.8** | 456.6 / 415.3 / 286.6 |
| 8 | 870.4 / 851.6 / 833.6 | 1,438.4 / 1,172.9 / 903.2 | 1,503.0 / 1,206.6 / 790.1 | **1,599.5** / **1,428.3** / **949.6** |

At eight clients, AX GDN replay improved on stock DFlash2 by
**11.2% / 21.8% / 5.1%**. A separate warmed AR check reached 953.0 tok/s
on chat. [B200 setup and per-pass results](docs/performance.md#b200-dflash2-serving-retest)

#### RTX PRO 6000 · Qwen3.8-27B BF16 and NVFP4

Two passes per route used 128-token greedy outputs. Cells give end-to-end
output tok/s in code / math / chat order.

![RTX PRO 6000 Qwen3.8-27B BF16 and NVFP4 throughput at concurrency eight by workload and serving route.](docs/figures/rtx-pro-6000-27b-retest-throughput.svg)

##### BF16

| Clients | Stock AR | Stock DFlash2 | AX tree / chain |
|---:|---:|---:|---:|
| 1 | 27.1 / 27.2 / 27.2 | 132.0 / 120.9 / 70.5 | **184.3** / **163.2** / **94.1** |
| 8 | 199.4 / 200.5 / 202.4 | 487.5 / 567.6 / 333.7 | **510.8** / **572.0** / **335.1** |

##### NVFP4

| Clients | Stock AR | Stock DFlash2 | AX tree / chain | AX GDN replay |
|---:|---:|---:|---:|---:|
| 1 | 54.2 / 54.1 / 54.4 | 229.3 / 188.5 / 125.2 | **324.7** / **270.9** / **152.2** | 235.8 / 195.2 / 128.9 |
| 8 | 376.0 / 377.3 / 367.0 | 819.2 / 760.5 / 505.3 | 907.7 / 823.3 / 487.0 | **996.8** / **898.1** / **580.8** |

At eight clients, BF16 chat reached 335.1 tok/s on AX tree/chain and
333.7 on stock DFlash2. NVFP4 GDN replay raised chat throughput from
505.3 to **580.8 tok/s (+14.9%)** and matched stock on all 48 greedy chat
outputs. The reverse-order run measured an 11.2% gain and matched 38 of 48
stock outputs.
[RTX PRO 6000 retest and latency results](results/rtx_pro_6000_retest/REPORT.md)

The RTX PRO 6000 hot-path follow-up matched serialized AR on all 40
1,024-token outputs with the verified width-eight chain, at **2.05×** AR
throughput at C1 and **1.81×** at C2 and C8. In a separate warmed chat
benchmark, accepted-prefix GDN replay improved native DFlash2 throughput
by **16–18% at C8** and **26–27% at C16** across both launch orders.
Replay's greedy outputs varied with batching. [Parity, cancellation, and profiling results](results/rtx_pro_6000_hotpath_validation/REPORT.md)

##### RTX PRO 6000 · FP8 verification tactics

At eight concurrent clients, both launch orders used 64 chat prompts
with exactly 512 greedy output tokens per request. The candidate cuBLAS
tactics improved throughput by **1.95–2.61%** and matched **57 of 64**
complete outputs in each order. The tactics remain experimental.

![RTX PRO 6000 throughput at eight clients with default and candidate FP8 tactics in both launch orders.](docs/figures/rtx-pro-6000-27b-fp8-verify-throughput.svg)

| First lane | Default (output tok/s) | Candidate (output tok/s) | Change | Exact outputs |
|---|---:|---:|---:|---:|
| Default | 699.74 | 718.01 | +2.61% | 57/64 |
| Candidate | 710.20 | 724.04 | +1.95% | 57/64 |

[RTX PRO 6000 operands and serving results](results/fp8_verify_tactics/REPORT.md#rtx-pro-6000-live-qwen38-27b-nvfp4) · [RTX 5090 serving results](#rtx-5090--fp8-verification-tactics) · [Thor-01 projection results](#thor-01--fp8-verification-projections)

With GDN replay already enabled, a separate draft KV graph added 0.8%
to eight-client streamed chat throughput and reduced throughput on the
frozen prompt set by 3.8%. [Draft KV graph measurements](results/rtx_pro_6000_kv_graph/REPORT.md)

[Streamed chat comparison for stock SGLang, ax-engine-sglang, and vLLM](results/rtx_pro_6000_plan_reuse/REPORT.md#sglang-ax-engine-sglang-and-vllm-on-rtx-pro-6000)

##### RTX PRO 6000 · Qwen3.8-27B NVFP4

The original speed sweep used SGLang 0.5.23, the same target checkpoint,
ten frozen prompts and fixed 128-token greedy outputs. Rates are end-to-end
output tok/s pooled over two passes in speed mode.

![RTX PRO 6000 throughput for stock SGLang AR, stock DFlash2 and the AX Engine add-on at concurrency 1 and 8. The add-on leads at concurrency 1. The two speculative routes are close at concurrency 8.](docs/figures/rtx-27b-nvfp4-throughput.svg)

| Concurrency and requests per pass | Stock SGLang AR | Stock SGLang + DFlash2 | SGLang + ax-engine-sglang |
|---|---:|---:|---:|
| 1, 10 requests | 54.41 | 124.50 (block 8) | **156.87** (tree 64) |
| 8, 40 requests | 388.29 | **489.86** (block 8) | 484.78 (tree 32, chain width 8) |

At concurrency one, tree 64 delivered **2.88×** stock AR and **1.26×**
stock DFlash2. At concurrency eight, stock DFlash2 and the add-on traded
the lead between passes and finished within about 1% of each other.
[Full RTX benchmark and launch settings](results/rtx_27b_nvfp4_speed/REPORT.md)

Fast tree 64 matched packed AR on one of ten complete sequences per pass.
The opt-in parity profile (`AX_NVFP4_VERIFY_PARITY=1`) gives AR decode and
verification the same projection kernels. Its tree and chain routes matched
its own AR reference on all ten complete sequences, and AR and tree passed
all 28 sampled distribution checks.

| Parity profile, 10 requests of exactly 128 tokens | C1 tok/s | C8 tok/s |
|---|---:|---:|
| Parity AR | 51.82 | 241.60 |
| Parity DFlash2 chain, width 8 | 115.43 | 474.94 |
| Parity DFlash2 tree, 16 nodes | 128.53 | 466.20 |
| Parity DFlash2 tree, 32 nodes | **131.58** | **538.61** |
| Parity DFlash2 tree, 64 nodes, `AX_SSM_POSITIONS=8` | 114.10 | 411.01 |

Every lane equalled parity AR on all ten inputs, and at concurrency eight
every request equalled the same request alone. The two C8 passes of one lane
differed by up to 45%, so compare C1. On the same pod, speed-profile tree 16
reached 126.59 tok/s at C1, and no speed-profile lane kept its outputs at
concurrency eight. The parity AR reference differs from packed AR.
[RTX parity results and limits](docs/reports/2026-11-20-rtx-nvfp4-parity-forced-128.md)

Those comparisons cover 128 tokens. With 1,024-token outputs, AR and chain 8
still equalled the AR reference on every sequence at concurrency 1, 2 and 8,
but the tree route differed on two of ten prompts, from output tokens 329 and
492. `AX_QUALITY_PROFILE=verified` applies the measured parity settings and
starts only on a route with exact-output evidence: AR, or the chain route
(`AX_TREE=0`, draft block 8) at 109–113 tok/s alone and 489–561 tok/s at
concurrency eight. It refuses the tree route. `ax-engine-sglang doctor` checks
a launch against these rules without loading the model.
[Verified profile gate](docs/reports/2026-11-21-verified-profile-gate.md) · [Profile settings](ax-engine-sglang/README.md#launch-check-quality-profile-and-adaptive-chain-width)

#### GeForce RTX 5090 · Qwen3.8-27B NVFP4

One 32 GB RTX 5090 ran SGLang 0.5.23 with two GPU request slots. Each
request generated 128 greedy tokens from the same frozen inputs used on
the RTX PRO 6000. Rates pool two passes and include prefill and local HTTP
overhead.

![GeForce RTX 5090 output throughput. AX Engine leads at concurrency one. Stock DFlash2 and AX Engine are within 0.6% at concurrency two.](docs/figures/rtx-5090-27b-nvfp4-throughput.svg)

| Concurrency and requests per pass | Stock SGLang AR | Stock SGLang + DFlash2, block 8 | SGLang + ax-engine-sglang |
|---|---:|---:|---:|
| 1, 10 requests | 57.64 | 125.01 | **135.18** (tree 32) |
| 2, 40 requests | 101.77 | **182.38** | 181.36 (tree 32, chain width 8) |

At concurrency one, AX Engine delivered **2.35×** stock AR and **8.1%**
more throughput than stock DFlash2. At concurrency two, the speculative
routes were within **0.6%** of each other. These are speed-profile runs
with verification parity disabled. Both speculative routes matched packed
AR on 0 of 10 complete outputs per pass at concurrency one.
[Full RTX 5090 benchmark and launch settings](results/rtx_5090_27b_nvfp4/REPORT.md)

The optional FP8 parity retest used the same prompts and 128-token outputs.
AR, tree 16 and chain 8 each matched all 100 sequences from the serialized
AR reference across concurrency one and two.
[RTX 5090 parity results](results/rtx_5090_27b_nvfp4/REPORT.md#fp8-parity-retest)

##### Experimental recurrent GDN replay

The width-eight DFlash2 follow-up used the same checkpoint, inputs, and
fixed 128-token outputs. The C1 and C2 rates pool four passes per route
with two server slots. The C8 run used eight slots and two passes.

![RTX 5090 DFlash2 replay throughput at concurrency 1, 2, and 8. The stock eight-slot launch ran out of GPU memory during cache sizing.](docs/figures/rtx-5090-27b-gdn-replay-throughput.svg)

| Concurrency | Stock DFlash2 output tok/s | AX GDN replay output tok/s |
|---:|---:|---:|
| 1 | 123.21 | 124.39 |
| 2 | 178.67 | 181.19 |
| 8 | GPU memory limit | **389.63** |

At two slots, replay replaced **3.38 GiB** of speculative state snapshots
with about **0.017 GiB** of replay buffers. The two-slot runs matched all
**200 of 200** complete stock DFlash2 outputs. The eight-slot replay run
also passed cancellation and slot reuse checks.
[Replay benchmark and validation](results/rtx_5090_27b_nvfp4/REPORT.md#accepted-prefix-gdn-replay-follow-up)

##### RTX 5090 · FP8 verification tactics

The cuDNN candidate raised end-to-end throughput by **2.08–2.70%** on
Qwen3.8-27B NVFP4 at eight concurrent clients. It matched the default
cuBLAS lane on **10 of 64** complete greedy outputs in each launch order.

![RTX 5090 Qwen3.8-27B NVFP4 throughput for default and candidate FP8 tactics across both launch orders, at eight concurrent clients.](docs/figures/rtx-5090-27b-fp8-verify-throughput.svg)

| First lane | Default (output tok/s) | Candidate (output tok/s) | Change | Exact outputs |
|---|---:|---:|---:|---:|
| Default | 802.33 | 824.01 | +2.70% | 10/64 |
| Candidate | 807.25 | 824.02 | +2.08% | 10/64 |

[Latency, graph coverage, and output checks](results/fp8_verify_tactics/REPORT.md#paired-serving-comparison)

#### Jetson Thor · Qwen3.8-27B

Single-request SGLang 0.5.23 runs on arm64 with FlashInfer. Each row gives
code / math / chat output tok/s. NVFP4 used a 512-token output limit and
BF16 used a 384-token limit.

![Jetson Thor throughput for stock SGLang AR, stock DFlash2 and the add-on, grouped by BF16 or NVFP4 and prompt family.](docs/figures/jetson-thor-27b-throughput.svg)

| Target precision | Stock SGLang AR | Stock SGLang + DFlash2 chain 16 | SGLang + ax-engine-sglang tree 32 |
|---|---:|---:|---:|
| NVFP4 | 13.4 / 13.5 / 13.5 | 93.0 / 73.5 / 30.6 | **106.1 / 92.1 / 43.1** |
| BF16 | 4.6 / 4.6 / 4.6 | 43.1 / 29.6 / 12.5 | **47.5 / 37.6 / 18.3** |

[Thor workload details and launch settings](docs/performance.md#jetson-thor--qwen38-27b)

##### Thor-01 · FP8 verification projections

The fastest bit-exact tactics reduced 64-row projection time by **4.49%**
for GDN QKVZ and **5.88%** for attention QKV versus default cuBLAS. The
test timed individual projections on synthetic CUDA inputs.

![Thor-01 synthetic FP8 projection time reduction versus default cuBLAS at 48, 64, 96, and 128 rows. The largest measured reduction was 5.88% for 64-row attention QKV.](docs/figures/thor-01-fp8-projection-time.svg)

| Verification rows | GDN QKVZ time reduction | Attention QKV time reduction |
|---:|---:|---:|
| 48 | 4.65% | 4.55% |
| 64 | **4.49%** | **5.88%** |
| 96 | 0.62% | 1.70% |
| 128 | 0.84% | 1.07% |

[Before and after timings](docs/performance.md#fp8-verification-projections) · [Benchmark report](results/fp8_verify_tactics/REPORT.md)

### Qwen3.8-Flash-Next

#### RTX PRO 6000 · Qwen3.8-Flash-Next NVFP4

One GPU served Flash Next with pinned-host FP8 PLE and 128 greedy output
tokens per request. Values pool two passes and show code / math / chat
output tok/s.

![RTX PRO 6000 Flash Next NVFP4 throughput for stock AR, stock MTP3, AX MTP3, and AX MTP7 at one and eight clients.](docs/figures/rtx-pro-6000-flash-next-nvfp4-throughput.svg)

| Clients | Stock AR | Stock MTP3 | AX MTP3 | AX MTP7 |
|---:|---:|---:|---:|---:|
| 1 | 83.1 / 83.8 / 83.7 | 183.5 / 164.1 / 134.0 | 182.1 / 164.8 / **135.0** | **218.9** / **174.0** / 114.7 |
| 8 | 414.1 / 421.5 / 399.3 | 679.0 / **582.6** / **455.4** | 594.2 / 572.4 / 405.5 | **728.0** / 542.7 / 356.5 |

At eight clients, AX MTP7 led code at **728.0 tok/s**, and stock MTP3
led chat at **455.4 tok/s**. The first AX MTP3 chat pass was slower than
its repeat (363.7 vs. 458.2 tok/s).
[RTX PRO 6000 Flash Next retest](results/blackwell_nvfp4_serving_retest/REPORT.md#rtx-pro-6000-qwen38-flash-next-nvfp4)

#### B200 · Qwen3.8-Flash-Next NVFP4

The B200 served the same prompts with the same 128-token output budget.
Values are code / math / chat output tok/s pooled over two passes.

![B200 Qwen3.8-Flash-Next NVFP4 serving retest at one and eight clients, with stock AR, stock MTP3, AX MTP3, and AX MTP7.](docs/figures/b200-flash-next-nvfp4-serving-retest.svg)

| Clients | Stock AR | Stock MTP3 | AX MTP3 | AX MTP7 |
|---:|---:|---:|---:|---:|
| 1 | 136.9 / 136.4 / 130.6 | 304.9 / 273.3 / **223.4** | 302.8 / 270.9 / 213.3 | **389.7** / **297.1** / 208.9 |
| 8 | 614.7 / 717.4 / 594.2 | 720.7 / 947.0 / 684.5 | 745.1 / 976.8 / **715.4** | **1,126.9** / **978.8** / 607.5 |

At eight clients, AX MTP7 led code and AX MTP3 led chat.
[B200 setup and per-pass results](docs/performance.md#b200-flash-next-serving-retest)

#### Two B300 GPUs · Qwen3.8-Flash-Next BF16

SGLang 0.5.23 ran the BF16 checkpoint with TP2 and an on-GPU BF16 KV cache.
The table and chart show end-to-end output tok/s from the second pass, with
24 greedy prompts per family and a 256-token output limit.

![Qwen3.8-Flash-Next BF16 on two B300 GPUs. Stock AR, stock MTP3, and AX Engine MTP7 throughput at concurrency 1, 4, and 16.](docs/figures/b300-flash-next-bf16-throughput.svg)

| Workload | Concurrency | Stock SGLang AR | Stock SGLang MTP3 | AX Engine MTP7 |
|---|---:|---:|---:|---:|
| Code | 1 | 184.9 | 306.6 | **390.0** |
| Code | 4 | 389.6 | 520.7 | **615.4** |
| Code | 16 | 794.5 | 1,199.7 | **1,649.8** |
| Math | 1 | 206.8 | 353.9 | **409.9** |
| Math | 4 | 550.4 | 649.4 | **735.2** |
| Math | 16 | 1,258.0 | 1,606.1 | **1,708.0** |
| Chat | 1 | 207.2 | **294.4** | 268.0 |
| Chat | 4 | 534.1 | **584.5** | 521.1 |
| Chat | 16 | 1,069.6 | **1,272.6** | 1,169.6 |

MTP7 beat stock MTP3 by **18–38% on code** and **6–16% on math**.
Stock MTP3 led on chat. Both MTP routes matched AR on six of seven greedy
references. AR and MTP7 passed all 28 sampled distribution checks.
[Full B300 benchmark and launch settings](results/b300-flash-next-bf16/REPORT.md)

#### Qwen3.8-Flash-Next NVFP4 · MTP

These single-request results use SGLang 0.5.23. The B200 figures come from
the second benchmark pass with a 256-token output limit. The Thor figures use
two arm64 hosts with pipeline parallelism over 2.5 GbE. Each cell gives
code / math / chat output tok/s.

![Flash Next on B200. MTP7 with the QSA fix leads on code and math. Stock MTP3 leads on chat.](docs/figures/b200-flash-next-mtp-throughput.svg)

![Flash Next across two Jetson Thor hosts. MTP7 with the QSA fix leads on code and math. MTP3 with the fix leads on chat.](docs/figures/jetson-thor-flash-next-mtp-throughput.svg)

| Platform | Stock SGLang AR | Stock SGLang MTP3 | MTP7 + ax-engine-sglang QSA fix |
|---|---:|---:|---:|
| One B200 | 204.9 / 212.5 / 211.7 | 435.2 / 432.3 / 334.4 | **547.9 / 487.7** / 296.8 |
| Two Jetson Thor hosts | 18.3 / 19.0 / 19.2 | 32.0 / 32.4 / 24.8 | **43.1 / 37.5** / 22.8 |

On both platforms, MTP7 with the QSA fix led on code and math. Stock MTP3
was faster than MTP7 on chat. The Thor study also measured MTP3 with the QSA
fix at **25.7 tok/s** on chat.
[Flash Next benchmark conditions and all routes](docs/performance.md#qwen38-flash-next-nvfp4--mtp)

## Documentation and repository layout

| Path | Use |
|---|---|
| [docs/](docs/) | Architecture, supported routes and validation records |
| [docs/performance.md](docs/performance.md) | Stock AR, stock speculation and add-on comparisons |
| [ax-engine-sglang/](ax-engine-sglang/) | Installable add-on and operational reference |
| [benchmarks/sglang/](benchmarks/sglang/) | Serving benchmarks and correctness checks |
| [benchmarks/baselines/](benchmarks/baselines/) | Shared prompts, speed and quality evaluation, and the vLLM baseline |
| [contracts/](contracts/) | Pinned benchmark evidence and speculative acceptance contracts |
| [results/](results/) | Benchmark reports |

This project grew out of Next-Concept Model experiments before moving to
target-verified speculative decoding. [Research history](docs/results-history.md)

## License

Apache License, Version 2.0. [License text](LICENSE) · [Third-party notices](ax-engine-sglang/NOTICE)
