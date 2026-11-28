<div align="center">

<img src="docs/images/ax-engine-banner.svg" alt="" width="100%">

# AX Engine for SGLang

[![CI](https://img.shields.io/github/actions/workflow/status/defai-digital/ax-engine-sglang/ci.yml?branch=main&label=CI)](https://github.com/defai-digital/ax-engine-sglang/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ax-engine-sglang?label=PyPI)](https://pypi.org/project/ax-engine-sglang/)
[![Linux](https://img.shields.io/badge/Linux-amd64%20%7C%20arm64-FCC624?logo=linux&logoColor=black)](#supported-platforms)
[![CUDA](https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia&logoColor=white)](#quick-start)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

**AX Engine's Linux CUDA edition for NVIDIA Hopper, NVIDIA Blackwell, and the NVIDIA Jetson series.**

`ax-engine-sglang` is a Python add-on for SGLang. SGLang runs the server,
schedules requests, and executes the model on CUDA. The add-on changes
speculative verification: a DFlash tree for one eligible request, chain
verification when requests are grouped, and a QSA pending-ring fix so
Flash Next multi-token prediction can verify a longer draft. The target
model still decides every emitted token.

The Apple Silicon edition runs on MLX. The editions share acceptance rules
and the benchmark record schema. Serving paths and checkpoints stay
specific to each runtime.
[Documentation](docs/README.md) · [How it works](docs/design.md) · [Benchmarks](docs/performance.md) · [AX Engine](https://github.com/defai-digital/ax-engine)

## Contents

- [Components and workflow](#components-and-workflow)
- [Quick start](#quick-start)
- [What it accelerates](#what-it-accelerates)
- [Supported platforms](#supported-platforms)
- [Measured throughput](#measured-throughput): [B200](#b200) · [H200](#h200) · [RTX PRO 6000](#rtx-pro-6000) · [RTX 5090](#rtx-5090) · [Jetson Thor](#jetson-thor) · [Exact outputs](#exact-outputs)
- [Repository](#repository)
- [Contributing](#contributing)
- [License](#license)

## Components and workflow

![What AX Engine adds to SGLang](docs/images/ax-engine-sglang-components.svg)

Callers use SGLang's HTTP API. An OpenAI-compatible client, or SGLang's own
client, is the supported caller. This edition does not ship a client SDK.
`ax-engine-sglang serve` loads the plugin into that server and forwards its
flags. SGLang keeps the scheduler, CUDA execution, and the stock
autoregressive, MTP, and DFlash routes. The plugin adds the verify shape:
a DFlash tree for one eligible greedy request, a chain when requests are
grouped, and a wider QSA ring so Flash Next MTP can check a longer draft.
The target model still decides every emitted token.
[How it works](docs/design.md)

## Quick start

Use Python 3.11 or 3.12 on Linux with a Hopper, Blackwell, or Jetson GPU
and a CUDA 13-compatible driver. SGLang 0.5.23 is the reference runtime. Install the
v0.5.1 wheel in a fresh virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install https://github.com/defai-digital/ax-engine-sglang/releases/download/v0.5.1/ax_engine_sglang-0.5.1-py3-none-any.whl
```

From a source checkout, use `python -m pip install ./ax-engine-sglang`.
If SGLang 0.5.23 is already installed, add `--no-deps`.
[Install matrix](docs/README.md#installation-and-release-status) · [Release notes](https://github.com/defai-digital/ax-engine-sglang/releases)

`serve` loads the AX plugin and forwards SGLang server flags. The target
and drafter below are the measured dense NVFP4 route on B200, RTX PRO
6000, RTX 5090, and Jetson Thor. The GPU needs memory for both checkpoints.

```bash
ax-engine-sglang serve RadixArk/Qwen3.8-27B-NVFP4 \
  --speculative-algorithm FC1_DFLASH_TREE \
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 16
```

On SGLang 0.5.23 the launcher fills the measured tree settings: 32 nodes,
a DFlash2 lattice, and fused verification. One eligible greedy request
uses that tree. Grouped requests use a DFlash2 chain.
`ax-engine-sglang doctor` checks a planned launch against the same rules
and does not load weights.
[Launch defaults](ax-engine-sglang/README.md#launch-defaults)

## What it accelerates

Stock autoregressive decoding, MTP, and DFlash stay available wherever
SGLang supports them. The add-on adds these routes.

| Model | Measured precision | Add-on route |
|---|---|---|
| Qwen3.5-9B | BF16 | DFlash v1 tree on an eligible request |
| Qwen3.8-27B dense | BF16 and NVFP4 | DFlash2 tree for one eligible request, and chain verification when requests are grouped. Accepted-prefix GDN replay is an experimental chain option. `AX_BTREE` (off by default) gives each request in a batch of 2–8 its own tree in one target pass |
| Qwen3.8-Flash-Next (Qwen4-Exp) | BF16 and NVFP4 | SGLang MTP plus a QSA pending-ring fix. This target does not expose the auxiliary states DFlash2 reads, so the tree is unavailable. BF16 needs two GPUs |

A route is faster when the tokens accepted per target pass save more time
than drafting and verification add.
[Model routes](docs/README.md#model-routes) · [Design](docs/design.md)

## Supported platforms

Support follows the CUDA compute capability. The measured board is the
run recorded in this repository. Another product on the same capability
uses the same kernels; it is not a second report. On Linux amd64 that is
NVIDIA Hopper and NVIDIA Blackwell. On Linux arm64 it is NVIDIA Jetson at
SM110, which is Jetson Thor.

Hopper at SM90 covers H100 and H200. Package 0.5.1 ran the 27B tree
and chain routes on one H200 for BF16, FP8 and NVFP4 (through Marlin
W4A16). NVFP4 on SM90 now runs on Humming when `humming-kernels` is
installed, as vLLM does. The NVFP4 verify and GDN recurrent kernels are
SM12-only and stay off there.
[H200 retest](#h200) · [Earlier H200 baselines](results/track_a_stage0_h200/REPORT.md)

| Platform | Host | Capability | Measured board |
|---|---|---|---|
| NVIDIA Hopper | Linux amd64 (x86_64) | SM90 | H200 |
| NVIDIA Blackwell | Linux amd64 (x86_64) | SM100 | B200 |
| | | SM120 | RTX PRO 6000 Blackwell Server Edition |
| | | SM120 | GeForce RTX 5090 |
| NVIDIA Jetson | Linux arm64 (aarch64) | SM110 | Jetson Thor |

The plugin also accepts SGLang 0.5.22 when its source anchors match. On
0.5.22, supply the launch settings yourself. `AX_AUTO_DEFAULTS` fills them
on 0.5.23.
[Platform records](docs/README.md#platforms)

## Measured throughput

Compare numbers inside one study. The current cross-route comparison is
the serving retest: SGLang 0.5.23, 128 greedy output tokens, eight request
slots, two streamed passes, and the same prompt corpus. Cells are code /
math / chat output tokens per second. Studies with 256, 512, or 1,024
tokens, a different slot count, or a pooled prompt mix do not substitute
for these rates. This section shows NVFP4 checkpoints only, by GPU and then
model (Flash Next before 27B). BF16 and FP8 results, full tables, the
TensorRT-LLM comparison, and the kernel ablations are in
[Serving performance](docs/performance.md).

### B200

#### Qwen3.8-Flash-Next NVFP4

The 128-token retest on one B200. AX MTP7 led code at both concurrencies.
Chat stayed with a three-step MTP window.

![B200 Qwen3.8-Flash-Next NVFP4 serving retest at one and eight clients, with stock AR, stock MTP3, AX MTP3, and AX MTP7.](docs/figures/b200-flash-next-nvfp4-serving-retest.svg)

| Clients | Stock MTP3 | AX MTP7 | Chat leader |
|---:|---:|---:|---|
| 1 | 304.9 / 273.3 / 223.4 | 389.7 / 297.1 / 208.9 | stock MTP3, 223.4 |
| 8 | 720.7 / 947.0 / 684.5 | 1,126.9 / 978.8 / 607.5 | AX MTP3, 715.4 |

[B200 Flash Next tables](docs/performance.md#b200-flash-next-serving-retest)

#### Qwen3.8-27B NVFP4

The 128-token retest on one B200 with stock vLLM on the same pod, rerun
with the current launch defaults: one DFlash prefill admitted at a time
and FlashInfer Gated-DeltaNet prefill (SGLang picks it on SM100 only for a
bf16 state). The 32-node tree led every workload at one client; at eight
clients GDN replay came within 2–7% of vLLM.

![B200 Qwen3.8-27B NVFP4 on one pod: stock SGLang DFlash2, stock vLLM DFlash2, AX tree 32 and AX GDN replay at one and eight clients.](docs/figures/b200-27b-nvfp4-prefill-graph.svg)

| Clients | Stock SGLang DFlash2 | Stock vLLM DFlash2 | AX tree 32 | AX GDN replay |
|---:|---:|---:|---:|---:|
| 1 | 692.7 / 581.7 / 370.2 | 838.4 / 695.6 / 451.6 | **945.6 / 766.5 / 459.1** | 717.1 / 645.7 / 406.2 |
| 8 | 2,271.3 / 1,976.9 / 1,329.0 | **2,660.1 / 2,851.6 / 1,819.6** | 2,634.3 / 2,551.6 / 1,538.8 | 2,613.2 / 2,656.8 / 1,707.0 |

Against stock vLLM on the same pod (code / math / chat):

| Clients | Stock SGLang DFlash2 | AX tree 32 | AX GDN replay |
|---:|---:|---:|---:|
| 1 | −17.4% / −16.4% / −18.0% | +12.8% / +10.2% / +1.7% | −14.5% / −7.2% / −10.1% |
| 8 | −14.6% / −30.7% / −27.0% | −1.0% / −10.5% / −15.4% | −1.8% / −6.8% / −6.2% |

Change from the previous run:

| Clients | AX tree 32 | AX GDN replay |
|---:|---:|---:|
| 1 | +10.1% / +4.8% / +1.6% | −0.8% / +6.0% / +4.0% |
| 8 | +8.5% / +23.0% / +14.7% | +8.1% / +17.2% / +16.5% |

Previous run: the same pod and harness before the two defaults (admission, FlashInfer GDN prefill).

FlashInfer prefill alone added 9–11% to replay at eight clients. Tokens per verify match vLLM, so the rest of the
eight-client gap is per-round cost: the replay fold, the verify conv
window writes, and RMSNorm + quantization that vLLM fuses.
[Cross-engine rerun](results/cross_engine_2026-11-28/REPORT.md#b200) ·
[GDN prefill report](results/b200_gdn_prefill/REPORT.md) ·
[Earlier B200 results](docs/performance.md#b200-qwen38-27b)

### H200

#### Qwen3.8-Flash-Next NVFP4

Only stock vLLM served it on one H200, through its FP4 Marlin fallback
with MTP3. SGLang 0.5.23 ran out of memory while repacking the 512
experts for Marlin (138 GB allocated, PLE table already on the host), so
the SGLang and AX lanes did not run.

| Clients | Stock vLLM MTP3 | Tokens per verify |
|---:|---:|---:|
| 1 | 259.9 / 247.1 / 246.1 | 3.84 / 3.42 / 2.68 |
| 8 | 846.4 / 727.1 / 710.0 | 3.83 / 3.42 / 2.61 |

The second sweep ran slower than the first by up to 30%; the cells pool
both.
[H200 Flash Next](results/cross_engine_2026-11-25/REPORT.md#h200-qwen38-flash-next-nvfp4)

#### Qwen3.8-27B NVFP4

The 128-token retest on one H200 (SM90, 141 GB), with stock vLLM on the
same pod. Hopper has no FP4 tensor cores: SGLang runs NVFP4 through Marlin
W4A16 and vLLM through humming-kernels. With `humming-kernels` installed
the launcher now runs the AX routes on Humming too, and admits one DFlash
prefill at a time. Both AX routes led every workload at eight clients, and
the tree led at one.

![H200 Qwen3.8-27B NVFP4 on one pod: stock SGLang DFlash2, stock vLLM DFlash2, AX tree 32 and AX GDN replay at one and eight clients.](docs/figures/h200-27b-nvfp4-throughput.svg)

| Clients | Stock SGLang DFlash2 (Marlin) | Stock vLLM DFlash2 | AX tree 32 | AX GDN replay |
|---:|---:|---:|---:|---:|
| 1 | 485.4 / 407.4 / 259.9 | 486.3 / 431.5 / 282.8 | **587.0 / 483.6 / 294.2** | 463.2 / 396.0 / 252.0 |
| 8 | 1,514.0 / 1,387.4 / 876.1 | 1,570.1 / 1,574.0 / 1,063.5 | **1,714.0 / 1,720.7** / 1,074.9 | 1,658.4 / 1,687.0 / **1,075.7** |

Against stock vLLM on the same pod (code / math / chat):

| Clients | Stock SGLang DFlash2 | AX tree 32 | AX GDN replay |
|---:|---:|---:|---:|
| 1 | −0.2% / −5.6% / −8.1% | +20.7% / +12.1% / +4.0% | −4.8% / −8.2% / −10.9% |
| 8 | −3.6% / −11.9% / −17.6% | +9.2% / +9.3% / +1.1% | +5.6% / +7.2% / +1.1% |

Change from the previous run:

| Clients | AX tree 32 | AX GDN replay |
|---:|---:|---:|
| 1 | −0.7% / +0.2% / −1.5% | +1.7% / +3.7% / +3.7% |
| 8 | +12.4% / +21.3% / +17.3% | +7.9% / +19.1% / +18.1% |

Previous run: the same pod and harness with Marlin and SGLang's admission rule.

With
Marlin both AX routes trailed vLLM by 10–14% on math and chat at eight
clients. AX cells pool five servers, vLLM the faster two of three (the
third ran 12–27% slower at eight clients; with it the tree leads by
+22.9 / +17.3 / +3.8%).
[H200 Humming report](results/h200_humming/REPORT.md) ·
[H200 tree sweep](results/h200_tree_sweep/REPORT.md)

### RTX PRO 6000

#### Qwen3.8-Flash-Next NVFP4

The 128-token retest on one RTX PRO 6000. AX MTP7 led code at both
concurrencies.

![RTX PRO 6000 Flash Next NVFP4 throughput for stock AR, stock MTP3, AX MTP3, and AX MTP7 at one and eight clients.](docs/figures/rtx-pro-6000-flash-next-nvfp4-throughput.svg)

| Clients | Stock MTP3 | AX MTP7 | Chat leader |
|---:|---:|---:|---|
| 1 | 183.5 / 164.1 / 134.0 | 218.9 / 174.0 / 114.7 | AX MTP3, 135.0 |
| 8 | 679.0 / 582.6 / 455.4 | 728.0 / 542.7 / 356.5 | stock MTP3, 455.4 |

[RTX PRO 6000 Flash Next tables](docs/performance.md#rtx-pro-6000-flash-next-serving-retest)

#### Qwen3.8-27B NVFP4

The 128-token retest on one RTX PRO 6000 (Workstation Edition), with stock
vLLM on the same pod. SGLang used FlashInfer cute-dsl FP4 GEMMs (the
add-on's SM12x default) and admitted one DFlash prefill at a time. The
32-node tree led every workload at one client, and GDN replay every
workload at eight.

![RTX PRO 6000 Qwen3.8-27B NVFP4 on one pod: stock SGLang DFlash2, stock vLLM DFlash2, AX tree 32 and AX GDN replay at one and eight clients.](docs/figures/rtx-pro-6000-27b-nvfp4-prefill-graph.svg)

| Clients | Stock SGLang DFlash2 | Stock vLLM DFlash2 | AX tree 32 | AX GDN replay |
|---:|---:|---:|---:|---:|
| 1 | 318.4 / 258.6 / 168.8 | 294.1 / 283.4 / 179.5 | **431.4 / 363.4 / 209.5** | 329.4 / 267.1 / 174.9 |
| 8 | 1,236.8 / 1,127.7 / 678.8 | 897.9 / 926.7 / 690.1 | 1,151.8 / 1,216.0 / 684.9 | **1,194.8 / 1,313.8 / 790.1** |

Against stock vLLM on the same pod (code / math / chat):

| Clients | Stock SGLang DFlash2 | AX tree 32 | AX GDN replay |
|---:|---:|---:|---:|
| 1 | +8.3% / −8.8% / −6.0% | +46.7% / +28.2% / +16.7% | +12.0% / −5.8% / −2.6% |
| 8 | +37.7% / +21.7% / −1.6% | +28.3% / +31.2% / −0.8% | +33.1% / +41.8% / +14.5% |

Change from the previous run:

| Clients | AX tree 32 | AX GDN replay |
|---:|---:|---:|
| 1 | −0.4% / −0.0% / −0.3% | +0.5% / −0.1% / +0.2% |
| 8 | −1.3% / +6.8% / +4.9% | −13.1% / +3.6% / +2.7% |

Previous run: the same pod and harness with SGLang's admission rule. Eight-client code varied by up to 13% between servers of one lane (replay 1,079–1,338 tok/s).

[Cross-engine rerun](results/cross_engine_2026-11-28/REPORT.md#rtx-pro-6000) ·
[Prefill-graph report](results/rtx_pro_6000_prefill_graph/REPORT.md) ·
[FP4 backend report](results/rtx_pro_6000_fp4_backend/REPORT.md) ·
[Earlier cross-engine table with TensorRT-LLM](docs/performance.md#rtx-pro-6000-qwen38-27b-nvfp4)

### RTX 5090

#### Qwen3.8-27B NVFP4

The 128-token retest on one 32 GB RTX 5090 with two request slots, with
stock vLLM on the same pod. SGLang used FlashInfer cute-dsl FP4 GEMMs, the
add-on's SM12x default. The 32-node tree led every workload at one and
two requests. TensorRT-LLM DFlash2 ran out of memory creating the target
and drafter on the 32 GB card.

![RTX 5090 Qwen3.8-27B NVFP4 on one pod: stock SGLang DFlash2, stock vLLM DFlash2, AX tree 32 and AX GDN replay at one and two requests.](docs/figures/rtx-5090-27b-nvfp4-prefill-graph.svg)

| Requests | Stock SGLang DFlash2 | Stock vLLM DFlash2 | AX tree 32 | AX GDN replay |
|---:|---:|---:|---:|---:|
| 1 | 294.9 / 237.2 / 155.1 | 245.3 / 213.6 / 173.1 | **389.4 / 318.0 / 186.4** | 317.0 / 259.4 / 160.3 |
| 2 | 378.0 / 393.3 / 234.9 | 316.2 / 345.9 / 231.2 | **418.2 / 421.8** / 243.7 | 402.9 / 405.5 / **255.7** |

Against stock vLLM on the same pod (code / math / chat):

| Requests | Stock SGLang DFlash2 | AX tree 32 | AX GDN replay |
|---:|---:|---:|---:|
| 1 | +20.2% / +11.0% / −10.4% | +58.7% / +48.9% / +7.7% | +29.2% / +21.4% / −7.4% |
| 2 | +19.5% / +13.7% / +1.6% | +32.3% / +21.9% / +5.4% | +27.4% / +17.2% / +10.6% |

vLLM's first-token median was 0.16–0.20 s
against 0.07–0.10 s for SGLang.
[Cross-engine rerun](results/cross_engine_2026-11-28/REPORT.md#rtx-5090) ·
[5090 prefill-graph report](results/rtx_5090_prefill_graph/REPORT.md) ·
[FP4 backend report](results/rtx_pro_6000_fp4_backend/REPORT.md#rtx-5090) ·
[Earlier 5090 studies](docs/performance.md#geforce-rtx-5090-qwen38-27b-nvfp4)

### Jetson Thor

#### Qwen3.8-27B NVFP4

On one Jetson Thor, tree 32 reached 106.1 / 92.1 / 43.1 tok/s. It led
Edge-LLM DDTree 32 (71.4 / 68.8 / 44.2) on code and math and was within 3%
on chat.

![Jetson Thor Qwen3.8-27B NVFP4 throughput for stock SGLang, stock vLLM, TensorRT Edge-LLM, and the add-on by prompt family.](docs/figures/jetson-thor-27b-nvfp4-throughput.svg)

[Jetson Thor](docs/performance.md#jetson-thor-qwen38-27b)

### Exact outputs

`AX_QUALITY_PROFILE=verified` admits autoregressive decoding and the
width-eight DFlash2 chain on a qualified GPU. On RTX PRO 6000 those routes
matched serialized AR at 128 and 1,024 tokens. The 16-node tree diverged
on two of ten prompts at 1,024 tokens, so the profile refuses the tree.
The speed tables above do not carry that match.
[Verified profile](docs/reports/2026-11-21-verified-profile-gate.md)

## Repository

| Path | Use |
|---|---|
| [docs/](docs/) | Install, platforms, routes, and design |
| [docs/performance.md](docs/performance.md) | Full comparisons, older studies, and ablations |
| [ax-engine-sglang/](ax-engine-sglang/) | Package, launch defaults, and environment variables |
| [benchmarks/sglang/](benchmarks/sglang/) | Serving benchmarks and correctness checks |
| [contracts/](contracts/) | Benchmark record and acceptance contracts |
| [results/](results/) | Campaign reports |

The project started as Next-Concept Model experiments and moved to
target-verified speculative decoding.
[Research history](docs/results-history.md)

## Contributing

Issues, wishlist items, reproducible benchmarks, and docs feedback are welcome.
Unsolicited code PRs for verification, kernel, or performance paths are
generally not accepted — open an issue first. See [CONTRIBUTING.md](CONTRIBUTING.md).
Report security issues privately. See [SECURITY.md](SECURITY.md).
Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Community

- Website: [automatosx.com](https://automatosx.com)
- Discord: [Join us](https://discord.gg/MECsKdA6zF)
- Email: [enquiry@defai.digital](mailto:enquiry@defai.digital)

## License

Apache License, Version 2.0. [License text](LICENSE) · [Third-party notices](ax-engine-sglang/NOTICE)
