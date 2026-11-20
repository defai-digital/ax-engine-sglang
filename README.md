# AX Engine for SGLang

**AX Engine's Linux CUDA edition for NVIDIA GPUs on amd64 and arm64.**

`ax-engine-sglang` brings AX Engine's inference acceleration approach to
SGLang as a Python add-on. SGLang runs the server, schedules requests and
executes the model on CUDA. The add-on extends speculative decoding, state
verification and model-specific attention behavior.

The Apple Silicon edition runs on MLX. The two editions share speculative
acceptance rules and benchmark evidence contracts, with serving paths and
checkpoint formats specific to each runtime.
[AX Engine](https://github.com/defai-digital/ax-engine) · [Architecture and support](docs/README.md)

## Quick start

Use Python 3.12 on Linux with an NVIDIA GPU and a CUDA 13-compatible driver.
Install a published release from PyPI in a fresh virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install ax-engine-sglang
```

To install from a checkout, run `python -m pip install ./ax-engine-sglang`
from the repository root. In an existing SGLang 0.5.23 environment, add
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

The serving code was validated with this target and drafter on Jetson Thor. A single greedy
request returned 32 tokens and the trace contained six DFlash2 tree rounds.
[Thor smoke test and flags](docs/README.md#installed-wheel-smoke-on-jetson-thor) · [Launch defaults and multi-user settings](ax-engine-sglang/README.md#launch-defaults)

## Supported platforms

| Host | NVIDIA configurations with recorded runs |
|---|---|
| Linux amd64 (x86_64) | B200 and RTX PRO 6000 |
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
| Qwen3.8-Flash-Next (Qwen4-Exp) | NVFP4 | SGLang MTP with the QSA pending-ring fix. DFlash2 tree serving remains unavailable |

For NVFP4, load an existing SGLang-compatible NVFP4 checkpoint with the
GPU kernels required by that checkpoint. Dense 27B and Flash Next were
verified on separate serving routes. [Model routes and precision support](docs/README.md#model-routes)

## Measured throughput

### B200 · Qwen3.8-27B NVFP4

Greedy runs used 64 prompts per family, a 1,024-token output limit and 64
server slots. Each rate averages two passes. Values are output tok/s in
code / math / chat order.

![B200 NVFP4 throughput for stock DFlash2 and the add-on at concurrency 1 and 16.](docs/figures/b200-27b-nvfp4-throughput.svg)

| Concurrency | Stock SGLang + DFlash2 block 8 | SGLang + ax-engine-sglang |
|---|---:|---:|
| 1 | 816 / 733 / 401 | **1,160 / 971 / 458** (tree 16) |
| 16 | 3,628 / 4,670 / 2,980 | **4,379.5 / 4,989 / 3,110** (chain 8) |

The paired dense-model run compared the two speculative routes. The B200
Flash Next chart and table include a matched stock SGLang AR lane and use
MTP as the speculative baseline. [Dense 27B benchmark](docs/performance.md#b200--qwen38-27b) · [Flash Next chart and table](docs/performance.md#qwen38-flash-next-nvfp4--mtp)

### RTX PRO 6000 · Qwen3.8-27B NVFP4

All three routes used SGLang 0.5.23, the same target checkpoint, frozen
inputs and fixed 128-token greedy outputs.
Rates are end-to-end output tok/s pooled over two passes in speed mode.

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
The separate opt-in parity profile matched its own AR reference on all ten
greedy prompts and passed all 28 sampled distribution checks in the recorded
comparison. It uses different verification numerics and carries a throughput
cost. [RTX parity and sampling results](docs/reports/2026-11-19-rtx-nvfp4-parity.md)

### Jetson Thor · Qwen3.8-27B

Single-request SGLang 0.5.23 runs on arm64 with FlashInfer. Each row gives
code / math / chat output tok/s. NVFP4 used a 512-token output limit and
BF16 used a 384-token limit.

![Jetson Thor throughput for stock SGLang AR, stock DFlash2 and the add-on, grouped by BF16 or NVFP4 and prompt family.](docs/figures/jetson-thor-27b-throughput.svg)

| Target precision | Stock SGLang AR | Stock SGLang + DFlash2 chain 16 | SGLang + ax-engine-sglang tree 32 |
|---|---:|---:|---:|
| NVFP4 | 13.4 / 13.5 / 13.5 | 93.0 / 73.5 / 30.6 | **106.1 / 92.1 / 43.1** |
| BF16 | 4.6 / 4.6 / 4.6 | 43.1 / 29.6 / 12.5 | **47.5 / 37.6 / 18.3** |

[Thor workload details and launch settings](docs/performance.md#jetson-thor--qwen38-27b)

### Qwen3.8-Flash-Next NVFP4 · MTP

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
