# AX Engine for SGLang

**AX Engine's Linux CUDA edition for NVIDIA GPUs on amd64 and arm64.**

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

## Quick start

Use Python 3.11 or 3.12 on Linux with an NVIDIA GPU and a CUDA
13-compatible driver. SGLang 0.5.23 is the reference runtime. Install the
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
| Qwen3.8-Flash-Next | NVFP4 | SGLang MTP plus a QSA pending-ring fix. This target does not expose the auxiliary states DFlash2 reads, so the tree is unavailable |

A route is faster when the tokens accepted per target pass save more time
than drafting and verification add.
[Model routes](docs/README.md#model-routes) · [Design](docs/design.md)

## Platforms

| Host | Recorded NVIDIA GPUs |
|---|---|
| Linux amd64 | B200, RTX PRO 6000 Blackwell Server Edition, GeForce RTX 5090 |
| Linux arm64 | Jetson Thor |

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
for these rates. Full tables, the TensorRT-LLM comparison, and the kernel
ablations are in [Serving performance](docs/performance.md).

### Qwen3.8-27B NVFP4

The add-on used a 32-node tree at one client and a width-eight chain when
requests were grouped. GDN replay is the experimental accepted-prefix path
on that chain. These are speed-profile runs. The B200 rows are package
0.5.1. The RTX PRO 6000 rows are package 0.4.0 on the same prompt corpus.

![B200 Qwen3.8-27B NVFP4 serving retest at one and eight clients, with stock AR, stock DFlash2, AX tree/chain, and AX GDN replay.](docs/figures/b200-27b-nvfp4-serving-retest.svg)

| GPU | Clients | Stock DFlash2 | AX tree/chain | AX GDN replay |
|---|---:|---:|---:|---:|
| B200 | 1 | 448.5 / 408.1 / 281.3 | **632.5 / 516.5 / 333.8** | 456.6 / 415.3 / 286.6 |
| B200 | 8 | 1,438.4 / 1,172.9 / 903.2 | 1,503.0 / 1,206.6 / 790.1 | **1,599.5 / 1,428.3 / 949.6** |
| RTX PRO 6000 | 1 | 229.3 / 188.5 / 125.2 | **324.7 / 270.9 / 152.2** | 235.8 / 195.2 / 128.9 |
| RTX PRO 6000 | 8 | 819.2 / 760.5 / 505.3 | 907.7 / 823.3 / 487.0 | **996.8 / 898.1 / 580.8** |

At one client the tree led on both GPUs. At eight clients, GDN replay led
stock DFlash2 by 11.2% / 21.8% / 5.1% on B200. On RTX PRO 6000 it raised
chat from 505.3 to 580.8 tok/s. BF16 on that GPU is in the same retest:
the tree led at one client, and at eight clients chat was 335.1 tok/s
against 333.7 for stock DFlash2.
[B200 per-pass results](docs/performance.md#b200-qwen38-27b) · [RTX PRO 6000](docs/performance.md#rtx-pro-6000-qwen38-27b-nvfp4)

On a 32 GB RTX 5090 the same checkpoints used two slots and a pooled
ten-prompt mix. Tree 32 reached 135.18 tok/s at one request (2.35× stock
AR, 8.1% above stock DFlash2). At two requests the speculative routes
finished within 0.6%.
[RTX 5090](docs/performance.md#geforce-rtx-5090-qwen38-27b-nvfp4)

On one Jetson Thor, tree 32 led Edge-LLM DDTree 32 on code and math and
tied it on chat, for both BF16 and NVFP4.
[Jetson Thor](docs/performance.md#jetson-thor-qwen38-27b)

### Qwen3.8-Flash-Next NVFP4

The same 128-token retest. AX MTP7 led code on every row below. Chat
stayed with a three-step MTP window.

| GPU | Clients | Stock MTP3 | AX MTP7 | Chat leader |
|---|---:|---:|---:|---|
| B200 | 1 | 304.9 / 273.3 / 223.4 | 389.7 / 297.1 / 208.9 | stock MTP3, 223.4 |
| B200 | 8 | 720.7 / 947.0 / 684.5 | 1,126.9 / 978.8 / 607.5 | AX MTP3, 715.4 |
| RTX PRO 6000 | 1 | 183.5 / 164.1 / 134.0 | 218.9 / 174.0 / 114.7 | AX MTP3, 135.0 |
| RTX PRO 6000 | 8 | 679.0 / 582.6 / 455.4 | 728.0 / 542.7 / 356.5 | stock MTP3, 455.4 |

[Flash Next tables](docs/performance.md#qwen38-flash-next-nvfp4-mtp)

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

## License

Apache License, Version 2.0. [License text](LICENSE) · [Third-party notices](ax-engine-sglang/NOTICE)
