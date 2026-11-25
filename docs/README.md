# AX Engine for SGLang documentation

AX Engine for SGLang is AX Engine's CUDA edition for Linux amd64 and arm64
hosts with NVIDIA GPUs. Install the `ax-engine-sglang` Python package and run
`ax-engine-sglang serve` to start SGLang with the `ax-engine` plugin enabled.
[Quick start](../README.md#quick-start) · [How it works and why](design.md) · [Benchmarks](performance.md) · [Configuration reference](../ax-engine-sglang/README.md)

## Installation and release status

AX Engine for SGLang 0.5.1 supports Python 3.11 and 3.12 and pins SGLang
0.5.23. For a fresh install from wheels, use glibc 2.34 or later and a CUDA
13-compatible NVIDIA driver. [SGLang 0.5.23 wheel files](https://pypi.org/project/sglang/0.5.23/#files)

| Environment | Install command |
|---|---|
| Fresh virtual environment | `python -m pip install ./ax_engine_sglang-0.5.1-py3-none-any.whl` after downloading the [release wheel](https://github.com/defai-digital/ax-engine-sglang/releases/download/v0.5.1/ax_engine_sglang-0.5.1-py3-none-any.whl) |
| Fresh virtual environment, from the repository root | `python -m pip install ./ax-engine-sglang` |
| Existing SGLang 0.5.23 stack, including the pinned container | `python -m pip install --no-deps ./ax-engine-sglang` |
| Existing SGLang 0.5.23 stack with the 0.5.1 wheel | `python -m pip install --no-deps ./ax_engine_sglang-0.5.1-py3-none-any.whl` |

Run `ax-engine-sglang --version` to verify the install. `serve` forwards
SGLang server flags, enables AX and reports a startup error if AX fails to load.
[Launch defaults and checkpoint validation](../ax-engine-sglang/README.md#launch-defaults)

### Publishing a release

Push a version tag to the public `defai-digital/ax-engine-sglang` repository
to build and check the distributions. The workflow checks the tag against
the installed wheel version. Attach the wheel and source distribution to
the GitHub release. [Release workflow](../.github/workflows/pypi.yml) · [GitHub releases](https://github.com/defai-digital/ax-engine-sglang/releases)

## Runtime boundaries

| Component | Responsibility |
|---|---|
| SGLang | Server and client-facing API, model loading, request batching, KV cache allocation and CUDA execution |
| AX Engine for SGLang | SGLang launch defaults, tree and chain speculative verification, GDN state handling and the Flash Next QSA pending-ring fix |
| AX Engine for Apple Silicon | The MLX inference runtime and its AXQ model packs |
| AX Serving | Fleet orchestration and NVIDIA/CUDA worker management |

This edition ports the acceleration approach into SGLang's execution
model. Both editions use target-verified speculative decoding and pin the
same speculative commit rules and benchmark record schema. The CUDA edition
uses SGLang-compatible target and draft checkpoints and executes SGLang's
CUDA path. The Apple Silicon edition uses its own MLX runtime and model
formats. [Shared contracts](../contracts/) · [AX Engine](https://github.com/defai-digital/ax-engine) · [AX Serving](https://github.com/defai-digital/ax-serving)

## Platforms

The supported deployment scope is Linux with NVIDIA CUDA on amd64
(`x86_64`) or arm64 (`aarch64`). The package depends on SGLang 0.5.23.
The plugin also accepts 0.5.22 when its source anchors match. On 0.5.22,
supply the launch settings explicitly. `AX_AUTO_DEFAULTS` fills them on
0.5.23.

| Architecture | Recorded deployment | Evidence |
|---|---|---|
| Linux amd64 | NVIDIA B200 (SM100) | [27B and Flash Next serving retest](../results/blackwell_nvfp4_serving_retest/REPORT.md) · [Dense NVFP4 verification](reports/2026-11-18-b200-27b-nvfp4.md) |
| Linux amd64 | NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120) | [27B NVFP4 serving](reports/2026-11-19-rtx-27b-nvfp4.md) · [Speed sweep and launch commands](../results/rtx_27b_nvfp4_speed/REPORT.md) |
| Linux amd64 | NVIDIA GeForce RTX 5090 (SM120) | [27B NVFP4 throughput and launch settings](../results/rtx_5090_27b_nvfp4/REPORT.md) |
| Linux arm64 | NVIDIA Jetson Thor (SM110) | [Container, aarch64 kernel checks and model measurements](reports/2026-11-19-jetson-thor.md) · [Single-host Flash Next](reports/2026-11-21-flash-next-single-thor.md) · [TensorRT Edge-LLM comparison](../results/edgellm_thor/REPORT.md) |

Each report records its exact model, precision, CUDA stack and workload.
The Flash Next arm64 reports cover one Thor with the PLE table on NVMe
and two Thors with pipeline parallelism.

## Model routes

SGLang owns standard serving on Linux. Stock AR, MTP and built-in DFlash
are available according to SGLang's model and backend support. The add-on
adds the verification and attention paths listed here. Each validation
record applies to its checkpoint, GPU and serving configuration.

| Model family | Measured target precision | Add-on route | Evidence |
|---|---|---|---|
| Qwen3.5-9B (Gated DeltaNet) | BF16 | DFlash v1 tree on eligible single requests. The tree launcher takes a matching drafter and n-gram table | [Jetson Thor 9B benchmark](reports/2026-11-19-jetson-thor.md#2-qwen35-9b-bf16-dflash-drafter-z-labqwen35-9b-dflash) |
| Qwen3.8-27B dense (Gated DeltaNet) | BF16 and NVFP4 | DFlash2 tree for eligible single requests and chain verification for grouped requests. The current default tree has 32 nodes. Accepted-prefix GDN replay is an experimental chain option | [Dense 27B performance](performance.md) |
| Qwen3.8-Flash-Next (Qwen4-Exp) | NVFP4 | SGLang MTP with the QSA pending-ring fix. DFlash2 stays unavailable: this target does not expose the auxiliary hidden states the drafter reads | [NVFP4 serving retest](../results/blackwell_nvfp4_serving_retest/REPORT.md) · [Single Thor](reports/2026-11-21-flash-next-single-thor.md) |

### NVFP4 deployment

Start with a checkpoint quantized for SGLang. The measured dense target is
`RadixArk/Qwen3.8-27B-NVFP4` with the BF16 drafter
`incoai/Qwen3.8-27B-DFlash2`. The target used FlashInfer CUTLASS NVFP4 GEMM
and an FP8 E4M3 KV cache. Choose the GEMM and attention backend from the
report for your GPU. Flash Next uses its own NVFP4 MoE and PLE settings.
[Current serving retest](../results/blackwell_nvfp4_serving_retest/REPORT.md) · [Earlier RTX sweep commands](../results/rtx_27b_nvfp4_speed/REPORT.md) · [Flash Next launch details](../ax-engine-sglang/README.md#maintenance-and-validation)

The target verifies each proposal on the configured numerical path. The
default speed profile uses batched verification. `AX_QUALITY_PROFILE=verified`
uses matched AR and verification settings. On RTX PRO 6000, AR and the
width-eight chain matched serialized AR at 128 tokens and at 1,024 tokens.
The 16-node tree matched at 128 tokens and diverged on two of ten prompts
at 1,024 tokens, so the profile refuses it.
[Verified profile](../ax-engine-sglang/README.md#launch-check-quality-profile-and-adaptive-chain-width) · [Hot-path validation](../results/rtx_pro_6000_hotpath_validation/REPORT.md)

## Validate a deployment

- `ax-engine-sglang doctor <model> <speculative flags>` applies the launch
  rules to a planned launch without loading weights. It reports whether the
  launch is supported, which settings it would derive, and whether that GPU,
  target, and route have an exact-output measurement.
- `AX_QUALITY_PROFILE=verified` starts only AR, or the width-eight DFlash2
  chain, on a qualified GPU. On RTX PRO 6000 those routes matched serialized
  AR at 128 and 1,024 tokens. The profile refuses the tree route.
  [Gate report](reports/2026-11-21-verified-profile-gate.md)
- [Serving performance](performance.md) is the current comparison of stock
  AR, stock speculation, and the add-on. The serving retests are the tables
  to read first. Older campaigns are labeled by output length and slot count.
- [SGLang benchmark guide](../benchmarks/sglang/) covers baseline launches,
  mixed requests, cancellation, greedy token identity, and sampled distributions.
- [Regression runner](../scripts/check_model_regression.py) compares
  servers on the same model and inputs.
- [Benchmark evidence schema](../contracts/benchmark-result-v1.schema.json)
  records hardware, checkpoint, workload, and throughput.
- [Results history](results-history.md) records the research through
  2026-11-17 and points at the later serving campaigns. [Original NCM plan](plan.md)

### Installed-wheel smoke on Jetson Thor (v0.2.1)

The current release is 0.5.1. The check below used the 0.2.1 candidate
wheel. It is the recorded proof that an installed wheel, with no source
checkout on the import path, can serve the dense 27B tree inside the pinned
container. It is not a throughput result for 0.5.1.

On 2026-11-20, the 0.2.1 candidate wheel was installed with `--no-deps` in
`lmsysorg/sglang:v0.5.23-cu130` on one Jetson Thor (Linux arm64, SM110).
The server started from `/tmp` with no source checkout on its import path.
The test used these cached checkpoint revisions.

| Role | Checkpoint | Revision |
|---|---|---|
| Target | `RadixArk/Qwen3.8-27B-NVFP4` | `319f741cce68d7914884900c138a1fbb70a42f30` |
| Drafter | `incoai/Qwen3.8-27B-DFlash2` | `015e795645c74b1a0eeef3b570031fb62e769bc5` |

Inside the pinned container, set `TARGET` and `DRAFTER` to the corresponding
local snapshot directories. These were the launch settings for the shared
GPU, with one active request, prefix cache disabled and CUDA graphs enabled.

```bash
unset SGLANG_PLUGINS
AX_ROUND_DUMP=/tmp/ax-wheel-rounds.jsonl ax-engine-sglang serve "$TARGET" \
  --speculative-algorithm FC1_DFLASH_TREE \
  --speculative-draft-model-path "$DRAFTER" \
  --speculative-num-draft-tokens 16 \
  --attention-backend flashinfer --disable-flashinfer-autotune --disable-radix-cache \
  --mem-fraction-static 0.75 --context-length 1024 --chunked-prefill-size 512 \
  --max-total-tokens 1024 --host 127.0.0.1 --port 30000
```

After warmup, send the greedy request from a second shell in the container.

```bash
curl --fail --max-time 60 http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"rid":"ax-wheel-021-verified-request","text":"Write a short Python function that returns the first n Fibonacci numbers.\n","sampling_params":{"temperature":0,"max_new_tokens":32,"ignore_eos":true}}'
```

The launch derived `AX_GDN_HEADS=48`, `AX_LATTICE=1` and
`FC1_DRAFT_BLOCK=16`. The request returned HTTP 200 and 32 completion tokens.
The trace for that request ID recorded six tree rounds. A second check with
`AX_GDN_HEADS=1` rejected the mismatched setting and reported the expected
value of 48 before loading model weights.

The tested artifact was `ax_engine_sglang-0.2.1-py3-none-any.whl` with SHA-256
`67e8765b99d949f0cfc1c51825c6cc73bce376795f9910831bac829e53ab7d97`.
