# AX Engine for SGLang documentation

AX Engine for SGLang is AX Engine's CUDA edition for Linux amd64 and arm64
hosts with NVIDIA GPUs. Install the `ax-engine-sglang` Python package and run
`ax-engine-sglang serve` to start SGLang with the `ax-engine` plugin enabled.
[Quick start](../README.md#quick-start) · [How it works and why](design.md) · [Benchmarks](performance.md) · [Configuration reference](../ax-engine-sglang/README.md)

## Installation and release status

AX Engine for SGLang 0.3.0 supports Python 3.11 and 3.12 and pins SGLang
0.5.23. For a fresh install from wheels, use glibc 2.34 or later and a CUDA
13-compatible NVIDIA driver. [SGLang 0.5.23 wheel files](https://pypi.org/project/sglang/0.5.23/#files)

| Environment | Install command |
|---|---|
| Fresh virtual environment | `python -m pip install ax-engine-sglang==0.3.0` |
| Fresh virtual environment, from the repository root | `python -m pip install ./ax-engine-sglang` |
| Existing SGLang 0.5.23 stack, including the pinned container | `python -m pip install --no-deps ./ax-engine-sglang` |
| Existing SGLang 0.5.23 stack with the 0.3.0 wheel | `python -m pip install --no-deps ./ax_engine_sglang-0.3.0-py3-none-any.whl` |

Run `ax-engine-sglang --version` to verify the install. `serve` forwards
SGLang server flags, enables AX and reports a startup error if AX fails to load.
[Launch defaults and checkpoint validation](../ax-engine-sglang/README.md#launch-defaults)

### Publishing a release

Build and check distributions by pushing a version tag in the public
`defai-digital/ax-engine-sglang` repository. Run `pypi.yml` manually on that
tag with `publish=true` to upload through the `pypi` GitHub environment and
PyPI Trusted Publishing. The workflow checks the tag against the version
reported by the installed wheel. [Release workflow](../.github/workflows/pypi.yml) · [PyPI project](https://pypi.org/project/ax-engine-sglang/)

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
(`x86_64`) or arm64 (`aarch64`). SGLang 0.5.23 is the reference release.
Historical SGLang 0.5.22 tree runs used explicitly supplied launch settings.

| Architecture | Recorded deployment | Evidence |
|---|---|---|
| Linux amd64 | Two NVIDIA B300 GPUs | [Flash Next BF16 serving](../results/b300-flash-next-bf16/REPORT.md) |
| Linux amd64 | NVIDIA B200 (SM100) | [Dense NVFP4 verification](reports/2026-11-18-b200-27b-nvfp4.md) · [Flash Next release checks](reports/2026-11-18-b200-v0.1.1.md) |
| Linux amd64 | NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120) | [27B NVFP4 serving](reports/2026-11-19-rtx-27b-nvfp4.md) · [Speed sweep and launch commands](../results/rtx_27b_nvfp4_speed/REPORT.md) |
| Linux arm64 | NVIDIA Jetson Thor (SM110) | [Container, aarch64 kernel checks and model measurements](reports/2026-11-19-jetson-thor.md) |

Each report records its exact model, precision, CUDA stack and workload.
Use the Thor report for the measured container and memory settings. The
Flash Next measurements on arm64 used two Thor hosts.

## Model routes

SGLang owns standard serving on Linux. Stock AR, MTP and built-in DFlash
are available according to SGLang's model and backend support. The add-on
adds the verification and attention paths listed here. Each validation
record applies to its checkpoint, GPU and serving configuration.

| Model family | Measured target precision | Add-on route | Evidence |
|---|---|---|---|
| Qwen3.5-9B (Gated DeltaNet) | BF16 | DFlash v1 tree on eligible single requests. The tree launcher takes a matching drafter and n-gram table | [Jetson Thor 9B benchmark](reports/2026-11-19-jetson-thor.md#2-qwen35-9b-bf16-dflash-drafter-z-labqwen35-9b-dflash) |
| Qwen3.8-27B dense (Gated DeltaNet) | BF16 and NVFP4 | DFlash2 tree for eligible single requests and chain verification for grouped requests | [Dense 27B performance](performance.md) |
| Qwen3.8-Flash-Next (Qwen4-Exp) | NVFP4 | SGLang MTP with the QSA pending-ring fix. DFlash2 tree requires hidden states and a PLE/QSA verification path that are currently unavailable | [Flash Next validation](reports/2026-11-18-b200-v0.1.1.md) |

### NVFP4 deployment

Start with a checkpoint quantized for SGLang. In the RTX dense 27B sweep,
the target was `RadixArk/Qwen3.8-27B-NVFP4` and the DFlash2 drafter was
BF16. The target used FlashInfer CUTLASS NVFP4 GEMM and FP8 E4M3 KV cache.
Choose the GEMM and attention backend from the report for your GPU and
checkpoint. Flash Next uses its own NVFP4 MoE and PLE settings.
[Dense 27B benchmark and commands](../results/rtx_27b_nvfp4_speed/REPORT.md) · [Flash Next launch details](../ax-engine-sglang/README.md#maintenance-and-validation)

The target verifies each speculative proposal on the configured numerical
path. The RTX speed profile uses batched verification. The opt-in parity
profile uses matched AR and verification settings, so its speculative
lanes reproduce its own AR reference. [Speed and parity results](performance.md#rtx-speed-and-parity-profiles)

## Validate a deployment

- [Serving performance](performance.md) compares stock AR, stock speculation and the add-on under the recorded conditions.
- [SGLang benchmark guide](../benchmarks/sglang/) covers baseline launches,
  mixed requests, cancellation, greedy token identity and sampled distributions.
- [Regression runner](../scripts/check_model_regression.py) compares
  servers with the same model and inputs. Use the RTX parity report for
  the exact settings of the NVFP4 identity checks.
- [Benchmark evidence schema](../contracts/benchmark-result-v1.schema.json)
  records hardware, checkpoint, workload and throughput details.
- [Results history](results-history.md) and the [original NCM plan](plan.md)
  record the research that preceded the SGLang add-on.

### 0.2.1 installed-wheel smoke on Jetson Thor

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
