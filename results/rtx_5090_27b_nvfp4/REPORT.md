# GeForce RTX 5090 · Qwen3.8-27B NVFP4

On one RTX 5090, AX Engine delivered **135.18 output tok/s** at concurrency
one. That is **2.35×** stock AR and **8.1%** faster than stock DFlash2.
At concurrency two, AX Engine delivered **181.36 tok/s**, **1.78×** stock
AR and within **0.6%** of stock DFlash2.

| Concurrency | Route | Pass 1 tok/s | Pass 2 tok/s | Pooled tok/s | Median request (s) | Exact AR matches in passes 1 and 2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Stock AR | 57.61 | 57.67 | 57.64 | 2.148 | Reference |
| 1 | Stock DFlash2, block 8 | 125.59 | 124.44 | 125.01 | 0.972 | 0/10, 0/10 |
| 1 | AX Engine, tree 32 | 135.05 | 135.32 | **135.18** | 0.832 | 0/10, 0/10 |
| 2 | Stock AR | 101.48 | 102.08 | 101.77 | 2.411 | Reference |
| 2 | Stock DFlash2, block 8 | 182.73 | 182.04 | **182.38** | 1.180 | 5/40, 4/40 |
| 2 | AX Engine, tree 32 with chain width 8 | 180.42 | 182.32 | 181.36 | 1.265 | 4/40, 4/40 |

Verification parity was disabled. Exact matches compare all 128 token IDs
against the same request in the corresponding packed AR pass.

## Measurement conditions

- RunPod, November 20, 2026 UTC. One NVIDIA GeForce RTX 5090 (SM120,
  32,607 MiB, 575 W power limit), Ubuntu 24.04 x86_64, NVIDIA driver
  595.112.07 and CUDA toolkit 13.2. SGLang 0.5.23, PyTorch 2.14.1+cu130,
  FlashInfer 0.7.2 and ax-engine-sglang 0.3.0 at `c9dc13c80418`.
- RadixArk NVFP4 target at `319f741cce68d7914884900c138a1fbb70a42f30` and
  Inco AI BF16 drafter at `015e795645c74b1a0eeef3b570031fb62e769bc5`. All
  weight files matched their pinned SHA-256 hashes.
  [Target revision](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4/tree/319f741cce68d7914884900c138a1fbb70a42f30) · [Drafter revision](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/015e795645c74b1a0eeef3b570031fb62e769bc5)
- FlashInfer attention, FlashInfer CUTLASS NVFP4 GEMM, two GPU request
  slots, 4,096 context limit, 4,096 KV tokens and a 0.94 static memory
  fraction. All lanes used 512-token prefill chunks, 64 MiB FlashInfer
  workspaces and seed 42. The target KV cache was FP8 E4M3 and the
  draft KV cache was BF16.
  Radix prefix caching, FlashInfer autotuning and prefill graphs were
  disabled. Decode graphs were enabled.
- Each C1 pass used the same ten frozen inputs, spanning 23–2,266 input
  tokens. Each C2 pass repeated that input list four times. All measured
  requests generated exactly 128 tokens with `temperature=0` and
  `ignore_eos=true`. The runner checked the finish reason and the presence
  of verification rounds on speculative lanes.
- The runner replaced SGLang's startup warm-up with the longest input and
  a concurrent two-request burst, both with 32 output tokens per request.
  Timed passes ran C1, C2, C2, C1 per lane. Pooled throughput divides total
  output tokens by total timed wall seconds,
  including prefill and loopback HTTP. Model loading, graph capture and
  warm-ups are outside the timed intervals.
- The eight-slot stock DFlash2 launch at a 0.90 static memory fraction
  failed during KV budget allocation after loading the target and drafter.
  The two-slot AX launch also exhausted VRAM when allocating the default
  384 MiB FlashInfer workspace for its chain draft graph. With 128 MiB
  workspaces it captured the graphs but exhausted VRAM on the longest
  prefill. A subsequent launch with chunked prefill hit VRAM limits in
  SGLang's multimodal startup warm-up. The measured routes use the
  shared text-only workload and launch settings above.

## Reproduce the comparison

Use an activated Python 3.12 environment with SGLang 0.5.23, the v0.4.0
plugin and Ninja on `PATH`. Set `TARGET` and `DRAFT` to the local pinned
checkpoint directories. The runner serves each lane in turn and writes
launch arguments, server logs, timings and output token IDs to the output
directory. The table above records the original benchmark checkout.
[Installation](../../docs/README.md#installation-and-release-status)

Download `parity-expanded-ids.json` from the public v0.4.0 release. Its SHA-256
is `e2b5edfeb7a18b4b01c2a67501a6659fe06eeb3ff2f98e95ef3cd957e17a7fe0`.
[Frozen input token IDs](https://github.com/defai-digital/ax-engine-sglang/releases/download/v0.4.0/parity-expanded-ids.json)

```bash
python benchmarks/sglang/bench_27b_nvfp4.py \
  --target "$TARGET" --draft "$DRAFT" --input-ids parity-expanded-ids.json \
  --out results/rtx_5090_27b_nvfp4/c2 --slots 2 --kv-tokens 4096 \
  --mem-fraction-static 0.94 --lanes tree32 stock8 ar
```

## Accepted-prefix GDN replay follow-up

The experimental checkout at `c6d8e91` measured native width-eight DFlash2
with stock GDN snapshots and with AX replay. Both routes used the pinned
checkpoints and frozen inputs above, with FP32 GDN state, Triton verify,
64 MiB FlashInfer workspaces, and the same 128-token greedy output length.
The next table pools four timing passes per route from two runs with the
lane order reversed. Each C1 pass used 10 requests and each C2 pass used 40.

| Concurrency | Stock DFlash2 tok/s | AX GDN replay tok/s | Exact matches vs stock |
|---:|---:|---:|---:|
| 1 | 123.21 | 124.39 | 40/40 |
| 2 | 178.67 | 181.19 | 160/160 |

At two slots, stock allocated 3.38 GiB of speculative SSM snapshots. AX
allocated about 0.017 GiB of replay buffers and zero SSM snapshots. At
eight slots, AX delivered 377.69 and 402.35 tok/s across two runs of 40
requests at concurrency eight, or 389.63 tok/s pooled. Its replay buffers
occupied about 0.055 GiB. Stock exhausted the GPU memory budget during KV
cache sizing with the same eight-slot settings.

An eight-slot stress run processed 48 requests with 12 clients and four
observed cancellations. The GPU pools returned to their idle usage and
all six serialized references matched after the stress run. The same
follow-up also measured fused tree 16 at 123.88 output tok/s and fused tree
32 at 135.95 output tok/s for one concurrent request.

Use the experimental checkout to run `benchmarks/sglang/bench_27b_nvfp4.py`
with `--lanes stock8 replay8` at two slots. For the eight-slot run, set
`--slots 8`. Use the same target, draft and input arguments as above.

## FP8 parity retest

The v0.4.0 FP8 projection differed from singleton decode in 23 of 45
shape and batch-size combinations on the RTX 5090. The updated parity
profile uses SGLang's Triton FP8 GEMM with a fixed 64 × 128 × 256 tile for
per-tensor scales in prefill, decode and target verification.

The chain-only launcher also sets `speculative_eagle_topk=1` during CUDA
graph capture. A graph captured with tree metadata and replayed in chain
mode produced different sequences from AR before this fix.

The serving retest used the same Qwen3.8-27B target, DFlash2 drafter and
ten frozen prompts. Every request generated 128 greedy tokens. Each lane
ran 10, 40, 40 and 10 requests at concurrency 1, 2, 2 and 1. Every sequence
was compared to that prompt's result from the first serialized AR pass.

| Parity lane | C1 output tok/s | C1 exact matches | C2 output tok/s | C2 exact matches |
|---|---:|---:|---:|---:|
| AR | 53.90 | 20/20 | 103.57 | 80/80 |
| DFlash2 tree 16 | 123.31 | 20/20 | 184.71 | 80/80 |
| DFlash2 chain 8 | 105.90 | 20/20 | 180.30 | 80/80 |

The GPU projection suite compared five Qwen projection shapes against
singleton calls at batch sizes from 2 to 512. It also passed a CUDA graph
replay check with prequantized FP8 activations and bias. The recurrent GDN
suite passed 32 chain and tree cases.

Prefill had a 4,096-token chunk limit, large enough for the prompts in
each batch. An exploratory run with a 512-token limit changed the first
output token of the 2,266-token prompt on all eight concurrent AR repeats.
The retest used two slots, a 4,096-token KV budget and 64 MiB FlashInfer
workspaces. Prefix caching, FlashInfer autotuning and prefill graphs were
off. Decode and verify graphs were enabled.

Install the plugin from the checkout with the parity fix. With the same
checkpoint paths and input file as above, run:

```bash
python -m pip install --no-deps ./ax-engine-sglang
python benchmarks/sglang/bench_27b_nvfp4.py \
  --target "$TARGET" --draft "$DRAFT" --input-ids parity-expanded-ids.json \
  --out results/rtx_5090_27b_nvfp4/parity --verify-parity
```
