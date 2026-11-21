# RTX PRO 6000 DFlash2 speed with parity disabled

On Qwen3.8-27B NVFP4, the 64-node plugin tree completed the ten serialized
requests at **156.87 output tok/s**, 2.88× the throughput of packed AR.
Median request latency fell from 2.379 s to 0.818 s. In both passes, one of
ten complete 128-token outputs was identical to AR.

With eight concurrent clients and 40 requests per pass, stock DFlash2 with
block size eight averaged 489.86 tok/s and plugin tree 32 averaged
484.78 tok/s. Each led one pass. Packed AR averaged 388.29 tok/s.
The separate parity profile reproduced a matched AR reference on all ten
prompts with a 32-token output budget.
[Parity verification report](../../docs/reports/2026-11-19-rtx-nvfp4-parity.md)

## Ten prompts per pass

| Serving route | C1 output tok/s | C1 median request (s) | C8 aggregate output tok/s | Exact 128-token matches vs AR out of 10, C1 passes 1 and 2 |
|---|---:|---:|---:|---:|
| Plugin tree, nodes 64 | 156.87 | 0.818 | 539.04 | 1, 1 |
| Plugin tree, nodes 32 | 148.67 | 0.838 | 546.53 | 0, 0 |
| Plugin tree, nodes 16 | 142.31 | 0.826 | 544.72 | 0, 0 |
| Plugin chain, width 8 | 125.39 | 0.991 | 517.51 | 1, 1 |
| Stock DFlash2, block 8 | 124.50 | 1.003 | 554.08 | 1, 1 |
| Stock DFlash2, block 16 | 116.01 | 1.063 | 408.92 | 1, 1 |
| Plugin chain, width 16 | 115.26 | 1.068 | 413.31 | 1, 1 |
| Plugin chain, width 4 | 102.98 | 1.220 | 443.78 | 1, 1 |
| Stock DFlash2, block 4 | 99.75 | 1.283 | 474.79 | 1, 1 |
| Packed AR | 54.41 | 2.379 | 244.44 | 10, 10 |

## Forty requests per pass, concurrency eight

The two speculative configurations with the highest pooled C8 rates in the ten-prompt sweep
were restarted and retested against AR. Each pass cycles through the same ten prompts four
times with eight client workers. Prefix caching stays disabled on all routes.

| Serving route | Pass 1 tok/s | Pass 2 tok/s | Pooled output tok/s | Relative to AR | Median request (s) |
|---|---:|---:|---:|---:|---:|
| Stock DFlash2, block 8 | 498.42 | 481.59 | 489.86 | 1.26× | 1.767 |
| Plugin tree, nodes 32 | 487.88 | 481.73 | 484.78 | 1.25× | 1.780 |
| Packed AR | 388.01 | 388.57 | 388.29 | 1.00× | 2.653 |

## Measurement conditions

- Single NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), driver 595.112.07, SGLang 0.5.23, plugin wheel v0.1.2, PyTorch 2.14.1+cu130 and FlashInfer 0.7.2.
- Target RadixArk/Qwen3.8-27B-NVFP4 at `319f741cce68d7914884900c138a1fbb70a42f30`, BF16 DFlash2 draft at `015e795645c74b1a0eeef3b570031fb62e769bc5`.
- Every lane loaded the same target. AR and stock DFlash2 excluded the plugin from the SGLang plugin whitelist. Plugin runs enabled fused verification, compiled tree construction, and set `AX_NVFP4_VERIFY_PARITY=0`. Tree routes draft a block of 16 and use chain width eight when the scheduler groups multiple requests. Chain-only routes set `AX_TREE=0`.
- Tree 64 initially exceeded the GDN/KV memory budget at eight slots. It completed all passes with `AX_SSM_POSITIONS=8`, which sizes intermediate GDN state for the width-eight chain. Its tree verification graph still captured 64 nodes. All ten successful routes retained the same slots, KV token cap and model precision.
- FlashInfer attention, flashinfer_cutlass NVFP4 GEMM, 8 request slots, 4,096 context limit and a 16,384-token KV pool. Greedy sampling with server seed 42. Radix cache, FlashInfer autotuning and prefill graphs disabled. Decode graphs enabled.
- The ten frozen inputs span 23–2,266 tokens. Each response returns exactly 128 output IDs using `ignore_eos=true`. All recorded requests finished at the length limit. This fixed-work comparison covers throughput and latency. Full-sequence equality is descriptive and does not measure task accuracy or the sampling distribution.
- Warm-up for each server consisted of one request and one eight-request burst of 32 output tokens per request. The sweep used C1, C8, C8, C1 in that order. Pooled rates divide all output tokens by the sum of timed wall-clock seconds. Median latency is the median individual HTTP request duration over both passes. Timings include prefill and local HTTP overhead and exclude model load and startup graph capture.
- Exact input IDs are available in the [v0.4.0 release asset](https://github.com/defai-digital/ax-engine-sglang/releases/download/v0.4.0/parity-expanded-ids.json) as `parity-expanded-ids.json`, SHA-256 `e2b5edfeb7a18b4b01c2a67501a6659fe06eeb3ff2f98e95ef3cd957e17a7fe0`. The local results directory contains the per-lane launch settings, per-request timings and output token IDs.

## Launch the measured configurations

Use Bash with the v0.1.2 plugin installed alongside SGLang 0.5.23. Set
`TARGET` and `DRAFT` to the local pinned checkpoint directories listed above.
Define the common options once. Run one server at a time.

```bash
common=(--model-path "$TARGET" --served-model-name qwen --attention-backend flashinfer
        --max-running-requests 8 --cuda-graph-max-bs-decode 8 --context-length 4096
        --max-total-tokens 16384 --mem-fraction-static 0.8 --disable-radix-cache
        --disable-flashinfer-autotune --disable-prefill-cuda-graph --random-seed 42
        --port 30517 --fp4-gemm-backend flashinfer_cutlass)
```

**Single-request tree 64**

```bash
SGLANG_PLUGINS=ax-engine AX_NVFP4_VERIFY_PARITY=0 AX_FAST=1 AX_LATTICE=1 \
AX_GDN_LAYERS=48 AX_GDN_HEADS=48 FC1_TREE64=1 FC1_TREE_N=64 FC1_DRAFT_BLOCK=16 \
FC1_BRANCH_PILOT=1 FC1_OURS=1 FC1_FUSED=1 FC1_COMPILE_BUILD=1 AX_TREE=1 \
AX_CHAIN_WIDTH=8 AX_CHAIN_GRAPH_WIDTHS=8 AX_CHAIN_DRAFT_WIDTHS=8 AX_SSM_POSITIONS=8 \
python -m sglang.launch_server "${common[@]}" --speculative-algorithm FC1_DFLASH_TREE \
  --speculative-draft-model-path "$DRAFT" --speculative-num-draft-tokens 16
```

**Tree 32 with width-eight chain fallback, concurrency-eight confirmation**

```bash
SGLANG_PLUGINS=ax-engine AX_NVFP4_VERIFY_PARITY=0 AX_FAST=1 AX_LATTICE=1 \
AX_GDN_LAYERS=48 AX_GDN_HEADS=48 FC1_TREE64=1 FC1_TREE_N=32 FC1_DRAFT_BLOCK=16 \
FC1_BRANCH_PILOT=1 FC1_OURS=1 FC1_FUSED=1 FC1_COMPILE_BUILD=1 AX_TREE=1 \
AX_CHAIN_WIDTH=8 AX_CHAIN_GRAPH_WIDTHS=8 AX_CHAIN_DRAFT_WIDTHS=8 \
python -m sglang.launch_server "${common[@]}" --speculative-algorithm FC1_DFLASH_TREE \
  --speculative-draft-model-path "$DRAFT" --speculative-num-draft-tokens 16
```

**Stock DFlash2, block 8**

```bash
SGLANG_PLUGINS=none python -m sglang.launch_server "${common[@]}" --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" --speculative-num-draft-tokens 8
```

**Packed AR baseline**

```bash
SGLANG_PLUGINS=none python -m sglang.launch_server "${common[@]}"
```
