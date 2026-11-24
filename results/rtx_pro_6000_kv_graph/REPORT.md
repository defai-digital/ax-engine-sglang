# DFlash2 draft KV graph on RTX PRO 6000

The CUDA fixture matched the eager KV cache exactly. At eight clients the
graph raised streamed chat throughput from 584.1 to 588.8 output tok/s
(+0.8%) and lowered frozen-prompt throughput from 586.5 to 564.5
(−3.8%). Single-client throughput was within 0.2% on both workloads.
Keep the graph experimental and opt-in.

## Streamed chat

| Order or cohort | Clients | Eager tok/s | Graph tok/s | Change | Eager / graph p95 | Eager / graph tokens per verify |
|---|---:|---:|---:|---:|---:|---:|
| Eager first | 8 | 584.4 | 586.6 | +0.4% | 2.15 / 2.13 s | 3.019 / 3.019 |
| Graph first | 8 | 583.9 | 591.0 | +1.2% | 2.15 / 2.11 s | 3.019 / 3.019 |
| Pooled | 8 | 584.1 | 588.8 | +0.8% | 2.15 / 2.11 s | 3.019 / 3.019 |
| Pooled | 1 | 133.5 | 133.6 | +0.1% | 1.16 / 1.17 s | 3.380 / 3.380 |

All 32 single-client sequence hashes matched. In both launch orders,
the graph and eager lanes generated the same two sets of 24 complete
outputs, in opposite pass order. Comparing by pass order gave 5 of 24
matches per pass. Pairing the opposite passes gave 24 of 24 matches and
identical verify counts for each pair. The graph replayed at live batch
sizes 1 through 8, with zero capture failures.

## Frozen prompts

Both lanes used native width-eight DFlash2 with accepted-prefix GDN replay.
The candidate added `AX_DFLASH_KV_GRAPH=1`. The server reported graph replay
at every live batch size from 1 through 8, with zero capture failures.

| Order or cohort | Clients | Eager tok/s | Graph tok/s | Change | Eager / graph p95 | Eager / graph tokens per verify |
|---|---:|---:|---:|---:|---:|---:|
| Graph first | 8 | 585.2 | 578.0 | −1.2% | 4.88 / 5.11 s | 2.969 / 2.928 |
| Eager first | 8 | 587.9 | 551.6 | −6.2% | 4.96 / 5.41 s | 2.959 / 2.816 |
| Pooled | 8 | 586.5 | 564.5 | −3.8% | 4.91 / 5.28 s | 2.964 / 2.871 |
| Pooled | 1 | 127.9 | 127.7 | −0.2% | 3.44 / 3.44 s | 3.088 / 3.088 |

All 120 paired serialized sequences matched. At eight clients the two lanes
matched on 34 of 40 and 31 of 40 sequences in the respective launch orders.
Comparing each lane against its own restart gave 33 of 40 matches for eager
and 32 of 40 for graph. Two of the differences in the first paired run began
with the first output token, which is produced during prefill.

## Setup

- One NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB),
  driver 595.112.07, SGLang 0.5.23, PyTorch 2.14.1, FlashInfer 0.7.2,
  Transformers 5.19.0, Triton 3.8.0.
- RadixArk/Qwen3.8-27B-NVFP4 at `319f741cce68d7914884900c138a1fbb70a42f30`
  and incoai/Qwen3.8-27B-DFlash2 at `015e795645c74b1a0eeef3b570031fb62e769bc5`.
  Eight server slots, 4,096-token context limit, 16,384-token KV pool,
  static memory fraction 0.90, FlashInfer CUTLASS FP4 GEMM,
  FP32 GDN state, and seed 42. Radix caching and prefill graphs were disabled.
- The CUDA fixture exercised the SGLang fused writer at batch sizes 1 and 8,
  comparing the complete draft KV cache against eager after zero, partial
  and full commits with changed inputs and reused physical slots.
- The chat benchmark reused the previous RTX retest's 24 chat inputs and
  streamed completions of 128 greedy tokens over loopback HTTP. Each lane
  order included two passes of eight serialized requests and two passes of
  24 requests at concurrency eight. [Chat corpus and settings](../rtx_pro_6000_retest/REPORT.md#measurement-conditions)
- The frozen benchmark used ten prompts, greedy decoding, 256 output tokens
  per request and loopback HTTP. Each lane order included 60 serialized and
  40 concurrent requests per lane. The harness repeated timed passes that
  encountered a first-time capture. Pooled throughput divides total tokens
  by total wall time. p95 uses the nearest rank across the pooled requests.
  [Benchmark procedure](../../docs/decisions/2026-11-21-dflash-hotpath.md#draft-context-kv-graph)
