# Direct DFlash2 draft KV writes on RTX PRO 6000

On streamed chat at eight clients, the direct BF16 writer moved pooled
throughput from 584.8 to 586.2 output tok/s (+0.25%). Both lanes emitted
3.024 tokens per verify. One frozen-prompt comparison measured a 1.08%
gain. The CUDA fixture matched the native cache exactly at batch sizes
1 and 8. Keep the direct writer opt-in.

## Streamed chat

| Launch order or cohort | Clients | Eager tok/s | Direct tok/s | Change | Eager / direct p95 |
|---|---:|---:|---:|---:|---:|
| Direct first | 8 | 583.5 | 585.1 | +0.28% | 2.15 / 2.17 s |
| Eager first | 8 | 586.1 | 587.4 | +0.22% | 2.15 / 2.13 s |
| Pooled | 8 | 584.8 | 586.2 | +0.25% | 2.15 / 2.14 s |
| Pooled | 1 | 134.3 | 133.1 | −0.93% | 1.16 / 1.17 s |

All 32 paired single-client sequences matched across both launch orders.
Each eight-client launch order matched 38 of 48 complete outputs.
Comparing the eager lane against its own restart gave the same
38-of-48 match, as did comparing the direct lane against its own restart.
The direct writer ran at batch sizes 1 through 8 in both launch orders.

## Frozen prompts

In one eager-first run at eight clients, throughput rose from 553.0 to
559.0 output tok/s (+1.08%) and p95 fell from 2.85 to 2.82 s. Both lanes
emitted 3.290 tokens per verify. All 30 grouped and 20 serialized output
sequences matched. The run used 128 greedy output tokens per request,
`--concurrency 8 --repeat 3`, and the frozen inputs from the earlier
27B benchmark. [Frozen input IDs](https://github.com/defai-digital/ax-engine-sglang/releases/download/v0.4.0/parity-expanded-ids.json)

## Setup

- Both lanes used the same RTX PRO 6000, driver, SGLang 0.5.23 runtime,
  target and DFlash2 checkpoints, and width-eight GDN replay settings
  as the previous KV graph experiment. The candidate added
  `AX_DFLASH_KV_DIRECT=1`. [Runtime and checkpoint pins](../rtx_pro_6000_kv_graph/REPORT.md#setup)
- Both launch orders reused the previous 24 streamed chat prompts.
  Each lane ran two passes of eight serialized requests and two passes
  of 24 requests at concurrency eight, with 128 greedy output tokens
  per request. Pooled throughput divides total output tokens by total
  measured wall time. p95 uses the nearest rank across the requests.
  [Chat corpus and settings](../rtx_pro_6000_retest/REPORT.md#measurement-conditions)
- The CUDA fixture compared the entire BF16 cache after zero, partial,
  and full commits with changed inputs and reused physical slots. The
  direct writer retains the native RoPE fused multiply-add order to
  preserve rounding. [GPU fixture and benchmark procedure](../../docs/decisions/2026-11-21-dflash-hotpath.md#direct-writes-after-the-stacked-draft-kv-projection)
