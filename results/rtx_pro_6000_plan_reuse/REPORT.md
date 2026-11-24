# DFlash2 replay planning on RTX PRO 6000

Reusing SGLang's FlashInfer plan inputs lowered pooled eight-client
streamed chat throughput from 584.3 to 575.9 output tok/s. The prototype
remains opt-in. The native 32-column GDN fold tile also remains the default.

## SGLang plan reuse

| Launch order or cohort | Clients | Native tok/s | Plan reuse tok/s | Change | Native / reuse p95 |
|---|---:|---:|---:|---:|---:|
| Plan reuse first | 8 | 583.3 | 582.6 | −0.11% | 2.16 / 2.15 s |
| Native first | 8 | 585.4 | 569.3 | −2.76% | 2.14 / 2.14 s |
| Pooled | 8 | 584.3 | 575.9 | −1.45% | 2.14 / 2.15 s |
| Pooled | 1 | 134.5 | 133.0 | −1.13% | 1.16 / 1.17 s |

Native and plan-reuse runs used the same RTX PRO 6000, model and draft
checkpoints, width-eight DFlash2, FP32 GDN state, and FlashInfer settings
as in the prior experiment. The candidate added `AX_DFLASH_PLAN_REUSE=1`.
Both launch orders replayed the candidate path at captured batch sizes
1 through 8. Timed passes used 128 greedy tokens per request, streamed
over loopback, on the 24 original chat prompts. Throughput pools token
totals and wall time. [Checkpoints and runtime](../rtx_pro_6000_kv_graph/REPORT.md#setup)

All 32 serialized outputs matched. Eight-client matches were 48 of 48
in the candidate-first order and 31 of 48 in the native-first order.
The two native launches matched all 48 grouped outputs against each
other. The two candidate launches matched 31 of 48 against each other.
Pooled tokens per verify were 3.019 native and 3.021 with plan reuse.
These results leave the plan-reuse lane outside the exact-output profile.

## SGLang, ax-engine-sglang, and vLLM on RTX PRO 6000

vLLM `0.34.0rc1.dev1105+g2f170a0fe` from the local reference commit
used the same target checkpoint, DFlash2 drafter, prompt token IDs,
128-token greedy responses, and eight client slots. The launch selected
seven speculative tokens plus the anchor. It used seed 42, context
length 4,096, a 512-token prefill batch budget, FP32 GDN state, and
prefix caching disabled. Two passes per concurrency ran after warmup.

| Engine and route | C1 tok/s | C8 tok/s | C8 p95 | C8 tokens per verify | Benchmark |
|---|---:|---:|---:|---:|---|
| Stock SGLang AR | 54.4 | 367.0 | 3.13 s | — | RTX retest |
| Stock SGLang DFlash2 | 125.2 | 505.3 | 2.41 s | 3.07 | RTX retest |
| ax-engine-sglang tree / chain | 152.2 | 487.0 | 2.48 s | 3.01 | RTX retest |
| ax-engine-sglang GDN replay | 134.5 | 584.3 | 2.14 s | 3.019 | Plan reuse |
| ax-engine-sglang GDN replay + direct KV | 133.1 | 586.2 | 2.14 s | 3.024 | Direct KV |
| ax-engine-sglang GDN replay + KV graph | 133.6 | 588.8 | 2.11 s | 3.019 | KV graph |
| ax-engine-sglang GDN replay + plan reuse | 133.0 | 575.9 | 2.15 s | 3.021 | Plan reuse |
| ax-engine-sglang GDN replay + FP8 draft KV | 132.8 | 570.7 | 2.14 s | 2.974 | FP8 draft KV |
| vLLM with CUDA GDN | 155.2 | 700.3 | 1.63 s | 3.054 | vLLM comparison |
| vLLM with Triton GDN | 155.9 | 696.3 | 1.75 s | 3.097 | vLLM comparison |

All rows use the Qwen3.8-27B NVFP4 streamed chat workload on one
RTX PRO 6000. The three earlier benchmark cohorts came from separate
server launches. [RTX retest](../rtx_pro_6000_retest/REPORT.md#chat-at-concurrency-eight) ·
[Direct KV](../rtx_pro_6000_kv_direct/REPORT.md#streamed-chat) ·
[KV graph](../rtx_pro_6000_kv_graph/REPORT.md#streamed-chat)

vLLM led the ax-engine-sglang GDN replay control by 19.8% at eight clients.
vLLM used FP8 KV for the target and draft, a different hybrid cache
layout, and its streamed completions endpoint. The SGLang lanes used
FP8 target KV and streamed `/generate`. SGLang used BF16 draft KV
aside from the explicit FP8 draft KV check.
vLLM used PyTorch 2.14.0 and FlashInfer 0.7.4 with its default
FlashInfer autotuning, and captured FULL graphs for target and
DFlash2 drafting. SGLang used PyTorch 2.14.1 and FlashInfer 0.7.2,
with FlashInfer autotuning disabled. Reported vLLM tokens per verify
divide output tokens by the change in its speculative draft counter
across each pass.
[vLLM source revision](https://github.com/vllm-project/vllm/tree/2f170a0fed20fd84dfcf6da2a3a8188a32f67e4e)

The second vLLM launch changed only the requested GDN decode backend
to Triton with `VLLM_GDN_DECODE_KERNEL=triton`. C8 output throughput
fell 0.57% and remained 19.2% above the AX GDN replay control. All 64
paired output sequences differed across the two vLLM runs. The
measured rates include that change in outputs and acceptance.

One additional SGLang launch enabled its native
`--speculative-draft-kv-cache-dtype fp8_e4m3` option on the replay lane.
The logs confirmed FP8 caches for both models. Its two C8 passes
pooled to 570.7 tok/s, 2.33% below the pooled BF16-draft baseline,
and emitted 2.974 tokens per verify. This was a single-startup check.

## GDN fold tile sweep

The CUDA microbenchmark used the production dimensions of 48 layers,
48 value heads, 16 key heads, 128-dimensional heads, and a verify
width of eight. State checkpoints matched the native 32-column
launch bit for bit for each tile and both tested batch sizes.

| Value columns per program | Batch 1 fold | Batch 8 fold |
|---:|---:|---:|
| 8 | 0.180 ms | 1.662 ms |
| 16 | 0.179 ms | 1.661 ms |
| 32 (native) | 0.179 ms | 1.660 ms |
| 64 | 0.264 ms | 2.019 ms |

Each observation timed twenty fold launches in a warmed CUDA graph.
All tiles kept the native K reduction and one warp per program.

## Thor

On Jetson Thor with SGLang 0.5.23 and driver 595.78, the CUDA fixture
matched native page IDs, query offsets, and KV offsets exactly across
changing lengths and padded batches. It also checked that the direct
packer stayed within its live slice. Both configured Thor hosts were
serving workloads, so this check does not establish Thor throughput.
The RTX PRO 6000 passed the same fixture.
