# RTX PRO 6000 Qwen3.8 retest

Throughput is end-to-end output tokens/s on one GPU. Bold marks the highest rate in each row.

## Qwen3.8-27B BF16

| Workload | Clients | Stock AR | Stock DFlash2 | AX tree / chain |
|---|---:|---:|---:|---:|
| Code | 1 | 27.1 | 132.0 | **184.3** |
| Math | 1 | 27.2 | 120.9 | **163.2** |
| Chat | 1 | 27.2 | 70.5 | **94.1** |
| Code | 8 | 199.4 | 487.5 | **510.8** |
| Math | 8 | 200.5 | 567.6 | **572.0** |
| Chat | 8 | 202.4 | 333.7 | **335.1** |

## Qwen3.8-27B NVFP4

| Workload | Clients | Stock AR | Stock DFlash2 | AX tree / chain | AX GDN replay |
|---|---:|---:|---:|---:|---:|
| Code | 1 | 54.2 | 229.3 | **324.7** | 235.8 |
| Math | 1 | 54.1 | 188.5 | **270.9** | 195.2 |
| Chat | 1 | 54.4 | 125.2 | **152.2** | 128.9 |
| Code | 8 | 376.0 | 819.2 | 907.7 | **996.8** |
| Math | 8 | 377.3 | 760.5 | 823.3 | **898.1** |
| Chat | 8 | 367.0 | 505.3 | 487.0 | **580.8** |

## Qwen3.8-Flash-Next NVFP4

| Workload | Clients | Stock AR | Stock MTP3 | AX MTP3 | AX MTP7 |
|---|---:|---:|---:|---:|---:|
| Code | 1 | 82.2 | 179.1 | 181.5 | **209.1** |
| Math | 1 | 82.9 | 163.8 | 163.3 | **170.7** |
| Chat | 1 | 79.2 | **137.0** | 132.7 | 116.9 |
| Code | 8 | 412.5 | 621.7 | 650.4 | **679.7** |
| Math | 8 | 415.9 | 572.4 | **576.5** | 555.3 |
| Chat | 8 | 376.5 | 423.0 | **452.0** | 356.9 |

## Chat at concurrency eight

| Model | Route | Pass 1 / 2 tok/s | p50 TTFT (ms) | p95 request (s) | Tokens per verify | Same prompt on repeat | Matched AR |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B BF16 | Stock AR | 202.0 / 202.9 | 104 | 5.12 | — | 22/24 | — |
| Qwen3.8-27B BF16 | Stock DFlash2 | 332.3 / 335.2 | 191 | 3.34 | 3.13 | 24/24 | 18/48 |
| Qwen3.8-27B BF16 | AX tree / chain | 330.4 / 339.9 | 238 | 3.29 | 3.15 | 16/24 | 16/48 |
| Qwen3.8-27B NVFP4 | Stock AR | 352.8 / 382.5 | 193 | 3.13 | — | 22/24 | — |
| Qwen3.8-27B NVFP4 | Stock DFlash2 | 499.8 / 510.9 | 183 | 2.41 | 3.07 | 24/24 | 0/48 |
| Qwen3.8-27B NVFP4 | AX tree / chain | 483.0 / 491.1 | 249 | 2.48 | 3.01 | 11/24 | 0/48 |
| Qwen3.8-27B NVFP4 | AX GDN replay | 572.7 / 589.2 | 174 | 2.10 | 3.07 | 24/24 | 0/48 |
| Qwen3.8-Flash-Next NVFP4 | Stock AR | 357.6 / 397.4 | 160 | 3.34 | — | 0/24 | — |
| Qwen3.8-Flash-Next NVFP4 | Stock MTP3 | 409.3 / 437.8 | 152 | 3.27 | 2.61 | 1/24 | 0/48 |
| Qwen3.8-Flash-Next NVFP4 | AX MTP3 | 448.3 / 455.8 | 146 | 2.66 | 2.62 | 1/24 | 0/48 |
| Qwen3.8-Flash-Next NVFP4 | AX MTP7 | 352.8 / 361.0 | 172 | 3.54 | 2.98 | 22/24 | 1/48 |

Replay reached **580.8 tok/s** on chat at C8, 14.9% higher than stock DFlash2 (505.3 tok/s) and 19.3% higher than AX tree/chain (487.0 tok/s).

Chat p95 request latency fell from 2.41 s to 2.10 s. Replay and stock each produced 3.07 output tokens per verification, and replay matched stock on 48/48 chat sequences.

SGLang reported 10.12 GB of intermediate SSM snapshots for stock and zero for replay, which allocated about 0.055 GB for replay buffers.

Replay matched stock on 43/48 code sequences and 35/48 math sequences at C8.

With lane order reversed and two fresh C8 chat passes per route, replay reached 560.5 tok/s against 503.9 tok/s for stock (11.2% higher). Replay matched 38/48 stock sequences in that check.

Flash Next chat showed substantial greedy output variation. On repeat at C8, AR matched 0/24, stock MTP3 and AX MTP3 each matched 1/24, and AX MTP7 matched 22/24. TileLang also logged a QSA build-time data-race warning. The cause of the output variation remains open.

## Measurement conditions

- One NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), 97,887 MiB,
  driver 595.112.07. SGLang 0.5.23, PyTorch 2.14.1+cu130,
  FlashInfer 0.7.2, Triton 3.8.0,
  Transformers 5.19.0, Python 3.12.3. AX Engine source
  checkout `9b763e79e3697a8d635f4fd655c279b63f62f092` (package version 0.4.0).
- Two timed passes per model, route, workload and concurrency. Each pass sent
  the same first eight distinct prompts at C1 and 24 at C8. Prompts came from
  the [B300 benchmark corpus](../b300-flash-next-bf16/REPORT.md) (prompt file
  SHA-256 `7f5953285f237f81eda3a502e3e2454719d9352106c824c67889c9d1f2855dd8`).
- Each chat-templated request was greedy with thinking disabled, seed 42 and
  exactly 128 output tokens. The client used SGLang's streamed `/generate` API
  on loopback. Output rates pool total tokens divided by total wall time and
  include prefill and streaming. One-request and eight-request chat warm-ups
  preceded timing for each server. TTFT measures the first token-bearing event.
  Output agreement compares the complete sequence of token IDs for matched
  prompts in the same pass.
- Eight server slots, context limit 4,096, decode/verify CUDA graphs enabled.
  Radix cache, FlashInfer autotune and prefill CUDA graphs disabled. Dense
  lanes used FlashInfer attention, a 16,384-token KV pool and static memory
  fraction 0.90. Flash Next used FlashInfer attention, an 8,192-token KV pool
  and static memory fraction 0.94. The prompt lengths spanned 22–203 tokens.
- Checkpoints and revisions:
  - `Qwen/Qwen3.8-27B` at `37561816e66b5b9153b1b64c10968c18c1e6f4a0` (BF16).
  - `RadixArk/Qwen3.8-27B-NVFP4` at `319f741cce68d7914884900c138a1fbb70a42f30`.
  - `incoai/Qwen3.8-27B-DFlash2` at `015e795645c74b1a0eeef3b570031fb62e769bc5`.
  - `nvidia/Qwen3.8-Flash-Next-NVFP4` at `fc694b54fb0174e0913e6adf86691ef85a4ead47`,
    with its verified compact MTP snapshot.
- The 27B AX speed route uses a 32-node tree for eligible single requests
  and a width-eight chain for grouped requests. NVFP4 uses FlashInfer CUTLASS
  FP4 GEMM. Stock DFlash2 and replay use DFlash2 block 8, FP32 GDN state and
  Triton verification. Replay enables `AX_GDN_CHAIN_REPLAY=1` with
  `AX_FI_OFFSET_FIX=0` for the matched comparison.
- Flash Next uses FlashInfer CUTLASS for NVFP4 GEMM and target/draft MoE.
  Its FP8 PLE table uses pinned-host offload in every lane. Stock MTP3
  uses SGLang's native ring. AX MTP3 and MTP7 use a 16-slot ring.
