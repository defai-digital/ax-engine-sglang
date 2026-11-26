# Stock vLLM, TensorRT-LLM and SGLang · November 25, 2026

This report adds the stock engines to the README benchmark tables on four
platforms. Every lane reuses the workload of the table it joins: same
checkpoint, drafter, prompts, output length and concurrency. Throughput is
end-to-end output tokens per second, pooled over two timed passes (total
output tokens divided by total timed seconds).

Stock vLLM is `0.34.0rc1.dev1105+g2f170a0fe` from the commit wheel index,
with `VLLM_PLUGINS` empty. TensorRT-LLM is `1.4.0rc3` (PyTorch backend,
`trtllm-serve`). SGLang is 0.5.23 with ax-engine-sglang 0.5.1.

TensorRT-LLM needed three launch fixes on every pod: plain `mpirun` with
the virtual environment first on `PATH`, serial weight loading, and the
system NVRTC 13.2 (`LD_PRELOAD=/usr/local/cuda/lib64/libnvrtc.so.13`). The
pip `nvidia-cuda-nvrtc 13.0.88` that PyTorch pins fails TensorRT-LLM's
kernel JIT with `NVRTC Internal Error`.

## Two B200 GPUs · Qwen3.8-Flash-Next BF16

Two NVIDIA B200 GPUs (driver 595.112.07) served `Qwen/Qwen3.8-Flash-Next`
at `812d96e10a0dd82b69d749e68693698cf6c36daa` (360 GB, 131 shards) with TP2.
The BF16 weights total 335 GiB against 358 GiB of GPU memory, so every lane
kept the PLE n-gram table (about 100 GB) in pinned host memory: SGLang
`--ple-offload-embedding`, vLLM's default `EngramConfig(cpu_offload=True)`,
and `TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1` for TensorRT-LLM. The checkpoint
was copied from the pod's S3 FUSE volume to local storage (`/dev/shm` plus
the root disk) before the measured lanes; loading from the FUSE volume took
over 12 minutes and missed the first readiness deadline.

The client is `benchmarks/baselines/bench_speed.py`: the frozen B300 prompt
file (SHA-256 `7f5953285f237f81eda3a502e3e2454719d9352106c824c67889c9d1f2855dd8`),
24 prompts per workload, greedy, a 256-token limit, concurrency 1, 4 and 16,
two passes. SGLang lanes ran through the B300 harness (`run_matrix.py`) with
`--tp-size 2`, Triton attention, FlashInfer GDN prefill/decode, Triton
verify, BF16 Mamba state, 16 slots, a 16,384-token KV pool and static memory
fraction 0.85. The stock MTP lanes use the 32-tensor BF16 MTP draft derived
from the checkpoint (6.49 GB, verified). AX MTP7 enables `AX_QSA_RING=16`.

Cells give code / math / chat output tok/s.

| Lane | C1 | C4 | C16 |
|---|---:|---:|---:|
| Stock SGLang AR | 215.9 / 230.1 / 227.9 | 549.2 / 644.8 / 615.0 | 980.5 / 1,301.5 / 1,239.6 |
| Stock vLLM AR | 220.1 / 223.6 / 223.3 | 598.6 / 622.4 / 597.8 | 1,233.5 / 1,383.4 / 1,278.4 |
| Stock SGLang MTP3 | 406.7 / 430.3 / 336.9 | 877.1 / 979.5 / 754.3 | 1,042.3 / 2,117.8 / 1,599.7 |
| Stock vLLM MTP3 | 508.9 / 483.5 / **369.5** | **1,260.3 / 1,270.7 / 945.0** | **2,589.4 / 2,671.9 / 1,943.0** |
| AX MTP7 | **535.9 / 506.6** / 303.8 | 943.2 / 1,029.8 / 654.3 | 2,078.6 / 2,230.6 / 1,443.4 |
| TensorRT-LLM AR, MTP3, MTP7 | failed | failed | failed |

Stock vLLM MTP3 led every cell at C4 and C16 and chat at C1; AX MTP7 led
code and math at C1. Against AX MTP7, vLLM MTP3 ran 1.34 / 1.23 / 1.44
times as fast at C4 and 1.25 / 1.20 / 1.35 times at C16.

Mean acceptance length was 3.93 / 3.50 / 2.64 for stock SGLang MTP3,
3.92 / 3.52 / 2.62 for stock vLLM MTP3 and 7.40 / 5.56 / 3.12 for AX MTP7.
The two MTP3 lanes accepted the same number of tokens per verify, so the
gap between them is per-step cost in the engine, not draft quality. vLLM
ran with `--gpu-memory-utilization 0.93`, 16 sequences and 8,192 batched
tokens; its server log reports 143,155 KV tokens (35 concurrent
4,096-token sequences). The SGLang lanes had a 16,384-token pool at
memory fraction 0.85. All SGLang lanes passed the mixed-request
and 32-request multi-user stress checks; AR and MTP7 also passed the
sampling collection.

Code ran first in every pass, and the first C16 code batch of each server
was slower in pass 1 than in pass 2: AR 896 → 1,083, stock MTP3
880 → 1,279 and AX MTP7 1,918 → 2,268 tok/s. The pooled values include
both passes.

**Discarded runs.** The first MTP7 run overlapped a second client that
reused its port, and the first AR and MTP3 runs missed a 12-minute
readiness deadline while loading from the FUSE volume. They were rerun;
`run_matrix.py` now refuses to launch on a busy port. The first stock vLLM
MTP3 server, at `--gpu-memory-utilization 0.85`, had 5.91 GiB of KV cache
(4,505 tokens, 1.1 concurrent 4,096-token sequences) and served one request
at a time, so its 509.4 / 484.8 / 371.6 tok/s stayed flat from C1 to C16.
The table uses the 0.93 rerun. TensorRT-LLM's first lanes failed before
loading: `trtllm-llmapi-launch` started `/usr/bin/python3` because Debian's
OpenMPI prepends `/usr/bin` to `PATH` when called as `/usr/bin/mpirun`.
With plain `mpirun`, the parallel weight loader was killed by the 647 GB
container memory limit (exit 137).

**TensorRT-LLM did not serve Flash Next on B200.** With serial loading and
NVRTC 13.2, every lane (AR, MTP3, MTP7) loaded both ranks and then failed in
the first decode step. Flash Next's attention backend,
`QSATrtllmAttention`, JIT-compiles the trtllm-gen kernel
`fmhaSm100aKernel_QkvBfloat16OBfloat16H256PagedKvDenseP32VarSeqQ16Kv256StaticSwapsAbForGen`,
and NVRTC stops with `could not open source file "cuda.h" (no directories in
search list)`. The wheel ships the header under
`tensorrt_llm/include/trtllm_gen_kernels/fmha/cuda/`, but the compile call
passes no include path. Setting `CUDA_HOME`, swapping NVRTC 13.2 for 13.4,
and an `LD_PRELOAD` shim that appends `--include-path` to
`nvrtcCompileProgram` did not change the error. The shim compiled a
`#include <cuda.h>` test kernel standalone, so the failing call does not go
through the exported NVRTC symbol. This is a TensorRT-LLM 1.4.0rc3 defect,
not a memory limit; more GPUs would not change it.

## RTX PRO 6000 · Qwen3.8-27B and Flash Next DFlash2

One RTX PRO 6000 Blackwell Server Edition (97,887 MiB, driver 595.112.07).
The client reproduces `results/rtx_pro_6000_retest/run.py` against vLLM's
OpenAI completions API: the same chat-templated token IDs with thinking
disabled, the first 8 (C1) and 24 (C8) prompts per workload from the B300
corpus, greedy, exactly 128 output tokens with EOS ignored, streamed, one-
and eight-request warm-ups, then two sweeps (C8 then C1, C1 then C8).
Tokens per verify come from vLLM's speculative-decoding counters.

vLLM used 8 sequences, a 4,096-token context, 8,192 batched tokens,
`--gpu-memory-utilization 0.90`, prefix caching off, FP32 Mamba state and
`{"method":"dflash","num_speculative_tokens":7}` with
`incoai/Qwen3.8-27B-DFlash2` at `015e7956`, the drafter's eight-position
block.

| Model | Clients | Stock vLLM DFlash2 (code / math / chat) | Tokens per verify |
|---|---:|---:|---:|
| 27B NVFP4 | 1 | 279.8 / 266.4 / 154.5 | 6.54 / 5.97 / 3.21 |
| 27B NVFP4 | 8 | 1,002.3 / 980.1 / 704.0 | 5.95 / 5.76 / 3.09 |
| 27B BF16 | 1 | 135.3 / 120.6 / 73.7 | 6.50 / 5.77 / 3.34 |
| 27B BF16 | 8 | 520.8 / 664.0 / 382.0 | 5.75 / 5.95 / 3.11 |

Against the README's SGLang lanes from the same workload, stock vLLM led all
three workloads at eight clients on both checkpoints, and led chat at one
client on NVFP4. The AX tree led code and math at one client. Stock SGLang
DFlash2 reported about 3.1 tokens per verify on chat at eight clients;
vLLM's DFlash2 proposer reached 5.8–6.5 on code and math.

**Flash Next with DFlash2.** Stock vLLM refuses
`incoai/Qwen3.8-Flash-Next-DFlash2` during configuration:
`NotImplementedError: Qwen4Exp speculative decoding supports only its native
MTP checkpoint and linear n-gram proposers`. The drafter's model card
documents serving only through a patched SGLang.

**Autoregressive baselines.** Stock vLLM and TensorRT-LLM ran AR with the
same client. TensorRT-LLM used `trtllm-serve` with 8 sequences, a
4,096-token context, 8,192 batched tokens, KV fraction 0.85, block reuse
off and padded CUDA graphs.

| Model | Clients | Stock vLLM AR | TensorRT-LLM AR |
|---|---:|---:|---:|
| 27B NVFP4 | 1 | 65.6 / 65.7 / 66.0 | 65.0 / 65.0 / 65.1 |
| 27B NVFP4 | 8 | **461.3 / 462.9 / 455.3** | 396.9 / 392.8 / 396.0 |
| 27B BF16 | 1 | 26.8 / 26.8 / 26.8 | 27.1 / 27.2 / 27.2 |
| 27B BF16 | 8 | 198.8 / 200.2 / 201.3 | **203.8 / 203.7 / 207.0** |

At one client the engines were within 1.5%. At eight clients vLLM led
NVFP4 by 15–18% and TensorRT-LLM led BF16 by 2–3%.

**Flash Next NVFP4 with MTP3.** Stock vLLM ran the checkpoint's native MTP
head with three speculative tokens.

| Clients | Stock vLLM AR | Stock vLLM MTP3 | Tokens per verify |
|---:|---:|---:|---:|
| 1 | 87.8 / 88.5 / 87.5 | 199.3 / 178.7 / 150.6 | 3.85 / 3.35 / 2.66 |
| 8 | 438.1 / 444.3 / 425.5 | 691.7 / 632.9 / 481.9 | 3.82 / 3.44 / 2.58 |

Against the README's SGLang rows from the same workload, vLLM MTP3 led
stock SGLang MTP3 on every cell, by 5–12%. AX MTP7 led code at both
concurrencies (218.9 and 728.0 tok/s); vLLM MTP3 led math and chat.

**TensorRT-LLM DFlash2.** DFlash on SM120 cannot use the `TRTLLM` drafter
attention backend (`requires SM100 or SM103, got SM120`). The `VANILLA`
backend imports `flash_attn_with_kvcache` from FlashAttention 2, which the
TensorRT-LLM environment does not ship. A FlashAttention 2.8.3 wheel built
for SM120 (`FLASH_ATTN_CUDA_ARCHS=120`, `-std=c++20` for the PyTorch 2.14
headers) supplies it: installed after TensorRT-LLM, it adds
`flash_attn/__init__.py` beside the FA4 package's `flash_attn/cute`. The
first attempt's install had not completed, so the import failed there. The
rerun used a fresh pod with the same client, `max_draft_len: 7` and
`attention_backend: VANILLA`.

| Model | Clients | TensorRT-LLM DFlash2 | Stock vLLM DFlash2 | AX tree/chain |
|---|---:|---:|---:|---:|
| 27B NVFP4 | 1 | 246.0 / 210.3 / 128.9 | 279.8 / 266.4 / **154.5** | **324.7 / 270.9** / 152.2 |
| 27B NVFP4 | 8 | 878.7 / **1,178.0** / 605.3 | **1,002.3** / 980.1 / **704.0** | 907.7 / 823.3 / 487.0 |
| 27B BF16 | 1 | 133.1 / 119.3 / 73.8 | 135.3 / 120.6 / 73.7 | **184.3 / 163.2 / 94.1** |
| 27B BF16 | 8 | **529.5 / 720.6 / 402.4** | 520.8 / 664.0 / 382.0 | 510.8 / 572.0 / 335.1 |

TensorRT-LLM led every BF16 workload at eight clients and NVFP4 math at
eight clients. Its first eight-client sweep ran slower than the second on
NVFP4 (code 759.5 → 1,042.2, chat 567.5 → 648.5 tok/s); the cells pool
both. At one client the two sweeps agreed within 1.1%. TensorRT-LLM
exposes no acceptance counters to this client.

Flash Next NVFP4 MTP3 ran out of GPU memory while loading experts (94.2 of
95.0 GiB in use), so the checkpoint does not fit one RTX PRO 6000 under
TensorRT-LLM.

## GeForce RTX 5090 · Qwen3.8-27B NVFP4 DFlash2

One 32 GB RTX 5090 (driver 595.112.07). The harness reproduces
`results/rtx_5090_27b_nvfp4`: the ten frozen inputs of
`parity-expanded-ids.json`
(SHA-256 `e2b5edfeb7a18b4b01c2a67501a6659fe06eeb3ff2f98e95ef3cd957e17a7fe0`,
23–2,266 input tokens), 128 greedy tokens with EOS ignored, two request
slots, and passes C1, C2, C2, C1. Each C1 pass sent 10 requests and each C2
pass 40. vLLM used an FP8 KV cache, FP32 Mamba state, 2,048 batched tokens
and `--gpu-memory-utilization 0.92`. TensorRT-LLM used `max_num_tokens`
4,096; at 2,048 it rejected the 2,266-token prompt with HTTP 400.

Pooled output tok/s; the SGLang rows are from the README study.

| Route | C1 | C2 |
|---|---:|---:|
| Stock SGLang AR | 57.64 | 101.77 |
| Stock vLLM AR | 73.7 | 129.2 |
| TensorRT-LLM AR | 70.2 | 120.4 |
| Stock SGLang DFlash2, block 8 | 125.01 | 182.38 |
| AX Engine tree 32 (chain 8 at C2) | 135.18 | 181.36 |
| Stock vLLM DFlash2, 7 tokens | **168.5** | **279.6** |

Stock vLLM DFlash2 ran 1.25 times as fast as the AX tree at one request
and 1.53 times stock SGLang DFlash2 at two, with 3.29 and 3.25 tokens per
verify on this pooled mix. The four passes of each lane agreed within
0.5%.

TensorRT-LLM DFlash2 does not fit the 32 GB card. With the FlashAttention 2
wheel from the RTX PRO 6000 section installed, model creation for the 27B
NVFP4 target and the DFlash2 drafter allocated 29.3 GiB of the 31.4 GiB and
failed on the 2.37 GiB embedding, with both `checkpoint_io_policy: auto`
(rank-striped read-ahead) and `native`. TensorRT-LLM AR, without a
drafter, fit.

The first vLLM attempt stalled for 25 minutes in FlashInfer's SM120 JIT
build: parallel `nvcc` jobs filled the pod's 62 GB memory limit and were
OOM-killed. The measured lanes reused the RTX PRO 6000's SM120 JIT cache
with `MAX_JOBS=4`.

## H200 · Qwen3.8-27B BF16, FP8 and NVFP4

One H200 SXM (143,771 MiB, driver 595.112.07, SM90). The SGLang lanes
reuse `results/rtx_pro_6000_retest/run.py`; the vLLM and TensorRT-LLM
lanes use the RTX PRO 6000 client above. Both send the same chat-templated
token IDs: the first 8 (C1) and 24 (C8) prompts per workload, exactly 128
greedy tokens with EOS ignored, one- and eight-request warm-ups, then two
sweeps (C8 then C1, C1 then C8).

- SGLang 0.5.23 with ax-engine-sglang 0.5.1, FlashInfer attention, 8
  slots, a 16,384-token KV pool, static memory fraction 0.90, chunked
  prefill 512. Stock DFlash2 used 8 draft tokens with FP32 Mamba state and
  Triton verify. The AX lane is tree 32 at one client and a width-eight
  chain at eight (`AX_CHAIN_WIDTH=8`, `AX_SSM_POSITIONS=8`). The
  SM12-only NVFP4 verify and GDN recurrent kernels stayed off.
- Checkpoints: BF16 `Qwen/Qwen3.8-27B`, FP8 `Qwen/Qwen3.8-27B-FP8` at
  `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` (128×128 block scales), and
  the NVFP4 checkpoint of the other tables through
  `--fp4-gemm-backend marlin` (W4A16) on SGLang and vLLM's FP4 Marlin
  fallback.
- vLLM: 8 sequences, 8,192 batched tokens, utilization 0.90, prefix
  caching off, `{"method":"dflash","num_speculative_tokens":7}`.
- TensorRT-LLM: `trtllm-serve`, 8 sequences, 8,192 batched tokens, KV
  fraction 0.85, block reuse off, padded CUDA graphs, DFlash with
  `max_draft_len: 7` and `attention_backend: FA4`.

| Precision | Clients | Stock SGLang AR | Stock SGLang DFlash2 | TensorRT-LLM DFlash2 | Stock vLLM DFlash2 | AX tree/chain |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 67.4 / 67.3 / 67.3 | 268.9 / 244.4 / 153.0 | 286.2 / 259.8 / 164.7 | 309.1 / 285.5 / 175.9 | **366.2 / 325.4 / 193.3** |
| BF16 | 8 | 457.4 / 463.7 / 451.0 | 1,029.1 / 946.3 / 619.4 | 931.7 / 1,011.1 / 702.6 | **1,084.5 / 1,313.6 / 830.1** | 1,061.0 / 955.4 / 626.7 |
| FP8 | 1 | 93.8 / 93.9 / 94.0 | 311.3 / 289.6 / 189.1 | 350.1 / 316.4 / 198.9 | 402.2 / **373.7 / 236.9** | **410.5** / 352.4 / 226.1 |
| FP8 | 8 | 611.8 / 612.2 / 637.4 | 1,122.1 / 895.0 / 690.7 | 1,083.0 / 1,115.2 / 721.7 | **1,160.3 / 1,359.2 / 1,018.3** | 1,132.4 / 938.0 / 705.0 |
| NVFP4 | 1 | 102.4 / 102.4 / 102.4 | 342.7 / 304.5 / 204.2 | needs Blackwell | **465.7 / 429.9 / 282.2** | 409.5 / 336.3 / 221.3 |
| NVFP4 | 8 | 642.5 / 643.9 / 672.3 | 1,063.8 / 873.5 / 609.4 | needs Blackwell | **1,357.3 / 1,382.4 / 971.8** | 1,071.7 / 913.2 / 619.9 |

Within a lane, the two sweeps agreed within 7.6% for SGLang and 14.1% for
vLLM. TensorRT-LLM's first C8 sweep ran slower than its second
(BF16 chat 650.4 → 763.8, FP8 chat 654.4 → 804.5 tok/s), a 17–23% spread;
the table pools both. The BF16 SGLang AR lane, which ran first on the pod,
spread 11.5% while FlashInfer JIT-compiled.

**Setup fixes.**

- SGLang NVFP4 with Marlin failed at the first decode with
  `AttributeError: 'tuple' object has no attribute 'dtype'`: Qwen3.5's
  dense MLP enables the fused SiLU+FP4-quant kernel whenever both
  projections are W4A4 ModelOpt layers, and hands Marlin an
  `(fp4, scale)` tuple. SGLang's kill switch
  `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1` fixed it. The harness strips
  `SGLANG_*` from the server environment, so the switch is set in the
  lane flags and recorded in `launch.json`.
- TensorRT-LLM DFlash on SM90 uses `attention_backend: FA4`, which needed
  `quack-kernels` 0.6.5 (0.5.0 fails to import against
  `nvidia-cutlass-dsl` 4.8.0). `VANILLA` needs FlashAttention 2.
- TensorRT-LLM FP8 failed in warm-up when DeepGEMM's NVRTC JIT could not
  compile its FP8 block-scaled GEMM. `TRTLLM_DG_JIT_USE_NVCC=1` switches
  DeepGEMM to nvcc, which then missed `deep_gemm/nvrtc_cutlass.cuh`;
  `TRTLLM_DG_NVCC_COMPILER` pointing at a wrapper that adds
  `-I <wheel>/tensorrt_llm/include` fixed the lane. No TensorRT-LLM code
  was changed.
- TensorRT-LLM NVFP4 kernels need SM100 or SM120, so that lane was not
  run.

## H200 · Qwen3.8-Flash-Next NVFP4

The same pod and client as the 27B section above, after the 27B
checkpoints were removed. The NVFP4 checkpoint (132.6 GB, 51.3 GB of it
the PLE table) was staged to `/dev/shm` and the root disk.

**SGLang did not load.** With `--fp4-gemm-backend marlin`,
`--moe-runner-backend marlin`, the PLE table in pinned host memory (FP8)
and static memory fraction 0.94, SGLang 0.5.23 ran out of GPU memory in
`prepare_moe_nvfp4_layer_for_marlin`: 138.0 GiB allocated of 139.8 GiB
when the repack asked for 800 MiB. The PLE embedding is built on the meta
device, so the table was not on the GPU. A retry with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` failed the same way
(138.2 GiB allocated), so the limit is the repack's peak, not
fragmentation. The stock MTP3 and AX MTP7 lanes would load the same way
and were stopped.

**Stock vLLM MTP3** (8 sequences, 8,192 batched tokens, utilization 0.90,
prefix caching off, `{"method":"mtp","num_speculative_tokens":3}`) loaded
in 210 s through vLLM's FP4 Marlin fallback.

| Clients | Stock vLLM MTP3 | Tokens per verify |
|---:|---:|---:|
| 1 | 259.9 / 247.1 / 246.1 | 3.84 / 3.42 / 2.68 |
| 8 | 846.4 / 727.1 / 710.0 | 3.83 / 3.42 / 2.61 |

The second sweep was slower than the first: C1 code 278.7 → 243.6, C1
math 261.5 → 234.2, C8 math 882.4 → 618.2 and C8 chat 767.2 → 660.8
tok/s, with unchanged tokens per verify. The cells pool both sweeps; the
pod was stopped before a rerun.

## Jetson Thor · Qwen3.8-27B DFlash2

One Jetson Thor (SM110, 120 W mode) ran the vLLM build in the
`lmsysorg/sglang:v0.5.23-cu130` container with the server flags of the
earlier Thor vLLM retest: one sequence, a 4,096-token context, FP32 Mamba
state, `VLLM_GDN_DECODE_KERNEL=cuda`, prefix caching off and
`{"method":"dflash","num_speculative_tokens":7}`. The client and prompts
match the README's Thor table: 12 prompts per workload with a 512-token
limit for NVFP4, 8 with a 384-token limit for BF16, C1, two passes.

| Checkpoint | Stock vLLM DFlash2 chain 8 (code / math / chat) | Mean acceptance length |
|---|---:|---:|
| 27B NVFP4 | 65.9 / 57.6 / 33.1 | 7.56 / 6.20 / 3.50 |
| 27B BF16 | 28.7 / 24.4 / 13.6 | 7.78 / 6.24 / 3.42 |

The two passes agreed within 1% on both checkpoints, and NVFP4 is within
4% of the earlier vLLM retest (67.21 / 59.93 / 33.20). The add-on's tree 32
ran 1.61 / 1.60 / 1.30 times as fast on NVFP4 and 1.66 / 1.54 / 1.35 times
on BF16. This vLLM build disables CUDA graphs for speculative decoding on
Thor, so verification runs eagerly.

Flash Next with DFlash2 fails on Thor with the same `NotImplementedError`
as on the RTX PRO 6000.

## B200 · Qwen3.8-27B NVFP4 batched trees

The same two-B200 pod first ran the batched-tree checks from
`results/rtx_pro_6000_tree_width/REPORT.md` on one GPU each: 200 prompts,
256 greedy tokens, tree lanes with `AX_TREE_TEMP=1.55`.

| Route | C2 tok/s | C4 tok/s | Tokens per verify C2 / C4 |
|---|---:|---:|---:|
| Tree 16 | 605.8 | 987.3 | 4.93 / 4.94 |
| Batched trees (`AX_BTREE=4`) | **710.8** | **1,107.1** | 5.84 / 5.81 |

Batched trees raised throughput by 17.3% at C2 and 12.1% at C4. The GPU
kernel check `gpu_btree_kernels.py` passed on SM100, including in-place
snapshots with `AX_SSM_POSITIONS`.

`AX_SSM_POSITIONS=8` caps the per-node snapshot buffers that batched trees
commit from. With a width-eight chain, btree4 with and without the cap
produced identical output on all 200 C4 requests (51,200 tokens, 9,069
verify rounds each). At C2, 194 of 200 requests matched and the first
difference came at token 254.2 on average, the usual batch-composition
drift. Throughput changed by under 1% (691.5 → 686.6 at C2,
1,088.3 → 1,082.5 at C4).
