# Qwen3.8-Flash-Next BF16 on two B300 GPUs (2026-11-20)

Qwen3.8-Flash-Next served BF16 inference on two NVIDIA B300 SXM6 GPUs using
tensor parallelism of two (TP2). Stock SGLang MTP3 improved aggregate throughput
over stock autoregressive decoding (AR) in all nine workload and concurrency
combinations. AX Engine MTP7 improved throughput over stock MTP3 by 18–38% on
code and 6–16% on math. MTP7 was 8–11% slower on chat.

## Configuration

- Two NVIDIA B300 SXM6 AC GPUs, 275,040 MiB each, with an NV18 link and NVIDIA
  driver 595.112.07.
- AX Engine for SGLang 0.2.0 from `08de0c698deef9146e7ec808a28b427222c5958b`,
  SGLang 0.5.23, PyTorch 2.14.1+cu130, FlashInfer 0.7.2, and Transformers 5.19.0.
- `Qwen/Qwen3.8-Flash-Next` BF16 revision
  `812d96e10a0dd82b69d749e68693698cf6c36daa`. The draft used the same
  byte-verified, 32-tensor MTP extraction in both speculative lanes.
- TP2 in all lanes. BF16 weights, KV cache, QSA index, and GDN recurrent state.
  PLE embeddings resided on the GPUs. Triton attention, FlashInfer TRT-LLM MoE,
  FlashInfer GDN prefill/decode, and Triton GDN verification. Decode CUDA graphs
  were enabled. Radix cache and prefill CUDA graphs were disabled.
- 16 running request slots, 16,384 pooled tokens, 4,096-token context limit.
  Native MTP3 verified up to four tokens per round. AX Engine MTP7 verified up
  to eight tokens per round and used a 16-slot QSA pending ring.
- Server RNG seeds in AR, MTP3, MTP7 order were 314159, 161803, and 271828.
  Requests used the same ordered prompt snapshot in all timed lanes.

## Throughput and first-token latency

The benchmark reused the [B200 NVFP4 measurement protocol](../../docs/reports/2026-11-18-b200-v0.1.1.md)
with the BF16 checkpoint on the B300 platform. Two passes used 24 prompts each
from HumanEval, GSM8K, and a fixed set of chat and UltraChat prompts at
concurrency 1, 4, and 16. Requests used greedy decoding, thinking disabled, and
a 256-token output limit. All 1,296 measured requests completed. Aggregate
throughput counts the emitted output tokens and includes prefill, decode, and
queueing. The frozen prompt file SHA-256 was
`7f5953285f237f81eda3a502e3e2454719d9352106c824c67889c9d1f2855dd8`.

The table reports the second pass. Compared with the first pass, C16 code
throughput increased by 142% for AR, 121% for MTP3, and 133% for MTP7. Both
passes are in the raw evidence. The B200 experiment used a different
checkpoint, GPU, and tensor parallelism.

| Workload | Concurrency | AR tok/s | Stock MTP3 tok/s | AX Engine MTP7 tok/s | MTP7 / MTP3 | p50 TTFT AR / MTP3 / MTP7 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Code | 1 | 184.9 | 306.6 | 390.0 | 1.272× | 217 / 251 / 222 |
| Code | 4 | 389.6 | 520.7 | 615.4 | 1.182× | 219 / 256 / 237 |
| Code | 16 | 794.5 | 1,199.7 | 1,649.8 | 1.375× | 500 / 486 / 509 |
| Math | 1 | 206.8 | 353.9 | 409.9 | 1.159× | 214 / 242 / 225 |
| Math | 4 | 550.4 | 649.4 | 735.2 | 1.132× | 379 / 286 / 409 |
| Math | 16 | 1,258.0 | 1,606.1 | 1,708.0 | 1.063× | 503 / 554 / 475 |
| Chat | 1 | 207.2 | 294.4 | 268.0 | 0.911× | 212 / 243 / 234 |
| Chat | 4 | 534.1 | 584.5 | 521.1 | 0.892× | 209 / 245 / 245 |
| Chat | 16 | 1,069.6 | 1,272.6 | 1,169.6 | 0.919× | 406 / 460 / 498 |

At concurrency one, median per-request decode speed for MTP7 was 887.8 tok/s
on code, 641.8 tok/s on math, and 378.5 tok/s on chat. Its mean committed
tokens per verify step were 7.34–7.42, 5.56–5.60, and 3.08–3.14 for the same
workloads. Each second-pass chat cell had 22 of 24 requests reach the output
limit.

## Serving and output checks

Each lane passed 15 mixed requests covering greedy, sampled, defaults,
structured responses, and log probabilities. Each also completed 32 stress
requests from eight clients, returned the tracked memory pools to idle, and
repeated all seven greedy outputs before and after stress. AR, MTP3, and MTP7
recorded two, three, and two observed cancellations, respectively.

The seven references were decoded with a 64-token budget and included one
2,266-token prompt. Greedy output matched AR on six of seven prompts for each
speculative lane. MTP3 first differed at output index 34 on prompt index 5.
MTP7 first differed at output index 48 on prompt index 1. Indices are zero
based. The long reference matched in all three lanes. These results establish
repeatability within each tested lane and leave exact cross-route greedy
equivalence unqualified.

QSA Top-K, QSA diagnostics, PLE gather, EAGLE sampling, and GDN recurrent
CUDA controls passed. The first EAGLE control timed out building its CUDA
sampling module and passed on rerun. The AR startup logged TileLang data-race
warnings while compiling QSA. CUDA graph capture and the serving runs
completed.

## Sampling distribution

AR and MTP7 each generated 2,000 independent samples, 500 per prompt on four
fixed prompts, with a six-token limit. Both used temperature 1, top-p 0.95,
and top-k 20. The 28 two-sample chi-square comparisons covered six token
positions and the joint first two tokens for each prompt. Zero comparisons
fell below either the 0.001 acceptance threshold or 0.01. The sampling
distribution gate passed.

## Evidence and completion

Raw request timings for both passes, prompt snapshots, launch records, greedy
references, sampling cohorts and metadata, GPU telemetry, and kernel-check
logs are under `raw/evidence/` in the local checkout. All timed and serving
commands exited with status 0. Both GPUs reported 0 MiB allocated after the
last serving process exited. Cgroup OOM and OOM-kill counters remained zero.
