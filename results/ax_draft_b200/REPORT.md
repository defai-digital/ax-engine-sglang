# Independent AX Draft on B200

AX Draft v1 loaded its own 3.03 MB checkpoint on SGLang 0.5.23. The
worker uses the target's last committed normalized hidden row and predicts
seven tokens. It runs the native eight-row chain verifier with accepted-prefix
GDN ReplaySSM. The server loaded only the target and AX Draft weights in
this lane. It allocated no separate draft KV cache.

## Conditions

- One NVIDIA B200 with driver 595.112.07, PyTorch 2.14.1 and FlashInfer 0.7.2.
- Same Qwen 3.8 27B NVFP4 target pack, FlashInfer CUTLASS, FP32 GDN state,
  FP8 KV, Triton verification, width eight and eight request slots in both arms.
- Decode CUDA graphs enabled. Prefill CUDA graphs and radix caching disabled.
- Four held-out prompts named `code-cache`, `math-cistern`, `prose-operations`
  and `code-checksum`, each with 64 greedy output tokens. Both arms ran two
  C1 passes of four requests and two C8 passes of sixteen requests.
- The 757,862-parameter AX Draft checkpoint was trained from random initialization
  with 456 windows from eight MLX teacher requests. Its weight SHA-256 is
  `76af949be4531bce6a0d973e20a9a9ec04ba6526c848b230bf1bc795eece2488`.

## Results

| Concurrent clients | DFlash2 output tok/s | AX Draft output tok/s | DFlash2 tokens/verify | AX Draft tokens/verify | Exact output match |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 173.01 | 83.07 | 4.13 | 1.13 | 8/8 |
| 8 | 466.73 | 285.05 | 3.76 | 1.12 | 12/32 |

Throughput pools both passes at each concurrency. Both routes used GDN
replay in every measured verify round. The exact match column compares
paired AX Draft and DFlash2 token IDs. The batch-eight comparison used
overlap scheduling and does not isolate the source of token differences.
The pilot's acceptance and throughput keep it on the experimental path.

[Aggregate receipt](summary.json) · [AX Draft passes](axdraft8.json) · [DFlash2 passes](replay8.json)

The experiment is available through the
[AX Draft launch settings](../../ax-engine-sglang/README.md#experimental-ax-draft-checkpoint).

## Backend-matched teacher and proposal follow-up

On the same GPU class, the target-only NVFP4 server generated 96 greedy
tokens per prompt. The recorder paired every output token with the target
hidden state that produced it. The target ran fixed batches of up to eight
with decode graphs and FP8 KV; prefill graphs and overlap were disabled.
The prompt split comprised 66 training and four held-out requests.

| Requested shortlist | Proposal hidden width | Training steps | Held-out top-1 | Held-out tokens per verify, seven drafts |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 128 | 350 | 13.4% | 1.131 |
| 4,096 | 256 | 550 | 12.9% | 1.131 |

This is an offline replay of each model's proposed chain on recorded
target features. Backend-matched labels alone did not materially
increase acceptance on these prompts.

A CUDA selector fuses the greedy walk over the existing compact scores,
transition edges and vocabulary IDs. The CUDA test matched the PyTorch
reference across batch sizes 1 and 8, depths 3 and 7, and shortlist
widths 4, 5 and 8, including ties and nonfinite scores. An early
proposal-only microbenchmark at batch eight and depth seven reduced
mean wall time across 500 synchronized calls from about 490 µs to
284 µs. This reduces proposal cost independently of the target
verification and commit.

### Fixed-batch serving comparison

The native 1,024-token checkpoint ran on one B200 with SGLang 0.5.23,
PyTorch 2.14.1 and FlashInfer 0.7.2. Both draft paths used eight rows,
FP32 GDN ReplaySSM, FP8 KV and eager prefill. Decode CUDA graphs and
overlap were enabled. Four 64-token prompts were sent serially, then
in two passes of 16 requests grouped into fixed batches of eight, and
then serially again. The table uses the last pass at each concurrency
to exclude the cold first pass.

| Concurrency | Target-only tok/s | DFlash2 tok/s | AX tok/s | DFlash2 tokens/verify | AX tokens/verify |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 104.17 | 173.96 | 86.90 | 4.063 | 1.164 |
| 8 | 799.32 | 1,117.73 | 567.75 | 3.606 | 1.180 |

AX matched DFlash2 for 8/8 serial and 32/32 concurrent outputs. Both
drafted routes matched target-only for 2/8 serial and 0/32 concurrent
outputs. The target-only route also diverged between serial and
concurrent batches, including a first-token difference on one prompt.
These figures compare both draft engines on identical emitted token
streams. They leave the separate target-only parity gate open.

### Short verification windows

The same native checkpoint and graph/overlap policy were used for the
two- and four-row routes. Each lane warmed a full eight-request batch
before timing. The last pass at each concurrency is shown again.

| AX verification rows | C1 tok/s | C8 tok/s | C8 tokens/verify | C1 exact vs DFlash2 | C8 exact vs DFlash2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 93.12 | 675.55 | 1.185 | 0/8 | 32/32 |
| 4 | 91.02 | 645.63 | 1.191 | 0/8 | 32/32 |
| 8 | 86.90 | 567.75 | 1.180 | 8/8 | 32/32 |

At C8, the two-row lane delivered 19.0% more tokens per second than
AX at eight rows and reached 60.4% of DFlash2's throughput. Neither
short width passed the target-only parity gate. With both decode
graphs and overlap disabled, the eight-row draft engines continued
to match each other, and the differences against target-only remained.
The final eager C8 runs were 141.58 tok/s for DFlash2 and 87.69 tok/s for
AX. Those rates describe the diagnostic run with graphs disabled.
