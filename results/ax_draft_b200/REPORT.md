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
