# RTX PRO 6000 NVFP4 parity with a row-count-invariant FP8 GEMM

The [row-group profile](2026-11-20-rtx-nvfp4-parity-row-groups.md) matched its
AR reference with one request in flight, but its outputs changed when requests
ran together. The parity profile now runs every FP8 projection on SGLang's
generic FP8 path, which gave a row the same bits at every row count. Tree,
chain and AR then produced the same tokens alone and under concurrency two,
three and eight, and tree 16 at concurrency eight rose from 240.68 to 384.84
output tok/s. This is the default (`AX_NVFP4_PARITY_ROWS=0`).

## Which FP8 kernels depend on the row count

Each row of a 64-row input was projected alone and then inside calls of 2, 4,
8, 9, 16, 32 and 64 rows, on synthetic weights with the Qwen3.8-27B FP8
shapes. A path passes when every row has the same bits in every call.

| FP8 path in SGLang 0.5.23 | 5120→8192 | 5120→16384 | 6144→5120 | 5120→5120 | 5120→34816 |
|---|---|---|---|---|---|
| SM120 dispatch (small-row kernels, then generic) | differs from 2 rows | differs from 2 rows | differs from 2 rows | differs from 2 rows | identical |
| FlashInfer BMM | differs from 32 rows | differs from 16 rows | differs from 2 rows | differs from 2 rows | identical |
| Generic path, `torch._scaled_mm` | identical | identical | identical | identical | identical |

The generic path dispatches on the weight scale. This checkpoint's FP8 layers
carry one scale per tensor, which takes `torch._scaled_mm`. The probe and the
serving runs both used that route. Layers with one scale per channel take a
CUTLASS or tuned Triton kernel instead, which was not measured; the profile
turns the row-count-tuned Triton choice off.

At a fixed 16 rows, all three paths gave a row the same bits at either end of
the call. That is why the row-group profile matched with one request: it
fixed the row count for decode and verification. Prefill and mixed batches
kept their own row counts.

## Concurrency against the same requests alone

One server per row, eight request slots, 128-token budget, the ten frozen
inputs. Each input was first sent alone, then three times at each
concurrency (30 requests). "Equal" counts complete outputs identical to the
output of the same input alone. Every lane repeated its alone outputs 10/10.

| Lane | Equal at C2 | Equal at C3 | Equal at C8 | Alone tok/s | C8 tok/s |
|---|---:|---:|---:|---:|---:|
| AR, 16-row groups | 30/30 | 26/30 | 20/30 | 47.02 | 222.76 |
| AR, invariant GEMM | 30/30 | 30/30 | 30/30 | 48.84 | 237.80 |
| Tree 16, 16-row groups | 26/30 | 27/30 | 21/30 | 111.44 | 240.68 |
| Tree 16, invariant GEMM | 30/30 | 30/30 | 30/30 | 112.33 | 384.84 |
| Tree 32, invariant GEMM | 30/30 | 30/30 | 30/30 | 97.68 | 367.14 |
| Chain 8, invariant GEMM | 30/30 | 30/30 | 30/30 | 118.02 | 401.66 |
| Tree 64, invariant GEMM, two slots | 30/30 | 30/30 | 30/30 | 71.16 | 173.12 |

With the invariant GEMM, the alone outputs of tree 16, tree 32, tree 64 and
chain 8 were identical to the alone outputs of AR on all ten inputs. Tree 64
did not start with eight slots (no memory left for the KV cache), so it ran
with two, and its concurrency rows queue behind those two slots.

## Single-request lanes on the same pod

Two request slots, one request in flight, the regression runner's greedy pass.
Tree and chain matched their own AR on 10/10 inputs in every row below.

| Budget | Lane | Invariant GEMM tok/s | 16-row groups tok/s |
|---|---|---:|---:|
| 32 tokens | AR | 43.24 | 42.52 |
| 32 tokens | Tree 16 | 69.06 | 71.68 |
| 32 tokens | Chain 8 | 80.88 | 75.43 |
| 128 tokens | AR | 48.86 | 47.65 |
| 128 tokens | Tree 16 | 114.25 | 111.54 |
| 128 tokens | Chain 8 | 114.52 | not run |

The 128-token pass stops at end-of-sequence: 906 output tokens with the
invariant GEMM and 921 with 16-row groups. The two modes are different AR
references and agreed on one of ten complete outputs.

## Acceptance checks for the default

With the invariant GEMM and a 32-token budget, AR, tree 16 and chain 8 each
passed the mixed-request and state-reuse check, the repeated reference and a
32-request stress run with cancellations. AR and tree each generated 500
six-token responses on four prompts with temperature 1.0, top-k 20 and top-p
0.95, using server seeds 324161 and 324299. All 28 distribution checks passed
and none fell below 0.01.

A tree server launched with only `AX_NVFP4_VERIFY_PARITY=1` and the
speculative arguments, using 16-row groups, matched the explicitly configured
16-row AR on 10/10 inputs. This checks the launch defaults added with the
parity guards. The GPU projection check passed four tests, including the
row-count test above through the installed patch for calls of 2 to 64 rows.

## Conditions

- One NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), driver
  595.112.07, SGLang 0.5.23, PyTorch 2.14.1+cu130 and FlashInfer 0.7.2.
- Target RadixArk/Qwen3.8-27B-NVFP4, BF16 DFlash2 draft at
  `015e795645c74b1a0eeef3b570031fb62e769bc5`, 16 draft tokens per round.
- FlashInfer attention, flashinfer_cutlass NVFP4 GEMM, Triton GDN verify,
  recurrent GDN verification, 4,096 context limit. Radix cache, FlashInfer
  autotuning and prefill graphs disabled. Concurrency servers used seed 42
  and chain width 8 for grouped requests.
- Each number is one pass. Timings include prefill and local HTTP overhead.
- This pod is a different machine from the one in the row-group report, so
  compare modes within this report only.

## Limits

- The NVFP4 MLP kernels were not tested for row-count dependence here. A
  later probe on real weights found identical rows for 2 to 16 rows, the
  nine-row kernel included. [Follow-up](2026-11-20-rtx-nvfp4-parity-gdn-one-launch.md)
- Every run here disabled the radix cache, FlashInfer autotune and prefill
  graphs. The follow-up measured matched outputs with all three enabled.
- A tree server that used 16-row groups and no chain graphs matched alone
  outputs on only 20/30 requests at concurrency eight and ran at 52.84 tok/s.
  The same launch was not repeated with the invariant GEMM.
- One-row groups (`AX_NVFP4_PARITY_ROWS=1`) under concurrency were not rerun.
- Sampling was checked for tree 16 only, and no run forced a fixed 128-token
  length, so these rates are not comparable with the speed benchmark.
