# RTX PRO 6000 NVFP4 parity with fixed projection row groups

The [first parity profile](2026-11-19-rtx-nvfp4-parity.md) reproduced its AR
reference on all ten prompts, but its tree and chain routes were slower than
AR. This follow-up keeps the exact-match result and removes that cost. With a
32-token output budget, tree 16 reached 86.77 output tok/s and chain 8 reached
88.24, against 46.34 for the matched AR reference. With a 128-token budget
they reached 123.64 and 122.88 against 49.09.

## Cause

SGLang 0.5.23 selects the target projection kernel from the row count of each
call. On SM120, an FP8 projection with one row uses a streaming GEMV. Two,
four, eight or nine rows use a CUTLASS GEMM for the listed shapes, and other
row counts use the generic FP8 path. NVFP4 has its own row-count table. AR
decode calls these kernels with one row and verification calls them with one
row per candidate, so the two paths produced different bits.

The first profile ran every target projection and the output head once per
candidate. Single-row decode is bound by reading the weights, so verifying
`W` candidates cost about `W` AR steps and accepted fewer than `W` tokens.

## Change

`AX_NVFP4_PARITY_ROWS` sets a fixed number of rows per projection call. AR
decode and target verification both pad their inputs with zero rows to a
multiple of that number. Each call then has the same row count and selects the
same kernel, and a verification round reads the weights once per group. The
default is 16, which covers a 16-node tree or two width-eight chains in one
group. A value of 1 restores the previous behaviour. Prefill is unchanged.

The AR reference depends on the row count. AR with 16 rows matched AR with one
row on one of ten prompts, and AR with eight rows matched on four of ten.
Compare speculative lanes only with an AR lane that uses the same value.

## Measured results

| Rows per call | AR tok/s | Tree 16 tok/s | Chain 8 tok/s | Tree and chain prompts identical to AR |
|---:|---:|---:|---:|---|
| 1 | 47.95 | 14.80 | 22.08 | Not rerun, 10/10 each in the earlier report |
| 4 | 46.99 | 46.04 | 67.20 | 10/10 each |
| 8 | 42.14 | 54.26 | 91.16 | 10/10 each |
| 16 | 46.34 | 86.77 | 88.24 | 10/10 each |

Each rate is one warm sequential pass of the ten inputs at concurrency one
with a 32-token output budget, after one untimed warm-up request. It includes
prefill and local HTTP overhead. Each speculative lane is compared with the
AR lane in the same row. The row for 16 comes from the full check below. Two
greedy-only runs with 16 rows gave 46.05, 86.16 and 87.66, then 45.79, 86.99
and 88.39 with the default setting, each with 10/10 for tree and chain.

A second pass with the default 16 rows raised the output budget to 128
tokens. Each lane returned 921 tokens over the ten inputs. Tree and chain
matched AR on all ten complete sequences.

| 16 rows per call, 128-token budget | Output tokens | Wall time (s) | Output tok/s | Relative to AR |
|---|---:|---:|---:|---:|
| Parity AR | 921 | 18.761 | 49.09 | 1.00× |
| Parity DFlash2 tree, 16 nodes | 921 | 7.449 | 123.64 | 2.52× |
| Parity DFlash2 chain, width 8 | 921 | 7.495 | 122.88 | 2.50× |

With 16 rows, the three lanes each passed the mixed-request and state-reuse
check, the repeated reference and a 32-request stress run. Each stress run sent four cancellations,
recorded zero errors and reproduced the reference outputs after reuse. AR and
tree each generated 500 six-token responses on four prompts with temperature
1.0, top-k 20 and top-p 0.95, using server seeds 324161 and 324299. All 28
distribution checks passed the 0.001 threshold and none fell below 0.01.

On the GPU, four synthetic FP8 projection shapes gave bit-identical rows
between a zero-padded AR group and a full verification group, for groups of
4, 8 and 16 rows and verification widths of 4, 8, 11, 16 and 64. Groups of 16
used the generic FP8 path with a synthetic weight scale.

## Limits

- These runs cover 16-node trees, width-eight chains, two request slots and
  concurrency one. A 64-node tree with 16 rows per call needs four groups per
  round and was not measured.
- The 128-token pass here stops at end-of-sequence and ran once. The speed
  benchmark forces exactly 128 tokens per request with eight slots, so its
  156.87 tok/s tree 64 result is not directly comparable.
- Throughput above concurrency one was not measured.
- NVFP4, BF16 and output head kernels have no separate bit-level test. The
  full-model greedy comparison covers them on the checkpoint weights.
- Sampling was checked for 16 rows only.
- Prefill and mixed prefill-decode batches are not regrouped. Requests that
  prefill together can start from different bits than the same requests alone.
- Row groups did not hold under concurrency, and the default is now a
  row-count-invariant FP8 GEMM. [Follow-up](2026-11-20-rtx-nvfp4-parity-invariant-gemm.md)

## Validation environment

NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 595.112.07, SGLang 0.5.23,
PyTorch 2.14.1+cu130 and FlashInfer 0.7.2, with CUDA graphs enabled. The
target was RadixArk/Qwen3.8-27B-NVFP4 with shard SHA-256 hashes matching the
checkpoint manifest. The draft was incoai/Qwen3.8-27B-DFlash2 at
`015e795645c74b1a0eeef3b570031fb62e769bc5`. Inputs were the ten-prompt
`parity-expanded-ids.json` release asset, SHA-256
`e2b5edfeb7a18b4b01c2a67501a6659fe06eeb3ff2f98e95ef3cd957e17a7fe0`.

## Reproduce

Use the commands in the [first parity report](2026-11-19-rtx-nvfp4-parity.md#reproduce).
`--nvfp4-verify-parity` now uses 16 rows per call. Add
`--nvfp4-parity-rows 1` to all lanes for the previous single-row behaviour.
