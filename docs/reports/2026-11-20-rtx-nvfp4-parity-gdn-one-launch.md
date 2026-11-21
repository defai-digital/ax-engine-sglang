# RTX PRO 6000 NVFP4 parity with one-launch GDN verification

The [parity profile](2026-11-20-rtx-nvfp4-parity-invariant-gemm.md) matched
AR but slowed down as the tree grew: tree 64 ran at 71.16 output tok/s
against 112.33 for tree 16. GDN verification called the decode kernel once
per candidate. It now uses SGLang's verify kernel, which walks every
candidate in one launch, with a parent table for every route. Tree 64 rose to
130.54 tok/s and tree 32 to 142.46, with outputs still identical to AR. With
prefill graphs and FlashInfer autotune enabled, tree 16 reached 160.70 tok/s
and 570.58 at concurrency eight.

## Why one launch gives the same bits

SGLang's Triton verify kernel and its unpacked decode are the same recurrent
function. They differ in how a candidate gets its starting state. With a
parent table, the kernel reloads the state from the snapshot buffer, as
stored. Without one, as SGLang calls it for a chain, it carries the state
between steps unrounded, which differs from decode when the state is BF16.
The plugin now passes a parent table always, linear for a chain.

On the GPU, the one-launch call and the former per-candidate loop each
matched independent single-candidate decodes bit for bit, in outputs and
snapshots, on 28 cases: FP32 and BF16 state, chain and random trees, 4 to 64
candidates and 1, 2 and 8 requests. Without the parent table, a BF16 chain
differed on 7 of 8 to 63 of 64 candidates.

| GDN verification, 48 layers, FP32 state, one request | Per-candidate loop (ms) | One launch (ms) |
|---|---:|---:|
| Tree, 16 candidates | 11.2 | 2.5 |
| Tree, 32 candidates | 21.7 | 4.7 |
| Tree, 64 candidates | 46.5 | 11.6 |

These times are CUDA graph replays of the GDN core alone on synthetic inputs.

## Serving results

One server per row, eight request slots, 128-token budget, ten frozen inputs.
Each input was sent alone, then three times at concurrency two, three and
eight. Every lane equalled its alone outputs on 30/30 requests at each
concurrency, repeated its alone outputs 10/10, and equalled the alone outputs
of the reference AR on 10/10 inputs. The reference AR ran with prefill graphs,
autotune and the radix cache disabled.

| Lane | Prefill graphs and autotune | Alone tok/s | C8 tok/s |
|---|---|---:|---:|
| AR (reference) | off | 49.04 | 237.32 |
| AR | on | 51.19 | 268.70 |
| Tree 16 | off | 139.96 | 401.11 |
| Tree 16 | on | 160.70 | 570.58 |
| Tree 32 | off | 142.46 | 416.55 |
| Tree 32 | on | 160.21 | 559.08 |
| Tree 64, two slots | off | 130.54 | 211.92 |
| Tree 64, two slots | on | 140.44 | 248.27 |
| Chain 8 | off | 103.30 | 409.18 |
| Chain 8 | on | 140.74 | 575.09 |

With all three features enabled, AR and tree 16 also equalled the reference
AR on 10/10 inputs and their alone outputs on 30/30 requests per concurrency.
Those runs repeat whole prompts, so the radix cache skipped prefill and their
rates are not reported.

Before this change, on a different pod, the same lanes with the features off
measured 112.33 (tree 16), 97.68 (tree 32), 71.16 (tree 64) and 118.02
(chain 8) tok/s alone.

## NVFP4 MLP rows

The previous report had no direct check of the NVFP4 MLP kernels. A probe in
the serving process recomputed real prefill inputs of both MLP projections
(5120→34816 and 17408→5120) as single rows and as calls of 2, 3, 4, 8, 9, 10
and 16 rows. Every row had the same bits in every call, including the
nine-row call that selects a separate SGLang kernel. The probe covered three
calls per projection.

## Acceptance checks

Regression runner, two slots, 32-token budget, features off. Tree 16 and
chain 8 matched AR on 10/10 inputs. All three lanes passed the mixed-request
check, the repeated reference and a 32-request stress run with cancellations.

| Lane | 32-token tok/s | 128-token tok/s |
|---|---:|---:|
| AR | 43.75 | 48.82 |
| Tree 16 | 88.94 | 137.76 |
| Chain 8 | 87.17 | 124.23 |

AR and tree 16 each generated 500 six-token responses on four prompts with
temperature 1.0, top-k 20 and top-p 0.95. All 28 distribution checks passed
the 0.001 threshold in two runs with different server seeds. The first run
(seeds 324161 and 324299) had two checks below 0.01, where chance predicts
0.3. The second (seeds 770001 and 771003) had none.

## Launch defaults

Superseded: with exactly 128 tokens per request, FlashInfer autotune changed
one or two of ten outputs at concurrency eight, so `AX_NVFP4_VERIFY_PARITY=1`
disables it as well as the radix cache. Prefill graphs stay on.
[Fixed-length results](2026-11-20-rtx-nvfp4-parity-forced-128.md) This run
stopped at end-of-sequence. The radix cache matched on repeated whole prompts; a hit on part of
a prompt was not tested.

## Conditions

- One NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), SGLang 0.5.23,
  PyTorch 2.14.1+cu130 and FlashInfer 0.7.2.
- Target RadixArk/Qwen3.8-27B-NVFP4, BF16 DFlash2 draft at
  `015e795645c74b1a0eeef3b570031fb62e769bc5`, 16 draft tokens per round.
- FlashInfer attention, flashinfer_cutlass NVFP4 GEMM, Triton GDN verify,
  `AX_NVFP4_PARITY_ROWS=0`, 4,096 context limit, server seed 42 and chain
  width 8 for grouped requests in the concurrency servers.
- Each number is one pass. Timings include prefill and local HTTP overhead.
  Requests stop at end-of-sequence, 906 output tokens over the ten inputs.

## Limits

- No run forced a fixed 128-token length, so these rates are not comparable
  with the speed benchmark's 156.87 tok/s for tree 64.
- Tree 64 still needs two slots. Its snapshot buffer is sized per slot and
  per candidate, and eight slots leave no memory for the KV cache.
- Verification attention still reads the prefix once per candidate.
- The serving runs did not record whether the GDN state is FP32 or BF16. The
  bit test covers both.
