# RTX PRO 6000 dense NVFP4 target verification

This follow-up uses the RTX PRO 6000 Blackwell Server Edition, SGLang 0.5.23
and the pinned Qwen3.8-27B NVFP4 and DFlash2 checkpoints from the
[initial RTX retest](2026-11-19-rtx-27b-nvfp4.md). The greedy suite covers
seven original prompts, including a 2,266-token input, and three held-out prompts covering Traditional Chinese, Python and
arithmetic. Each request has a 32-token output budget.

## Measured results

At temperature zero, both DFlash2 routes reproduced the complete AR token IDs
and output lengths on all ten inputs. Each passed all seven original prompts
and all three held-out prompts, including the Traditional Chinese case.

| RTX PRO 6000 serving profile | Output tokens | Wall time (s) | Output tok/s |
|---|---:|---:|---:|
| Packed AR control, parity disabled | 320 | 6.576 | 48.66 |
| Parity AR | 318 | 6.635 | 47.93 |
| Parity DFlash2 chain | 318 | 14.665 | 21.68 |
| Parity DFlash2 tree | 318 | 21.703 | 14.65 |

The table measures one warm sequential pass of the ten requests at concurrency
one. It includes prefill, decoding and local HTTP overhead. Each server ran one
untimed warm-up request. On this workload, parity AR was faster than both
speculative routes. Production throughput at higher concurrency was outside
this measurement. The packed AR control used the same inputs and produced
two more output tokens than the parity routes.

A separate speed benchmark disabled verification parity and generated exactly
128 tokens per request. Tree 64 reached 156.87 tok/s against packed AR at
54.41 tok/s at concurrency one. Each benchmark has its own AR baseline.
[RTX speed benchmark](../../results/rtx_27b_nvfp4_speed/REPORT.md)

Each route passed 15 mixed requests and a separate 32-request stress run.
The server observed four cancellations on AR, five on chain and five on
tree. All three stress runs recorded zero errors, released their slots and
reproduced the reference outputs after reuse.

For the final sampling comparison, parity AR and the mixed tree/chain service
each generated 500 six-token-budget responses on four prompts, for 2,000
samples per service. The server seeds were 324161 and 324299, with
temperature 1.0, top-k 20 and top-p 0.95. All 28 distribution checks passed
the prespecified 0.001 threshold. One p-value was below 0.01. On prompt zero,
given first token 760, the next token was 13071 in 44/178 AR responses
and 39/168 DFlash2 responses. These checks cover the recorded sampling
parameters and prompts.

## Fault isolation

The checkpoint mixes NVFP4 MLPs, FP8 attention projections and BF16 B/A
projections. On six real layer-zero activation batches of shape [8, 5120],
the BF16 B/A linear on the whole batch differed from the same weights
applied to each row. The maximum absolute difference ranged from 0.03125
to 0.0625. Simply replaying the recurrent state update used to leave these
projection differences upstream of the verifier.

At layer two, two real prefixes had bit-identical incoming SSM states. With
batch-invariant dense operators and grouped decode attention, B/A agreed but
the mixed QKV projection still differed on identical input. SM120's ModelOpt
FP8 dispatcher selects a dedicated GEMV for eligible single-row projections
and different kernels for wider batches. The second RTX control changed
only FP8 projection dispatch during target verification. Exact eager chain
outputs improved from three to seven of the seven prompts.

| Incremental control | Exact greedy prompts |
|---|---:|
| Eager chain, grouped attention, invariant dense operators, row-wise BF16/FP4 and output head | 3/7 |
| Same eager chain with row-wise ModelOpt FP8 projections | 7/7 |
| Graph tree and chain with grouped attention and row-wise projections, each against AR without invariant operators | 5/7 each |

The opt-in verification profile runs the target projections one row at a time,
uses upstream batch-invariant operators for dense arithmetic, and gives each
candidate its own FlashInfer paged-decode history. A tree node can see the
committed prefix and its ancestors. AR and verification both disable KV
splitting. The drafter can still propose multiple tokens, and the target still
verifies tree and chain rounds. The target KV cache remains FP8 E4M3.

In the earlier RTX run, tree and chain each matched four of seven prompts
against unpacked-GDN AR. That earlier AR matched one of seven responses
from the new AR reference. The new comparison therefore qualifies a matched
serving profile. This also matters for sampling. For sampling prompt zero,
conditional on token 760, the next token was 13071 in 25/247 original AR
responses and 68/263 original recurrent DFlash2 responses. An intermediate AR build and the parity tree
produced 42/168 and 39/168 respectively. All 28 preliminary checks passed at
the preselected 0.001 threshold. The final comparison uses a fresh AR run
and the same tree cohort.

## Validation environment

The pod used driver 595.112.07, PyTorch 2.14.1+cu130 and FlashInfer 0.7.2
with two serving slots. The target revision was
`319f741cce68d7914884900c138a1fbb70a42f30` and the drafter revision was
`015e795645c74b1a0eeef3b570031fb62e769bc5`. CUDA graphs were enabled.
The local CPU suite passed 107 tests, including chain and sibling page-list
cases. The companion engine contract and JSON Schema checks passed.

The full three-route comparison used a single pinned plugin source digest
(`df161fe37eac2d306f8eb5febd1c8d23a3bd3e28c4afb2cfb6abcf108a76f6f1`).
The release wheel adds startup checks for the worker's CUDA device and the
NVIDIA CUTLASS DSL, and rejects an uncaptured graph wrapper during replay.
Its plugin sources hash to
`96513b9b465618614847fe0597c9666f84dc81645ebc360921873668a988a31b`.

An isolated wheel environment initially omitted the Python startup hook that
exposes `cutlass.cute`. With the same command and weights, it reproduced only
3/10 reference outputs. Restoring the installed dependency's normal site
initialization restored 10/10. The release checks that the DSL imports when
a target worker starts the parity profile. Reinstalling the final wheel
reproduced all ten greedy outputs. With round tracing enabled, ten fixed
greedy requests, warm-up, two concurrent requests, one sampled request and
server initialization recorded 80 tree and 12 chain target graph rounds.

All ten standalone GPU checks passed using the installed wheel. They include
real CUDA dispatch for synthetic FP8 weights of shape [8192, 5120], direct
comparison to eight single-token projections, CUDA graph GDN replay, and
the existing sampling, MoE and runtime checks. The full serving tests used
the pinned checkpoint weights.

## Reproduce

Set `target` and `draft` to the pinned checkpoint directories and `runner`
to the absolute path of the pod's Python. Use the same checkout for both
lanes. Download the ten-input `parity-expanded-ids.json` release asset into
the working directory. To run just the seven original prompts, replace
`--input-ids parity-expanded-ids.json` with `--long-context 2600`.
[Frozen input token IDs](https://github.com/defai-digital/ax-engine-sglang/releases/download/v0.4.0/parity-expanded-ids.json)

```bash
common=(--candidate-only --python "$runner" --baseline "$PWD" --candidate "$PWD"
        --model "$target" --draft "$draft" --nvfp4-verify-parity
        --input-ids parity-expanded-ids.json --max-tokens 32 --context-limit 4096 --slots 2
        --repeat-reference --stress-requests 32)
for lane in ar tree chain; do
    seed=324299
    sampling=(--sampling-n 500)
    if [[ $lane == ar ]]; then seed=324161; fi
    if [[ $lane == chain ]]; then sampling=(); fi
    "$runner" scripts/check_model_regression.py "${common[@]}" --route "$lane" \
        --server-seed "$seed" "${sampling[@]}" --out "results/rtx-parity-$lane"
done
for lane in tree chain; do
    "$runner" scripts/check_model_regression.py compare-greedy \
        --reference results/rtx-parity-ar/candidate --candidate "results/rtx-parity-$lane/candidate" \
        --out "results/rtx-parity-$lane/comparison.json"
done
"$runner" benchmarks/sglang/sampling_dist_check.py compare results/rtx-parity-ar/candidate/sampled.jsonl \
    results/rtx-parity-tree/candidate/sampled.jsonl --require-independent-seeds
```
