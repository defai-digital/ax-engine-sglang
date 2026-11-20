# Flash Next EAGLE sampling follow-up on B200 (2026-11-18)

With SGLang 0.5.23 and FlashInfer 0.7.2, autoregressive sampling applies
top-k and top-p to the original distribution together. EAGLE's target-only
verifier applies top-p after top-k has renormalized the probabilities. On an
identical probability vector `[0.40, 0.27, 0.18, 0.15]`, with `k=2` and `p=0.55`,
the sequential verifier assigns the first token all the probability. The
joint filter assigns the first two tokens probabilities of approximately
0.597 and 0.403. A 40,000-draw CUDA test matched the joint filter to FlashInfer's
AR sampler. A second CUDA test verified that the worker uses the updated
function and that a failed startup rolls back the patch.

Enable `AX_EAGLE_JOINT_SAMPLING=1` with `SGLANG_PLUGINS=ax-engine` to apply
the joint filter to FlashInfer target-only EAGLE verification when both
cutoffs are active and min-p is disabled. The patch is opt-in on SGLang 0.5.23.
The 86 CPU tests passed with Python 3.10, 3.12, and 3.14.

The earlier matched-prefix probe found NVFP4 logit differences before
sampling. Switching both routes to FlashInfer CUTLASS still produced six
argmax differences in 186 identical prefixes on B200. The sampling filter
change leaves model logits as emitted by the target. [Earlier B200 report](2026-11-18-b200-v0.1.1.md)

## Sampled throughput

All three lanes used one B200, the same NVIDIA Flash Next NVFP4 checkpoint,
FlashInfer TRT-LLM MoE, on-device FP8 PLE, and decode CUDA graphs. Both MTP7
lanes used ring 16. The legacy lane used sequential top-k/top-p and the
candidate used joint filtering. After warmup at concurrency 1, 4 and 16, each
lane ran 24 ordered prompts per workload at each concurrency, for 216 timed
requests. Sampling used temperature 1, top-k 20 and top-p 0.95. The output
limit was 256 tokens. All three lanes passed mixed requests and reproduced
the six short greedy references.

| Workload | Concurrency | AR tok/s | Legacy MTP7 tok/s | Joint MTP7 tok/s | Joint vs. legacy | p50 TTFT AR / legacy / joint (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Code | 1 | 200.2 | 528.8 | 524.4 | −0.8% | 86 / 93 / 92 |
| Code | 4 | 541.5 | 1,058.0 | 1,001.0 | −5.4% | 94 / 116 / 112 |
| Code | 16 | 1,118.1 | 2,227.8 | 2,178.3 | −2.2% | 190 / 187 / 185 |
| Math | 1 | 208.0 | 445.7 | 455.8 | +2.3% | 87 / 94 / 94 |
| Math | 4 | 636.4 | 1,029.5 | 1,015.1 | −1.4% | 87 / 110 / 111 |
| Math | 16 | 1,419.0 | 2,265.7 | 2,122.1 | −6.3% | 175 / 179 / 184 |
| Chat | 1 | 207.9 | 259.3 | 253.2 | −2.4% | 84 / 94 / 97 |
| Chat | 4 | 600.1 | 615.9 | 593.3 | −3.7% | 88 / 111 / 114 |
| Chat | 16 | 1,353.1 | 1,259.6 | 1,226.0 | −2.7% | 169 / 177 / 181 |

The run has one timed pass per lane. The speed differences include changes in
generated output and accepted draft tokens as well as the filtering work.

## Independent distribution checks

Each lane collected 500 samples per prompt on four prompts with a six-token
budget. The server seeds were 314159 for AR, 271828 for legacy MTP7, and
161803 for joint MTP7. All pairwise comparisons verified the hashes of the
independent cohorts and used α = 0.001.

| Comparison | Checks below α out of 28 |
|---|---:|
| AR vs. legacy MTP7 | 5 |
| AR vs. joint MTP7 | 11 |
| Legacy MTP7 vs. joint MTP7 | 0 |

Both MTP7 lanes failed the comparison with AR. On the autumn prompt, all
1,500 samples agreed on the first four tokens. At the fifth token, AR chose
token 4105 in all 500 samples. Legacy MTP7 chose token 3514 in 31 of 500
samples and joint MTP7 chose it in 26 of 500. The earlier saved target logits
at this exact prefix put token 4105 at 97.96% in AR and 93.55% in MTP7. The
same `top_p=0.95` threshold therefore includes the next token during MTP7
verification and excludes it in AR. This difference is present before
either filter runs.

## Gated DeltaNet follow-up

On four prompts captured in eager mode, the first Gated DeltaNet layer
received identical input activations at all 28 matched decode positions.
All 28 layer outputs differed with the standard packed AR decode kernel.
Disabling packed AR decode reduced this to one of 28 positions. At the
autumn prompt's fifth token the unfiltered AR probability fell from 97.96%
to 94.82%, placing it below the 95% cutoff as in MTP7. The maximum raw
softmax total variation over the small full-model replay changed from 0.1133
to 0.1773. This isolates a source of drift in the first layer and shows
that differences still accumulate later in the model.

A warmed CUDA graph run collected 2,000 unpacked AR samples with server seed
141421. Its autumn prompt produced token 3514 in 32 of 500 trials. At α = 0.001,
both MTP7 lanes passed all 28 checks against this reference. Packed and
unpacked AR differed on 15 of 28 checks. These comparisons qualify the tested
unpacked reference and workload. The original packed AR equivalence gate
remains open.

The unpacked AR run used the same workload and prompt order. Code throughput
was 200.9 tok/s at one request and 1,133.1 tok/s at concurrency 16, compared
with 200.2 and 1,118.1 tok/s for packed AR. Median TTFT was 82 and 170 ms for
unpacked AR, compared with 86 and 190 ms for packed AR. Each value comes from
one timed pass.

To verify the plugin entry point, a final B200 run started a two-slot AR
server with `AX_GDN_UNPACKED_DECODE=1` and `AX_FI_OFFSET_FIX=0`. The dispatcher
reported `packed_decode=False` before graph capture. The server passed all
15 mixed requests and replayed six short greedy references. Four CUDA
runtime tests and both CUDA sampling tests passed after the startup change.
The server shut down, and GPU allocation returned to 0 MiB.

Raw throughput, sample cohorts, stage captures and launch records were in the
sibling directory `ax-engine-sglang-b200-evidence-eagle-joint/`. That directory
was removed on 2026-11-19 after this report was recorded on main.
