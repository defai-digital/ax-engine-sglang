# Track B: lossy parallel block decoding — decision record (2026-11-14)

## Decision

Concept-Spine (Track A v2, a lossless drafter with concept conditioning) is stopped before any
training run. The project moves to Track B: commit whole draft blocks without requiring exact
agreement with the target. A no-training decisive experiment comes first.

## Why

The goal is a speed-up large enough that people would want to try NCM-style generation. Lossless
speculative decoding cannot give that (review: `docs/reviews/2026-11-14-direction-review.md`):

- One round commits at most one block. A perfect block of 16 at the measured 15 ms round is about
  1040 tok/s: 2.1x today's best public drafter on code, about 10x plain decoding. That is the
  ceiling, not an expectation.
- The accepted length is bounded by the target's own uncertainty (chat: 3.3-3.8 tokens), which a
  better drafter does not change. Concept conditioning was expected to add 3-5%.
- NCM's speed comes from generating in parallel without token-by-token verification. A native NCM
  at the quality of a 27B LLM needs pretraining-scale compute; published NCM models are 25M-130M
  parameters and are compared with diffusion language models, not with autoregressive LLMs.

Stage S results kept from Track A: the pipeline (data, capture, evaluation) is validated; the
released Qwen3-8B DSpine-b16 reaches an offline accepted length of 5.67 on 900 held-out prompts
(code 6.40, math 12.12, chat 3.80), greedy consistency 0.9906.

## Decisive experiment (no training)

`.internal/research/lossy_block_decoding/block_decode.py`, `.internal/research/lossy_block_decoding/score.py`, `.internal/research/lossy_block_decoding/run_decisive.sh`. Target Qwen3-8B, greedy,
thinking off; released DSpine-b16 drafter; HumanEval (164) and GSM8K (first 300). Policies: `ar`,
`exact`, `topk:2`, `topk:5`, `ratio:0.5/0.3/0.1/0.03`, `typical:0.09,0.3`, `force:4/8/15`.
Speed is tokens per round (one round = one target pass); quality is pass@1 / accuracy with a
paired bootstrap interval against `exact`.

Thresholds, fixed before the full run (only a 6-prompt smoke test had been seen):

- **Go**: some policy commits at least 1.5x the tokens per round of `exact` on both tasks with an
  accuracy drop of at most 2% relative on both. Then train a decoder that makes whole blocks
  acceptable (the NCM paper's remedies: training on corrupted inputs so the decoder tolerates its
  own errors, refinement passes, selection among candidates by the target).
- **No-go for untrained relaxation**: every policy within 2% relative accuracy stays below 1.2x.
  Then relaxed acceptance alone is worthless and Track B depends entirely on training a better
  block decoder; `force:15` shows the quality floor that training has to lift.
- Between the two: judgement, reported with the numbers.

Tokens per round is an upper bound on the wall-clock gain over exact speculative decoding (the
round costs the same). It says nothing about high concurrency, where verification dominates.

## Outcome (2026-11-14)

Measured in `results/track_b_decisive/REPORT.md` (Qwen3-8B, released DSpine-b16, 300 GSM8K and 164
HumanEval prompts). Go was not met. The best policies within 2% relative accuracy (`ratio:0.1`,
`typical`) give 1.25x tokens per round on GSM8K and 1.05x on HumanEval; committing unverified
draft tokens collapses accuracy even at 4 tokens per round. Relaxing the acceptance rule of a
drafter trained for lossless speculation is closed as a direction. What remains open is a drafter
trained for standalone block quality (corruption training, refinement passes, target-rescored
candidates), which needs training and was not tested here. Code QA: `docs/reviews/2026-11-14-track-b-qa.md`.
