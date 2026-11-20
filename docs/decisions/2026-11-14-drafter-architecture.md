# Decision: chunk-mode drafter replaces the discrete-plan drafter (2026-11-14)

## What happened
The first end-to-end GPU test of the NCM-style drafter (coarse codes -> tokens) showed 2.0 accepted
tokens under teacher forcing but 0.07 in a free-running speculative simulation. Cause: the coarse
codes are computed from the target block, so gold codes leak the answer; predicted codes were only
11-29% accurate and the token step collapsed.

## Reviews
codex (gpt-6.1-sol), grok, muse and Qwen 4 Max reviewed the diagnosis and options
(`docs/reviews/2026-11-14-drafter-rework-*.md`). They agreed on every main point:
- A discrete plan of the whole block is a poor fit for exact-match greedy acceptance: it asks the
  model to predict the hardest thing first.
- Left-to-right dependence inside the block plus a prefix-weighted loss matches the acceptance
  process. Teacher forcing on gold earlier tokens is correct there, because a draft token only counts
  if all earlier tokens were accepted.
- Stop the lossless tokenizer: speculative decoding is lossless through the verifier, not the tokenizer.
- Judge only free-running numbers.

## Decision
- Main drafter: **chunk mode**. Frozen (later fine-tuned) Qwen3.5-0.8B context encoder; a 6-layer
  decoder drafts 16 tokens in chunks [1, 1, 2, 4, 8]. Tokens in a chunk are predicted in parallel and
  attend to the tokens of earlier chunks. Prefix-weighted loss (0.8^j, 20% uniform floor).
- **Encoder-state init**: queries start from the encoder's last hidden state and decoder residuals are
  zero-initialized, so token 1 starts as the 0.8B's own next-token prediction.
- No tokenizer is needed. Phase 1 tokenizer training is paused; its code and the plan mode stay in
  the repo as an ablation.
- Context window 2,560 tokens so the offline simulation matches a cached incremental encoder.
- Deferred, in priority order: verifier-logit (top-k KL) distillation, calibrated confidence for
  draft-length selection, plan + rollout rerun under the free-running metric.

## Evidence
`results/phase2_ablation/REPORT.md`: at 2,000 steps, accepted prefix 0.20 (direct) -> 0.65 (chunks +
prefix loss) -> 2.15 (chunks + prefix loss + encoder-state init), on a 0.8B-distilled stand-in set.

## What this means for "NCM"
The paper's coarse-to-fine code hierarchy does not transfer to lossless speculative drafting. What
carries over is the idea of a few parallel passes per block. The final system is closer to a
semi-autoregressive drafter with growing chunks than to NCM proper. Track B (rebuilding the 35B model
itself around code levels) is not supported by this evidence.

## Hardware
All training, baselines and final timing now run on one 2x B200 pod, so speed comparisons share a
GPU type. Distillation runs at about 10k output tokens/s per B200.
