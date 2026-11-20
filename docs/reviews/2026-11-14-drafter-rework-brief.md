# Design decision request: NCM block drafter rework (2026-11-14)

You are a senior ML research engineer. Be critical and concrete. No code. Answer in <= 700 words
using the numbered questions at the bottom.

## Setting
Goal: speculative decoding for Qwen3.6-35B-A3B (verifier, greedy, lossless) with a small drafter
that proposes a block of B=32 tokens in a few parallel passes, in the style of Next-Concept
Modeling (NCM, arXiv 2610.14162). The bar is Qwen's built-in MTP head: about 2.2-2.8 tokens per
verifier round with 2 draft tokens.

Drafter: frozen Qwen3.5-0.8B (same tokenizer) as causal context encoder -> memory. A 6-layer
decoder (d=1024) cross-attends to memory and predicts, coarse to fine: level-0 codes (4), level-1
(8), level-2 (16), then 32 Qwen token ids. Codes come from a frozen block multiscale residual
VQ-VAE over Qwen tokens (8192 codes per level). Training data: the verifier's own greedy outputs
(self-distillation), so held-out acceptance can be simulated exactly offline.

## Evidence so far (small smoke-scale runs, RTX PRO 6000)
- Tokenizer, lossless variant (levels 4/8/16/32): 84% code-token reconstruction after 2,500 of
  150,000 planned steps. Codebook collapse at level 0 was reduced by lowering the dead-code restart
  threshold (accuracy at 3,000 steps: 53% -> 66% in plan mode); level-0 usage is still only ~4%.
- Drafter, 1,500 steps, teacher-forced on gold codes: token accuracy 0.67, expected accepted prefix
  2.0 tokens. Code-level accuracy is low: 17% (level 0), 11.5% (level 1), 29% (level 2).
- The same drafter in free-running speculative simulation: 1.07 tokens per round (acceptance ~1%).
  Diagnosis: exposure bias. Gold codes are computed from the target block, so they leak the answer;
  at inference the predicted codes are mostly wrong and the token step collapses.

## Rework already implemented
1. Rollout training: in 50% of batches the decoder conditions on its own greedy predicted codes.
2. Corrective targets: the level-i target is the nearest code to the residual left by the codes
   actually fed in (z - zhat(input codes < i)), so targets stay consistent after earlier mistakes.
3. Validation reports free-running token accuracy and accepted prefix length.
4. Ablation: no plan at all; one decoder pass predicts all 32 tokens from learned queries.
A 2,000-step comparison of (rollout, teacher-forced, direct) is running now.

Budget: about 200 GPU-hours on 2x RTX PRO 6000 for training, then 2x H200 for final timing.

## Questions
1. Is the diagnosis right, and is rollout + corrective targets the right fix? What failure modes
   remain (e.g. greedy deterministic codes for a multimodal continuation)?
2. For exact-match greedy speculative drafting, is a discrete coarse plan sensible at all, or is
   the latent plan inherently harder to predict than the first few tokens? What result from the
   running ablation should make us drop the plan?
3. Name the 2-3 highest-value alternatives or additions, ranked: e.g. distilling verifier logits
   (KL) instead of hard tokens, a prefix-weighted loss (early positions matter most for acceptance),
   left-to-right dependency inside the block (a few AR steps over chunks), mask-predict refinement,
   continuous (non-VQ) plan, confidence calibration for draft-length selection.
4. Should we keep training the lossless tokenizer variants (>=99.95% code-token gate), or put that
   budget into the drafter and plan-mode tokenizers?
5. Give a concrete go/no-go decision rule for the next 24 hours of experiments.
