## Verdict

The owner is right on both counts, but the advantage is narrower than "NCM is better." NCM was never fairly tested here, and its one structural edge over a one-pass block drafter is **joint-block modeling under sampled decoding** — exactly where DFlash2 is weak (4.2/8 sampled, 3.6/8 chat). Greedy code is already at the block-8 cap; NCM cannot win there except via longer blocks. The current plan discards the only mechanism that could win. The key that unlocks it is the verification primitive the owner suspects: **shared-Gumbel exact-match**, not rejection sampling.

## 1. What can and cannot produce an advantage

**Can:** (a) A latent hierarchy models the *joint* block. Conditioned on coarse codes z, fine tokens are nearly deterministic, so the drafter's per-position conditional q(x_j|z) is far sharper than DFlash2's marginal or its rank-256 Markov selector. (b) Coarse-to-fine correction lets later levels fix earlier mistakes, flattening the per-position acceptance decay that kills one-pass drafters at L>8. (c) Early levels are short → cheap early steps, adaptive compute.

**Cannot:** raw "few steps without distillation" (irrelevant — you already distill); the paper's quality-vs-diffusion-LM results (no AR/spec-decoding baseline, never token-exact); VQ compression itself (a fidelity liability at 248k vocab, not a speed source).

**The verification crux — the owner is correct.** Rejection sampling min(1,p/q) needs the *marginal* proposal q̃(x_j)=Σ_z q(z)q(x_j|z), which is intractable; using q(x_j|z) in the ratio is biased. **Gumbel-max coupling makes this irrelevant.** Share per-position Gumbel noise g_j (seed+index) between draft and target. Target's own sample is x_j=argmax_v(log p_j(v)+g_j(v)) — exact by the Gumbel-max trick. Draft proposes y_j=argmax_v(log q(v|z)+g_j(v)) with the *same* g_j. Accept iff y_j==x_j; on mismatch emit x_j and stop. Output is always a prefix of the target's exact samples → **lossless for any drafter, no ratio, no tractable q̃.** Acceptance = E_z[1−TV(p_j,q_z)] ≥ 1−TV(p_j,q̃) by convexity of TV. Latent conditioning *strictly* raises acceptance whenever z carries joint information. Caveat: the draft must argmax with the shared noise (independent categorical sampling drops acceptance to Σpq); target must expose logits (SGLang does). Greedy is g=0, ordinary exact-match.

## 2. Was the negative result fair? No.

Confounders, all visible in the repo: gold codes computed from the target block leak the answer (the 0.07 free-running is mostly this, not NCM); tokenizer 66–84% vs paper >99.8%; `corrupt_prob=0.1` vs paper 0.50; K=4 levels (`[4,8,16,32]`) vs 8–12; 2,000 steps; 0.8B stand-in verifier; separate encoder, no target hidden states. The decision "coarse-to-fine does not transfer" is unsupported.

**Minimum fair test:** tokenizer ≥99% exact-token reconstruction (hard prerequisite); corruption mean 0.5 decreasing over levels; conditioning on the 27B's tapped hidden states; K=8, L=32; ≥50k steps on ≥1B target-distribution tokens; judge *free-running* acceptance against the real 27B with Gumbel coupling. **Continue** if ≥1.3× DFlash2 accepted tokens on sampled chat at ≤1.2× drafter cost; **stop** pure NCM below 1.1×.

## 3. NCM-Draft

L=32, K=8, levels ≈[1,2,4,8,12,16,24,32]. Discrete VQ for coarse levels (l≤8); **continuous token-space residual at the last level with a tied 248k head** to guarantee fidelity. Replace the 0.8B encoder with projected 27B last-hidden states, cross-attended (EAGLE-style). Block-diagonal attention; corruption 0.5; w_k∝l_k^−α. Verification: Gumbel coupling (sampling) / exact match (greedy). Adaptive compute: halt after level k on a confidence head, verify only the first l_k tokens. Cost: 8 passes of a ~0.5–1B drafter over ≤32 positions ≈0.1–0.25× one 27B verify pass. **Estimated** tokens/round: code-greedy 14–22, math 10–16, sampled chat 6–10.

## 4. Long blocks — yes, this is the battleground

One-pass acceptance decays with position; NCM's correction flattens it. Estimated per-position acceptance, sampled chat: DFlash2 ~0.5 falling to ~0.15 by pos 32 (≈5–7 accepted of 32); NCM ~0.6–0.7 flat (≈12–18). NCM wins only if the K-pass overhead < the acceptance gain; at L=8 it cannot pay, at L=32 it can.

## 5. Copy exactly / must change

Copy: corruption 0.5 decreasing (repo's 0.1 is wrong), nested dropout (present), block-diagonal attention (present), w_k∝l_k^−α, K=8–12 (repo's 4 is too few). Change: 248k vocab → tied target lm_head, far more tokenizer data and a larger codebook; exact fidelity → continuous last level; DeltaNet hybrid target → tapped hidden states still valid as context, verification stays causal, drafter's non-causal block attention is fine (separate model).

## 6. Hybrids, ranked

1. **NCM levels as refinement on DFlash2's first pass** — keeps greedy-code strength, adds joint modeling where weak; best expected value/risk.
2. **NCM for positions 9–32 only** — directly attacks measured decay, easy to validate.
3. **NCM codes replacing the rank-256 Markov selector** — surgical but loses coarse-to-fine correction.
4. **Non-lossless NCM for thinking tokens** — last; breaks the lossless selling point and is hard to quality-validate.

## 7. Honest probabilities (≥10% end-to-end over DFlash2)

(a) greedy code/math **~25%** (DFlash2 near cap; only L=32 helps, and it extends too). (b) sampled thinking **~50%**. (c) chat **~45%**. Reaching **1.5× tuned MTP ~55–60%** (DFlash2 already clears it in several settings); **3× ~15%** — consistent with the decision doc dropping 3× as a target.

## 8. First three experiments (2× B200, <20 GPU-h each)

- **E1 — Tokenizer fidelity gate (~15h).** Multiscale RVQ, L=32, K=8, nested dropout, tied 248k decoder, ~500M 27B tokens. **Go ≥99.0%** exact reconstruction (target 99.8%); **no-go <97%** → pure NCM dead, pivot to hybrid #1.
- **E2 — Gumbel coupling + latent oracle (~6h).** Implement shared-Gumbel exact-match in SGLang; prove losslessness on a fixed-seed suite; measure acceptance vs rejection on DFlash2 proposals (sampled chat/thinking). Upper-bound the latent gain with an oracle coarse conditioning. **Go:** losslessness confirmed **and** oracle ≥+20% acceptance on chat.
- **E3 — Minimal NCM-Draft vs DFlash2 @ L=32 (~20h).** 0.5B generator, target-hidden conditioning, corruption 0.5, K=8, ~1B tokens, ~50k steps. Free-running accepted tokens/round + tok/s vs real 27B, greedy+sampled, c=1 and c=8. **Go:** ≥1.3× DFlash2 accepted tokens on sampled chat at ≤1.2× cost **and** ≥1.1× end-to-end tok/s; **no-go <1.1×** → hybrid only.

**Bottom line:** run E2 first — it is cheap and decides whether the whole thesis (latent joint modeling + lossless Gumbel verification) is real. If the oracle gain is there, E1/E3 are worth their hours; if not, keep DFlash2 and stop calling it NCM.