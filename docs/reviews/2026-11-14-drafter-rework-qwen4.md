Grounded in `phase2/drafter.py` and the configs. Key code facts that shape the answer: rollout batches still use **gold token targets**; semi-AR chunks and prefix-weighting **already exist**; the lossless path emits via `tok.decode`, the plan path via a token head.

## 1. Diagnosis — half right
The train/test mismatch is real, but "exposure bias" undersells it. The 0.67→~0.01 collapse is mostly an **information** problem: gold codes are a deterministic function of the whole 32-token block, so the token head learns to *read the answer off the codes* rather than predict it. Rollout + corrective targets repairs code-target consistency, not predictability. Two remaining failure modes, one confirmed in your code:
- In `forward`, rollout batches still call `_token_loss(tl, targets, …)` with **gold tokens**. So 50% of training is "wrong plan → gold tokens," an impossible mapping that teaches the head to ignore the plan. Corrective targets fix codes only; the token objective stays mismatched.
- Greedy argmax codes commit to one mode of a multimodal continuation; corrective targets then make later levels *self-consistent with the wrong mode* — coherent but wrong. Exact-match needs the verifier's specific path. Level-0 at 17% with ~4% usage (~330 effective codes) caps exact-block accuracy.

## 2. The plan is misaligned with the metric
Acceptance is prefix-weighted: token 20 is worth nothing if token 2 is wrong. A coarse code summarizes the *entire* block — high entropy, arbitrary index, sensitive to late tokens — so it is inherently harder to predict than the first token, the lowest-entropy quantity. You pay an upfront hard prediction to ease later positions that earn no acceptance. **Drop the plan** if `free_accept_len(direct) ≈ free_accept_len(rollout-plan)` (within ~0.2), or if rollout-plan stays <~1.5 while teacher-forced-plan is high (the leak gap survived rollout → its value was leakage, not structure). Keep it only if it specifically lifts acceptance@1–4.

## 3. Ranked additions
1. **Semi-AR chunks — already built (`b32_chunks_q08.yaml`); make it the primary baseline.** Your comment is correct: for accepted-prefix, gold earlier chunks are the *right* conditioning (a token counts only if all earlier were accepted), so chunks need no rollout. This fixes coherence and matches how acceptance is earned. Also try `[2,2,4]` with B=8.
2. **Verifier-logit KL distillation.** Prefix weighting already exists (`prefix_decay=0.9`); the missing signal is soft targets. Hard CE on greedy tokens discards argmax margins; KL teaches which positions are easy and calibrates confidence. Dump verifier logits during self-distillation.
3. **Confidence-gated draft length.** `conf`/`max_len` are wired; use calibrated per-position acceptance to stop early. Cheap, orthogonal. Mask-predict and continuous plans rank lower: CMLM needs many iterations to match AR; a continuous latent removes index arbitrariness but reintroduces mean-of-modes blur.

## 4. Stop the lossless tokenizer
The ≥99.95% gate matters only for the code-direct emitter (`tok.decode`); the plan path emits via the token head, so reconstruction fidelity is off the acceptance critical path. Lossless also requires predicting *all* levels (incl. 32 level-3 codes) near-perfectly — strictly harder. At 84%/2.5k steps and 4% level-0 usage, 99.95% is a budget sink. Freeze the current checkpoint for the ablation; redirect GPU to the drafter. Revisit only if the plan proves value.

## 5. 24h go/no-go
Metric: held-out `free_accept_len` + acceptance@1–4 + drafter ms/round. MTP is ~free, so you need `tokens/(1 + t_draft/t_verify) > 2.2–2.8` — roughly **3+ tokens/round if overhead ≈40%**.
- Run the 2k comparison **plus chunks** at equal steps; add one KL variant of the leader.
- **Drop plan** if chunks ≥ rollout-plan, or rollout-plan < 1.5.
- **GO** if any variant ≥ 2.5 free-accept with overhead < ~35% → scale it.
- **NO-GO/pivot** if best < 1.8: the B=32 parallel premise is falsified at this scale; shrink to short semi-AR blocks and/or EAGLE-style verifier-feature reuse instead of a separate 0.8B encoder.
- Judge only free-running numbers; never teacher-forced.