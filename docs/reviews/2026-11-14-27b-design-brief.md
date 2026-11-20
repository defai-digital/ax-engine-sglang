# Design review: ax-engine-sglang retargeted to Qwen3.8-27B

You are an independent design reviewer. Answer in English, under 900 words, quantitative and
decisive. Do not edit files. Repo context if useful: docs/plan.md,
docs/decisions/2026-11-14-drafter-architecture.md, phase2/drafter.py, phase2/README.md,
results/phase2_ablation/REPORT.md.

## Decision already taken by the owner (not up for review)
The Qwen3.6-35B-A3B target is dropped (MoE with 3B active parameters makes the verifier step too
cheap for an external drafter). All 35B jobs were stopped. The new target is **Qwen/Qwen3.8-27B**.
The owner wants "a Qwen3.8-27B NCM model" (NCM = Next-Concept Modeling, arXiv 2610.14162: coarse-
to-fine parallel generation over multiscale residual VQ codes) and hopes for >= 3x the speed of the
built-in MTP mode. Hardware: one pod with 2x B200 (183 GB each); budget up to ~200 GPU-hours of
training and ~100 h of evaluation, more if justified.

## Target facts (HF config / model card)
Dense 27B, 64 layers = 16 x (3 Gated-DeltaNet + 1 full attention), hidden 5120, FFN 17408,
24 heads / 4 KV heads, head_dim 256, vocab 248,320, untied embeddings, 1 MTP layer ("trained with
multiple steps"), thinking on by default, Apache-2.0, context 262k.

## Published state of the art on this exact target (one H200, concurrency 1, temp 1.0, top-p 0.95,
## thinking, xhigh effort) -- tok/s
From the DFlash2 card (incoai/Qwen3.8-27B-DFlash2, Apache-2.0, 2B, block diffusion: one pass
predicts 8 positions with top candidates per position, a small selector picks a path):
| Task | AR | MTP (k=7) | DSpark | DFlash2 | DFlash2 / MTP |
|---|---|---|---|---|---|
| GSM8K | 68.9 | 178.5 | 185.3 | 236.1 | 1.32 |
| HumanEval | 69.0 | 151.9 | 159.9 | 214.6 | 1.41 |
| MBPP | 69.0 | 153.1 | 163.3 | 226.9 | 1.48 |
| MT-Bench | 68.9 | 134.9 | 137.6 | 184.0 | 1.36 |
Acceptance length MTP / DSpark / DFlash2: GSM8K 5.02/4.36/5.46, HumanEval 3.91/3.30/4.39,
MT-Bench 3.74/3.01/4.10. At concurrency 32 MTP and DSpark fall below AR; DFlash2 is 1.0-1.45x AR.
From the DSpark card (RadixArk/Qwen3.8-27B-DSpark v2, 1.86B, 5 full-attention layers at d=5120,
consumes target hidden states from layers 5/19/33/47/61, rank-256 Markov head, trained with
SpecForge for 16 future positions, served at block 7): DSpark v2 / MTP at concurrency 1 =
1.65 (GSM8K), 1.54 (HumanEval), 1.57 (MBPP), 1.37 (MT-Bench); AR 94-96 tok/s, MTP 157-180.
So: MTP already gives 1.6-2.6x AR, and the best public drafters give about 1.35-1.65x MTP.
3x MTP would mean roughly 6-7x AR at batch 1.

## What we have
- Chunk drafter code (phase2/drafter.py): separate Qwen3.5-0.8B encoder -> 6-layer decoder drafting
  16 tokens in 5 passes over chunks [1,1,2,4,8]; prefix-weighted loss; early stop after any chunk.
  Free-running accepted length 2.15 after 2,000 steps with a 0.8B stand-in verifier. It never sees
  the target's hidden states.
- NCM plan-mode code (block multiscale VQ tokenizer + code-level drafter). It failed free-running:
  teacher-forced 2.0 accepted tokens, about 1.07 tokens per round when codes are self-predicted.
- Self-distillation generator on the vLLM offline engine, offline greedy speculative simulator,
  verify-cost timer, Phase 0 vLLM benchmark harness (AR vs MTP).

## Draft proposal from the lead (criticise it)
Two tracks, lossless first.

Track A, "ax-engine-sglang drafter v2" (lossless speculative decoding, output identical to the target):
1. Condition on the target: fuse hidden states from ~5 tapped target layers (as DSpark does) plus
   the embeddings of accepted tokens; drop the separate 0.8B encoder.
2. Coarse-to-fine block drafting in 2-3 passes instead of 1: pass 1 predicts all 8-16 positions in
   parallel with top-k candidates; passes 2-3 re-predict later positions conditioned on the argmax
   (or soft embedding mix) of earlier positions, i.e. our chunk schedule [2,6] or [1,3,12] used as
   refinement. This is the "NCM" element kept: scale-wise refinement, but in token space, not VQ
   codes.
3. Adaptive draft length: a calibrated per-position confidence head decides how many tokens to send
   to verification (verify width costs time; MTP k=7 and DFlash2 block 8 are fixed).
4. Train on the 27B's own outputs (thinking and non-thinking, sampled and greedy), loss = prefix-
   weighted CE + top-k KL to the target's logits at each position.
5. Gate: same engine, same GPU, same precision, vs tuned MTP, DSpark v2 and DFlash2: >= 1.5x MTP on
   code and chat at concurrency 1 and no regression below AR at concurrency 32.

Track B, "27B-NCM" (not lossless): fine-tune the 27B itself (LoRA or last-N-layer adapters plus a
block head) to emit a block of 8-16 tokens per forward pass with coarse-to-fine refinement, with
the original AR path kept as verifier/fallback. Only this track could plausibly reach >= 3x MTP;
quality regressions are expected and must be measured (HumanEval, MBPP, GSM8K, MT-Bench).

## Questions
1. Realistic ceiling: for lossless decoding on this target, what speedup over MTP is attainable at
   concurrency 1 (give a number and the arithmetic: accepted length, draft cost, verify cost)? Is
   3x MTP reachable by any lossless design? If not, say so plainly.
2. Track A architecture: which of proposal items 1-4 are worth doing, which are not, and what is
   missing (e.g. tree/multi-candidate verification, reusing the MTP head as the first-token
   predictor, distilling into the MTP head itself, d=5120 vs a projected narrower drafter, number of
   layers, KV reuse, DeltaNet state)? Give one concrete recommended architecture with sizes.
3. Does any real NCM component (multiscale VQ codes, concept tokens) earn a place, or should the
   name be kept only for scale-wise refinement in token space? Is there a design where coarse codes
   help free-running acceptance rather than hurt it?
4. Build vs reuse: train from scratch with our code, fine-tune DFlash2 (Apache-2.0) or use
   SpecForge / z-lab/dflash training code? What is the fastest path to a result that beats
   DFlash2, and what would be a genuinely new contribution?
5. Data and compute on 2x B200: how many target tokens of self-distillation are needed, expected
   27B generation throughput, how to cache tapped hidden states (storage: 5 layers x 5120 floats per
   token is ~50 KB/token in bf16) or whether to run the frozen target online during training;
   training steps and wall-clock.
6. Track B: is it worth starting at all? Smallest experiment that would show whether a block-
   emitting 27B keeps quality, and the kill criterion.
7. Engine: vLLM (MTP built in, DFlash via PR 52816) or SGLang (DSpark and DFlash native)? Which one
   for the benchmark and the integration?
8. Your recommended plan as an ordered list with go/no-go gates, and the first three experiments.
