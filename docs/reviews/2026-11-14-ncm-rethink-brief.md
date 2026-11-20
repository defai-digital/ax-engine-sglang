# Rethink: are we using the real advantages of Next-Concept Modeling? (ax-engine-sglang, Qwen3.8-27B)

You are an independent research reviewer. Answer in English, under 1,000 words, concrete and
decisive. Do not edit files. Repo context if useful: docs/decisions/2026-11-14-retarget-qwen38-27b.md,
docs/decisions/2026-11-14-drafter-architecture.md, results/phase2_ablation/REPORT.md,
phase2/drafter.py, phase1/vqvae.py. The NCM paper is arXiv 2610.14162 (code: HazyResearch/ncm).

## The owner's concern
"We did not optimise for the NCM design advantages." The current plan (fine-tune the DFlash2 block
drafter, add adaptive width / tree verification) keeps almost nothing of NCM. The owner wants the
project to win *because of* NCM ideas, and asks us to re-read the paper and rethink.

## What NCM actually is (from the paper and the released code)
- Tokenizer: multiscale residual VQ-VAE. Level k downsamples the current residual to l_k latent
  vectors (l_1 < ... < l_K = L), quantises them, upsamples and adds to the cumulative
  reconstruction; a decoder maps the final reconstruction to tokens (argmax). Trained with nested
  dropout over the cutoff level. Exact-token reconstruction 99.79-99.99% on text (98.79% WMT14).
- Generator: one transformer predicts the code hierarchy coarse to fine. All codes of a level are
  sampled in parallel and independently, conditioned on the resized cumulative reconstruction of
  the coarser levels. K = 8-12 levels for 128 tokens (K in {4,8,16,32} ablated: more levels give
  better quality, with diminishing returns). Block-diagonal attention (each level attends to the
  prompt and itself): 1.4x faster and better than block-causal.
- Training tricks: conditioning-code corruption (random codes from the same codebook replace input
  codes, mean rate 0.50, decreasing over levels; targets stay clean) "greatly improves quality" -
  this is their fix for error accumulation; level loss weights w_k ~ l_k^-alpha.
- Why it is fast: few steps without distillation, early levels are short so early steps are cheap,
  later levels can correct earlier partial solutions (central to Sudoku: 86.1% hard at 12 NFE).
- Results are against diffusion LMs only (9-31x faster at similar quality, ~100M-725M generators),
  never against an AR model with speculative decoding, never token-exact against a teacher.
- Stated limitations: fixed depth and level lengths (an input-dependent schedule is future work),
  tokenizer trained separately from the generator, small scale, artefacts at coarse transitions.
- Probes: fixing the first 3 of 12 levels keeps the theme while wording varies - coarse levels carry
  "what is said", fine levels "how".

## Our NCM evidence so far (weak, be critical of it)
- Plan-mode drafter (codes from a block VQ tokenizer, then tokens): teacher-forced 2.0 accepted
  tokens, about 1.07 tokens per round when codes are self-predicted. But: 2,000 steps, a 0.8B
  stand-in verifier, tokenizer only 66-84% token-accurate (paper: >99.8%), NO code corruption, no
  target hidden states, a separate small encoder. So NCM was never tested fairly here.
- Chunk drafter [1,1,2,4,8] in token space: 2.15 accepted (same weak setting).

## New measurements today: Qwen3.8-27B on one B200, SGLang 0.5.23 (preliminary, tok/s)
| Mode | Setting | code c=1 | math c=1 | chat c=1 | tokens per verify round |
|---|---|---|---|---|---|
| AR | greedy, no thinking | 99 | 102 | 102 | 1 |
| DFlash2 (block 8, 2B) | greedy, no thinking | 471 | 466 | 276 | 7.5 / 6.5 / 3.6 |
| DFlash2 | temp 1.0, top-p 0.95, thinking | 312 (code) | - | - | 4.2 |
| MTP 3 steps (smoke, 1 prompt) | greedy | ~250 | - | - | 3.1 |
So with greedy decoding DFlash2 already accepts 7.5 of a maximum 8 on code: the block cap, not
drafter accuracy, limits it there. With sampling, acceptance drops to ~4.2 of 8, and on chat it is
3.6 even with greedy. At concurrency 32 DFlash2 falls from 601 to 256 tok/s per request.
Public numbers (H200, sampling + thinking): MTP 2.0-2.6x AR; DFlash2 1.32-1.48x MTP.

## A point to check
Lossless speculative *sampling* does not require guessing a random token: with rejection sampling
(accept with prob min(1, p/q)) or a Gumbel-max coupling of the draft and target noise, a drafter
whose conditional distribution equals the target's is accepted every time, even at temperature 1.
A one-pass block drafter can only output per-position marginals (plus DFlash2's rank-256 Markov
selector), so its q for position j is not conditioned on the tokens it sampled earlier in the block.
NCM's hierarchy is a latent-variable model of the *joint* block: sample the coarse codes, and the
fine tokens become nearly deterministic. Is this where NCM has a structural advantage (sampled
decoding and high-entropy chat, exactly where DFlash2 is weak)? What breaks: q(tokens) under a
latent model is not tractable for min(1, p/q); does Gumbel coupling or exact-match verification
make that irrelevant?

## Questions
1. List the NCM properties that can produce a real advantage for lossless drafting against a strong
   one-pass block drafter, and those that cannot. Be specific about mechanism, not slogans.
2. Was our negative NCM result a fair test? What is the minimum fair test (tokenizer accuracy,
   corruption schedule, target-hidden conditioning, number of levels, steps, data) and what result
   would make you continue or stop?
3. Design the best NCM-based drafter you can for this target ("NCM-Draft"): block length (16? 32?
   64?), levels and schedule, where the target's hidden states enter, whether codes are discrete VQ
   or continuous/residual token-space refinements, how verification stays lossless under greedy
   and under sampling (rejection sampling vs coupled noise vs exact match), how early levels give
   adaptive compute (stop after level k when confidence is low), expected passes and cost per
   round relative to one 27B step, and the expected tokens per round on code/math/chat.
4. Long blocks: since greedy acceptance is already at the block-8 cap, would a 16-32 token block
   with K levels (cost growing ~log L) be the place where NCM beats a one-pass drafter, whose
   later positions decay? Estimate the per-position acceptance curve each approach can reach.
5. Best practices we should copy exactly from the paper (corruption, nested dropout, block-diagonal
   attention, loss weighting, level count) and what must change for a 248k-token vocabulary, exact
   token fidelity and a causal hybrid DeltaNet target.
6. Hybrid options: NCM levels as extra refinement on top of DFlash2's first pass; NCM codes as the
   "selector" replacing the rank-256 Markov head; NCM only for positions 9-32 after a DFlash2
   block; a non-lossless NCM mode for thinking tokens only. Rank them.
7. Honest probability that an NCM-based drafter beats DFlash2 by >= 10% end to end (a) greedy code
   and math, (b) sampled thinking, (c) chat; and that it reaches 1.5x and 3x tuned MTP.
8. The three experiments to run first on 2x B200 (each under ~20 GPU-hours), with go/no-go numbers.
