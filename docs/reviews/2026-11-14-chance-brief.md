# Review request: winning chance and design correctness of NCM-Draft for Qwen3.8-27B

Independent review wanted. Be blunt; disagree with us where the evidence says so. Design docs are in
the repository at /Volumes/Ext16TR0/code/ax-engine-sglang (read-only): `docs/decisions/2026-11-14-ncm-draft.md`,
`docs/decisions/2026-11-14-retarget-qwen38-27b.md`, code in `phase2/drafter.py`, `track_a/`.

## Goal
Lossless speculative decoding for Qwen/Qwen3.8-27B (dense, 64 layers = 48 Gated DeltaNet + 16 full
attention, hidden 5120, 1 built-in MTP layer, thinking on by default). "Win" = measured end-to-end
tok/s in SGLang on the same B200 GPU, byte-identical greedy output, faster than the public drafters
incoai/Qwen3.8-27B-DFlash2 and RadixArk/Qwen3.8-27B-DSpark and than built-in MTP. Stretch: 3x MTP.

## Our design (NCM-Draft v1)
Idea from Next-Concept Modeling (arXiv 2610.14162): draft a 32-token block coarse-to-fine.
- Block tokenizer: multiscale residual VQ-VAE, levels [1,2,4,8,16,32] codes, 8192 codes per level,
  nested dropout. Gate: >=99% token reconstruction.
- Drafter: d=1024, 6 layers, 16 heads; cross-attends to hidden states tapped from the frozen 27B at
  layers 5/19/33/47/61 (same taps as DFlash2), anchor convention (memory stops before the last
  committed token, which enters as an embedding). 6 code passes then 1 token pass (7 parallel
  passes per block); token logits through the frozen target LM head.
- Training: frozen 27B online, greedy non-thinking self-distillation (40k prompts each of code,
  math, chat); NCM recipe: input-code corruption mean 0.5 decreasing over levels, clean labels,
  code dropout 0.1. 30k steps.
- Controls with identical conditioning: `direct` (one pass, 32 tokens in parallel, DFlash-like) and
  `chunks` (token-space coarse-to-fine [1,1,2,4,8,16], no tokenizer).
- Gates: NCM >=10% better accepted length than both controls and better survival at position 16;
  then beat DFlash2 offline on the benchmark prompts; only then SGLang integration.

## Measurements so far (1x B200, SGLang 0.5.23, c = concurrency, tok/s)
- AR: ~102 at c=1 on every set; c=32: 1170-2000.
- DFlash2 (block 8, 5 layers d=5120), greedy non-thinking c=1: code 489 (accept 7.5 of 8),
  math 453 (6.2), chat 250 (3.3). Sampled+thinking c=1: code 310 (4.1), math 366 (5.0), chat 242 (3.2).
  c=32 sampled: math 1662, chat 2508.
- DSpark (block 7), sampled+thinking: code c=1 448 (accept 6.96), c=32 2734; math c=1 360 (5.06),
  c=32 2946.
- MTP3: ~250 tok/s, accept 3.1 (single-prompt smoke; full MTP2/3/5/7 sweep still running).
- So one DFlash2 round costs ~17 ms vs 9.8 ms per AR token; speedup shrinks from ~4.4x AR at c=1 to
  ~2.2x at c=32.
- Our own evidence is weak: earlier NCM tokenizer on Qwen tokens reached only 66-84% reconstruction;
  earlier chunk drafter (separate 0.8B encoder, 35B target) accepted 2.15 tokens; in a CPU toy task
  only the one-pass `direct` variant learned, `chunks` had teacher-forced acc 0.95 but free-running
  0.31. No GPU run of the new design yet.

## Our current estimate (subjective)
Offline accept length beats DFlash2 on greedy code (any variant): 55-65%. NCM beats its own controls
by >=10%: 20-30%. End-to-end >=10% faster than DFlash2/DSpark on greedy code/math: 25-30%; sampled+
thinking: 15-20%; chat: 10-15%. 3x MTP: ~5% code, <1% chat.

## Questions
1. Your own probabilities for each line above, with the arithmetic (accept length, round latency).
2. Is the design correct? Name the biggest mistakes. In particular: (a) is a VQ code hierarchy the
   right use of the NCM idea for a *lossless drafter*, where only the exact-match prefix counts, or
   does coarse-to-fine help nothing because errors at position 1 kill the block? (b) block 32 vs
   adaptive length; (c) 7 passes vs latency; (d) drafter size d=1024x6 vs DFlash2's d=5120x5;
   (e) greedy non-thinking training while the deployment default is sampled+thinking; (f) chat,
   where accept is ~3.2 for everyone.
3. What would you build instead with the same budget (2x B200, ~200 GPU-hours training) to maximise
   the chance of a measurable win over DFlash2/DSpark? Give a concrete architecture and training
   recipe, and the first experiment that would falsify it within 6 GPU-hours.
4. Where exactly can an NCM-style idea add value that DFlash2/DSpark do not already capture
   (e.g. confidence/length prediction, re-drafting the tail after the first rejection, multi-candidate
   tree verification, coarse plan carried across blocks, correction of a rejected block)?
5. (If you have web access) Search for the current state of the art, 2026: DFlash / DFlash2, DSpark,
   SpecPilot, JetSpec, EAGLE-3 and successors, block/parallel/diffusion drafters, lossless sampling
   couplings (Gumbel / optimal transport) for block drafts, speculative decoding for hybrid
   linear-attention (DeltaNet) models, anything published on Qwen3.8 speculative decoding. Cite URLs.
   Which published techniques are we missing?

Answer in at most 900 words, numbered by question.
