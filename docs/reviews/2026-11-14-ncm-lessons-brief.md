# Research request: what can we learn from existing Next-Concept Modeling (NCM) implementations?

Use web search. Cite a URL for every factual claim; say "not found" rather than guess. Separate
what a source measured from what you infer.

## Context
We are building a lossless speculative-decoding drafter for Qwen/Qwen3.8-27B and want to use ideas
from Next-Concept Modeling (arXiv 2610.14162, Stanford Hazy Research; code github.com/HazyResearch/ncm,
checkpoints huggingface.co/hazyresearch/ncm-checkpoints). Our first NCM attempts went badly and we
do not want to keep coding blind:
- Our block tokenizer (multiscale residual VQ-VAE, block 32, levels 4/8/16/32 or 1/2/4/8/16/32,
  8192 codes per level, d=512, 2+2 layers, tied 248k-token Qwen vocabulary, nested dropout) reached
  only 66-84% token reconstruction on Qwen tokens (code/math/web/chat); we need >=99%.
- Our coarse-to-fine drafter (codes then tokens, 7 passes) has no GPU result yet; a token-space
  coarse-to-fine variant had teacher-forced accuracy 0.95 but free-running 0.31 on a toy task.
- Reviewers argue a VQ hierarchy cannot help a lossless drafter because only the exact-match
  prefix counts and an error in the coarsest code breaks position 1.
The repository is at /Volumes/Ext16TR0/code/ax-engine-sglang (read-only), our tokenizer is `phase1/vqvae.py`,
the NCM source is cloned at /private/tmp/claude-502/-Volumes-Ext16TR0-code-automatosx/b1fa9bed-7128-4a15-9f2e-a27580b78cf0/scratchpad/ncm-src .

## Questions
1. NCM paper and official code: the exact tokenizer recipe and its reconstruction accuracy per
   dataset (vocabulary size, sequence/block length, number of levels and level sizes, codebook
   size, model width/depth, training steps, tricks such as codebook restarts, EMA, entropy loss,
   nested-dropout distribution). Which ablations did they report (levels K, corruption rate, level
   weighting, block size, codebook size) and what were the numbers? What do they list as
   limitations? Compare with `phase1/vqvae.py` and name every difference that could explain our
   66-84%.
2. Independent implementations, reproductions, forks, GitHub issues/discussions, blog posts,
   X/Reddit/HN threads, OpenReview comments on NCM: what did people actually measure, what failed,
   what fixes worked? Any on large-vocabulary (100k+) tokenizers, long documents, code, or
   pretrained-LLM token spaces?
3. Related lines we can borrow results from: NCP-ArchPreview "Next Concept Prediction"
   (arXiv 2609.10715), Meta Large Concept Models, VAR / next-scale prediction and its text
   variants, residual-VQ text autoencoders, latent/continuous-plan language models, hierarchical
   or coarse-to-fine drafters in speculative decoding (e.g. Hierarchical Speculative Decoding
   arXiv 2601.05724, DSpine arXiv 2609.36173, JetSpec++ arXiv 2610.19792, SpecPilot
   arXiv 2611.02501, SpecOPD arXiv 2611.02846, D-REAL arXiv 2611.10432). For each: the experiment,
   the number, and the lesson for us.
4. Has anyone used NCM or any discrete/continuous concept hierarchy as a *draft model verified by
   an autoregressive LLM*? Results?
5. Given the evidence, what is the best-supported way to get a measurable gain from an NCM idea in
   a lossless drafter, and which of our choices are contradicted by published results? Rank at most
   five concrete changes by evidence strength.

Answer in at most 1000 words, numbered by question, with a final table: change | evidence (URL) |
expected effect | cost.
