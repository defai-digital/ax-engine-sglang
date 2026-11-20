# Review request: best practices and corrections for the Track A v2 plan

You are one of three reviewers (Gemini 4 Hydrogen, gpt-6.1-sol, Qwen 4 Max). Be concrete and blunt.
Everything you need is in this message; you may also search the web and cite URLs. Do not edit files.

## Situation
Lossless speculative decoding for Qwen/Qwen3.8-27B (dense, 48 Gated DeltaNet + 16 attention layers,
hidden 5120, vocab 248k, thinking on by default). Hardware: 2x B200 (183 GB), about 200 GPU-hours.
Measured on one B200 with SGLang 0.5.23 (tok/s at concurrency 1; accepted length in brackets):
AR 102; DFlash2 greedy non-thinking code 489 (7.5 of 8), math 453 (6.2), chat 250 (3.3);
DFlash2 sampled+thinking code 310 (4.1), math 366 (5.0), chat 242 (3.2); DSpark sampled+thinking
code 266 (4.1), math 310 (4.7), short-output set code 448 (7.0), math 360 (5.1), chat 195 (2.7);
MTP3 about 250 (3.1, smoke). At concurrency 32 the speedup over AR falls to 1.0-2.2x.
Our first design (7-pass VQ coarse-to-fine drafter, "NCM-Draft v1") was rejected by four reviewers.
Published evidence we rely on: NCP-ArchPreview arXiv 2609.10715 section 5.3 (gated continuous concept
injection into a DFlash2-like drafter: accepted length +4.17%, HumanEval +7.59%, 0.04M params, drafter
1.1B, trained on 5B tokens x 10 epochs); DSpine arXiv 2609.36173 (adjacent causal injection, block 16,
plugin for SGLang 0.5.22, weights tencent/Qwen3.8-27B-DSpine-b16 and tencent/Qwen3-8B-DSpine-b16,
training in SpecForge); SpecOPD arXiv 2611.02846 (on-policy rollouts); D-REAL arXiv 2611.10432
(acceptance-length objective); NCM arXiv 2610.14162 (tokenizer needs K=8-12 levels, 30k-120k updates
at batch 2048-4096 to reach 99%+).
The owner's instruction: validate the thinking at small scale first, then plan and implement the
larger run. They want the project to gain from Next-Concept-Modeling ideas.

## The plan under review

The draft of `docs/decisions/2026-11-14-track-a-v2-concept-spine.md` (before the corrections these reviews led to).

## Questions
1. Best practices for fine-tuning a released block drafter (DSpine/DFlash2) on new target-generated
   data: learning rate, warmup, which parameters to train, data volume needed for a measurable
   change, anchors per sequence, loss weighting over positions, how to avoid regressing the released
   weights, how to cache target hidden states versus running the target online. Cite sources.
2. Concept source for a target with no concept module: is C1a (pooled last-chunk hidden states) or
   C1b (predicted next-block coarse latent) better supported? Is there a better third option (e.g.
   the target's own MTP head hidden state, a look-ahead probe of the anchor hidden state, a
   product-quantised concept as in NCP)? What exactly should the small-scale experiment be so that
   a positive result is not a capacity or seed artefact, and how many held-out tokens are needed to
   detect +3% in accepted length?
3. Is Qwen3-8B the right small-scale testbed, or should the small test be a small data slice on the
   27B itself (the 8B is a plain transformer without DeltaNet layers; the 27B is the product)?
   Which findings transfer and which do not?
4. Stage 0 block sweep and stop rule: correct? What is missing (e.g. profiling draft/verify split,
   CUDA-graph tiers, calibration of the verify-length policy)?
5. Correct the plan: list what to cut, what to add, what to reorder, and the corrected gates and
   budget. State the single most likely way this plan still fails.
6. Your probability that the corrected plan yields a measured end-to-end gain >= 5% over the best
   public drafter on B200 at concurrency 1 (code/math), and >= 3% accepted-length gain from the
   concept injection specifically.

Answer in at most 900 words, numbered by question.
