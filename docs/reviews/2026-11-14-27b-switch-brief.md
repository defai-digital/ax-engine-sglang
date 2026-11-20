# Review request: should ax-engine-sglang switch its target from Qwen3.6-35B-A3B to Qwen3.8-27B?

You are an independent reviewer. Be critical, quantitative and short (under 700 words). Do not edit files.
Repo context (read if useful): docs/plan.md, docs/decisions/2026-11-14-drafter-architecture.md,
phase2/README.md, phase2/drafter.py, results/phase2_ablation/REPORT.md.

## Project
Goal: generate text faster than the target model's built-in MTP speculative decoding, with output
identical to the target in greedy mode (lossless speculative decoding). Gate: >= 1.5x the best MTP
mode at batch 1 on code and chat. The owner hopes for 3x.

Current drafter ("chunk mode"): Qwen3.5-0.8B causal encoder over the context (hidden 1024, 24 layers,
same tokenizer) -> 6-layer decoder (d=1024) that drafts 16 tokens in 5 parallel passes over chunks
[1,1,2,4,8], prefix-weighted loss, queries initialised from the encoder's last state. Trained on the
target's own greedy outputs (self-distillation). Evidence so far: 2.15 free-running accepted tokens
per round after only 2,000 steps on stand-in data (0.8B as verifier). Full 80k-step training against
the 35B has not started; 35B distillation (150k prompts x 3 domains) is about 40% done on 2x B200.

## Current target: Qwen3.6-35B-A3B
MoE, 36B total / ~3B active per token, 40 layers, hidden 2048, 256 experts top-8, 1 MTP layer.
35B MTP numbers are not measured yet (Phase 0 runs in a few hours). Concern: the verifier step is
cheap (3B active), so a 0.8B encoder + 5 decoder passes is maybe 30-50% of one verifier step, while
the MTP head is nearly free and reads the target's own hidden state.

## Proposed target: Qwen/Qwen3.8-27B (Apache-2.0, released 2026-11)
Facts checked today from the HF config and model card:
- Dense, 27B, 64 layers (48 Gated DeltaNet + 16 full attention, 3:1), hidden 5120, intermediate 17408,
  vocab 248,320 (same tokenizer family as Qwen3.5-0.8B), `mtp_num_hidden_layers: 1`,
  "MTP: trained with multiple steps". Thinking mode on by default.
- So one verifier step costs roughly 9x the 35B-A3B's active compute; a 0.8B drafter is ~3% of it.

The owner's premise is "MTP mode is basically not working in this model". Public measurements do NOT
support that for Qwen3.8-27B:
- vLLM recipe: MTP acceptance 0.77 (FP8), 0.79-0.90 (NVFP4) at k=3 on 2x RTX 5090.
- llama.cpp community table (~90 rows): typical +33% to +145% decode speed over AR; code acceptance
  ~0.87-0.90, prose ~0.23-0.48; per-position acceptance 0.85/0.69/0.57/0.47; k=2-4 is the sweet spot,
  k=8 is worse. On prose some setups gain ~0% .
- DGX Spark study (vLLM, concurrency 1): no MTP 4.5 / 7.9 / 9.7 tok/s vs MTP k=3 9.9 / 17.1 / 18.5
  (BF16 / FP8 / NVFP4), i.e. 1.9-2.2x.
- External block drafters already exist for this model: RadixArk DSpark (1.36B, 7 tokens per round)
  36.6 tok/s and z-lab DFlash2 (2B, 8 draft tokens) 47.9 tok/s at concurrency 1 on SGLang NVFP4,
  versus 18.5 tok/s for vLLM+MTP (engine and method are confounded). That is about 2.0x and 2.6x MTP.
(It is possible the owner means MTP is weak on prose/chat, or weak on the 35B-A3B; say which reading
your answer assumes.)

## Questions
1. Does switching the target to Qwen3.8-27B materially raise the chance of (a) >= 1.5x MTP and
   (b) >= 3x MTP at batch 1? Give rough probabilities for 35B-A3B vs 27B and the arithmetic
   (tokens per round, draft cost, verify cost) behind them.
2. Given DSpark / DFlash2 already report ~2-2.6x MTP on the 27B, what is left for ax-engine-sglang to
   contribute? Should we benchmark against them, adopt their design (e.g. conditioning on the
   target's hidden states instead of a separate 0.8B encoder), or keep the chunk drafter?
3. What should change in our drafter for a 27B dense target (block length, chunk schedule, use of
   target hidden states / KV, drafter size, thinking-mode data)?
4. Costs and risks of switching now: distillation throughput on 2x B200 for a dense 27B, wasted 35B
   distillation, thinking-mode default, vLLM integration, anything else.
5. Recommendation: (A) stay on 35B-A3B, (B) switch to 27B now, (C) finish the cheap 35B screen
   already queued (about 1 day) and run 27B in parallel/afterwards, or (D) something else.
   One paragraph, decisive.
