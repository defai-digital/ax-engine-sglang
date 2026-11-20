# Track C: draftable trunk (joint trunk adaptation + block decoder) — 2026-11-14

## Why

Track B showed that relaxing the acceptance rule of an existing drafter gives 1.05-1.25x tokens per
round at unchanged quality and that unverified blocks collapse accuracy
(`results/track_b_decisive/REPORT.md`). With a frozen trunk the only lever left is the number of
tokens committed per trunk pass, and that is bounded by how predictable the trunk's own
continuation is (chat: about 3 tokens per round with every public drafter).

The owner raised the budget to about 2000 GPU-hours and proposed Qwen3.5-9B as the development
scale. Four reviewers (`docs/reviews/2026-11-14-budget2000-review.md`) agree on one route and on
gating it early:

- Route R2, "draftable trunk": adapt the trunk (LoRA) jointly with a parallel block decoder so the
  trunk's continuation becomes predictable in blocks, and verify drafts against the adapted trunk
  (the verification pass is free: the trunk has to read the committed block anyway). Quality is
  then a task-metric comparison with the original model, not token identity.
- Their probability that this reaches 3x measured tok/s over the best public drafter at <= 2%
  relative loss on 27B: 5% (Qwen), 8-10% (muse), 10% (codex), 12% (grok). No published result is
  within 3x of that bar; the closest (Medusa-2, CLLM, Fast-dLLM v2) report 2.3-3.6x over AR.
- Bigger lossless drafter 1-2%, native NCM student <= 1%, latent reasoning 1-3% (wrong metric).
- All four: spend at most about 100 GPU-hours at the small scale and stop on a miss.

Decision: run Track C on Qwen3.5-9B up to the gate below. Nothing is spent on 27B before the 9B
gate passes.

## Why Qwen3.5-9B

Same family as the 27B target (`qwen3_5`: hybrid linear/full attention, vocabulary 248320, built-in
MTP), Apache-2.0, and a public block-16 drafter exists (`z-lab/Qwen3.5-9B-DFlash`, Apache-2.0,
1.29B parameters) that serves as baseline and as initialisation. Qwen3-8B, used in Track B, is a
different architecture with a different vocabulary.

## System

- Trunk T' = Qwen3.5-9B + LoRA; teacher T = the original weights (frozen).
- Block decoder D = DFlash-style drafter reading T' hidden states (initialised from the public
  drafter).
- Decoding: D proposes a block, T' reads it in one pass and commits the prefix it agrees with
  (exact, or ratio 0.1-0.3 as in Track B) plus its own next token. Output = T' greedy decoding.
- Speed = committed tokens per round / measured round time. Quality = task metrics of the T'
  system against T greedy decoding.

## Training signal

1. Block loss: D's cross-entropy on the next block given T' states, back-propagated into the LoRA
   (the boundary state is trained to carry the plan of the next block).
2. Preservation: KL(T || T') on next-token distributions over the same sequences.
3. From stage C2: agreement loss on T' own rollouts (T' is pushed towards D's proposal only where
   T rates the proposal at least r times its best token), corrupted/self-generated prefixes for D
   with a decaying teacher-forcing rate, block schedule 16 -> 32.

Data: T greedy responses (thinking off) to a code / math / chat prompt mix, disjoint from the
evaluation prompts.

Main failure mode named by every reviewer: acceptance rises because the adapted trunk becomes
generic (lower-entropy boilerplate), so acceptance alone is never a success metric.

## Stages, budget and gates (all thresholds fixed before the runs)

2000 GPU-hours is the owner's ceiling, not a target: each stage budget below is an upper bound, and
the aim is to reach a go / stop answer at 9B with as little of it as possible.

Evaluation set at 9B: HumanEval (164), GSM8K (300), 200 held-out chat prompts. Baseline B0 = T with
the public drafter under exact acceptance. "x B0" = committed tokens per round relative to B0, mean
of the three domains. Chat quality: blind pairwise judgement by Qwen3.8-27B of T' output against T
output (order randomised); T' must not lose more than it wins by over 4 points of the prompts.

| Stage | Budget (GPU-h) | Content | Continue if | Stop if |
|---|---|---|---|---|
| C0 | 5 | B0 baseline; harness for the qwen3_5 trunk and DFlash drafter; data generation | harness reproduces AR token for token under `exact` | - |
| C1 | 25 | premise test, equal data and steps: A0 drafter-only fine-tune vs A1 drafter + LoRA (losses 1+2); `exact` in this harness equals greedy decoding up to bf16 near-ties (margins below 0.15 logits) | A1 >= 1.3x B0 and >= 1.15x A0, task accuracy within 2% relative | A1 < 1.15x A0 (the trunk adaptation adds nothing) |
| C2 | 70 | scale data (100-200M tokens), loss 3, block 32, one optional refinement pass | >= 1.8x B0 on average, chat >= 2x B0 chat and >= 6 tokens per round, every quality metric >= 98% of T, measured wall-clock >= 1.5x B0 in the same harness | average < 1.3x B0, or chat < 4.5 tokens per round, or any quality metric below 98% before 1.5x |
| C3 | 250 | 27B pilot (LoRA + decoder, 200-300M tokens) | >= 2x DFlash2 tokens per round on code and math at <= 2% loss, chat >= 2x | otherwise |
| C4 | 1100 | 27B full run, serving integration, measured benchmark against DFlash2 / DSpine on the same GPU | - | - |
| reserve | 550 | reruns, evaluation | | |

A result between "continue" and "stop" is reported to the owner with the numbers; it is not
silently promoted.

GPU-hours are logged per stage in `results/track_c/BUDGET.md`.

## Stage 0 (Track A) closure

The H200 baseline lane is stopped after MTP7: DSpine, DFlash2 and the best built-in MTP on the same
GPU are what the comparison needs; MTP5, DSpark and AR exist on B200
(`results/track_a_baselines/REPORT.md`).
