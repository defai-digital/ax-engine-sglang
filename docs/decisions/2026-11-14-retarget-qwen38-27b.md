# Decision: retarget to Qwen3.8-27B, lossless Track A only (2026-11-14)

## What changed
- **Target model: Qwen/Qwen3.8-27B** (dense, 64 layers = 16 x (3 Gated DeltaNet + 1 full attention),
  hidden 5120, vocab 248,320, 1 MTP layer, Apache-2.0). The Qwen3.6-35B-A3B target is dropped by the
  project owner. All 35B jobs were stopped and the 35B weights deleted from the pod. About 180k
  partial 35B responses are kept only as a control set.
- **Track A only**: a lossless speculative drafter with the 27B as verifier. Track B (a block-emitting
  27B) is deferred.
- Qwen3.8-Flash-Next was considered and rejected: MoE with 6B active parameters and a 4B MTP head
  (the same cheap-verifier problem as the 35B, with a stronger MTP), about 360 GB in bf16, a new
  `qwen4_exp_text` architecture, and a non-Apache license.

## Why the 35B was a poor target
Speed is roughly accepted tokens per round / (draft time + verify time). With ~3B active parameters
the 35B verify step is cheap, so an external drafter costs an estimated 30-50% of one step, while the
MTP head is nearly free and reads the target's own hidden state. The 27B is dense, so one verify
step is far more expensive and a drafter is a small fraction of it. The 35B MTP baseline was never
measured; this is a design argument, not a result.

## Public numbers on the 27B (not ours; one H200, concurrency 1, temperature 1.0, thinking on)
| Task | AR | MTP k=7 | DSpark | DFlash2 | DFlash2 / MTP |
|---|---|---|---|---|---|
| GSM8K | 68.9 | 178.5 | 185.3 | 236.1 | 1.32 |
| HumanEval | 69.0 | 151.9 | 159.9 | 214.6 | 1.41 |
| MBPP | 69.0 | 153.1 | 163.3 | 226.9 | 1.48 |
| MT-Bench | 68.9 | 134.9 | 137.6 | 184.0 | 1.36 |

tok/s, from the DFlash2 model card. The DSpark card reports DSpark v2 at 1.37-1.65x MTP on a
different stack. MTP on this model is not weak: it is already 2.0-2.6x AR.

## Reviews
Four reviewers (codex gpt-6.1-sol, grok, muse, Qwen 4 Max) answered
`docs/reviews/2026-11-14-27b-design-brief.md`; their answers are next to it.

Agreed by all four:
- Drop the separate 0.8B encoder; condition the drafter on hidden states tapped from the target.
- No VQ codes on the critical path. What remains of NCM is scale-wise refinement in token space.
- At most one refinement pass after the parallel pass, kept only if it pays for its latency.
- Run the frozen target online during training; caching tapped states costs about 51 KB per token.
- SGLang is the benchmark and integration engine (DSpark and DFlash are native there).
- First step: reproduce AR, tuned MTP, DSpark v2 and DFlash2 on our own hardware.
- Track B is deferred.

Ceiling for lossless decoding against MTP at concurrency 1: grok 1.5x (code, chat) to 1.8x (math),
muse 1.3-1.45x, Qwen 4 Max 1.5-1.85x, codex 1.5-1.8x with about 2x as a stretch. codex notes that
3x is unproven rather than impossible and points to tree and two-level verification work
(SpecPilot, JetSpec++) that should be screened.

Disagreements:
| Topic | codex | grok | muse | Qwen 4 Max |
|---|---|---|---|---|
| Drafter size | keep DFlash2 (~2B, d=5120) | ~120M, d=1024 | 1.2-1.6B, d=2048 | ~2B, d=5120 |
| Starting point | fine-tune DFlash2 with SpecForge | fork the trainer, new small model | fine-tune DFlash2 | fine-tune DFlash2 |
| Adaptive draft length | yes | threshold sweep only | low value | main contribution |
| First data size | 10M tokens | 100M | 30-80M | 1B |
| MTP head for token 1 | no, use the verifier bonus token | yes | yes | yes |

## Decision
1. **Gate (replaces the 35B gate).** Same engine, GPU and precision: >= 1.5x tuned MTP on code and
   chat at concurrency 1, >= 1.1x the strongest reproduced public drafter, and not slower than AR at
   concurrency 32. 3x MTP is not a target for the lossless track.
2. **Start from DFlash2** (Apache-2.0) and the SpecForge trainer rather than from
   `.internal/research/ncm/drafting/drafter.py`. Three of four reviewers chose this, and it gives a measured baseline on day
   one. The Phase 2 simulator, verify-cost timer and benchmark harness are reused; the chunk drafter
   is kept as an ablation only.
3. **Drafter size is decided by measurement.** Step 1 measures draft and verify cost on B200 at
   concurrency 1 and 32. A narrow drafter (grok's ~120M, muse's d=2048) becomes the main line only if
   the 2B drafter's cost is what limits concurrency-32 speed.
4. **Candidate improvements, each kept only for a measured end-to-end gain of >= 5%:** fine-tuning
   on the target's own sampled and greedy outputs (thinking and non-thinking), calibrated adaptive
   draft width, tree verification on the first positions, one gated refinement pass, and on-policy
   training.
5. **Data:** start at 10-30M target tokens and grow towards 100M while held-out accepted length
   still improves.

## Plan and gates
| Step | Work | Go / no-go |
|---|---|---|
| 1 | SGLang on B200: AR, MTP k sweep, DSpark v2, DFlash2 at concurrency 1 and 32 on code, math and chat; verify cost by width; greedy parity | Published ratios reproduced within ~10%, or the harness is debugged first |
| 2 | Fine-tune DFlash2 on 10-30M self-distilled tokens; offline free-running accepted length | >= 10% over unchanged DFlash2, else change the method before spending more |
| 3 | Ablate adaptive width, tree verification, refinement pass, drafter width | Keep changes worth >= 5% end to end |
| 4 | Full training and end-to-end measurement in SGLang | The gate in decision 1 |

## Naming
The result is a block drafter with scale-wise token refinement. It is NCM-inspired; it is not the
paper's NCM, whose multiscale VQ codes lost accuracy when self-predicted
(`results/phase2_ablation/REPORT.md`).
