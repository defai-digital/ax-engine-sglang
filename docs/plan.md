# ax-engine-sglang plan (2026-11-13)

This is the original NCM research plan. The project now extends SGLang for
CUDA inference on Linux. [Current architecture and support](README.md)

Based on Claude's draft design plus independent reviews by codex, grok and muse
(`docs/reviews/`). All four agreed on the sequencing below.

## Base model facts (Qwen/Qwen3.6-35B-A3B, Apache-2.0)
- 36B total / ~3B active parameters, 40 layers, hidden 2048.
- Hybrid: 30 Gated-DeltaNet linear-attention layers + 10 full-attention layers (pattern 3:1).
  Full attention: 16 Q / 2 KV heads, head_dim 256, partial RoPE.
- MoE: 256 experts, top-8, plus a shared expert; expert width 512.
- Vocab 248,320; 1 MTP layer ("trained with multi-steps"); 27-layer vision tower (dropped for v1).

## Design decisions
1. **Block-wise NCM**: NCM generates fixed lengths only, so open-ended output is produced in blocks
   of B=32–64 tokens, each refined coarse-to-fine in K≈4 levels.
2. **Last level = Qwen token ids**, predicted by Qwen's LM head. Coarser levels are a plan, so
   decoding adds no reconstruction error (needed for code).
3. **DeltaNet stays causal.** No backward scans. In-block parallelism comes from the full-attention
   layers or a small separate decoder. Accepted tokens are replayed causally to update KV and
   recurrent state.
4. **Track A before Track B**: a small NCM drafter with Qwen as verifier gives output identical to
   Qwen in greedy mode. Stochastic exactness needs care: latent-code probabilities are not token
   probabilities.
5. **Baseline is MTP**, not diffusion models. NCM's published speed-ups were against diffusion only.

## Phases and gates
| Phase | Work | Go / no-go |
|---|---|---|
| 0 | vLLM baseline: AR vs MTP1–3; speed at batch 1–64; HumanEval/MBPP/GSM8K | Numbers recorded (`benchmarks/baselines/`) |
| 1 | Reproduce ~100M NCM from the paper; block tokenizer on Qwen tokens (code/math/chat), B=32/64 | ≥99.95% exact code tokens, or last level = token ids |
| 2 | Track A: 0.3–1B NCM drafter + frozen Qwen verifier | ≥1.5× end-to-end vs best MTP mode at batch 1, verifier-identical output |
| 3 | Track B pilot: adapters on frozen Qwen, ≤2B tokens | ≤2-point HumanEval/MATH regression; exact cache parity in lossless mode |
| 4 | Full fine-tune, start at 5B tokens, self-distilled data | Parity with base Qwen and ≥2× decode in the serving mode |

## Hardware
- Phases 0–2: **1× H200** (bf16 model is ~72 GB).
- Phases 3–4: **8× H200**, single node with NVLink. A full fine-tune needs ~580 GB of states plus activations.
  Rough compute for 30B tokens is ~1.9e21 FLOPs, about 15–28 days at 10–20% MFU (MoE training is
  memory-bound). Use B200 only after a kernel qualification run: FLA issue #913 reports
  shape-dependent NaNs on sm_100 for linear-attention kernels.
- Train in BF16; FP8 only after correctness is established.

## Update 2026-11-14
- Phase 1a (NCM reproduction) passed: `results/phase1_repro/REPORT.md`.
- Phase 2 drafter was redesigned after the first GPU test and four reviews. The NCM-style code plan
  is replaced by a chunk-mode drafter; see `docs/decisions/2026-11-14-drafter-architecture.md`.
  Design decisions 1-2 above (code levels, final level = token ids) now apply only to the plan-mode
  ablation. Decisions 3-5 stand.
- Phase 1b tokenizer training is paused: the chosen drafter does not use a tokenizer, and
  speculative decoding is lossless through the verifier. A partial `b32_lossless` run reached 84%
  code-token reconstruction after 2,500 steps. Lowering the dead-code restart threshold from 1.0 to
  0.05 raised 3,000-step code accuracy from 53% to 66% (plan mode).
- Hardware: one 2x B200 pod for training, baselines and final timing (the earlier H200-first plan
  was dropped once the B200 pod ran vLLM and the drafter without kernel problems).
- Track B (rebuilding the 35B model around code levels) is not supported by the evidence so far.
  It stays gated on Phase 2 and would need a new design.

## Update 2026-11-14 (later): new target and NCM-Draft
- The target is now **Qwen/Qwen3.8-27B** and only the lossless Track A is pursued
  (`docs/decisions/2026-11-14-retarget-qwen38-27b.md`). The 35B work is stopped. The gate is
  >= 1.5x tuned MTP on code and chat at concurrency 1, >= 1.1x the best public drafter, and not
  slower than AR at concurrency 32, all on one engine (SGLang), GPU type and precision.
- NCM was reopened as a bounded, controlled experiment (`docs/decisions/2026-11-14-ncm-draft.md`):
  the earlier negative result used a weak tokenizer, almost no code corruption, few levels and
  2,000 steps. NCM-Draft conditions a code hierarchy on the frozen 27B's hidden states and is
  compared with one-pass and token-space controls. Code: `.internal/research/concept_drafting/`.

## Update 2026-11-14 (evening): NCM-Draft v1 dropped, Track A v2
- Four design reviews, a literature search and first B200 measurements (DFlash2 greedy code 489
  tok/s at accepted length 7.5 of 8; chat 3.2-3.3 for every drafter) led to dropping NCM-Draft v1
  before any GPU run: a VQ hierarchy does not fit exact-prefix acceptance and seven passes cost too
  much. Estimated chance of an end-to-end win was about 10%.
- Track A v2 (`docs/decisions/2026-11-14-track-a-v2-concept-spine.md`) fine-tunes the open DSpine
  drafter and adds the next-concept idea as a gated continuous vector, the only form with a
  published positive result (NCP-ArchPreview, +4.17% accepted length). Ideas are validated at small
  scale (Qwen3-8B, then a 2-3M token 27B pilot) before any large run.
- The gate is restated: >= 5% tok/s over the best tuned public drafter on code and math at
  concurrency 1 in the served mode. The earlier 1.5x-MTP and 3x-MTP targets are not planning goals.
