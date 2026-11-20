# Project review, 2026-11-13 to 2026-11-17

Scope: everything in this repository up to commit 733c4d3 plus this review. All speed numbers were
measured end to end on one NVIDIA B200 with SGLang 0.5.23 unless stated; ratios compare servers on
the same GPU in the same session. Details and raw data are linked per item.

## 1. Summary

- The original goal, converting Qwen3.6-35B-A3B into a Next-Concept Model, was reproduced
  (NCM paper numbers match) and then **stopped on measured evidence**: a 4.0-6.6x ceiling before
  quality loss, an 8.8% HumanEval loss from the tokenizer, and an MoE verify step too cheap for an
  external drafter.
- The project moved to **lossless tree speculative decoding with a frozen target** (Track L). The
  result is the SGLang plugin `ax-engine-sglang`.
- **Qwen3.8-27B (done):** 1.09x-1.42x over stock SGLang DFlash2 at one request, 1.01x-1.21x with
  2-64 requests on NVFP4; 1.07x-1.72x over TensorRT-LLM DFlash2 at one request.
- **Qwen3.8-Flash-Next (MTP done):** found and fixed a correctness bug in stock SGLang's sparse
  attention (QSA) and lifted its 4-token verify cap; MTP 7 is 1.22x-1.30x faster than stock MTP 3 on
  code.

## 2. Timeline

| Date | Work | Outcome | Record |
|---|---|---|---|
| 11-13 | Phase 0 harness, AR and MTP baselines | used by every later run | `benchmarks/baselines/` |
| 11-13/14 | NCM reproduction | matches the paper | `results/phase1_repro/REPORT.md` |
| 11-14 | NCM tokenizer and drafter on the 35B | stopped | `results/phase2_ablation/REPORT.md` |
| 11-14 | Retarget to Qwen3.8-27B; Tracks A, B, C, M, N | closed | `docs/results-history.md` sections 2-4 |
| 11-14/15 | Track L tree on 9B and 27B, one request | main line | `results/track_l_l0/REPORT.md` |
| 11-15 | Multi-user width policy, Stage 0 | approved, pinned widths pass | `docs/decisions/2026-11-15-multiuser-width-policy.md` |
| 11-16 | Two attention backends, NVFP4, sampling | done | `docs/decisions/2026-11-16-dual-attention-backend.md` |
| 11-17 | TensorRT-LLM comparison and chat c16 root cause | done | same, section 17 |
| 11-17 | Qwen3.8-Flash-Next feasibility, QSA ring fix, MTP sweep | MTP done | `results/flash_next/REPORT.md` |

## 3. Achievements

### 3.1 Qwen3.8-27B

| Comparison | 1 request | Several requests |
|---|---|---|
| vs. stock SGLang DFlash2, NVFP4 | code 1.42x, math 1.33x, chat 1.14x | 1.01x-1.21x up to 64 requests |
| vs. stock SGLang DFlash2, bf16 | 1.09x-1.15x (faster stock block); up to 1.49x vs. block 8 | 0.93x-1.09x up to 16 requests |
| vs. TensorRT-LLM 1.4 DFlash2 | bf16 1.07x-1.40x, NVFP4 1.32x-1.72x | code and NVFP4 math lead; chat c16 0.81x-0.90x |

Correctness evidence: token identity across plugin builds and SGLang 0.5.22 / 0.5.23 (192 / 192),
fp32 teacher audit at the stock mismatch rate (near-ties only), stress runs with aborts, and a
distribution test showing the sampled tree verify is lossless in bf16 (0 / 28 failures).

### 3.2 Qwen3.8-Flash-Next

- **Upstream bug found:** stock SGLang 0.5.23 overwrites the members of a QSA index-key group
  within one verify forward whenever the window crosses a group boundary; 44-66% of groups were
  corrupt on 6,000-token contexts with MTP 1-3.
- **Fix (`AX_QSA_RING=16`):** ring of 16 slots per request, CUDA-graph metadata kernels included;
  verify windows up to 13 tokens; zero cost at MTP 3.
- **Speed:** MTP 7 vs. stock MTP 3, repeated within 1.5%: code 1.22x-1.30x, math 1.05x-1.15x,
  chat 0.80x-0.86x.

### 3.3 Engineering

- A plugin that builds its worker from the installed SGLang and refuses to start when an edit no
  longer matches (tested on 0.5.22 and 0.5.23); no copy of SGLang code is shipped.
- Measurement tooling reused across models: paired benchmarks, token identity, fp32 teacher,
  sampling distribution tests, round profilers, TensorRT-LLM runner.
- Twelve design decisions recorded with evidence, each reviewed by several independent reviewers
  (codex, grok, muse, kimi, GLM) before GPU time was spent.
- Upstream defects documented: SGLang DFlash2 sampled decoding is not lossless (bf16, 7 / 28 tests
  fail); QSA ring overwrite (reported upstream: sgl-project/sglang#51009); Qwen4-Exp exposes no aux hidden states.

## 4. Open items and limits

| Item | Status | Next step | Estimate |
|---|---|---|---|
| 27B chat with 16 requests trails TensorRT-LLM (0.81x-0.90x) | cause located at engine level: SGLang's batch-16 DFlash2 round is 14-20% slower | kernel-level profile of both engines at batch 16 | 2-4 GPU hours |
| 27B bf16 math c8 / code c8 width 8 below the 0.98x gate | per-request rate equals stock; wall time longer (scheduling) | per-request timelines are recorded; analyse | under 1 GPU hour |
| Flash-Next chat slower with wide MTP | SGLang 0.5.23 has `--speculative-adaptive` (per batch-size bucket) | test it with candidates [1, 3, 5, 7] and ring 16; per-request depth if needed | 1-2 GPU hours, then code |
| Flash-Next DFlash2 and tree | blocked: no aux hidden states; PLE n-gram context; tree-aware QSA | engine work in SGLang's Qwen4-Exp model | days |
| QSA check residue (0.3-0.4% of groups, ring-independent) | unexplained | instrument the forward that produces it | under 1 GPU hour |
| Task accuracy | not measured for the tree or for Flash-Next MTP | HumanEval / GSM8K pass rates | 1-2 GPU hours per model |
| NVFP4 sampled losslessness | not definable beyond "as the verify forward computes it" | none planned | - |

## 5. Lessons

1. Measure the ceiling before training: NCM's ceiling and the tokenizer loss were cheap to measure
   and decided the direction.
2. Compare against the strongest public baseline on the same GPU and engine, and against another
   engine (TensorRT-LLM) to separate method gains from engine gains.
3. Keep plugin patches scoped to the plugin's own mode; a global patch silently cost stock MTP its
   multi-request CUDA graphs.
4. Engine invariants can fail under loads they were meant to allow: the QSA cap of 4 tokens did
   not prevent the overwrite it was meant to prevent.
5. Repeat speed measurements before stating a ratio, and report where the method loses (chat with
   many requests, chat with wide MTP).

## 6. Data and artefacts

- Code and records: this repository (`ax-engine-sglang/`, `benchmarks/sglang/`, `.internal/research/concept_drafting/`, `docs/`).
- Raw results: `results/` locally (git-ignored, about 740 MB) and `/workspace/ax-ncm/results` on the
  pod volume (historical path from before the rebrand); reports in `results/*/REPORT.md`.
- Models: Hugging Face (Qwen/Qwen3.8-27B, RadixArk/Qwen3.8-27B-NVFP4,
  RadixArk/Qwen3.8-Flash-Next-NVFP4, incoai/Qwen3.8-27B-DFlash2); pod copies in
  `/workspace/hf-cache-qwen38` and `/workspace/hf-flash`.
