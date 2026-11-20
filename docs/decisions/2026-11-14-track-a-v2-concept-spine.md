# Decision 2026-11-14: Track A v2 — concept-conditioned one-pass drafter ("Concept-Spine")

Status: **accepted** after review by Gemini 4 Hydrogen, gpt-6.1-sol and Qwen 4 Max
(`docs/reviews/2026-11-14-v2-plan-*.md`). Supersedes `2026-11-14-ncm-draft.md` (NCM-Draft v1),
which is kept only as a record.

## Why v1 is dropped

Evidence in `docs/reviews/2026-11-14-chance-*.md` and `2026-11-14-ncm-lessons-*.md`:

- Only the exact-match prefix counts in lossless drafting. A coarse VQ code summarises the whole
  32-token block, so an error in it breaks position 1.
- Seven drafter passes plus a 32-token verify cost 18-28 ms per round against 15.3 ms for DFlash2
  (measured on B200); accepted length would have to reach 11-13.5 to be 10% faster.
- Published drafters that win use one parallel pass, some causal conditioning between positions
  (DSpark, DSpine, JetSpec) and a dynamic verify length. None uses a VQ hierarchy.
- The only measured "concept + drafter" result (NCP-ArchPreview, arXiv 2609.10715 §5.3) injects a
  *continuous* concept vector of the last completed chunk through zero-initialised gates into a
  one-pass drafter: mean accepted length 5.933 -> 6.180 (+4.17%; HumanEval +7.59%), 0.04M parameters.
  That target was pretrained with a concept module; Qwen3.8 was not.
- Our tokenizer (66-84%) was shallow and undertrained next to the NCM recipe (K = 8-12 levels for
  48-128 tokens, 30k-120k updates at batch 2048-4096, 98.79% on a 151k vocabulary).
- Local check: with codes always dropped, the v1 NCM path should match the one-pass control but
  reached token accuracy 0.06 against 0.66. That path has an unresolved bug and is not used.

## What v2 is

Start from the strongest open one-pass drafter and add the next-concept idea in continuous form.

1. **Base**: `tencent/Qwen3.8-27B-DSpine-b16` (Apache-2.0; 5 layers, block 16, adjacent causal
   injection, dynamic verification, ReplaySSM). Serving: its SGLang plugin (`sglang-dspine`,
   SGLang 0.5.22). Training: SpecForge online mode (target hidden states streamed from SGLang;
   caching five 5120-wide taps costs about 51 KB per token, 1.5 TB per 30M tokens, so no bulk cache).
2. **Data in the served mode**: the target's own outputs, sampled + thinking (T=1.0, top-p 0.95,
   top-k 20) and greedy; code, math, chat, tool/agent style. Benchmark prompts stay out of training
   and out of model selection. 20-30% general-domain replay to protect the released weights.
3. **Fine-tuning recipe** (the released recipe is for training from scratch: AdamW, peak LR 6e-4,
   4% warmup, clip 1.0, 512 anchors per sequence, position decay gamma 7, CE + teacher L1 +
   refinement + alignment terms). For continuation: fresh optimizer, LR sweep 1e-5 / 3e-5 / 1e-4 on
   the drafter, 1e-4 to 3e-4 on new zero-gated modules, 3-5% warmup, loss coefficients unchanged,
   correct-predecessor curriculum off, early stopping on held-out accepted length, original
   checkpoint kept. Hard-label CE alone can lower sampled acceptance (D-REAL), so the teacher-
   distribution term stays on.
4. **Concept variants** (zero-gated additive injection `h += tanh(g_l) * RMSNorm_l(c)` at every
   draft position of every drafter layer; committed states only, fixed token chunks):
   - **C1a last-chunk concept**: mean pooling, then learned-query pooling, of the fused target
     hidden states over the last completed chunk.
   - **C1c predicted next-chunk concept**: a small MLP on the anchor state predicts the mean-pooled
     fused target state of the *next* chunk; trained with a stop-gradient auxiliary loss, and the
     prediction is what gets injected in training and inference. This is the "next concept" of NCM
     and NCP without a codebook.
   - Ablation: the target's built-in MTP head state as a free concept proxy, if its cost is small.
5. **Controls** for every concept run, same data order, seed and budget:
   - fine-tuned drafter without any addition;
   - equally sized anchor-conditioned adapter without concept supervision (capacity control);
   - wrong-content concept (the concept of position anchor // 2 of the same sequence: causal and
     independent of batching; a random permutation would leak later states to earlier anchors);
   - zero-gate parity test: with gates at zero the output is bitwise the control's.
6. **Verify length**: tune the existing SPS / survival policies of DSpine and DSpark for B200 per
   workload and concurrency before adding anything. A concept-informed survival head (C2) is tried
   only if G2 passes.
7. **Acceptance-oriented objective** (SpecOPD on-policy rollouts, D-REAL) is an optional later
   experiment applied to control and variant alike.

Cut: the VQ tokenizer and codes on the drafting path, the post-rejection repair pass (C3), block
retraining in Stage 0, any claim about three times MTP.

## Order of work

1. **Stage 0, no training (10 GPU-h, 27B)**
   - Pin versions: every baseline that enters a comparison is measured on the same SGLang version.
   - Add DSpine-b16; complete MTP2/3/5/7.
   - On released weights sweep the verify cap / dynamic verification, not the trained block size.
     Modes: greedy and sampled + thinking; code, math, chat; concurrency 1, 8, 32.
   - Profile one round: draft backbone, vocabulary projection, verification, ReplaySSM rollback,
     scheduling; record CUDA-graph padding tiers.
   - Choose by committed tokens per total round time; confirm with repeated end-to-end runs.
2. **Stage S, small scale (20 GPU-h, capped)** on Qwen/Qwen3-8B with
   `tencent/Qwen3-8B-DSpine-b16`: a debugging and idea filter, not a proxy for the 27B.
   - S0 export/import round trip and zero-gate parity. S1 reproduce the released drafter.
   - S2 control fine-tune on 1-2M target tokens with the LR sweep. S3 C1a and C1c against all
     controls, three paired seeds. Evaluation: free-running accepted length under the served
     verification policy, >= 300 held-out prompts per domain, bootstrap clustered by prompt.
   - Transfers to the 27B: gate behaviour, leakage checks, objective and recipe mechanics, whether
     a predicted concept beats a plain adapter. Does not transfer: acceptance magnitude, chunk
     size, verify/rollback cost on DeltaNet layers, serving gains.
3. **27B pilot (15 GPU-h)**: best concept variant and matched control on 2-3M target tokens, two
   seeds. Runs even if the 8B result is null, once, at this budget.
4. **27B scale-up**: data (25), control (25), winner (40), serving and statistics (30), reserve (35).

## Gates

| Gate | Criterion |
|---|---|
| G0 | Checkpoint export/import preserves outputs; zero-gate parity; byte-identical greedy output against AR; released DSpine reproduced on the pinned version |
| G1 | Fine-tuned control within a 1% non-inferiority margin of the released drafter in every domain and mode |
| G2 | Concept variant >= +3% accepted length over the control **and** over the capacity and shuffled controls; three paired seeds; prompt-clustered 95% interval above zero; on the 8B and confirmed on the 27B pilot |
| G3 | SGLang on B200, concurrency 1: >= +5% tok/s on both code and math in the served mode over the best tuned public drafter (DSpine, DFlash2, DSpark), interval above zero; chat and concurrency 8 regress by at most 1% |

Ideas that fail their gate are recorded and dropped, not scaled.

## Expectation

Reviewer estimates for the corrected plan: 25-35% for G3, 15-35% for a concept-specific gain of 3%
on the 27B. The most likely failure is redundancy: DSpine's five target taps and adjacent
injection may already carry what a pooled or predicted concept adds. SpecOPD measured only +2.83%
throughput at concurrency 1 on a 27B target despite a much larger acceptance gain.
