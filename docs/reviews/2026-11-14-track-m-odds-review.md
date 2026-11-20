# Track M odds review (2026-11-14)

Question: after Tracks B, C and N closed and M0b reached 97.9/98.3/99.5% token reconstruction
(chat/code/math), what are the chances of the NCM conversion of Qwen3.5-9B? Same prompt to four
reviewers (below); answers verbatim after it.

## Summary

| Item | Lead | codex | grok | muse | kimi |
|---|---|---|---|---|---|
| M0d >= 99.5% token acc in all domains | 45-55% | 40% | 40% | 30% | 30-40% |
| M1 within 5% quality at span 64 | 10-15% | 8% | 5% | 5% | 3-6% |
| >= 1.5x over lossless DFlash at <= 5% loss | ~5% | 3% | 2% | 1-2% | <1% |
| Overall success within 2000 GPU-h | 3-8% | 2% | 3% | 1-2% | 1-3% |

Points all four make:
- Span 64 gives 64/12 = 5.3 tokens per sequential 9B forward, below the measured DFlash
  7.3-10.8 tokens per round at no quality loss, so span 64 cannot beat DFlash; only longer spans
  could, at higher quality risk.
- 99.5% token accuracy is the wrong M0 gate: 0.995^64 = 73% exact blocks at best, and GSM8K /
  HumanEval fail on one wrong number or identifier. Use an encode/decode round trip of the 9B's own
  answers re-scored on GSM8K and HumanEval (<= 1-2% relative loss) as the primary gate.
- Biggest M1 risk: free-running error propagation from coarse levels; teacher-forced code accuracy
  will look better than generation. Kill tests: the round trip (~1-2 GPU-h, no training), then a
  10-30 GPU-h M1 pilot measured on free-running task quality with an oracle-code control.
- Better odds for >= 3x over plain AR: lossless speculative decoding (public DFlash, EAGLE-3 style
  drafters, tuning) - codex ~60%; muse ~25% for relaxed (synonym-neighbourhood) acceptance.

grok adds that the paper's matching WMT14 tokenizer reconstructs at 98.79%.

## Prompt

You are an independent reviewer for an ML research project. Give calibrated probability estimates. Be skeptical and concrete. Do not run code; answer from the facts below plus your knowledge of the literature.

## Goal
Convert Qwen (PoC: Qwen/Qwen3.5-9B, later Qwen3.8-27B) into a Next-Concept Model (NCM, HazyResearch, arXiv 2610.14162): a coarse-to-fine generator that emits a 64-token span (later 256) as 11 hierarchical levels of VQ codes (levels of sizes 1,4,8,14,20,26,32,40,48,56,64, codebook 4096), then a small decoder maps codes to tokens. The user's success criterion: "people want NCM because it may be much faster than an LLM; if it is only a little faster, no one will try it." Lossy parallel decoding is acceptable, but quality must stay close (target: within 5% of the 9B on GSM8K and HumanEval). Budget ceiling: 2000 GPU-hours (B200/H200), ~a few hundred already spent.

## Baselines (measured)
- Qwen3.8-27B on H200, greedy, concurrency 1, tok/s (accept length): built-in MTP-7: code 272.9 (6.99), math 242.7, chat 138.0. Public DFlash drafter: 351.2 (7.41)/323.5/178.0. DSpine: 511.5 (12.40)/363.5/154.6. All lossless.
- Qwen3.5-9B + public DFlash drafter (HF harness, exact): HumanEval 10.83 tokens/round, GSM8K ~7.3.

## Tracks already closed (measured)
- Track B (relaxed acceptance with a DSpine-b16 drafter on Qwen3-8B): best within 2% quality = 1.25x GSM8K, 1.05x HumanEval over exact. Unverified blocks collapse quality.
- Track C (fine-tune trunk LoRA r64 + KL/agreement so the 9B is easier to draft; 1 epoch, ~18M tokens): val accept 6.42 (public) -> 6.469 (drafter only) / 6.484 (LoRA) = +1%. Closed.
- Track N (NCM-style block decoder b32, 6 levels, trained as a drafter on 9B features, H200): free-running accept length 0.19 / 0.37 / 0.53 at steps 1k/2k/3k; gate was >= 6.4. Closed.

## Track M (current): the 9B itself becomes the NCM generator
- M0 = span tokenizer (official NCM VQ-VAE code, WMT14 config: width 384, 0 encoder / 1 decoder block, learned level-0 cascade, nested dropout 0.5, quant_resi 0.5), 64-token spans, Qwen 248k vocab. Gate: full-hierarchy token reconstruction >= 99.5% on held-out 9B answers per domain.
  - M0 (learned 248k tables, 9B answers + 150M public tokens): step 12.5k token acc chat 96.2 / code 96.9 / math 99.3. ~88% of errors were tokens seen < 100 times in training (unseen ~99% wrong).
  - M0b (frozen Qwen input embedding + lm_head behind learned projections): step 10k chat 97.9 / code 98.3 / math 99.5; block-exact (all 64 tokens right) 38% / 52% / 80%. Unseen-token error 38-68%; residual errors semantically close (litres->gallons, Iowa->Alaska).
  - M0d (M0b + 1.7B tokens public code/math/web/chat): running, 30k steps; step 2.5k matches M0b.
- M1 plan: LoRA/full-FT of the 9B to predict the 11 levels of the next span autoregressively over levels (context via KV cache), tokenizer decoder to tokens; ~30-40 GPU-h pilot; gate: quality within 5% on GSM8K/HumanEval. Theoretical speed ~ span/(levels+1) = 64/12 = 5.3 tokens per 9B forward; M2 = span 256 for larger gains.
- Proposed kill test: ~20-30 GPU-h M1 pilot; stop Track M if quality drops > 15-20%.

## The lead researcher's own estimates
M0d passes 99.5% in all domains: 45-55% (>= 99%: ~85%). M1 within 5% quality at span 64: 10-15%. Beating lossless DFlash by >= 1.5x wall-clock at <= 5% quality loss: ~5%. Overall project success within 2000 GPU-h: 3-8%.

## Questions
1. Your probability for each of the four items above, with one-line reasoning each. Say where you disagree with the lead's estimates and why.
2. The single biggest risk in M1, and the cheapest experiment (GPU-h) that would most reduce uncertainty about it.
3. Is the 99.5% token gate the right M0 gate for M1, or should it be something else (e.g. block-exact rate, task-level quality after encode/decode round trip)?
4. Any design change with a materially better chance of a >= 3x wall-clock speedup over plain AR at <= 5% quality loss on a 9B within the remaining budget (cite prior work)? Keep the whole answer under 600 words.

## codex

I interpret “within 5%” as **relative score loss on each benchmark**. These are subjective probabilities from today, within the remaining budget; they are overlapping events, not factors to multiply.

| Outcome | My probability | Reason / disagreement with lead |
|---|---:|---|
| M0d reaches ≥99.5% in all domains | **40%** | Slightly lower: more data should improve coverage, but chat requires eliminating roughly three quarters of M0b’s remaining errors, including systematic substitutions. |
| M1, span 64, preserves quality within 5% | **8%** | Below 10–15%: reconstruction does not establish that Qwen can learn to generate the hierarchy while retaining reasoning and code dependencies. |
| ≥1.5× DFlash wall-clock speed, ≤5% quality loss | **3%** | Below 5%: quality and latency must succeed together; 11 sequential backbone evaluations leave limited headroom against an effective parallel drafter. |
| Overall NCM project success within 2,000 GPU-h | **2%** | Below 3–8%: representation, adaptation, and systems risks compound; moving to 256 tokens adds uncertainty alongside potential speed. |

The [NCM paper](https://arxiv.org/html/2610.14162v1) demonstrates smaller-scale generation, translation, and summarization against diffusion baselines—not successful conversion of an instruction model with preserved GSM8K/HumanEval performance. That limits how strongly its results support this project.

**The biggest M1 risk is failure to transfer Qwen’s sequential computation into parallel hierarchy prediction.** An early mistaken code can constrain many later decisions; excellent reconstruction with oracle codes does not test this.

I would spend **20–30 aggregate GPU-h** on a diagnostic M1 pilot using a frozen tokenizer, comparing decoded outputs under oracle preceding levels, generated preceding levels, and complete multi-span rollouts. Measure task scores and the additional damage caused by generating early levels. Include a small training-set overfit check: failure even there points toward implementation or optimization problems. A >20% early deficit justifies stopping the current recipe, but does not distinguish insufficient training from an architectural limitation.

**Replace the primary M0 gate with task-level round-trip preservation.** Encode/decode complete held-out baseline answers, then rerun GSM8K grading and HumanEval execution. Require roughly **≤1–2% relative degradation**, reserving most of the 5% allowance for generation. Also test Qwen continuations from reconstructed prefixes.

Keep token accuracy, block-exact rate, and identifier/number/operator fidelity as diagnostics. Under independent errors, 99.5% token accuracy gives only approximately **73% exact 64-token blocks**; actual error clustering changes this. Neither token accuracy nor block-exact rate alone establishes task preservation.

**For ≥3× over plain AR, exact speculation has materially better odds.** Start with the existing public 9B DFlash drafter and optimize block length, verification, caching, and kernels before retraining. [DFlash](https://arxiv.org/abs/2602.06036) reports >6× acceleration in tested settings; [EAGLE-3](https://arxiv.org/abs/2503.01840) reports up to 6.5×. These are precedents, not predictions for this hardware/model. Given your acceptance measurements, I assign **about 60%** to achieving ≥3× on both target tasks within budget.

For a replacement generator, token-space block diffusion preserves more pretrained structure. However, [Fast-dLLM v2](https://nvlabs.github.io/Fast-dLLM/v2/) reports only up to 2.5× after approximately 1B adaptation tokens—insufficient evidence for your threshold.

Finally, the proposed hierarchy contains **313 code positions**. If processed by the backbone, their compute and cache costs matter: **64/12 counts sequential stages, not wall-clock speedup**. Measure that before committing to M2.

## grok

I'll ground the estimates in the NCM paper and the closest prior results on hierarchical codes and parallel decoding, then answer the four questions directly.**M0d hits 99.5% in every domain: 40%.** Math is already at the line. Chat and code still have to clear a long tail on a 248k vocabulary, and the paper’s matching WMT setup (width 384, 0 encoder / 1 decoder block, 11 levels, codebook 4096, span 64) reconstructs at 98.79%. All domains at or above 99%: about 70%. The lead’s 45–55% and ~85% are a little high.

**M1 stays within 5% of the 9B on GSM8K and HumanEval at span 64: 5%.** The lead’s 10–15% is high. At M0b’s 98.3% code-token accuracy, block-exact reconstruction is only 52%, so the frozen tokenizer already inserts about one error per 64 code tokens. HumanEval fails on one bad identifier. The published NCM generators are ~100M models trained from scratch; nothing there is a LoRA’d 9B, and the 30–40 GPU-hour pilot is far below their multi-billion-token generator budget.

**At least 1.5× lossless DFlash at no more than 5% quality loss: 2%.** Span 64 costs 12 sequential 9B forwards per 64 tokens (5.3 tokens per forward). On this 9B, DFlash already commits 10.8 tokens per round on HumanEval and ~7.3 on GSM8K, with one target forward plus a small drafter. Eleven extra 9B weight loads cannot beat that by 1.5×. The lead’s ~5% treats 5.3× against plain autoregressive (AR) decoding as if it were a win over DFlash.

**Project success inside 2000 GPU-hours: 3%.** That sits at the floor of the lead’s 3–8%. Tracks B, C, and N already removed the cheap ways to get a large speedup while holding quality.

The main M1 risk is that the oracle round trip spends the quality budget before any generator is trained. Fine levels are predicted independently given coarser codes, so a wrong coarse code is not refinable. Figure 3 of the NCM paper is still garbage at mid-depth, and the paper notes that one token error breaks code. The cheap test is about 1–2 GPU-hours and no training: encode and decode the 9B’s own correct GSM8K and HumanEval strings and rescore. If HumanEval moves by more than about 10 points, or GSM8K by more than about 5, stop. For another ~2 GPU-hours, compare teacher-forced and free-running level decoding on ~200 spans.

The 99.5% token gate is the wrong M0 gate. It can pass while most code blocks are still inexact. The gate that matches M1 is the task round trip above, with headroom of about 2 points left for the generator, plus free-running block-exact rate. Per-level code accuracy is the right diagnostic. Token accuracy only tracks the rare-token tail.

Keep the 9B frozen and verify against it. Train a small next-scale drafter on its traces for a few hundred GPU-hours, and accept only the prefix the 9B itself would have emitted (Leviathan et al.). Quality loss is then zero, and a 3× gain over plain AR is just the accept length. DFlash and EAGLE-3 already clear 3× losslessly. Track N at 0.53 accept after 3k steps is an under-trained drafter. A lossy 9B that emits all 11 levels has a worse shot. If the goal is to beat DFlash, use parallel coarse levels only as a sketch and let a short MTP cleanup by the 9B rewrite low-margin positions. Consistency-LM Jacobi distillation (Kou et al., 2024) is the precedent for turning an AR model into a few parallel steps with a small quality drop. VAR next-scale prediction tolerates perceptual error. This vocabulary does not.

## muse

1. Calibrated probabilities (lead in parens):

- **M0d ≥99.5% token-acc all domains: 30%** (45-55%; ≥99%: 65% vs 85%). Chat/code need 4x error cut (97.9/98.3→99.5) with a width-384 1-block decoder; errors are Zipf-tail, and block-exact is only 38%/52% at 98% token-acc. Disagree high: 11x data helps means, not tail composition.
- **M1 within 5% on GSM8K/HumanEval at span-64: 5%** (10-15%). Requires learning a new 313-code output distribution while preserving exact reasoning, after Track C gave +1% from 18M-token FT and Track N got 0.53/6.4 accept. Diffusion LLMs (LLaDA-8B, Seed Diffusion) still trail AR >5% after far more training. Lead 2-3x high.
- **≥1.5x wall-clock over lossless DFlash at ≤5% loss: 1-2%** (5%). Arithmetic: M1 = 64 tokens / 11 sequential 9B forwards = 5.3 tok/forward; DFlash already 7.3-10.8 tok per 1 target forward + cheap drafts. 11 expensive forwards cannot beat 1. Lead misses this.
- **Overall success in 2000 GPU-h: 1-2%** (3-8%). Conjunction of above plus M2 span-256 retrain. Agree direction, lower level.

2. **Biggest M1 risk:** next-span coarse codes (L0-L1: 5 codes must commit 64 tokens of reasoning) are unpredictable from current KV-cache; exact math/code dependencies cannot be decided coarsely. **Cheapest test (~2-5 GPU-h):** freeze 9B, train linear probe or LoRA-r8 to predict oracle L0+L1 teacher-forced on GSM8K/code traces. If top-1 near unigram or code-PPL barely moves, kill M1 before the 20-30h pilot.

3. **No, 99.5% token-avg is the wrong gate.** One wrong number/identifier fails GSM8K/HumanEval; M0b proves it: 98% token-acc yet 38-52% block-exact, with fatal "close" errors (litres→gallons). Use: (a) block-exact ≥85%/domain, (b) oracle encode→decode round-trip task loss ≤1-2%, (c) code predictability (probe accuracy), not just reconstructability.

4. **Yes — drop VQ, keep AR verifier.** Fine-tune 4 MTP heads (Gloeckle et al. 2024) or distill a 0.5B EAGLE-3 drafter on self-traces (EAGLE-3, Li et al. 2025; DistillSpec, Zhou et al. 2024), verify with embedding-neighborhood relaxed acceptance (Lantern, Lee et al. 2025; lossy/typical acceptance, Hu et al. 2024, Anagnostidis et al. 2023). Cost <200 GPU-h, proven 2.5-4x over AR lossless; 5% budget buys ≥3x. P(~25%) vs NCM ~1-2%, because it relaxes only synonyms under an exact verifier. Note lossless DFlash at 7-11 tok/round may already be ≥3x over plain AR.


## kimi

### 1. Probability estimates

- **M0d reaches ≥99.5% token accuracy in every domain: 30–40%.** The additional data should reduce tail-token errors, but moving chat/code from roughly 98% to 99.5% requires a 3–4× error reduction; unseen-token errors remain structurally difficult. **The lead’s 45–55% is optimistic.**
- **M1 stays within 5% task-quality loss at span 64: 3–6%.** Track N’s extremely low acceptance is direct evidence that pretrained Qwen features do not readily support accurate block generation; autoregressive level errors will compound within each span. **The lead’s 10–15% is too high.**
- **Beating lossless DFlash by ≥1.5× wall-clock at ≤5% loss: <1%.** Span 64 provides only **5.3 tokens per full-model forward**, below DFlash’s measured 7.3–10.83 tokens/round, before counting NCM decoding overhead. Span 64 therefore appears structurally unable to meet this target. **The lead’s ~5% is substantially too high.**
- **Overall success within 2000 GPU-hours: 1–3%.** Success requires both a difficult generative conversion and a speed result that the present span-64 design cannot plausibly deliver. **I agree with the lead’s pessimism, but would use the lower end or below.**

### 2. Biggest M1 risk and cheapest informative test

The biggest risk is **free-running conditional error propagation**: an early coarse-level mistake selects the wrong subsequent codes, and teacher-forced code accuracy can look healthy while complete spans—and consequently reasoning or programs—diverge. This is more fundamental than tokenizer reconstruction.

Run a **10–15 GPU-hour, span-64 LoRA pilot**, then evaluate actual free-running generation on roughly 200 GSM8K problems and the full HumanEval set. Measure:

- task quality and wall-clock speed;
- exact-code and exact-token span rates;
- first failing hierarchy level;
- an **oracle-code decoding control**, separating tokenizer failures from generator failures.

Do not rely on teacher-forced validation accuracy for the kill decision. If quality drops by more than 15–20%, stopping is justified; continuing to full fine-tuning is unlikely to repair a large free-running gap cheaply.

### 3. Is 99.5% token accuracy the right M0 gate?

**No.** Even independent errors at 99.5% token accuracy imply only \(0.995^{64}\approx72.6\%\) exact blocks. It also overweights frequent tokens while potentially missing rare but decisive identifiers, numbers, and operators.

Use **task-level encode/decode round-trip quality on genuine 9B answers** as the primary representation gate, requiring negligible degradation. Keep block-exact rate and errors on critical tokens as diagnostics. Block-exactness alone is too strict because semantically harmless substitutions exist; neither round-trip metric predicts M1’s free-running quality.

### 4. Better route to ≥3× speedup

**Prioritize lossless speculative decoding, not lossy NCM generation.** Exact verification preserves the target model’s output by construction (Leviathan et al., 2023; Chen et al., 2023). The measured DFlash/DSpine results already make this the strongest route; invest in longer or dynamically sized drafts, draft-tree optimization as in **EAGLE-2**, and serving/kernel optimization.

If NCM research must continue, use it only as a **proposal generator behind exact Qwen verification**. Track N suggests that variant currently has poor prospects, whereas lossless DFlash has both empirical acceptance lengths and no quality-loss risk.

