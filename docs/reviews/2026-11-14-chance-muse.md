# Winning chance and design review: muse

## 1. Probabilities, with arithmetic

Round economics from your numbers: AR token 9.8 ms; DFlash2 code-greedy round ≈ 17 ms for ≈ 8.3 committed tokens (7.5 accepted + bonus), giving 489 tok/s. Your NCM round costs 7 sequential draft passes plus one verify(32). Even at ~1 ms per 140M pass, verify(32) on a dense 27B will be ≥12–18 ms, so T_round ≈ 22–28 ms. To beat 489 tok/s you need (A+1)/0.025 ≥ 489, i.e. **accepted length A ≥ 10–13 just to tie on code**. On chat (A ≈ 3.2 for everyone), 4.2 tokens per 20+ ms round ≈ 210 tok/s — slower than DFlash2's 250. That is the whole game: block 32 buys headroom only if survival stays high past position 10, while verify(32) taxes every round including instant rejections.

| Line | Yours | Mine |
|---|---|---|
| Offline accept beats DFlash2, greedy code, any variant | 55–65% | ~50% — but hollow: DFlash2 is capped at 8, so A > 7.5 of 32 proves little |
| NCM ≥10% over its own controls | 20–30% | 10–15% |
| End-to-end ≥10% over DFlash2/DSpark, greedy code/math | 25–30% | 8–12%, and likelier via `direct` + truncation than via NCM |
| Same, sampled + thinking | 15–20% | 4–7% |
| Same, chat | 10–15% | 2–4% |
| 3× MTP | ~5% / <1% | Agree: ~2–5% code (needs A ≈ 13+ at 22 ms — no), <1% chat |

Your offline-vs-e2e gap is the tell: Gate 3's "projected draft cost" lets a cap-artifact offline win masquerade as progress. Measure e2e or nothing.

## 2. Design correctness

(a) **The VQ hierarchy is the wrong use of NCM.** Expected accepted length is Σ_j S(j), dominated by early survival; an error in the global level-0 code poisons position 1 and the other 31 predictions are discarded. Coarse codes summarize the unpredictable tail to predict the predictable head — backwards. Sudoku's "correction" works because of global constraints; open-ended text has none. Your own doc states this risk and hand-waves it with corruption/dropout; the toy result (teacher-forced 0.95 → free-running 0.31) is the evidence winning the argument. Note the exposure-bias asymmetry your code comment gets half-wrong: token-space `chunks` on gold tokens is matched under greedy acceptance, but **code-space conditioning is never matched** — your own codes differ from gold codes even when tokens agree. Also Gate 1 (≥99% reconstruction) extrapolates from a 66–84% baseline with no bridging evidence.

(b) Fixed 32 with no adaptive length is strictly dominated: you pay verify(32) when position 2 rejects. DSpark-style scheduling and DSpine's dynamic verification already solve this; you omitted it.

(c) 7 sequential passes reload weights and cross-attend 7× at batch 1, where launches dominate. DSpine propagates position dependencies through network depth inside one forward pass — same motivation, ~7× less draft latency.

(d) 140M/d=1024 squeezing 5×5120-dim taps through a 1024 bottleneck, predicting 32 tokens + 63 codes, versus DFlash2's ~2B/d=5120 predicting 8. Capacity is your binding constraint, not draft FLOPs.

(e) Greedy non-thinking training for a sampled+thinking default is a distribution double-miss: thinking traces are OOD, and greedy tokens teach mode-seeking while sampled acceptance needs distribution matching (accept ∝ min(1, q/p)). No KL/logits distillation, no thinking data — sampled was lost at the data section.

(f) Chat: entropy wall at A ≈ 3.2. The fix is cheaper drafts + shorter adaptive blocks + tree on early positions — the opposite of your design in all three axes.

## 3. What to build instead

Bet the budget on the line 3/4 of your own reviewers picked. **Base: [DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16)** (5-layer d=5120, block 16, SGLang-native, Apache-2.0), DFlash2 as fallback. Keep the tapped backbone; add (i) per-position confidence head → adaptive verify truncation, (ii) top-2 tree on positions 1–3 (verify width ≤ 20), (iii) one gated token-space refinement pass that reuses the verifier's rejected-position logits (two-level verification). Train with SpecForge/DeepSpec: 50–100M tokens mixed greedy + sampled (T=1, top-p 0.95, top-k 20), thinking on/off, code/math/chat; KL against cached top-64 target logits; stage 2 on-policy (EAGLE-2 style). Ablate MTP-head-for-token-1.

**Falsifier ≤ 6 GPU-h:** no training first — offline oracle on 2k bench blocks: confidence truncation + top-2 tree at positions 1–2 on the frozen checkpoint (~2 h with target forwards). Then a 500-step fine-tune probe (~4 h). Kill the line if oracle headroom < 8% tokens/round *and* extrapolated fine-tune lift < 2%. Either bar passing green-lights the full run.

## 4. Where NCM-style ideas could add value, ranked

1. **One refinement pass after rejection, in token space.** The verifier's logits at the rejected position are free, exact supervision — re-draft the tail from the corrected prefix within the same round. This is the NCM "correction" idea minus VQ, and the only one I'd fund first.
2. **Coarse block embedding as an accept-length/confidence predictor** for the verify scheduler — a use where semantic coarseness helps instead of poisoning position 1.
3. **Coarse plan carried across blocks** (block-embedding memory) for structured code — testable last, likely ~zero.
4. Tree stems, tail re-drafting: no — standard trees and the next round already do these; VQ adds latency, not information.

## 5. State of the art you are missing

Block drafters: [DFlash](https://en.papernotes.org/ICML2026/image_generation/dflash_block_diffusion_for_flash_speculative_decoding/) (ICML'26, one-pass block diffusion), [DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), [DSpark explainer](https://www.sevydevy.in/learn/dspark-speculative-decoding-explained-block-drafting) (confidence head + verify scheduler — the adaptive length you lack), [vLLM Aug'26 measurements](https://www.aloka.tech/a/1k7geg/choosing-a-speculative-decoding-method-in-vllm-eagle-3-dflash-dspark-and-tuning-n) (DFlash ≤2.87×, EAGLE-3 below baseline — EAGLE is dead for this target). Trees/verification: [JetSpec](https://arxiv.org/html/2606.18394v2) (parallel tree drafting), [SpecPilot](https://arxiv.org/html/2611.02501v1) (tree + two-level verification — screen before building), [Sequoia](https://www.emergentmind.com/papers/2402.12374) (Gumbel-top-k trees), SpecTr/OPT-Tree (optimal-transport block/tree acceptance). Couplings: per-token spec sampling is already optimal (1−TV, see [this explainer](https://astralord.github.io/posts/speculative-decoding-the-art-of-guessing-well/)); block-level gains need multi-draft couplings ([MC-SJD](https://arxiv.org/pdf/2510.24211)). **Biggest miss: [DSpine](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16)** (Tencent, [arXiv 2609.36173](https://www.explainx.ai/blog/tencent-dspine-speculative-decoding-drafter-qwen3-8b-27-percent-longer-acceptance-vs-dflash-2026)) — block 16, in-depth dependency propagation, dynamic verification, ReplaySSM, claims above DFlash2 (e.g. [104.8 vs 58.2 tok/s](https://alphasignal.ai/news/tencent-s-dspine-pushes-qwen3-8-27b-to-104-8-tok-s-on-a-single-gpu) in one community test). It is your block-32 thesis executed better. Also unaddressed: DeltaNet/SSM state replay on rejection (KV tricks don't transfer; DSpine ships ReplaySSM) — a correctness risk for any SGLang integration you write.
