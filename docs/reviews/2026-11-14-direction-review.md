# Direction review: is a lossless NCM drafter the wrong goal? — 2026-11-14

Question put to gpt-6.1-sol, grok, muse and Qwen 4 Max preview (brief first, answers below).

```
Strategic review, no code. Answer in at most 500 words, numbered, blunt. Do not run commands. You may read docs in /Volumes/Ext16TR0/code/ax-engine-sglang/docs (docs/decisions/2026-11-14-track-a-v2-concept-spine.md, docs/reviews/2026-11-14-ncm-evidence-notes/*.md) and results/track_a_baselines/REPORT.md, and use the web.

The owner's challenge: "People want to try NCM because it may be MUCH faster than an LLM. If the speed is only a little better, no one will try NCM. Are we going in the wrong direction? Is applying NCM to an existing LLM the wrong idea, so that it must be a totally NCM design?"

Facts.
1. NCM (Next-Concept Modeling, arXiv 2610.14162, Hazy Research): multiscale residual VQ-VAE tokenizer + transformer that generates K levels coarse-to-fine, all codes of a level in parallel (8-32 NFE per sequence). It IS the generator; no verification. Paper speed claims are versus diffusion LMs (10-31x). Models are 25M-130M parameters, trained from scratch. Quality: OpenWebText Gen.PPL 83 (data 14.2); WMT14 De-En BLEU 22.6 at 11 NFE vs AR 25.2; XSum ROUGE-L 28.0 at 8 NFE vs AR 24.4. "6-10x faster than strong AR" appears only in an interview, not in a table.
2. Our project so far: keep an existing open LLM (Qwen3.8-27B) and use NCM ideas inside a LOSSLESS speculative-decoding drafter (output identical to the target). Measured on 2x B200, SGLang, greedy, concurrency 1, tok/s (accepted tokens per round): AR 101; built-in MTP7 code 396 (7.07), math 350 (5.65), chat 183 (3.16); public DFlash2 drafter code 489 (7.5 of 8), math 453 (6.2), chat 250 (3.3). One DFlash2 round costs about 15 ms versus 9.8 ms per AR token. At concurrency 32 the drafters lose most of their advantage (chat: DFlash2 0.88x of best MTP).
3. Our current plan ("Concept-Spine") adds a continuous concept vector to the public DSpine drafter (block 16). Reviewers estimated 15-25% chance of +3% accepted length and 10-20% chance of +5% measured tok/s over the best public drafter. Released Qwen3-8B DSpine drafter, our offline measurement: accepted length code 6.4, math 12.1, chat 3.8 of 16.
4. Budget: about 200 GPU-hours on H200/B200 class GPUs, one engineer-equivalent. No ability to pretrain a 27B model from scratch.

Questions.
Q1. Is "NCM as a lossless drafter for an existing LLM" structurally capped at a small gain? Give the ceiling argument with numbers (what bounds tokens per target forward pass).
Q2. What are the realistic ways to get a LARGE speedup (>=3x over today's best public drafter, i.e. roughly >=10-15x over AR) on an LLM of useful quality, and what does each cost in quality and compute? Consider: (a) native NCM student distilled from the LLM, (b) frozen LLM trunk + parallel block/concept decoder without exact verification (lossy or relaxed/typical acceptance), (c) latent/concept reasoning with fewer target passes, (d) anything else credible.
Q3. Which of these is feasible within ~200 GPU-hours as an honest proof of concept, at what model scale, and what would the headline result be? What is the cheapest decisive experiment (<=20 GPU-h) that tells us whether to pursue it?
Q4. Your verdict in one line: continue the lossless drafter, pivot (to what), or stop. Give a probability that the pivot shows >=3x over AR-with-best-drafter at <=2% relative quality loss on some benchmark.
```

---

## codex

1. **Concept-Spine targets an incremental win, not your stated ambition.** Native NCM removes tokenwise target verification; your project retains it. NCM’s published speed comparison is against diffusion models, with small models and substantial quality gaps—not evidence of 10–15× acceleration at Qwen quality. Reusing pretrained weights is sensible; exact output preservation is the constraint. [NCM](https://arxiv.org/abs/2610.14162)

2. **Lossless has a practical ceiling, not a universal small-gain theorem.** Speed is committed tokens divided by round latency. At 15 ms, tripling today’s DFlash2 requires approximately **22 code / 20 math / 11 chat tokens per round**. Block 8 cannot deliver that: even perfect acceptance gives roughly 533–600 tok/s, depending on bonus-token accounting—only 1.09–1.23× today’s code result. Block 16 still cannot reach the code/math targets at that latency. Your 8B DSpine acceptance-only headroom is 2.5× code, 1.32× math, 4.21× chat. Those are optimistic ratios, not speedups. Longer blocks encounter earlier mismatches; branching increases verification work. There is no model-independent bound of eight tokens per target pass.

3. **Native NCM student:** potentially the largest architectural gain because it eliminates verification entirely. Cost: tokenizer reconstruction errors, student capacity loss, distillation error, and substantial training/data generation. Your budget supports roughly a 100–300M research prototype, subject to profiling—not a credible replacement for general 27B capability. A narrow translation/summarization result could be honest; broad quality parity would be fantasy.

4. **Frozen LLM plus parallel decoder:** the most affordable architectural pivot. Start with an existing 8B trunk and train a small block/concept head; jointly adapting a pretrained 1–3B model is another plausible budget-scale experiment. Dropping exact verification permits longer blocks, but errors compound, and processing committed tokens to refresh trunk caches still costs compute. Typical acceptance retains target evaluation and merely relaxes rejection. Neither guarantees large gains: Medusa reports 2.3–3.6× over AR; Fast-dLLM v2 converts pretrained AR weights using approximately 1B training tokens and reports up to 2.5×. Neither establishes your target. [Medusa](https://arxiv.org/abs/2401.10774), [Fast-dLLM v2](https://arxiv.org/abs/2509.26328)

5. **Latent reasoning and other routes:** compressing long reasoning traces could produce large *answer-latency* gains, with task-specific distillation and possible reasoning degradation. Each latent step still costs a forward pass; visible output remains a floor. Coconut supports task-specific promise, not broad parity. Smaller distilled specialists, routing, and retrieval are cheaper alternatives, with coverage/quality tradeoffs. [Coconut](https://arxiv.org/abs/2412.06769)

6. **Spend ≤20 GPU-hours first:** on released 8B DSpine, measure exact decoding, relaxed-acceptance sweeps, and an all-accepted timing ceiling on predeclared XSum examples. Include prefill, cache refresh, and full response latency. Compare against the fastest public drafter on identical hardware. If the ceiling misses 3×, stop that configuration. If quality collapses, stop that acceptance policy. Only then spend the remaining budget training a specialist head. Headline: “8B approximate block decoding: measured latency versus summarization quality.”

7. **Verdict: pivot to narrowly evaluated approximate block decoding; stop Concept-Spine. Subjective probability: 5% of achieving ≥3× over the same model’s best drafter with ≤2% relative quality loss on a predeclared benchmark within 200 GPU-hours.**

---

## grok

I'll read the decision note, the NCM evidence notes, and the baseline report, then answer the four questions from those numbers.I'll count the draft against the 500-word cap and keep the ceiling arithmetic tied to the measured round times.1. Yes. Lossless tok/s is accepted tokens per round over round time. One target forward emits at most one draft block, and only the matching prefix counts. DFlash2 code keeps 7.52 of 8 in 15.4 ms (489 tok/s) versus AR at 9.8 ms/token (101 tok/s): 4.9×.

A perfect block of 16 at that same 15.4 ms is ~1040 tok/s (2.1× DFlash, ~10× AR). Qwen3-8B DSpine accepts 6.4/16 code, 12.1/16 math, 3.8/16 chat, and a wider block makes the round slower. 3× DFlash on code (~1470 tok/s) needs ~23 accepted tokens inside 15 ms. No such block exists.

Chat at 3.3/8 is unused length. Measured concept injection (NCP) added +4% accepted length. Extra coarse-to-fine passes (the v1 cost, 18–28 ms) need accepted length 11–13.5 for a 10% edge over DFlash. At concurrency 32, chat DFlash is 0.88× best MTP.

2. (a) Native NCM as the generator. This is what cuts a sequence from O(L) token steps to 8–32 NFE. Published speed is 10–31× versus diffusion LMs. An outside same-shape AR loop saw ~7–8× at batch 1, and the lead reversed by batch 128. At ≤130M, OWT Gen.PPL is 83 against data 14.2 and AR-sample ~35; WMT BLEU is 22.6 against AR 25.2. Same-size AR still beats ~87M NCM on TinyStories (Gen.PPL ~11 vs ~24) with a 99.8% codec. A 1.2% tokenizer miss on a 152k vocab breaks code. Distilling into it still needs a pretrain-scale codec and will not keep 27B quality.

(b) Frozen trunk with unverified block or typical acceptance. This removes the verify tax. It also leaves the 2% quality bar: chat’s draft disagrees near position 4, so a force-accepted block is drafter text.

(c) Latent reasoning. Coconut ran GSM8K ~3× faster (0.26s → 0.09s) at accuracy 42.9 → 34.1. That misses 2%, and ordinary token decode is unchanged.

(d) A smaller AR model. An 8B is ~3× less weight traffic and can land near 2–3× decode when bandwidth-bound, with a hard-task gap above 2%. SpecPilot reached 10.87× over AR at batch 1 by systems search, and 3.93× over AR when batched.

3. A 100M–1B NCM inside 200 GPU-hours repeats a known loss to same-size AR. The ≤20 GPU-hour test is an oracle, no training: verify the 27B’s own greedy future at blocks 8, 16, and 32, concurrency 1, code/math/chat, tok/s versus DFlash. Stop the drafter if block-16 is under 2× DFlash on code and math. Force-accept one DFlash block on one code set and one math set. A quality drop well past 2% kills lossy block decode.

4. Stop the lossless drafter. Do not pivot to a native NCM on this budget. Probability of ≥3× over AR-plus-best-drafter at ≤2% relative quality loss on some benchmark: ~5%.

---

## muse

1. Q1 — Yes, lossless NCM-drafter is capped at small gain. Lossless speculative decoding yields at most `block+1` tokens per target forward, and in practice `1+sum(prod p_i)` where `p_i` is per-position accept prob. You are already at ceiling: block 8 DFlash2 code 7.5/8, math 6.2, chat 3.3; block 16 DSpine math 12.1/16. Even perfect `p=1` on block 16 gives 16 tokens/round. Round costs ~15ms vs 9.8ms/AR-token, so perfect = 16/15ms ≈1060 tok/s ≈2.2x current DFlash2-code, ~10x AR — and perfect is impossible on high-entropy chat. Realistic +0.3 tokens from a concept vector (NCP measured +4.17%: 5.93→6.18) is +3-5% tok/s. At conc 32 verify dominates and you already lose: chat DFlash2 0.88x best MTP. Concept cannot fix prefix-match + Amdahl.
2. Q2 — >=3x over best drafter (≈10-15x AR) requires leaving losslessness:
  a) Native NCM student distilled from LLM: only true NCM path. 8-32 NFE/sequence vs 64-128 AR. Cost: pretrain-scale. NCM quality today is weak (OWT GenPPL 83 vs data 14.2; WMT BLEU 22.6 vs AR 25.2) at 25-130M. Matching Qwen-27B needs 10k-100k GPU-h. Impossible on 200h.
  b) Frozen trunk + parallel block decoder, lossy/relaxed accept: only credible large gain on budget. Emit 16 tokens per 1-2 trunk passes, accept by top-k/typical/MBR, no exact distribution match. Cost: 1-5% quality loss if sloppy; train 0.5-2B head on 10-30M tokens, ~100-200 GPU-h on 8B.
  c) Latent/concept reasoning: saves thinking tokens only, ~20-50% on math, not 3x. Coconut GSM8K 34.1% vs CoT 42.9% — loses accuracy.
  d) Other: routing/cascade small-model-handles-most + system autotune (SpecPilot claims 4.4x→10.8x AR). Real but not NCM, and chat still bounded.
3. Q3 — Feasible in 200h: only (b) on Qwen3-8B, not 27B. PoC headline: 2-3x over best lossless drafter on code/math (≈6-9x AR greedy) at ≤2-5% HumanEval/GSM8K drop. 27B + ≤2% + 3x-over-drafter is not feasible here. Cheapest decisive test ≤20h, no training: offline on released DSpine-8B drafts, 300 prompts/domain, compute exact-match length vs relaxed length (target top-5 or p>0.1). If relaxed/exact <2x, kill. Second: force-accept full 16 and measure task drop — that is your quality floor. ~5-10 GPU-h on 8B.
4. Q4 — Verdict: stop lossless Concept-Spine, pivot to frozen-8B lossy parallel decoder or stop. Probability that pivot hits ≥3x over AR-with-best-drafter at ≤2% relative loss on any benchmark in 200h: ~8%.


---

## qwen

已讀完決策文件、四份證據筆記與 `track_a_baselines/REPORT.md`。以下數字除標「估」者皆來自實測或文件。

1. **Q1：是，結構性封頂。** 無損投機每輪＝draft＋一次 target 驗證，accepted A≤block B；加速＝A·t_AR/t_round。concurrency 1 下 27B 驗證是 memory-bound，驗 B 個 token≈一個 AR token（9.8 ms）。DFlash2 已 489 tok/s＝4.86x。要再 3x（≈1467 tok/s、≈14.5x AR）需每輪 commit ~15 token 且 draft 近乎免費——等於 drafter 就是 target。逐位置匹配率 p 連乘：code p≈0.94→A 飽和 7–8；chat p≈0.4→A≈3.3（與實測吻合）；DSpine-b16 離線 code 也只 6.4/16。天花板由 target 熵決定，不是架構；NCM coarse-to-fine 不改這個熵，粗碼錯一個就斷在 position 1（v1 已證）。concurrency 32 只剩 0.88x。Concept-Spine 的 +3–5% 是天花板下的灰塵。

2. **Q2：唯一保住 27B 品質又能碰標竿的是 (b)。** 硬體允許（估）：B≤32 的 pass 仍 memory-bound（~10–13 ms），B/12 ms→B=16≈13x、B=24≈20x AR。瓶頸不是硬體，是「一次能 commit 幾個 token 而不崩品質」，而無損正是把它壓小的原因。(a) 原生 NCM student：證據全面負面——ncm-mlx TinyStories Gen.PPL 24–48 vs AR 11–14、87M 仍輸、MASSIVE-de EM 51 vs 57.6、OWT 83 vs data 14.2；≤2% 品質損失基本不可能，「全 NCM」是品質死路。(b) 凍結 trunk＋平行 block head、放寬/typical acceptance：trunk 扛品質，把 memory-bound pass 攤提到 B 個 token，訓練數十 GPU-h，是唯一可行 pivot。(c) latent reasoning 是「減少 token 數」的另一軸，Coconut GSM8K 還掉到 34.1 vs 42.9，非本题。(d) 誠實強基線：蒸餾 27B→3B＋drafter，或 SpecPilot 12-bit 系統增益（10.87x 無損）——NCM 要贏的是這個，不是裸 AR。

3. **Q3：先做 ≤5 GPU-h 的決定性實驗。** 拿已發布 DSpine/DFlash2，關掉驗證、直接 commit draft block（B=8/16/24），量 tok/s＋pass@1/exact-match。純推理、零訓練，直接畫出 lossy 的速度-品質 Pareto：若 B=16 在 code 達 ≥1467 tok/s 且 ≤2% 損失→pivot 成立；若品質崩→連 (b) 也封頂，答案就是「27B 品質＋15x 不可兼得」。PoC（若成立）：凍結 27B trunk、code 域、B=16–24 平行 head＋relaxed acceptance，~50–100 GPU-h；headline 只宣稱 code、concurrency 1。

4. **Q4：停止加碼無損 drafter；先跑無驗證實驗，再決定是否 pivot 到 (b)，不要轉「全 NCM」。** pivot 在「某 benchmark」達 ≥3x-over-best-drafter 且 ≤2% 相對品質損失的機率 ≈15%（code pass@1 是唯一可能；chat/math≈0）。殘酷現實：owner 的前提錯了——保住 27B 品質就沒有 15x；要 15x 就得放品質或換小模型。

（選擇記錄：依既有證據筆記與 REPORT 判斷，未另查 web；B-position pass 成本為 memory-bound 推估，非實測；依偏好以繁中回覆。）