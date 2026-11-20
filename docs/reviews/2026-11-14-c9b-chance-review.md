# Odds of the Qwen3.5-9B experiment (Track C, stage C1) — 2026-11-14

Brief: `docs/reviews/2026-11-14-c9b-chance-brief.md`. Answers of gpt-6.1-sol (codex), grok, muse, Qwen 4 Max preview and Kimi, unedited apart from removed tool preambles.

| Reviewer | C1 passes | C2 passes (unconditional) | 27B bar (unconditional) | 9B result worth scaling |
|---|---|---|---|---|
| codex | 20% | 4% | 0.2% | 10% |
| grok | 12% | 4% | 1% | 4% |
| muse | 15-20% | 2-3% | 0.3-0.5% | 2-4% |
| Qwen | 25% | 7% | 2-3% | 5-8% |
| Kimi | 25% | 7.5% | 1.5% | 8% |

Shared points: the most likely C1 outcome is A0 about 1.05-1.08x B0 and A1 about 1.06-1.09x A0 (both speed gates missed, quality kept); C1 tests a weaker claim than the route itself because the block loss reaches the trunk only through the tap features and nothing trains the adapted trunk to agree with the drafter on its own rollouts; chat is the wall. The single change all five ask for before C2: an agreement loss that reaches the adapted trunk's next-token logits directly, trained on rollouts of the adapted trunk and drafter with corrupted prefixes, keeping the KL anchor to the original trunk.

---

## codex

1. **C1 pass: 20%**; plausible range **10–35%**. These are subjective forecasts, not fitted probabilities. Assuming B0 chat = 3.5, B0 averages **7.21 tokens/pass**; C1 requires **9.37**. My point forecast, code/math/chat:
   - **A0:** 11.1 / 7.8 / 4.1 → **7.67**, **1.06× B0**; unchanged trunk accuracy.
   - **A1:** 11.5 / 8.2 / 4.6 → **8.10**, **1.06× A0**, approximately **1% relative accuracy loss**.
   - **A1k:** smaller gains, approximately **0–0.5% relative accuracy loss**.

   There is a mechanism: future-token supervision can reshape trunk features to expose information the small drafter cannot otherwise extract. DFlash already benefits from target features ([paper](https://arxiv.org/html/2602.06036v2)). But your loss rewards predicting **T’s recorded tokens**, not directly agreeing with **T′’s evolving decisions**. Expect modest representation improvement before a dramatic draftability breakthrough.

2. **C2 pass: 20% conditional on C1; 4% unconditional**, assuming gated progression. With the illustrative B0 above, C2 needs roughly **13.0 tokens/pass overall and 7.0 on chat**. That is substantial improvement over an already trained drafter, within **100 GPU-hours**, while preserving every quality metric. Block 32 supplies headroom; it does not supply prediction accuracy.

3. **Final 27B bar: 5% conditional on C2; 0.2% unconditional—about 1 in 500.** C2 demonstrates **1.8× tokens/pass**; the final requirement is **3× measured throughput over the best public drafter**. Larger verification blocks, drafting cost, and serving overhead consume acceptance gains. DFlash’s own speedup analysis explicitly includes those costs ([paper](https://arxiv.org/html/2602.06036v2)). Scaling and the competing baseline add further uncertainty.

4. **Both false negatives and false positives are credible.** Tap-only gradients are legitimate, but can miss useful changes to T′’s output decisions. KL **1 versus 10** means little without loss normalization and gradient magnitudes: either can freeze adaptation or permit drift. **18M unique tokens** can support adaptation; two epochs do not establish saturation. Exact verification preserves **T′**, leaving drift from T entirely to quality evaluation. bf16 near-ties can change greedy trajectories. If quality uses only **51 GSM8K items**, one answer moves accuracy **1.96 percentage points**—too coarse for a convincing 2% relative claim. **Single change:** make the planned on-policy, differentiable prefix-agreement loss reach **T′’s logits directly**, retaining the KL anchor.

5. **Overall: 10% odds of evidence warranting a 27B pilot; best fallback: a better frozen-trunk drafter with measured throughput gains and a credible quality–speed frontier.**

---

## grok

**1.** P(C1 passes) = **12%**. Most likely: A0 = **1.04× B0** (code **11.0**, math **7.6**, chat **3.8**; mean **7.5** vs B0 mean **7.2** from 10.83 / 7.3 / 3.6). A1 = **1.06× A0** and **1.10× B0** (11.2 / 8.1 / 4.4, mean **7.9**). Accuracy: HumanEval ~**99%** of T, GSM8K ~**97–98%** of T. A1k (KL 10) sits on A0, mean ~**7.6**. The only honest mechanism is LoRA reshaping the tap states so the same argmax is easier to draft. Medusa-2’s joint fine-tune added about **5–25%** over a frozen backbone, with a direct head loss. Here the block gradient reaches the trunk only through a 1.29B drafter already fitted to frozen features, γ=7 gives position 15 weight e^{−15/7} ≈ **0.12**, and 18M tokens × 2 epochs will not rewrite chat. Code is already **10.83 of 16**, so 1.3× (mean ~**9.4**) needs chat to go ~**3.6 → ~7**. A bigger A1−A0 gap is the trunk getting duller: KL weight 1.0 is added to a block cross-entropy of order 1 while KL starts near 0.

**2.** P(C2 | C1) = **20%**. P(C2) = **4%**, counting the gray zone between your stop and continue lines. C2 wants mean ≥ **13** tokens/pass and chat ≥ ~**7**, at ≥98% quality. More teacher-forced tokens move acceptance like a log. Block 32 does nothing until the first 16 are reliable. Chat is the miss.

**3.** P(27B bar | C2) = **15%**. Unconditional **1%**. Published Qwen3.5-27B DFlash, block 16, thinking on, is already accept **8.9** and **6.2×** autoregressive on HumanEval. 1.8× tokens/pass is about **1.5–1.8×** tok/s after a longer verify, not 3× against that drafter. Acceptance length does not jump another 1.7× because the trunk is 27B. The old 5–12% R2 odds assumed a stronger recipe than this C1.

**4.** You will get a false negative. The tap Jacobian is weak, 18M is small, and γ=7 ignores the tail, so A1 ≈ A0 and the “trunk adds nothing” stop fires even if on-policy agreement would have moved T′. The false positive, if A1 does move: verification against T′ is free by construction, 2% relative on HumanEval is ~**1.7** points while Track B’s 95% intervals were about **±3**, and chat is not in the C1 accuracy line. bf16 margins under 0.15 are tenths of a token. One change before C2: spend the 70 GPU-h on agreement loss on corrupted T′ rollouts with the block loss flat across the block, and make the blinded chat judge a hard stop. Do not buy more teacher-forced tokens of T.

**5.** **4%** that 9B ends worth a 27B run. If it misses, keep the paired table of tokens/pass versus argmax agreement: a clean no on tap-only LoRA, which saves the rest of the budget.

---

## muse

1. P(C1)=15-20%. Likely B0 mean 7.2 (10.8 code/7.3 math/3.5 chat); A0 7.9, 1.08x B0 (11.5/7.9/3.7); A1 8.4, 1.06x A0, 1.16x B0 (12.0/8.3/3.9) — misses 1.3x (9.4). Accuracy: A0=T exact; A1 HE/GSM within 1%, chat -1 to -3%. Claimed mechanism: LoRA encodes future-block plan in boundary-state null space, drafter reads via taps without moving current argmax. In C1 effectively none: teacher-forced block CE+KL with 36M exposures pays for narrower/memorized outputs, drafter (1.29B) eats the easy gain; no rollout pressure so no generalizable +15%.
2. P(C2|C1)=10-15%; through-gate unconditional 2-3%; run-C2-anyway 4-6%. C2 needs mean ~13 (1.8x) and chat 3.5→6+ at ≥98%. Nothing published doubles chat committed length at fixed quality; Fast-dLLM-v2 2.5xAR is slower than your B0 (~5xAR); a C1 pass would likely be genericization, weak evidence for C2.
3. P(27B-3x-tok/s|C2)=~10%; unconditional ~0.3-0.5%. C2 mean 13 is half the 22-25/round at ~15ms the bar needs; tokens/round≠tok/s after block-32/refinement inflate round time; committed-at-quality is near scale-independent, so 9B 1.8x predicts 27B ~1.8x over a stronger 27B DFlash2/DSpine, not 3x.
4. FP risk dominates: self-verification vs T' + HE164/GSM300 ±3-6% CIs + mean-of-3 lets code/math carry flat chat; templated/generic T' drafts itself and still passes 2%; bf16 <0.15 ties add silent drift. FN risk: 18M teacher-forced-only tokens, blind KL 1-vs-10, gamma-7 downweighting late positions. One change: add T'-rollout+corrupted-prefix training/eval to C1 now and require force-8 accuracy and frozen-T shadow verification to move before promoting.
5. Overall 2-4% that 9B earns 27B spend; fallback: keep A0 in-distribution DFlash tune (~1.1x, free) + harness, stop trunk work.

---

## qwen

1. **C1 通過 ~25%。** 最可能：A0 ≈ 1.05–1.12×B0（self-distill 微調，code/math 小漲、chat 不動）；A1 ≈ 1.10–1.25×B0，A1/A0 ≈ 1.03–1.12，accuracy 在 2% 內（KL 守住）。即 A1 小贏 A0 但達不到 1.15×A0，avg 也難到 1.3×B0 → 多半 fail。A1>A0 機制存在但弱：LoRA 把中繼 tap 特徵改成更能 encode 下一 block 的 plan，drafter 預測更準，KL 保住輸出分佈（code 確認：block CE 只經 taps 進 trunk，next-token logits  solely 由 KL 管，train_joint.py:176-182,338-347）。但 C1 的 KL 把續寫熵釘死，chat 仍 ~3–4；R2 真正「移動熵上限」的 agreement loss 留到 C2，所以 C1 測的是比 R2 更弱的命題。avg 被 chat 拖死：chat 卡 3.5 時要 1.3×avg，code/math 各需 +30–50%，光雕 tap 很難。

2. **C2|C1 ~30%；unconditional ~7%。** C2 加 10× 資料 + agreement loss + block32 才是真 R2 測試，avg≥1.8× 有機會；但 chat≥2×B0 且 ≥6 tokens 是死關——B0 chat ~3–4 要翻倍，Track B（force_4→GSM 0.413）與四 reviewer 都指 chat 熵是牆。最可能：avg 過、chat 不過。

3. **27B|C2 ~30%；unconditional ~2–3%。** bar＝3× 最佳公開 drafter ≈ 14×AR；C2 過也才 ~8–9×AR@9B，外推到 22–25 committed/round 仍是大跳。我的 2–3% 低於 reviewer 5–12%，因 C1 前提弱 + chat 牆（與 qwen「含 chat 實為 2–3%」一致）。

4. **最大 false-negative：** 18M tokens 太少 + block loss 只經 tap、KL 釘住續寫 → A1≈A0，機制有效也被判死。**False-positive：** exact 對 T' 自我驗證是循環的，acceptance 漲≠能力漲；但 C1 有 2% task gate + KL，大漂移難，bf16 near-tie（<0.15 logits）與 HE 164 題可能漏過小漂移。**C2 前最值得改的一件事：** 讓 block loss 直接監督 trunk 的 next-token（加 trunk-side block/agreement CE），不要只雕 tap 特徵——把 C1 從弱特徵測試變成真前提測試。

5. **一句：** 9B 走到「值得上 27B」約 5–8%；最有價值 fallback＝實測 ~1.2–1.3×、零品質損失的 self-distill DFlash + ratio0.1 drafter，加上可重用 joint-train/eval harness 與 chat 熵上限的硬數字，足以把 27B 預算轉向小 specialist 或系統最佳化。

---

## kimi

1. **C1 passes: 25%.** Median prediction, tokens/pass for code/math/chat:
     - **B0:** 10.8 / 7.3 / 3.5; mean **7.2**.
     - **A0:** 11.7 / 7.9 / 3.7; mean **7.8**, or **1.08× B0**. Accuracy unchanged because the trunk is frozen.
     - **A1:** 12.8 / 8.6 / 4.1; mean **8.5**, or **1.18× B0 and 1.09× A0**. Accuracy likely **0.5–1.5% relative worse**, with chat drifting most.
     
     Most likely result: quality narrowly passes; both speed gates fail. The real mechanism is LoRA making tap features encode a multi-token plan and making continuations slightly more predictable. But block CE reaches trunk behavior only indirectly through shared LoRA/taps (`track_c/train_joint.py:337-347`); nothing directly trains T′ to accept D’s proposals on its own rollouts.

  2. **C2 passes: 30% conditional on C1; 7.5% unconditional.** C2 requires mean tokens/pass around **13.0**, chat **≥6**, and measured **≥1.5× wall-clock**. That is another **53%** mean gain over my median C1 prediction, while adding block-32 latency and stricter quality requirements. More data helps; on-policy agreement training is the only plausible step-change.

  3. **Final 27B bar: 20% given C2; 1.5% unconditional.** Passing C2 still means only **1.5× B0 measured**, versus the final **3× measured** requirement. Scale transfer, block-32 overhead, stronger public baselines, and chat entropy remain independent failure points.

  4. **Both error directions are real.**
     - **False negative:** 18M teacher-forced tokens, clean prefixes, and no rollout agreement loss can make C1 undersell the eventual recipe. KL weight **10** may simply freeze away the treatment effect.
     - **False positive:** verification against T′ rewards self-consistency, not fidelity to T. Generic, low-entropy output can raise acceptance while damaging usefulness. HumanEval **164**, GSM8K **300**, and chat **200** cannot reliably certify a **2%** noninferiority margin.
     - KL **1** may permit drift; KL **10** may suppress gains. BF16 near-ties add measurement noise but are not the main threat.
     
     **Single best C2 change:** train on **T′/D’s own rollouts with corrupted prefixes and direct trunk–drafter agreement loss**, retaining the original-T quality anchor. More data alone does not fix the training/decoding mismatch.

  5. **Overall odds of a 9B result worth scaling: about 8%; best fallback is a published speed/quality frontier plus a reproducible negative result showing where trunk adaptation stops buying accepted tokens.**
