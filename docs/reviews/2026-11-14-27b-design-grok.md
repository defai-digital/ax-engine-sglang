**3× MTP is unreachable for any lossless drafter on this target.** The realistic concurrency-1 ceiling is about **1.5× MTP on code and chat, and about 1.8× on GSM8K**. Track A is worth running against that ceiling. Track B does not raise it unless you skip verification and accept a quality drop.

Use only the DFlash2 card as the joint scoreboard. DSpark’s own 1.37–1.65× claims sit on a different stack (AR 94–96 tok/s versus 69).

## 1. Ceiling

Let \(T = 1/68.9\,\text{s} = 14.5\,\text{ms}\). Tokens per round over tok/s gives the round cost:

| | \(L\) | tok/s | round |
|---|---:|---:|---:|
| MTP, GSM8K | 5.02 | 178.5 | **1.94 \(T\)** |
| DFlash2, GSM8K | 5.46 | 236.1 | **1.59 \(T\)** |
| MTP, HumanEval | 3.91 | 151.9 | 1.78 \(T\) |
| DFlash2, HumanEval | 4.39 | 214.6 | 1.41 \(T\) |

Fit \(L=(1-\alpha^{k+1})/(1-\alpha)\) with \(k=7\): GSM8K \(\alpha\approx0.86\) (asymptote \(1/(1-\alpha)=7.2\)); HumanEval \(\alpha\approx0.78\) (asymptote 4.6); MT-Bench \(\alpha\approx0.76\) (asymptote 4.2). DFlash2’s longer \(L\) is mostly \(k=8\) at nearly the same \(\alpha\). Its speed edge is the cheaper round (1.59\(T\) versus 1.94\(T\)), not a new acceptance regime.

Infinite-\(k\), free draft, verify \(=1.0T\): GSM8K \(7.2/2.59=\) **2.8× MTP**, code \(4.6/2.20=\) **2.1× MTP**. A sane \(k=12\) at \(\alpha=0.86\) yields \(L\approx6.2\), and 3× MTP on GSM8K needs \(L/7.77\le0.80T\). The verify forward alone is at least \(1.0T\). Code never gets there: 3× MTP is 6.6× AR, above the \(\alpha=0.78\) asymptote. No published drafter has moved \(\alpha\) by the \(+0.10\) that would matter. DSpark’s acceptance is below MTP.

A narrow drafter plus a width-2 tree can plausibly land near **1.5× (code/chat) and 1.7–1.8× (GSM8K)** at concurrency 1. That is the bar.

## 2. Track A

Do 1 and 4. Do 2 as two passes, not three. Do 3 as a threshold sweep after the train, not as a research head.

Concrete model, ~120M trainable:

- Token 1 comes from the target’s MTP head (one GEMV on a hidden state you already have). The drafter predicts positions 2–8 only.
- 6 layers, \(d=1024\), 8 heads, FFN 4096. Output through an up-projection into the **frozen target LM head** (1024×5120). No private 248k embedding.
- Four taps, layers 16/32/48/64, each projected 5120→1024 and cross-attended. Full \(d=5120\) is how DSpark became 1.86B and then fell below AR at concurrency 32.
- Pass 1: positions 2–8 in parallel. Pass 2: refine 5–8 from embeddings of 1–4. Drop a third pass unless it adds \(>0.4\) to \(L\) at \(<0.15T\).
- Loss: prefix-weighted CE (\(0.8^j\)) plus KL to the teacher’s top-32 logits. Train and gate at temp 1.0, top-p 0.95, thinking on. Greedy \(L\) is a debug metric.
- After training, sweep \(k\in\{4,6,8\}\) and a top-2 branch at position 2 inside one verify forward. Extra verify tokens are cheap while the step is bandwidth-bound; a third decoder launch is not.

Missing from the proposal, and required before more layers: MTP composition, shared LM head, a width-2 tree, and a Gated-DeltaNet rollback spike. Accepted tokens must advance the recurrent state once; the rejected suffix is discarded. That integration, not the loss, is the long pole.

## 3. NCM codes

No multiscale VQ and no concept tokens. Teacher-forced acceptance of 2.0 collapsed to ~1.07 free-running because a future-block code either carries the tokens’ entropy (so it is as hard as the tokens, and one miss poisons the block) or it does not (so the token head ignores it). A linear projection of past hidden states is the useful “coarse” object. Keep scale-wise refinement in token space, and call the system a block drafter.

## 4. Build versus reuse

Fork the Apache-2.0 DFlash / SpecForge trainer and add MTP token-1 plus the four projected taps. `phase2/drafter.py` is the right loss and chunk schedule, and the wrong starting point: \(L=2.15\) after 2,000 steps against a 0.8B stand-in that never saw a 27B hidden state. A from-scratch 2B clone of DFlash2 is not a contribution. The contribution is MTP composition on a DeltaNet hybrid, the concurrency-1 versus concurrency-32 Pareto at ~120M, and the VQ negative result you already measured.

## 5. Data and compute

100M target tokens for the gate, 200M only if \(L\) is still climbing. The 27B bf16 weights are ~54 GB, so one replica per B200. Plan on ~3k output tok/s per GPU, ~6k aggregate: **100M tokens is about 5 h of generation.**

Do not cache raw taps. Four layers × 5120 × 2 bytes is 40 KB/token, and 100M tokens is **4 TB**. Run the frozen 27B forward online during training. The teacher dominates the 120M backward. At a global batch of ~16k tokens that is ~6k steps, **under ~12 h wall, ~24 GPU-hours**, inside a 200 GPU-hour budget with room for three ablations.

## 6. Track B

One experiment, then stop or proceed on evidence. Freeze the 27B. Train an 8-token head on the final hidden state and the MTP state for 10M tokens (~4 GPU-hours). **Kill it** if the exact prefix is no better than MTP (\(L\le5.0\) GSM8K, \(\le3.9\) HumanEval) or HumanEval drops by ≥2 points. A head that only ties MTP has reproduced MTP. Shipping the block **without** a verify forward is the only way to 3×, and it is a quality bet, not a speculative-decoding bet. The moment the AR path verifies, Track B inherits Track A’s ceiling.

## 7. Engine

**SGLang for the gate.** DFlash2 and DSpark are native there; vLLM DFlash is PR 52816. One engine. If SGLang cannot run this model’s MTP on the B200 pod, measure MTP against yours in vLLM and label every DFlash2 number as external.

## 8. Plan

1. **Spike, ~8 GPU-h.** SGLang, thinking, temp 1.0, top-p 0.95, concurrency 1 and 32: AR, MTP \(k=7\), DFlash2, plus \(t_v(k)\) for \(k=1,4,8,16\). Go if MTP/AR on GSM8K and code is in 1.8–2.8×.
2. **Offline prototype, ~20 GPU-h, 20M tokens.** MTP token-1, four taps, 6×1024, two passes, online teacher. Go if held-out \(L\) matches DFlash2 (GSM8K ≥5.5, HumanEval ≥4.4, MT-Bench ≥4.1) at temp 1.0. A miss of more than 0.3 ends the architecture search.
3. **100M train and SGLang integration, ~40 GPU-h,** including DeltaNet rollback. Go if end-to-end speed is ≥10% above same-engine DFlash2 on at least two of the four tasks and worse on none, and ≥1.0× AR at concurrency 32. If concurrency 32 fails, shrink the drafter. Do not add layers.
4. Track B head only beside step 3. Apply the kill rule above. Drop 3× lossless from the plan.
