# Winning chance and design review: grok

I'll read the design docs and drafter code, then check 2026 speculative-decoding results before answering the five questions.**1. Probabilities**

AR is 9.8 ms/token. DFlash2 greedy rounds are 15.3 ms (code, τ=7.5, 489 tok/s), 13.7 ms (math), 13.2 ms (chat): about 4–6 ms of draft on top of one verify. That draft is not bandwidth-bound (1.92B weights are ~0.5 ms at 8 TB/s). 7.5/8 implies conditional survival r≈0.96. Expected length (1−r³²)/(1−r) beats 7.5 only if r≳0.89. Your only free-running number is 2.15 tokens.

- Offline τ > DFlash2, greedy code, any variant: **30%** (yours 55–65). The chance is `direct_b32`. Block 32 clears 7.5 if r≈0.90. NCM is the worse variant.
- NCM ≥10% over both controls and better survival at 16: **12%** (yours 20–30). Gate 2 requires beating `direct`. The CPU toy went the other way (teacher-forced 0.95, free-run 0.31).
- End-to-end ≥10% vs DFlash2 and DSpark, greedy code/math: **8%** (yours 25–30). 538 tok/s needs τ≥8.6 at 16 ms, or τ≥10.8 at 20 ms. Seven passes sit in the second column.
- Sampled + thinking: **4%** (yours 15–20). You would be training the other distribution.
- Chat: **3%** (yours 10–15). τ is 3.2–4.5 for MTP, DFlash2, and DSpine.
- 3× MTP: **2% code, <1% chat** (yours ~5 and <1). 3×250 tok/s is 750. At a 14 ms round that needs τ≥10.5 without growing the round. Chat would need code-like entropy.

The c=32 “2.2×” is not in the ranges you listed. DFlash2 at 1662–2508 vs AR at 1170–2000 is about 1.3–2.1×.

**2. Design**

This is the wrong experiment.

(a) A VQ hierarchy fights exact-prefix acceptance. NCM scores a full block softly. You score “token 1 matches, else the round is worth ~1.” A level-1 code depends on all 32 tokens, including ones you cannot predict, so a bad coarse code flips position 1. Corruption 0.5 and dropout 0.1 train the token head to ignore the codes, and you still pay six passes. Oracle reconstruction at 99% is not the hurdle: 0.99³²≈0.72 even with gold codes, and you will not have gold codes. The 66–84% run already showed this vocab is hard.

(b) Fixed 32 loses to a truncated K. Only code has headroom (7.5/8). Chat is dead by about position 4.

(c) Seven serial passes spend the only thing 140M parameters buy. DFlash2’s extra ~5 ms is launches, cross-attention, and the 248k LM head.

(d) d=1024×6 will not hold r≈0.96. The retarget note said to pick width after the draft/verify split. You measured ~5 ms of ~15 ms, then chose a stack that is both narrower and more serial.

(e) Greedy non-thinking distillation does not transfer to sampled thinking. Thinking traces are a different language. Sampled losslessness needs a shared-noise coupling, not argmax match.

(f) Chat will barely move. DSpine’s own MT-Bench gain is 4.48 vs 4.11. That is entropy.

Gate 2 also ignores latency, so it can promote a slower model. Four online-27B runs of 30k steps will eat the 200 GPU-hours before a tuned baseline exists.

**3. What to build**

Fine-tune DSpine (Apache-2.0, already in SGLang, 5 layers, block 16, adjacent causal injection). It is the published one-pass fix for independent positions. Add a linear stop-head and serve K∈{4,8,12,16} from predicted survival. Train the serving mixture (thinking on, T=1, top-p 0.95, plus a greedy half), loss weight 0.8ʲ, 10–20M target tokens, benchmark prompts held out. Cache taps for that slice so the 27B stays out of the loop. Skip a DeltaNet tree unless that kernel is already in your SGLang. Use shared-Gumbel verification for T>0.

Falsifier, under 6 GPU-hours, no training. Sweep DFlash2 and DSpine at K=8 and K=16, greedy code and sampled-thinking chat. Kill the long-block thesis if greedy-code tok/s rises under 8%. Then score a second DFlash2 block twice: teacher-forced on a correct first block, and free-running. If the oracle second block still accepts ≳4 tokens on code, chain blocks. If the free-run block adds ~0, fine-tune DSpine’s injection. If both fail, stop. A 32-code hierarchy will not win where a second real draft block does not.

**4. Where an NCM-shaped idea still helps**

Not as the proposal. The target hidden state is already a better plan than a code you must invent.

- One vector that predicts how far the block stays stable, used only to truncate.
- One repair pass after the first reject, conditioned on the verifier token. That is `chunks` with a single extra pass, paid only if position 1 survived.
- A tree at the first low-margin token. Chat’s τ≈3 is several plausible first tokens. JetSpec and SpecPilot are this, with a latency argument.
- A sketch carried across blocks is weak. The next verify hidden state replaces it.

**5. What you are missing**

- DSpine. On Qwen3-8B, matched training, mean τ 7.83 vs DFlash2 7.55 at T=0. The 27B card confounds block 16 plus dynamic verify against DFlash2’s block 8, and MT-Bench only moves 4.11→4.48. https://arxiv.org/abs/2609.36173 https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16 https://github.com/Tencent/DSpine
- DFlash/DFlash2, one denoising step on target features. https://arxiv.org/abs/2602.06036 https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
- Domino, the post-backbone predecessor head DSpine argues is too late (discussed in 2609.36173).
- JetSpec, parallel trees with branch-wise causal scores. https://arxiv.org/abs/2606.18394
- SpecPilot: hybrid-recurrent tree verify, two-level verify, batching, no retrain. Claims 4.46×→10.87× at batch 1. https://arxiv.org/abs/2611.02501
- EAGLE-3. DFlash’s claim is about 2.5× it by dropping sequential drafting. https://arxiv.org/abs/2602.06036
- Shared-Gumbel block verification. https://arxiv.org/abs/2610.22543 Block-diffusion draft trees: https://arxiv.org/abs/2604.12989
- NCM itself is a diffusion-LM result, speed versus diffusion models, not versus a verifier. https://arxiv.org/abs/2610.14162
- Longer K on stock DFlash2 already raises tokens/pass on edit-heavy work (GB10, K=16, ~15.9 vs ~7). https://github.com/fattchris/qwen-dflash2-spark
- A 22M phrase head gets most of the math gain from target features alone. https://arxiv.org/abs/2611.12464

The missing pieces are DSpine, a K sweep, truncation, one rejection repair, a hybrid tree, and a sampling coupling. A six-level residual VQ on the critical path is not one of them.
