# Track A v2 plan review: gemini

### 1. Fine-tuning a released block drafter (DSpine/DFlash2)
- **LR & warmup:** SpecForge trains DFlash/DSpine from scratch at peak LR `6e-4` (AdamW, cosine, `0.04` warmup, grad clip `1.0`, ~6 epochs). Using `6e-4` to fine-tune released weights on 1–30M tokens will break them. Use a **two-group LR**: `2e-5`–`4e-5` on the 5-layer drafter + existing projections, and `1e-4`–`3e-4` on newly added zero-gated concept modules; 5–10% linear warmup, cosine decay to `0.1×`.
- **Trainable params:** Keep target `embed_tokens` and `lm_head` frozen (248k vocab). Train the drafter layers, layer-fusion projections, and concept adapter. Do **not** train the survival/verify-length head with token CE; calibrate or train it separately on rollout acceptance outcomes.
- **Data volume:** 1–2M tokens in Stage S can catch regressions or train a small adapter, but is borderline for shifting a strong 5-layer drafter without overfitting. For 27B, generate **15–30M target tokens** and prioritize **on-policy rollouts** (SpecOPD, [arXiv:2611.02846](https://arxiv.org/abs/2611.02846)) over static trajectories.
- **Anchors & loss weighting:** Keep SpecForge’s **512 random anchors/sequence** sampled across both `<think>` and final answers. Standard exponential decay (`gamma=7.0` at `b=16`, `~4.0–5.0` at `b=8–12`) ignores prefix survival; follow with SpecOPD’s E2E-TV surrogate or D-REAL ([arXiv:2611.10432](https://arxiv.org/abs/2611.10432)), which optimize consecutive accepted length directly.
- **Avoiding regression:** On `T=1.0` sampled data, hard-label CE over-sharpens draft probabilities and reduces speculative acceptance. Use **soft KL/TV distillation** against target logits, mix greedy + sampled prompts, and early-stop on **held-out accepted length**, not loss.
- **Caching vs. online:** Run **SpecForge online mode** (streaming tapped hidden states from SGLang). Caching 5 tapped layers × 5120 dim × bf16 over 30M tokens requires ~1.5 TB and prevents on-policy rollouts.

### 2. Concept source, controls, and statistical power
- **C1a vs. C1b:** **C1a** (zero-gated continuous pooling over the last completed chunk) has the only empirical backing: NCP-ArchPreview ([arXiv:2609.10715](https://arxiv.org/abs/2609.10715) §5.3) injects the *previous* completed chunk’s concept. C1b asks a small head on one anchor state to predict a future NCM level-1 latent; given your shallow/undertrained tokenizer (66–84% vs. NCM’s 98.79%, [arXiv:2610.14162](https://arxiv.org/abs/2610.14162)), C1b injects prediction noise.
- **Third option & core caveat:** Check whether DSpine already taps Qwen3.8-27B’s native MTP pre-head hidden state; if not, test tapping it or training a continuous future-chunk summary probe jointly (no VQ/PQ). Blunt caveat: unlike NCP-ArchPreview, Qwen3.8 was **not** pretrained with NCP, so pooled past hidden states may carry near-zero signal beyond the 5 target layers DSpine already reads.
- **Eliminating artefacts:** A constant-input control degenerates into a learnable bias. Match both parameter count and compute path using **shuffled/wrong-chunk hidden states** or **local-window-only pooling** (same FLOPs, no chunk-level concept), 3 seeds, identical batch order.
- **Sample size for +3%:** For $\mu \approx 4.8$, $\sigma \approx 3.3$ (`b=16`), an unpaired $+3\%$ ($\Delta \approx 0.144$) needs ~8,200 rounds ($\alpha=0.05$, 80% power). With **paired prompt evaluation** ($\rho \approx 0.7$) and **prompt-clustered bootstrap** (never per-round, as rounds in a trace autocorrelate), **~2,500–4,000 rounds (~150–250 held-out prompts / ~30k–50k generated tokens)** suffices.

### 3. Qwen3-8B vs. a 27B small-data slice
- Qwen3-8B is a **cheap code/idea filter, not a valid proxy** for Qwen3.8-27B.
- **Transfers:** LR stability, SpecOPD/D-REAL mechanics, zero-gate convergence, and whether C1a adds signal on a plain Transformer.
- **Does not transfer:** Qwen3.8-27B is **48 Gated DeltaNet + 16 attention layers** (`3:1`) with **248k vocab** and **ReplaySSM** ([arXiv:2609.36173](https://arxiv.org/abs/2609.36173)); fixed-size recurrent state alters both tapped hidden-state information and verify/rollback latency. B200 compute-to-bandwidth ratio also shifts between 8B and 27B. Use 8B (≤25h) only to kill duds, then run a **2–3M token 27B pilot** before full training.

### 4. Stage 0 block sweep and stop rule
- **Missing / flawed:**
  1. **Truncation validity:** Verify `Qwen3.8-27B-DSpine-b16` degrades gracefully or works at `b=8/12` without retraining its adjacent causal injection/survival head.
  2. **Wrong stop rule:** Gating on **greedy-code at concurrency 1** misses your worst bottleneck—at `c=32` speedup falls to `1.0–2.2×`, and sampled+thinking is far slower. Larger blocks (`b>8`) often lose at `c=8/32` because `B × batch` verification saturates tensor cores. Gate on **c=1 and c=8/32**, greedy-code **and** sampled+thinking.
  3. **Missing profiling:** Measure draft vs. verify split across `verify_len ∈ {4,8,12,16}` and batch `{1,8,32}`, **ReplaySSM rollback overhead**, **CUDA-graph padding cliffs**, and **recalibrate DSpine/DSpark survival thresholds specifically for B200**.

### 5. Plan corrections and top failure mode
- **Cut:** **C1b** (noisy unpretrained NCM latent), **C3** (second draft pass kills batch/latency scaling), and **offline hidden-state caching** on 27B.
- **Add:** Stage-0 breakdown + B200 threshold calibration; 2-group LR + soft-KL/SpecOPD on-policy training; shuffled/local-pooling control + prompt-clustered bootstrap; **2–3M token 27B pilot**.
- **Reorder:** Stage 0 (with B200 threshold calibration) → 8B Stage S (C1a + C2) → 27B 2–3M pilot → 27B Control + SpecOPD/D-REAL → 27B Best Concept + SpecOPD/D-REAL → SGLang eval.
- **Corrected Gates:**
  - **G0:** DSpine reproduces within 10%; greedy parity; Stage-0 profiling + B200 threshold calibration complete.
  - **G1:** Fine-tuned 27B control ≥ **recalibrated** released DSpine on greedy code/math AND sampled+thinking.
  - **G2:** C1a/C2 ≥ +3% accepted length over **FLOP-matched control** (3 seeds, prompt-clustered bootstrap `p<0.05`) on 8B and confirmed on 27B slice.
  - **G3:** SGLang on B200 (`c=1, 8`): ≥ +5% tok/s over **best of {recalibrated DSpine, DFlash2, DSpark}** on code/math, no chat regression, greedy parity.
- **Budget (200h):** Stage 0 **10h**, 8B Stage S **25h**, 27B data **25h**, 27B pilot **15h**, 27B control+SpecOPD **35h**, 27B concept+SpecOPD **40h**, Eval **25h**, Reserve **25h**.
- **Single most likely failure mode:** **Target-representation redundancy**—because Qwen3.8-27B lacks NCP pretraining, C1a pooling adds `<1.5%` accepted length over DSpine’s existing 5-layer target taps while extra gate/verify overhead erases end-to-end tok/s on B200.

### 6. Probabilities
- **≥ +5% end-to-end tok/s over best public drafter (B200, c=1, code/math):** **35%**. If you hit this, it will likely come from SpecOPD/D-REAL on-policy tuning on Qwen3.8 thinking traces + B200 threshold calibration, not the concept module.
- **≥ +3% accepted-length gain from concept injection specifically:** **15–20%**. NCP-ArchPreview gained `+4.17%` on a target natively pretrained with NCP; bolting an ad-hoc concept pool onto a 5-layer DSpine drafter over a standard LLM rarely clears `+3%`.
