# Track A v2 plan review: codex

1. **Fine-tuning: preserve the released recipe before improving it.** DSpine’s published 8B recipe uses AdamW, peak LR \(6e{-4}\), 4% warmup, clipping 1.0, 512 anchors/sequence and positional decay \(\gamma=7\). Its objective includes refinement and layerwise alignment; “prefix-weighted CE” is incomplete. Target embeddings/head remain frozen. These are training-from-scratch settings. [Paper, Appendix A](https://arxiv.org/html/2609.36173v2#A1)

   My continuation starting point: fresh optimizer, LR sweep **1e−5/3e−5/1e−4**, 3–5% warmup, 128–512 randomly refreshed response anchors, preserving loss coefficients. Train adapters first, then compare against full-drafter tuning; match that choice across control and variants. Disable the correct-predecessor curriculum initially. Use early stopping on development acceptance, retain the original checkpoint, and replay representative general-domain data.

   Treat 1–2M tokens as debugging scale; evaluate a 2M→10M→30M learning curve. No defensible universal minimum exists. Match thinking, sampling, length and domain distributions; the arbitrary 50/50 mixture needs justification.

   Cache frozen target taps for repeated off-policy comparisons. Five BF16 5120-wide taps cost **512 GB per 10M tokens**, before other features; avoid full-vocabulary logits. Online draft-rollout supervision requires fresh target evaluation on draft prefixes.

   Audit implementation first: DSpine has an open [loss-normalization correction](https://github.com/sgl-project/SpecForge/pull/1038) and reported [stale-config bug](https://github.com/sgl-project/SpecForge/issues/1170).

2. **C1a is the cheapest supported starting hypothesis; neither proposed source is validated.** NCP’s result uses an already pretrained Concept Module, not generic pooled Qwen states. Its HumanEval +7.59% is **accepted-length improvement**, not task accuracy. [§5.3](https://arxiv.org/pdf/2609.10715)

   Start with last-chunk mean pooling, then learned-query pooling. Use fixed token boundaries and committed states only. Add explicit projections into each drafter layer and count their parameters and latency.

   For C1b, predict a **frozen, normalized pooled representation of future target states** using prefix-only inputs and an auxiliary prediction loss. Future states are training labels only; inject predictions during both training and inference. A jointly learned bottleneck needs collapse prevention. Do not involve the unresolved VQ tokenizer.

   The essential third option is an equally sized **anchor-conditioned adapter without concept supervision**. A constant-input control cannot represent comparable functions. Also test shuffled concepts and removal of the auxiliary loss. MTP features deserve consideration only after measuring availability and incremental cost; product quantization adds unnecessary dependencies.

   Compare matched initialization, batches, updates and three paired seeds, initially with a frozen backbone. Confirm the winner after unfreezing. Bootstrap by prompt, keeping turns together; report seed variation separately.

   Power comes from a pilot: \(n_{\rm eff}\approx7.85\sigma_\Delta^2/(0.03\mu)^2\). At mean length five and paired SD three, approximately **3,140 independent pairs** give 80% power. Correlated rounds require more. Start with **1,000 prompts/domain, roughly 0.5–2M generated tokens/domain**, then recompute power.

3. **Use 8B for debugging, followed by a mandatory small 27B experiment.** Transferable findings include gate behavior, leakage checks, objective implementation and whether auxiliary prediction beats an ordinary adapter. Acceptance magnitude, optimal concept boundaries, caching costs, recurrent-state handling and serving gains do not transfer reliably.

   Cap 8B work; run the winning concept and matched control on 1–2M 27B tokens before authorizing scale. An 8B null result should permit one tightly budgeted 27B check.

4. **The sweep is useful; the stop rule is wrong.** Greedy-code performance cannot determine sampled-thinking block choices. Test code, math and chat in the intended serving mode, stratified by context/output length.

   Separate proposal horizon from verification length; distinguish anchor, draft, correction and bonus tokens. Compare acceptance at common anchors and fixed horizons.

   Profile draft backbone, vocabulary projection/selector, concept extraction, verification, ReplaySSM rollback and scheduling. Record CUDA-graph padding tiers and fallback paths. Tune existing SPS policies per workload/concurrency before inventing C2; the project explicitly requires deployment-specific [SPS calibration](https://github.com/Tencent/DSpine/blob/main/docs/sps_tuning.md).

   Choose by **committed tokens/total round time**, then confirm through repeated end-to-end runs.

5. **Cut C3, VQ dependencies, mandatory sequential objective changes and blanket claims about winning architectures.** Repair after rejection needs another verification and largely overlaps the next normal round.

   Reorder: version/serving audit → baseline profiling → training/export round trip → 8B checks → small 27B comparison → scale one winner → optional acceptance-objective experiment applied to both arms.

   Corrected gates:

   - **G0:** checkpoint import/export preserves outputs; zero-gate parity; greedy parity; sampled-verifier correctness, including actual proposal probabilities and recurrent rollback.
   - **G1:** control satisfies a predefined 1% noninferiority margin by domain/mode. Card reproduction requires matching its protocol.
   - **G2:** three paired seeds **and** prompt-cluster inference: ≥3% acceptance gain, corrected 95% interval above zero, beating the informative capacity control.
   - **G3:** ≥5% measured throughput gain on both code and math at concurrency one against the tuned best public baseline, interval above zero; chat and concurrency-eight regression ≤1%.

   Budget, GPU-hours: baseline/integration **12**, data/cache **20**, small experiments **24**, control **24**, winner **45**, serving/statistics **30**, reserve **45**—**200 total**.

   Most likely failure: existing target taps already contain the useful concept information; marginal acceptance gains cannot pay for added inference work.

6. **My subjective probabilities:** **25%** for ≥5% throughput improvement on both code and math at concurrency one; **30%** for ≥3% acceptance gain attributable specifically to concept injection on 27B. SpecOPD’s measured 27B mean throughput gain over DFlash V2 was only **2.83% at concurrency one**, despite stronger acceptance results. That is the relevant cautionary comparison. [Appendix A.8](https://arxiv.org/html/2611.02846v1#A1.SS8)
