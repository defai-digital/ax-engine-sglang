1. **The diagnosis is substantially right; the fix is necessary but insufficient.** Gold-code conditioning is legitimate teacher forcing, but its accuracy measures an oracle-assisted task. [NCM itself documents this mismatch](https://arxiv.org/html/2610.14162v1#A6.SS3).

   Keep rollout training; ramp toward 100% predicted-code conditioning once stable. Corrective targets are a reasonable experiment, not an established solution: nearest residual code minimizes latent distance, not token-prefix error. Incorrect coarse codes may leave residuals outside the finer codebooks’ capacity. Compare rollout with fixed versus corrective targets to isolate their value.

   Remaining risks: incompatible within-level argmax choices, frozen-encoder limitations, code losses competing with token loss, and a decoder learning to ignore unreliable plans. Greedy verifier continuations are deterministic given full context; apparent multimodality comes from the drafter’s approximation. Sampling diverse plans does not directly help single-path greedy acceptance.

2. **A discrete plan is plausible, but presently unproven for this objective.** It is not inherently harder to predict: compression can remove irrelevant detail. However, reconstructing an entire block can require distinctions unnecessary for predicting its first two tokens. Your hierarchy may impose a harder prerequisite before doing useful work.

   NCM’s generation-quality results do not establish exact greedy agreement with another model. Treat its architecture as a hypothesis.

   Drop the plan if direct drafting matches or beats rollout on free-running prefix survival **and** costs less. If the plan improves acceptance, require enough improvement to cover all extra passes. Compare equal GPU-hours and training curves, not just equal steps. With this budget, require roughly **20% higher projected throughput than direct** to justify keeping the hierarchy; that is an engineering threshold, not a scientific falsification.

3. **Ranked priorities:**

   **First: prefix-weighted token loss and shorter draft policies.** Try normalized weights proportional to \(0.8^{j-1}\), retaining a small uniform component. Report \(P(A\ge j)\), particularly positions 1–8. Optimize draft lengths 2/4/8/16/32 on validation; confidence thresholds must predict prefix survival, not merely token correctness. Truncation reduces verification cost but saves drafting cost only if computation also stops early.

   **Second: introduce token dependencies.** Try four sequential passes producing two tokens each, with later chunks conditioned on predicted earlier chunks during training. Compare against the existing four-pass hierarchy. Also benchmark the unchanged 0.8B AR drafter: it establishes whether the frozen encoder already contains better usable predictions. Prioritize accepted prefixes over completing 32-token blocks.

   **Third: add verifier-logit distillation to hard-token CE.** Test a modest KL term, emphasizing early positions. Soft targets may improve generalization, but KL is not the exact greedy-agreement objective and should not replace hard supervision automatically. I would defer continuous plans and mask refinement until these simpler controls establish the bottleneck. [EAGLE-3’s move toward direct token prediction](https://arxiv.org/abs/2503.01840) supports questioning auxiliary latent targets.

4. **Pause lossless-tokenizer training.** Speculative decoding obtains losslessness through verifier correction; tokenizer reconstruction need not be lossless. Moreover, 99.95% average token reconstruction does not guarantee exact blocks or strong prefixes.

   Freeze the current plan tokenizer during comparisons. Spend primarily on drafters; resume tokenizer work only if an oracle-plan experiment demonstrates substantial headroom that better codes plausibly unlock. Four-percent code usage alone is not a reason to spend more.

5. **Next-24-hour rule:**

   - Cap expenditure at **48 GPU-hours**, including evaluation. Finish the three-way comparison, then allocate remaining time to direct versus rollout winners, prefix weighting, and a confirming seed.
   - Use prompt-disjoint code/chat sets, simulated round advancement, prompt-bootstrap confidence intervals, and domain-specific results. Audit EOS/padding. Distinguish accepted drafts \(A\) from tokens/round \(R=1+A\): **1.07 tokens/round means only 0.07 accepted drafts**.
   - Rank by projected throughput \(R/[t_{\text{draft}}+t_{\text{verify}}(m)]\), including encoder-cache updates. Require **≥1.5× best MTP on both code and chat**, robust to timing uncertainty, before funding H200 integration.
   - If neither architecture approaches that gate and acceptance has plateaued, stop this design. H200 timing remains a projection check until integrated end-to-end measurement.
