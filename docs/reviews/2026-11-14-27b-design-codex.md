Fund a DFlash2-based Track A. Treat **1.5–1.8× tuned MTP as realistic, ~2× as stretch**, and 3× as an unproven research objective.

**1. Ceiling.** Let τ include the verifier bonus:

\[
S_{\mathrm{MTP}}=\frac{\tau}{t_{\mathrm{draft}}+t_{\mathrm{verify}}+t_{\mathrm{overhead}}}\div R_{\mathrm{MTP}}.
\]

Illustrative H200-scale arithmetic: MTP=178.5 tok/s; drafting plus overhead=3 ms; verification=20 ms. Then τ=6.5 gives 283 tok/s, **1.58× MTP**; τ=8 gives **1.95×**. Threefold requires τ=12.3 at 23 ms, or 16.1 at 30 ms. An eight-position chain cannot deliver that. These costs are assumptions, not B200 measurements.

However, **“no lossless design can reach 3×” is unjustified**. Newer [SpecPilot](https://arxiv.org/html/2611.02501v1) reports 2.13× stock DFlash2 on this target using trees, kernels and two-level verification. It evaluates greedy decoding and permits numerical near-tie differences. [JetSpec++](https://arxiv.org/html/2610.19792v2) already combines refinement and overlap, using two GPUs. Neither establishes 3× tuned MTP under your exactness and resource contract.

**2. Architecture.** Item 1: yes; remove the encoder. Item 2: one pass initially; permit a second only when acceptance improves proportionally more than round time. Reject unconditional three-pass refinement. Item 3: yes, optimize expected committed tokens/time using calibrated conditional acceptance. Item 4: yes, including sampled histories; truncated KL needs a residual probability bucket.

Concrete recommendation: retain DFlash2’s approximately 2B drafter: five layers, d=5120, FFN=17408, 32 query/8 KV heads, head_dim=128, 2048-token draft window, taps 5/19/33/47/61, rank-256 top-16 selector, block eight. Share frozen target embeddings/head. This preserves [checkpoint compatibility](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json). Defer projected d=2048 distillation.

Use the verifier bonus as anchor; MTP is not a free additional predictor. MTP-only distillation is a useful cheap control. Cache accepted feature projections and drafter KV; target KV is not interchangeable. Trees require branch-correct **DeltaNet and convolution state**, rollback and commitment; attention masks alone are insufficient.

**3. NCM.** Token refinement is NCM-inspired, not the paper’s NCM. Your 2.15-prefix result used a stand-in target; VQ’s free-running collapse supplies no positive evidence. Exclude codes initially.

A defensible later experiment uses self-predicted coarse codes as optional residual hints, with code dropout and a direct-token bypass. Keep codes only if they improve free-running acceptance at matched latency. Semantic coherence alone earns nothing under exact-token verification.

**4. Build versus reuse.** Fine-tune DFlash2 through [SpecForge’s existing DFlash2 training](https://github.com/sgl-project/SpecForge); use z-lab/dflash for evaluation support. First screen reusable SpecPilot verification code. Rebuilding your encoder/decoder is slower.

None of proposal items 1–4 establishes novelty. Refinement and [on-policy distillation](https://arxiv.org/abs/2611.02846) already have prior art. A genuine contribution requires causal concept proposals beating token-only drafting at equal latency, or a demonstrably better exact hybrid-state verification algorithm.

**5. Data and compute.** Start with **10M unique target response tokens**; expand to 30–100M only while held-out performance improves. Match thinking/non-thinking traffic, retain sampled and greedy subsets, and exclude evaluation prompts.

Planning estimate: **1–3k output tok/s/B200**, batched at moderate contexts; two replicas generate 30M tokens in 1.4–4.2 hours. Measure this first: the repository’s old 10k MoE rate does not transfer.

Five bf16 taps cost 51,200 bytes/token: **1.54 TB/30M**, 5.12 TB/100M, before logits. Store token IDs; replay sequences through a frozen teacher online, one GPU teacher/one trainer. Replay is parallel teacher forcing, not repeated autoregressive generation.

Try 10–20k steps ×8192 supervised positions: 82–164M exposures. Reserve **20–50 pod-hours, 40–100 GPU-hours**, recalibrated after 200 steps. This is a fine-tuning budget, not a convincing scratch-training budget.

**6. Track B.** Defer it. LoRA does not remove causal state updates; emitted blocks require replay, included in timing. If desired, cap a pilot at 10 GPU-hours: block eight, auxiliary head plus rank-16 adapters on the last eight layers, 5M tokens. Evaluate unverified free-running outputs. Kill for >2-point HumanEval/MBPP/GSM8K degradation, >0.2 MT-Bench loss, or <1.5× MTP including replay, using paired uncertainty estimates. Passing only justifies further investigation.

**7. Engine.** Choose SGLang for primary benchmarking and integration. Pin versions, precision and kernels. vLLM is a replication path; [PR 52816](https://github.com/vllm-project/vllm/pull/52816) is already merged. Report two-GPU overlap separately with equal-resource baselines. Never combine the cards’ precision-dependent results into one ranking.

Run these experiments in order:

1. Reproduce AR, tuned MTP, DSpark v2 and DFlash2 at concurrency 1/32. Measure actual verification costs, pooled tokens/round, greedy parity and recurrent-state correctness.
2. Fine-tune DFlash2 on 10M tokens; compare the unchanged checkpoint and MTP-only tuning. Continue for ≥10% measured improvement over DFlash2.
3. Ablate adaptive width and gated second-pass refinement; test trees if replay predicts headroom. Retain only ≥5% end-to-end gains.
4. Release only at ≥1.5× tuned MTP on code/chat, ≥1.1× the strongest reproduced drafter, correct stochastic rejection using actual proposal probabilities, and concurrency-32 AR fallback meeting the no-regression gate.
