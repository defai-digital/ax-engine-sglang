# Winning chance and design review: codex

1. **I would not make seven-pass NCM the main bet.** My subjective probabilities, within your budget at concurrency 1:

   | Outcome | Probability |
   |---|---:|
   | Any proposed variant beats DFlash2 offline, greedy code | 45% |
   | NCM beats both controls by ≥10% | 15% |
   | ≥10% faster than both public drafters, greedy code/math | 15% |
   | Same, sampled+thinking code/math | 7% |
   | Same, chat | 5% |
   | 3× tuned MTP: code / chat | 2% / <0.5% |

   These are judgments, not statistically estimated probabilities. Let \(A=1+\sum_j P(\text{accepted prefix}\ge j)\), including correction/bonus, and \(T\) be total round time: speed \(=A/T\). Your acceptance counter already includes the bonus. [DFlash’s block eight means seven proposals](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). Thus \(7.5/489\) implies **15.3 ms amortized**, not universally 17 ms.

   At a hypothetical **25 ms** NCM round, a 10% win requires \(A>13.45/12.46\) on greedy code/math, \(>12.32/10.07\) on sampled code/math, and \(>6.88\) on chat, against supplied leaders. At 20/30 ms, multiply thresholds by 0.8/1.2. Three times 250 requires \(A>18.75\); tuned MTP remains unknown. A 10% acceptance gain tolerates only 10% more latency **just to break even**. Concurrency-32 requires separate cost curves.

2. **Valid experiment; weak production design.**

   **(a)** Coarse-to-fine can coordinate exact identifiers and syntax; first-error termination does not make it useless. But reconstructing observed futures is much easier than predicting their codes. Even independent 99% reconstruction gives only \(0.99^{32}=72.5\%\) perfect blocks. Your final token head avoids that hard ceiling; reconstruction remains an inadequate success gate. [NCM’s generative results](https://arxiv.org/abs/2610.14162) do not establish speculative-prefix gains.

   **(b–d)** Train long, serve adaptive widths maximizing expected committed tokens/time. Seven small passes require measured timings, including projections, LM head, verification and rollback. Width 1024 imposes a substantial information/output-rank bottleneck; compare pretrained-width models before assuming fewer parameters wins.

   **(e–f)** Train sampled thinking from day one. Chat’s ~3.2 is **not an entropy ceiling**: correct rejection sampling with \(q=p\) accepts everything. The \(\sum p^2\) argument applies to independent sample matching, not maximal coupling. [SpecTr](https://arxiv.org/abs/2310.15141)

   Your baseline characterization understates the competition: [DFlash2](https://inco.ai/blog/dflash2/) already has candidate-path selection and local convolutions; [DSpark](https://arxiv.org/abs/2607.05147) has sequential correction and confidence scheduling. Neither is your plain `direct` control.

   Implementation findings: [NCM]( /Volumes/Ext16TR0/code/ax-engine-sglang/phase2/drafter.py) ignores `max_len`, recomputes cross-attention K/V each pass, and uses random corruption without predicted-code rollouts. [Cached evaluation](/Volumes/Ext16TR0/code/ax-engine-sglang/phase2/eval_drafter.py:109) excludes sequences exceeding 2,560 tokens. [The anchor mask](/Volumes/Ext16TR0/code/ax-engine-sglang/phase2/drafter.py:327) exposes the anchor hidden state when `n_ctx=1`.

   Losslessness additionally requires correct DeltaNet **and convolution-state** rollback, correct sampling verification, and SGLang greedy parity. Offline vLLM continuations cannot establish that. The chunk teacher-forced/free-running gap alone does not diagnose a bug: prefix survival compounds conditional errors.

3. **Fine-tune pretrained DFlash2.** Retain five 5120-wide layers, taps, convolutions, selector and frozen target head. Extend supervision to 16 proposals; add a 256-hidden-unit MLP predicting prefix survival. Select widths `{0,4,8,12,16}` from measured costs per concurrency; zero means AR. No VQ.

   Train all drafter weights with AdamW, LR \(10^{-5}\), 200-step warmup. Start with 10–30M target-generated tokens: equal greedy/non-thinking and sampled/thinking, balanced code/math/chat. Follow with draft-rollout, acceptance-oriented distillation, retaining a CE anchor. Keep final benchmark prompts outside training **and model selection**. Reserve 40 GPU-hours for validation and serving experiments.

   **Six-GPU-hour falsifier**, using the existing harness: one hour profiling/baselines, four hours fine-tuning, one hour paired held-out evaluation. Reject this budgeted recipe if no width improves measured committed-tokens/time by 5%, or greedy parity fails. This is a spending gate, not proof that longer training cannot work.

4. The plausible NCM-specific contribution is a **persistent coarse latent for nonlocal choices**—identifier reuse, argument structure—amortized across blocks. Test against equally sized continuous memory; otherwise VQ gets undeserved credit.

   Tail repair should condition on the verifier’s actual correction, then undergo verification again. Rejected-tail states cannot become committed context. One gated refinement may pay; seven mandatory passes probably will not. Confidence scheduling, trees and rejection repair are useful, but already established rather than NCM novelty.

5. **As of November 14, these are the most relevant missing techniques:**

   [DFlash](https://arxiv.org/abs/2602.06036) establishes one-pass diffusion drafting. [EAGLE-3](https://arxiv.org/abs/2503.01840) combines multilayer features with training-time rollout; [EAGLE-4](https://github.com/eagle-hq/EAGLE-4) now provides a Qwen3.8-27B drafter and cost-adaptive trees.

   [JetSpec](https://arxiv.org/abs/2606.18394) supplies causal parallel trees; [JetSpec++](https://arxiv.org/abs/2610.19792) adds refinement, foresight and pipelining. [SpecPilot](https://arxiv.org/abs/2611.02501) directly studies Qwen3.8, hybrid trees, two-level verification, lossless weight packing and batching. Headline gains are not transferable B200 predictions.

   [SpecOPD](https://arxiv.org/abs/2611.02846) covers branch-parallel on-policy training, including hybrid 27B targets; [D-REAL](https://arxiv.org/abs/2611.10432) optimizes acceptance directly. [DSpine](https://arxiv.org/abs/2609.36173) injects predecessor predictions through depth.

   For hybrid verification, examine [TreeWY](https://arxiv.org/abs/2608.20961) and [ReplaySSM](https://arxiv.org/abs/2610.13778). For sampling, examine [shared-Gumbel coupling](https://arxiv.org/abs/2408.07978) and [SpecTr-GBV’s block optimal transport](https://arxiv.org/abs/2604.25925). Their assumptions matter; latent coherence alone proves neither correctness nor acceptance gains. Apply general engine improvements to every baseline before attributing a win to NCM.
