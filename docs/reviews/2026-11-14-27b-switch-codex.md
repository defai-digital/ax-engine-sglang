**The 27B materially improves the opportunity, but 3× MTP remains a long shot.** I assume “MTP is weak” means particular prose/chat workloads. The blanket claim is contradicted by [vLLM’s measured acceptance](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

1. **Probabilities and arithmetic.** My subjective probabilities of passing **both code and chat**, after full training and competent integration of the current chunk architecture:

   | Target | ≥1.5× best MTP | ≥3× best MTP |
   |---|---:|---:|
   | 35B-A3B | 10–20% | <3% |
   | 27B | 25–40% | 3–8% |

   These are judgments, not statistical estimates; the unmeasured 35B baseline could change them substantially.

   Let \(A\) be accepted draft tokens, \(d\) drafting latency, \(v\) verification latency, and \(M\) MTP speedup over AR. Normalize latencies by one AR token step:

   \[
   S_{\mathrm{vs\,MTP}}=\frac{A+1}{(d+v)M}.
   \]

   Illustrative assumptions for 16-token drafts—not measurements:

   | Target | \(d\) | \(v\) | \(M\) | Required \(A\): 1.5× / 3× |
   |---|---:|---:|---:|---:|
   | 35B | 0.40 | 1.80 | 2.0 | 5.6 / 12.2 |
   | 27B | 0.10 | 1.30 | 2.2 | 3.6 / 8.2 |

   Even the favorable 27B assumptions turn today’s \(A=2.15\) into only **1.02× MTP**. Moreover, that result uses the encoder itself as verifier, making first-token agreement unusually easy ([ablation report](/Volumes/Ext16TR0/code/ax-engine-sglang/results/phase2_ablation/REPORT.md)). Neither ninefold active compute nor \(0.8/27\) predicts latency: weight bandwidth, kernels, five decoder passes, encoder updates, and verification width matter. Longer drafts also increase \(v\).

2. **Contribution and competition.** Benchmark DSpark and DFlash2 before expensive training. The cited 2–2.6× comparisons confound engines. A cleaner counterexample: DFlash2’s own same-engine H200 results give **214.6/151.9 = 1.41× MTP** on HumanEval and **184/134.9 = 1.36×** on MT-Bench. Those use sampled, thinking-mode generation and fixed seven-token proposals, so they do not establish your greedy gate against tuned MTP. [DFlash2 evaluation](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

   A useful ax-engine-sglang contribution would be better latency/acceptance through adaptive chunking, lower drafting cost, or stronger non-thinking chat performance. Merely beating MTP on another dense target offers limited novelty. Keep the chunk drafter as an ablation; prioritize target-feature conditioning.

3. **Drafter changes.** Start with eight tokens, comparing `[1,1,2,4]`, `[1,2,5]`, and one-pass drafting; retain sixteen only when marginal accepted tokens justify their cost. Ablate projected target hidden states against the separate encoder. DSpark already consumes several target feature layers. [Architecture](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)

   Test roughly 0.3–1B draft capacity before scaling. Cache decoder context K/V projections: the current attention implementation recomputes them each pass. Target KV reuse needs explicit adapters; dimensions and recurrent states prevent automatic substitution. Verify tokenizer mappings exactly. Train non-thinking data first; evaluate thinking separately with matched templates, effort, and retained reasoning.

4. **Switching costs.** Measure dense distillation throughput immediately. As a planning sensitivity—not a forecast—2–6k aggregate output tokens/s on two B200s means **21–64 hours** for 450k responses averaging 1,024 tokens, excluding prefill. Thinking increases token volume. Keep the completed approximately 180k 35B responses as controls/warm-start data, not 27B supervision.

   Integration remains substantial: target-feature extraction, CUDA graphs, accepted-token advancement, and rollback of both attention KV and DeltaNet state. Require exact greedy parity against the same checkpoint and precision. The repo’s prefill-difference projection cannot establish that or the end-to-end gate.

**Recommendation: C.** Finish the queued one-day 35B screen, while measuring 27B AR, tuned MTP, DSpark, and DFlash2 on the same B200 protocol. Defer further bulk distillation and 80k-step training until those measurements identify an attainable acceptance/latency threshold. Favor 27B for subsequent research if confirmed, with target-conditioned drafting as the main architectural experiment. Treat 3× as an exploratory upside, not a planning assumption.
