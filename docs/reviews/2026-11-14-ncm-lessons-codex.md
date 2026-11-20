# NCM implementation lessons: codex

1. **NCM establishes that accurate reconstruction is possible; it does not provide an unambiguous training reproduction.** Training scripts are missing, and paper/config discrepancies remain. [Official issue](https://github.com/HazyResearch/ncm/issues/10)

Paper-reported tokenizer settings and held-out **token** reconstruction—not whole-sequence accuracy: [Tables 1/5](https://arxiv.org/html/2610.14162v1#S4.SS1)

| Dataset | Text vocabulary | Length / levels | Codes/level | Width; encoder+decoder blocks | Updates | Accuracy |
|---|---:|---:|---:|---|---:|---:|
| OWT | 50,304 | 1024 / 32 | 4096* | 384; 0+1 | 78,432 | 99.99% |
| LM1B | 30,528 | 128 / 12 | 8192 | 384; 0+4 | 25,957 | 99.86% |
| TinyStories | 8192 | 128 / 12 | 4096 | 384; 0+1 | 30,000 | 99.98% |
| XSum | 50,304 | 48 / 8* | 4096 | 384; 0+1 | 120,001 | 99.79% |
| WMT14 | 151,680 | 64 / 11 | 4096 | 384; 0+1 | 60,000 | 98.79% |
| Sudoku | 12 | 89 / 12 | 16 | 256; 0+1 | 10,000 | 100% |

Released level lengths:

- [OWT](https://github.com/HazyResearch/ncm/blob/main/config/eval/owt.yaml): \(1²,2²,\ldots,32²\); **8192** codes/level, contradicting Table 5.
- [LM1B](https://github.com/HazyResearch/ncm/blob/main/config/eval/lm1b-small.yaml)/[TinyStories](https://github.com/HazyResearch/ncm/blob/main/config/eval/tinystories.yaml): 1,8,16,24,32,40,52,64,76,92,108,128.
- [XSum](https://github.com/HazyResearch/ncm/blob/main/config/eval/xsum.yaml): 4,6,10,14,20,26,32,40,48—**nine** tokenizer levels.
- [WMT14](https://github.com/HazyResearch/ncm/blob/main/config/eval/wmt14.yaml): 1,4,8,14,20,26,32,40,48,56,64.
- [Sudoku](https://github.com/HazyResearch/ncm/blob/main/config/eval/sudoku-easy.yaml): 1,2,4,6,9,13,18,26,37,50,67,89.

Code uses EMA, commitment 0.25, per-level residual convolutions mixed at 0.5, learned first-level sampling, and no entropy loss. Restarts default off. Nested dropout normally gives full hierarchy probability 0.5, otherwise uniformly selects a proper prefix. Code averages VQ losses across levels; paper describes cutoff-only loss. LM1B’s config requests dual full/truncated losses, absent from the released quantizer implementation. [Quantizer](https://github.com/HazyResearch/ncm/blob/main/models/quant.py), [LM1B config](https://github.com/HazyResearch/ncm/blob/main/config/eval/lm1b-small.yaml)

**Measured ablations:** K=4/8/16/32 all reconstruct ≥99.95%; successive Gen-PPL reductions are 52%,32%,20%. Schedule PPL: linear/quadratic/sqrt/log/exponential/cosine =88.4/59.6/196.7/168.2/57.7/77.8. Weighting α≤1 gives 18.9–19.0, α=1.5 gives 19.4. Selected corruption mean 0.5 gives 15.41; exact complete curve values: **not found**. Block-size/codebook-size ablations: **not found**. Limitations: fixed schedules, separately trained tokenizer, scaling only to 725M, transition artifacts. [Appendices F/I](https://arxiv.org/html/2610.14162v1#A6)

**Inspection/inference:** plausible contributors versus [your implementation](/Volumes/Ext16TR0/code/ax-engine-sglang/phase1/vqvae.py) are vocabulary/domain difficulty; shorter/sparser ladders; contextual encoder versus embedding lookup; tied versus untied output; LayerNorm versus GroupNorm; missing QK normalization; decoder positional embeddings; different initialization; zero versus 0.1–0.3 dropout; missing learned first-level resampling; restart frequency/threshold; and different batch/exposure budgets. Width/depth also differ; their effects are unmeasured. Your EMA updates per microbatch, making accumulation relevant. These are hypotheses, not established explanations. [Official architecture](https://github.com/HazyResearch/ncm/blob/main/models/vae.py), [quantizer](https://github.com/HazyResearch/ncm/blob/main/models/quant.py), [training config](../../phase1/configs/b32_lossless.yaml)

2. **Independent measurements exist, but remain small and self-reported.**

- MLX reproduction: identical decoded IDs; rare-entity survival 89%/96% on XSum/WMT14; known-span surrogate masking restored 100%. Structured sentences round-tripped 99.9%, JSON-like formats 8%. MASSIVE-de exact match: NCM 51.0%, AR 57.6%; long-form invented names fell 8.1%→2.2%. [Report](https://github.com/HazyResearch/ncm/issues/19)
- Python tokenizer restart fix: reconstruction CE 6.25→2.55 at step 350. **Your code already implements this fix.** [Reproducer](https://github.com/HazyResearch/ncm/issues/16), [local code](/Volumes/Ext16TR0/code/ax-engine-sglang/phase1/vqvae.py)
- RTX5090 LM1B PPL: 94.69/88.96/81.25 at 97M/317M/725M. Reported 7.0–8.5× batch-one speedups compare against randomly initialized, eager AR models—not matched quality. [Report](https://github.com/HazyResearch/ncm/issues/26)

Additional measured reproductions in X/Reddit/HN/OpenReview: **not found**; [HN](https://news.ycombinator.com/item?id=50422168) discusses drafting speculatively. Large-vocabulary evidence includes official WMT14 above; pretrained-Qwen/code reconstruction ≥99%: **not found**.

3. **Related measured results; lessons below are my inferences.**

| Work | Experiment/result | Lesson |
|---|---|---|
| [Meta LCM/RVQ](https://arxiv.org/html/2412.08821v2#S2.SS3.SSS5) | 64×8192 RVQ reaches ~70% of continuous SONAR autoencoding BLEU | Semantic fidelity does not establish exact recovery |
| [VAR](https://arxiv.org/abs/2404.02905) | ImageNet256 FID 18.65→1.73 | Image quality cannot predict token acceptance; additional text-specific evidence beyond NCM: not found |
| [ConceptLM](https://arxiv.org/html/2602.08984v1#S4.SS3) | Llama-8B, 9.6B-token adaptation: average 47.2→47.6 versus parameter-matched baseline | Auxiliary concepts can help pretrained models |
| [Cola2](https://arxiv.org/abs/2610.31305) | Continuous hierarchy: eight-task average 60.10 after 346.03B source tokens | Test noisy/dropped latent conditioning |
| [HSD](https://arxiv.org/abs/2601.05724) | EAGLE-3 integration: >12% performance gain | Verification hierarchy, not semantic hierarchy |
| [DSpine](https://arxiv.org/abs/2609.36173) | Qwen3-8B accepted length 3.77→4.82; throughput +23.3% | Preserve predecessor dependence |
| [JetSpec++](https://arxiv.org/abs/2610.19792) | Seven-benchmark acceptance 8.94; up to 7.45× AR speed | Refinement needs latency amortization |
| [SpecPilot](https://arxiv.org/abs/2611.02501) | Single-request AR speedup 4.46×→10.87× | Systems optimization matters |
| [SpecOPD](https://arxiv.org/abs/2611.02846) | Across 8B/27B/125B targets: acceptance up to +41.45% | Train on draft-generated histories |
| [D-REAL](https://arxiv.org/abs/2611.10432) | Qwen3-8B: acceptance +0.23 tokens; throughput up to +9.4% | Optimize prefix survival directly |

4. **NCM itself as an AR-verified drafter: not found. Concept conditioning: yes.** NCP-ArchPreview adds gated completed-chunk concepts to a 1.1B drafter: 5.933→6.180 mean committed tokens, +4.17%; HumanEval +7.59%. Horizon=16; overhead=0.04M parameters. End-to-end speedup: **not found**. This uses the concept-capable target’s representations; transfer to Qwen remains untested. [§5.3](https://arxiv.org/pdf/2609.10715)

5. **Recommendation/inference:** retain token proposals and use concepts as optional conditioning. The reviewers’ impossibility claim is unsupported: verification concerns final tokens, and stochastic acceptance concerns distributional overlap—not matching an independently sampled reference. Teacher-forced accuracy alone is contradicted as a sufficient objective; seven passes and ≥99% codec accuracy as prerequisites remain unsupported. [NCP](https://arxiv.org/pdf/2609.10715), [D-REAL](https://arxiv.org/abs/2611.10432)

Ranked by evidence strength; effects/costs are estimates:

| Change | Evidence (URL) | Expected effect | Cost |
|---|---|---|---|
| 1. Train on draft rollouts | [SpecOPD](https://arxiv.org/abs/2611.02846) | Reduce exposure mismatch | High |
| 2. Optimize accepted-prefix length | [D-REAL](https://arxiv.org/abs/2611.10432) | Better useful proposals | Medium |
| 3. Inject gated concepts into token drafter | [NCP](https://arxiv.org/pdf/2609.10715) | Modest gain; Qwen uncertain | Medium–high |
| 4. Add adjacent predecessor conditioning | [DSpine](https://arxiv.org/abs/2609.36173) | Improve joint consistency | Medium |
| 5. Reproduce released codec before further scaling | [NCM configs](https://github.com/HazyResearch/ncm/tree/main/config/eval) | Isolate reconstruction failure | Medium |
