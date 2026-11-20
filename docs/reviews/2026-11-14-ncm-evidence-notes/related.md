# Concept hierarchies and lossless speculative drafters

Scope: source-measured facts for a lossless speculative-decoding drafter. Every number below has a URL. Claims not printed in the fetched source are marked not found. Inferences are separated from measurements.

## What did NCP-ArchPreview (arXiv 2609.10715) measure, and is it a drafter?

### Takeaway
NCP-ArchPreview is an 8.94B token-level AR LM that also predicts product-quantized concepts over 4-token spans. It does not report concept-code accuracy or concept-to-token reconstruction. The only drafting result is feature injection from the target’s concept state into a separate 1.1B token drafter, which raises mean accepted length by 4.17% under exact speculative verification.

### Cited Findings
- Architecture (Table 10): 8.94B parameters; Token Encoder 16 layers, Concept Module 8 layers, Token Decoder 16 layers; hidden size 4,096; FFN 11,008; 32 heads; vocab 100,278; max length 8,192. Concept compression factor \(k=4\) (mean-pool of 4 token states). Product quantization: \(S=32\) segments, \(N=128\) codewords per segment, codeword dimension 128, stated capacity \(128^{32}\). Concepts are predicted as a softmax-weighted sum of codewords (differentiable), not as argmax codes, and added back into the token stream after a causal shift. NCP loss is MSE to a stop-gradient continuous concept target; NTP remains token cross-entropy. — [arXiv HTML 2609.10715](https://ar5iv.labs.arxiv.org/html/2609.10715)
- Training scale and loss: trained on 5.73T Dolma-3 tokens. Matches OLMo-3-7B’s final pretraining loss after 51.3% of the tokens (stated 1.95× convergence). Final Stage-1 loss is 0.091 lower than OLMo-3-7B. Stage-2 converges 1.51× faster. At 85% of standard compute it approaches a parameter-aligned 8.9B baseline. Pareto compute efficiency stated as 1.74×. — [arXiv abs 2609.10715](https://arxiv.org/abs/2609.10715); [arXiv HTML](https://ar5iv.labs.arxiv.org/html/2609.10715)
- Downstream (Table 1, Stage-1 Overall AVG): OLMo-3-7B 46.59 vs NCP 49.04 (+2.45). Stage-2 Overall AVG 56.98 vs 57.57 (+0.59). GSM8K Stage-1 39.27 vs 45.26 (+5.99). Likelihood block is bits per UTF-8 byte (lower better), not concept bits; Stage-1 BPB average 0.824 vs 0.811 (−0.014). — [arXiv HTML 2609.10715](https://ar5iv.labs.arxiv.org/html/2609.10715)
- Concept prediction accuracy (top-1 code hit rate): not found. Token reconstruction from concepts: not found. The word “reconstruct” in the paper is about related work (JEPA), not a measured reconstruction rate. — [arXiv HTML 2609.10715](https://ar5iv.labs.arxiv.org/html/2609.10715)
- Speculative experiment (Section 5.3, Table 9): not the Concept Module as the drafter. A 1.1B drafter (DFlash2 two-tap dynamic convolution plus a path selector over the top-16 candidates, plus DFlare-style mix of five target hidden states) is trained by online target distillation on a fixed 5B-token subset × 10 epochs. Baseline+Concept adds the target concept \(c_{\lfloor t/k \rfloor}\) from the last fully completed target chunk into every draft position, +0.04M parameters, gates initialized at 0. Evaluation is “exact speculative verification,” proposal horizon 16. Mean accepted length (includes the target bonus/correction token): GSM8K 6.351 → 6.537 (+2.93%); MATH 6.105 → 6.240 (+2.22%); HumanEval 5.432 → 5.845 (+7.59%); MBPP 5.844 → 6.099 (+4.37%); macro 5.933 → 6.180 (+4.17%). No wall-clock speedup is reported for this experiment. — [arXiv HTML 2609.10715](https://ar5iv.labs.arxiv.org/html/2609.10715)
- VQ adaptation: freezing the backbone and updating the 17M-parameter VQ codebooks and concept heads is a domain-adaptation interface. That 17M figure is not the drafter. — [arXiv abs 2609.10715](https://arxiv.org/abs/2609.10715)

### Inferences
- The concept path is a training-time auxiliary target plus a conditioning feature. Losslessness, if any, comes from exact token-level speculative verification of the token drafter, not from accepting or rejecting concept codes.
- Because the injected vector is the target’s own completed-chunk concept, this is target-side feature conditioning, not a smaller concept model proposing a plan that the AR LLM then verifies.
- MSE-to-soft-codebook training does not produce a measured discrete concept accuracy, so a concept-code acceptance rate cannot be computed from this paper.

### Gaps
- Concept-code accuracy, codebook usage/collapse, and bits per concept: not found.
- Whether “exact speculative verification” is greedy token match or Leviathan rejection sampling: the paper says exact speculative verification and counts a target correction/bonus token, but does not write the acceptance probability formula. Distributional losslessness is therefore not explicitly proved in the fetched text.
- End-to-end tokens/s for Baseline+Concept: not found.

## What does Meta’s Large Concept Model measure?

### Takeaway
Official LCM treats a concept as a sentence in a continuous SONAR embedding and generates the next sentence embedding (MSE, diffusion, or residual-quantized SONAR). A frozen SONAR decoder turns embeddings back into text. It is not verified by a separate token-level AR LLM. Quantized LCM is discrete RVQ but underperforms diffusion LCM on the paper’s own pretraining metrics, and full 64-codebook RVQ reaches only about 70% of continuous SONAR auto-encoding BLEU.

### Cited Findings
- Concept unit: “a concept corresponds to a sentence,” using frozen SONAR (up to 200 text languages). Explorations at 1.6B parameters and on the order of 1.3T tokens. — [arXiv abs 2412.08821](https://arxiv.org/abs/2412.08821); [arXiv HTML v2](https://arxiv.org/html/2412.08821v2)
- Scale conflict: arXiv v2 abstract says the scaled model is 7B on about 2.7T tokens. The Meta research page says about 7.7T tokens. — [arXiv abs 2412.08821](https://arxiv.org/abs/2412.08821); [Meta research page](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)
- Concepts are continuous by default. Quant-LCM discretizes SONAR with residual VQ: FAISS iterative k-means, \(n_{\text{codebooks}}=64\), \(n_{\text{units-per-codebook}}=8192\), trained on 15M English Common Crawl sentences. Auto-encoding BLEU on FLORES devtest rises with more codebooks and, with all 64, reaches around 70% of the BLEU of unquantized SONAR. Quant-LCM-d predicts the next codebook index with an 8192-way softmax; Quant-LCM-c uses a continuous distance softmax. The SONAR decoder was fine-tuned on quantized vectors (1.2M English sentences; with probability 0.3 a random prefix of the codebooks is kept). — [arXiv HTML v2](https://arxiv.org/html/2412.08821v2)
- Table 3 pretraining comparison (the paper’s columns are \(\ell_2\), round-trip \(\ell_2\), paraphrasing similarity, contrastive accuracy). One reported Quant-LCM-d contrastive accuracy is 81.1%, next to One-Tower 80.2% and Two-Tower 80.6% on the same row-block; Quant-LCM-d’s other scalar on that row is 0.682 vs Two-Tower 1.137. A later row-block (Wikipedia-en / Gutenberg header follows in the HTML) shows Quant-LCM-d contrastive accuracy 76.0% and 72.0%. Exact dataset names for the 81.1% cell were not cleanly recoverable from the HTML line break, so only the printed cells are cited. — [arXiv HTML v2](https://arxiv.org/html/2412.08821v2)
- The paper states that directly minimizing MSE in embedding space does not yield good results, and that “the limited performance of the Quant-LCM approaches … may be explained by” SONAR not being trained to be quantizable, “yielding a significant number of codebooks and a large amount of units per codebook.” — [arXiv HTML v2](https://arxiv.org/html/2412.08821v2)
- Segmentation quality uses AutoBLEU (encode a segment, decode, BLEU against the segment) on 10k documents / about 500k sentences. Numeric AutoBLEU table values: not extracted. — [arXiv HTML v2](https://arxiv.org/html/2412.08821v2)
- No sentence in the fetched LCM text says a token-level autoregressive LLM verifies LCM outputs. Generation is decode-with-SONAR. — [arXiv HTML v2](https://arxiv.org/html/2412.08821v2)
- Related, not official LCM: SONAR-LLM keeps continuous SONAR thoughts but trains with token-level cross-entropy through the frozen SONAR decoder. On XSum / CNN-DM (same number of sentences as the reference): SONAR-LLM ROUGE-L 19.3 / 16.0; MSE LCM 12.2 / 7.6; diffusion LCM 12.0 / 10.2. This is teacher-decoded token likelihood, not speculative verification. — [arXiv abs 2508.05305](https://arxiv.org/abs/2508.05305); [arXiv HTML](https://arxiv.org/html/2508.05305v1)

### Inferences
- Quant-LCM is the closest official-LCM analogue to a discrete coarse-to-fine concept generator, but the residual steps refine one sentence embedding. They are not a draft tree for a target AR LLM.
- Round-trip \(\ell_2\) after SONAR decode-and-reencode is a reconstruction-consistency metric, not token exact-match accuracy. The ~70% relative auto-encoding BLEU is the explicit reconstruction number.

### Gaps
- Numeric AutoBLEU for the chosen sentence segmenter: not extracted.
- XLSum ROUGE-L for the 7B LCM vs Llama-3.1-8B (a secondary summary claimed 20.2 vs 19.7): not taken from the paper HTML in this pass, so not used.
- Token-level exact reconstruction accuracy of SONAR or Quant-LCM: not found (BLEU and \(\ell_2\) only).
- Any speculative-decoding acceptance length: not found.

## Do VAR / next-scale models reconstruct text, and does a coarse error ruin the sample?

### Takeaway
Tian et al. VAR is images only. No text token-reconstruction or LM-perplexity number exists in that paper, and no ablation was found that measures whether a wrong coarse scale ruins an image. The text next-scale system with measured token reconstruction is Stanford NCM (arXiv 2610.14162), which is a generator, not a verified drafter. NCM’s own figure says finer levels fill in and correct content; it also trains with corrupted conditioning codes because generation accumulates code errors. A numeric “coarse error ruins the sample” ablation was not found.

### Cited Findings
- VAR (images): next-scale / next-resolution prediction with a multi-scale VQVAE. ImageNet 256×256: AR baseline FID 18.65 and IS 80.4 improved to FID 1.73 and IS 350.2, with 20× faster inference. A 2B VAR reaches FID 1.73, below the validation-set reference FID 1.78, and beats L-DiT 3B/7B on the paper’s comparison. Text token reconstruction or LM perplexity: not found. The paper is image generation. — [arXiv HTML 2404.02905](https://arxiv.org/html/2404.02905)
- Search of that HTML for a coarse-scale error ablation (wrong/perturbed first scale, drop first scale): no matches. — [arXiv HTML v1 2404.02905](https://arxiv.org/html/2404.02905v1)
- Text next-scale: NCM encodes text with a multiscale residual VQ-VAE and generates one level at a time, all codes in a level in parallel, conditioned on the cumulative reconstruction of coarser levels. Abstract: better generative perplexity than the strongest diffusion LM baseline on LM1B at 21× lower runtime; competitive on OpenWebText at 10×; matches the strongest DLM’s ROUGE-L on XSum at 31×; Sudoku beats AR and diffusion baselines with fewer function evaluations. — [arXiv abs/html 2610.14162](https://arxiv.org/html/2610.14162v1)
- Tokenizer (Table 1), exact token reconstruction on held-out text with the full hierarchy, and mean codebook utilization: OWT 99.99% / 83.10%; LM1B 99.86% / 89.31%; TinyStories 99.98% / 90.12%; XSum 99.79% / 86.20%; WMT14 98.79% / 61.89%; Sudoku easy/medium/hard reconstruction 100% (utilization 80.73 / 83.85 / 80.73). The paper says image-invisible errors matter in text because one wrong digit can invalidate Sudoku, and that observed reconstruction errors are often substitutions of related words. — [arXiv HTML 2610.14162](https://arxiv.org/html/2610.14162v1)
- Generation quality is free-running next-scale generation scored by an external LM, not by a target AR LLM’s accept/reject. On LM1B, a bar chart lists Gen. PPL 93.1 for NCM vs 97.3 for DUO, 115.5 for MDLM, 112.3 for FLM, 124.6 for CANDI, 295.4 for RDLM, and the caption says NCM uses 4.9× fewer training tokens. A multi-scorer table lists NCM 97M trained on 13.36B tokens with average Gen. PPL 83.24 (Mistral-7B 59.69, Llama-3.1-8B 75.23, Gemma-2-9B 106.41, Qwen-3-14B 91.64) versus DUO 139M / 66B tokens at average 92.52. GPT-2-large is the reference for the main Gen. PPL plots. — [arXiv HTML 2610.14162](https://arxiv.org/html/2610.14162v1); [ar5iv](https://ar5iv.labs.arxiv.org/html/2610.14162)
- Coarse vs fine, qualitative and training, not a ruin-rate: Figure 3 caption says coarse levels fix the high-level idea and finer levels fill in and correct the content. One printed WMT trajectory starts garbled (“However, it is a important to the the the and the.”) and is still being corrected at later steps. Training uses conditioning corruption: a fraction of conditioning codes are replaced by other codes from the same level’s codebook, labels stay clean, “to improve robustness to accumulating VQ code errors.” Replacement probabilities “vary by level”; numeric ablation is deferred to Appendix F.3 and was not extracted. — [arXiv HTML 2610.14162](https://arxiv.org/html/2610.14162v1)
- NCM mentions DFlash only as an example of diffusion drafters used in speculative decoding. It does not report NCM as that drafter. — [arXiv HTML 2610.14162](https://arxiv.org/html/2610.14162v1)
- Cumulative reconstruction vs depth (appendix note on ar5iv): reconstruction accuracy increases across levels and reaches 99.99% at full resolution on the plotted setting; four hierarchy depths all reconstruct at least 99.95% of tokens; Gen. PPL improves as the number of levels grows, with diminishing returns. — [ar5iv 2610.14162](https://ar5iv.labs.arxiv.org/html/2610.14162)

### Inferences
- High full-hierarchy reconstruction means the tokenizer is not the bottleneck. Free-running Gen. PPL far worse than a reconstruction ceiling means generator errors, including coarse-code errors, dominate sample quality. That is an inference from the two metrics, not a paper-stated causal ablation.
- Conditioning corruption is evidence the authors expect earlier-level code errors to hurt later levels, and that they try to make later levels robust rather than assuming a coarse error is fatal. The qualitative trajectory shows later levels editing earlier text. Neither is a measured “irrecoverable coarse error” rate.
- VAR’s image FID drop from 18.65 to 1.73 does not transfer to text tokens.

### Gaps
- VAR coarse-error ablation: not found.
- Any other text VAR / “visual autoregressive language model” with LM perplexity besides NCM: not found in this pass.
- NCM codebook size, level lengths, and Appendix F.3 replacement-rate deltas: not extracted from the paper (secondary blogs state schedules; those numbers are not used).
- Exact XSum ROUGE-L digits: the abstract says “matches”; the body says NCM beats AURORA-LM on ROUGE-1/2/L at 16× fewer NFE without guidance. The numeric ROUGE cells were not reliably extracted.

## Residual-VQ text autoencoders: reconstruction and collapse

### Takeaway
Two text-side residual quantizers report reconstruction: NCM’s residual VQ tokenizer recovers 98.79–99.99% of tokens, and LCM’s SONAR RVQ recovers about 70% of continuous auto-encoding BLEU at 64×8192. Neither reports classic dead-code collapse as the failure mode. The weakest utilization number found is NCM’s 61.89% mean codebook use on WMT14. Large-vocab collapse is discussed for generic VQ, not measured as a text-LM BPE residual-VQ result here.

### Cited Findings
- NCM residual hierarchy reconstruction and utilization: see Table 1 numbers above. Lowest utilization 61.89% (WMT14). No “codebook collapse” sentence in the fetched NCM sections. — [arXiv HTML 2610.14162](https://arxiv.org/html/2610.14162v1)
- LCM SONAR RVQ: 64 codebooks × 8192 units; auto-encoding BLEU around 70% of continuous SONAR at 64 codebooks; Quant-LCM called limited because the space was not trained to be quantizable and needs many large codebooks. — [arXiv HTML 2412.08821v2](https://arxiv.org/html/2412.08821v2)
- NCP product quantization (\(32 \times 128\)) is a different quantizer (partitioned features, not residual scales) and has no reconstruction accuracy. — [ar5iv 2609.10715](https://ar5iv.labs.arxiv.org/html/2609.10715)
- Residual VQ applied to LLM KV cache, not to text tokens: depth 8 “recovers most of the performance of the unquantized model”; depth 6 “brings significant drop,” especially MMLU and GSM8K; about 5.5× compression vs FP16. — [arXiv 2410.15704](https://arxiv.org/abs/2410.15704); [HTML](https://arxiv.org/html/2410.15704)
- Codebook-collapse literature found in this pass is not a text BPE residual autoencoder: a VQ study says random init and limited encoder capacity cause token and embedding collapse — [arXiv 2603.17052](https://arxiv.org/abs/2603.17052). A survey-style page describes the classical dead-code failure but is not a primary experiment — [Emergent Mind topic page](https://www.emergentmind.com/topics/codebook-collapse).

### Inferences
- For a lossless text drafter, NCM shows a residual hierarchy can be an almost-lossless tokenizer (exact token match ≥98.79% on the reported sets) while still leaving generation errors to the prior. Tokenizer exact-match is not the same as free-running exact-match.
- LCM’s 70% relative BLEU shows residual quantization of a frozen sentence space can destroy a large fraction of decodability even with 64 codebooks. That is a poor substrate for a lossless token drafter unless a token model re-verifies.
- \(128^{32}\) (NCP) and \(8192^{64}\) (LCM RVQ, if treated as a product) are capacities, not measured used-code counts. Collapse cannot be inferred from capacity alone.

### Gaps
- A SoundStream/RQ-VAE paper that residual-quantizes BPE or token sequences and reports both exact reconstruction and dead-code rate on a large codebook: not found. NCM is the closest.
- Numeric dead-code fraction for LCM’s 8192-unit codebooks: not found.

## Latent or continuous-plan LMs under verification

### Takeaway
Coconut reports final-answer accuracy, not speculative verification and not a teacher-forced versus free-running generation gap. No Planning-Tokens or Quiet-STaR result with a verification metric was found. SONAR-LLM’s token cross-entropy is through a frozen sentence decoder, not rejection sampling.

### Cited Findings
- Coconut feeds the last hidden state back as the next input embedding instead of decoding it to a token. Table 1 accuracy (and generated tokens): GSM8K CoT 42.9% / 25.0 tokens, No-CoT 16.5% / 2.2, Coconut 34.1% / 8.2; ProntoQA CoT 98.8% / 92.5, Coconut 99.8% / 9.0; ProsQA CoT 77.5% / 49.4, Coconut 97.0% / 14.2. These are task accuracies after latent thoughts, not acceptance rates. — [arXiv HTML 2412.06769](https://arxiv.org/html/2412.06769v2); [abs](https://arxiv.org/abs/2412.06769)
- A within-paper comparison says that even when Coconut is forced to emit a full language chain (\(k=0\)), answer accuracy is still higher than CoT. That is not a teacher-forced versus free-running perplexity gap, and no numeric \(k=0\) accuracy was extracted. — [arXiv HTML 2412.06769v1](https://arxiv.org/html/2412.06769v1)
- Clock time on one A100, batch 1: GSM8K No-CoT 0.03s, CoT 0.26s, Coconut 0.09s. — [arXiv HTML 2412.06769v1](https://arxiv.org/html/2412.06769v1)
- Planning Tokens and Quiet-STaR, generation metric under verification or teacher-forced vs free-running gap: not found in this pass.
- SONAR-LLM XSum ROUGE-L 19.3 vs MSE LCM 12.2 is free generation scored by overlap, with token-level supervision through the decoder during training. — [arXiv HTML 2508.05305](https://arxiv.org/html/2508.05305v1)

### Inferences
- Continuous thoughts are not discrete draft tokens. Nothing in Coconut supplies an acceptance length against a target AR distribution.
- Coconut’s GSM8K regression versus CoT (34.1 vs 42.9) is a warning that an unverified latent plan can lose task accuracy even when logical-search tasks improve. It is not evidence about lossless drafting.

### Gaps
- Teacher-forced vs free-running token NLL for Coconut, Planning Tokens, and Quiet-STaR: not found.
- Any of these used as a speculative drafter: not found.

## What do the six hierarchical or parallel speculative-decoding papers actually draft?

### Takeaway
All six draft or verify tokens (or token-tree branches), not concepts. Five state a lossless or distribution-preserving guarantee in the abstract or introduction. None reports a recoverable coarse concept error. The closest “recovery” mechanisms are intra-block self-refinement before verification (JetSpec++) and hierarchical resampling of token branches after the last accepted token (HSD). A wrong early draft token still ends the accepted prefix under ordinary verification; DSpine’s own motivation is that an inconsistent predecessor shortens that prefix.

### Cited Findings
- HSD, arXiv 2601.05724. Draft unit: tokens from an existing drafter (integrated into EAGLE-3); the “hierarchy” is successive resampling distributions over branches, not concepts or layers of meaning. Verification: “provably lossless”; recovers the target distribution in expectation. Headline: average +6.7% decoding speed across benchmarks and model sizes, up to +12.3% on individual datasets; “integrating HSD into EAGLE-3 yields over a 12% performance gain” versus that EAGLE-3 baseline, “without compromising distribution fidelity.” Whether a coarse semantic error is recoverable: not applicable; resampling happens immediately after the last accepted token. — [abs](https://arxiv.org/abs/2601.05724); [HTML v2](https://arxiv.org/html/2601.05724v2)
- DSpine, arXiv 2609.36173. Draft unit: a block of token positions, drafted in parallel; each layer injects the predecessor’s predicted feature into the successor. Not concepts. Paper abstract: at temperature 0 on Qwen3-8B, seven-benchmark mean acceptance length 3.77 (DFlash) → 4.82 (+27.8%); SGLang throughput +23.3% vs DFlash. AlphaXiv’s reading of the paper adds: same ShareGPT retrain, five-layer backbones, block size 16, six epochs, 15 proposed tokens; DSpine mean 4.82 vs strongest baseline DFlash2 at 4.39 (+10.0%). A later Hugging Face card for a retrained checkpoint reports much higher lengths (Qwen3-8B, T=0, Open-PerfectBlend retrain: DSpine average 7.83 vs DFlash2 7.55 vs DFlash 5.92). Those card numbers are a different setup from the abstract’s 3.77→4.82. Coarse error: the paper says independent token picks produce inconsistent continuations that shorten the accepted prefix, so an early bad token is not recovered at verification; injection is an attempt to avoid that inside the drafter. Explicit “lossless” sentence: not in the abstract; the method is standard speculative acceptance length. — [abs](https://arxiv.org/abs/2609.36173); [alphaXiv](https://www.alphaxiv.org/abs/2609.36173); [HF card](https://huggingface.co/tencent/Qwen3-8B-DSpine-b16)
- JetSpec++, arXiv 2610.19792. Draft unit: token blocks. Intra-block self-refinement iteratively corrects the drafter’s own parallel predictions; inter-block foresight starts the next block before verification; drafting is pipelined with verification. “Preserving the target model’s output distribution.” Headline: up to 7.45× wall-clock vs autoregressive decoding and 1.54× vs DSpark; up to 1,361 tokens/s for one request on GB200; average acceptance length 8.94 across seven benchmarks. Project page: at T=0 and verification budget 16, 8.94 tokens/round vs 6.89 for DSpark on Qwen3-8B. Self-refinement can change a draft before it is verified; the paper does not say a rejected coarse concept is repaired after verification. — [abs](https://arxiv.org/abs/2610.19792); [HTML](https://arxiv.org/html/2610.19792); [project page](https://jetspec-project.github.io/jetspecpp-web/)
- SpecPilot, arXiv 2611.02501. Draft unit: existing parallel token drafters (three drafters, four target models). An agent searches system/algorithm changes. “Token forcing checks every output token against stock.” “Without training or quality loss.” Headline speedup over AR: stock 4.46× → 10.87× at one request, and 1.94× → 3.93× with batching. Discoveries named in the abstract include a 12-bit weight format, tree verification for hybrid recurrent targets, two-level verification, and three batching mechanisms. What the two levels are (tokens vs concepts): not found in the abstract. — [abs](https://arxiv.org/abs/2611.02501); [HTML](https://arxiv.org/html/2611.02501v1)
- SpecOPD, arXiv 2611.02846. Draft unit: parallel token blocks. Architectures named: DFlash, DSpark, DFlash V2. Targets: 8B full attention, 27B full-linear, 125B sparse-linear. Trains with an E2E total-variation surrogate aimed at acceptance length. Headline: up to 41.45% higher mean acceptance length, 16.09× supervised-token throughput vs sequential-rollout on-policy distillation, up to 3.56× generation speedup vs autoregressive decoding. The abstract does not say “lossless,” “exact match,” or “rejection sampling.” Coarse-error recovery: not found. — [abs](https://arxiv.org/abs/2611.02846)
- D-REAL, arXiv 2611.10432. Draft unit: tokens from DSpark and DFlash2; target Qwen3-8B; ten benchmarks. Post-training maximizes expected acceptance length under stochastic verification (prefix survival along a draft-sampled prefix), with an unbiased gradient. Headline: one epoch improves benchmark-averaged mean acceptance length by up to 0.23 tokens at T=0 and T=1, and up to 9.4% end-to-end serving throughput. HTML detail at T=1: continued training / D-PACE drops macro-average \(\tau\) from 4.67 to 4.60 (DSpark) and 4.69 to 4.60 (DFlash2), while D-REAL changes those averages by +0.07 and +0.23. This is stochastic verification, not greedy exact-match only. Coarse concept recovery: not found. — [abs](https://arxiv.org/abs/2611.10432); [HTML](https://arxiv.org/html/2611.10432v1)

### Inferences
- “Hierarchical” in HSD and “two-level” in SpecPilot are verification or systems hierarchies over tokens. They are not evidence that a concept code was drafted and then checked by an AR LLM.
- If the acceptance rule is prefix exact-match or standard speculative sampling, a wrong token at position \(i\) is not repaired by a finer token at \(i+1\). DSpine states the prefix-shortening failure directly. JetSpec++ moves correction to before verification.
- D-REAL’s +0.23 token gain on an already-plateaued token drafter is small next to architecture gaps such as DSpine’s +1.05 vs DFlash (3.77 → 4.82) or JetSpec++’s 8.94 vs 6.89. Those comparisons cross papers and setups, so they are not a controlled ranking.

### Gaps
- SpecOPD and DSpine abstracts: explicit lossless theorem statement not found (SpecOPD omits it; DSpine reports acceptance length only).
- SpecPilot “two-level verification” mechanism: not found beyond the name.
- Per-benchmark acceptance tables for HSD, JetSpec++, and SpecOPD: not extracted beyond the headlines above.
- Whether any of the six can accept a later token after rejecting an earlier one inside the same concept: not found, and no concept unit is defined.

## Has anyone used NCM, or any discrete or continuous concept hierarchy, as a draft that an autoregressive LLM verifies?

### Takeaway
Not found as a concept-code drafter. The only measured link from a discrete concept module to lossless token verification is NCP-ArchPreview’s +4.17% mean accepted length (5.933 → 6.180) when a target concept vector conditions a token drafter. Stanford NCM, Meta LCM, and Quant-LCM generate in their own latent spaces and decode with their own decoders. No acceptance rate for concept IDs under AR-LLM verification was found.

### Cited Findings
- NCP injection result: macro mean accepted length 5.933 → 6.180 under exact speculative verification, horizon 16, four benchmarks, +0.04M parameters on a 1.1B token drafter. The concept is \(c_{\lfloor t/k \rfloor}\) from the target, not a sampled concept draft. — [ar5iv 2609.10715](https://ar5iv.labs.arxiv.org/html/2609.10715)
- NCM evaluates free-running generation with Gen. PPL, BLEU, ROUGE, and Sudoku accuracy. Speculative decoding appears only as a citation to DFlash. No acceptance length. — [arXiv HTML 2610.14162](https://arxiv.org/html/2610.14162v1)
- LCM and Quant-LCM decode with SONAR. No target-LLM verification experiment. — [arXiv HTML 2412.08821v2](https://arxiv.org/html/2412.08821v2)
- Searches for speculative decoding with vector quantization or “concept” draft-and-verify returned weight-quantized token drafters, not concept-code drafters: SPEQ reports a draft-token accept rate of 0.976 using a 4-bit bit-sharing draft of the same model — [arXiv HTML 2510.18525](https://arxiv.org/html/2510.18525v1). ML-SpecQD reports up to 2.72× over BF16 using MXFP4 weight drafts — [arXiv abs 2503.13565](https://arxiv.org/abs/2503.13565). Heterogeneous-vocabulary lossless SD reports up to 2.8× and is about tokenizer mismatch, not concepts — [arXiv abs 2502.05202](https://arxiv.org/abs/2502.05202).
- A search for “coarse-to-fine speculative decoding acceptance rate” did not return a concept-hierarchy drafter. Unrelated token-level acceptance figures that did appear: domain node-acceptance near 0.54 for a 1B drafter vs a 7B target (chat 0.565, code 0.538, math 0.518, reasoning 0.532; n=99,768 nodes) — [arXiv HTML 2604.14682](https://arxiv.org/html/2604.14682v2). Those are not concept accuracies.

### Inferences
- The negative result is about the role, not about the existence of concept models. Discrete concept hierarchies exist (NCP product quantization, NCM residual scales, LCM RVQ). What was not found is the experiment “propose concept codes, reject or accept them by a token AR LLM, emit the target’s tokens.”
- NCP’s +4.17% is evidence that a concept vector can help a token drafter slightly. It is not evidence that drafting at concept granularity is lossless or that a coarse concept error can be corrected by the verifier. The verifier never sees a concept decision; it sees tokens.
- For a lossless drafter, NCM’s ≥98.79% tokenizer reconstruction is the encouraging half, and the missing half is a measured acceptance length when those codes are proposed instead of ground-truth codes. That half is not in the paper. LCM’s ~70% relative auto-encoding BLEU is evidence that a sentence-level discrete concept, decoded without a token verifier, is not lossless.

### Gaps
- A paper that drafts NCM codes, LCM sentence embeddings, or NCP codewords and reports acceptance length or speedup against an AR LLM: not found.
- Concept-level top-1 accuracy inside NCP-ArchPreview, which would be the natural upper bound on concept acceptance: not found.
- Numeric NCM conditioning-corruption ablation (how much a replaced coarse code changes token recovery or Gen. PPL): referenced as Appendix F.3, not extracted.
