# NCM design and settings review with codex and grok (2026-11-14)

Question (user): research the original paper and other NCM projects; are our design and settings
right? Same brief to codex (gpt-6.1-sol) and grok, plus our own reading of the paper (Table 5,
App. C/F/I), the released configs and `models/quant.py` (commit c702742). Reviewer answers are
verbatim below; claims marked "checked" were re-verified by us, the rest are the reviewers'.

## Findings

Tokenizer (M0)
1. M0e (LM1B shape: L=128, 12 levels, codebook 8192, 4 decoder blocks, no pre-quant dropout) is the
   best-supported released recipe; both reviewers agree, codex calls it "defensible, not established
   best". Measured: step 5,000 token accuracy 96.5 / 97.6 / 99.2 (chat/code/math) vs M0d 94.3 / 95.7 /
   98.4 at the same step, with spans twice as long.
2. Remaining deviations from the LM1B recipe (checked in `config/eval/lm1b-small.yaml`): dual nested
   dropout (every step reconstructs the full hierarchy with weight 0.7 and a truncated one with 0.3;
   the released `quant.py` does not implement `dual`, only one batch-wide cutoff with probability
   0.5, which is what we run), GroupNorm 4 (ours 8), encoder positions off / decoder positions on
   (ours the reverse), level-0 cascade off, AdamW beta2 0.99, weight decay 0.05, peak lr 3e-4, batch
   2,048. Table 5 and the released YAMLs disagree in places (OWT codebook 4096 vs 8192; EMA 0.95 vs
   0.99 for XSum/WMT14).
3. Exposure: M0e sees ~2.0B token positions, 29% of LM1B's 6.8B and 3% of OWT's 61.7B (checked).
4. Vocabulary is the confound: the >= 99.8% tokenizers have 8k-50k vocabularies; the only
   large-vocabulary one (WMT14, 151k, learned tables, 144M parameters) is the worst (98.79%,
   utilisation 61.9%). Frozen pretrained tables are not in the paper.
5. No lossless tokenizer and no code benchmark exist in the paper or in any project found. The
   authors left dead-code restarts out because they improve reconstruction but hurt generation
   (issue #16, checked). A commenter there reports continuous RVQ pooling blurring brackets and
   indentation on 64-token Python, fixed by a deterministic dyadic token subsample (self-reported).
6. Both reviewers propose an exact side channel (sparse position + token-id corrections for the
   positions the decoder gets wrong) as the only way to guarantee code round trips; oversampling
   rare tokens and trainable low-rank corrections of the frozen tables are cheaper first steps.

Generator (M1)
7. Qwen3.5-9B is a hybrid: 24 `linear_attention` layers (causal, recurrent state) and 8
   `full_attention` layers (checked in its config.json). NCM needs bidirectional attention inside a
   level and block-diagonal isolation between levels; changing a mask only affects 8 of 32 layers.
   This must be resolved before M1 (codex's top architectural risk).
8. A span costs 641 generator positions (1+8+...+128) for 128 tokens over 12 sequential forwards:
   ~5x the token-positions of AR. Latency at concurrency 1 can improve; throughput per GPU does not.
9. M1 must condition on the cumulative continuous reconstruction, train with clean labels and
   corrupted conditioning (mean ~0.5, decreasing over levels), use level embeddings and
   block-diagonal attention (Gen. PPL 15.41 vs 19.63 block-causal), alpha in [0, 1], greedy (top-k 1)
   sampling for conditional tasks. Multi-span generation with a KV cache and conversion of a
   pretrained AR model are not in the paper.
10. Other projects (reviewers' findings, self-reported): ncm-mlx (checked: its NCM models "still lose
    clearly to an autoregressive baseline on text", TinyStories Gen. PPL 24.4 vs AR 11.5 at 87M;
    less corruption lowered teacher-forced loss but made samples worse), MCL4 grafts on Llama/Granite
    (Granite reconstruction 85%), ncm-modular (consecutive spans, seams unresolved), ConceptLM (adapts
    Llama 8B but keeps token decoding). None reaches a <= 5% quality loss on a pretrained LLM.

## Decisions

- Keep M0e running; judge it by the HumanEval/GSM8K round trip, not token accuracy.
- Next tokenizer run (only if M0e's round trip justifies it): dual 0.7/0.3 nested dropout, LM1B
  positions/GroupNorm/optimizer settings, rare-token oversampling, >= 7B positions.
- Measure how many corrections per span an exact side channel would need at M0e's final accuracy.
- Do not start M1 before (a) the round trip passes and (b) there is a design for the hybrid
  linear-attention layers.
- Stop: WMT14 recipe, pre-quant dropout 0.25, dead-code restarts, counting 128/12 as a speedup.

## Brief

You are a research reviewer. Task: check whether our NCM tokenizer/generator design and settings are right, by researching primary sources. Use web search. Cite URLs / file paths and quote numbers. Separate what you verified from what you infer. Do not modify files.

## Sources to research
1. The NCM paper: "Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation", arXiv 2610.14162 (https://arxiv.org/abs/2610.14162, HTML https://arxiv.org/html/2610.14162v1), especially Sec. 3.3-3.4, Tables 1, 4, 5, Appendices C (training details), F (ablations: number of levels, level schedule, attention pattern, loss weighting, code corruption, sampling), I (limitations).
2. Official code https://github.com/HazyResearch/ncm (a local clone is at ./ncm-src if you can read files: models/vqvae.py, models/quant.py, models/vae.py, config/eval/*.yaml), the HF collection https://huggingface.co/collections/hazyresearch/ncm, and the two blog posts (hazyresearch.stanford.edu/blog/2026-10-12-ncm, scalingintelligence.stanford.edu/blogs/ncm/).
3. Any OTHER NCM projects: third-party reproductions, forks, issues/discussions on the official repo, follow-up papers, or attempts to apply NCM / multiscale residual VQ text tokenizers / next-scale prediction (VAR-style) to large pretrained LLMs, large vocabularies, code, or chat. Also closely related text work (e.g. hierarchical/residual VQ text autoencoders, latent/"concept" LMs, parallel coarse-to-fine text decoders) that reports reconstruction accuracy or a way to guarantee exactness.

## Our project
Goal: turn Qwen/Qwen3.5-9B (vocab 248,320) into an NCM generator that is much faster than AR, with quality within ~5% on GSM8K and HumanEval. Stage M0 = span tokenizer; stage M1 = the 9B (LoRA or full FT) predicts the code hierarchy of the next span given the prompt and previous spans via KV cache; long answers = consecutive spans.

Tokenizer runs so far (official VQVAE code, our own training loop: CE + vq loss, AdamW 0.9/0.95, lr 5e-4 -> 5e-5 cosine, warmup 1000, wd 0.01, clip 1, bf16 autocast, EMA codebooks decay 0.99, commitment 0.25, fine_drop prob 0.5 min_keep 1, quant_resi 0.5, learned_l1_sampling + cascade base 4, pre_quant_groupnorm 8, use_qk_norm, width 384, 6 heads, 0 encoder blocks):
- M0 (WMT14 recipe: L=64, levels 1,4,8,14,20,26,32,40,48,56,64, codebook 4096, 1 decoder block, pre_quant_dropout 0.25, cascade norm on, learned 248k embedding + output tables; data 9B answers + 150M public tokens): 12.5k steps x 1024 blocks: token acc chat 96.2 / code 96.9 / math 99.3. 88% of errors on tokens seen < 100 times.
- M0b (same but frozen Qwen input embedding -> learned Linear(4096->384), and output = frozen Qwen lm_head applied to RMSNorm(Linear(384->4096))): 10k steps: 97.9 / 98.3 / 99.5.
- M0d (M0b + 1.7B tokens of public code/math/web/chat, 30% 9B answers): 10k steps: 98.1 / 98.4 / 99.6. Round trip (encode->decode the 9B's own correct answers, re-grade): GSM8K 86.3% -> 86.3%; HumanEval 82.9% -> 50.0%. Typical breaks: `*=` -> `+=`, `(paren` -> `(character`, ` MAD` -> ` SAM`, bracket/quote swaps, ` litres` -> ` gallons`.
- M0e (running; LM1B-style: L=128, levels 1,8,16,24,32,40,52,64,76,92,108,128, codebook 8192, 4 decoder blocks, pre_quant_dropout 0, cascade norm off, dropout 0.1, frozen Qwen tables, batch 512 blocks x 30k steps = ~2B token positions): step 2.5k token acc 83.6 / 88.7 / 95.3 (M0d at the same step: 76.1 / 82.5 / 92.1).
Our estimate: HumanEval needs ~99.99% token reconstruction for <= 2% relative loss.

M1 plan (not started): 9B predicts levels coarse to fine, positions within a level in parallel, one 9B forward per level (12 forwards per 128 tokens), tokenizer decoder -> tokens; planned additions from the paper: code corruption of conditioning codes, conditioning on cumulative continuous reconstruction z^(k-1), level loss weights, block-diagonal attention inside the span, optional CFG / MBR.

## Questions (answer each, numbered)
1. Tokenizer recipe: from the paper's Table 5 / App. C and the released configs, what exactly distinguishes the >= 99.86% tokenizers (OWT, LM1B, TinyStories, XSum) from WMT14 (98.79%)? Give the full per-dataset table you can verify (L, K, schedule, codebook, dec blocks, dropout, pre-quant dropout, nested dropout prob, EMA, lr, batch, updates, params, training token count). Is our M0e the best-supported choice for a 248k-vocab, code-heavy domain, or should it be different (e.g. OWT-style 1-block decoder with square-ish schedule, more levels, larger codebook, wider latent, encoder blocks)?
2. Training-loop fidelity: the repo ships no training loop. From the paper, is anything we are likely missing or doing wrong: loss terms and weights (Eq. 9), nested-dropout sampling, codebook init (k-means? dead-code restarts?), EMA details, LR schedule, label smoothing, weight tying, mixed precision, padding handling (do they mask pad in the loss? how are short sequences handled?), how many updates before reconstruction saturates?
3. Is replacing the embedding/output tables with frozen pretrained-LM tables supported or contradicted by anything in the literature? Better alternatives for a huge vocab with a long tail (e.g. a copy/pointer path, byte/char fallback, tied or factorized embeddings, larger width, an exact residual "correction" channel)?
4. Exactness: does the paper or any related work give a way to make the tokenizer lossless or near-lossless for code (e.g. extra finest level, residual token-id channel, verification by the decoder, error-correcting pass)? What reconstruction do they report on anything code-like?
5. Generator: list every design feature and hyperparameter of the paper's generator that M1 must replicate, with the ablation numbers showing which matter most (corruption schedule, attention pattern, level weighting alpha, level-0 handling, CFG, sampling temperature/top-k/top-p, MBR). Does anything in the paper or elsewhere indicate whether a pretrained AR LLM can be converted (LoRA/full FT) into such a generator, and whether multi-span (segment-wise) generation with a prefix has been tried?
6. Other NCM projects: what did you find (links), what settings/results did they report, and what lessons apply to us?
7. Verdict: top 5 concrete changes to our design/settings, ranked by expected impact, and anything we should stop doing. Keep the whole answer under 900 words.

## grok

I'll research the NCM paper, official code/configs, and related work first, then answer the seven questions from primary sources.I’ll pull the paper tables, official configs, and related work in parallel so the numbered answers can cite primary numbers.**Verdict:** M0e is the right *paper* recipe (LM1B). It is not enough for 248k-vocab code. M0d’s HumanEval 82.9% → 50.0% is the tokenizer, not M1. Do not start M1 until round-trip HumanEval is near the AR baseline.

**Verified vs inferred:** numbers from the paper/HTML, `ncm-src` configs, GitHub issues, and third-party writeups are **verified**. Design recommendations for Qwen 9B are **inferred**.

### 1. Tokenizer recipe
Table 1 ([arxiv.org/html/2610.14162v1](https://arxiv.org/html/2610.14162v1)): recon **OWT 99.99 / LM1B 99.86 / TinyStories 99.98 / XSum 99.79 / WMT14 98.79 / Sudoku 100**; util **83.1 / 89.3 / 90.1 / 86.2 / 61.9 / ~81**. XSum is **99.79**, not ≥99.86.

Table 5 + App. C (token positions = updates × batch × L):

| | OWT | LM1B | TinyStories | XSum | WMT14 | Sudoku |
|---|---|---|---|---|---|---|
| L / K / V_k | 1024 / 32 / **4096** | 128 / 12 / 8192 | 128 / 12 / 4096 | 48 / 8 / 4096 | 64 / 11 / 4096 | 89 / 12 / 16 |
| text vocab | 50304 | 30528 | 8192 | 50304 | **151680** | 12 |
| enc / dec | 0 / **1** | 0 / **4** | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 |
| d / heads | 384 / 6 | 384 / 6 | 384 / 6 | 384 / 6 | 384 / 6 | 256 / 4 |
| dropout | 0.3 | 0.1 | 0.3 | 0.1 | 0.1 | 0.3 |
| nested drop | 0.5 | **0.7/0.3 dual** | 0.5 | 0.5 | 0.5 | 0.5 |
| EMA / β2 / wd | 0.99 / 0.95 / 0.01 | **0.99 / 0.99 / 0.05** | 0.95 / 0.95 / 0.01 | 0.95 / 0.95 / 0.01 | 0.95 / 0.95 / 0.01 | 0.99 / 0.99 / 0.01 |
| lr / warmup | 3e-4→3e-5 / 1k | 3e-4 / 1k | 5e-4→5e-5 / 1k | 5e-4 / 1k | 5e-4 / 1k | 5e-4 / 500 |
| updates × batch | 78432 × 768 | 25957 × 2048 | 30k × 4096 | 120001 × 4096 | 60k × 4096 | 10k × 1024 |
| tok positions | 61.68B | 6.80B | 15.73B | 23.59B | 15.73B | 0.91B |
| params | 111.2M | 111.4M | 36.4M | 60.6M | 144.0M | 14.9M |

Released yaml (`ncm-src/config/eval/`): **pre_quant_dropout 0** on OWT/LM1B/TS/Sudoku, **0.25** on WMT14/XSum; LM1B `dec_layers: 4`, `learned_l1_cascade: false`, `fine_drop dual w_full=0.7 w_trunc=0.3`; WMT/XSum `cascade_norm` on. **Mismatch:** Table 5 OWT V_k=4096 vs `owt.yaml` **8192**; Table 5 WMT/XSum EMA 0.95 vs yaml **0.99**. OWT schedule is squares 1…1024.

WMT14 is the only large-vocab analog (Qwen3-0.6B, 151k) and has the worst recon and util. High-acc runs use 8k–50k vocabs.

**M0e is the best-supported recon recipe** (LM1B: L=128, same 12-level ladder, 8192, 4 decoder blocks, dropout 0.1, pre-quant drop 0, cascade off). **Inferred:** do not copy WMT14 1-block / pre-quant 0.25 / cascade-norm for 248k code. Encoder blocks stay 0 in every released tokenizer.

### 2. Training-loop fidelity
**Verified (Eq. 9, `quant.py`, App. C):** CE over all L positions + β=0.25 commitment MSE + EMA codebook (codebook term is EMA, not a learned MSE). Nested dropout = `fine_drop` (LM1B dual 0.7/0.3). Codebooks `normal_()`, `requires_grad_(False)`; `kmeans_init` default **False**; `restart_threshold` default **0**. STE **level 0 only**. AdamW 0.9/0.95 (LM1B 0.99), cosine to 0.1× peak, clip 1, **bf16**. No label smoothing. Tokenizer embed/output **untied**. Padding: `_adjust_length` right-pads; App. C counts pad in token budgets; **no pad mask in Eq. 9**. Recon saturates by ~10k–30k on small L; OWT used 78k.

**Likely missing vs LM1B:** dual 0.7/0.3 nested drop, wd 0.05, β2 0.99, batch ~2048, tokenizer code-corruption in LM1B yaml (`aux.code_corruption`). **Inferred:** M0e 512×30k ≈ 2B positions is **3× below** LM1B’s 6.8B.

### 3. Frozen LM tables
**Not in the paper.** Closest: frozen T5-small as *prefix* encoder, +0.24 ROUGE-L, +35M params (Table 10); they still used the VQ encoder. WMT14 *learns* 151k tables (144M tokenizer). **Inferred:** frozen Qwen tables help the tail (your M0→M0b jump) but cannot invent unused ids (`*=`→`+=`). Better: oversample rare ids; larger d; **exact token residual**; byte/char fallback. Copy/pointer is **inferred**, not NCM.

### 4. Exactness
**No lossless code tokenizer in the paper.** Sudoku 100% with |V|=12. OWT errors (Table 11): entities, numerals, markup, word boundaries — same class as your operator/unit swaps. App. I: fixed K,ℓ_k; tokenizer trained separately. **Inferred:** extra finest **token-id residual** (or dyadic exact anchors) is the only way to 99.99% on 248k code. Issue #16 toy “dyadic stride → 100% `ast.parse`” is **not** a published NCM result.

### 5. Generator (must replicate)
From §3.4, Alg. 1–2, App. F, Table 2, yaml:

- Condition on **cumulative ẑ^(k−1)** (not codes alone).
- **Code corruption**, linear-decreasing; TinyStories mean ~**0.50** (F.3); “moderate helps, too much hurts.”
- Train all levels **one forward**; sample **K sequential** levels, parallel within level.
- **Block-diagonal** attn: GenPPL **15.41** vs block-causal **19.63** (Table 9).
- Level weights α: **0–1** GenPPL 18.9–19.0; α=1.5 → 19.4 (F.2.3). Default α=0 is ELBO.
- Prefix LM: XSum P=512, WMT P=96, Sudoku P=91; `raw_token_prefix: true`.
- `predict_level0` **true** on WMT; **false** + `query_level0` on LM1B/XSum/Sudoku; unigram L0 on OWT/TS.
- Sample: T=1.0; WMT/XSum **top_k=1**; Sudoku **T=0**; uncond top_k=0 top_p=1.
- CFG: XSum **w=0.5**, WMT **w=1.0** (Fig. 14). MBR: XSum **K=16** (29.6 RL), WMT **K=12** (25.2 BLEU).
- `tie_heads_to_codebook` on OWT/TS/XSum; `post_upsample_conv` on OWT/LM1B.
- Generators **from scratch**, 88–120M (scale study to **725M** on frozen tokenizer). **No LoRA/FT of a pretrained AR LLM.** Prefix-conditioned **one span of length L**. Consecutive multi-span + KV cache is **not** in the paper (App. I: even short text runs the full hierarchy).

### 6. Other NCM projects
- Official repo: **eval only** ([issue #10](https://github.com/HazyResearch/ncm/issues/10)). HF: [hazyresearch/ncm](https://huggingface.co/collections/hazyresearch/ncm).
- [janmus1c/ncm-mlx](https://github.com/janmus1c/ncm-mlx): TinyWorld codec **100%** exact; own TinyStories codec **99.816%** non-pad; 87M NCM GenPPL **24.40** vs AR **11.48**.
- [#16](https://github.com/HazyResearch/ncm/issues/16): dead-code restart bug; **Kumbong: restarts help recon, hurt generation.** Commenter tried Granite/Llama “frozen-donor” grafts (anecdote).
- [#19](https://github.com/HazyResearch/ncm/issues/19) / ncm-modular: prefix-conditioned training, no published 9B result.
- Inference ports: HexaField apple-silicon; no 248k-code generator.

### 7. Top 5 changes
1. **Block M1** until encode→decode HumanEval is ≳80% (preferably ≳99.5% token acc on code). 98.4% cannot hit ~5% relative loss.
2. **Keep M0e (LM1B)**; add **dual 0.7/0.3 nested drop**, **wd 0.05, β2 0.99**, batch closer to **2048**, **~7B+ positions**.
3. **Add an exact finest channel** (token-id residual / copy of rare ids). Frozen 248k softmax will keep swapping `*=`/`+=`.
4. **Oversample tokens with freq &lt;100** (your 88% error mass).
5. **M1 must be ẑ-conditioned next-scale**, not “9B predicts a flat code list.” Use block-diagonal, ~0.5 decreasing corruption, α∈[0,1], greedy for code.

**Stop:** WMT14 1-block L=64 for code; pre-quant dropout 0.25; expecting CFG/MBR to fix tokenizer substitutions; dead-code restarts without a gen metric; treating multi-span KV as a paper feature.

## codex

**1. Tokenizer recipe — verified.** XSum reports **99.79%**, below your stated threshold. P/R below means paper/released config; training positions are calculated, including padding and repeated exposure. Sources: [Tables 1/5](https://arxiv.org/html/2610.14162v1#A3), [released configs](https://github.com/HazyResearch/ncm/tree/main/config/eval).

| Setting | OWT | LM1B | TinyStories | XSum | WMT14 |
|---|---|---|---|---|---|
| Accuracy % |99.99|99.86|99.98|99.79|98.79|
| L; K |1024;32|128;12|128;12|48;8/9 P/R|64;11|
| Schedule |1²,…,32²|A|A|4,6,10,14,20,26,32,40,48|B|
| Codebook P/R |4096/8192|8192|4096|4096|4096|
| Decoder blocks |1|4|1|1|1|
| Dropout/prequant |.3/0|.1/0|.3/0|.1/.25|.1/.25|
| Nested dropout |.5|.7 full+.3 truncated|.5|.5|.5|
| EMA P/R |.99|.99|.95|.95/.99|.95/.99|
| Peak→floor LR |3e−4→3e−5|3e−4→3e−5|5e−4→5e−5|5e−4→5e−5|5e−4→5e−5|
| Batch×updates |768×78,432|2048×25,957|4096×30,000|4096×120,001|4096×60,000|
| Paper parameters, M |111.2|111.40|36.43|60.63|144.00|
| Positions, B |61.68|6.80|15.73|23.59|15.73|

A=1,8,16,24,32,40,52,64,76,92,108,128; B=1,4,8,14,20,26,32,40,48,56,64. All use width384, zero encoder blocks, six decoder heads.

**Inference:** M0e is defensible, not established best. Dataset, vocabulary and training-budget differences confound architectural comparisons.

**2. Training fidelity — verified discrepancies.** [LM1B config](https://github.com/HazyResearch/ncm/blob/main/config/eval/lm1b-small.yaml) additionally specifies **GroupNorm4, encoder positions off, decoder positions on, cascade off, Adam β₂=.99, wd=.05**.

Implement dual reconstruction explicitly: released `models/quant.py:263–268,519–531` ignores dual weights, averages VQ losses across all levels, then truncates decoder input. Eq.9 instead describes cutoff-specific VQ terms. Ordinary dropout samples one batch-wide uniform cutoff **1…K−1**; commitment **.25** is already included.

Codebooks initialize Gaussian; k-means/restarts default off; EMA uses distributed sums and ε=1e−5 smoothing. An author confirms restarts improved reconstruction but **worsened generation**. [Issue16](https://github.com/HazyResearch/ncm/issues/16)

BF16 is documented; label smoothing, training pad masking, exact LR interpolation and saturation time remain unverified. The wrapper right-pads/chunks; that does not establish training masking. M0e’s **1.97B positions ≈29%** of LM1B’s exposure.

**3. Frozen tables.** Indirect support exists: [ELF](https://arxiv.org/html/2605.10938v1#S5) favors pretrained embeddings, but trains unembedding separately. Your improvement supports retaining pretrained initialization, not necessarily freezing both tables.

**Inference:** test trainable low-rank table corrections/biases and rare-token-balanced batches before widening everything. Pointer paths only recover strings available to copy; byte fallback requires explicit reversible encoding.

**4. Exactness.** NCM supplies no lossless guarantee or code reconstruction benchmark. Sudoku reaches **100% with vocabulary12**; extra levels alone guarantee nothing. [Tables1/5](https://arxiv.org/html/2610.14162v1)

**Proposal:** sparse position+token-ID corrections guarantee round-trip when supplied by the encoder; generated corrections remain fallible. [Speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html) provides an AR distribution guarantee, but adapting sampling to latent NCM requires handling draft probabilities. Greedy AR verification is simpler.

Your estimate corresponds to approximately **200 independent vulnerable tokens**: .9999²⁰⁰≈98.02%. Executable round-trips are the stronger acceptance criterion.

**5. Generator.** Verified implementation requirements: frozen tokenizer; resized cumulative continuous reconstructions; level embeddings, RoPE/QK normalization; clean labels; corrupted conditioning; within-level bidirectional attention plus prefix access; isolated target levels; code heads. LM1B adds kernel3 residual convolutions/shared head. Level0 uses unigram seeds for OWT/TinyStories, queries for LM1B/XSum/Sudoku, prefix prediction for WMT. [Code](https://github.com/HazyResearch/ncm/blob/main/models/next_concept.py)

Reference LM1B training: **12×768,12 heads;100k×512;LR3e−4→3e−5,warmup1000;AdamW(.9,.95),wd.1,clip1,dropout.1**. These are not validated 9B settings.

[Ablations](https://arxiv.org/html/2610.14162v1#A6): diagonal versus block-causal **PPL15.41/19.63;5.65/7.90s**. α=0…1 gives **18.9–19.0**, versus **19.4** at1.5. Exponential/quadratic/linear schedules give **57.7/59.6/88.4**. Increasing levels4→8→16→32 reduces PPL approximately52%/32%/20%. Corruption’s selected mean is **.50**, linearly decreasing; endpoints remain [unresolved](https://github.com/HazyResearch/ncm/issues/24).

Released sampling: unconditional T1/k0/p1; conditional T1/k1/p0; Sudoku T0. CFG ramps across levels; WMT w1 improves BLEU22.6→23.9. MBR12/16 candidates yields BLEU25.2/ROUGE-L29.6, multiplying compute.

**Critical inference:** Qwen has **24 causal DeltaNet layers plus8 full-attention layers**; mask replacement alone cannot reproduce NCM. Recurrent-state isolation and within-level behavior require redesign. [Config](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json), [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py). Your ladder processes **641 positions**, not128.

**6. Other projects — self-reports, not independent validation.**

- [MCL4](https://github.com/mESmaC/mcl4-coe-ncm): pretrained Llama/Granite grafts,4/6 levels,1024 codes; Granite reconstruction85.35%.
- [WB-catboy](https://huggingface.co/banan332/WB-catboy-NCM-Instruct-725M): NCM725M chat tuning,128-token prefix; weak factuality.
- [ncm-modular](https://github.com/Red-fuchs/ncm-modular): consecutive spans reduce invented names8.1→2.2%; seams unresolved; exact surrogate restoration requires known entities.
- [Issue26](https://github.com/HazyResearch/ncm/issues/26): LM1B PPL94.69/88.96/81.25 at97/317/725M.
- [ConceptLM](https://arxiv.org/abs/2602.08984): adapts Llama8B while retaining token decoding.

None establishes your ≤5% retention target.

**7. Ranked changes — inference:** **(1)** Gate M1 on executable reconstruction; prototype exact corrections/greedy verification. **(2)** Resolve Qwen’s hybrid-state semantics first. **(3)** Complete LM1B fidelity, especially dual loss/positions/cascade. **(4)** Increase comparable exposure and target rare-token errors. **(5)** Compare1/4 decoder blocks and quadratic/exponential ladders before enlarging width/codebooks. Stop treating aggregate accuracy or NFE ratios as quality/speed guarantees; defer CFG/MBR.
