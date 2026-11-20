# NCM implementation lessons: grok

# NCM drafter evidence

Measured facts are tagged. Other causal claims are inferences. "not found" means these sources do not state it.

## 1. Official tokenizer, ablations, limitations, local gap

**Measured** ([Table 1](https://arxiv.org/html/2610.14162v1)): full-hierarchy exact token recovery is OpenWebText 99.99%, LM1B 99.86%, TinyStories 99.98%, XSum 99.79%, WMT14 98.79%, Sudoku 100%. Mean codebook use is 83.10 / 89.31 / 90.12 / 86.20 / 61.89 / ~81–84%. Block-exact rate: not found.

**Measured** (Tables 4–5): vocab / L / K / codebook / updates — OWT 50,304 / 1024 / 32 / 4,096 / 78,432; LM1B 30,528 / 128 / 12 / 8,192 / 25,957; TinyStories 8,192 / 128 / 12 / 4,096 / 30,000; XSum 50,304 / 48 / 8 / 4,096 / 120,001; WMT14 (Qwen3-0.6B) 151,680 / 64 / 11 / 4,096 / 60,000; Sudoku 12 / 89 / 12 / 16 / 10,000. Text models use d=384, 0 encoder blocks, 1 decoder block (LM1B: 4), finest level = L, cross-entropy plus commitment and codebook MSE, straight-through, nested dropout 0.5 (LM1B written 0.7/0.3), EMA 0.99 or 0.95. Entropy loss, restarts, numeric β, and integer level lists: not found. Exposure: OWT ~61.68B positions, LM1B ~6.80B (Appendix C.2).

**Released YAMLs** ([repo](https://github.com/HazyResearch/ncm)): commitment 0.25, Phi 0.5, learned level-0 sampler, GroupNorm, restarts off. Versus Table 5: OWT codebook 8,192, XSum has 9 levels and EMA 0.99, WMT EMA 0.99. LM1B 0.7/0.3 weights are unread by `quant.py`.

**Ablations** (Appendix F): K∈{4,8,16,32} on TinyStories all reconstruct ≥99.95%. Schedule Gen. PPL spans 196.7 (square-root) to 57.7 (exponential). Loss-weight α moves Gen. PPL only 18.9–19.4. Corruption "greatly improves" generation; bar heights not printed; one setup uses mean rate 0.50. Codebook-size and block-length reconstruction ablations: not found.

**Limitations** (Appendix I): fixed K, separate VQ-VAE, scale stops at 725M, artifacts at coarse-level transitions. Reconstruction failure and drafting are not listed. §2 says one wrong token can break code or Sudoku.

**Local diff** (`phase1/vqvae.py`, `b32_lossless.yaml`): vocab 248,320 Qwen3.6 ids, code/math/web/chat, block 32, levels 4/8/16/32, 8,192 codes, d=512, 2+2 blocks, LayerNorm, tied head, area/linear only, residual detached, restarts on (0.05/200). The 66% figure is a comment: 3,000-step code accuracy moved 53%→66% after loosening restarts. A finished 84% log was not found. Inference: K=4 at ≥99.95% means four levels do not explain 66% on TinyStories. Untested together: 248k vocab, code mix, no learned level 0, restarts on, ~3k steps versus 10k–120k. WMT at 151,680 still hits 98.79%.

## 2. Outside measurements

**Measured:** RTX re-evals match Sudoku ~99.4/96.4/86.2 and put LM1B Gen. PPL near 94/89/81 for 97M/317M/725M ([#26](https://github.com/HazyResearch/ncm/issues/26)). Batch-1 versus an untuned GPT-2 loop is ~7–8× and reverses by batch 128. [ncm-mlx](https://github.com/janmus1c/ncm-mlx): codec 99.816% tokens; 12M Gen. PPL ~46 with strong corruption versus ~95 with none, while code CE moves 2.25→1.39; same-size AR ~14.6. Author on [#16](https://github.com/HazyResearch/ncm/issues/16): restarts helped reconstruction and hurt generation, so they were left out. [ncm-modular](https://github.com/Red-fuchs/ncm-modular) calls the WMT codec a Qwen tokenizer (96% entity keep, 8% JSON round-trip). A new 100k+ or Llama/Qwen-id training run: not found. OpenReview reviews: not found ([forum](https://openreview.net/forum?id=d83mBTrpiJ)). X, Reddit, HN: discussion only ([HN](https://news.ycombinator.com/item?id=50422168)).

## 3. Related numbers

**Measured.** NCP ([2609.10715](https://ar5iv.labs.arxiv.org/html/2609.10715)): product quant, k=4, 32×128; concept accuracy not found. Injecting the target's last concept into a 1.1B token drafter raises accepted length 5.933→6.180 (+4.17%). LCM RVQ 64×8192 reaches ~70% of continuous SONAR BLEU ([2412.08821](https://arxiv.org/html/2412.08821v2)); no token verifier. VAR is images only, FID 18.65→1.73 ([2404.02905](https://arxiv.org/html/2404.02905)); text reconstruction not found. Token drafters: HSD +6.7% speed ([2601.05724](https://arxiv.org/abs/2601.05724)); DSpine 3.77→4.82, and an early bad token shortens the prefix ([2609.36173](https://arxiv.org/abs/2609.36173)); JetSpec++ 8.94 vs 6.89 ([2610.19792](https://arxiv.org/html/2610.19792)); SpecPilot 4.46×→10.87× at one request ([2611.02501](https://arxiv.org/abs/2611.02501)); SpecOPD up to +41.45% acceptance ([2611.02846](https://arxiv.org/abs/2611.02846)); D-REAL up to +0.23 tokens ([2611.10432](https://arxiv.org/html/2611.10432v1)).

## 4. Used as a verified draft?

**Measured:** not found. NCM samples codes coarse-to-fine and argmaxes the VQ decoder ([Algorithm 2](https://arxiv.org/pdf/2610.14162)). "Lossless" and "acceptance": not found. Speculative decoding is one DFlash citation. LM1B Gen. PPL is 93.05 at 12 steps versus DUO 97.30 at 128 (21×). F.3 describes teacher-forced codes versus free-running accumulation; paired token accuracies: not found. A co-author calls drafting a possible integration ([TechTalks](https://bdtechtalks.com/2026/11/06/next-concept-modeling/)). Later levels can clean earlier code errors ([X](https://x.com/KumbongHermann/status/2110471684344135741)), inside NCM's decoder, before an AR verifier sees tokens.

## 5. Ranking

**Inference.** Under DSpine's prefix rule, a wrong token at position 1 ends the draft, and no paper shows an AR LLM repairing a coarse code. NCM's finer-level correction stays inside its own decoder. ≥98.79% reconstruction can coexist with losing to AR on free text (ncm-mlx). The 66% comment is outside every published reconstruction run. Four levels and 8,192 codes are not what the K=4 ≥99.95% result blames. The 0.95 versus 0.31 gap matches F.3 and the corruption tradeoff.

Supported use: match the released tokenizer until token accuracy is in their range, then use codes only as features for a token drafter under ordinary speculative verification.

| Change | Evidence | Expected effect | Cost |
| --- | --- | --- | --- |
| Match the release: ℓ_K=L, learned level-0 sampler, GroupNorm, 0 encoder blocks, untied head, restarts off, ≥30k steps; report token accuracy | [Table 1](https://arxiv.org/html/2610.14162v1), [code](https://github.com/HazyResearch/ncm) | The only published route into 98.8–99.99% tokens. 248k code mix is untested | OWT exposure was ~61.68B positions |
| Verify tokens, not codes. Optionally condition a token drafter on a concept vector | [NCP +4.17%](https://ar5iv.labs.arxiv.org/html/2609.10715); concept-code acceptance not found; [DSpine](https://arxiv.org/abs/2609.36173) | The only measured concept effect on acceptance is small. A bad coarse code is not repaired after verify | Drop the 7-pass code proposal |
| Train the code prior with conditioning corruption; score free-running output | [F.3](https://arxiv.org/pdf/2610.14162); [ncm-mlx](https://github.com/janmus1c/ncm-mlx) ~46 vs ~95 Gen. PPL | Free-running samples improve while teacher-forced loss gets worse | One schedule; does not fix 66% reconstruction |
| Leave dead-code restarts out of the generator | [#16](https://github.com/HazyResearch/ncm/issues/16); local comment 53%→66% at 3k steps | Restarts can raise reconstruction and, per the author, hurt generation | An ablation |
| If the metric is acceptance length, start from a token block drafter | [JetSpec++ 8.94 vs 6.89](https://arxiv.org/html/2610.19792); NCM's 10–31× is vs diffusion ([abstract](https://arxiv.org/html/2610.14162v1)) | A measured lossless speedup. NCM-as-drafter speedup: not found | A different architecture |
