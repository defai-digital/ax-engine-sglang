# NCM paper design audit for Track M (2026-11-14)

Question (user): are we using the NCM paper's design features, and what are its best practices?
Sources: arXiv 2610.14162 (HTML v1, Tables 1, 4, 5, Sections 3.3-3.4, Appendices C, F, I) and the
released configs in HazyResearch/ncm `config/eval/*.yaml` (commit c702742).

## 1. What the paper actually shows

- Generators are ~100M parameters (scaling runs 97M / 317M / 725M), trained from scratch per
  dataset. There is no conversion of a pretrained AR model, no AR verifier, no hybrid with AR. The
  only pretrained model tried (T5-small as context encoder) was dropped.
- Speed and quality claims are against diffusion LMs (MDLM, DUO, ...): "9-31x faster" than those,
  not faster than AR or speculative decoding. OWT unconditional Gen. PPL is 83.1 at 32 NFE.
- Long outputs: fixed target length per task (48-1,024 tokens); no multi-span generation.
- The paper says one token error can "prevent code from running"; it evaluates no code task.
  Reconstruction errors are "substitutions of related words or entities" (what we see).
- Limitations (App. I): separately trained tokenizer, wish for end-to-end training, "occasional
  artifacts at coarse-level transitions".

## 2. Tokenizer: we copied the paper's weakest recipe

Released tokenizers (Table 1 reconstruction; settings from the configs):

| Task | Vocab | L | Levels (K) | codes/token | Codebook | Dec. blocks | pre-quant dropout | cascade norm | Recon. |
|---|---|---|---|---|---|---|---|---|---|
| OWT | GPT-2 50k | 1024 | 32 (1,4,9,...,1024) | ~11.2 | 8192 | 1 | 0.0 | false | 99.99% |
| TinyStories | BPE 8k | 128 | 12 (1,8,16,...,128) | ~5.0 | 4096 | 1 | 0.0 | false | 99.98% |
| LM1B | BERT 30k | 128 | 12 (1,8,16,24,32,40,52,64,76,92,108,128) | ~5.0 | 8192 | 4 | 0.0 | - | 99.86% |
| XSum | GPT-2 50k | 48 | 8 | ~4.2 | 4096 | 1 | 0.25 | true | 99.79% |
| WMT14 | Qwen3 151k | 64 | 11 (1,4,8,...,64) | ~4.9 | 4096 | 1 | 0.25 | true | 98.79% |
| **Ours M0/M0b/M0d** | Qwen3.5 248k | 64 | 11 (WMT14) | ~4.9 | 4096 | 1 | 0.25 | true | 97.6-99.0% |

Training budget (token positions): OWT 78,432 x 768 x 1024 = ~62B; LM1B 25,957 x 2,048 x 128 = ~6.8B;
ours M0d 30,000 x 1,024 x 64 = ~2.0B (M0b stopped at ~0.65B).

Differences that plausibly cost us reconstruction:
1. Recipe choice: WMT14 is the only released tokenizer under 99.79%. Translation is scored by BLEU,
   which tolerates related-word substitutions; code is not.
2. pre_quant_dropout 0.25 (only the two conditional tasks use it; all >= 99.86% tokenizers use 0).
3. Codebook 4096 vs 8192 (OWT, LM1B), and decoder depth 1 vs 4 (LM1B).
4. Training tokens 3-30x below the high-accuracy tokenizers.
5. Our vocabulary (248k) is the largest; WMT14 (151k, the next largest) is also the worst. Frozen
   Qwen tables (M0b) already cut unseen-token errors from ~99% to 38-68%.

Speed implication missed in our plan: LM1B and TinyStories cover 128 tokens with 12 levels (10.7
tokens per NFE) at 99.86-99.98%; our 64 tokens / 11 levels gives 5.3. OWT covers 1,024 with 32
(32 tokens per NFE). Span length, not level count, is the speed lever, and the paper shows the
longer spans reconstruct better, not worse.

## 3. Generator features not yet in the M1 plan

| Feature (paper / config) | Purpose | In M1 plan? |
|---|---|---|
| Code corruption: conditioning codes replaced uniformly at a random rate in [0, 1], level 0 skipped; mean ~0.5, decreasing over levels (Sec. 3.4, App. F) | train-test mismatch / error propagation; ablation: "greatly improves quality" | no |
| Condition each level on the cumulative continuous reconstruction z^(k-1), resized to l_k and projected | coarse-to-fine refinement on continuous latents, not on discrete ids | no (planned discrete codes) |
| Level loss weighting w_k ~ l_k^-alpha (alpha <= 1 fine) | balance coarse vs fine levels | no |
| Block-diagonal attention within the target; prefix-LM attention to the prompt | fastest and best in ablation (Gen. PPL 15.4 vs 19.6 block-causal) | partially (KV cache context) |
| Level 0 from a unigram prior (predict_level0 false) for unconditional tasks; predicted for WMT14 | stable first level | no |
| Classifier-free guidance (w 0.5-1.0; 2 NFE per step) | conditional quality | no |
| MBR over K = 12-16 samples (+4.5 ROUGE-1, best results use it) | quality at extra NFE | optional |
| Two-stage generator training: unpaired text first, then prompt/answer pairs | data efficiency | no |

## 4. Consequences for Track M

- The measured HumanEval collapse (83 -> 46% after an M0b round trip) is partly self-inflicted by
  the recipe choice; the paper's own tokenizers reach 99.86-99.99% on general text. A tokenizer at
  >= 99.95% is a reasonable target, not a stretch.
- The better operating point is the LM1B/TinyStories shape: L = 128, 12 levels, codebook 8192,
  pre-quant dropout 0, decoder 4 blocks, frozen Qwen tables, long training on broad data; gate on the
  task round trip (HumanEval and GSM8K lose <= 1-2% relative), with token accuracy as a diagnostic.
- None of this addresses the larger M1 risk: the paper never matched AR quality, never used a
  pretrained LM as generator and never ran code tasks. M1 should adopt code corruption, continuous
  z^(k-1) conditioning, block-diagonal attention and MBR from the start.
