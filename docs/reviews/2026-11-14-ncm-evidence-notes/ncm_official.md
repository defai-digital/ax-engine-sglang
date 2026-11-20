# Official NCM tokenizer recipe vs a local reimplementation at 66–84% reconstruction

Sources used: arXiv HTML [2610.14162v1](https://arxiv.org/html/2610.14162v1) (tables parsed from the HTML; PDF at [arxiv.org/pdf/2610.14162](https://arxiv.org/pdf/2610.14162) was downloaded but not text-extracted). Code clone at `/private/tmp/claude-502/-Volumes-Ext16TR0-code-automatosx/b1fa9bed-7128-4a15-9f2e-a27580b78cf0/scratchpad/ncm-src`, which is the [HazyResearch/ncm](https://github.com/HazyResearch/ncm) evaluation repo. Local tokenizer at `/Volumes/Ext16TR0/code/ax-engine-sglang/phase1/vqvae.py`. Hugging Face raw `owt/config.yaml` at `https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/main/owt/config.yaml` returned HTTP 404 in this session, so checkpoint bytes were not re-downloaded. The clone README says `config/eval/` is identical to the `config.yaml` shipped with each checkpoint ([README.md](https://github.com/HazyResearch/ncm/blob/main/README.md) lines 91–93 of the clone: `/private/tmp/claude-502/-Volumes-Ext16TR0-code-automatosx/b1fa9bed-7128-4a15-9f2e-a27580b78cf0/scratchpad/ncm-src/README.md:91-93`). Project pages linked from that README, not fetched: [Hazy blog](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm), [Scaling Intelligence blog](https://scalingintelligence.stanford.edu/blogs/ncm/), [HF collection](https://huggingface.co/collections/hazyresearch/ncm), [HF checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints).

"Source measured" means the number or quote is in a cited source. Calculations and causal claims are under Inferences. "not found" means the source set above does not state it.

## 1. Exact tokenizer recipe and reconstruction accuracy per dataset

### Takeaway
The paper reports full-hierarchy exact token reconstruction of 99.99% (OWT), 99.86% (LM1B), 99.98% (TinyStories), 99.79% (XSum), 98.79% (WMT14), and 100.00% (Sudoku easy/medium/hard). The recipe is a multiscale residual VQ-VAE with \(\ell_K = L\), commitment plus codebook MSE, straight-through, and nested dropout. Released eval configs match most of Table 5 but disagree on OWT codebook size (8192 vs 4096), XSum level count (9 vs 8), and XSum/WMT EMA decay (0.99 vs 0.95).

### Cited Findings
Source measured, Table 1, exact token recovery on held-out text with the full hierarchy ([HTML Table 1](https://arxiv.org/html/2610.14162v1)):

- Recon. acc. (%): OWT 99.99, LM1B 99.86, TinyStories 99.98, XSum 99.79, WMT14 98.79, Sudoku Easy 100.00, Medium 100.00, Hard 100.00.
- Utilization (%), mean percentage of codebook entries used across levels: OWT 83.10, LM1B 89.31, TinyStories 90.12, XSum 86.20, WMT14 61.89, Sudoku Easy 80.73, Medium 83.85, Hard 80.73. The HTML cells for the three Sudoku utilization numbers contain a `visibility:hidden` `0` used for alignment; the visible text is 80.73 / 83.85 / 80.73.
- Caption and prose: "Reconstruction accuracy measures exact token recovery on held-out text using the full hierarchy." "On Sudoku, where correctness requires exact recovery, the tokenizer achieves 100% reconstruction accuracy." "On other tasks, like summarization and translation, token-level reconstruction accuracy is close to 100%." "mean utilization ranging from 62% to 90% across datasets."
- Appendix G, OWT only: "Reconstruction accuracy increases across levels, reaching 99.99% at full resolution." "approximately 0.01% incorrectly reconstructed tokens." Error categories in Table 11: named entities, diacritics, numerals, usernames, markup, word boundaries. Per-level accuracy numbers in Figure 16 are not printed in the text. not found as a table.

Source measured, formulation ([§3.2–3.3](https://arxiv.org/html/2610.14162v1)):

- \(\ell_1 < \ell_2 < \cdots < \ell_K = L\). The finest level is the full target length.
- Downsample residual to \(\ell_k\), nearest codebook entry, reconstruction operator \(R_k\) maps codes back to length \(L\), residual \(z^{(k)} = z - \hat z^{(k)}\).
- Loss (eq. 9), at a sampled cutoff \(k\): token cross-entropy \(+\ \beta\,\mathrm{MSE}(z, \mathrm{sg}[\hat z^{(k)}]) + \mathrm{MSE}(\mathrm{sg}[z], \hat z^{(k)})\). "The VQ terms combine a commitment loss and a latent quantization loss." "We use a straight-through estimator to pass reconstruction gradients through the discrete quantization step to the encoder."
- Numeric \(\beta\): not found in Table 5 or in the equation prose. Entropy loss: not found. Codebook restarts: not found. The names Phi, quant_resi, tied embeddings: not found in the paper text that was extracted.

Source measured, Table 4 datasets ([Appendix B](https://arxiv.org/html/2610.14162v1)). One tokenizer per dataset. Text tokenizer / context / target \(L\):

- OpenWebText: GPT-2, null prefix, \(L=1024\).
- LM1B (One Billion Word): bert-base-uncased, null prefix, \(L=128\).
- TinyStories: Custom BPE, null prefix, \(L=128\).
- XSum: GPT-2, article context 512, target 48.
- WMT14 De–En: Qwen3-0.6B, German context 96, target 64.
- Sudoku: FMLM+, puzzle context, target 89.
- Bytes as the VQ-VAE input: not found. Training is on these text-tokenizer ids, not on raw bytes and not on Qwen3.6-35B ids.

Source measured, Table 5 tokenizers ([Appendix C.1](https://arxiv.org/html/2610.14162v1)). Columns OWT, LM1B, TinyStories, XSum, WMT14, Sudoku. Caption: "Text vocabulary size includes unused padding rows. Nested dropout is the probability of dropping the finest levels; LM1B trains on the complete and a truncated hierarchy, weighted 0.7 and 0.3."

- Text vocab size: 50304, 30528, 8192, 50304, 151680, 12.
- Target length \(L\): 1024, 128, 128, 48, 64, 89.
- Levels \(K\): 32, 12, 12, 8, 11, 12. The integer list \(\ell_k\) for the main models is not in Table 5. not found there.
- Codebook vocab \(V_k\): 4096, 8192, 4096, 4096, 4096, 16. The table prints one number per dataset, not a per-level list.
- Latent width \(d\): 384, 384, 384, 384, 384, 256.
- Encoder blocks: 0, 0, 0, 0, 0, 0.
- Decoder blocks: 1, 4, 1, 1, 1, 1.
- Decoder attention heads: 6, 6, 6, 6, 6, 4.
- Dropout: 0.3, 0.1, 0.3, 0.1, 0.1, 0.3.
- Parameters: 111.2M, 111.40M, 36.43M, 60.63M, 144.00M, 14.90M.
- Training updates: 78432, 25957, 30000, 120001, 60000, 10000.
- Global batch size: 768, 2048, 4096, 4096, 4096, 1024.
- Peak learning rate: \(3\times 10^{-4}\), \(3\times 10^{-4}\), \(5\times 10^{-4}\), \(5\times 10^{-4}\), \(5\times 10^{-4}\), \(5\times 10^{-4}\).
- Learning-rate floor: \(3\times 10^{-5}\), \(3\times 10^{-5}\), \(5\times 10^{-5}\), \(5\times 10^{-5}\), \(5\times 10^{-5}\), \(5\times 10^{-5}\).
- Warmup updates: 1000, 1000, 1000, 1000, 1000, 500.
- AdamW \(\beta_1\): 0.9 for all six. \(\beta_2\): 0.95, 0.99, 0.95, 0.95, 0.95, 0.99.
- Weight decay: 0.01, 0.05, 0.01, 0.01, 0.01, 0.01. Gradient clip: 1 for all.
- Codebook EMA decay: 0.99, 0.99, 0.95, 0.95, 0.95, 0.99.
- Nested dropout: 0.5, 0.7/0.3, 0.5, 0.5, 0.5, 0.5.
- Paper's own token-exposure totals (Appendix C.2, includes padding): LM1B tokenizer \(25{,}957\times 2{,}048\times 128 \approx 6.80\)B positions; OWT tokenizer \(78{,}432\times 768\times 1{,}024 \approx 61.68\)B. Other datasets' token totals: not found as a stated sum.
- Seed, BF16, torch.compile: "All experiments employed seed 42, BF16 mixed-precision and torch.compile."

### Inferences
- None of the Table 1 numbers is a block-exact-match rate. The metric is token accuracy. Exact-block rate: not found.
- WMT at vocab 151680 still has 98.79% token accuracy, so a six-figure vocabulary is not, in their setup, enough by itself to land in the 66–84% range.

### Gaps
- Per-level reconstruction percentages (Figures 9c and 16) are not printed.
- Numeric commitment \(\beta\) in the paper: not found.
- Main-model integer level schedules: not found in the paper (only \(K\) and \(\ell_K=L\)).
- HF checkpoint files were not downloaded (404 on the resolve URL tried).

## 2. Ablations they reported

### Takeaway
The only printed tokenizer-reconstruction ablation number is that TinyStories ladders with \(K\in\{4,8,16,32\}\) all reconstruct at least 99.95% of tokens. Other ablations (schedule shape, attention, loss weight \(\alpha\), code corruption, CFG, MBR) report generation quality, not a tokenizer accuracy table. Codebook-size and block-length ablations of reconstruction: not found.

### Cited Findings
Source measured, Appendix F, TinyStories unless noted ([HTML Appendix F](https://arxiv.org/html/2610.14162v1)):

- F.1.1 number of levels. \(K\in\{4,8,16,32\}\), linear schedule from 1 to 128, 4096 codes per level, tokenizer trained 40k steps, then a generator for 10k steps, no code corruption. "longer hierarchies achieve slightly better final reconstruction, although all four reconstruct at least 99.95% of tokens." Gen. PPL falls as \(K\) grows, with diminishing returns. Reference text Gen. PPL 9.67 and unigram entropy \(H=3.96\). The per-\(K\) Gen. PPL values and the per-level accuracy curve are in Figure 9 and are not printed. not found as numbers.
- F.1.2 level schedule. Six schedules (linear, quadratic, square-root, logarithmic, exponential, cosine), \(K=12\), start at 1 code, end at 128, lengths rounded to integers. Figure 10 compares Gen. PPL. Printed Gen. PPL per schedule: not found.
- F.2.1 attention (generator, same 12-level tokenizer, "mean rate 0.50" corruption). Table 9: block-causal Gen. PPL 19.63, \(H\) 4.01, time 7.90s; sliding window (2 levels) 17.84, 3.99, 6.83s; block-diagonal 15.41, 3.98, 5.65s. Times are for 1024 TinyStories samples at 12 steps, batch size 16. Block-causal and sliding-window times are \(1.40\times\) and \(1.21\times\) block-diagonal.
- F.2.2 context encoder on XSum, constant-LR recipe, greedy, 8 NFE, full test set. Table 10: T5-small encoder ROUGE-1/2/L 34.89 / 12.92 / 28.19 (123.4M) vs VQ-VAE encoder 34.79 / 12.77 / 27.95 (88.0M). Deltas stated as +0.10, +0.15, +0.24 ROUGE and +35.3M parameters.
- F.2.3 loss weighting \(w_k \propto \ell_k^{-\alpha}\). Raising \(\alpha\) from 0 to 1.5 moves the coarse levels' share of the loss from 19% to 63%. "Gen. PPL after 10k steps stays at 18.9–19.0 for \(\alpha\le 1\) and rises slightly to 19.4 at 1.5."
- F.3 code corruption (generator, not the tokenizer objective). TinyStories, 128 tokens, 12-level tokenizer of Table 5, 88.6M transformer, 10,000 steps, batch 1024. Gen. PPL averaged over five seeds of 256 samples. "We use a linearly decreasing corruption rate across levels." "moderate corruption greatly improves generation quality, while excessive corruption weakens the conditioning signal." The rate that wins Figure 13: not found as a printed number. Replacement is uniform from the same level's codebook. Targets stay clean.
- F.4 sampling. CFG: XSum ROUGE peaks at \(w=0.5\); WMT BLEU peaks at \(w=1.0\). MBR: "gaining 4.48 ROUGE-1 in total from \(K=1\) to \(K=256\)." Main table uses \(K=16\) on XSum (128 NFE) and \(K=12\) on WMT14 (132 NFE), "yielding 29.6 ROUGE-L and 25.2 BLEU."
- Codebook-size ablation: not found. Block-size / \(L\) ablation: not found. The \(K=4\) run above is the closest reconstruction ablation, and it still says at least 99.95%.

### Inferences
- A 4-level ladder is not, on their TinyStories BPE setup, the reason reconstruction would fall to 66–84%. Their own \(K=4\) tokenizer ablation still states \(\ge 99.95\%\) token reconstruction.
- Corruption, \(\alpha\), attention, CFG, and MBR change generation, not the frozen tokenizer's Table 1 accuracy.

### Gaps
- Figure-only ablation numbers (Gen. PPL vs \(K\), vs schedule, vs corruption rate, per-level accuracy) are not in the HTML text.

## 3. Limitations they list

### Takeaway
Appendix I does not say the tokenizer fails to reconstruct, does not discuss hierarchy token-errors as a limitation, and does not discuss using NCM as a speculative drafter. It does say generations have occasional artifacts at coarse-level transitions, that \(K\) and \(\ell_k\) are fixed, and that the VQ-VAE is trained separately from the generator.

### Cited Findings
Source measured, Appendix I ([HTML](https://arxiv.org/html/2610.14162v1)), full substance of the section:

- "NCM fixes both the hierarchy depth \(K\) and the per-level lengths \(\ell_k\), so even short continuations traverse the full hierarchy; an input-dependent schedule would adapt compute to example complexity."
- "NCM relies on a separately trained multiscale VQ-VAE tokenizer, decoupling representation learning from the generative objective. An end-to-end training approach could benefit NCM by aligning representation learning with the generative objective."
- "Our main experiments use generators of roughly 100M parameters, and our scaling study extends to 725M parameters on LM1B. Scaling to larger datasets and model sizes is another exciting direction."
- "Generations are generally fluent but exhibit occasional artifacts at coarse-level transitions, presenting another opportunity to further improve NCM."
- F.3 also defers "Corruption based on the model's actual errors" to future work. That is a training/inference mismatch for the generator, not a Table 1 limitation.
- Drafter: the introduction mentions "diffusion drafters like DFlash" as background for diffusion LMs. Appendix I does not say NCM is or should be a drafter. not found as a limitation.
- Section 2, why reconstruction matters, is not in the limitations list: "Accurate reconstruction is also important: a single token error can alter meaning, prevent code from running, or invalidate a Sudoku solution."
- Section 4.1: reconstruction errors "often involve substitutions of related words or entities, with their effect on meaning depending on the context."

### Inferences
- Nothing in Appendix I predicts a 66–84% tokenizer. The paper treats full-hierarchy reconstruction near 99–100% as an achieved property, and treats coarse-level transition artifacts as a generation issue.

### Gaps
- No limitation statement about exact-match block reconstruction, code tokens, or drafter acceptance rates. not found.

## 4. Official code vs the paper

### Takeaway
The released eval YAMLs are the concrete \(\ell_k\) lists and the quantizer switches. They agree with Table 5 on vocab, \(L\), \(d\), encoder depth 0, decoder depth, and head count for the datasets checked, but OWT's codebook is 8192 in the YAML vs 4096 in Table 5, XSum has 9 levels ending at 48 vs paper \(K=8\), and XSum/WMT EMA decay is 0.99 in YAML vs 0.95 in Table 5. The quantizer always treats the last level as length \(L\). There is no entropy loss and no dead-code restart unless `restart_threshold` is set, which the YAMLs do not set.

### Cited Findings
Source measured, what `n_layers` means. `models/vae.py` uses the constructor argument `n_layers` as `n_head`, not depth (`/private/tmp/.../ncm-src/models/vae.py:130-136` and the comment at `vae.py:224-227`: "n_layers is the head count here, not a depth"). Encoder transformer blocks exist only if `enc_layers>0` (`vae.py:130-136`). `VQVAE` defaults `enc_layers=0`, `dec_layers=1` (`models/vqvae.py:26-28`). Decoder `output_layer` is a separate `Linear`; it is not tied to `wte` (`vae.py:126`, `vae.py:233`, `vae.py:257`). Pre-quant norm is GroupNorm or LayerNorm with `affine=False` (`vqvae.py:59-67`). Comment: "affine=False: ||z-e||^2 is not scale-invariant."

Source measured, quantizer mechanics (`models/quant.py`):

- Codebooks: `nn.Embedding`, `weight.data.normal_()` (default Normal(0,1)), `requires_grad_(False)`, EMA update (`quant.py:294-300`, `quant.py:485-498`). `kmeans_init` defaults False (`quant.py:246`). YAMLs do not set it.
- `restart_threshold` defaults 0.0 (`quant.py:244`). None of the eval YAMLs set it, so `_restart_dead_codes` does not run (`quant.py:499-500`). When it does run, it runs every forward, copies the new vector into `ema_w` unscaled, and sets the cluster size to the threshold (`quant.py:437-446`).
- Down: level 0 uses a learned strided conv if `learned_l1_sampling` (`quant.py:395-398`, init as an average pool at `quant.py:145-151`). Other non-final levels use `F.interpolate(..., mode="area")` (`quant.py:399`). Up: learned conv-transpose on level 0 if that flag is set, initialized as nearest-repeat (`quant.py:155-161`, `quant.py:404-407`); otherwise `F.interpolate(..., mode="linear")` (`quant.py:408`). `align_corners` is not passed.
- The last level is not resampled: `if i != num_levels - 1` downsample/upsample, else quantize `z_rest` at full length (`quant.py:472-475`, `quant.py:507-510`). `idx_to_z_q` allocates `self.levels[-1]` as the sequence length (`quant.py:713`). So the implementation requires \(\ell_K = L\).
- Phi / `quant_resi`: `(1-r)*h + r*Conv1d(h)` (`quant.py:10-38`). Released configs set `enabled: true`, `ratio: 0.5`, `share_mode: 0` (one Phi per level), `learnable_ratio: false`. Example OWT: `config/eval/owt.yaml:128-132`.
- Straight-through: final `z_hat = z + (z_hat - z).detach()` (`quant.py:531`). Extra STE only at level 0 and only if `learned_l1_sampling`. Comment: "Straight-through estimator at level 0 only; at higher levels it diverges within ~100 steps." (`quant.py:503-505`).
- VQ loss inside the loop, then divided by the number of levels (`quant.py:519-524`): `commitment_cost * MSE(z_hat.detach(), z) + MSE(z_hat, z.detach())`. Nested dropout replaces `z_hat` after that loss is computed (`quant.py:525-530`). That is not "both CE and VQ evaluated only at the sampled cutoff" as eq. 9 states. Entropy term: not found in `quant.py` forward.
- Fine-drop keys actually read: `prob`, `min_keep`, `prob_start`, `anneal_steps` (`quant.py:262-269`). Keys `dual`, `w_full`, `w_trunc` are not read. `code_corruption` is not referenced in any `.py` file in the clone (search of `ncm-src`). LM1B YAMLs still contain those keys (`config/eval/lm1b-small.yaml:113-117`).

Source measured, released tokenizer YAMLs under `config/eval/` (clone path prefix `/private/tmp/claude-502/-Volumes-Ext16TR0-code-automatosx/b1fa9bed-7128-4a15-9f2e-a27580b78cf0/scratchpad/ncm-src/`). `vqvae_vocab_size` is one integer repeated \(K\) times. `learned_l1_sampling: true` on every task. `commitment_cost: 0.25` on every task. `quant_resi.ratio: 0.5`, `share_mode: 0`.

- OWT (`owt.yaml:76-84`, `121-136`, levels `140-171`): vocab 50304, \(L=1024\), \(d=384\), `n_layers` 6 (heads), dropout 0.3, GroupNorm groups 8, EMA decay 0.99, fine-drop prob 0.5, min_keep 1, cascade true, 32 codebooks of 8192, levels \(1^2,2^2,\ldots,32^2\) ending at 1024. Paper Table 5 codebook is 4096, not 8192. EMA 0.99 matches the paper.
- LM1B (`lm1b-small.yaml:56-67`, `84-88`, levels `99-110`, fine-drop `113-117`; medium and large match): vocab 30528, \(L=128\), \(d=384\), heads 6, dropout 0.1, GroupNorm 4, `dec_layers: 4`, `enc_positional: false`, EMA 0.99, cascade false, 12 codebooks of 8192, levels 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128. Paper \(K\), \(V_k\), \(d\), decoder blocks, EMA match. The YAML fine-drop is `prob: 1.0` plus unread `w_full: 0.7`, `w_trunc: 0.3`. The paper's 0.7/0.3 weights are therefore not implemented in this quantizer.
- TinyStories (`tinystories.yaml:55-63`, `80-95`, levels `99-110`): vocab 8192, \(L=128\), \(d=384\), heads 6, dropout 0.3, GroupNorm 8, EMA 0.95, cascade true, 12 codebooks of 4096, same level list as LM1B, fine-drop prob 0.5. Matches Table 5 on vocab, \(L\), \(K\), \(V_k\), \(d\), dropout, EMA, nested-dropout probability.
- XSum (`xsum.yaml:53-61`, `75-90`, levels `94-102`): vocab 50304, \(L=48\), \(d=384\), heads 6, dropout 0.1, GroupNorm 8, pre-quant dropout 0.25, EMA decay 0.99, cascade true with cascade norm, 9 codebooks of 4096, levels 4, 6, 10, 14, 20, 26, 32, 40, 48. Paper says \(K=8\) and EMA 0.95. Those two do not match. Finest level equals 48, so \(\ell_K=L\) holds.
- WMT14 (`wmt14.yaml:51-59`, `75-92`, levels `96-106`): vocab 151680, \(L=64\), \(d=384\), heads 6, dropout 0.1, GroupNorm 8, pre-quant dropout 0.25, EMA decay 0.99, cascade true with cascade norm, 11 codebooks of 4096, levels 1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64. Paper \(K=11\) and \(V_k=4096\) match. Paper EMA 0.95 does not match YAML 0.99.
- Sudoku easy/medium/hard (example `sudoku-easy.yaml:43-46`, `61-72`, levels `82-93`, fine-drop `100-101`): vocab 12, \(L=89\), \(d=256\), heads 4, `dec_layers: 1`, dropout 0.3, GroupNorm 8, EMA 0.99, cascade true, 12 codebooks of 16, levels 1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89, fine-drop prob 0.5. \(K\), \(V_k\), \(d\), \(L\), EMA match Table 5.

The clone has no training entry point. `README.md:70-87` is evaluation only. Tokenizer training hyperparameters that are not stored in these YAMLs (optimizer steps, batch, learning rate) exist in paper Table 5 and are not repeated in `config/eval/`.

### Inferences
- Where paper Table 5 and the YAML disagree, the checkpoint config is the one that would have been loaded for released weights, if the README's "identical" claim is true. This note did not hash the HF files.
- A reimplementation that follows Table 5's OWT codebook of 4096 is not the released OWT tokenizer. A reimplementation that follows Table 5's XSum \(K=8\) is not the released XSum level list.

### Gaps
- Training-loop source that would honor LM1B `w_full` / `w_trunc` or `code_corruption`: not in this repo. not found.
- Whether the paper's 4096-code OWT tokenizer was an earlier run than the released 8192-code checkpoint: not found. Both documents exist; they conflict.

## 5. Every local difference that could explain 66–84% vs their ~99%

### Takeaway
The local module is a real residual VQ-VAE with the same loss shape, area/linear resample, Phi blend 0.5, commitment 0.25, and \(p=0.5\) prefix dropout, but it is not the NCM tokenizer. The differences that the paper's own \(K=4\) result does **not** explain are: Qwen3.6 vocab 248320 and code/math/web/chat ids instead of their BPE, no learned level-0 sampler, dead-code restarts that the release leaves off, LayerNorm instead of GroupNorm, a 2-block encoder where theirs has none, tied embeddings, and much less token exposure if training stops early. The only in-repo accuracy numbers are 53% to 66% code-token accuracy at 3000 steps after loosening restarts, and 80% code-token accuracy for a tiny CPU model at 1500 steps. A finished-run 84% figure: not found.

### Cited Findings
Source measured, local defaults and the lossless config that the training script loads.

`phase1/vqvae.py` (`/Volumes/Ext16TR0/code/ax-engine-sglang/phase1/vqvae.py`):

- Vocab 248320, block 32, \(d=512\), 8 heads, encoder layers 2, decoder layers 2, dropout 0.0, levels `[4, 8, 16, 32]`, codebook 8192, commitment 0.25, EMA decay 0.99, `restart_threshold` default 1.0, `restart_every` 50, `quant_resi` 0.5, `fine_drop_prob` 0.5, tied embeddings True (`vqvae.py:26-42`).
- Down `interpolate` mode `area`; up mode `linear`, `align_corners=False` (`vqvae.py:94-104`). No learned conv. If length already matches, resample is a no-op, so a last level equal to \(L\) is full length (`vqvae.py:95-96`, `vqvae.py:101-102`).
- Codebook init `randn * 0.02`, then on the first train step a data-dependent reinit (`vqvae.py:82-84`, `vqvae.py:112-120`).
- EMA every step; dead codes (`ema size < restart_threshold`) are replaced only when `step % restart_every == 0`. Replacement sets the embedding to the sample and sets `ema_w` to `sample * restart_threshold` (`vqvae.py:123-146`).
- Residual subtracts `q.detach()` (`vqvae.py:170`). Official subtracts the non-detached quantized tensor (`quant.py:517`).
- VQ loss is the mean over levels of commitment MSE plus codebook MSE on the cumulative vector (`vqvae.py:173-179`). Nested dropout then picks a strict prefix with probability `fine_drop_prob` for the tensor that the decoder sees (`vqvae.py:181-182`). Straight-through is only the final `z + (sel - z).detach()` (`vqvae.py:183`). No per-level STE.
- Encoder is `enc_layers` pre-norm `TransformerEncoderLayer`s plus affine-free LayerNorm (`vqvae.py:207-225`). Output head is tied to the token embedding when `tie_embeddings` (`vqvae.py:214-216`). Official decoder head is not tied.
- Plan mode is allowed: `levels[-1] <= block_len` (`vqvae.py:47`). Official `quant.py` indexes the last level as length \(L\).

`phase1/configs/b32_lossless.yaml` overrides: `restart_threshold: 0.05`, `restart_every: 200`, levels `[4, 8, 16, 32]`, and the comment "ablation: 1.0 -> 0.05 raised 3000-step code accuracy 53% -> 66%" (`b32_lossless.yaml:1-18`). Same comment is in `b32_plan.yaml:14`, `b32_k6_lossless.yaml:14`, `b64_lossless.yaml:14`, `b64_plan.yaml:14`. Train plan: 150000 steps, global batch 512, micro-batch 256, lr \(3\times 10^{-4}\), min lr \(3\times 10^{-5}\), warmup 1000, weight decay 0.01, grad clip 1, `vq_weight` 1.0, mix code 0.4 / math 0.2 / web 0.2 / chat 0.2 (`b32_lossless.yaml:20-34`). `b32_k6_lossless.yaml:10` uses levels `[1, 2, 4, 8, 16, 32]`. `b64_lossless.yaml` uses block 64, levels `[8, 16, 32, 64]`, 75000 steps. `b32_plan.yaml:10` uses levels `[4, 8, 16]` and `final_level: tokens`, so the codes do not include a full-length level.

`phase1/train_tokenizer.py:119-120`: loss is token cross-entropy on the ids plus `vq_weight * vq_loss`. AdamW betas are hardcoded `(0.9, 0.95)` (`train_tokenizer.py:76`).

`phase1/prepare_data.py:109` and `phase1/README.md:55-63`: blocks are Qwen tokenizer ids from `Qwen/Qwen3.6-35B-A3B`, domains codeparrot Python, open-web-math, fineweb-edu, ultrachat. Not GPT-2, not bert-base-uncased, not the paper's custom TinyStories BPE, not Qwen3-0.6B.

`phase1/README.md:82-84`: "A tiny model (d=128, 1,500 steps) reached 80% code-token accuracy, and accuracy rose with each added level." The gate the local project set for itself is \(\ge 99.95\%\) code-token accuracy (`README.md:73`, `vqvae.py:10`). That gate is the local project's, not a number measured by the NCM paper. The paper's \(\ge 99.95\%\) sentence is the TinyStories \(K\) ablation, not this gate.

`phase1/configs/smoke_cpu.yaml`: \(d=64\), codebook 256, `restart_threshold: 1.0`, `restart_every: 10`, 40 steps. This is the pipeline smoke config, not the 1500-step \(d=128\) run. The \(d=128\) config file: not found in `phase1/configs/`.

Differences, each with why it can move reconstruction. Official side is the released YAML plus `quant.py`, not only Table 5.

1. Input symbols. Local: Qwen3.6 token ids, vocab 248320 (`vqvae.py:26`, `prepare_data.py:109`). Official text vocabs: 8192 to 151680, Sudoku 12 (Table 5 and the YAMLs above). Their best reported text accuracy on the largest vocab is WMT 98.79% at 151680, not 66%.
2. Data. Local mix is code-heavy (`b32_lossless.yaml:34`). Official tokenizers are per dataset (Table 4). The local logs that exist talk about code-token accuracy, which is the slice the paper says is fragile ("prevent code from running", §2).
3. Learned coarse resample is absent locally. Every released YAML sets `learned_l1_sampling: true`. Official level 0 is a trained strided conv / conv-transpose initialized as average-pool and nearest-repeat (`quant.py:145-161`, `395-411`). Local always uses area down and linear up (`vqvae.py:94-104`), including the coarsest level, which nested dropout forces the decoder to use alone half the time.
4. Dead-code restarts are on locally and off in the release. Official default threshold 0 and YAMLs omit it (`quant.py:244`). Local threshold 0.05 every 200 steps, and the comment records 53% code accuracy at threshold 1.0 versus 66% at 0.05, both at 3000 steps (`b32_lossless.yaml:14`, `vqvae.py:136-146`). Official restart, if enabled, also writes `ema_w` differently (unscaled vector, every step).
5. Codebook init scale. Official `normal_()` then EMA (`quant.py:299`). Local \(0.02\) Gaussian plus a one-batch data init (`vqvae.py:82-84`, `112-120`).
6. Pre-quant norm. Official GroupNorm, groups 8 or 4, affine false (`owt.yaml:82`, `lm1b-small.yaml:63`, `vqvae.py:59-67`). Local LayerNorm, affine false (`phase1/vqvae.py:210`).
7. Depth. Paper and code: 0 encoder blocks (Table 5; `enc_layers` default 0). Local: 2 encoder blocks and 2 decoder blocks (`vqvae.py:30-31`, `209-212`). Paper decoder blocks are 1 except LM1B which is 4. Extra local depth is more capacity, not less. It is listed because it is a real mismatch, not because it predicts worse accuracy.
8. Width. Local \(d=512\) vs official 384 (text) or 256 (Sudoku). Again more width, not less.
9. Tied softmax. Local ties the 248320-way head to the input embedding (`vqvae.py:214-216`). Official does not (`vae.py:233`).
10. Dropout. Local 0 (`b32_lossless.yaml:9`). Official text tokenizers 0.1 or 0.3 (Table 5). XSum and WMT also drop 0.25 before quantization (`xsum.yaml:61`, `wmt14.yaml:59`). Local has none.
11. Hierarchy. Local lossless \(K=4\), lengths 4, 8, 16, 32 on \(L=32\) (`b32_lossless.yaml:10`). Official finest level is \(L\) with \(K\) from 8 to 32 and much longer lists (YAMLs above). Their \(K=4\) TinyStories ablation still says \(\ge 99.95\%\) tokens (Appendix F.1.1), on vocab 8192, 4096 codes, 40k steps. So \(K=4\) did not produce 66% for them.
12. Plan mode (`b32_plan.yaml`) stops at length 16 on a block of 32. Official `quant.py` cannot represent a last level shorter than \(L\). Their Table 1 numbers are full-hierarchy reconstructions with \(\ell_K=L\). Comparing plan-mode token accuracy to Table 1 is not the same measurement. The file header says plan mode expects the generator, not the VQ codes, to name the tokens (`vqvae.py:11-12`).
13. Nested dropout distribution. Local: with probability 0.5, uniform over strict prefixes, which can be a single level (`vqvae.py:181-182`). Official non-LM1B YAMLs: prob 0.5, `min_keep: 1`, and the dropped latent is chosen after the VQ loss (`quant.py:525-530`). Similar. LM1B YAML `prob: 1.0` is not what this quantizer interprets as 0.7/0.3, because `w_full` is unread. Not a local-only bug.
14. Loss. Both average commitment 0.25 and codebook MSE over levels and put cross-entropy on the possibly truncated latent. Official commitment in released configs is 0.25 (`owt.yaml:123`). Paper \(\beta\) numeric value: not found. No entropy bonus on either side.
15. Optimizer details that differ even when the schedule matches OWT/LM1B peak lr: local AdamW \(\beta_2=0.95\) for every run (`train_tokenizer.py:76`). Paper LM1B tokenizer \(\beta_2=0.99\) and weight decay 0.05 (Table 5). Local weight decay 0.01 (`b32_lossless.yaml:27`), which matches OWT/TinyStories/XSum/WMT/Sudoku, not LM1B.
16. Token exposure if the local 150000-step config is actually finished: \(150000 \times 512 \times 32 = 2.4576\times 10^9\) token positions. Paper OWT tokenizer exposure is stated as about 61.68B; LM1B tokenizer about 6.80B (Appendix C.2). The in-repo 66% comment is at step 3000, which is \(3000/150000 = 1/50\) of that planned budget. Whether any local run reached 150000 steps: not found in `phase1/`.
17. Resample of the residual into the next level. Local detaches before the subtract (`vqvae.py:170`). Official does not (`quant.py:517`). Later levels' inputs therefore do not train earlier Phi layers through the residual in the local code. Phi itself still gets the codebook MSE on `z_hat`.
18. What is not a difference: straight-through on the decoder latent exists on both sides; commitment-plus-codebook MSE exists on both sides; area/linear is what official uses for every level except the learned level 0; both use EMA; both use a 0.5 Phi blend in the released text configs; neither implements an entropy loss.

### Inferences
- The paper's \(K=4\) result (\(\ge 99.95\%\) on TinyStories) rules out "only four levels" as a sufficient explanation of 66–84% on their data. It does not rule it out for vocab 248320.
- The in-repo 66% number is explicitly a 3000-step code-token accuracy after a restart-threshold change, not a finished comparison to Table 1. The 80% number is a \(d=128\), 1500-step CPU model. Using those as if they were full-budget, full-width, all-token accuracies would overstate the gap.
- The combination that is actually different in a way their ablations never tested is: large Qwen vocab, code-heavy ids, interpolate instead of a trained level-0 sampler, restarts left on, and (if the 66% comment is the run in question) two orders of magnitude fewer updates than Table 5.
- WMT shows that vocab 151680 can still reach 98.79% with their recipe, so vocab size is not a complete explanation either. It is one contributing mismatch, not a proof.

### Gaps
- No local `eval_*.json` or `log.jsonl` was in `phase1/` at write time. A measured 84% token accuracy: not found. The 66% and 53% figures exist only as a config comment about 3000-step code accuracy. The 80% figure is the README's tiny-model note, not Table 1's metric on the same data.
- Official training code that might differ from `quant.py` as checked in: not in the clone.
- HF config bytes: not fetched (404). The YAML facts above are from the git clone the README says matches the checkpoints.
- Per-level official reconstruction percentages beyond the full-hierarchy Table 1 and the OWT 99.99% sentence: not found.
