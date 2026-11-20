# Phase 1a: NCM reproduction (released checkpoints)

Run on 2026-11-14, 1x RTX PRO 6000, HazyResearch/ncm at commit c702742, full test sets, no CFG / MBR.

| Task | Metric | Reproduced | Paper | Gate (within ~1 point) |
|---|---|---|---|---|
| XSum | ROUGE-1 / 2 / L | 34.92 / 12.71 / 27.93 | 35.0 / 12.8 / 28.0 | pass |
| WMT14 De-En | BLEU (cased / lowercased) | 21.73 / 22.62 | 22.6 | pass (lowercased matches; cased is 0.9 lower) |
| Sudoku hard | solved | 86.25% (1725 / 2000) | 86.1% | pass |
| LM1B small | gen. PPL (GPT-2 large), 5 seeds | 94.19 ± 2.18, entropy 4.31 | better than DUO (see paper) | recorded |

The released checkpoints reproduce the paper.
