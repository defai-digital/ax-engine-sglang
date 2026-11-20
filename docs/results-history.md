# Results history (2026-11-13 to 2026-11-17)

A record of what was tried, what was measured and what was decided, in order, so later work does
not repeat a closed experiment or lose a number. Each entry gives the headline result and points to
the full report or decision record. Numbers are measured end to end unless marked "projected" or
"offline". Numbers from different GPUs (B200, H200, RTX PRO 6000, Apple M5) are not comparable with
each other.

Throughout: greedy decoding, thinking off, "code / math / chat" = HumanEval / GSM8K / chat prompts
(`benchmarks/baselines/prompts.py`), "x" = ratio of throughput (total output tokens / total time), 95% intervals
from paired bootstrap over prompts (`benchmarks/sglang/paired_ci.py`) where given.

## 1. Original goal: convert Qwen3.6-35B-A3B into an NCM model (2026-11-13/14)

| Step | Result | Record |
|---|---|---|
| Reproduce NCM (HazyResearch/ncm c702742, released checkpoints, RTX PRO 6000) | XSum R-1/2/L 34.92/12.71/27.93 (paper 35.0/12.8/28.0); WMT14 De-En BLEU 22.62 lowercased (paper 22.6); Sudoku hard 86.25% (paper 86.1%); LM1B gen. PPL 94.19 ± 2.18. The paper reproduces. | `results/phase1_repro/REPORT.md` |
| First NCM-style drafter (coarse plan codes -> tokens) | 2.0 accepted tokens teacher-forced, 0.07 free-running: the gold codes leak the target; predicted codes 11-29% accurate. | `docs/decisions/2026-11-14-drafter-architecture.md` |
| Drafter ablation (0.8B stand-in verifier, 2,000 steps) | Accepted prefix: one-pass 32 tokens 0.20; chunks 0.45; + prefix-weighted loss 0.65; + encoder-state init 2.15. | `results/phase2_ablation/REPORT.md` |
| Target change | 35B-A3B dropped (cheap MoE verifier, strong MTP); target Qwen/Qwen3.8-27B (dense, Apache-2.0). 35B weights deleted. | `docs/decisions/2026-11-14-retarget-qwen38-27b.md` |
| NCM-Draft v1, Concept-Spine (Track A v2) | Designed and reviewed, stopped before any GPU run: in lossless drafting only the exact-match prefix counts, and a block-level code breaks position 1. | `docs/decisions/2026-11-14-ncm-draft.md`, `docs/decisions/2026-11-14-track-a-v2-concept-spine.md` |

## 2. Public drafter baselines (Track A, 2026-11-14)

H200, SGLang 0.5.22 + DSpine plugin, Qwen3.8-27B, concurrency 1, tok/s (accepted length):

| set | MTP7 | DFlash2 (block 8) | DSpine (block 16) |
|---|---|---|---|
| code | 272.9 (6.99) | 351.2 (7.41) | 511.5 (12.40) |
| math | 242.7 (5.67) | 323.5 (6.17) | 363.5 (7.36) |
| chat | 138.0 (3.16) | 178.0 (3.28) | 154.6 (2.99) |

Lesson: every public drafter accepts about 3 tokens per round on chat; that number limits any
speed-up. Reports: `results/track_a_stage0_h200/REPORT.md`, B200 lanes in
`results/track_a_baselines/REPORT.md`.

## 3. Lossy block decoding (Track B, 2026-11-14) — closed

Qwen3-8B, released DSpine-b16, no training. Best policies within 2% relative accuracy give 1.25x
tokens per round on GSM8K and 1.05x on HumanEval (Go needed 1.5x on both). Committing 4 unverified
draft tokens per round costs 53 points on GSM8K and 22 on HumanEval. Relaxing the acceptance rule of
a lossless-trained drafter is not a path to a large speed-up.
`results/track_b_decisive/REPORT.md`, `docs/decisions/2026-11-14-track-b-lossy-block-decode.md`.

## 4. Draftable trunk and NCM decoder on 9B (Tracks C, M, N, 2026-11-14) — stopped

Proof-of-concept scale moved to Qwen3.5-9B (same architecture family) with a 2,000 GPU-hour
ceiling. Measured before stopping:

- NCM span tokenizer round trip (Track M0): HumanEval 82.9 -> 75.6 at 23% of training (GSM8K unchanged).
- NCM span cost (`.internal/research/ncm_trunk/bench_span.py`, 9B, B200, HF kernels): 128 greedy AR tokens 3,302 ms; one
  128-token NCM span 499 ms (6.6x), 822 ms with the reverse scan of the recurrent layers (4.0x). This
  is a ceiling before any generator quality loss.
- The lossless DFlash chain already gives 5.06x / 3.88x / 1.61x over AR on the 9B (next section),
  with no quality loss.
- Reviewers put the odds of the trunk-conversion options at about 5% (Track C C1: 12-25%).

Decision: NCM and target training stopped; main line becomes lossless tree speculation with a
frozen target. `docs/decisions/2026-11-14-track-c-draftable-trunk.md`,
`docs/decisions/2026-11-14-track-m-m1-hybrid-trunk.md`,
`docs/decisions/2026-11-14-track-n-ncm-decoder-9b.md`,
`docs/decisions/2026-11-14-track-l-lossless-tree-speculation.md`.

## 5. Lossless tree speculation (Track L, 2026-11-14/15) — main line

### 9B (Qwen3.5-9B + z-lab DFlash, B200)

Served baselines, SGLang 0.5.23, DFlash chain block 16:

| set | c=1 vs AR | c=8 vs AR | c=32 vs AR | accept |
|---|---|---|---|---|
| code | 5.06x | 2.81x | 2.05x | 11.9 |
| math | 3.88x | 2.83x | 2.07x | 7.2-7.5 |
| chat | 1.61x | 1.29x | **0.73x** | 2.79 |

Tree (DForkTree plugin, 32 nodes, overhead fixes, block 24), paired, 3 repeats, c=1:
1,682 / 1,330 / 574 tok/s; vs best stock chain 1.158 [1.120, 1.191] / 1.257 [1.203, 1.319] /
1.164 [1.147, 1.187]. Lesson: the first plugin runs lost most of the tree's gain to per-round host
bookkeeping; removing the device-to-host readbacks (`benchmarks/sglang/dft_fast.py`) recovered it.

### 27B (Qwen3.8-27B + DFlash2, B200, c=1)

DFlash2 lattice tree (`tree32cfl_b16`: 32 nodes from DFlash2's 16 x 16 transition lattice, block
16), final paired run, 2 repeats:

| lane | code | math | chat | vs chain8n (code / math / chat) |
|---|---|---|---|---|
| chain8n (DFlash2 as shipped) | 500.4 | 456.3 | 250.9 | 1 |
| chain16n | 665.6 | 523.9 | 224.7 | 1.330 / 1.148 / 0.896 |
| tree32cfl_b16 | 741.7 | 625.0 | 295.9 | 1.482 / 1.370 / 1.179 |
| tree64cfl_b16 | 667.2 | 594.4 | 279.8 | 1.334 / 1.303 / 1.115 |

- 64 nodes are slower than 32 everywhere.
- vs the faster stock chain of each family: 1.114 / 1.193 / 1.179.
- Offline replay had predicted chat as the largest gain; live, chat gains less because a 32-node
  verify costs more than an 8-token one.
- The released plugin overflowed an int32 offset on the 27B (fixed, commit 1913f33).
- Token identity: no speculative lane, the shipped chain included, is token-identical to AR in bf16;
  the tree differs from AR on 84 of 192 prompts against 86 for chain8n. fp32 adjudication of every
  first divergence finds numerical flips (both tokens within 0.22 nats of the fp32 winner); none is a
  token the fp32 reference rejects.

16-node tree, same protocol, 1 pass (`track_l_t16`; the second pass was stopped part-way on
2026-11-15 to free the GPU for multi-user Stage 0): 717.2 / 598.1 / 284.6 tok/s = 1.426 / 1.308 /
1.134 vs chain8n (32 nodes in the same pass: 1.480 / 1.370 / 1.180).

Report: `results/track_l_l0/REPORT.md`.

### Standalone plugin `ax-engine-sglang` (2026-11-15)

Installable plugin that generates its worker from the installed SGLang. 0.5.22: 192/192 outputs
token-identical to the patched DForkTree plugin at the same speed; 0.5.23: 192/192 identical. Speed
on 0.5.23, 2 passes: 744.3 / 625.0 / 296.0 tok/s = 1.489 [1.462, 1.516] / 1.369 [1.339, 1.398] /
1.180 [1.165, 1.196] vs chain8n (chain8n 499.9 / 456.4 / 250.8; chain16n 675.0 / 525.1 / 224.6).
`ax-engine-sglang/README.md`, `results/track_l_l0/REPORT.md`.

## 6. Multi-user serving, Stage 0 (2026-11-15)

### Stock DFlash2 on the 27B, bf16, B200, SGLang 0.5.23 (total output tok/s)

From `docs/decisions/data/2026-11-15-multiuser/` (64 requests per family, 1,024 tokens). Chain
lanes at c = 1, 4, 8, 32 are the mean of two runs (mu0, mu2), which differ by up to 10% at one level
(chain8n code c=8); c = 16 and AR are one run (mu2):

| set | c | AR | chain8n | chain16n |
|---|---|---|---|---|
| code | 1 | 100 | 491 | 660 |
| code | 4 | 355 | 1,349 | 1,667 |
| code | 8 | 618 | 1,959 | 2,412 |
| code | 16 | 1,013 | 2,821 | 2,974 |
| code | 32 | 1,224 | 3,453 | 3,231 |
| math | 1 | 102 | 454 | 522 |
| math | 4 | 360 | 1,366 | 1,549 |
| math | 8 | 654 | 2,181 | 2,223 |
| math | 16 | 1,131 | 3,161 | 2,830 |
| math | 32 | 1,432 | 3,636 | 3,349 |
| chat | 1 | 102 | 250 | 225 |
| chat | 4 | 376 | 830 | 715 |
| chat | 8 | 694 | 1,386 | 1,093 |
| chat | 16 | 1,181 | 2,066 | 1,523 |
| chat | 32 | 1,997 | 2,726 | 1,772 |

Lessons:
- The best block length depends on the load: block 16 wins on code up to 16 requests and loses at
  32; on chat block 8 wins at every load. This is the basis of the load-adaptive width design.
- Round profile (mu1): the verify round grows with requests (block 8 / 16: 11.5 / 13.2 ms at c=1,
  21.1 / 33.3 ms at c=32); the drafter stays at 1.5-2.9 ms.
- Memory: the recurrent-state scratch is sized slots x draft tokens, about 0.14 GB each on the 27B
  (block 16 with 32 slots: 74.3 GB scratch, 69,696 KV tokens left). A 32-node tree cannot serve many
  requests; the multi-user server uses 16 draft tokens (16-node tree at one request).
- Test limit set by the project owner: 16 concurrent requests (c = 1, 2, 4, 8, 16). The c=32 rows
  above are kept as reference; 64 requests may be tested later.

Design: `docs/decisions/2026-11-15-multiuser-width-policy.md` (Stage 0 approved by codex, kimi,
muse, GLM 5.3 and grok). Stage 0 (pinned widths, eager, correctness first) is being tested.

### NVFP4 27B (RadixArk/Qwen3.8-27B-NVFP4, Apache-2.0), stock DFlash2, B200 (2026-11-15)

NVFP4 MLP and lm_head, FP8 attention and FP8 KV cache; weights 20.1 GB (bf16 about 54 GB). With
block 16 and 32 slots the server still allocates the 74.25 GB scratch and has 1,499,456 KV tokens
(bf16: 69,696). Same prompts and settings as the bf16 table (`track_n_nvfp4`, 1 pass, no c=16):

| set | c | AR | chain8n | chain16n | chain8n / chain16n vs bf16 |
|---|---|---|---|---|---|
| code | 1 | 188 | 720 | 893 | 1.47 / 1.35 |
| code | 4 | 622 | 1,838 | 2,063 | 1.36 / 1.24 |
| code | 8 | 1,041 | 2,579 | 2,874 | 1.32 / 1.19 |
| code | 32 | 1,776 | 4,023 | 4,549 | 1.17 / 1.41 |
| math | 1 | 195 | 696 | 745 | 1.53 / 1.43 |
| math | 4 | 675 | 1,982 | 2,005 | 1.45 / 1.29 |
| math | 8 | 1,164 | 3,075 | 2,614 | 1.41 / 1.18 |
| math | 32 | 2,397 | 5,530 | 4,193 | 1.52 / 1.25 |
| chat | 1 | 197 | 394 | 336 | 1.58 / 1.49 |
| chat | 4 | 708 | 1,258 | 1,018 | 1.52 / 1.42 |
| chat | 8 | 1,259 | 1,944 | 1,441 | 1.40 / 1.32 |
| chat | 32 | 3,321 | 3,772 | 2,348 | 1.38 / 1.33 |

NVFP4 is a different model from the bf16 target: speculation on it reproduces NVFP4 decoding, not
bf16 decoding, and its task accuracy against bf16 has not been measured yet. Data:
`docs/decisions/data/2026-11-15-nvfp4/`. On NVFP4 block 16
stays ahead of block 8 on code at 32 requests (4,549 vs 4,023), unlike bf16.

## 7. Mac port (ax-engine, Apple M5, 2026-11-15, in progress)

Qwen3.8-27B MXFP4, 16 prompts per family, 512 tokens, 2 repeats, tok/s (code / math / chat):

| MLX | direct | DFlash2 |
|---|---|---|
| 0.32.4 | 33.8 / 33.2 / 33.1 | 116.2 / 88.3 / 55.2 and 97.4 / 81.2 / 53.2 |
| 0.32.5 (built from main 914732419) | 31.7 / 31.8 / 31.9 | 150.4 / 121.9 / 66.9 and 143.9 / 118.1 / 65.3 |

- A trace showed the block-16 override never verified more than 9 rows, so an earlier "block 16 is
  slower" conclusion was withdrawn.
- A 9-row verify costs 104-113 ms against 60-67 ms for 8 rows.
- Reviewers' next step: measure a branched draft's acceptance gain and the packed 2-4 x <=8-row cost
  (about 44 ms on 0.32.5) before writing kernels. Work is on branch `track-l-lattice-tree` of the
  ax-engine repository, not merged.

## 8. Multi-user server with two attention backends, 27B (2026-11-16/17) — done

Tree rounds on `triton` at one request, chain rounds of width 16 / 8 on private `trtllm_mha`
backends with their own CUDA graphs. Record: `docs/decisions/2026-11-16-dual-attention-backend.md`.

| Step | Result | Section |
|---|---|---|
| Stage 1a correctness | identity, teacher audit and stress pass after four bugs fixed (buffer swap capacities, FP8 probe slot, teacher counting) | 10 |
| bf16 speed, 16 slots, 2 passes | c = 1: 1.09x / 1.15x / 1.15x (code / math / chat) vs. the faster stock block; c = 2-16: 0.93x-1.09x, below the 0.98x gate at math c8 and code c8 width 8 (scheduling, not rounds) | 11 |
| NVFP4 speed, 32-64 slots | after the fix: c = 1 code 1.42x, math 1.33x, chat 1.14x; c = 2-64 1.01x-1.21x | 12, 14 |
| Sampled, grammar, logprob requests | stopped the server before; now run as chain rounds; acceptance equals stock | 15 |
| Sampled tree verify (`AX_TREE_SAMPLING=1`) | bf16: 0 of 28 distribution tests fail (stock DFlash2 sampling fails 7, an upstream defect); NVFP4 kernels give row-count-dependent distributions; kept off by default (no speed gain shown) | 16 |
| TensorRT-LLM 1.4 DFlash2 | ours leads at c = 1 (1.07x-1.72x) and on code; TensorRT-LLM leads chat c = 16 (0.81x bf16, 0.90x NVFP4). Same acceptance; its batch-16 round is 14-20% cheaper, for stock SGLang too | 17 |

## 9. Qwen3.8-Flash-Next (2026-11-17) — MTP done, tree blocked

Record: `docs/decisions/2026-11-17-qwen38-flash-next-feasibility.md`; report:
`results/flash_next/REPORT.md`.

| Step | Result |
|---|---|
| Feasibility | DFlash2 refused above 4 tokens (QSA cap) and crashes at 4 (Qwen4-Exp returns no aux hidden states); a tree also needs the PLE n-gram context and tree-aware QSA selection |
| Stock baselines (RadixArk NVFP4) | AR 222 / 227 / 229 tok/s, MTP 3 508 / 493 / 380 at c = 1 |
| QSA ring bug (stock 0.5.23) | 44-66% of compressed index groups built from overwritten keys on 6,000-token contexts, MTP 1-3 included; ring of 16 slots leaves only a ring-independent 0.3-0.4% |
| Wider MTP with the ring | MTP 7 vs. stock MTP 3, two passes within 1.5%: code 1.22x-1.30x, math 1.05x-1.15x, chat 0.80x-0.86x; MTP 9 / 12 no better (worse at c = 16) |
| Plugin bug found | the tree's capture patches limited every server's target graph to batch 1; now scoped to the tree algorithm |

## Lessons that apply across tracks

- Compare against the best public baseline on the same GPU and serving stack; stock settings change
  with load (block 8 vs 16).
- Measure live before believing an offline replay: per-round overhead and verify cost decide the
  gain.
- Lossless in bf16 means "same as the target's own speculative decoding up to numerical flips";
  check token identity and adjudicate divergences.
- Memory, not only speed, limits multi-user tree decoding on the recurrent (Gated DeltaNet) layers.
- A plugin's global patches must check that the server runs the plugin's own algorithm; a stock
  mode loaded with the plugin silently lost its multi-request CUDA graphs.
- Check the engine's own invariants under the new load: a cap that looked safe (verify window <= the
  QSA ratio) still corrupted state whenever a window crossed a group boundary.
- Run the speed comparison twice before claiming a ratio; here every cell repeated within 1.5%.
