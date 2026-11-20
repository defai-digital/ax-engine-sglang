# Track M best-practice review, round 2 (grok, muse, agy, codex)

Date: 2026-11-14. Same question as round 1 (`2026-11-14-track-m-best-practice-review.md`), now for the
whole approach (tokenizer, generator design, probe experiments, speed target), with four reviewers
on one brief and two required comparisons: against the NCM paper / official code, and against other
projects. "Verified" below means we opened the source ourselves (paper page, model card, or the
local clone of HazyResearch/ncm at c702742).

## Verdict (all four agree)

- Best practice **for the mechanics**: LM1B tokenizer recipe copied faithfully; causal cached
  prefix; block-diagonal levels; empty-state reverse scan; codebook-tied heads; task round trip as
  the tokenizer gate (stricter than the paper, which reports token accuracy only).
- **Not** best practice **for the stated goal** (<=5% loss incl. code, clearly faster than the same
  model token by token and than lossless speculative decoding). The projects that reached that kind
  of goal stay in token space with small blocks; the paper never compares with AR decoding of a
  large model; nothing about our speed is measured yet.

## T1. Ours vs the NCM paper and official code (verified in `config/eval/lm1b-large.yaml`, `models/`)

| Item | Official (LM1B) | Ours | Status | Ruling |
|---|---|---|---|---|
| Tokenizer recipe | width 384, 0 enc / 4 dec blocks, GroupNorm 4, dual nested dropout 0.7/0.3, AdamW b2 0.99, lr 3e-4, batch 2048, 25,957 steps | same | same | keep |
| Levels / codebook | 1,8,16,24,32,40,52,64,76,92,108,128; 8192 per level; EMA 0.99 | same | same | keep for the tokenizer; number of levels is a speed question (below) |
| Token tables | trained embedding/head, 30k-50k vocabulary | frozen Qwen tables (248k) behind learned maps | differs | necessary at 248k; unproven for rare code tokens - the round trip decides |
| Tokenizer data | task corpus, 6.8B positions | 30% own answers + 70% public, 2.2B tokens | differs | reasonable; fewer unique tokens than exposures in the paper |
| Dead-code restart (issue #16) | bug only when `restart_threshold > 0` | default 0 | n/a | not affected |
| Generator | transformer trained from scratch, 97M-725M | pretrained 9B hybrid (8 attention + 24 Gated DeltaNet layers) | differs | the goal; no precedent in the paper |
| Masks | prefix two-sided, targets block-diagonal | prefix causal, targets block-diagonal; recurrent layers restart from the prefix state, optional reverse scan from an empty state | differs | justified (cache, multi-span); same pattern as Fast-dLLM v2, BDLM Mamba-H, Clover, FLARE |
| Positions | RoPE reset to 0..l_k-1 per level | mapped to token positions | differs | keep, ablate |
| Level 0 | learned query (`query_level0`) | last prefix state (official `predict_level0` variant) | differs | keep, add a level-0-only metric |
| Inputs | projected cumulative z_hat + level embedding + zero-init post-upsample conv (k=3) | projected z_hat + level embedding | **missing** | add the post-upsample conv in M1 |
| Heads | one shared head + per-level bias (tied codebook head is an option) | codebook-tied distance heads | differs | keep; shared head as a control arm |
| Corruption | uniform replacement, rate ~ U[0,1], level 0 kept | first probes clean | **missing** | added to the probe (`--corrupt`) and required for M1 |
| Loss weights | weights by level length | plain mean over levels | missing | low stakes; specify in M1 |
| Generator training | 100k steps x 512 x 128 = 6.55B positions | ~1B tokens planned | differs | compare with Fast-dLLM v2 (~1B) and FLARE (~10B), not with from-scratch NCM |
| Sampling | temperature 1 + frequency/adjacency penalties (LM1B); CFG and MBR on XSum/WMT14 | greedy, none | missing | no CFG/MBR (they multiply the passes); penalties only if repeats appear |
| Speed claim | 10-31x **versus diffusion LMs**, ~100M models, one H100, batch 16 | none measured | - | benchmark added (`track_m/bench_span.py`) |

## T2. Ours vs other projects

| Project | What they did | Conversion data | Measured quality and speed | Verified |
|---|---|---|---|---|
| **FLARE** (arXiv 2606.01774) - same base model | Qwen3.5-2B/4B/9B to block diffusion: causal clean stream + noisy blocks (block size 4) two-sided inside the block; Gated DeltaNet blocks start from the clean boundary state and read a shared block-end state; AR loss + diffusion loss; logit shift | ~10B tokens, 9,000 steps | 9B diffusion mode vs AR parent: GSM8K 93.1 vs 89.2, HumanEval 82.3 vs 95.1, IFEval 63.2 vs 91.3; verified-AR mode HumanEval 92.1 | yes (speed numbers not read) |
| Fast-dLLM v2 (2509.26328) | Qwen2.5-7B to block diffusion, block 32, causal across blocks, shift kept, complementary masks | ~1-3B tokens, 2,500 steps | average 60.3 vs 58.2; up to 2.5x over AR | yes |
| Dream 7B (2508.15487) | Qwen2.5-7B to full-sequence diffusion, shift kept | ~580B tokens | HumanEval 57.9 vs 56.7, GSM8K 77.2 vs 78.9 | yes |
| DiffuGPT / DiffuLLaMA (2410.17891) | full fine-tune to diffusion, shift kept | 30B / ~65B tokens | no shift: -12 points (GPT-2, GSM8K-symbolic); DiffuLLaMA well below LLaMA2 zero-shot | yes |
| BYOD (2610.41574) | frozen trunk, LoRA r=1024 on attention Q/V | <=102M tokens | average 49-51% vs 59-72% for the parents; no speed claim | yes |
| BDLM Mamba-H (2607.02805) | from scratch; forward scan from the cached prefix state, reverse scan inside the block from nothing, summed | 5B (87M model) | perplexity 61.6 vs 83.7 (full two-sided Mamba) | yes |
| DiffuMamba (2511.15927) | from scratch, separate weights per direction, summed | - | 4.3x over a diffusion transformer on long inputs | yes |
| Clover-35B-A3B (HF Diffuspace) | Qwen3.6-35B hybrid, causal prefix, extra reverse-convolution parameters | not stated | vendor numbers only (not checked by us) | card only |
| Diffusion-Qwen3.5-9B (HF summerAIO) | shared reverse scan with per-channel zero gate | 20 QLoRA steps | not benchmarked | yes |
| CLLM (2403.00835) | Jacobi consistency + AR loss, token space | small, per domain | 2.4-3.4x, quality kept (reviewers' reading) | no |
| ncm-mlx | small from-scratch NCM | TinyStories | generative perplexity 24.4 vs AR 11.5 | earlier check |
| **DForkTree** (2611.12356) - same base model | draft trees on the public DFlash drafter, no training | none (n-gram counts of 100k responses) | Qwen3.5-9B: **4.75x over AR**, accept length 7.96, RTX 5090, batch 1; distribution-preserving | yes |
| Our Track C | public drafter + own training | - | accept length 6.4-6.5 | own measurement |

Two entries change the picture and were missing from round 1: FLARE has already converted this exact
model with a token-space block design, and DForkTree already gives 4.75x on it without training.

## Points all reviewers raised, and rulings

| # | Finding | Ruling |
|---|---|---|
| 1 | Speed is unmeasured; the paper's speedups are against diffusion LMs. 12 levels = 641 positions per 128 tokens. Reviewer estimates: realistic 2-4x over AR; beating accept-6.4 speculative decoding by 1.5x would need ~5-7x over AR. | **Measure now**: `track_m/bench_span.py` times one span (11 level passes + commit + decode) against token-by-token decoding in the same stack. Runs right after the first probe batch on idle GPUs. |
| 2 | Tokenizer on code: token accuracy is not evidence; HumanEval round trip must clear ~78.8 (5% relative) and leaves the generator no allowance. | Hard gate stays the round trip. Step 6,000: GSM8K 86.3 -> 86.3, HumanEval 82.9 -> **75.6** (step 4,000: 65.9). Continue training to 25,957 steps; if the final value is below 78.8, add the exact side channel (item 3). |
| 3 | Cheapest route to code-safe output: sparse exact residual (position, token id) where the decoder disagrees; the generator must then predict the corrections or fall back to verified AR on those positions. agy instead proposes encoder blocks and shorter spans. | Measure corrections per span after the tokenizer run; prefer the residual channel (no retraining of the tokenizer). Encoder blocks / shorter spans only as a second tokenizer run if the residual is too dense. |
| 4 | Probes: clean teacher-forced code accuracy can rank options wrongly (ncm-mlx saw this); 600 x 16 is a smoke test, not rejection evidence. Missing: corrupted conditioning, free-running rollout scored on decoded tokens, level-0-only. | Added to `probe_trunk.py`: `--corrupt`, accuracy under 20% corrupted conditioning, free-running rollout decoded to tokens (`free_token`, `free_span`) next to the tokenizer ceiling. Second batch: LoRA with corruption, 1,500 steps, with and without the reverse scan. Still a small-scale signal. |
| 5 | Shift: the 11-12 points are a GPT-2 token-space result; with NCM inputs already aligned to the target position, shifting inner levels may misalign (agy). | Treat as open; the probe's shift on/off arm decides. Prefix next-token loss must be trained in M1, not only logged. |
| 6 | Adapters: LoRA on projections only leaves the DeltaNet convolution and gates as pretrained (agy). | M1: also train the DeltaNet conv, gate projections and norms. |
| 7 | Fewer levels / shorter spans / early exit are the only way to a large speedup (all). | Decide after the benchmark; the tokenizer's nested dropout already allows decoding from fewer levels, which makes an early-exit measurement cheap. |
| 8 | Alternatives to privilege-check: FLARE-style token-space block design on the same trunk (agy, codex). | Not started. This is a change of track and is the owner's decision; the trunk code (`ncm_trunk.py`) would be reused. Raised in the session report. |

## Reviewer odds (percent)

| Outcome | grok | muse | agy | codex |
|---|---|---|---|---|
| Tokenizer passes incl. code | 25 (70 with residual channel) | 35 | 25 (70 with changes) | 65 |
| Generator within 5% | 15 | 25 | 15 | 20 |
| >=2x measured over AR at that quality | 20 | 30 | 65 (speed only) | 15 |
| >=1.5x over lossless speculative decoding | 8 | 8 | <10 | 5 |

## Code changed with this review

- `track_m/probe_trunk.py`, `track_m/run_probes.sh`: probe experiments, now with corruption,
  free-running rollout and decoded-token metrics.
- `track_m/bench_span.py`: span latency against token-by-token decoding.
- `track_m/ncm_trunk.py`: per-layer gradient checkpointing (every level block carries its own
  recurrent state; the stored graph did not fit otherwise).

Results of the probes and the benchmark are recorded separately when they finish.
