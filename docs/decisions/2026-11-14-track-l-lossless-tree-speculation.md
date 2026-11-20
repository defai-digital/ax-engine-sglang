# Track L: lossless tree speculative decoding (replaces Track M / NCM as the main line)

Date: 2026-11-14. Status: design accepted by all five reviewers (grok, agy, kimi, codex on v4; muse
on v5, which only tightened v4); L0 partly measured; no training started.
Review record: `docs/reviews/2026-11-14-track-l-consensus-review.md`. Measurements:
`results/track_l_l0/REPORT.md`.

## Goal

Make Qwen/Qwen3.5-9B (Apache-2.0; 8 full-attention + 24 causal Gated DeltaNet layers), then
Qwen/Qwen3.8-27B, generate as fast as possible with the output of the original model, measured end
to end in one serving stack. Ceiling 2000 GPU-hours on 2x B200. Standing rule from the owner: no GPU
time on a design known to be wrong; small tests first.

## Why the line changed

- NCM (Track M), measured: tokenizer round trip HumanEval 82.9 -> 75.6 at 23% of training (GSM8K
  unchanged); one 128-token span costs 499 ms (822 ms with reverse scans) against 3,302 ms for 128
  greedy tokens in the same unoptimised stack, i.e. a ceiling of 4.0-6.6x before any generator
  loss. The lossless baseline below is already at 3.9-5.1x on code/math with zero loss.
- Converting the target itself (FLARE on this exact 9B, Fast-dLLM v2, Dream) costs 1-580B tokens and
  loses quality (FLARE's verified mode: IFEval 71.4 vs 91.3 for the parent); block 4 caps at 4 tokens
  per forward; full-model self-speculation with long blocks accepts 5.5-6.8 tokens
  (Nemotron-Labs-Diffusion) at the price of full-model draft passes.
- All five reviewers preferred a frozen target with a small feature-conditioned block-diffusion
  drafter and a tree over every option that trains the target.

Ideas taken from FLARE (no code, no weights; its licence is PolyForm Noncommercial): Gated DeltaNet
blocks restart from the clean boundary state; verification and next-block drafting can share one
target forward; transfer-data match matters more than loss/mask design.

## Design

The target is frozen and never fine-tuned. A small block-diffusion drafter (public Apache-2.0
weights: z-lab/Qwen3.5-9B-DFlash, z-lab/Qwen3.8-27B-DFlash2) proposes; candidates are arranged in a
tree; the target verifies the tree in one forward; acceptance is exact.

- **Tree.** One drafter pass gives per-position marginals; the top-16 candidates per position are
  kept; the tree is the best prefixes under a hard node budget from {8, 16, 32, 64}, scored by the
  product of the marginals re-scored with a trigram PMI of the target's own answers (weight 0.3 on
  the top-1 spine, 0.6 after a fork; DForkTree, arXiv 2611.12356). Siblings have unique token ids.
- **Acceptance (sample-and-match).** The tree is built without looking at any target output. From
  the root, the next token is drawn from the target's own processed distribution at the current
  node (argmax when greedy); if it equals a child, move there and repeat; otherwise emit it and end
  the round. Every emitted token is a sample of the target's conditional distribution given the
  emitted prefix, so the joint distribution is the target's for any tree.
- **Gated DeltaNet verification.** No per-node recurrent state: key, corrected value and cumulative
  log-decay per node (log-decay in fp32), state after acceptance in closed form, checked against
  native replay (relative L2 <= 1e-3 in fp32; bf16 reported).
- **Load awareness.** The node budget shrinks under load; per (task class, temperature, load) cell
  the server uses the best of tree, chain and AR.
- **Cost identity** for every decision: speedup S = L * t_AR / T_round, L = tokens emitted per
  round, T_round = draft + tree build + verify + state commit, measured in the serving stack.

## Stages, caps and gates

Caps in GPU-hours on B200: L0 <= 20, L1 <= 60, L2 <= 60, each L3 pilot <= 40. A stage that reaches
its cap without passing its gate stops and is reported.

| Stage | What | Gate |
|---|---|---|
| L0 | Measurement only. (a) tuned served baselines: AR and DFlash chain at block 8/16/24 (27B: DFlash2 with its selector and the block-16 chain), concurrency 1/8/32, temperature 0 and 1, code/math/chat/instruction following, long-context smoke. (b) verify-width microbenchmark, single and batched, 8-64 nodes, native replay vs closed form. (c) offline tree headroom (done), plus top-M recall and share of non-coherent nodes. (d) the released Apache-2.0 DForkTree plugin as is on the 9B against a verifier-matched chain. | **GL0**, per task family and temperature: measured (L0d) tokens/s, or a labelled projection if L0d cannot run, >= 1.15x the fastest tuned chain at concurrency 1 on at least two of three families with a bootstrap CI lower bound > 1.0, and a node budget with S >= 1.0x at concurrency 8 and 32. Else ship the tuned chain and stop. |
| L1 | Engine work, no training: tree drafting, single-pass tree verification, PMI re-scoring, load-aware budget in the current SGLang, starting from the DForkTree plugin (attributed). | **GL1**. Correctness: coupled-randomness identity (per request seed and absolute position) between AR, chain and tree, token for token, first divergence checked against an fp32 single-step reference with the logit gap recorded; sequence log-probability identity; joint test on the first 4 sampled tokens with a pre-registered sample size; EOS/stop, rollback, long context. Speed, per task family, temperature and concurrency 1/8/32: paired 95% bootstrap CI of tokens/s against the best of {AR, tuned chain}, lower bound >= 1.0x everywhere and >= 1.15x at concurrency 1; TTFT, p50, p99 non-inferior (5% margin). Task scores are a screen at noise tolerance (<= 1%). |
| L2 | Qwen3.8-27B with DFlash2 (does its selector compose with a tree?). | same gates |
| L3 | Optional training of the small drafter only: fork re-anchoring, a Jacobi/xPress-style refiner, drafter capacity, chat-domain data. | Before training: an online oracle/ceiling test per item; a capped pilot only if >= 15% tokens/s is at stake at concurrency 1 and survives at concurrency 8. |

Not done: fine-tuning the target; block-diffusion, FLARE-style or NCM conversions of the target;
speed claims taken from other hardware.

## Expectation (projection, not measured)

Offline gains in tokens per round of +13% (code), +29% (math), +39% (chat) for a 32-node tree over
the best chain block. If a tree round costs about as much as a chain round at concurrency 1
(DForkTree reports +1%; to be measured here), that is roughly 5.5-6x on code, 5x on math and
2.2x on chat over AR on the 9B at concurrency 1, and less under load.
