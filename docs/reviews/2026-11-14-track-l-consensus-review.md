# Track L design: five-reviewer consensus record

Date: 2026-11-14. Reviewers: grok, muse, codex (gpt-6.1-sol), agy, kimi CLIs, each given the same
brief with web search, asked to compare with the NCM paper and related projects and to end with
`VERDICT: BEST_PRACTICE=<yes|no>` and their blocking changes. The loop ran until every reviewer
answered yes. Result: `docs/decisions/2026-11-14-track-l-lossless-tree-speculation.md`.

| Round | Design | grok | muse | codex | agy | kimi |
|---|---|---|---|---|---|---|
| 1 | v1: train a large-block (16-32) self-speculative 9B, FLARE-style two-stream objective | no | no | no | no | (no final answer; reasoning against) |
| 2 | v2: frozen target + public block-diffusion drafter + tree, measurement first | no | yes* | yes* | no | no |
| 3 | v3: + tuned chain/DFlash2 baselines, verify-width first, released DForkTree run, correctness gate, stage caps, measured offline headroom | no | no | no | yes | no |
| 4 | v4: + exact sample-and-match sampler and its tests, fastest chain per task as reference, measured beats projected, provenance of the offline numbers | yes | no | yes | yes | yes |
| 5 | v5: + GL1 coverage at every concurrency with CI and latency, AR fallback | (v4 yes) | yes | (v4 yes) | (v4 yes) | (v4 yes) |

\* yes with listed blocking changes.

## What the reviewers changed

- **Round 1 (all):** a full-model self-drafter cannot beat a small feature-conditioned drafter per
  unit of compute: the draft costs full target forwards, FLARE shipped horizon 4, Nemotron-Labs-
  Diffusion accepts 5.5-6.8 of 31 drafted tokens; "lossless against the fine-tuned stream" can still
  lose against the original model (FLARE: -22 IFEval). Accept length per verify is the wrong gate;
  gate on tokens/s with draft passes included. Preferred option for all: frozen target + drafter +
  tree. Projects we had missed: xPress, Multi-Future Drafting, DFlare, DSpark, HybridSpec,
  Packed-SS, Nemotron-Labs-Diffusion, DDTree, CaDDTree, the public Qwen3.8-27B DFlash2 drafter.
- **Round 2:** tuned served baselines (incl. DFlash2 and its path selector) before any tree work;
  verify-width microbenchmark before headroom decisions; correctness (token agreement, sampling
  identity) as the quality gate at noise tolerance instead of a 5% task tolerance; no regression at
  the highest served concurrency; licence check (DForkTree is Apache-2.0); stage caps; ceiling tests
  before any drafter training; drop the claim that no larger speedup has evidence.
- **Round 3:** specify the exact tree sampler (sample-and-match) and test joint, not marginal,
  distributions; compare against the fastest chain block per task (block 24 beats a block-16 tree on
  HumanEval); a measured plugin run overrides the offline projection; show that the offline tree
  numbers come from a budget-constrained tree on drafted prefixes with an uncontaminated n-gram
  table (unit tests added).
- **Round 4 (muse):** GL1 must hold per task, temperature and concurrency 1/8/32 with paired CIs and
  TTFT/p50/p99 non-inferiority, with AR fallback where the chain itself regresses.

## Comparison with the NCM paper and related projects (as used in the brief; sources verified by us unless marked)

| Project | Approach | Evidence | Role here |
|---|---|---|---|
| NCM (arXiv 2610.14162) | latent coarse-to-fine spans, generators trained from scratch (97M-725M) | speed 10-31x against diffusion LMs only; no code tasks; no pretrained-LM conversion | left: measured ceiling 4.0-6.6x before quality loss, tokenizer loses 8.8% on HumanEval at step 6,000 |
| FLARE (2606.01774) | Qwen3.5-9B converted to two-stream block diffusion, block 4, ~10B tokens | GSM8K 93.1 vs 89.2, HumanEval 82.3 / 92.1 vs 95.1, IFEval 63.2 / 71.4 vs 91.3; no speed vs AR | ideas only (licence PolyForm NC) |
| Fast-dLLM v2 (2509.26328) | Qwen2.5-7B block diffusion, block 32, ~1-3B tokens | up to 2.5x over AR at parity with an NTP-tuned baseline | below the lossless baseline |
| Dream 7B / DiffuLLaMA | full diffusion conversions, 65-580B tokens | at or below the AR parent | out of budget |
| Nemotron-Labs-Diffusion (2607.05722) | full-model self-speculation, drafts 31 | accepts 5.5-6.8, ~4x AR (reviewer-reported) | evidence against v1 |
| DFlash / DFlash2 (z-lab) | small block-diffusion drafter reading target features, chain | ours: 9B 5.06x / 3.88x / 1.61x (code/math/chat) at concurrency 1; 27B 4.86x / 4.43x / 2.43x | baseline |
| DDTree, EAGLE-3 tree | trees over drafter marginals | 9B: 4.27x / 4.41x average (DForkTree's table) | L0/L1 |
| DForkTree (2611.12356, Apache-2.0) | fork-aware PMI re-scoring, closed-form Gated DeltaNet tree verification | 9B: 4.75x average, accept 7.96, batch 1, RTX 5090 | L0(d), L1 starting point |
| xPress, Multi-Future Drafting, DFlare, DSpark, HybridSpec, Packed-SS, CaDDTree | refiners, multi-future drafters, larger drafters, load-aware verify budgets | reviewer-reported, not verified by us | L3 candidates behind ceiling tests |

## Odds given by the reviewers (round 2, for the v2 direction)

Reviewers put v1 (large-block self-speculation) at 5-6% to beat the lossless baseline by 1.5x, and
the lossless tree line at 70-85% to hold a large gain over AR on this model.
