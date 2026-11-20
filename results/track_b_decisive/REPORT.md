# Block decoding under relaxed acceptance: `/dev/shm/ax/results/track_b_decisive`

Settings: target `Qwen/Qwen3-8B`, draft `/dev/shm/ax/spine8b_t/train_init`, draft_config `None`, max_new_tokens `512`

## gsm8k (300 prompts)

| policy | accuracy | vs exact (95% CI) | tokens/round | x exact | non-argmax tokens | same output as greedy | mean tokens |
|---|---|---|---|---|---|---|---|
| exact | 0.947 | - | 9.38 | 1.00 | 0.000 | 1.000 | 259 |
| force_15 | 0.087 | -0.860 [-0.897, -0.820] | 15.61 | 1.66 | 0.146 | 0.000 | 297 |
| force_4 | 0.413 | -0.533 [-0.590, -0.473] | 4.94 | 0.53 | 0.052 | 0.000 | 265 |
| force_8 | 0.237 | -0.710 [-0.760, -0.657] | 8.86 | 0.94 | 0.090 | 0.000 | 267 |
| ratio_0.03 | 0.930 | -0.017 [-0.043, +0.010] | 12.26 | 1.31 | 0.047 | 0.003 | 261 |
| ratio_0.1 | 0.947 | +0.000 [-0.030, +0.030] | 11.77 | 1.25 | 0.040 | 0.003 | 258 |
| ratio_0.3 | 0.953 | +0.007 [-0.020, +0.033] | 10.97 | 1.17 | 0.027 | 0.027 | 259 |
| ratio_0.5 | 0.960 | +0.013 [-0.010, +0.037] | 10.42 | 1.11 | 0.017 | 0.063 | 257 |
| topk_2 | 0.523 | -0.423 [-0.483, -0.367] | 12.10 | 1.29 | 0.048 | 0.000 | 267 |
| topk_5 | 0.250 | -0.697 [-0.750, -0.643] | 13.47 | 1.44 | 0.077 | 0.000 | 269 |
| typical_0.09_0.3 | 0.940 | -0.007 [-0.037, +0.023] | 11.69 | 1.25 | 0.038 | 0.003 | 259 |

## humaneval (164 prompts)

| policy | accuracy | vs exact (95% CI) | tokens/round | x exact | non-argmax tokens | same output as greedy | mean tokens |
|---|---|---|---|---|---|---|---|
| exact | 0.848 | - | 11.11 | 1.00 | 0.000 | 1.000 | 202 |
| force_15 | 0.030 | -0.817 [-0.872, -0.756] | 15.67 | 1.41 | 0.149 | 0.000 | 286 |
| force_4 | 0.628 | -0.220 [-0.293, -0.146] | 4.92 | 0.44 | 0.037 | 0.116 | 198 |
| force_8 | 0.305 | -0.543 [-0.622, -0.463] | 8.79 | 0.79 | 0.088 | 0.098 | 211 |
| ratio_0.03 | 0.829 | -0.018 [-0.055, +0.018] | 11.85 | 1.07 | 0.012 | 0.317 | 203 |
| ratio_0.1 | 0.841 | -0.006 [-0.037, +0.024] | 11.66 | 1.05 | 0.008 | 0.396 | 201 |
| ratio_0.3 | 0.841 | -0.006 [-0.018, +0.000] | 11.40 | 1.03 | 0.004 | 0.549 | 203 |
| ratio_0.5 | 0.848 | +0.000 [-0.018, +0.018] | 11.31 | 1.02 | 0.002 | 0.683 | 204 |
| topk_2 | 0.695 | -0.152 [-0.220, -0.085] | 12.16 | 1.10 | 0.030 | 0.116 | 202 |
| topk_5 | 0.561 | -0.287 [-0.360, -0.213] | 12.91 | 1.16 | 0.049 | 0.085 | 198 |
| typical_0.09_0.3 | 0.841 | -0.006 [-0.037, +0.024] | 11.63 | 1.05 | 0.008 | 0.415 | 201 |

## Reading against the pre-declared thresholds

Setup: Qwen3-8B target, released `tencent/Qwen3-8B-DSpine-b16` drafter (block 16), greedy, no
training, 512 new tokens, B200, HF implementation (`.internal/research/lossy_block_decoding/block_decode.py`). Tokens per round is
measured; no wall-clock speed was measured, and "x exact" is an upper bound on the wall-clock gain
over lossless speculative decoding. The partial `ar` run (5 HumanEval prompts) was token-identical
to `exact`.

Thresholds (`docs/decisions/2026-11-14-track-b-lossy-block-decode.md`): Go = a policy with at least
1.5x the tokens per round of `exact` on both tasks at no more than 2% relative accuracy loss;
No-go = every policy within 2% stays below 1.2x.

- Policies within 2% relative accuracy on both tasks: `ratio:0.5`, `ratio:0.3`, `ratio:0.1`,
  `typical`. Best of them: `ratio:0.1` / `typical` with 1.25x on GSM8K and 1.05x on HumanEval.
  `ratio:0.03` (1.31x / 1.07x) loses 1.8% and 2.2% relative; its intervals include zero.
- No policy reaches 1.5x on both tasks at any accuracy except `force:15` on GSM8K (1.66x), where
  accuracy falls from 0.947 to 0.087.
- Unverified blocks collapse quality at every length: committing 4 draft tokens per round without
  checking already costs 53 points on GSM8K and 22 on HumanEval, and is slower than `exact`.
  Accepting any top-2 token costs 42 and 15 points.

Result: Go is not met. Strict No-go is missed only by GSM8K at 1.25x; on HumanEval the gain is
1.05x. Relaxed acceptance of an untrained (lossless-trained) drafter is not a path to a multiple
of speculative decoding: the headroom between `exact` and the block size is small (11.1 of 16 on
code) and the tokens the target would reject are mostly wrong, not merely different. Any further
Track B work has to change the drafter (training for standalone block quality), not the
acceptance rule.
