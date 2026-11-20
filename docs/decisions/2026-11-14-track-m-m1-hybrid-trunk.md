# Track M / M1: running the hybrid Qwen3.5 trunk as an NCM generator (2026-11-14)

## Problem

NCM generates a span as K level blocks (level k has l_k positions). Every position of a block must
see the prompt/prefix and its own block in both directions, and nothing of the other blocks
(block-diagonal attention; Gen. PPL 15.4 vs 19.6 for block-causal in the paper's ablation).
Qwen3.5-9B has 32 layers, of which only 8 are `full_attention`. The other 24 are `linear_attention`
(Gated DeltaNet): a causal depthwise conv (kernel 4) over the q/k/v projections, then a recurrent
delta-rule state. A different attention mask does nothing in those 24 layers.

## Design (implemented in `.internal/research/ncm_trunk/ncm_trunk.py`)

| Part | Full-attention layer (8) | Linear-attention layer (24) |
|---|---|---|
| Prefix (prompt + generated spans) | unchanged causal attention; keys/values cached | unchanged causal scan; recurrent state and conv tail at the prefix end cached |
| Level block | queries attend to the cached prefix keys and to the same block, both directions | the block restarts from the cached prefix state and conv tail and is scanned left to right |
| Level block, `bidi=True` | - | plus a right-to-left scan with the same weights from an empty state and empty conv tail; `out = forward + gate_l * backward`, one learned gate per layer and channel, initialised to 0 |

- Positions: position j of level k gets RoPE position `P + floor((j + 0.5) * L / l_k)`, i.e. where
  the tokens it stands for are; the finest level sits exactly on the token positions.
- Restarting every block from the prefix state (instead of carrying the state from block to block)
  is what makes the recurrent layers block-diagonal: a block depends only on the prefix and its own
  inputs. All K levels are trained in one teacher-forced pass, and at generation time each level is
  run alone against the cached prefix with the same result.
- The right-to-left scan is the only new computation. It starts empty because the prefix lies to
  the left of the block and is not the history of a right-to-left scan (BDLM Mamba-H, arXiv
  2607.02805, eq. 12; changed after the review in
  `docs/reviews/2026-11-14-track-m-best-practice-review.md`). With the gate at 0 the trunk starts as the
  pretrained model; whether two-sided recurrent layers are needed is an M1 ablation (`bidi` on/off;
  `causal=True` gives the block-causal baseline).
- The blocks of one span are batched across levels (padded to the longest level, initial state
  repeated), one kernel call per layer.

Alternatives not taken: carrying the recurrent state across levels (breaks block independence, the
levels could then not be trained in one pass or cached per level); replacing the 24 layers by new
attention layers (throws away most of the pretrained trunk); zero-state blocks (no access to the
prefix in 24 layers).

## Verification (`.internal/research/ncm_trunk/test_ncm_trunk.py`, Qwen3.5-9B, B200)

| Check | float32 | bfloat16 |
|---|---|---|
| prefix pass = Hugging Face forward (relative L2 / same next token) | 0.0000 / 100% | 0.012 / 100% |
| causal block on the real continuation tokens = Hugging Face forward | 0.0000 / 100% | 0.012 / 100% |
| left-padded prefix = unpadded | 0.0000 | 0.012 |
| changing one level moves no other level | 0 | 0 |
| last position of a level reaches the first (block-causal: 0) | 0.022 | 0.035 |
| one level alone = its slice of the all-levels pass | 1e-5 | 0.030 |
| bidi gate 0 = one-directional; gate 0.5 differs; one level alone at gate 0.5 = its slice | 0 / 0.80 / 8e-6 | 0 / 0.80 / 0.028 |
| gradients reach linear layers, attention layers, the prefix and the gates | yes | yes |

The bfloat16 differences are rounding (they vanish in float32). Two pitfalls found: a boolean
attention mask and the cuDNN attention backend both give NaN gradients in torch 2.14; the module
uses an additive finite mask and excludes that backend.

## Cost

One span = one prefix extension + sum(l_k) block positions: 641 for the 12 LM1B levels of a
128-token span, in 12 sequential trunk calls at generation time (or one call in training). That is
~5x the token-positions of autoregressive decoding for the same 128 tokens; the possible gain is in
latency at low concurrency, not in throughput.

## Not done yet (M1 proper)

Block inputs (cumulative tokenizer reconstruction resized to l_k and projected to 4096, level
embedding, code corruption), code heads, loss weights, LoRA/full fine-tuning, multi-span training
data, and the generation loop. M1 starts only if the span tokenizer passes the round-trip gate.
A cheap pre-test remains: a frozen-trunk probe on the coarsest levels.
