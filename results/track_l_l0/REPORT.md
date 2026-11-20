# Track L, stage L0: measurements (no training)

Hardware: 1x NVIDIA B200 per lane, bf16. Dates: 2026-11-14/15 (UTC). Target Qwen/Qwen3.5-9B, drafter
z-lab/Qwen3.5-9B-DFlash (public weights). The two 27B sections use Qwen/Qwen3.8-27B with
incoai/Qwen3.8-27B-DFlash2. Measured numbers only; projections are in the decision
document.

## L0(a) Served baselines, SGLang 0.5.23, greedy, thinking off (`benchmarks/sglang/run_baselines.sh`)

Same 64 requests per task for both modes. "accept" = tokens per verification round as reported by
the server. DFlash chain block 16. Block 24 at concurrency 1 is in L0(d); block 8, temperature 1 and
instruction following are still to be measured.

| Task | Concurrency | AR tok/s | DFlash chain tok/s | vs AR | accept |
|---|---|---|---|---|---|
| code | 1 | 265.6 | 1,344.5 | 5.06x | 11.89 |
| code | 8 | 1,551.3 | 4,355.5 | 2.81x | 11.21 |
| code | 32 | 3,297.7 | 6,756.1 | 2.05x | 11.94 |
| math | 1 | 273.2 | 1,060.4 | 3.88x | 7.22 |
| math | 8 | 1,723.0 | 4,867.9 | 2.83x | 7.39 |
| math | 32 | 3,837.6 | 7,934.9 | 2.07x | 7.52 |
| chat | 1 | 275.8 | 444.0 | 1.61x | 2.79 |
| chat | 8 | 1,884.0 | 2,421.6 | 1.29x | 2.79 |
| chat | 32 | 5,270.5 | 3,846.9 | **0.73x** | 2.79 |

On chat at concurrency 32 the chain is slower than plain AR: the design's fallback to AR under load
is needed, not optional.

## L0(d) Measured tree speculation: released DForkTree plugin (`benchmarks/sglang/run_tree_plugin.sh`)

Plugin: github.com/Perseus037/DForkTree v0.1.0 (commit eeed8b6, Apache-2.0), run from an overlay with
its pinned SGLang 0.5.22. Same GPU, same 64 greedy requests per task, concurrency 1 (the plugin
asserts one request per batch, so concurrency 8 and 32 cannot be measured with it). Block 24 for the
trees. Chain lanes use the stock DFLASH algorithm on the same SGLang 0.5.22 without the plugin.

The release does not run as published. Two fixes were needed (`benchmarks/sglang/patch_dforktree.py`): the
scoring closure of `tree_ops.ngram_path_fn` is missing and was reconstructed from the paper, and
`verify_branch` needed a fallback for an empty host-side length field (one extra device-to-host copy
per round). These numbers are therefore "plugin + our fixes", not the authors' build.

Output tok/s over the whole run (accept = tokens per verification round):

| Lane | code | math | chat |
|---|---|---|---|
| chain 16 (0.5.22) | 1,262.4 (11.55) | 986.2 (7.14) | 406.2 (2.80) |
| chain 24 (0.5.22), run 1 / run 2 | 1,354.2 / 1,353.3 (12.85) | 952.9 / 960.5 (7.49) | 370.1 / 369.4 (2.81) |
| tree 32, eager tree builder | 1,167.1 (15.34) | 959.9 (9.88) | 414.7 (4.04) |
| tree 64, eager tree builder | 1,102.3 (15.66) | 947.4 (10.72) | 402.6 (4.32) |
| tree 32, compiled builder (`FC1_COMPILE_BUILD=1`) | **1,387.3** (15.34) | **1,188.9** (9.88) | **520.1** (4.04) |
| tree 64, compiled builder | 1,287.6 (15.66) | 1,144.6 (10.72) | 493.5 (4.32) |

| Tree 32 (compiled) versus | code | math | chat |
|---|---|---|---|
| best chain on the same SGLang 0.5.22 | 1.02x | 1.21x | 1.28x |
| best chain on any stack (SGLang 0.5.23: block 16 for code and math, block 8 for chat) | 1.03x | 1.12x | 1.06x |
| AR (L0(a)) | 5.22x | 4.35x | 1.89x |

Findings:

- The acceptance gain predicted offline is real in the serving stack: 15.34 / 9.88 / 4.04 tokens per
  round against the L0(c) prediction 14.77 / 9.86 / 4.28 for tree 32 + PMI at block 24.
- Speed follows only once the per-round overhead is removed. The tree builder alone costs 2.67 ms
  per round eagerly and 0.79 ms compiled (B200, measured standalone), against about 5.5 ms for a
  whole chain round. With the eager builder the tree is slower than the chain on code and math.
- 64 nodes is slower than 32 everywhere: the extra accepted tokens do not pay for the wider verify.
- First token is about 10 ms later than the chain (42 ms against 31-33 ms), which costs most on
  short code answers.
- Total output tokens differ between lanes on identical greedy requests (code: 11,066 chain 24,
  10,879 tree 32, also chain 16 against chain 24), so outputs are not token-identical across lanes.
  Losslessness must be shown by the GL1 token-identity test, not assumed.

Gate GL0 (>= 1.15x the fastest tuned chain at concurrency 1 on >= 2 of 3 families, and >= 1.0x at
concurrency 8 and 32): **not met as written.** Against the chain on the same engine version the
tree passes on math and chat; against the fastest chain measured on any stack (SGLang 0.5.23, block
8 on chat: 488.5 tok/s) it reaches 1.15x on no family; no confidence intervals yet (one repeat of chain 24 agreed within 0.8%); concurrency 8 and
32 are unmeasured because the plugin is single-request.

## L0(d) follow-up: per-round overhead removed, paired measurement (2026-11-15)

Root cause of the small margin above (profiled with `benchmarks/sglang/axprof.py`, GPU synchronised around
each phase): the tree target pass is cheaper than the chain-24 pass, but the plugin's per-round
bookkeeping (tree build, accept/commit, KV append and host round trips) ate the gain. The fix,
`benchmarks/sglang/dft_fast.py` (installed as `ax_fast.py`, active with `AX_FAST=1`, lane suffix `f`): no
device-to-host readback per round (the KV plan uses the scheduler's host-side length bound), one
Triton launch writes the verify mask from the device prefix length, and the snapshot kernel
launches only its live half. Correctness checks: `benchmarks/sglang/test_dft_fast.py` (B200: 90/90 random
trees and prefix lengths give the release mask bit for bit; buffers alternate; an absurd bound
raises) and an `AX_CHECK=1` serving run (15,400 rounds, 0 mask mismatches, 0 short bounds).

Protocol (`benchmarks/sglang/run_paired.sh`, fixed before the run): B200, greedy, thinking off, concurrency 1,
128 prompts per family (HumanEval / GSM8K / chat), up to 2,048 new tokens, 4 unrecorded warmup
requests, 3 repeats (GPU 0: repeats 1 and 3, GPU 1: repeat 2), each repeat a complete pass over all
lanes in an order rotated per repeat. Per-request wall time and token ids from
`benchmarks/sglang/token_identity.py`; statistics from `benchmarks/sglang/paired_ci.py`: throughput = sum(tokens) /
sum(time) per family, repeats averaged within a prompt, 95% interval from 10,000 paired bootstrap
resamples of prompts. "Best chain" = the fastest of chain 8/16/24 on SGLang 0.5.22 and chain 8n/16n
(stock DFLASH on SGLang 0.5.23), re-picked inside every resample. Lanes: tree32cf = tree 32,
compiled builder, overhead fix, block 24, plugin on SGLang 0.5.22.

| tok/s (3 repeats) | code | math | chat |
|---|---|---|---|
| AR (0.5.22) | 271.4 | 274.9 | 277.2 |
| chain 24 (0.5.22) | 1,354.9 | 915.8 | 364.1 |
| best chain | 1,452.9 (16n) | 1,057.6 (16n) | 493.3 (8n) |
| **tree32cf** | **1,682.4** | **1,329.5** | **574.2** |

| tree32cf against | code | math | chat |
|---|---|---|---|
| best chain (any stack), ratio [95% CI] | **1.158 [1.120, 1.191]** | **1.257 [1.203, 1.319]** | **1.164 [1.147, 1.187]** |
| chain 24, same engine | 1.242 [1.196, 1.298] | 1.452 [1.411, 1.508] | 1.577 [1.548, 1.609] |
| AR | 6.20x | 4.84x | 2.07x |
| output length vs best chain | 1.036 | 0.954 | 1.031 |
| ratio on prompts with identical output in every repeat of both lanes | 1.177 (112 prompts) | 1.249 (63) | 1.205 (5, too few to read) |

Causal ablation (GPU 1, same protocol, 2 repeats, all three lanes produce token-identical outputs on
128/128 prompts in every repeat, so only time differs):

| vs tree32c (release, compiled builder) | code | math | chat |
|---|---|---|---|
| tree32cf (overhead fix) | 1.123 [1.120, 1.128] | 1.123 [1.120, 1.127] | 1.113 [1.112, 1.115] |
| tree32cfs (fix, plus one forced GPU sync per round) | 1.014 [1.010, 1.018] | 1.020 [1.017, 1.022] | 1.019 [1.019, 1.019] |

Putting one host synchronisation per round back removes about 85-90% of the gain: synchronisation
explains most of it (the fix also removes a second copy and some launch work).

Token identity against AR (greedy, bf16; every lane reproduced its own output exactly in all 3
repeats, 128/128 per family):

| identical to AR / 128 | code | math | chat |
|---|---|---|---|
| chain 24 (0.5.22) | 112 | 59 | 9 |
| chain 16n (0.5.23) | 109 | 72 | 6 |
| tree32cf | 106 | 61 | 11 |

Divergence from AR is the normal bf16 behaviour of every speculative lane (a verify pass over many
tokens rounds differently from one-token decoding); the tree is within the chains' range. FP32
adjudication of the first differing token (`benchmarks/sglang/adjudicate.py`, Hugging Face Qwen3.5-9B in
fp32 on prompt + common prefix, repeat 2):

| AR vs lane, at the first differing token | prompts that differ | fp32 picks AR's token | fp32 picks the lane's | neither | largest regret of AR / of the lane (nats) |
|---|---|---|---|---|---|
| chain 24 | 204 / 384 | 89 | 112 | 3 | 0.277 / 0.216 |
| chain 16n | 197 / 384 | 87 | 110 | 0 | 0.277 / 0.259 |
| tree32cf | 206 / 384 | 103 | 99 | 4 | 0.277 / 0.259 |

Regret = fp32 log-prob of the fp32 argmax minus that of the chosen token. At every divergence of
every lane both tokens are within 0.28 nats of the fp32 winner, AR's own token included, and the
fp32 reference sides with AR or the tree about equally often: the tree's divergences are small
numerical flips of the same size as the chains'. This is consistent with a correct verifier but
does not prove one: it checks only the first differing token, of one repeat, greedy only, and the
reference is a full-prefix fp32 forward rather than the single-step reference GL1 specifies.
Strict token identity to AR (GL1 as written) is met by no speculative lane in bf16.

Measurement QA: three review rounds (codex, muse, kimi) on the scripts; every finding that could
bias a ratio or an identity count was fixed (strict prompt-set / hash / repeat validation, best
chain re-chosen per resample, identity over all repeats, aborted answers and retried requests
rejected, diagnostic runs excluded, completion sidecars, stale outputs invalidated before a rerun,
the patcher validates the whole install before writing). The runs above predate the completion
sidecar and prompt hashes, so they were analysed with `--legacy`; their prompt sets and repeat
counts were still checked. Artifact audit of the earlier single-pass runs: client token counts
equal the server's counters, and the server-side decode throughput moved by the same +13% as the
client-side figure.

Gate GL0, concurrency 1, greedy part (as written in the Track L decision: point estimate >= 1.15x
the fastest tuned chain on >= 2 of 3 families with a bootstrap CI lower bound > 1.0): **passes on
all three families** against the fastest tested chain (the lower bound also clears 1.15 on math;
code 1.120, chat 1.147; a lower bound >= 1.15 is the stricter GL1 requirement). Still open for GL0:
temperature 1, and concurrency 8 and 32 (the plugin asserts one request per batch). "Best chain" is
the fastest tested configuration (0.5.22 blocks 8/16/24, 0.5.23 blocks 8/16; 0.5.23 block 24 was
run later and is slower, see the review follow-up), and the cross-stack comparison mixes engine version with algorithm; the same-engine ratio
(vs chain 24) is the algorithmic one. The three repeats ran on two GPUs (repeat 2 on GPU 1); the
interval covers prompt sampling, not run-to-run variation. Scope of every number here: Qwen3.5-9B,
B200, greedy, batch 1.

Tree size after the fix (same protocol, 1 repeat, each GPU against its own tree32cf; the plugin
accepts only N = 16, 32, 64): tree16cf 0.895 / 0.910 / 0.954 and tree64cf 0.948 / 0.958 / 0.944 of
tree32cf (code / math / chat), all intervals below 1. Tree 32 stays the best budget on the 9B.

### Where a round's time goes after the fix (profile, 2026-11-15)

`benchmarks/sglang/axprof.py`, B200, 32 prompts per family, up to 1,024 tokens. With `AXPROF_SYNC=1` every
phase is bracketed by GPU synchronisations (GPU time of that phase, but host work is serialised);
with `AXPROF_SYNC=0` nothing is synchronised and "interval" is the real time between round starts.

| ms per round | tree32cf | chain 24 (0.5.22) |
|---|---|---|
| draft forward (synced) | 1.62 | 1.60 |
| target verify forward (synced) | 4.53 | 6.13 |
| sum of the two forwards | 6.15 | 7.73 |
| tree build / KV append / rest of verify (synced, host work included) | 0.70 / 0.62 / 2.80 | - |
| real interval between rounds (unsynced) | **6.85** | **7.58** |
| host time inside a round (unsynced) | 3.32 | 2.24 |

Unsynchronised, the host runs ahead of the GPU, so the round time is set by the GPU: the tree round
is 6.85 ms against 6.15 ms of model forwards, leaving about 0.7 ms (10%) of non-model GPU work and
bubbles; the chain round has none measurable. So further per-round engineering (fused build /
accept / KV append, whole-round CUDA graph) can gain at most about 10% at concurrency 1, less than
the synced profile suggested. Per request, the 23-26 ms prefill is shared by all lanes; on short
code answers (~170 tokens, ~11 rounds) it is about a quarter of the request time.

## 27B offline replay: DFlash2 + tree (`.internal/research/tree_headroom/dflash2_headroom.py`, 2026-11-15)

Qwen3.8-27B greedy answers (SGLang 0.5.23 AR, 64 prompts per family, up to 1,024 tokens, B200),
replayed through incoai/Qwen3.8-27B-DFlash2 (SpecForge `DFlash2DraftModel`: block 8, top-16
candidates, learned low-rank transition lattice). Tokens per round (accepted + the target's token);
a block of 8 caps every scheme at 8. Tree budgets in the plugin convention (N includes the root).

| tokens / round | code | math | chat |
|---|---|---|---|
| live DFlash2 server, earlier run (`results/track_a_baselines`) | 7.52 | 6.17 | 3.27 |
| replay: chain, unary top-1 | 7.43 | 5.96 | 3.06 |
| replay: chain, selector walk (what DFlash2 serves) | 7.49 | 6.20 | 3.24 |
| tree 16, unary / lattice | 7.59 / 7.66 | 6.57 / 6.95 | 3.81 / 4.02 |
| tree 32, unary / lattice | 7.66 / 7.74 | 6.84 / 7.22 | 4.08 / 4.40 |
| tree 64, unary / lattice | 7.70 / 7.76 | 6.96 / 7.37 | 4.30 / 4.70 |
| tree 32 lattice vs selector chain | 1.03x | 1.17x | 1.36x |

The replay reproduces the live server (7.49 / 6.20 / 3.24 against 7.52 / 6.17 / 3.27). Scoring the
tree with DFlash2's own learned transition (the lattice row of each node's parent) beats unary
scoring by 1-8 points, so the selector is a ready-made, trained replacement for DForkTree's n-gram
PMI. Code is saturated by the block of 8 (7.5 of 8), so a tree cannot help there; math and chat
have +17% and +36% more tokens per round at 32 nodes. Not a speed result: a 27B tree round verifies
32 tokens instead of 8, and no tree verifier for DFlash2 exists in the serving stack yet.

## 27B live: DFlash2 block length and lattice tree (`benchmarks/sglang/run_tree_plugin.sh`, 2026-11-15)

Qwen/Qwen3.8-27B with incoai/Qwen3.8-27B-DFlash2, one B200 per lane, bf16, greedy, concurrency 1,
64 prompts per family, up to 1,024 tokens. Ratios are paired per prompt against `chain8n` (DFlash2
as shipped: block 8, SGLang 0.5.23) with bootstrap 95% intervals (`benchmarks/sglang/paired_ci.py`).

### Block length alone (stock DFlash2 chain, 2 repeats, `results/track_l_27b_live`)

| lane | code | math | chat | round (ms) |
|---|---|---|---|---|
| chain8n (tok/s) | 503 | 458 | 251 | 13.8 |
| chain16n | 1.332 [1.298, 1.364] | 1.148 [1.122, 1.175] | 0.891 [0.877, 0.905] | 15.5 |
| chain24n | 1.219 | 0.985 | 0.786 | 17.3 |
| chain32n | 1.057 | 0.870 | 0.694 | 18.7 |

The drafter was trained with block 8 but drafts 16 positions usefully: code was capped by the
block (7.5 of 8 tokens per round), so block 16 is the best chain for code and math. Chat accepts
about 3 tokens per round, so a longer block only costs time there. Past 16 every family loses.

### A bug the 27B exposed in the released plugin

The first 27B tree runs accepted exactly 1.00 token per round and later emitted token id 0. Root
cause: the `snapshots` kernel of the plugin's fused Gated-DeltaNet verifier (`fc1_fused.py`)
computed the per-layer offset `l * OS0` in int32. The 9B state (24 layers x 32 heads) fits; the 27B
intermediate state (48 layers, 48 heads, 32 nodes, fp32) passes 2^31 elements from layer 43 on (22
at 64 nodes), the offset wraps, and the kernel writes out of bounds, into the KV cache among other
places (NaN keys in a full-attention layer). Fix in `benchmarks/sglang/patch_dforktree.py`: the layer index
and the slot / destination offsets are int64. `benchmarks/sglang/test_tree_gdn.py` (GPU) covers it: the
snapshot contract at 32 and 64 nodes with 27B shapes, the split of 16 to 48 value heads, and the
verify-to-commit seam; the unpatched kernel dies there with an illegal memory access. The plugin's
conv and tree kernels still index slots in int32, which is safe for one request and has to be
widened before batching.

### Lattice tree, final paired run (`results/track_l_27b_final`, 2 repeats on 2 GPUs, lane order rotated)

`tree<N>cfl_b16`: N-node tree built from DFlash2's own 16 x 16 transition lattice
(`benchmarks/sglang/dft_lattice.py`), drafter block 16, compiled builder, per-round overhead fixes, the
plugin's fused verifier on SGLang 0.5.22. `chain16` is the stock chain on that same 0.5.22 overlay.

| lane | code tok/s | math tok/s | chat tok/s | code vs chain8n | math vs chain8n | chat vs chain8n |
|---|---|---|---|---|---|---|
| chain8n (DFlash2 as shipped) | 500.4 | 456.3 | 250.9 | 1 | 1 | 1 |
| chain16n | 665.6 | 523.9 | 224.7 | 1.330 [1.296, 1.362] | 1.148 [1.122, 1.175] | 0.896 [0.883, 0.909] |
| chain16 (0.5.22) | 655.2 | 501.0 | 215.0 | 1.309 [1.272, 1.345] | 1.098 [1.063, 1.133] | 0.857 [0.840, 0.878] |
| **tree32cfl_b16** | **741.7** | **625.0** | **295.9** | **1.482 [1.457, 1.509]** | **1.370 [1.340, 1.398]** | **1.179 [1.164, 1.195]** |
| tree64cfl_b16 | 667.2 | 594.4 | 279.8 | 1.334 [1.295, 1.367] | 1.303 [1.274, 1.328] | 1.115 [1.100, 1.131] |

Against the fastest stock chain per family (re-chosen in every resample; chain16n for code and
math, chain8n for chat) the 32-node tree is 1.114 [1.098, 1.130] on code, 1.193 [1.164, 1.220] on
math and 1.179 [1.164, 1.195] on chat. Both trees beat DFlash2 as shipped on all three families; no
stock chain does (block 16 loses on chat). 64 nodes are slower than 32 everywhere: the extra accepted tokens do not pay for
the larger verify. Against plain decoding from an earlier, unpaired baseline run (about 101 / 102 /
103 tok/s) the tree is roughly 7.3x / 6.1x / 2.9x; treat that column as indicative.

Offline replay predicted the tree's gain over the chain at block 8 as 1.03x / 1.17x / 1.36x in
tokens per round; live, with a block-16 tree against the fastest chain of each family (block 16 for
code and math, block 8 for chat), the wall-clock gain is 1.11x /
1.19x / 1.18x. Chat keeps less than the replay suggested because a 32-node verify round costs more
than an 8-token one.

Every lane reproduced its own output exactly in both repeats (64 / 64 per family).

Token identity against plain decoding (AR, an earlier SGLang 0.5.23 run; repeat 2), and FP32
adjudication of the first differing token (`benchmarks/sglang/adjudicate.py`, Hugging Face Qwen3.8-27B in
fp32 on prompt + common prefix):

| lane | identical to AR, code / math / chat (of 64) | prompts that differ (of 192) | fp32 picks AR's token | fp32 picks the lane's | neither | largest regret of AR / of the lane (nats) |
|---|---|---|---|---|---|---|
| chain8n (DFlash2 as shipped) | 60 / 39 / 7 | 86 | 43 | 43 | 0 | 0.186 / 0.156 |
| chain16n | 59 / 36 / 6 | 91 | 45 | 46 | 0 | 0.186 / 0.183 |
| tree32cfl_b16 | 60 / 38 / 10 | 84 | 47 | 37 | 0 | 0.214 / 0.183 |
| tree64cfl_b16 | 58 / 32 / 7 | 95 | 43 | 52 | 0 | 0.186 / 0.155 |

As on the 9B, no speculative lane is token-identical to AR in bf16, the shipped DFlash2 chain
included, and the tree is inside the chains' range. At every first divergence of every lane both
tokens are within 0.22 nats of the fp32 winner and one of them is the fp32 winner; the reference
sides with AR and with the lane about equally often. The tree's first divergences are consistent
with numerical flips of the same size as the stock chain's; none is a token the fp32 reference
clearly rejects. The same limits apply as for the 9B: first differing token only, one repeat, greedy,
full-prefix fp32 reference.

The review follow-up below adds the per-token audit, the same-engine ratios and the block-8 trees.
Not measured: sampling, concurrency above 1 (the plugin's verifier takes one request), the
standalone plugin on SGLang 0.5.23, and other tree drafters (DDTree, Bastion, DSpine, EAGLE4).

## Review follow-up: budgets, baselines, verifier checks (2026-11-15)

Three reviewers (codex, muse, kimi) asked for the controls below before the concurrency-1 design
could be called settled. Same protocol as above (64 prompts per family, 1,024 tokens, greedy, two
repeats on two GPUs with rotated lane order, `results/track_l_27b_rv7`, `results/track_l_9b_rv7`).

### 27B: node budget and drafter block

| lane, vs chain8n | code | math | chat |
|---|---|---|---|
| tree32cfl_b16 (rerun) | 1.475 [1.448, 1.503] | 1.363 [1.333, 1.391] | 1.179 [1.165, 1.195] |
| tree16cfl_b16 | 1.425 [1.368, 1.474] | 1.305 [1.280, 1.331] | 1.135 [1.122, 1.148] |
| tree32cfl_b8 | 0.945 [0.938, 0.953] | 1.062 [1.043, 1.082] | 1.180 [1.168, 1.193] |
| tree16cfl_b8 | 0.976 [0.967, 0.985] | 1.061 [1.044, 1.079] | 1.130 [1.116, 1.143] |
| tree32cfln_b16 (SGLang's own tree verify) | 1.066 [1.027, 1.099] | 0.983 [0.964, 1.001] | 0.825 [0.814, 0.836] |

The rerun reproduces the final run (1.482 / 1.370 / 1.179). A block-8 tree is no faster on chat
than the block-16 tree (1.180 against 1.179) and much slower on code and math, so one setting, 32
nodes with block 16, is best or tied on every family; 16 nodes are 3 to 4 points behind. With
SGLang's own verifier the same tree is slower than the shipped chain on chat: the fused verifier is
what turns the tree's extra accepted tokens into wall-clock gain.

### 27B: same engine, output length, identical outputs (final run)

Tree and stock chain both on the SGLang 0.5.22 overlay (`chain16`), so engine version is not part
of the ratio. The comparison is block-matched (block 16 on both sides); on chat the fastest stock
chain is block 8, which this run did not include on 0.5.22, so the chat column here overstates the
gain over the best chain (1.18 against chain8n above):

| vs chain16 (0.5.22) | code | math | chat |
|---|---|---|---|
| tree32cfl_b16 | 1.132 [1.113, 1.151] | 1.248 [1.222, 1.274] | 1.376 [1.343, 1.404] |
| tree64cfl_b16 | 1.018 [1.004, 1.034] | 1.187 [1.162, 1.214] | 1.302 [1.270, 1.327] |
| tree32: output tokens, tree / chain | 1.009 | 0.996 | 0.989 |
| tree32: ratio on prompts with identical output in both lanes | 1.123 (60 prompts) | 1.227 (38) | 1.300 (7, too few to read) |

Against chain8n the tree's output-length ratio is 0.996 / 1.018 / 0.993 and the identical-output
ratio 1.493 (59) / 1.387 (39) / 1.239 (7). These controls support a timing gain on code and math;
aggregate length differences do not explain it. On chat only 7 prompts have identical output, too
few to say the same from that subset. Run-to-run: the two independent runs of tree32cfl_b16 against chain8n differ by 0.007 or less
per family. All intervals are over prompts, not over runs.

### 9B: the missing stock baselines, with full provenance

Rerun with completion sidecars and prompt hashes (no `--legacy`), adding block 24 on SGLang 0.5.23.
`chain8n` was added in a second pass on the same GPUs and repeats, after the other lanes rather
than interleaved with them.

| lane | code tok/s | math tok/s | chat tok/s | vs chain16n (code / math / chat) |
|---|---|---|---|---|
| chain8n | 1,070.7 | 972.6 | 490.6 | 0.738 / 0.895 / 1.099 |
| chain16n | 1,450.4 | 1,086.4 | 446.6 | 1 |
| chain24n | 1,407.7 | 1,021.7 | 400.5 | 0.971 / 0.940 / 0.897 |
| chain24 (0.5.22) | 1,361.3 | 961.2 | 370.4 | 0.939 / 0.885 / 0.829 |
| tree32cf | 1,673.5 | 1,373.0 | 586.7 | 1.154 / 1.264 / 1.314 |

Block 24 on the newer engine is slower than block 16 on every family. The fastest stock chain is
chain16n on code and math and chain8n on chat, as in the earlier run. Against the fastest of the
four stock lanes, re-chosen in every resample, the tree is **1.154 [1.131, 1.171] on code, 1.264
[1.227, 1.312] on math and 1.196 [1.182, 1.210] on chat** (earlier three-repeat run, 128 prompts
and 2,048 tokens: 1.158 / 1.257 / 1.164).

### Fused verifier against SGLang's own verifier, same tree

`tree32cfl_b16` and `tree32cfln_b16` build the same tree from the same drafter; only the verifier
differs. Their outputs are identical on 61 / 42 / 8 of 64 prompts (code / math / chat), the same
level as any two lanes here (tree against chain8n: 59 / 39 / 7), because two bf16 implementations
round differently and a near-tie then goes either way. Identity between the two therefore neither
proves nor refutes the fused verifier; the per-token audit below can expose output-level faults
that this comparison cannot.

### Per-token audit on the lane's own prefix (`benchmarks/sglang/prefix_audit.py`)

For every answer, the target is run once in fp32 (Hugging Face, no speculation) on prompt + the
lane's whole output, teacher-forced, and every emitted token is compared with the fp32 argmax on
that same prefix. Unlike the first-divergence adjudication this covers every round of every
request, through the tokens it emitted: a wrong acceptance, rollback, committed recurrent state or
cache would have to leave the emitted tokens plausible to go unseen. It does not observe the
served bf16 acceptance decisions or the internal state themselves.
Regret = fp32 log-prob of the argmax minus that of the emitted token.

| Qwen3.8-27B, 192 prompts | tokens | not the fp32 argmax | rate | largest regret (nats) | regret >= 0.1 / >= 0.25 / >= 0.5 |
|---|---|---|---|---|---|
| AR (plain decoding, control) | 88,048 | 527 | 0.60% | 0.257 | 63 / 1 / 0 |
| chain8n (DFlash2 as shipped, control) | 87,966 | 500 | 0.57% | 0.185 | 63 / 0 / 0 |
| tree32cfl_b16 | 88,048 | 533 | 0.61% | 0.259 | 64 / 1 / 0 |
| tree32cfln_b16 (SGLang's own verify) | 87,963 | 487 | 0.55% | 0.199 | 49 / 0 / 0 |
| tree32cfl_b8 | 87,649 | 529 | 0.60% | 0.235 | 57 / 0 / 0 |
| tree64cfl_b16 | 88,381 | 520 | 0.59% | 0.252 | 63 / 1 / 0 |

| Qwen3.5-9B, 192 prompts | tokens | not the fp32 argmax | rate | largest regret (nats) | regret >= 0.1 / >= 0.25 / >= 0.5 |
|---|---|---|---|---|---|
| AR (control) | 102,073 | 791 | 0.77% | 0.277 | 98 / 1 / 0 |
| chain16n (control) | 86,471 | 593 | 0.69% | 0.259 | 71 / 1 / 0 |
| chain24 (0.5.22, control) | 86,060 | 598 | 0.69% | 0.253 | 75 / 1 / 0 |
| tree32cf | 85,259 | 561 | 0.66% | 0.259 | 71 / 1 / 0 |

Plain bf16 decoding itself departs from the fp32 argmax on 0.6 to 0.8% of tokens, never by more
than 0.28 nats. The tree lanes sit at the same rate and the same largest regret as the controls on
both models, and no token of the about 437,000 emitted by tree lanes (352,000 on the 27B, 85,000 on
the 9B) has a regret of 0.5 nats or more. A wrong verifier decision, rollback or committed state
would be expected to produce tokens the fp32 target clearly rejects; none were observed. This is
empirical evidence, not a proof: a fault that happens to pick a near-tied token would not show, and
0.28 nats is the observed size of the bf16 discrepancy here, not an established tolerance. One
repeat of one GPU per lane was audited (repeat 2, GPU 1; every lane reproduced its output in both
repeats). What this does not cover: sampling, more than one request at a time, and
other prompt sets; the reference is a full-prefix fp32 forward, which is not bit-identical to
step-by-step decoding either.

The tree is not built chain-first: `run_tree_plugin.sh` leaves `FC1_CHAIN_FIRST` unset and the
lattice builder defaults to off, as the DForkTree paper recommends.

## Standalone plugin `ax-engine-sglang` on SGLang 0.5.22 and 0.5.23 (2026-11-15)

Everything above ran the DForkTree v0.1.0 plugin patched in place on an SGLang 0.5.22 overlay.
`ax-engine-sglang/` packages the same code as an installable plugin (`SGLANG_PLUGINS=ax-engine`)
that carries no copy of SGLang's DFlash worker: `worker_gen.py` builds the tree worker from the
installed SGLang at start-up (four text edits, each must match exactly once). The package's files
have the same AST as the patched plugin's, and for 0.5.22 the generated worker has the same AST as
the worker file of the DForkTree release.

Qwen3.8-27B, DFlash2 drafter, one B200 per run, greedy, one request at a time, 64 prompts per
family, 1,024 tokens. Lanes `axotree32cfl_b16` (plugin on the 0.5.22 overlay) and `axtree32cfl_b16`
(plugin on the venv's SGLang 0.5.23, no overlay); attention backend triton in both.

| Check | Result |
|---|---|
| 0.5.22, plugin vs. patched DForkTree, same GPU (`results/track_l_ax_parity`, 1 pass) | 192/192 outputs token-identical; 740.5 / 624.5 / 296.1 tok/s vs. 743.2 / 625.0 / 295.6 (code / math / chat) |
| 0.5.23, plugin vs. patched DForkTree on 0.5.22 | 192/192 outputs token-identical |
| 0.5.23 with `AX_CHECK=1` (mask and length bound checked every round), 24 requests | no check failed |

Speed on 0.5.23 (`results/track_l_ax_0523`, 2 passes on 2 GPUs, lane order rotated; ratio and 95%
interval from `paired_ci.py`, baseline the stock DFlash2 chain at block 8 on the same SGLang):

| Lane | code tok/s | math tok/s | chat tok/s | vs. chain8n (code / math / chat) |
|---|---|---|---|---|
| chain8n (DFlash2 as shipped) | 499.9 | 456.4 | 250.8 | 1 |
| chain16n | 675.0 | 525.1 | 224.6 | 1.350 / 1.150 / 0.896 |
| axtree32cfl_b16 | 744.3 | 625.0 | 296.0 | 1.489 [1.462, 1.516] / 1.369 [1.339, 1.398] / 1.180 [1.165, 1.196] |

Against the faster stock chain of each family (re-picked inside every bootstrap resample): code
1.103 [1.086, 1.119], math 1.190 [1.161, 1.218], chat 1.180 [1.165, 1.196]. These agree with the
final run of the patched plugin (1.482 / 1.370 / 1.179 and 1.114 / 1.193 / 1.179).

Not covered: more than one running request (the fused verifier still asserts one), sampling,
grammar-constrained decoding, and the prefix audit was not repeated because the outputs are
token-identical to the audited lane.

## L0(c) Offline tree headroom (`.internal/research/tree_headroom/tree_headroom.py`)

Replay of lossless decoding on saved greedy answers (one target pass per answer, one drafter pass
per answer position, schemes simulated against the true continuation). Tokens per round = accepted
tokens + the target's own token. Hardware independent; **not** a speed measurement (a tree round
also has a wider verify pass). Top-16 candidates per position; PMI = trigram model over 12.3M tokens
of the target's own answers to 30k training prompts (disjoint from the evaluation prompts), weight
0.3 on the top-1 spine and 0.6 after a fork. The replay's chain numbers match the live decoder on
the same answers (11.54 / 7.37 / 3.07).

| Block | Scheme | HumanEval (100 answers) | GSM8K (100) | chat (60) |
|---|---|---|---|---|
| 16 | chain | 11.62 | 7.39 | 3.07 |
| 16 | tree 16 | 12.17 | 8.24 | 3.83 |
| 16 | tree 16 + PMI | 12.12 | 8.57 | 3.92 |
| 16 | tree 32 | 12.67 | 8.85 | 4.13 |
| 16 | tree 32 + PMI | 12.82 | 9.41 | 4.28 |
| 16 | tree 64 | 12.94 | 9.27 | 4.37 |
| 16 | tree 64 + PMI | 13.20 | 9.91 | 4.55 |
| 24 | chain | 13.06 | 7.65 | 3.08 |
| 24 | tree 16 | 12.53 | 8.31 | 3.82 |
| 24 | tree 32 | 14.56 | 9.28 | 4.12 |
| 24 | tree 32 + PMI | 14.77 | 9.86 | 4.28 |
| 24 | tree 64 | 15.13 | 9.74 | 4.38 |
| 24 | tree 64 + PMI | 15.60 | 10.54 | 4.56 |

Budget convention: the table above was computed with "tree N" = N drafted nodes. The plugin's N
counts the root too (N-1 drafted nodes), so plugin tree 32 corresponds to "tree 31" here; the
replay now uses the plugin convention by default (`--budget-includes-root 1`) and builds the tree
independently of how many answer tokens remain. It is a global best-first (DDTree) replay, not the
plugin's per-depth beam, so it estimates headroom rather than replaying the served tree.

Against the best chain block (24): tree 32 + PMI gives +13% (code), +29% (math), +39% (chat); tree
64 + PMI +19% / +38% / +48%.

`.internal/research/tree_headroom/test_tree_headroom.py` (CPU): 5/5 checks pass (budget respected, unlimited budget equals
top-M recall depth, chain consistency, prefix-closed node sets of exactly the budget size, node set
independent of the true continuation).

## NCM span cost, for the record (`.internal/research/ncm_trunk/bench_span.py`)

Qwen3.5-9B, B200, batch 1, transformers reference kernels (not a serving stack), random block
inputs: 128 greedy AR tokens 3,302 ms (HF generate); one NCM span of 128 tokens (11 level passes +
commit + tokenizer decode) 499 ms = 6.6x; with the reverse scan of the recurrent layers 822 ms =
4.0x. Each level pass costs about one AR token step (~40 ms) regardless of its length. Speed only;
no quality is involved.
