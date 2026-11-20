# Multi-user serving for the tree decoder: load-adaptive verify width (draft, 2026-11-15)

Status: **Stage 0 approved (revision 5 confirmed by codex, kimi, muse, GLM 5.3, grok); revision 6 sets the 16-request test limit**. Numbers marked "measured" are from this repository's runs;
everything else is a projection and is labelled so.

## Goal

Qwen3.8-27B + DFlash2 on SGLang 0.5.23, one B200. With several requests running, be faster than
DFlash2 as shipped (chain, block 8) where the GPU has headroom, and not measurably slower elsewhere
(a measured gate, see Proposal; not a guarantee). Outputs stay the
target's own (same guarantee as at one request).

## What is measured

Stock DFlash2 chain, greedy, 64 prompts per family, 1,024 tokens, `results/track_l_mu0`
(1 pass, one GPU, `benchmarks/baselines/bench_speed.py`). Total output tok/s; "acc" = mean accepted tokens per
verify round.

| Family | c | block 8 total | block 16 total | block 16 / block 8 |
|---|---|---|---|---|
| code (acc 7.5 / 12.3) | 1 | 492 | 662 | 1.35 |
| | 4 | 1,351 | 1,672 | 1.24 |
| | 8 | 2,062 | 2,443 | 1.18 |
| | 32 | 3,473 | 3,222 | 0.93 |
| math (acc 6.2 / 8.2) | 1 | 454 | 523 | 1.15 |
| | 4 | 1,346 | 1,558 | 1.16 |
| | 8 | 2,179 | 2,172 | 1.00 |
| | 32 | 3,628 | 3,359 | 0.93 |
| chat (acc 3.3 / 3.3) | 1 | 250 | 225 | 0.90 |
| | 4 | 833 | 714 | 0.86 |
| | 8 | 1,378 | 1,106 | 0.80 |
| | 32 | 2,726 | 1,771 | 0.65 |

Round time implied by (per-request decode speed p50) / acc, code family: 8 rows 12.3 ms, 16 rows
13.5, 32 rows 16.7, 64 rows 20.6 to 22.3, 128 rows 29.2, 256 rows 38, 512 rows 58. So a verify round
costs about 12 ms plus 0.1 to 0.15 ms per row up to ~128 rows: rows are nearly free for one request
and become the bottleneck from about 100 rows.

At one request (measured, `results/track_l_ax_0523`): tree 32 nodes on drafter block 16 is 1.49 /
1.37 / 1.18 against block 8 and 1.10 / 1.19 / 1.18 against the faster chain.

## What this implies

1. The right number of verify rows per request depends on the load and on how predictable the text
   is. Block 8 is the best single setting only for chat and for 32 requests.
2. A 32-node tree spends 32 rows per request. At 4 requests that is 128 rows (about 29 ms a round
   against 16.7 ms for block 8). Projection from the one-request gains: the 32-node tree loses to a
   chain from about 4 requests; a 16-node tree may still win on chat and math at 2 to 4 requests
   (projected +10 to 15%, not measured).
3. Most of the multi-user gain available on code and math comes from the block length, not from
   the tree: +24% / +16% at 4 requests, +18% / 0% at 8 (measured, table above).

## Round profile: what drafting 16 costs when only 8 are verified (measured)

`results/track_l_mu1`: the stock chain under the round profiler (`benchmarks/sglang/axprof.py`, GPU
synchronised around every phase, so these are upper bounds per phase and not speeds), one server per
concurrency, all three families pooled. Mean ms per round.

| Block | c | drafter forward | target verify forward | whole round |
|---|---|---|---|---|
| 8 | 1 | 1.51 | 11.52 | 15.69 |
| 8 | 4 | 1.58 | 12.95 | 18.01 |
| 8 | 8 | 1.70 | 14.84 | 20.06 |
| 8 | 32 | 2.08 | 21.10 | 27.58 |
| 16 | 1 | 1.51 | 13.17 | 17.31 |
| 16 | 4 | 1.71 | 15.53 | 20.33 |
| 16 | 8 | 1.85 | 19.47 | 24.83 |
| 16 | 32 | 2.85 | 33.33 | 40.13 |

Drafting 16 positions instead of 8 adds 0.0 / 0.13 / 0.15 / 0.77 ms to the drafter forward at 1 / 4
/ 8 / 32 requests, i.e. under 1% of a round up to 8 requests and about 2.8% of a block-8 round at
32. The rest of a round outside the two forwards is the same for both blocks (2.6 to 4.4 ms). So
"draft 16, verify 8" should cost about what stock block 8 costs up to 8 requests and up to ~3% more
at 32; this is a projection from the profile and stage 0 measures it directly.

The verify column is the cost table the width rule needs, indexed by (requests, width). The earlier
revision derived one cost per row count; reviewers showed that this is wrong (the same number of
rows costs more with more requests) and that it would pick width 16 for code at 32 requests, where
block 8 is measured 7% faster.

## Memory: the recurrent-state scratch decides the tree size (measured)

SGLang keeps a per-step Gated-DeltaNet state scratch for every request slot, sized
`max_running_requests x speculative_num_draft_tokens` (`intermediate_ssm_state_cache`, from the
server logs): 0.14 GB per slot and step on Qwen3.8-27B.

| Server | Scratch | KV cache left (tokens) |
|---|---|---|
| stock block 8, 32 slots | 37.1 GB | 542,208 |
| stock block 16, 32 slots | 74.3 GB | 69,696 |
| tree 32 nodes, 1 slot | 9.0 GB | 955,663 |
| tree 32 nodes, 32 slots (this design as written) | ~148 GB | does not fit on one B200 |

So a 32-deep buffer cannot serve 32 requests. A one-request 16-node tree (16 deep) measured against
the 32-node tree, same GPU, one pass (`results/track_l_t16`), vs. stock block 8:

| Lane | code | math | chat |
|---|---|---|---|
| tree 32 | 1.480 [1.454, 1.507] | 1.370 [1.339, 1.398] | 1.180 [1.166, 1.196] |
| tree 16 | 1.426 [1.367, 1.476] | 1.308 [1.283, 1.334] | 1.134 [1.121, 1.147] |

Tree 16 still beats the fastest stock chain of each family (block 16 for code and math, block 8
for chat: 717 vs. 675, 598 vs. 525, 285 vs. 251 tok/s, unpaired across runs), at about 4% below
tree 32.

Decision: the multi-user server uses `speculative_num_draft_tokens = 16`: a 16-node tree at one
request, chain widths 8 and 16 above it. Its scratch and KV capacity are then exactly those of
stock block 16. The 32-node tree stays the one-request configuration (`--max-running-requests 1`).

Test limit: 16 concurrent requests (`--max-running-requests 16`, c = 1, 2, 4, 8, 16), set by the
project owner; c = 32 results above are kept as reference only, and a 64-request test may be added
later. Estimated from the two server logs above (0.14 GB per slot and step, about 80 GB shared by the
scratch and the KV cache after the weights; to be read from the server log at launch): 16 slots at
16 steps take about 38 GB of scratch and leave about 530k KV tokens; 16 slots at 32 steps (a 32-node
tree) about 77 GB and 40k tokens, so the 32-node tree is not used in the multi-user server.

Capacity is part of the comparison: against stock block 8 this server holds 70k instead of 542k
KV tokens at 32 slots, so long-context or high-concurrency deployments can hit the KV limit sooner
(the benchmark here, 1,024-token answers to short prompts, fits). Every multi-user result reports
the KV capacity next to the speed, and the slot count is an operator setting
(`--max-running-requests`): 16 slots halve the scratch to 37 GB. Reducing the scratch itself (for
example SGLang's ReplaySSM, which 0.5.23 rejects for DFlash on this model) is out of scope.

## Proposal

One server, drafter block fixed at 16, verify width chosen every round for the whole batch:

| Running requests | Verify | Rows per request |
|---|---|---|
| 1 | lattice tree, fused verifier (today's path), 16 nodes | 16 |
| 2 or more | stock chain verify on the first W drafted tokens, W in {8, 16} | W |

W counts the anchor: W rows verify the anchor and W - 1 drafted tokens.

Width rule. For each W, score(W) = sum over requests of E_i(W) / cost(b, W), where b is the number
of running requests and E_i(W) the expected tokens request i commits at width W.

- cost(b, W) is the round time implied by a measured stock sweep, not a synchronised phase profile:
  cost(b, W) = b x acc(b, W) / total_tok_s(b, W), with acc the mean accepted tokens per round.
  Reviewers showed that the synchronised profile understates the whole-round ratio at 32 requests
  (verify 33.33 / 21.10 = 1.58 against 1.74 implied by throughput), which made the rule pick 16 for
  code at 32 requests. With the throughput-implied cost the code case gives
  E(16)/E(8) = 12.21 / 7.56 = 1.62 against cost ratio (3473 / 7.56) / (3222 / 12.21) = 1.74, so the
  rule keeps 8, as measured. By construction the rule then reproduces the measured winner whenever
  acceptance is predicted correctly; the offline replay below tests exactly that. The table is
  measured for b in {1, 2, 4, 8, 16, 32} (`results/track_l_mu2`, running); between measured b the
  cost is interpolated linearly, and the replay checks the interpolated points too.
- E_i(W) = 1 + sum over j < W - 1 of the product of the acceptance rates of positions 0..j, from
  the request's decayed per-position counts (as ax-engine's `WindowStats`): a position counts as
  observed only if every earlier draft was accepted and the position was verified. An unobserved
  position takes the rate of the previous position times the population ratio rate(j)/rate(j-1)
  measured on the stock sweeps (not the previous rate unchanged, which overestimates deep
  positions and biases the rule to 16).
- Cold start: a request with fewer than 4 observed rounds is scored with the per-position rates of
  the stock sweeps of the family with the lowest acceptance (chat), so an unknown request is
  treated as hard to predict until its own counts say otherwise (a flat 0.85 prior, or a pool
  dominated by easy text, would send chat to 16).
- Switching margin 3%; every 64th round of a batch that has stayed at 8 verifies 16 once, so deep
  positions are observed again. Counts and a fixed table only, never wall time: the same batch
  composition and token history give the same widths.

Before it is used, the rule is replayed offline on the saved per-round acceptance of the stock
sweeps and must pick, for every (family, c), the block that the measurement shows faster (or be
within 2% of it).

What is claimed, and what is not. Target: at every tested (family, c) the adaptive server is not
slower than stock block 8 by more than the run-to-run interval, and faster where block 16 or the
tree is faster. It is a measured gate, not a guarantee: a batch that mixes chat and code shares one
width, so one request can be slower than it would be alone even when the batch total is higher;
per-request tail latency is reported next to totals.

Projected totals against DFlash2 as shipped (block 8), from the two measured stock sweeps:

| c | code | math | chat |
|---|---|---|---|
| 1 | +43% (measured, tree 16) | +31% (measured) | +13% (measured) |
| 4 | +24% | +16% | 0% |
| 8 | +18% | 0% | 0% |
| 32 | 0% to -3% | 0% to -3% | 0% to -3% |

Later, only if stage 1 holds up: a batched fused verifier so 2 to 4 requests can each verify a
small tree, and per-request widths (SGLang's ragged verify layout exists for DSpark only). Before
either, a one-request 16-node tree lane and the stage-1 cost table decide whether the margin is
real.

## Stage 0: pinned widths, correctness first (no speed claim)

Stage 0 adds a pinned mode (`AX_CHAIN_WIDTH=8|16`, `AX_TREE=0|1`) and no rule. It exists to answer
the questions the review raised, in this order, each a gate for the next:

1. Width/commit contract. In a chain round of width W, with drafter block B = 16:
   - Verify input: the first W columns of the drafted block; `draft_token_num = W`; positions
     `prefix + 0..W-1`; standard causal mask (no custom mask).
   - Server configuration, pinned: `speculative_num_draft_tokens = 16` (the tree's node count,
     set by the plugin from `FC1_TREE_N = 16`; see the memory section) and drafter block
     `B = FC1_DRAFT_BLOCK = 16`. Every per-request buffer that SGLang sizes from
     `speculative_num_draft_tokens` is therefore 16 deep, which covers the tree (16) and both chain
     widths (8, 16). A multi-request configuration with a different count is rejected at start-up.
     (The one-request 32-node tree server keeps 32 and `--max-running-requests 1`.)
   - KV (checked in the 0.5.23 source): `DFlashDraftInputV2.prepare_for_decode` keeps
     `reserve = 2 x speculative_num_draft_tokens = 32` page-aligned slots allocated ahead of each
     request's committed length (the over-allocation that overlap scheduling needs; the host-side
     length lags by one round). The unverified drafted positions W..B-1 lie inside that reserve:
     they are allocated, never freed during the request and released with the rest of the
     over-allocation when the request ends (`_release_overallocated_kv_indices`). Each round writes
     a new window at the new committed length into slots that are already mapped.
     `assign_extend_cache_locs_func` only gathers slot ids from `req_to_token`, it allocates nothing.
     Every post-draft use of `self.block_size` and the `speculative_num_draft_tokens` field of the
     `GenerationBatchResult` (the scheduler uses it as the output stride) take W in a chain round.
     The reserve equals stock block 16's (32 slots per request).
   - KV plan split. The stock worker plans one window of B slots for the draft, the verify and the
     drafter-cache write (`kv_index_translator.plan(write_virtual=..., read_extent=block_size)`).
     A chain round of width W < B plans the draft window as today (B slots) and plans a second,
     W-slot window for the verify, as `fc1_branch64.verify_branch` already re-plans n slots for the
     tree. Slots W..B-1 then hold drafter-written values past the commit; they are never read,
     because at most W tokens are committed and the next round's window starts at the new committed
     length. A check reads the committed prefix's KV after every round in the stress test and
     compares it with a fresh prefill of the committed text (tolerance as in the audit).
   - Accept: `_accept_block` receives W candidates and W logit rows per request; at most W tokens
     are committed (W - 1 drafts and the bonus token); `new_seq_lens = prefix + commit_len`.
   - Recurrent state: the chain verify writes its per-step Gated-DeltaNet and convolution
     snapshots at steps 0..W-1 of the 16-deep intermediate buffers, and the stock commit
     (`DFlashWorkerV2._update_target_mamba_state_after_verify`, the only commit path the DFlash
     worker uses in 0.5.23; ReplaySSM is off) copies step `commit_len - 1` of each request into its
     state slot. The tree's fused snapshot path (`fc1_fused.snapshots`) is never used by a chain
     round (see the seam below).
   - Drafter cache: the first `min(W, B)` target hidden rows are appended with the commit lengths.
   - Everything in the worker that reads `self.block_size` after the draft step is listed in the
     implementation and takes W instead (acceptance reshaping, logit adjustments, logprob stride,
     hidden-state view, result metadata).
2. Tree/chain seam (one request <-> several). Explicit flag per round (`round_kind = tree | chain`)
   instead of shape predicates: the fused Gated-DeltaNet, convolution and attention-plan hooks run
   only when the flag says tree (today they key on `cache_steps == N` and `x.shape[0] == 1`, which a
   chain of width N would also match); `fc1_plan.HOST_KV_LEN` is reset to None at the start of
   every chain round; the tree context (`CTX.anc`, keys, values) is never read by a chain round;
   `batch.seq_lens_cpu` and `batch.seq_lens_sum` are both restored on every exit path (`try/finally`), as
   the stock worker restores both around its verify.
   Tree-builder re-entry: the builder conditions on the token before the anchor (`fc1_pre1`), which
   today is updated only by tree rounds and reset only when the request id changes. When a request
   returns to tree mode after chain rounds, `fc1_pre1` is re-read from the request's committed
   output (the last committed token before the anchor). Checks: (a) after re-entry `fc1_pre1` equals
   that token; (b) the builder is deterministic in its inputs: fed the saved drafter outputs of a
   re-entry round, it rebuilds the same tree (nodes and order). A tree-only run on the same text is
   not a reference, because chain and tree rounds append different target hidden states to the
   drafter cache, so its drafts can differ while both are correct. The 32-bit slot indexing in `fc1_conv.py`
   and `tree_kernels.py` is widened to 64-bit now, although the tree stays at one request, because a
   request that ran in a batch can hold a high slot index when it is alone again.
3. Graphs. First with the verify forward eager for chain rounds (SGLang falls back to eager by
   itself when the width differs from the captured one), which is enough for every correctness
   check. Then CUDA graphs per (kind, width): the plugin's capture hook returns the tree verify
   input only for the tree runner and leaves SGLang's own for chain runners; one runner object per
   width, built after the main runner; a test replays tree, 8, 16, tree, 16, 8 for 200 rounds each
   after all graphs exist and compares with the eager outputs, to catch a runner that re-initialised
   shared attention buffers. Memory of the extra graphs and the remaining KV capacity are reported.
   Every (b, W) in the speed grid must report graph replay, not eager fallback.

Serving configuration for every Stage 0 check, pinned: SGLang 0.5.23, triton attention, overlap
scheduling on (the plugin registers `supports_overlap=True`), page size 1, radix cache off, CUDA
graphs off for the eager phase and on for the graph phase; the forced-switching checks run in both.
Sampling, grammar-constrained and logprob requests are rejected by the plugin with an explicit error
in multi-request mode until each is tested (a test asserts the error). The widened 64-bit slot
indexing gets a GPU unit test at 27B shapes (extending `benchmarks/sglang/test_tree_gdn.py` to the convolution
and tree kernels) before any batch runs.

Stage 0 checks (greedy, unconstrained):

| Check | Passes when |
|---|---|
| Pinned W = 16 at c = 1, 4, 8, 16 vs. the stock block-16 server (same drafts, same width) | outputs token-identical per request, up to the rate at which the stock server differs from itself across concurrency |
| Batch-composition invariance, pinned W | a prompt alone vs. with companions: same output, or a first divergence inside the fp32 regret threshold used in the report (0.5 nats) |
| Pinned W = 8 (drafted at 16) vs. stock block 8 | not required to be identical (the drafts can differ). Per-position acceptance on identical saved prefixes within noise of block 8's; fp32 prefix audit at the stock rate; tok/s measured at c = 1, 4, 8, 16 and reported with its interval. The profile projects up to ~3% slower at 32; if the measurement confirms that, the adaptive server at high load drafts at block 8 (a second drafter graph) or the claim is narrowed to "not slower up to 16 requests" |
| Forced switching 8 -> 16 -> 8 every k rounds, and tree -> chain -> tree by adding and removing a second request | per-round accept positions, bonus tokens and committed recurrent state equal to an eager reference run with the same forced schedule; prefix audit of the outputs including the regret tail and the tokens right after each switch |
| Independent accept oracle (forced switching and pinned runs) | for every round, the accepted length equals the length of the longest drafted prefix that matches the target's own greedy choice on the committed prefix, recomputed offline by a Hugging Face teacher-forced pass (bf16, the serving precision) over the dumped per-round drafts. Mismatches only where the bf16 reference and the server differ on a near-tie (regret under 0.5 nats, the cutoff of the prefix audit), at no more than the stock chain's rate in the same check |
| Stock self-divergence baseline | stock block 16 at c = 1 vs. c = 8 and c = 16 (same prompts): the identity rate measured first, because the pinned-W identity pass line is defined relative to it |
| Edge cases | rejection at every position, full acceptance, EOS and max-token cuts across a width change, unequal prefix lengths, request-slot reuse, cancellation |
| Allocator accounting | sampled right after each `prepare_for_decode`: `kv_allocated_len >= kv_committed_len + 32` (page size 1) for every running request, equal to the stock DFlash server's value in the same state; `kv_allocated_len` never decreases while a request runs (between prepares the gap shrinks by the published commit, which is expected); and pages and recurrent-state slots in use return to the start value after 2,000 requests with joins and leaves |
| Fused hooks | counters show zero fused-kernel calls in chain rounds and zero stock-kernel calls in tree rounds |

The raw numbers cited here (`results/track_l_mu0`, `track_l_mu1`, `track_l_mu2`) are copied as
`docs/decisions/data/2026-11-15-multiuser/*.json` so every number in this note can be checked.

## Stage 1: the rule

Only after stage 0 passes. Adds the width rule, the offline replay check of the rule, and the
paired speed run: adaptive vs. stock block 8 vs. stock block 16, interleaved, c = 1, 2, 4, 8, 16,
three families plus a mixed batch, at least 2 repeats, totals and per-request p50 / p10 speed.
The comparison at high concurrency also includes plain decoding (the Track L gate is the best of AR
and chain).

## Review record

Round 1 (2026-11-15; codex gpt-6.1-sol, kimi, muse, GLM 5.3, grok, agy). All agreed with the scope
(tree at one request, chain otherwise, one width per round). codex did not approve; the others
approved with required changes. Taken from the review: the rows-only cost table was wrong (GLM);
"never slower" was not supported and draft-16-verify-8 had to be measured (codex, kimi, grok, GLM);
the width/commit contract and the tree/chain seam were unspecified (codex, grok, agy); the capture
hook returned the tree input for every runner and the 32-bit slot indexing had to be widened (agy,
grok); the checks did not exercise batching or switching (all). This revision answers each.

Round 2 (codex, kimi, muse, grok; GLM and agy did not return a verdict). codex approved Stage 0;
kimi, muse and grok did not. Taken from the review: the KV description was wrong (the unverified
positions are inside the allocated 2x reserve, not unallocated; grok, muse); the buffer depth had to
be derived and pinned (`speculative_num_draft_tokens = 32`; muse, grok assumed 16 and would be right
for a server started without the plugin); the stock worker plans one B-slot window, so W < B needs a
second plan (muse); tree-builder re-entry state (muse); an oracle independent of the code under test
and a measured stock self-divergence baseline (muse, kimi); the synchronised profile was the wrong
cost source and still picked 16 for code at 32 requests (grok, muse); the flat prior and the
unchanged backfill biased the rule to 16 (grok, muse); the raw data was not in the repository
(muse); the stated round time 12.3 ms used per-request p50 speed (611 tok/s), which is the right
denominator for a per-request round time, so that sentence is kept (grok computed it from the total).

Round 3 (codex, kimi, muse, GLM 5.3, grok; agy out of quota). codex, muse and GLM approved Stage 0;
kimi and grok each named two items, all taken: `seq_lens_sum` must be restored with `seq_lens_cpu`
(kimi); the allocator gap is 64 only right after `prepare_for_decode`, so the gate is sampled there
(kimi, grok); the re-entry check must not compare with a tree-only run, whose drafter cache history
differs (grok); the goal sentence still said "never slower" (grok); `results/track_l_mu2` was cited
but not yet copied (grok; now copied); the cold-start pool was unspecified (grok); the oracle's
near-tie cutoff and the result's stride field (grok).

Revision 5 (after the Stage 0 approval): the server logs show the recurrent-state scratch is
sized slots x draft tokens (0.14 GB each); 32 slots x 32 steps does not fit. The multi-user server
therefore pins 16 draft tokens and a 16-node tree (measured 1.43 / 1.31 / 1.13 against stock block
8 at one request, 4% below the 32-node tree), with the scratch and KV capacity of stock block 16;
the projections at c = 1 and the Stage 0 depth, reserve and allocator numbers are updated to 16.

Revision 6: the test limit is 16 concurrent requests; the checks and the speed run use c = 1, 2, 4,
8, 16.

## Stage 0 implementation notes (2026-11-15)

Code: `ax-engine-sglang/ax_engine_sglang/ax_chain.py` and three worker edits in `worker_gen.py`.
Checks run on one B200, Qwen3.8-27B + DFlash2, SGLang 0.5.23, 16 prompts per family, 512 tokens,
greedy, outputs collected with 1, 4 and 16 requests in flight (`token_identity.py collect
--concurrency`).

- Reference: stock DFlash on the triton attention backend (the plugin's backend), eager, 16 slots.
  Two runs are token-identical at every concurrency (48/48 at c = 1, 4, 16): the stock server is
  deterministic under load, so identity against it is a valid test. A stock run on the default
  backend (trtllm_mha) is not a valid reference for the plugin.
- Found by that test: the plugin declares `speculative_eagle_topk = 2`, and SGLang's hybrid
  linear-attention backend reads `topk` at run time and takes its tree verify path for every target
  verify. Chain rounds therefore ran stock logic on other kernels and diverged from stock. Fix:
  `ax_chain` sets the backends' `topk` to 1 for a chain round and restores it for a tree round.
- After the fix, pinned width 16 with tree rounds off is token-identical to stock block 16 at
  c = 1, 4 and 16 (48/48 each).
- Pinned width 8 (block-16 draft, first 8 verified) is not comparable token for token with stock
  block 8: the DFlash2 drafter drafts the block jointly, so its first 8 tokens differ from an 8-token
  draft. Code is identical (16/16 at every concurrency); chat and math diverge after a median of
  50-220 tokens. First divergences are adjudicated in fp32 (`benchmarks/sglang/adjudicate.py`).
- The lattice tree builder reads the in-graph draft sampler, which SGLang builds only with CUDA
  graphs. The tree lane runs with `--cuda-graph-max-bs-decode 1`: graphs for one request (tree
  rounds), eager for several (chain rounds). The fused tree hooks set up their context during graph
  capture, so "no round decided yet" counts as a tree round.
- fp32 adjudication of every first divergence (stock on triton as lane a, `benchmarks/sglang/adjudicate.py`):
  width 8 vs. stock block 8 at c = 1 / 4 / 16: 17 / 21 / 22 prompts differ, ours picks the fp32
  winner 5 / 14 / 13 times and stock 12 / 7 / 9, never neither; largest regret ours 0.185, stock
  0.235 nats. Tree lane vs. stock block 16: 18 / 19 / 26 differ, ours 9 / 8 / 13, stock 9 / 11 / 13,
  never neither; largest regret ours 0.183, stock 0.152 nats. All within the 0.22-nat range seen at
  one request.
- The draft sampler of SGLang is sized for the largest graph batch; with graphs at one request only,
  the stock draft step still staged sampling parameters for the full batch (crash at c = 4). The
  staging is skipped for a batch larger than the sampler (that batch drafts eager).
- Independent accept oracle (`benchmarks/sglang/accept_oracle.py`: per dumped round, the accepted length and
  the bonus token recomputed from a teacher-forced Hugging Face bf16 pass over the request's
  committed text; 144 requests per lane, c = 1, 4, 16 combined):

  | lane | rounds | tokens checked | mismatches | share | over 0.5 nats | worst (nats) |
  |---|---|---|---|---|---|---|
  | pinned 16, tree off (token-identical to stock: the stock rate) | 8,914 | 49,181 | 214 | 0.44% | 0 | 0.250 |
  | pinned 8, tree off | 10,327 | 48,835 | 205 | 0.42% | 0 | 0.250 |
  | tree at one request, chain 16 otherwise | 8,508 | 48,915 | 181 | 0.37% | 0 | 0.250 |
  | forced switching 16 x 5 / 8 x 3, tree off | 9,272 | 48,962 | 205 | 0.42% | 0 | 0.250 |
  | forced switching, tree on | 8,730 | 49,079 | 191 | 0.39% | 0 | 0.250 |

  Every mismatch is a near-tie between the server and the reference (which runs the PyTorch
  reference Gated DeltaNet), at no more than the stock lane's rate.

- CUDA graph phase (`AX_CHAIN_GRAPH_WIDTHS="16 8"`, graphs up to 16 requests). The main runner
  captures the tree graph at one request; each chain width has a private runner and attention
  backend (topk 1, built like SGLang's UNO runner), capturing batch sizes 1-8, 10, 12, 14, 16 for
  0.65 GB (width 16) and 0.34 GB (width 8). A chain round swaps the target's attention backend and
  graph runner to its width's pair for the verify and the recurrent-state commit.
  - Stock block 16 with graphs differs from itself across two runs at c = 4 (chat 9/16 identical),
    not at c = 1 or 16: with graphs, which requests share a batch depends on arrival timing. It also
    differs from stock eager. The pinned width-16 graph lane against stock with graphs: identical
    at c = 1 (48/48), c = 16 47/48, c = 4 chat 6/16 (inside the stock run-to-run range).
  - Accept oracle on the five graph lanes (pinned 16, pinned 8, tree + chain 16, forced switching
    without and with the tree): 0.43 / 0.44 / 0.39 / 0.38 / 0.36% of about 49,000 tokens each,
    none over 0.5 nats, worst 0.25.
  - Allocation check on every round of every graph lane: the gap between allocated and committed
    length never fell below 32 slots, and no running request's allocation shrank. Hook counters:
    pure chain lanes consulted the fused hooks only in chain rounds, where they declined.

- Edge cases (`benchmarks/sglang/stress_multiuser.py`, graph configuration, tree + chain 16 and forced
  switching with the tree): 2,000 requests from 24 client threads into 16 slots, random
  max_new_tokens in {1, 2, 7, 16, 64, 256, 512}, 179 abort calls of which 50-55 hit a running
  request, prompts repeated (slot reuse). 0 errors; 24 reference prompts collected alone before and
  after the stress are token-identical (no state leaks); the recurrent-state pool returns to one
  slot of 16 (the probe request) and the KV usage to 0.00 (log resolution about 5,000 tokens); the
  allocation check held on about 38,000 samples per run.
- First speed look (one pass, graphs, 64 prompts per family, 1,024 tokens; stock lanes from the
  1-16 request baseline `track_l_mu3`), total tok/s relative to stock block 8:

  | c | code: tree + 16 / pinned 16 / pinned 8 | math | chat |
  |---|---|---|---|
  | 1 | 1.42 / 1.32 / 0.95 | 1.31 / 1.10 / 0.95 | 1.13 / 0.86 / 0.90 |
  | 4 | 1.25 / 1.27 / 0.96 | 1.07 / 1.04 / 0.93 | 0.82 / 0.82 / 0.88 |
  | 16 | 1.05 / 1.08 / 0.99 | 0.95 / 0.93 / 0.82 | 0.69 / 0.69 / 0.87 |

  Pinned 16 is within 1-2% of stock block 16 at every concurrency (the round machinery costs
  little). Pinned 8 accepts as many tokens as stock block 8 (for example 3.24 vs. 3.25 on chat at
  c = 16) but its rounds are 4-20% longer: it drafts the whole 16-token block to verify 8. As the
  design anticipated, width-8 rounds now draft at width 8 (`AX_CHAIN_DRAFT_WIDTHS="8"`: a drafter
  graph runner, sampler, spec info and draft buffers per drafting width, switched before the draft
  step since the width is known from the batch size); being tested.
- Same-backend comparison (`track_l_s0n`). The plugin lanes run triton attention; `track_l_mu3`
  stock ran the engine default, trtllm_mha, which is 3-15% faster than triton for stock block 8
  (most at 16 requests: chat 2,042 vs. 1,735 tok/s). Against stock on triton with the same graph
  settings, and with the allocation check off (it syncs every round), the plugin lanes match or
  beat stock (total tok/s; one pass):

  | | width 8 drafted at 8 / stock block 8 | pinned 16 / stock block 16 | best plugin lane / best stock |
  |---|---|---|---|
  | code c1 / c4 / c16 | 0.99 / 1.00 / 1.01 | 1.01 / 1.05 / 0.96 | 1.09 / 1.05 / 0.97 |
  | math c1 / c4 / c16 | 0.99 / 0.99 / 1.08 | 1.00 / 1.03 / 1.06 | 1.19 / 1.04 / 1.08 |
  | chat c1 / c4 / c16 | 0.98 / 0.99 / 1.03 | 1.00 / 1.03 / 1.02 | 1.22 / 0.99 / 1.04 |

  So the round machinery and width-8 drafting cost about nothing; the earlier "width 8 is 4-20%
  slower" was the backend and the check. What remains is the backend itself: at more than one
  request the plugin gives up the trtllm_mha speed. Next: whether chain rounds can run on
  trtllm_mha (or flashinfer for both kinds) inside the plugin (`track_l_s0o`).
- Attention backends inside the plugin (`track_l_s0o`, `track_l_s0p`). Chain rounds run correctly
  on trtllm_mha (oracle 0.45% mismatches, none over 0.5 nats) and on flashinfer (0.48%, none).
  Tree rounds on trtllm_mha are wrong (28 mismatches over 0.5 nats, worst 9.4, all in tree rounds:
  the backend ignores the tree mask); tree rounds on flashinfer crash in graph replay. The tree
  stays on triton. Paired with stock on trtllm_mha and the same graph settings (one pass):

  | | width 8 drafted at 8 / stock block 8 | pinned 16 / stock block 16 | best plugin / best stock |
  |---|---|---|---|
  | code c1 / c4 / c16 | 0.98 / 0.99 / 0.88 | 1.01 / 1.05 / 1.03 | 1.07* / 1.05 / 1.03 |
  | math c1 / c4 / c16 | 0.98 / 0.97 / 1.06 | 1.00 / 1.03 / 1.04 | 1.14* / 1.03 / 1.06 |
  | chat c1 / c4 / c16 | 0.98 / 0.98 / 1.01 | 1.00 / 1.02 / 1.04 | 1.13* / 0.98 / 1.01 |

  (* c = 1 best plugin lane is the tree on triton.) With the fast backend, the plugin's chain
  rounds equal stock at the same width, so above one request the plugin is at parity with the
  better of the two stock widths at each point, and ahead of any single fixed stock width across
  loads (chat c16: 2,063 vs. 1,454 for stock block 16; code c1: 701 vs. 493 for stock block 8).
  The one-request tree gain needs triton while the chain rounds want trtllm_mha: a server with
  both backends (triton for the tree round at one request, trtllm_mha otherwise) is the next
  design question, for review before Stage 1. A second pass is running (`track_l_s0p_r2`).
- NVFP4 plugin divergence (`track_n_nv5`): pinned 16 on eager triton, five runs, against four
  stock runs that are all identical. Three plugin runs equal stock at c = 4 and 16; two differ at
  c = 4 and equal each other. Their batch-size histograms differ from the other three (for example
  24 vs. 94 rounds at batch 2): requests joined and left at different rounds. NVFP4 GEMMs are not
  batch invariant (activation scales are static, so the token values of other requests do not
  matter, but the kernel choice per batch size does), so a different schedule moves near-ties.
  Not an acceptance error on this evidence; open: why the plugin's schedule varies between runs
  while stock's does not, and an NVFP4 oracle (teacher-forced through the SGLang server itself).
- Second bf16 pass (`track_l_s0p_r2`; stock in `track_l_s0p_r2_r2`): passes agree within 1-2% at
  most points. Mean of two passes, plugin / stock at the same width on trtllm_mha: width 8 drafted
  at 8 0.96-1.01 (about 2% behind at one to four requests: per-round Python), except code at
  c = 16, 0.87 in both passes (not seen on triton, 1.01; open); pinned 16 0.98-1.05.
- NVFP4 (`track_n_nv7`, `track_n_nv8`; 32 slots, graphs to batch 32, trtllm_mha, two passes; tree
  lane on triton, one pass, c <= 4): width 8 / stock block 8 0.96-1.04; pinned 16 / stock block 16
  0.97-1.13. Best plugin lane / best stock: c = 1 (tree) code 1.06, math 1.19, chat 1.13;
  c = 2-32 0.96-1.13. The tree on triton loses at c = 2 and 4 (its chain rounds are on triton),
  the same reason as for bf16.
