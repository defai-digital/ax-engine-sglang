# Qwen3.8-Flash-Next: can the lossless tree path support it? (feasibility study, 2026-11-17)

Status: QSA ring fix done (section 8): MTP up to 13 tokens per round works on SGLang 0.5.23 with the
plugin (`AX_QSA_RING=16`); MTP 7 is +20-30% on code over stock MTP 3. Tree and DFlash2 remain blocked. SGLang 0.5.23 caps Flash-Next speculation at 4 tokens per round,
which rules out stock DFlash2 and any tree until QSA's pending ring is rewritten. Qwen3.8-27B work is complete (2026-11-16 decision
record, sections 1-17); this is the candidate next target.

Note: the multi-user plan so far was dense models only. Flash-Next is a MoE, so taking it on changes
that scope; the payoff analysis in section 4 is what decides it.

## 1. The model

| | Qwen3.8-27B (done) | Qwen3.8-Flash-Next |
|---|---|---|
| architecture | `Qwen3_5ForConditionalGeneration` | `Qwen4ExpForConditionalGeneration` (`qwen4_exp_text`), new |
| parameters | 27 B dense | 180 B total; MoE 512 experts, top 10 + shared expert (expert width 640); about 3-4 B active |
| layers | 64 (16 full attention, 48 Gated DeltaNet) | 48 (12 full attention, 36 Gated DeltaNet) |
| hidden | 5120 | 2560, with 4 hyper-connection streams (`hc_count 4`) |
| full attention | dense, 4 KV heads | QSA sparse attention: an indexer picks 2048 tokens per query from 4-token compressed blocks; 2 KV heads |
| GDN dims | 16 key / 48 value heads, 128 | the same |
| new parts | - | per-layer n-gram embedding (PLE: 3-gram hash table, about 54 B parameters stored in FP8), built-in MTP (NEXTN) layer |
| DFlash2 drafter | incoai/Qwen3.8-27B-DFlash2 | incoai/Qwen3.8-Flash-Next-DFlash2 (0.6 B, block 8, selector top 16, target layers 5/15/25/35/45, apache-2.0) |
| checkpoints | bf16, RadixArk NVFP4 | bf16 (360 GB), Qwen FP8 (186 GB), nvidia/Qwen3.8-Flash-Next-NVFP4 (about 125 GB) |

Only the NVFP4 checkpoint fits one B200 (bf16 and FP8 do not). The KV cache is small (12 attention
layers x 2 KV heads), so about 50 GB is left for KV and the drafter. SGLang can also keep the n-gram
table in pinned host memory (`Qwen4ExpPinnedHostEmbedding`), which would make FP8 fit; untested.

## 2. Engine support today (SGLang 0.5.23, TensorRT-LLM 1.4.0rc3)

- SGLang 0.5.23 ships `models/qwen4_exp.py`; `Qwen4ExpModel` inherits `Qwen3_5ForCausalLM`, so
  `set_dflash_layers_to_capture` (the DFlash2 hidden-state capture) is inherited. The built-in MTP
  layer is wired up as NEXTN (`speculative_hook.py`).
- **Stock DFlash2 chain (topk 1) should run as is.** PLE accepts target verify with topk 1.
- **Tree verify is refused**: `prepare PLE batch` raises `Qwen4 PLE speculative decoding supports only
  topk=1` whenever `spec_info.topk != 1`, and our plugin declares `speculative_eagle_topk=2`.
- TensorRT-LLM 1.4.0rc3 has `modeling_qwen4_exp.py` and handles its hyper-connection hidden size for
  MTP; DFlash2 on it is untested.

## 3. What the tree path needs (engineering)

| # | Part | Why it breaks with a tree | Size |
|---|---|---|---|
| a | PLE n-gram context | each token's embedding hashes the 2 previous tokens; SGLang builds them from the row as a straight line. In a tree a node's previous tokens are its parent and grandparent (`retrieve_parent`), and after verify the n-gram history must be committed from the accepted path, like the GDN state | medium |
| b | QSA sparse attention | the indexer compresses keys in 4-token groups (a pending-group ring) and derives verify positions from `seqlens_expanded`, i.e. a line. Tree siblings must never be compressed together or select each other; the tree nodes have to be attended densely under the tree mask, with sparse selection only over the committed prefix | large, kernel work; **main risk** |
| c | hyper-connections | 4 residual streams change the captured hidden shape and the fused verify layer code written for Qwen3.5 | small-medium |
| d | GDN tree state, chain rounds, selector sampling, flashinfer RNG fix | same GDN dimensions as 27B; reusable | small |
| e | MoE in the tree verify | NVFP4 MoE kernels with 64 x batch rows; works in principle, needs graph widths checked | small |

Losslessness checks reuse what we have (greedy identity against plain decoding, the 28 distribution
tests, kernel Monte-Carlo checks).

## 4. Will a tree pay off on this model?

On the dense 27B a wide tree was nearly free: verifying 64 nodes costs about the same as 8 because the
weights are read once. On a 512-expert MoE every extra node pulls in more experts. Expected distinct
experts per layer for n tokens x top 10: 512 x (1 - (1 - 10/512)^n).

| verify tokens (c = 1) | distinct experts / layer | share of expert weights read | NVFP4 bytes (about 60 GB of experts) | time at about 7 TB/s |
|---|---|---|---|---|
| 1 (plain decode) | 10 | 2% | 1.2 GB | 0.17 ms |
| 8 (chain) | 74 | 14% | 8.7 GB | 1.2 ms |
| 16 | 138 | 27% | 16 GB | 2.3 ms |
| 32 | 240 | 47% | 28 GB | 4.0 ms |
| 64 | 367 | 72% | 43 GB | 6.1 ms |

So verify cost grows almost linearly with tree width up to about 32 nodes, and a 64-node tree costs
about 5 x a chain round on expert weights alone. On the 27B the tree lifted tokens per round from
about 3.2 to 4.1 (+28%). The same gain at 5 x the verify cost loses at c = 1, so the useful tree is
narrow (12-16 nodes), where the gain is maybe +10-15%. At c = 16 the batch already touches most
experts, so tree width costs less there, but the 27B tree was already worth least at high
concurrency.

Expectation: the tree's gain over stock DFlash2 on Flash-Next is much smaller than on 27B, and parts
3a-3b are real work. Whether it is worth it depends on two measurements in F0.

## 5. Plan and gates

- **F0 smoke (about 2-3 GPU hours, no code):** on nvidia/Qwen3.8-Flash-Next-NVFP4 measure AR, built-in
  MTP and stock DFlash2 (chain) at c = 1/4/16 on code/math/chat (speed, tokens per round), plus one
  microbenchmark: target verify time for 8 / 16 / 32 / 64 tokens.
  - Gate 1: stock DFlash2 runs and beats MTP; otherwise Flash-Next is MTP territory and the tree is moot.
  - Gate 2: verify(16) / verify(8) <= 1.3 at c = 1. Above that a tree cannot pay off; stop at F0.
- **F1 tree port (about 1-2 weeks engineering, 20-40 GPU hours):** 3a, 3c, 3d, 3e first with QSA in a
  dense fallback for short contexts (assumption to confirm in F0: under the 2048-token budget QSA
  selects every block, so the tree can start with that case); then 3b. Lossless checks at each step.
- **F2 speed and comparison (about 10 GPU hours):** ours vs stock DFlash2 vs MTP vs TensorRT-LLM, the
  same benchmark matrix as 27B.

## 6. F0 result (2026-11-17, B200, nvidia/Qwen3.8-Flash-Next-NVFP4, SGLang 0.5.23, greedy, 64 requests, 1024 tokens)

Results: `results/f0` (AR, MTP 3 steps) and `results/f0b` (DFlash2 at 4 tokens). The model must be read
from /dev/shm: from /workspace (an S3 FUSE mount) loading ran at one shard per 6 minutes.

| tok/s (tokens per round) | c = 1 | c = 4 | c = 16 |
|---|---|---|---|
| AR code / math / chat | 222 / 227 / 229 | 658 / 705 / 718 | 1219 / 1650 / 1666 |
| MTP 3 code | 509 (3.86) | 1217 | 1959 |
| MTP 3 math | 496 (3.51) | 1290 | 2692 |
| MTP 3 chat | 380 (2.60) | 1033 | 2199 |
| stock DFlash2, 8 tokens | refused at start | | |
| stock DFlash2, 4 tokens | crashes at graph capture | | |

- **QSA caps every speculative method at 4 tokens per round.** With 8 draft tokens the server stops at
  verify graph capture: `Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4):
  the pending index-key ring holds one group; got 8` (`qwen_sparse_attn_backend.py`
  `_require_chain_speculation`). A tree of 12+ nodes is further out of reach than section 3b assumed:
  even a chain of 8 is refused.
- **DFlash2 is not wired for Qwen4Exp.** At 4 tokens the target forward fails in
  `qwen3_vl.py` (`hidden_states, aux_hidden_states = hidden_states`: too many values):
  `Qwen4ExpModel.forward` returns `(hidden_states, hc_hidden_states)` and never returns the captured
  layers. Fixable in the plugin (small), but pointless under the 4-token cap.
- **MTP 3 sits at the cap.** On code it accepts 3.86 of a possible 4 tokens per round, so a better
  drafter at the same width has almost nothing to gain there; on math (3.51) and chat (2.60) the room is
  0.5-1.4 tokens per round.
- AR per-request speed falls from 232 (c = 1) to 102-116 tok/s (c = 16): each extra request pulls in
  more experts, as estimated in section 4.

Gate 1 cannot pass without first rewriting QSA's pending index-key ring to hold more than one group
for verify rows (engine kernel work, needed by chain DFlash2, MTP 7 and any tree alike). Decision:
stop Flash-Next here. Reopen if SGLang lifts the 4-token cap (watch `_require_chain_speculation`), or as
a separate project whose first deliverable is that QSA change, measured with MTP 7 before any tree work.

## 7. Verify width inside the cap (F0r, F0w: RadixArk/Qwen3.8-Flash-Next-NVFP4, SGLang only)

The SGLang cookbook lists RadixArk/Qwen3.8-Flash-Next-NVFP4 for B200; F0 used nvidia's build. Rerun
(`results/f0r`, AR and MTP 1/2/3): within 2% of the nvidia build except code at c = 16 (AR 1463 vs
1219, MTP 3 2398 vs 1959 tok/s). vLLM was not used (SGLang only).

Round time = tokens per round / per-request decode rate (p50), so it includes the MTP draft steps:

| tokens verified per round | c = 1 | c = 4 | c = 16 |
|---|---|---|---|
| 1 (AR) | 4.3 ms | 5.4-5.6 ms | 8.6-9.6 ms |
| 2 (MTP 1) | 5.5 ms | 7.2-7.8 ms | 13.0-15.6 ms |
| 3 (MTP 2) | 6.1 ms | 8.7-9.9 ms | 15.4-19.6 ms |
| 4 (MTP 3) | 6.6 ms | 9.7-11.3 ms | 17.3-22.6 ms |

| tok/s at c = 1 (tokens per round) | AR | MTP 1 | MTP 2 | MTP 3 |
|---|---|---|---|---|
| code | 222 | 334 (1.99) | 432 (2.95) | 508 (3.87) |
| math | 227 | 338 (1.93) | 430 (2.77) | 493 (3.50) |
| chat | 229 | 315 (1.76) | 365 (2.28) | 380 (2.60) |

- At c = 1 each extra verified token (with its draft step) adds about 0.5 ms once speculation is on
  (5.5 -> 6.1 -> 6.6 ms): widening is cheap for one user, which the section 4 bandwidth estimate
  overstated (the c = 1 round is dominated by fixed costs, not expert reads).
- At c = 16 each extra token adds 2-4 ms: 16 requests x 4 tokens already pull in most experts, and the
  round grows 1.45 x from 2 to 4 tokens. Wide verification is expensive for many users.
- MTP 3 still gains on every set at c = 1, most on code (3.87 of 4 tokens accepted: the cap, not the
  drafter, is the limit there); on chat MTP 2 to MTP 3 adds only 4% (2.28 -> 2.60).

Reading for gate 2: inside the cap the c = 1 cost curve is flat enough (doubling 2 -> 4 tokens costs
1.2 x) that a wider round would likely pay for one user on code and math, if the cap were lifted. It
cannot be measured beyond 4 tokens in SGLang 0.5.23. The decision in section 6 stands: the next step on
Flash-Next is the SGLang QSA change (pending index-key ring of more than one group for verify rows),
measured with MTP 7 at c = 1; tree work only after that.

## 8. The QSA ring fix (`AX_QSA_RING`, ax-engine-sglang `ax_qsa.py`), 2026-11-17

**Stock SGLang 0.5.23 bug.** Each request keeps the index keys of its open 4-token group in a ring of
4 slots addressed by `position % 4`. A forward writes all its tokens' keys into the ring first and then
compresses the groups its rows complete, reading members back from the ring
(`QSAIndexer.project_qk` then `update_key_state_and_compress`). A verify window that starts at
position p and completes a group overwrites that group's earlier members whenever (p % 4) + W > 4, so
the compressed key is built from the wrong tokens. SGLang refuses W > 4 but W = 2..4 (MTP 1-3) hit it
too. Only the indexer's block selection is affected, so it shows once a context exceeds the 2048-token
budget.

**Fix.** `AX_QSA_RING=R` gives the ring R slots per request (`req * R + position % R`): pool
allocation, the two slot builders, and the CUDA-graph metadata Triton kernels (copied with a RING
constant); the verify-window cap becomes R - 3 (13 at R = 16). Opt-in; off leaves SGLang unchanged.

**Plugin bug found on the way.** `ax_chain`'s capture patches (written for the 27B tree) limited the
target CUDA graph of *every* server that loads the plugin to batch 1, so stock MTP with the plugin ran
eager at c > 1. They now apply only when the speculative algorithm is `FC1_DFLASH_TREE` (the 27B tree
path is unchanged).

**Ring consistency (q2, `AX_QSA_CHECK=1`, eager, 6000-token contexts, code + math, 24 requests x 256
tokens).** Groups compressed from ring members that do not hold the group's own positions:

| run | target pool | MTP draft pool |
|---|---|---|
| stock ring 4, MTP 1 | 7024 of 16000 (44%) | overwritten members seen (e.g. 124 replaced by 128) |
| stock ring 4, MTP 3 | 10636 of 16000 (66%) | overwritten members seen |
| ring 16, MTP 3 | 48 of 16000 | none |
| ring 16, MTP 7 | 72 of 20000 | none |

The 48 / 72 left are one pattern in all four runs ("group ending at 123 holds 64-67", 12 layers x 4 or
6 forwards per server), independent of the ring size and of the request count; open, probably a
startup or warmup forward rather than a verify. Greedy token identity against AR on the same long
contexts cannot tell the variants apart (stock MTP 3 19/24 code, 3/24 math identical; ring 16 MTP 3
17/24, 4/24; MTP 7 17/24, 0/24): batch-verify numerics dominate on this MoE.

**Speed (q3, RadixArk NVFP4, greedy, 64 requests, 1024 tokens; tok/s, tokens per round).** Stock MTP 3
from F0r; the others with the plugin and `AX_QSA_RING=16`:

| set, c | AR | MTP 3 stock | MTP 3 ring 16 | MTP 5 ring 16 | MTP 7 ring 16 |
|---|---|---|---|---|---|
| code 1 | 222 | 508 (3.87) | 509 | 596 (5.68), 1.17x | **662 (7.27), 1.30x** |
| code 4 | 658 | 1216 | 1218 | 1318, 1.08x | **1461, 1.20x** |
| code 16 | 1463 | 2398 | 2203 | 2671, 1.11x | **3027, 1.26x** |
| math 1 | 227 | 493 (3.50) | 494 | 543 (4.69), 1.10x | **567 (5.56), 1.15x** |
| math 4 | 705 | 1290 | 1294 | 1338, 1.04x | 1362, 1.06x |
| math 16 | 1644 | 2559 | 2558 | 2703, 1.06x | 2771, 1.08x |
| chat 1 | 229 | 380 (2.60) | 380 | 357 (2.94), 0.94x | 327 (3.04), 0.86x |
| chat 4 | 719 | 1021 | 1021 | 898, 0.88x | 820, 0.80x |
| chat 16 | 1654 | 2200 | 2173 | 2012, 0.91x | 1819, 0.83x |

(ratios against stock MTP 3). The wider ring costs nothing at MTP 3 (code c = 16 2203 vs 2398 is
within the run-to-run spread seen between F0 and F0r). Wider MTP pays where acceptance keeps growing:
code (7.27 tokens per round at MTP 7) +20-30% at every concurrency, math +6-15%; on chat acceptance
stops at about 3 tokens and MTP 5 / 7 lose 6-20%. The step count should follow the workload (or adapt
per request); this is the first measurement past the 4-token cap on this model.

Next steps on Flash-Next, in order of value: the ring bug is reported upstream
(sgl-project/sglang#51009, 2026-11-17); per-request adaptive MTP depth; DFlash2 needs the
Qwen4Exp aux-hidden-state capture (section 6) and then works within the 13-token window; a tree still
needs the PLE n-gram context and a tree-aware QSA selection (section 3).

**Wider still (q4, same setup, code and math only).** tok/s (tokens per round):

| set, c | MTP 3 stock | MTP 7 | MTP 9 | MTP 12 (13 tokens) |
|---|---|---|---|---|
| code 1 | 508 (3.87) | 662 (7.27) | 669 (8.74) | **683 (10.58)** |
| code 4 | 1216 | 1461 | 1499 | **1531** |
| code 16 | 2398 | **3027** | 2508 | 2335 |
| math 1 | 493 (3.50) | **567 (5.56)** | 546 (6.31) | 517 (7.00) |
| math 4 | 1290 | **1362** | 1326 | 1267 |
| math 16 | 2559 | 2771 | **2798** | 2457 |

MTP 12 needs `--max-running-requests 32`: the GDN Mamba state pool grows with the draft-token count
and at 13 tokens and 48 requests leaves no KV budget. Past MTP 7 the gain flattens at c <= 4 and turns
negative at c = 16 (the verify batch grows faster than acceptance). MTP 7 is the best single setting
here; a per-request depth (wide for code, narrow for chat) is where the rest is.

**Second pass (q5, stock MTP 3 without the plugin vs ring-16 MTP 7, same settings).** Every cell is
within 1.5% of the first pass, and the MTP 7 / MTP 3 ratios repeat to within 0.02: code 1.30 / 1.22 /
1.26x, math 1.15 / 1.05 / 1.08x, chat 0.86 / 0.80 / 0.83x at c = 1 / 4 / 16.
