# Two attention backends in one multi-user server (design, for review)

Date: 2026-11-16. Status: implemented and measured (sections 10-17; approved in revision 3, sections 7-8). Parent design:
`docs/decisions/2026-11-15-multiuser-width-policy.md` (Stage 0 done; this note is the
precondition for Stage 1). Target: Qwen3.8-27B (bf16) and RadixArk/Qwen3.8-27B-NVFP4 with the
DFlash2 drafter on one B200, SGLang 0.5.23, ax-engine-sglang plugin.

## 1. Problem (measured)

The plugin runs two kinds of decode rounds: a tree round (16-node lattice, only when exactly one
request is running) and chain rounds (one chain per request, width 16 or 8). Stage 0 measured
(two passes, paired with stock on the same backend and graph settings):

| backend | tree rounds | chain rounds |
|---|---|---|
| triton | correct (oracle 0.36-0.39% near-tie mismatches, none over 0.5 nats) | correct, but triton is 3-15% slower than trtllm_mha for stock (most at 16 requests) |
| trtllm_mha | **wrong**: 28 mismatches over 0.5 nats, worst 9.4, all in tree rounds (no tree mask) | correct (0.45%, none over 0.5); plugin = stock at the same width (width 16: 0.98-1.05, width 8: 0.96-1.01) |
| flashinfer | crashes in graph replay | correct (0.48%) but slower than trtllm_mha |

So one backend for the whole server forces a choice: triton keeps the one-request tree gain
(bf16 c = 1: code 1.07, math 1.14, chat 1.13 vs. the best stock width on trtllm_mha; NVFP4: 1.06,
1.19, 1.13) but loses 3-15% above one request; trtllm_mha wins above one request but has no tree.
The NVFP4 tree lane on triton is behind stock at c = 2 and 4 for the same reason.

Goal: tree rounds on triton, every chain round on trtllm_mha, in one server, so that c = 1 keeps
the tree numbers and c >= 2 keeps the trtllm_mha chain numbers.

## 2. What SGLang 0.5.23 already provides (read from source, tag v0.5.23, 297e54f)

- `layers/attention/hybrid_attn_backend.py`, `HybridAttnBackend`: two full-attention backends
  (prefill, decode) on **one** KV pool (`model_runner.token_to_kv_pool`), routed by forward mode;
  TARGET_VERIFY goes to the decode backend when `--speculative-attention-mode decode`, else to the
  prefill backend. Flags `--prefill-attention-backend`, `--decode-attention-backend`. Marked
  experimental by SGLang.
- `model_executor/model_runner_components/attention_backend_setup.py`: when prefill and decode
  differ, it builds both full backends with `_build_full_attention_backend_from_str` and wraps the
  pair once with `attn_backend_wrapper` (the comment names Qwen3.5: one Gated-DeltaNet dispatcher,
  not two). `_build_backend_from_str(model_runner, backend_str, init_new_workspace)` builds a single
  wrapped backend from a name.
- KV layout: both triton and trtllm_mha address the same NHD pool. trtllm_mha views it as pages
  (`paged_kv_view`, `_reshape_paged_kv_cache`); with page_size 1 every slot is a page. An HND pool
  (`SGLANG_USE_HND_KVCACHE`) is a different layout and is not used here.
- Page size: `arg_groups/model_overrides/qwen3_5.py` sets page_size 64 only when the attention
  backend is chosen automatically (and picks trtllm_mha only for topk 1). With `--attention-backend`
  given explicitly, page_size stays at its default (1). `arg_groups/speculative_hook.py` rejects
  topk > 1 with page_size > 1 except on some backends. Every Stage 0 run (triton and trtllm_mha)
  passed the backend explicitly, so all ran page_size 1; stock with the automatic choice
  (page_size 64, `track_l_mu3`) and stock with explicit trtllm_mha (page_size 1, `track_l_s0p`)
  give the same speed (chat c = 16: 2,042 vs. 2,041 tok/s).

## 3. What the plugin already does

`ax_engine_sglang/ax_chain.py` (Stage 0):
- `_build_chain_runners`: per chain width W in `AX_CHAIN_GRAPH_WIDTHS`, a private
  `DecodeCudaGraphRunner` with its own attention backend, built under
  `get_spec().override(speculative_eagle_topk=1, speculative_num_draft_tokens=W)` with
  `mr._get_attention_backend(init_new_workspace=True)` (the main runner's backend string).
- `_use_chain_runner`: for a chain round, swaps `model_runner.attn_backend` and
  `decode_cuda_graph_runner` to the private pair; `begin_round` restores them. The swap also
  covers eager chain rounds (batch above the graph sizes), and
  `update_mamba_state_after_mtp_verify` then reads the private backend's linear-attention metadata
  (verified in Stage 0: identity, oracle, stress).
- `_build_draft_runners`: per drafting width a private drafter graph runner (draft backend).
- The main runner keeps the tree round (graph at batch 1) and prefill.

So the private chain backends already are separate objects; today they are simply built with the
same backend string as the main runner.

## 4. Proposal

D1. **Chain backends by name, built in the trtllm_mha role.** Build each private chain backend
with `_build_backend_from_str(model_runner=mr, backend_str=AX_CHAIN_ATTN, init_new_workspace=True)`
(default `trtllm_mha`) instead of `mr._get_attention_backend(...)`, inside the same
`get_spec().override(topk=1, num_draft_tokens=W)`. `_build_backend_from_str` returns the
`attn_backend_wrapper`-wrapped backend (`attention_backend_setup.py:264-274`), so each private
backend has its own Gated-DeltaNet dispatcher, as now. The trtllm_mha constructor decides its KV
access checks and native-FP4 setup from the runner's role strings
(`trtllm_mha_backend.py:218-240`: `model_runner.prefill_attention_backend_str`,
`decode_attention_backend_str`), and the GDN wrapper asserts on them
(`attention_registry.py:471-475`). So during construction both strings on `mr` are set to
`AX_CHAIN_ATTN` and restored afterwards (try/finally), which reproduces the construction of the
Stage 0 all-trtllm_mha server, where chain rounds were correct. The same two attributes are set
on the returned backend object (SGLang records them there for model dispatch,
`attention_backend_setup.py:158-160`); the implementation also greps the model code for readers
of these attributes and lists them.

D2. **Main runner on triton.** The server starts with `--attention-backend triton` (tree rounds
need its tree mask; the main runner's graph captures batch 1 only). Every chain round, at any
batch size, graph or eager, goes through a private trtllm_mha backend (D1). Consequence: with
`AX_CHAIN_GRAPH_WIDTHS` covering every chain width (16 and 8), no chain round runs on triton.
The plugin refuses to start if a chain width has no private runner while `AX_CHAIN_ATTN` differs
from the main backend (otherwise an unlisted width would silently run on triton). The mixed mode
needs CUDA graphs: `_build_chain_runners` builds no private backend when the main runner has no
graph runner (`ax_chain.py:228-229`), so with `--disable-cuda-graph` and `AX_CHAIN_ATTN` set the
plugin also refuses to start. ("Eager chain rounds" above means batches larger than the captured
sizes inside a graph-enabled server, which still use the private backend.)

D3. **Prefill.** Two options, to be measured, not assumed:
- D3a: prefill on triton (simplest; what the main runner does now).
- D3b: `--prefill-attention-backend trtllm_mha --decode-attention-backend triton
  --speculative-attention-mode decode`, so EXTEND runs on trtllm_mha and TARGET_VERIFY (tree) on
  triton, through SGLang's own `HybridAttnBackend`. The private chain backends are built by name
  (D1), so they are not affected by the split.
Pick by TTFT and total tok/s; D3a is the default if D3b shows no gain or any identity issue.

D4. **Drafter backend pinned.** The draft runner has its own backend
(`ModelRunner.draft_attention_backend`; the trtllm_mha runs logged "Overriding draft attention
backend to trtllm_mha"). It is set explicitly to trtllm_mha in every lane, mixed and stock, and
the server log line is checked, so paired lanes cannot differ in drafter backend.

D5. **page_size 1** for the whole server (both backends, explicit flags). No HND pool.

D6. **KV written by one backend is read by the other.** Prefill (triton or trtllm_mha) writes the
prompt KV; tree rounds (triton) and chain rounds (trtllm_mha) both write committed and draft KV
through the layer's `set_kv_buffer` path of their own backend, into the same pool, and read KV
written by the other. Requirements to check in code before implementation:
- both backends write K/V with the same layout and dtype into `token_to_kv_pool` (bf16; FP8 for
  NVFP4 with the same `k_scale`/`v_scale` handling);
- no backend keeps a private KV copy or cache of pages across rounds (trtllm_mha's
  `_fmha_v2_paged_kv` copies pages per forward; check that this is per-forward only);
- the Gated-DeltaNet (recurrent) state is backend-independent (one mamba pool; each wrapper
  holds per-forward metadata). This is asserted, not yet cited: the implementation cites the
  wrapper code and the KV handoff test (section 5, item 2) also covers the recurrent state, since a
  stale state shows up as a wrong committed token.
- source reading so far (review rounds 1-2): triton writes through the pool's `set_kv_buffer`
  with the layer's `k_scale`/`v_scale` (`triton_backend.py:1735-1743`); trtllm_mha's FP8 path does
  not call `set_kv_buffer` but writes the pool buffers directly with `fused_fp8_qkv_kv_cache`
  (`trtllm_mha_backend.py:1086-1098`), with the layer's scales or a ones tensor when the checkpoint
  has none (`:406-408`); both read with the same scales; trtllm_mha on SM100 reads a view of the pool (`:1457-1474`); the
  `_fmha_v2_paged_kv` copy is per forward and used on SM90/SM120 prefill only (`:367`,
  `:1438-1455`). No layout or dtype mismatch is visible; bitwise equivalence of FP8 writes is not
  proven by reading and is tested (section 5).

D7. **Memory.** trtllm_mha does not allocate a workspace per backend: `init_new_workspace` is
not used by it, and its workspace is one process-wide buffer (`get_buffer("trtllm_mha_zero_workspace")`,
`trtllm_mha_backend.py:310-317`) reused by every instance. That is safe because rounds run one
after another on one stream; the design relies on it and says so. The extra memory per width is
the private graphs, metadata buffers and the GDN wrapper's buffers (Stage 0 private triton
runners: 0.65 GB for width 16, 0.34 GB for width 8, drafter width 8: 0.27 GB). It comes out of
the KV reserve; report KV tokens before and after.

## 5. Validation plan (Stage 1a, before the width rule)

Correctness (Stage 0 gates plus the handoff checks asked for in review round 1):
1. Token identity, eager reference is not possible for the mixed server (no stock server has the
   same mix), so the gate is the accept oracle: bf16, 16 prompts per family, 512 tokens, c = 1, 4,
   16, plus the forced-switching lane (`AX_CHAIN_SCHEDULE`) with the tree. Pass: mismatches at the
   level of the Stage 0 lanes (under 0.5%), none over 0.5 nats.
2. Cross-backend KV handoff:
   - the round dump records, per round, the backend that ran it, graph or eager, and the width;
     the run fails if any chain round ran on the main (triton) backend;
   - requests whose rounds alternate tree and chain (arrivals that move the batch between 1 and 2
     requests) must pass the oracle;
   - committed KV vs. fresh prefill: after such a request finishes, its prompt plus committed
     text is prefilled again and the greedy choice at every committed position is compared with
     what the decode rounds committed; any difference beyond the oracle's near-tie rule fails.
     The reference must recompute the whole prefix: it runs on a separate stock server started
     with `--disable-radix-cache` (no reuse of KV or recurrent state from the mixed server), and
     its log must show `#cached-token: 0` for every reference prefill. This also exercises the
     recurrent state;
   - FP8 write-by-A / read-by-B probe (NVFP4, FP8 KV): write the same K/V through each backend's
     write path into scratch slots and compare the stored bytes; then read them through the other
     backend's attention for one query and compare with the same-backend result.
3. Stress test (`benchmarks/sglang/stress_multiuser.py`): 0 errors, reference outputs identical before and
   after, pools back.
4. Allocation check (`AX_CHAIN_CHECK=1`) on a correctness run only (it syncs every round).
5. NVFP4 mixed server: the same items 1-4 on RadixArk/Qwen3.8-27B-NVFP4. The accept oracle there
   needs an NVFP4 reference: teacher forcing through a stock SGLang server on the same checkpoint
   (`--disable-radix-cache`, prompt + committed text, `return_logprob` with the input-token
   log-probabilities and the top-k per position), so that at each position both the reference's
   best token and the committed token have a score and the 0.5-nat near-tie rule applies as for
   bf16. Building this oracle is part of Stage 1a and comes before the NVFP4 gate. Until it exists
   the interim gate, stated as weaker, is: (a) fresh-prefill check (item 2) on tree and chain
   rounds; (b) at c = 1 with the tree off (`AX_TREE=0`, chain rounds only) on both sides, the mixed
   server against the plugin's all-trtllm_mha server (the all-trtllm_mha tree lane is known wrong
   and is never a reference); (c) chain lanes at c >= 2 against the same all-trtllm_mha lanes,
   where differences are only counted, not gated, because of the schedule-dependent NVFP4
   numerics seen in Stage 0. Tree rounds on NVFP4 are gated by (a) alone until the oracle exists.
6. Each new private graph runner records its replays per (batch size, width); a correctness run
   must show replays for every captured size it used, so no width silently runs eager.
Not used as a gate: exact token identity of pinned 16 (no tree) against stock on trtllm_mha.
Stock on trtllm_mha with graphs is not deterministic run to run at c > 1 (Stage 0), so a
mismatch would not localise an error.

Speed (paired, two passes, 64 prompts per family, 1,024 tokens, c = 1, 2, 4, 8, 16; NVFP4 also 32).
Gates hold per family, width and concurrency, not on averages; each lane also reports TTFT
median and per-request p50 and p10 tok/s:
- c = 1: mixed server >= 0.98 x the tree lane on triton (`track_l_s0k` tree16; NVFP4 `track_n_nv8`).
- c >= 2: mixed server >= 0.98 x the trtllm_mha chain lanes at the same width (`track_l_s0p`,
  `track_l_s0p_r2`; NVFP4 `track_n_nv7`, `track_n_nv8`).
- TTFT median for D3a vs. D3b.
- The switching cost (a round after the batch moves between 1 and 2 requests) appears in the
  round profile, not only in averages.

## 6. Risks and open questions (for the reviewers)

R1. SGLang marks the hybrid backend experimental; D3b leans on it, D1/D2 do not (they only build
another backend object by name).
R2. trtllm_mha under `get_spec().override(topk=1)` while the process-wide spec says topk 2: does
its constructor or metadata path read topk or draft-token counts elsewhere (not through
`get_spec()` at call time)? Stage 0 ran it with the whole server on trtllm_mha (topk 2 declared)
and chain rounds were correct, but that was not the mixed case.
R3. FP8 KV (NVFP4): do triton and trtllm_mha apply the same scales when writing and reading?
R4. Memory per private backend: no extra workspace (shared, D7); the per-width cost is graphs,
metadata and wrapper buffers, which may differ between trtllm_mha and triton; measured (D7).
R5. Open Stage 0 items, not solved by this design: code at c = 16 with width 8 is 0.87 x stock
block 8 on trtllm_mha in both passes (1.01 on triton); width 8 is about 2% behind stock at one to
four requests (per-round Python); NVFP4 run-to-run schedule variation. The code c = 16 width-8
point is a hard failure of the speed gate: Stage 1 either explains and fixes it or the width rule
must not choose width 8 where it loses, and says so.
R6. Alternative not chosen: a tree mask inside trtllm_mha (a kernel change; out of scope), or
flashinfer for both kinds (its tree rounds crash in graph replay and its chain rounds are slower
than trtllm_mha).

## 7. Revision 2 (review round 1: codex, kimi, muse, GLM revise; grok proceed)

- D1: private chain backends are built with the runner's role strings set to trtllm_mha
  (trtllm_mha reads them for KV access checks and native FP4; the GDN wrapper asserts on them),
  and the strings are recorded on the backend object; `_build_backend_from_str` confirmed to wrap.
  (kimi, codex)
- D2: mixed mode requires CUDA graphs; graph-off refuses to start. (kimi)
- D4: drafter backend pinned, not only recorded. (muse)
- D6: FP8 read/write scale paths cited; GDN state independence marked as asserted and tested.
  (muse, all)
- D7 corrected: one process-wide trtllm_mha workspace, reused sequentially, not one per width.
  (kimi, grok)
- Section 5: per-round backend in the dump with a fail on chain-on-triton; committed-KV vs.
  fresh-prefill check; FP8 write/read probe; NVFP4 mixed-server lane with an SGLang-based NVFP4
  oracle; per-size replay counters; gates per family/width/concurrency with TTFT and tails; code
  c = 16 width 8 is a hard failure. (codex, kimi, GLM, muse)
- Not adopted: exact identity of pinned 16 against stock trtllm_mha (muse 4a), because stock on
  trtllm_mha with graphs is not deterministic at c > 1, so it cannot serve as a gate.

## 8. Revision 3 (review round 2: kimi, muse, GLM proceed; codex, grok revise)

- NVFP4 oracle specified with scores for both the reference-best and the committed token (needed
  for the near-tie rule), on a stock server without radix cache; built in Stage 1a. (codex)
- NVFP4 interim gate: c = 1 comparison with the tree off on both sides (the all-trtllm_mha tree is
  known wrong); c >= 2 differences counted, not gated; tree rounds gated by the fresh-prefill
  check only. (grok, codex)
- Fresh-prefill reference on a separate stock server with `--disable-radix-cache`, checked by
  `#cached-token: 0`. (codex)
- Wording: trtllm_mha's FP8 write is `fused_fp8_qkv_kv_cache` directly into the pool, not
  `set_kv_buffer`; R4 reconciled with D7. (codex, kimi, grok)

## 9. Implementation notes (code review round 1: codex, grok, kimi)

- The FP8 probe (`ax_fp8_probe.py`) compares the two write paths byte by byte on poisoned scratch
  bytes and fails when requested on a non-FP8 pool or with no layer found. The cross-backend read
  half of section 5 item 2 is not a separate probe: it is covered end to end by the teacher-forced
  fresh-prefill check (`benchmarks/sglang/sglang_teacher_check.py`), where every chain round that read
  tree-written KV and every tree round that read chain-written KV must commit the reference's greedy
  token. Calling either backend's attention kernel outside a forward pass would need its per-forward
  metadata built by hand, which is itself a source of error.
- The teacher check requests the span from the last prompt token (SGLang returns an unscored
  leading row), fails above a 0.5% near-tie mismatch rate, on any mismatch over 0.5 nats, and
  unless the reference log shows radix cache off, a non-speculative server and zero cached tokens.
- `benchmarks/sglang/run_stage1a.sh` fails (exit 1, no speed step) on any failed lane, oracle, teacher check,
  stress result or round-dump check (chain rounds on trtllm_mha from a graph, tree rounds on triton
  from a graph, requests crossing tree <-> chain present); NVFP4 runs 32 slots.

## 10. Stage 1a correctness results (single B200, SGLang 0.5.23, 2026-11-16)

Mixed server: tree rounds on triton (batch-1 graph), chain rounds on private trtllm_mha backends
(graphs at widths 16 and 8, drafter at width 8 with its own graph), drafter on trtllm_mha, 16 slots
(bf16) / 32 slots (NVFP4). `benchmarks/sglang/run_stage1a.sh correct`.

Bugs found on the first GPU runs, all fixed before the results below:
- Draft-at-width swapped four of the six draft buffers that SGLang sizes together; with aborts the
  host length buffer kept a smaller capacity and the draft step crashed (2 of 12 stress runs). All
  six are swapped now and every swap checks the capacities (commit 63d5261).
- The FP8 probe wrote slot 0, which SGLang's store kernel skips (padding slot) and trtllm's fused
  kernel writes; it now uses slot 1 (commit 27fce1e).
- The teacher check counted prefill log lines instead of prefilled requests (commit 5eaaaa4) and
  did not read 0.5.23's dict-style server_args.

bf16 (Qwen/Qwen3.8-27B): all gates pass.
- HF accept oracle (mtree): 0.43% near-ties, none over 0.5 nats.
- Teacher-forced fresh prefill on a stock AR server (radix cache off, 0 cached tokens on all 290
  prefilled requests): 288 outputs, 97,271 tokens, 0.28% near-ties, worst 0.375 nats, none over 0.5.
- Round dumps: every chain round on trtllm_mha from a graph, every tree round on triton from a
  graph, requests crossing tree <-> chain in both lanes.
- Stress: 4 of 4 passes after the buffer fix (2,000 requests each, aborts, EOS/max-token cuts,
  width changes, references identical, pools back).

NVFP4 (RadixArk/Qwen3.8-27B-NVFP4, FP8 KV): FP8 write probe 32/32 byte-identical; stress 2,000
requests, 0 errors. The teacher check does not meet the absolute bf16 gate, and neither does stock
DFlash2: decoding NVFP4 and prefilling the same text disagree on about 3% of tokens whatever the
decoder. Same check, same prompts (16 per set x 3 sets, 512 tokens, c = 1 / 4 / 16):

| decoder | mismatch rate | over 0.5 nats (share of tokens) |
|---|---|---|
| stock DFlash2 width 16 | 3.13 / 2.97 / 3.02% | 1.21 / 1.06 / 1.12% |
| stock DFlash2 width 8 | 2.94 / 3.20 / 3.32% | 1.06 / 1.08 / 1.30% |
| ax-engine mixed (mtree) | 3.08 / 3.05 / 3.05% | 1.04 / 1.16 / 1.22% |
| ax-engine mixed (msw) | 3.08 / 2.78 / 2.97% | 1.04 / 0.98 / 0.97% |
| ax-engine chain only (mtree0, c = 1) | 3.09% | 1.12% |

Worst regret: stock 6.9 nats, ax-engine 5.6. At c = 1 every ax-engine round is a triton tree round
with no backend switch and the rate is the same, so the 3% is NVFP4 prefill-versus-decode numerics,
not the tree or the dual backend. NVFP4 gate, as used from here on: mismatch and over-tie rates no
worse than stock DFlash2 under the same teacher check. Passed.

## 11. Stage 1a speed, bf16 (single B200, 2026-11-16)

Total output tok/s, mean of two passes each side, 64 requests per point, 1,024 max tokens, greedy.
Ours: mixed server (tree on triton at one request, chain rounds on trtllm_mha at width 8 or 16,
`run_stage1a.sh speed`); stock: SGLang DFlash2 at block 8 / 16 on trtllm_mha, same slots.
Pass-to-pass spread: ours within about 1% everywhere; stock within 0-1.4% except math c16 (13%).

| set | c | ours W8 | stock W8 | W8 ratio | ours W16 | stock W16 | W16 ratio | best ours / best stock |
|---|---|---|---|---|---|---|---|---|
| chat | 1 | 284 | 248 | 1.145 | 284 | 223 | 1.276 | 1.145 |
| chat | 2 | 453 | 465 | 0.975 | 410 | 409 | 1.003 | 0.975 |
| chat | 4 | 829 | 839 | 0.988 | 731 | 717 | 1.020 | 0.988 |
| chat | 8 | 1393 | 1395 | 0.999 | 1149 | 1102 | 1.043 | 0.999 |
| chat | 16 | 2031 | 2023 | 1.004 | 1511 | 1437 | 1.052 | 1.004 |
| code | 1 | 768 | 515 | 1.492 | 766 | 704 | 1.088 | 1.091 |
| code | 2 | 907 | 892 | 1.016 | 1174 | 1176 | 0.999 | 0.999 |
| code | 4 | 1518 | 1519 | 1.000 | 1933 | 1964 | 0.984 | 0.984 |
| code | 8 | 2143 | 2329 | 0.920 | 2828 | 2777 | 1.018 | 1.018 |
| code | 16 | 3262 | 3295 | 0.990 | 3328 | 3204 | 1.039 | 1.010 |
| math | 1 | 607 | 457 | 1.328 | 606 | 527 | 1.150 | 1.151 |
| math | 2 | 826 | 830 | 0.995 | 929 | 924 | 1.006 | 1.006 |
| math | 4 | 1402 | 1406 | 0.997 | 1534 | 1640 | 0.936 | 0.936 |
| math | 8 | 2238 | 2412 | 0.928 | 2264 | 2446 | 0.926 | 0.926 |
| math | 16 | 3073 | 3243 | 0.948 | 3189 | 2932 | 1.088 | 0.983 |

- One request: the triton tree is 9-15% faster than the faster stock block (1.09 code, 1.15 math,
  1.15 chat).
- Code c16 at width 8, 0.87 when chain rounds ran on triton (section 1), is 0.99 now.
- Below the 0.98 gate: code c8 W8 (0.920), math c8 at both widths (0.928 / 0.926), math c4 W16
  (0.936), chat c2 W8 (0.975). At math c8 and code c8 acceptance, per-request decode rate and TTFT
  equal stock; only wall time is longer (about 1 s on 8-12 s runs), which points at scheduling
  (fewer requests running at some point), not at the rounds. Per-request start/end times are now
  recorded by bench_speed.py to locate it.

## 12. Stage 1a speed, NVFP4 (single B200, 2026-11-16)

RadixArk/Qwen3.8-27B-NVFP4 (FP8 KV), 32 slots, same protocol as section 11, two passes each side.
NVFP4 outputs depend on the batch schedule (static activation scales), so the two sides generate
different text and points are noisier than bf16: pass 2 / pass 1 per point is 0.83-1.04 for ours at
width 8 and 0.95-1.03 for stock.

| set | c | ours W8 | stock W8 | W8 ratio | ours W16 | stock W16 | W16 ratio | best ours / best stock |
|---|---|---|---|---|---|---|---|---|
| chat | 1 | 455 | 402 | 1.133 | 455 | 341 | 1.337 | 1.133 |
| chat | 2 | 711 | 730 | 0.973 | 610 | 611 | 1.000 | 0.973 |
| chat | 4 | 1242 | 1294 | 0.959 | 1063 | 1035 | 1.027 | 0.959 |
| chat | 8 | 1986 | 2012 | 0.987 | 1604 | 1499 | 1.070 | 0.987 |
| chat | 16 | 2977 | 2968 | 1.003 | 2186 | 2026 | 1.079 | 1.003 |
| chat | 32 | 3836 | 3832 | 1.001 | 2484 | 2354 | 1.055 | 1.001 |
| code | 1 | 1129 | 811 | 1.393 | 1126 | 1025 | 1.099 | 1.102 |
| code | 2 | 1359 | 1348 | 1.008 | 1538 | 1640 | 0.938 | 0.938 |
| code | 4 | 2161 | 2148 | 1.006 | 2541 | 2592 | 0.980 | 0.980 |
| code | 8 | 3004 | 3173 | 0.947 | 3525 | 3068 | 1.149 | 1.111 |
| code | 16 | 4075 | 3086 | 1.320 | 4212 | 4103 | 1.026 | 1.026 |
| code | 32 | 5635 | 4529 | 1.244 | 5112 | 5637 | 0.907 | 1.000 |
| math | 1 | 964 | 728 | 1.325 | 962 | 777 | 1.237 | 1.241 |
| math | 2 | 1264 | 1268 | 0.997 | 1362 | 1319 | 1.032 | 1.032 |
| math | 4 | 2103 | 2108 | 0.998 | 2224 | 2123 | 1.048 | 1.048 |
| math | 8 | 3131 | 3097 | 1.011 | 3050 | 3224 | 0.946 | 0.971 |
| math | 16 | 4198 | 4659 | 0.901 | 3956 | 3561 | 1.111 | 0.901 |
| math | 32 | 6298 | 5472 | 1.151 | 4839 | 4600 | 1.052 | 1.151 |

- One request: 1.10 code, 1.24 math, 1.13 chat over the faster stock block.
- Several requests: best/best 0.90-1.15, most points within 0.96-1.05; the width-16 chain is ahead
  of stock width 16 at 13 of 15 points. Lowest: math c16 (0.90, ours W8 vs stock W8, where ours
  W16 is 1.11 against stock W16), code c2 (0.94), chat c4 (0.96). With the run-to-run spread above
  these need more passes before they count as losses.

### 11a. The c = 8 shortfall depends on run context (2026-11-16)

Rerun of c = 8 alone (fresh server, no other levels before it), width 8, bf16, with per-request
start/TTFT/end times (bench_speed.py `requests`):

| c = 8, width 8 | ours tok/s | stock tok/s | ratio | in the full sweeps (2 passes each) |
|---|---|---|---|---|
| math | 2,222 | 2,174 | 1.022 | 0.928 |
| code | 2,356 | 2,317 | 1.017 | 0.920 |
| chat | 1,393 | 1,386 | 1.005 | 0.999 |

Stock math c8 was 2,412 / 2,411 in the two full sweeps and 2,174 alone: what ran earlier on the same
server moves a point by up to about 10% on either side, consistently between passes of the same
sweep. Per-request timing shows no scheduling gap on our side: slots refill within 0.2-0.3 ms on
both, TTFT equal, slot occupancy 0.853 vs 0.838 (math) and 0.957 vs 0.944 (code). The full-sweep
shortfall at c = 8 is therefore not a slower round or a scheduling stall in the mixed server; it is a
history effect of the sweep. Speed gates should be judged on interleaved or isolated points with
several passes, not on one sweep order.

## 13. Drafter share of a round, and drafter-free methods (2026-11-16)

Question: could a drafter-free self-speculative method (Apple, arXiv 2507.11851: gated LoRA plus
mask tokens, the target drafts for itself) beat DFlash2 on the 27B? The most it can save is the
drafter's time. Stock DFlash2, bf16, B200, trtllm_mha, benchmarks/sglang/axprof.py with GPU syncs (each phase
an upper bound; tok/s of these runs is not a speed result):

| run | drafter forward | target verify | round interval | drafter share |
|---|---|---|---|---|
| c = 1, block 8 | 1.40 ms | 11.65 ms | 15.16 ms | 9.2% |
| c = 1, block 16 | 1.42 ms | 13.33 ms | 16.83 ms | 8.4% |
| c = 8, block 8 | 1.58 ms | 14.72 ms | 19.05 ms | 8.3% |
| c = 8, block 16 | 1.71 ms | 19.20 ms | 23.27 ms | 7.4% |

A drafter-free method must reach at least about 91% of DFlash2's acceptance length just to break
even. Accepted tokens per round, c = 1:

| | code | math | chat |
|---|---|---|---|
| Apple, Tulu3-8B, 8 masks (paper, Table 1) | 5.35 | 5.22 | 2.4-2.5 |
| DFlash2 on the 27B, block 8 | 7.52 | 6.17 | 3.27 |
| DFlash2 on the 27B, block 16 | 12.29 | 8.12 | 3.31 |
| ax-engine lattice tree, 16 nodes | 12.63 | 9.27 | 4.02 |

Different base models and prompts, so not a controlled comparison, but the gap (Apple at 55-80% of
DFlash2 block 8) is far below the 91% break-even, the quadratic variant verifies k^2 extra positions
(costly with several requests), and gated LoRA does not fuse into the weights. Not pursued.

### 11b. Reverse-order sweep (16 -> 1), bf16, one pass each side

| point | forward sweeps (2 passes) | reverse sweep | c = 8 alone | reading |
|---|---|---|---|---|
| code c8 W8 | 0.920 | 1.012 | 1.017 | sweep-history effect |
| math c16 W8 | 0.948 | 1.095 | - | sweep-history effect |
| math c8 W8 / W16 | 0.928 / 0.926 | 0.932 / 0.935 | 1.022 (W8) | ours steady at about 2,230 tok/s in every run; stock 2,174-2,413 |
| math c4 W16 | 0.936 | 0.937 | - | steady in both orders: a real loss of about 6% (1,534 vs 1,639) |
| other points, c >= 2 | 0.97-1.05 | 0.97-1.07 | | parity |

The one steady shortfall, math at c = 4 on the width-16 chain, is being profiled on both sides
(axprof now also attaches to ax-engine-sglang's worker).

### 11c. The math c = 4 width-16 shortfall is text divergence, not a slower round (2026-11-16)

- Round time, c = 4, width 16, all three sets (axprof): ours 18.90 ms, stock 18.73 ms (+0.9%).
- Acceptance is lower and identical in all three sweeps: math 8.00 vs 8.29, code 11.77-11.93 vs 12.28.
- Mixed server with the tree on vs off at c = 4: same output length (429.9 math), acceptance 7.94
  vs 7.92, tok/s within 1%. Round dumps: no request ever returned to chain rounds after a tree
  round (the tree only takes the last running request), so no tree <-> chain handoff is involved.
- The two servers generate different text: greedy bf16 near-ties resolve differently because our
  prefill and tree rounds run on triton and stock runs everything on trtllm_mha (mean math length
  435 vs 416 at this point). Both pass the token-by-token teacher check. Across points the acceptance
  ratio moves both ways (0.966 to 1.076 at c >= 1 on width 16), and this point sits at the low end.
- Next: put prefill on trtllm_mha too (SGLang hybrid attention: `--prefill-attention-backend
  trtllm_mha --decode-attention-backend triton --speculative-attention-mode decode`; tree verify
  stays on triton). Then chain rounds see the same text as stock and the comparison is like for like.

### 11d. Hybrid attention: prefill on trtllm_mha (2026-11-16, one pass, width 16)

`--prefill-attention-backend trtllm_mha --decode-attention-backend triton --speculative-attention-mode
decode`: prefill as stock, tree verify on triton (2,844 tree rounds, all triton with graphs), chain
rounds on trtllm_mha (5,788, graphs). Width-16 sweep, c = 1 2 4 8 16, against the best of three stock
passes:

| set | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|
| code | 1.050 | 0.998 | 1.009 | 1.027 | 1.038 |
| math | 1.152 | 1.040 | 1.012 | 1.039 | 1.031 |
| chat | 1.293 | 1.005 | 1.014 | 1.033 | 1.055 |

At math c4 and c8 the two servers now generate the same text (mean length 416.0 / 417.9 on both)
with the same acceptance (8.29 / 8.27). run_stage1a.sh uses hybrid attention by default (HYBRID=0
for all-triton). Teacher check of these outputs and two full passes at both widths follow.

### 11e. Hybrid attention, full bf16 speed (two passes ours, three passes stock)

Ours: run_stage1a.sh speed with hybrid attention, passes s1a_hyb and s1a_hyb_r2 (forward order).
Stock: chain8n / chain16n, s1a_stock_bf16, _r2 (forward) and _rev (reverse order). Means of the
passes. Teacher check of the hybrid outputs (c = 1, 4, 16; 144 outputs, 48,629 tokens, fresh
prefill): 0.33% mismatches, all within the 0.5 tie (worst 0.25), passed.

| set | c | W8 ours | W8 stock | W8 ratio | W16 ours | W16 stock | W16 ratio | best ours / best stock |
|---|---|---|---|---|---|---|---|---|
| code | 1 | 739 | 516 | 1.434 | 741 | 702 | 1.056 | 1.056 |
| code | 2 | 908 | 893 | 1.017 | 1184 | 1171 | 1.011 | 1.011 |
| code | 4 | 1524 | 1519 | 1.003 | 1979 | 1951 | 1.015 | 1.015 |
| code | 8 | 2371 | 2331 | 1.017 | 2837 | 2724 | 1.042 | 1.042 |
| code | 16 | 3321 | 3289 | 1.010 | 3303 | 3175 | 1.041 | 1.010 |
| math | 1 | 607 | 457 | 1.328 | 608 | 527 | 1.152 | 1.152 |
| math | 2 | 824 | 830 | 0.993 | 961 | 924 | 1.040 | 1.040 |
| math | 4 | 1418 | 1407 | 1.008 | 1659 | 1639 | 1.012 | 1.012 |
| math | 8 | 2440 | 2403 | 1.015 | 2537 | 2438 | 1.041 | 1.041 |
| math | 16 | 3110 | 3166 | 0.982 | 2948 | 2868 | 1.028 | 0.982 |
| chat | 1 | 288 | 248 | 1.160 | 288 | 223 | 1.293 | 1.160 |
| chat | 2 | 457 | 465 | 0.983 | 411 | 409 | 1.005 | 0.983 |
| chat | 4 | 826 | 838 | 0.985 | 724 | 717 | 1.010 | 0.985 |
| chat | 8 | 1396 | 1390 | 1.004 | 1146 | 1099 | 1.043 | 1.004 |
| chat | 16 | 2016 | 2020 | 0.998 | 1500 | 1437 | 1.044 | 0.998 |

Ours pass to pass: 0.973-1.032 (width 8), 0.950-1.000 (width 16).

- Width 16 is at or above stock width 16 at every point (1.005-1.293).
- Width 8 is 0.982-1.017 against stock width 8 at c >= 2, within the run-to-run spread.
- Best ours / best stock: one request 1.06 code, 1.15 math, 1.16 chat; several requests
  0.982-1.042. Every c >= 2 point that was below 0.98 in section 11 (code c8 0.920, math c8 0.928,
  math c4 0.936, chat c2 0.975) is now at or above 0.983.
- Bf16 Stage 1a speed gate (no point below 0.98 of the best stock block with paired passes): passed.

### 12a. NVFP4 with hybrid attention (two passes each side, 2026-11-16)

Correctness (run_stage1a.sh all, hybrid): identity lanes, stress (passed), FP8 probe (32 writes
byte-identical), round dumps (tree on triton 2,925 / chain on trtllm_mha 5,840, graphs, 2 handoffs).
Teacher: 3.10% mismatches, 1.07% over the tie, worst 4.75; stock NVFP4 control 2.9-3.3%,
1.06-1.30%, worst 6.9; passed under the NVFP4 gate now encoded in run_stage1a.sh (mismatches <= 3.5%,
over the tie <= 1.3%, worst <= 7.0).

Speed (ours s1a_hyb_nvfp4 + _r2; stock s1a_stock_nvfp4 + _r2):

| set | c | ours W8 | stock W8 | W8 ratio | ours W16 | stock W16 | W16 ratio | best ours / best stock |
|---|---|---|---|---|---|---|---|---|
| chat | 1 | 461 | 402 | 1.148 | 461 | 341 | 1.355 | 1.148 |
| chat | 2 | 708 | 730 | 0.969 | 607 | 611 | 0.993 | 0.969 |
| chat | 4 | 1253 | 1294 | 0.968 | 1053 | 1035 | 1.017 | 0.968 |
| chat | 8 | 2001 | 2012 | 0.994 | 1588 | 1499 | 1.059 | 0.994 |
| chat | 16 | 2988 | 2968 | 1.007 | 2162 | 2026 | 1.067 | 1.007 |
| chat | 32 | 3890 | 3832 | 1.015 | 2485 | 2354 | 1.056 | 1.015 |
| code | 1 | 1141 | 811 | 1.407 | 1151 | 1025 | 1.123 | 1.123 |
| code | 2 | 1347 | 1348 | 1.000 | 1649 | 1640 | 1.006 | 1.006 |
| code | 4 | 2154 | 2148 | 1.003 | 2588 | 2592 | 0.998 | 0.998 |
| code | 8 | 3160 | 3173 | 0.996 | 3445 | 3068 | 1.123 | 1.086 |
| code | 16 | 3880 | 3086 | 1.257 | 4669 | 4103 | 1.138 | 1.138 |
| code | 32 | 5415 | 4529 | 1.196 | 5514 | 5637 | 0.978 | 0.978 |
| math | 1 | 945 | 728 | 1.299 | 946 | 777 | 1.217 | 1.217 |
| math | 2 | 1234 | 1268 | 0.973 | 1354 | 1319 | 1.026 | 1.026 |
| math | 4 | 2117 | 2108 | 1.005 | 2185 | 2123 | 1.029 | 1.029 |
| math | 8 | 3179 | 3097 | 1.026 | 3412 | 3224 | 1.058 | 1.058 |
| math | 16 | 4416 | 4659 | 0.948 | 3892 | 3561 | 1.093 | 0.948 |
| math | 32 | 5470 | 5472 | 1.000 | 4800 | 4600 | 1.044 | 1.000 |

Ours pass to pass 0.97-1.08 (width 8), 0.85-1.03 (width 16).

- One request: 1.12 code, 1.22 math, 1.15 chat over the faster stock block.
- Several requests (c = 2-32): best/best 0.948-1.138, 14 of 15 points at or above 0.968, the exception being
  math c16 (0.948, where stock width 8 is the noisiest point: 4,116-4,712 between its passes).
  Against the all-triton mixed server (section 12): code c2 0.938 -> 1.006, math c16 0.901 ->
  0.948, chat c4 0.959 -> 0.968; code c32 1.000 -> 0.978.

## 14. NVFP4 with 64 slots (2026-11-16)

Qwen3.8's GDN layers keep an intermediate SSM state per slot and per draft position for
speculative verify. Stock chain 8 at 64 slots: `intermediate_ssm_state_cache size: 73.12GB`. Our
server reserves 16 positions for the 16-node tree (about 146 GB): it does not start at 64 slots, and
neither does stock chain 16. At 64 slots ours therefore runs an 8-node tree on SGLang's own tree
verify (lane axtree8cfln_b8; the plugin's fused GDN tree kernel has inverses for 16/32/64 nodes only)
plus the width-8 chain on trtllm_mha, hybrid attention; stock runs chain 8.

Correctness (c = 1, 16, 64; 576 outputs): stress passed, FP8 probe 32 writes byte-identical, chain
rounds with more than 16 requests on trtllm_mha with graphs (53,362), tree rounds on triton. Teacher:
ours 3.46% mismatches, 1.22% over the tie, worst 5.37; stock control at 64 slots under the same check
3.47%, 1.27%, worst 3.87. Within the NVFP4 gate.

Speed, two passes each (pass 2 stock first):

| set | c | ours p1 | ours p2 | stock p1 | stock p2 | ratio of means |
|---|---|---|---|---|---|---|
| code | 32 | 5385 | 5228 | 5340 | 5275 | 1.000 |
| code | 48 | 5769 | 5723 | 5672 | 5695 | 1.011 |
| code | 64 | 6209 | 7247 | 6511 | 6353 | 1.046 |
| math | 32 | 6132 | 5972 | 5816 | 5876 | 1.035 |
| math | 48 | 6620 | 6492 | 6567 | 6540 | 1.000 |
| math | 64 | 7450 | 7459 | 7366 | 7406 | 1.009 |
| chat | 32 | 3966 | 3894 | 3884 | 3879 | 1.013 |
| chat | 48 | 4211 | 4176 | 4172 | 4157 | 1.007 |
| chat | 64 | 4598 | 4537 | 4479 | 4484 | 1.019 |

Several requests at 64 slots: 1.000-1.046, parity or better at every point.

Open problem: one request at 64 slots loses the tree's advantage. Code c = 1: 700 tok/s (8-node tree,
native verify) vs 1,141 with 32 slots and the 16-node fused tree, and below stock chain 8 at 32
slots (811). The tree only ever serves one running request, yet the state cache is sized for every
slot at the largest draft width. Fix to design: size the intermediate state cache as 64 slots x 8
positions plus 16 positions for the slot(s) that run tree rounds (or fall back to 8 positions when
memory is short), so the 16-node fused tree stays available at 64 slots.

### 14a. AX_SSM_POSITIONS: the 16-node fused tree at 64 slots (2026-11-16)

ax-engine-sglang/ax_engine_sglang/ax_state_cap.py sizes the intermediate SSM cache for
AX_SSM_POSITIONS draft positions (8) instead of speculative_num_draft_tokens (16): MambaPool.__init__
(allocation) and KVCacheConfigurator._handle_max_mamba_cache (budget; the conv windows above the cap,
still allocated, are charged at a 5% conv share) are rebuilt from the installed source with exact-match
edits. Tree rounds no longer use the cache: fc1_fused's commit writes the accepted node's state straight
into the slot's persistent state with the same snapshots kernel (Out = temporal, position stride 0) and
rolls back only the conv windows with SGLang's fused_conv_window_scatter_with_mask, as SGLang's own
scatter does after its SSM copy. Start-up refuses the cap unless FC1_FUSED=1, AX_TREE=1, AX_CHAIN_WIDTH is
set and every chain width (graph widths, schedule, pin) is <= the cap: SGLang's own verify, chain rounds
at the tree's width and the tree-off main graph (captured as a 16-wide stock chain) write one position per
node. Reviews: codex found the tree-off capture path and the conv budget gap (both fixed); grok OK.

Server log at 64 slots: intermediate_ssm_state_cache 73.12 GB (as stock chain 8), conv windows 2.86 GB,
KV cache 591,680 tokens.

Correctness (NVFP4): one request, the same 48 prompts and 512 tokens as s1a_hyb_nvfp4/mtree (32 slots,
uncapped): 48 of 48 outputs token-identical. c = 1, 16, 64 with AX_CHAIN_CHECK: stress passed, FP8 probe 32
writes byte-identical, chain rounds on trtllm_mha with graphs (53,463 with more than 16 requests), tree
rounds on triton (12,996), 3 requests crossing tree <-> chain. Teacher: 3.49% mismatches, 1.24% over the
tie, worst 6.19; stock control at 64 slots 3.47%, 1.27%, worst 3.87.

Speed, 64 slots, one pass (stock: c = 1-16 one pass, c = 32-64 mean of two):

| set | c | capped, 16-node tree | 8-node tree | stock chain 8 | capped / stock |
|---|---|---|---|---|---|
| code | 1 | 1164 | 700 | 812 | 1.433 |
| code | 2 | 1399 | 1290 | 1359 | 1.029 |
| code | 4 | 2286 | 2105 | 2162 | 1.057 |
| code | 8 | 3307 | 3130 | 3194 | 1.035 |
| code | 16 | 4541 | 3705 | 3114 | 1.458 |
| code | 32 | 5566 | 5385 | 5307 | 1.049 |
| code | 48 | 6067 | 5769 | 5684 | 1.068 |
| code | 64 | 6438 | 6209 | 6432 | 1.001 |
| math | 1 | 973 | 641 | 727 | 1.338 |
| math | 2 | 1278 | 1227 | 1266 | 1.010 |
| math | 4 | 2239 | 2037 | 2109 | 1.062 |
| math | 8 | 3427 | 3076 | 3146 | 1.089 |
| math | 16 | 4976 | 4620 | 4604 | 1.081 |
| math | 32 | 6073 | 6132 | 5846 | 1.039 |
| math | 48 | 6888 | 6620 | 6554 | 1.051 |
| math | 64 | 7530 | 7450 | 7386 | 1.020 |
| chat | 1 | 458 | 362 | 402 | 1.139 |
| chat | 2 | 702 | 709 | 730 | 0.962 |
| chat | 4 | 1270 | 1263 | 1295 | 0.981 |
| chat | 8 | 2050 | 2008 | 2017 | 1.016 |
| chat | 16 | 3035 | 2971 | 2957 | 1.027 |
| chat | 32 | 3924 | 3966 | 3881 | 1.011 |
| chat | 48 | 4207 | 4211 | 4164 | 1.010 |
| chat | 64 | 4584 | 4598 | 4481 | 1.023 |

- One request: 1.43 code, 1.34 math, 1.14 chat over stock chain 8 at 64 slots (the 8-node config was
  0.86-0.91); same as the 32-slot 16-node server (code 1,164 vs 1,141).
- Several requests: 0.96-1.46, 20 of 21 points at or above 0.98 (chat c2 0.962). Second pass queued.

Two passes each side (stock c = 1-16 from s1a_nv64lo_stock + _r2, c = 32-64 from s1a_nv64_stock + _r2;
ours s1b_cap/speed + s1b_cap_r2):

| set | c | ours p1 | ours p2 | stock p1 | stock p2 | ratio of means |
|---|---|---|---|---|---|---|
| code | 1 | 1164 | 1161 | 812 | 820 | 1.424 |
| code | 2 | 1399 | 1402 | 1359 | 1404 | 1.014 |
| code | 4 | 2286 | 2304 | 2162 | 2339 | 1.020 |
| code | 8 | 3307 | 3295 | 3194 | 3328 | 1.012 |
| code | 16 | 4541 | 4265 | 3114 | 4142 | 1.214 |
| code | 32 | 5566 | 5597 | 5340 | 5275 | 1.052 |
| code | 48 | 6067 | 5938 | 5672 | 5695 | 1.056 |
| code | 64 | 6438 | 6785 | 6511 | 6353 | 1.028 |
| math | 1 | 973 | 970 | 727 | 738 | 1.326 |
| math | 2 | 1278 | 1287 | 1266 | 1309 | 0.996 |
| math | 4 | 2239 | 2252 | 2109 | 2284 | 1.022 |
| math | 8 | 3427 | 3503 | 3146 | 3421 | 1.055 |
| math | 16 | 4976 | 4949 | 4604 | 4736 | 1.063 |
| math | 32 | 6073 | 6207 | 5816 | 5876 | 1.050 |
| math | 48 | 6888 | 6730 | 6567 | 6540 | 1.039 |
| math | 64 | 7530 | 7555 | 7366 | 7406 | 1.021 |
| chat | 1 | 458 | 458 | 402 | 400 | 1.143 |
| chat | 2 | 702 | 702 | 730 | 722 | 0.968 |
| chat | 4 | 1270 | 1272 | 1295 | 1300 | 0.980 |
| chat | 8 | 2050 | 2048 | 2017 | 2042 | 1.009 |
| chat | 16 | 3035 | 3045 | 2957 | 3003 | 1.020 |
| chat | 32 | 3924 | 3967 | 3884 | 3879 | 1.017 |
| chat | 48 | 4207 | 4172 | 4172 | 4157 | 1.006 |
| chat | 64 | 4584 | 4561 | 4479 | 4484 | 1.020 |

- One request: 1.42 code, 1.33 math, 1.14 chat over stock chain 8 at the same 64 slots. Ours pass to
  pass within 0.5% at c = 1.
- Several requests: 0.968-1.214; 19 of 21 at or above 0.996. Below: chat c2 0.968 and chat c4 0.980,
  steady in both passes (chat c2 was also the lowest bf16 point, 0.983): a small fixed cost of our
  width-8 chain round where acceptance is lowest (chat 3.2). Profiled next.
- NVFP4 64-slot gate (multi-user no worse than stock within run-to-run spread): passed except chat c2.

### 14b. Chain rounds no longer build the DFlash2 lattice (2026-11-16)

Cause of the chat c = 2-4 shortfall in 14a. A round profile (axprof, chat c = 2, 64 slots) put the
target verify level with stock (7.25 vs 7.34 ms) and the drafter 0.32 ms slower (1.49 vs 1.16 ms). A
torch trace (benchmarks/sglang/trace_kernels.py, 100 scheduler steps each side) found the one extra kernel:
`_tiled_select`, 0.31 ms per round, launched from the width-8 drafter CUDA graph. That graph had
captured ax_lattice's sampler wrapper, which runs a full-vocabulary top-k with log-sum-exp partition
and builds the 16x16 lattice. Only the tree builder reads them; the chain round drafts the same
greedy path without them. New graph counters in axprof show every round replaying its graph
(draft w8 / w16, target w8 / w16), so eager fallback was not the cause.

Fix: the samplers built for narrower drafting widths (ax_chain._build_draft_runners) carry
`ax_chain_only = True` and ax_lattice calls the release sampler for them. The full-width sampler is
unchanged, so tree rounds and full-width chain rounds still compute the lattice. (A first version
keyed this on ax_chain.KIND; code review found KIND is set at verify time, so an eager drafter would
read the previous round's kind, and the DForkTree site has no ax_chain module. Replaced before
commit.)
The committed version (site s1d): ids 192 / 192 identical to the KIND version, drafter 1.177 ms per
round at chat c = 2, every round on a graph.

Checks (NVFP4, 64 slots, AX_SSM_POSITIONS=8):
- Drafter per round, chat c = 2: 1.494 -> 1.179 ms (stock 1.163). Target verify unchanged.
- One request: ids 192 / 192 token-identical to the run before the fix.
- Teacher (c = 1, 16, 64): mismatch 3.55%, over-tie 1.27%, worst regret 6.19. The mismatch rate is
  0.05 points above the 3.5% NVFP4 gate. Per file: c = 1 2233 (the same outputs scored 2230 in the
  s1b run, so the reference itself moves by a few tokens), c = 16 2346 (stock control 2282, s1b 2247),
  c = 64 2135 (stock 2124, s1b 2149). Stock 64-slot control: 3.47%. Recorded as at the gate's noise
  floor rather than a pass; the gate leaves no margin over stock at 64 slots and is due for a
  per-file comparison against a paired stock run.

Speed, two passes after the fix (s1c/speed, s1c_r2), against the same stock means as 14a:

| set | c | stock tok/s (mean of 2) | before fix | after fix p1 / p2 | before / stock | **after / stock** |
|---|---|---|---|---|---|---|
| code | 1 | 816 | 1162 | 1160 / 1160 | 1.424 | **1.421** |
| code | 2 | 1381 | 1401 | 1441 / 1443 | 1.014 | **1.044** |
| code | 4 | 2250 | 2295 | 2365 / 2358 | 1.020 | **1.049** |
| code | 8 | 3261 | 3301 | 3353 / 3432 | 1.012 | **1.040** |
| code | 16 | 3628 | 4403 | 4308 / 4451 | 1.214 | **1.207** |
| code | 32 | 5307 | 5582 | 5645 / 5685 | 1.052 | **1.067** |
| code | 48 | 5684 | 6003 | 6171 / 5999 | 1.056 | **1.071** |
| code | 64 | 6432 | 6611 | 6647 / 6479 | 1.028 | **1.020** |
| math | 1 | 733 | 972 | 971 / 971 | 1.326 | **1.325** |
| math | 2 | 1288 | 1283 | 1328 / 1329 | 0.996 | **1.032** |
| math | 4 | 2196 | 2246 | 2307 / 2312 | 1.022 | **1.052** |
| math | 8 | 3284 | 3465 | 3539 / 3561 | 1.055 | **1.081** |
| math | 16 | 4670 | 4963 | 4989 / 4989 | 1.063 | **1.068** |
| math | 32 | 5846 | 6140 | 6307 / 6174 | 1.050 | **1.067** |
| math | 48 | 6554 | 6809 | 6851 / 6857 | 1.039 | **1.046** |
| math | 64 | 7386 | 7542 | 7689 / 7713 | 1.021 | **1.043** |
| chat | 1 | 401 | 458 | 458 / 458 | 1.143 | **1.143** |
| chat | 2 | 726 | 702 | 731 / 732 | 0.968 | **1.008** |
| chat | 4 | 1298 | 1271 | 1319 / 1318 | 0.980 | **1.016** |
| chat | 8 | 2030 | 2049 | 2111 / 2113 | 1.009 | **1.041** |
| chat | 16 | 2980 | 3040 | 3125 / 3095 | 1.020 | **1.044** |
| chat | 32 | 3881 | 3946 | 4015 / 3982 | 1.017 | **1.030** |
| chat | 48 | 4164 | 4189 | 4299 / 4225 | 1.006 | **1.024** |
| chat | 64 | 4481 | 4572 | 4625 / 4664 | 1.020 | **1.036** |

- One request unchanged (1.14-1.42): tree rounds were never affected.
- Several requests: all 21 points above stock, 1.008-1.207 (before: 0.968-1.214). chat c2 0.968 ->
  1.008, chat c4 0.980 -> 1.016. Pass to pass within 3% (code c16, c48, c64 move the most).
- NVFP4 64-slot speed gate: passed everywhere.

Teacher noise at 64 slots (same NVFP4 teacher, c = 1 + 16 + 64 outputs, about 189k tokens each):

| outputs | scored | mismatch | over-tie (regret > 0.5) |
|---|---|---|---|
| stock control, generation 1 | 3 times | 3.470 / 3.485 / 3.475% | 2403 / 2367 / 2375 |
| stock control, generation 2 | once | 3.429% | 2415 |
| ours before the fix (s1b) | once | 3.49% | 2350 |
| ours after the fix (s1c) | 3 times | 3.550 / 3.504 / 3.542% | 2403 / 2405 / 2399 |

- Scoring the same outputs again moves the rate by up to 0.05 points; generating them again (batch
  composition at c = 16 and 64 differs run to run) moves it by about 0.05 points more.
- Ours sits about 0.07 points above stock (3.49-3.55 vs 3.43-3.48), all of it in near-ties: the
  over-tie count, mismatches where the committed token is worse than the best by more than 0.5 logit,
  is the same (2350-2405 vs 2367-2415). The extra mismatches are greedy flips between near-equal
  tokens, as expected from a different verify kernel order, not wrong tokens.
- Gate for NVFP4 at 64 slots from here: over-tie no higher than a paired stock run plus 0.1 points
  and worst regret under 7.0; the raw mismatch rate is reported but not gated (3.5% was set at 32
  slots and leaves no room above stock at 64). s1c passes: over-tie 1.27% vs stock 1.25-1.28%.

### 14c. Stage 1a speed with the lattice skip (2026-11-16)

The Stage 1a speed lanes (run_stage1a.sh speed: sd8 = tree + chain drafted at width 8, sp16 = tree +
chain 16) drafted chain rounds through the same width-8 drafter graph, so 12a / 11e paid the 0.3 ms
too. Rerun on site s1d, two passes each (s1d_nvfp4 + _r2, s1d_bf16 + _r2), against the same stock
two-pass means as before (s1a_stock_nvfp4 / s1a_stock_bf16 + _r2). "best" = faster of sd8 / sp16
against faster of stock chain 8 / chain 16; "before" = s1a_hyb_nvfp4 / s1a_hyb (+ _r2).

NVFP4, 32 slots:

| set | c | best-of-ours / best-of-stock, before | **after** | sd8 / stock chain 8, before | **after** |
|---|---|---|---|---|---|
| code | 1 | 1.123 | **1.120** | 1.407 | **1.414** |
| code | 2 | 1.006 | **1.003** | 1.000 | **1.033** |
| code | 4 | 0.998 | **0.990** | 1.003 | **1.020** |
| code | 8 | 1.086 | **1.052** | 0.996 | **1.014** |
| code | 16 | 1.138 | **1.143** | 1.257 | **1.316** |
| code | 32 | 1.009 | **0.991** | 1.196 | **1.233** |
| math | 1 | 1.217 | **1.216** | 1.299 | **1.297** |
| math | 2 | 1.026 | **1.028** | 0.973 | **1.013** |
| math | 4 | 1.029 | **1.043** | 1.005 | **1.029** |
| math | 8 | 1.058 | **1.055** | 1.026 | **1.047** |
| math | 16 | 0.948 | **0.949** | 0.948 | **0.949** |
| math | 32 | 1.000 | **1.011** | 1.000 | **1.011** |
| chat | 1 | 1.148 | **1.148** | 1.148 | **1.148** |
| chat | 2 | 0.969 | **1.005** | 0.969 | **1.005** |
| chat | 4 | 0.968 | **1.004** | 0.968 | **1.004** |
| chat | 8 | 0.994 | **1.033** | 0.994 | **1.033** |
| chat | 16 | 1.007 | **1.035** | 1.007 | **1.035** |
| chat | 32 | 1.015 | **1.036** | 1.015 | **1.036** |

bf16, 16 slots:

| set | c | best-of-ours / best-of-stock, before | **after** | sd8 / stock chain 8, before | **after** |
|---|---|---|---|---|---|
| code | 1 | 1.053 | **1.052** | 1.436 | **1.436** |
| code | 2 | 1.007 | **1.003** | 1.017 | **1.034** |
| code | 4 | 1.008 | **1.007** | 1.003 | **1.017** |
| code | 8 | 1.021 | **1.038** | 1.018 | **1.033** |
| code | 16 | 1.018 | **1.029** | 1.008 | **1.029** |
| math | 1 | 1.152 | **1.152** | 1.328 | **1.328** |
| math | 2 | 1.040 | **1.040** | 0.993 | **1.016** |
| math | 4 | 1.012 | **1.010** | 1.008 | **1.034** |
| math | 8 | 1.037 | **1.032** | 1.012 | **1.030** |
| math | 16 | 0.959 | **1.043** | 0.959 | **1.043** |
| chat | 1 | 1.160 | **1.160** | 1.160 | **1.160** |
| chat | 2 | 0.983 | **1.001** | 0.983 | **1.001** |
| chat | 4 | 0.985 | **1.009** | 0.985 | **1.009** |
| chat | 8 | 1.001 | **1.019** | 1.001 | **1.019** |
| chat | 16 | 0.996 | **1.012** | 0.996 | **1.012** |

- Width-8 lane against stock chain 8, several requests: NVFP4 1.004-1.316 except math c16 0.949;
  bf16 1.001-1.043, all above stock. chat c2-c4 moved from 0.97-0.985 to 1.001-1.011 on both.
- One request unchanged (tree rounds).
- Math c16 is the one point left below stock (NVFP4 0.948 before and 0.949 after; bf16 0.981 and
  1.105 in the two passes, so noise there). Profiled next at NVFP4.

### 14d. Paired per-prompt speed: the large multi-user ratios were tail effects (2026-11-16)

Aggregate tok/s compares two servers fairly only if both write the same text. They do not: NVFP4
greedy near-ties resolve differently under different kernels (14b teacher noise), so at c = 16 math
37 of 64 prompts have a different output length on the two sides. In a small batch one extra answer
that runs to max_tokens lengthens the tail: math c16 NVFP4, ours 3 answers at 1,024 tokens against
stock's 2, last request finished at 6.60 s vs 5.93 s, aggregate 0.932, while per-request decode on the
27 identical-length prompts was 370.5 vs 366.9 tok/s (ours 1% faster). The same effect, the other way
round, made code c16 / c32 look like 1.2-1.33.

benchmarks/sglang/paired_speed.py pairs two runs prompt by prompt (same prompt index, same output length as a
proxy for the same text) and reports the decode-time-weighted ratio on those prompts. The bf16 stock
runs predate the per-request records, so this covers NVFP4.

Same-length prompts, decode-time-weighted ratio, ours / stock chain 8 (stock pass 2):

| set | c | 32 slots p1 | 32 slots p2 | 64 slots p1 | 64 slots p2 |
|---|---|---|---|---|---|
| code | 1 | | | 1.553 | 1.553 |
| code | 2 | 1.016 | 1.030 | 1.029 | 1.032 |
| code | 4 | 0.989 | 1.010 | 1.003 | 0.990 |
| code | 8 | 1.012 | 1.045 | 1.035 | 1.033 |
| code | 16 | 0.995 | 1.030 | 1.021 | 1.025 |
| code | 32 | 1.020 | 1.044 | 1.045 | 1.031 |
| code | 48 | | | 1.004 | 1.004 |
| code | 64 | | | 1.034 | 1.033 |
| math | 1 | | | 1.318 | 1.319 |
| math | 2 | 1.034 | 1.024 | 1.025 | 1.025 |
| math | 4 | 1.045 | 1.026 | 1.001 | 1.003 |
| math | 8 | 1.028 | 1.042 | 1.060 | 1.066 |
| math | 16 | 1.017 | 1.025 | 1.026 | 1.045 |
| math | 32 | 1.024 | 1.022 | 1.039 | 1.034 |
| math | 48 | | | 1.033 | 1.037 |
| math | 64 | | | 1.040 | 1.036 |
| chat | 1 | | | 1.151 | 1.151 |
| chat | 2 | 1.018 | 1.014 | 1.018 | 1.018 |
| chat | 4 | 1.014 | 1.013 | 1.015 | 1.019 |
| chat | 8 | 1.028 | 1.043 | 1.034 | 1.028 |
| chat | 16 | 1.044 | 1.043 | 1.041 | 1.037 |
| chat | 32 | 1.033 | 1.033 | 1.034 | 1.028 |
| chat | 48 | | | 1.033 | 1.025 |
| chat | 64 | | | 1.039 | 1.040 |

(32 slots: s1d_nvfp4 / _r2 sd8 against s1a_stock_nvfp4_r2 chain8n; 64 slots: s1c / s1c_r2 against
s1a_nv64lo_stock_r2 and s1a_nv64_stock_r2. 32-slot c = 1 not paired: its stock pass lacks records.)

- Several requests, NVFP4: ours is 0.99-1.07 of stock on identical text, mostly 1.01-1.04. Math c16
  is not a loss (1.017-1.045). Code c16 is not a 1.2x win (0.995-1.030). The honest multi-user claim
  is "no worse than stock, typically 2-4% faster".
- One request: code 1.55, math 1.32, chat 1.15 (64 slots), the tree's gain.
- From here speed tables report the paired ratio next to the aggregate; aggregate-only claims at
  c > 1 in 11e-14c are superseded by this section where they differ.

bf16, 16 slots, paired (s1d_bf16 / _r2 sd8 against fresh stock chain 8 passes with per-request
records, stock_bf16_pr_p1 / _p2), same-length prompts, decode-time-weighted ratio:

| set | c | aggregate p1 / p2 | paired p1 / p2 |
|---|---|---|---|
| code | 1 | 1.434 / 1.445 | 1.645 / 1.643 |
| code | 2 | 1.037 / 1.029 | 1.038 / 1.037 |
| code | 4 | 1.021 / 1.012 | 1.011 / 1.004 |
| code | 8 | 1.029 / 1.021 | 1.026 / 1.021 |
| code | 16 | 1.000 / 0.998 | 1.029 / 1.025 |
| math | 1 | 1.329 / 1.330 | 1.345 / 1.345 |
| math | 2 | 1.016 / 1.017 | 1.020 / 1.021 |
| math | 4 | 1.050 / 1.019 | 1.012 / 1.017 |
| math | 8 | 1.031 / 1.031 | 1.025 / 1.020 |
| math | 16 | 0.915 / 1.196 | 1.019 / 1.029 |
| chat | 1 | 1.160 / 1.160 | 1.165 / 1.165 |
| chat | 2 | 1.001 / 1.001 | 1.002 / 1.002 |
| chat | 4 | 1.012 / 1.006 | 1.007 / 1.011 |
| chat | 8 | 1.013 / 1.023 | 1.020 / 1.026 |
| chat | 16 | 1.005 / 1.020 | 1.014 / 1.034 |

- bf16 several requests: 1.002-1.038 paired, every point above stock in both passes. Math c16 shows
  the tail effect plainly: aggregate 0.915 in one pass and 1.196 in the other, paired 1.019 / 1.029.
- bf16 one request: code 1.64, math 1.35, chat 1.17.

## 15. Sampled, grammar and logprob requests no longer stop the server (2026-11-16)

Found while testing the model card setting (track_a qwen38 preset: temperature 1.0, top-k 20,
top-p 0.95, thinking on). ax_chain refused any multi-request batch that was not all greedy, had a
grammar or asked for logprobs by raising ValueError inside the scheduler, which stops the SGLang
server (scheduler exception, SIGQUIT), not just the request. SGLang applies the model's generation
config by default (temperature 1.0), so two concurrent requests that did not set temperature 0 were
enough. Every speed and identity run so far was greedy, so none hit it.

Fix (ax_chain.tree_round): a decode round is a tree round only for one greedy request without
grammar or logprobs. Every other batch, including one sampled request, runs a chain round, i.e.
SGLang's own DFlash verify, which supports selector sampling (the server log reports
"selector decode folded into the draft cuda graph (sampling_enabled=True)" for the main and the
width-8 drafter graphs). Sampled single requests therefore get no tree gain yet; a lossless sampled
tree verify is separate work.

Checks (NVFP4, 32 slots, site s1e):
- qwen38 preset, c = 1, 2, 4, 16, 32 prompts per family, 1,024 tokens: no error. Acceptance equals
  stock: code 4.21-4.38 (stock 4.16-4.28), math 4.85-5.00 (4.77-4.91), chat 3.02-3.11 (3.04-3.06).
  Aggregate tok/s 0.976-1.129 of stock (sampled text differs run to run, so this is level, not a
  claim).
- Greedy ids at c = 1: 192 / 192 identical to site s1d. Stress test passed.
- Stock with the same preset: no error at any c (s1e/stock_samp).
- 64 slots with AX_SSM_POSITIONS=8 (s1e64): qwen38 preset at c = 1, 16, 64 and the stress test, no
  error; acceptance code 4.35-4.47, math 4.78-4.89, chat 3.00-3.04; 5,363 / 4,981 / 4,128 tok/s at
  c = 64.

Review of the fix (codex, VERDICT BUG; all in configurations no shipped lane uses) and changes
(site s1f):
- One-request chain round on a tree server without private chain runners (AX_CHAIN_GRAPH_WIDTHS
  unset) could be admitted to the main runner's batch-1 tree graph at the tree's width. Such servers
  now run chain rounds on the main runner eager.
- A chain width below the block that drafts the full block and verifies a prefix keeps SGLang's
  full-width selector sample; a sampled request would fail in the scatter. Start-up now refuses a
  chain width below the block that is not in AX_CHAIN_DRAFT_WIDTHS (AX_CHAIN_NARROW_GREEDY=1 allows
  it for greedy-only measurement servers). Every Stage 1a lane drafts width 8 at 8.
- A grammar-constrained chain round built its grammar tree before narrowing; it is rebuilt after.
- begin_round and choose each decided tree vs chain; choose now reuses begin_round's decision for the
  same batch and refuses a tree round after drafting at a chain width.

Checks on s1f (NVFP4, 32 slots): greedy ids c = 1 192 / 192 identical to s1e; stress passed;
benchmarks/sglang/mixed_requests_check.py (new: greedy, sampled, model-default parameters, JSON schema and
logprob requests alone and together, three rounds) passed, JSON parses, one logprob per token, the
greedy output sent with the others equal to it alone in 6 / 6; qwen38 preset c = 1, 4 no error,
acceptance as stock; a server with chain width 8 and no drafting width refuses to start with the
message above. Stage 1a correctness on s1e NVFP4 passed (teacher 3.05%, worst regret 4.75).
Stage 1a correctness on s1f bf16 passed (teacher 0.30%, over-tie 0, worst regret 0.25; the s1e bf16
run broke at its last step because benchmarks/sglang/run_baselines.sh was replaced on the pod while it ran).

## 16. Is sampled decoding lossless? Tree sampling, stock DFlash2, NVFP4 (2026-11-16)

Question: can a single sampled request (temperature 1.0, top-k 20, top-p 0.95, the model card's
defaults) take tree rounds (experimental `AX_TREE_SAMPLING=1`)? A greedy run is checked token by
token; a sampled one needs a distribution test.

Tools (all in `benchmarks/sglang/`):
- `sampling_dist_check.py`: 4 prompts x 500 samples, first 6 output ids, one request in flight;
  chi-square homogeneity per (prompt, position) and on the first two tokens jointly (28 tests);
  alpha 1e-3 (family-wise false alarm about 3%).
- `target_only_kernel_check.py`, `chain_reject_kernel_check.py`: Monte-Carlo checks of SGLang's two
  sampled-verify kernels on synthetic trees / chains (first committed token vs the root's p).
- `logprob_probe.py`: the probability a server reports at output position 1 for a fixed first token.
- `dft_lattice.py` `AX_LATTICE_DUPCHECK=1`: counts duplicate sibling tokens in our trees.

Calibration: the server ignores per-request `sampling_seed` for DFlash (two stock runs with different
seeds wrote identical files) and draws from its own generator, seeded by `--random-seed`; independent
runs use different `--random-seed`. Plain decoding vs plain decoding (two server seeds, seeded and
unseeded requests): 0 of 28 tests below 0.01, every pair.

Results against plain decoding (no speculation), same precision:

| run | tests p < 0.01 | p < 0.001 |
|---|---|---|
| bf16 ours, tree sampling (4000 rounds, all tree) | **0** | **0** |
| bf16 stock DFlash2 sampled | 7 | 7 |
| NVFP4 ours, tree sampling (all tree rounds) | 11-14 | 5-9 |
| NVFP4 stock DFlash2 sampled (with or without CUDA graphs) | 13 | 11-13 |

- **Our tree sampling verify is lossless.** In bf16 it matches plain decoding on all 28 tests. It
  routes through SGLang `eagle_sample` -> `tree_speculative_sampling_target_only`, which the
  Monte-Carlo check finds lossless for trees with distinct sibling tokens (max |z| 2.9-3.7 over 512
  tokens) and biased only when two siblings carry one token (that token's probability doubles,
  z = 34-48). Our trees: 0 duplicate siblings and 0 repeated candidate ids in 2000 trees.
- **Stock SGLang DFlash2 sampled decoding is not lossless, in bf16 too.** Example, prompt 0 after
  "The": stock reports p(" l") = 0.225 at its verify and samples " l" 53.5% of the time; " lamp"
  (p 0.035) never appears in 202 samples. Its rejection kernel (`chain_speculative_sampling_triton`)
  passes the Monte-Carlo check, and the selector walk draws tokens from the q it reports, so the
  defect is in how `_selector_sampling_accept` / the worker wires them; not yet located. Upstream issue,
  not in this plugin (our chain rounds use it unchanged, as every stock DFlash2 server does).
- **NVFP4 gives different distributions on different paths.** Probability after "The", prompt 0:

  | token | NVFP4 decode (1 row) | NVFP4 prefill | NVFP4 verify (8 rows) | bf16 decode | bf16 verify |
  |---|---|---|---|---|---|
  | " l" | 0.180 | 0.202 | 0.339 | 0.215 | 0.225 |
  | " storm" | 0.140 | 0.202 | 0.097 | 0.102 | 0.106 |
  | " salt" | 0.124 | 0.035 | 0.059 | 0.023 | 0.024 |

  The verify value is identical in every request (no dependence on the draft), and bf16 verify
  matches bf16 decode within 1-2 points. NVFP4's own decode and prefill disagree as much as decode
  and verify do: a property of the quantized model's row-count-dependent kernels, not of speculative
  decoding. For NVFP4, "lossless" can only mean "as the verify-shaped forward computes it"; the tree
  sampling failures in NVFP4 are this effect.

Decision: keep `AX_TREE_SAMPLING` off by default for now. It is correct (bf16 evidence), but its
speed gain is not shown yet (accepted tokens per round +6-15%, tok/s flat at c = 1 in the first
NVFP4 run) and turning it on changes which verify path sampled requests use. Next: speed with
matched settings; locate the stock DFlash2 selector-sampling defect.

Review (agy, grok, muse, kimi; read-only, given the code, an SGLang source excerpt and the measured data):
- All four: the tree sampling construction is lossless given correct target logits (agy, muse, kimi
  "LOSSLESS"; grok "NOT LOSSLESS" citing the measured NVFP4 data, which the bf16 run resolves).
- Fixed: a main-runner-only server (no `AX_CHAIN_GRAPH_WIDTHS`) now captures batch 1 only with the
  tree on (the fused tree capture assumes one request) and runs narrowed chain rounds eager with the
  tree off (grok); every configured chain width needs a runner whenever `AX_CHAIN_GRAPH_WIDTHS` is
  set, checked at start-up (grok); requests with a custom logit processor or min-p never take tree
  rounds (the tree verify applies neither) (kimi); those attributes are read directly so a rename
  fails closed (muse); the chi-square pooled cell is merged when still under 5 expected (grok, kimi);
  `logprob_probe.py` can pin the first token and states that it measures chain verify (grok, kimi);
  the kernel check asserts the root slot is written (muse).
- Checked, not changed: the sampler-staging guard returning None (stock returns None too; batches
  larger than the sampler draft eager), accept buffers on a new width (stock reallocates all of them
  when the capacity is reset), distinct siblings by construction (measured 0 duplicates).
- Found by the follow-up test (s1i): the start-up refusal of a narrowed chain width without its
  drafting width (section 15) ran only on servers with `AX_CHAIN_GRAPH_WIDTHS`; a server with the
  tree off, no chain graph widths and chain width 8 started, and its first sampled request stopped it
  (the full-width selector scatter error). The check now runs at worker start-up on every server
  (`_check_narrowed`), and `AX_CHAIN_DRAFT_WIDTHS` without `AX_CHAIN_GRAPH_WIDTHS` is refused too.

Speed of tree sampling (s1l, site s1k, single B200, sampled requests: qwen38 preset, temperature 1.0,
top-k 20, top-p 0.95, thinking budget 1024, 32 prompts per set, c = 1, `AX_TREE_SAMPLING` 0 vs 1,
two passes each, decode-time-weighted tok/s pooled over both passes; sampled text differs between
runs, so per-request medians move with output length):

| precision | set | off tok/s | on tok/s | ratio | median per request, off -> on (pass 1 / pass 2) |
|---|---|---|---|---|---|
| NVFP4 | code | 535 | 552 | 1.032 | 575 -> 649 / 591 -> 606 |
| NVFP4 | math | 616 | 646 | 1.050 | 647 -> 728 / 647 -> 719 |
| NVFP4 | chat | 385 | 371 | 0.964 | 381 -> 376 / 390 -> 383 |
| bf16 | code | 336 | 353 | 1.051 | 358 -> 420 / 351 -> 416 |
| bf16 | math | 379 | 417 | 1.102 | 392 -> 454 / 395 -> 469 |
| bf16 | chat | 234 | 235 | 1.005 | 236 -> 253 / 243 -> 238 |

Accepted tokens per round rise 11-20% (NVFP4 code 4.24-4.28 -> 4.83-5.06, math 4.91 -> 5.58-5.88,
chat 3.03-3.13 -> 3.34-3.36; bf16 similar), but the round costs more: the tree's sampled verify
(`eagle_sample`) builds dense probabilities with softmax, top-k and top-p renormalisation over the
full 248k vocabulary on all 16 rows, where greedy takes an argmax and stock DFlash2 uses a sparse
top-k path (`build_speculative_verify_target_probs`). Decision: `AX_TREE_SAMPLING` stays off by
default.

Profile (s1m, torch profiler, NVFP4, c = 1, 100-200 scheduler steps): GPU kernel time per step is
5.149 ms for sampled tree rounds and 5.108 ms for greedy tree rounds. The sampling kernels (top-p
renorm, top-k, softmax, `TreeSpeculativeSamplingTargetOnly`) add about 0.18 ms per step, so a sparse
top-k build would save at most that; not worth doing. The small gain is the method's: with
`AX_TREE_SAMPLING=0` a sampled request runs width-8 chain rounds, cheaper than a 16-node tree round
(tree verify, `_tiled_select` 0.16 ms, `_tree_finish` 0.14 ms), and under sampling the target's
choices are spread, so the tree adds only 11-20% accepted tokens per round (greedy code reaches 12
per round), which about pays for the larger round.

Stock DFlash2 sampled-decoding bias: located and fixed (bf16, single B200, chi-square against plain
decoding s1g8 `ar_bf16`, 4 prompts x 500 samples x 6 tokens, 28 tests; diagnostics in
`benchmarks/sglang/selector_diag/sitecustomize.py`, loaded with `PYTHONPATH` on a stock server):

| run | change | tests below 1e-3 | P(" l" \| "The") |
|---|---|---|---|
| plain decoding (s1g8) | - | reference | 0.186 |
| stock (s1g8) | - | 7 | 0.535 |
| s1n | selector sampling off (greedy drafts, sampled verify) | 0 | 0.237 |
| s1r | selector accept replaced by a pure-torch rejection sampler | 5 | 0.502 |
| s1v | draft-walk uniforms drawn eagerly before each replay | 6 | - |
| s1w | flashinfer `get_seed_and_offset` returns the old offset | 0 | - |

Path to the cause: per-round counters (s1o-s1q) showed every drafted token is the walk's pick, lies
among the candidates with q > 0, no duplicate candidates, q sums to 1, the walk's uniforms change every
round, and the accept coins differ from them; the chain kernel stays lossless with the target's top-k
zeros, near one-hot q and gamma 7 (`benchmarks/sglang/chain_reject_kernel_check.py --target-topk 20`); the
target's row-0 probabilities do not depend on the drafts (s1s); every sampled round takes the selector
accept (s1u). The reference accept still failing (s1r) left the draft: in the first decode round of a
request the walk's first uniform falls in the prefill sampler's CDF interval of the first output token
(s1t: first token "E" -> u0 in 0.01-0.20, "The" -> 0.21-0.61, "For" -> 0.64-1.0, matching that token's
interval [0, 0.19], [0.19, 0.61], [0.61, 1]). Given "The", u0 lands mostly in " l"'s interval of the
draft's q ([0.289, 0.503]), so " l" is drafted about half the time while q reports 0.214, and the
rejection test, correct for the q it is given, over-accepts it.

Cause: flashinfer 0.7.2 `flashinfer.sampling.get_seed_and_offset` advances the CUDA generator's philox
offset and returns the advanced value to its own kernel, so the generator's next user starts from the
same (seed, offset) and redraws the same numbers. Prefill samples with flashinfer; the next generator
user is the draft graph's `uniform_()`. Drawing the uniforms eagerly with `torch.rand` (s1v) starts at
the same offset and does not help; returning the old offset does (s1w).

Fix: `ax_engine_sglang/ax_rng.py`, on by default (`AX_FI_OFFSET_FIX=0` turns it off): the plugin
replaces `get_seed_and_offset` in `flashinfer.sampling` and `flashinfer.cake_sampling` with one that
returns the start of the reserved range. Greedy decoding draws nothing and is unchanged. Our tree
sampling passed before the fix because its verify coins and the target-only kernel take explicit
`torch.rand` uniforms, and its drafts are not sampled. Stock SGLang 0.5.23 without the plugin keeps the
bias.

Upstream status (checked 2026-11-16): already fixed, so no issue was filed. flashinfer PR 5651
("reserve RNG offsets for the current call", merged 2026-10-22) returns the start of the range from
0.7.3 on, and 0.7.4 adds a lock around the reservation (PR 9148); SGLang main pins flashinfer 0.7.4,
while the v0.5.23 release pins 0.7.2. `ax_rng` therefore patches only flashinfer below 0.7.3; moving
the pod to SGLang's next release (or flashinfer >= 0.7.3) makes it a no-op.

NVFP4 after the fix (s1y, against two plain-decoding runs s1g4 `ar_u1` / `ar_u2`, tests below 1e-3):
shipped config with `AX_TREE_SAMPLING=0` 10 / 8, with `AX_TREE_SAMPLING=1` 4 / 8, stock chain 11 / 12.
Unchanged from before the fix: the NVFP4 differences are the row-count-dependent numerics above, not
this bug.

Checks of the fix with the plugin (site s1x, shipped config: tree width 16, chain width 8, chain
graph widths 16 and 8): bf16, sampled requests with `AX_TREE_SAMPLING=0` (chain rounds through the
stock selector, biased before the fix) 0 of 28 tests below 1e-3; with `AX_TREE_SAMPLING=1` (tree
rounds) 0 of 28; NVFP4 greedy ids c = 1: 192 of 192 identical to s1f over s1f's 512 tokens (this run
used 1024). The tree-sampling speed table above was measured before the fix: its `AX_TREE_SAMPLING=0`
side ran the biased selector chain, whose acceptance does not match an unbiased drafter's, so the
comparison is repeated with the fix.

Tree sampling speed after the fix (s1y, site s1x, same settings as s1l: qwen38 preset, thinking budget
1024, 32 prompts per set, c = 1, two passes each, decode-time-weighted tok/s pooled over both passes;
accept = mean accepted tokens per round):

| precision | set | off tok/s | on tok/s | ratio | accept off | accept on |
|---|---|---|---|---|---|---|
| NVFP4 | code | 536 | 540 | 1.007 | 4.28 | 4.84 |
| NVFP4 | math | 618 | 668 | 1.080 | 4.93 | 5.94 |
| NVFP4 | chat | 380 | 372 | 0.978 | 3.04 | 3.36 |
| bf16 | code | 332 | 360 | 1.083 | 4.37 | 5.18 |
| bf16 | math | 383 | 417 | 1.088 | 5.04 | 5.90 |
| bf16 | chat | 237 | 237 | 1.003 | 3.12 | 3.44 |

The fixed chain side accepts about as much as the biased one did (the bias moved which token was
accepted, not how often), so the picture matches s1l: tree sampling gains 8-9% on bf16 code and math
and on NVFP4 math, breaks even on bf16 chat and NVFP4 code, and loses 2% on NVFP4 chat (inside the
run-to-run spread of sampled runs, about 3%). Both sides are lossless in bf16 (0 of 28 each).
Decision: `AX_TREE_SAMPLING` stays off by default (one switch for both precisions, and NVFP4 gains
nothing outside math); bf16 deployments serving code or math can turn it on.

## 17. TensorRT-LLM DFlash2 comparison (single B200, 2026-11-17)

TensorRT-LLM 1.4.0rc3 (pypi.nvidia.com, own venv, torch 2.14.0+cu130), `trtllm-serve` PyTorch backend
under `mpirun -n 1` (`benchmarks/sglang/run_trtllm_dflash2.sh`): plain decoding (TRT AR) and
`DFlashDecodingConfig` with the same incoai DFlash2 drafter (max_draft_len 7, drafter attention
TRTLLM), max batch 32, CUDA graphs with padding, KV fraction 0.85, no prefix reuse. Our shipped config
(site s1x: tree width 16, chain width 8, chain graph widths 16 and 8) and stock SGLang DFlash2 block 8
(trtllm_mha) were measured right after on the same pod (t2). All with `benchmarks/baselines/bench_speed.py`: greedy,
64 requests per set, 1024 tokens, c = 1 / 4 / 16, aggregate output tok/s. Mean output lengths and
truncation agree across the four servers within 3% (same model, greedy).

| bf16 | c | TRT AR | TRT DFlash2 | SGLang stock DFlash2 | ours | ours / TRT DFlash2 | ours / stock |
|---|---|---|---|---|---|---|---|
| code | 1 | 115 | 526 | 515 | 736 | 1.401 | 1.429 |
| code | 4 | 378 | 1251 | 1515 | 1545 | 1.235 | 1.020 |
| code | 16 | 1107 | 2821 | 3221 | 3345 | 1.186 | 1.039 |
| math | 1 | 117 | 496 | 457 | 607 | 1.224 | 1.329 |
| math | 4 | 387 | 1501 | 1406 | 1434 | 0.956 | 1.020 |
| math | 16 | 1210 | 3465 | 3399 | 3500 | 1.010 | 1.030 |
| chat | 1 | 117 | 269 | 249 | 288 | 1.072 | 1.159 |
| chat | 4 | 398 | 947 | 838 | 848 | 0.895 | 1.012 |
| chat | 16 | 1290 | 2623 | 2117 | 2118 | 0.808 | 1.001 |

| nvfp4 | c | TRT AR | TRT DFlash2 | SGLang stock DFlash2 | ours | ours / TRT DFlash2 | ours / stock |
|---|---|---|---|---|---|---|---|
| code | 1 | 160 | 661 | 811 | 1136 | 1.720 | 1.402 |
| code | 4 | 540 | 1116 | 2161 | 2206 | 1.976 | 1.021 |
| code | 16 | 1240 | 2426 | 3093 | 3799 | 1.566 | 1.228 |
| math | 1 | 165 | 644 | 727 | 944 | 1.467 | 1.299 |
| math | 4 | 573 | 1762 | 2107 | 2157 | 1.224 | 1.024 |
| math | 16 | 1702 | 4097 | 4603 | 4306 | 1.051 | 0.935 |
| chat | 1 | 164 | 351 | 402 | 461 | 1.315 | 1.148 |
| chat | 4 | 600 | 1121 | 1294 | 1294 | 1.154 | 0.999 |
| chat | 16 | 1842 | 3386 | 2953 | 3057 | 0.903 | 1.035 |

- One request: ours leads TensorRT-LLM DFlash2 everywhere: bf16 1.07-1.40, NVFP4 1.32-1.72
  (per-request decode bf16 code 1002 vs 628 tok/s, NVFP4 code 1136 aggregate vs 661).
- Several requests: NVFP4 code and math lead (1.05-1.98); bf16 code leads (1.19-1.24), bf16 math is
  even (0.96-1.01). Chat at c = 16 is TensorRT-LLM's in both precisions (bf16 0.81, NVFP4 0.90; bf16
  chat c = 4 0.90): its per-request decode rate there is 190 vs 153 tok/s (bf16) and 251 vs 210
  (NVFP4), and stock SGLang DFlash2 is as slow as ours, so the gap is between the engines' DFlash2
  multi-request paths, not in our tree. TensorRT-LLM reports no acceptance through /metrics, so
  whether it accepts more (its own selector path) or verifies a batch more cheaply is open; next step
  for this point: its per-iteration stats (`return_perf_metrics` / iteration log) on chat c = 16.
- TensorRT-LLM's TTFT is higher with DFlash2 at c = 16 (bf16 code 0.148 s vs 0.078 s ours).
- Ours against stock in this session: 1.00-1.43 except NVFP4 math c = 16 (0.935: per-request decode
  355 vs 352 tok/s, equal; the wall-time loss is the scheduling effect of section 11).
- Setup notes: `/workspace` (network FS) rejects uv's file operations, so the venv is on the root disk
  (12 GB); `--prerelease=allow` pulls mpmath 1.5.0a2, which breaks sympy (pinned back to 1.3.0); the
  pip OpenMPI 5 cannot `Spawn` as a singleton in this container, hence `mpirun -n 1` with
  `trtllm-llmapi-launch`; `snapshot_download(local_files_only=True)` refuses the bf16 cache (missing
  .gitattributes), so the script reads the snapshot directory.

Why TensorRT-LLM leads on chat at c = 16 (t3: TensorRT-LLM DFlash2 with one HTTP frontend and
per-iteration stats, chat, c = 1 and 16; the pod rebooted during the first attempt, t3 was rerun):

- Frontends: TensorRT-LLM starts 8 HTTP frontend processes by default (`num_serve_frontends=8`; SGLang
  has one tokenizer manager). With one frontend chat c = 16 drops 2-4% (NVFP4 3320 vs 3386 tok/s, bf16
  2529 vs 2623), so the frontends are not the gap.
- Acceptance is the same: tokens per round (accepted drafts per request-round + 1, decode-only
  iterations) 3.20-3.22 NVFP4 and 3.18-3.26 bf16 at every batch size, against 3.18 / 3.22 (NVFP4) and
  3.27 / 3.25 (bf16) for ours / stock at c = 16. TensorRT-LLM's drafter path does not draft better.
- The round is cheaper. Round time = tokens per round / per-request decode rate at c = 16:

| precision | TensorRT-LLM | ours | stock SGLang |
|---|---|---|---|
| NVFP4 | 3.22 / 246.9 = 13.0 ms | 3.18 / 210 = 15.1 ms | 3.22 / 209 = 15.4 ms |
| bf16 | 3.24 / 188.8 = 17.2 ms | 3.27 / 153 = 21.4 ms | 3.25 / 151 = 21.5 ms |

  At 16 requests TensorRT-LLM's draft + verify round of 16 x 8 tokens costs 14-20% less than SGLang's,
  ours and stock alike (our chain rounds at c = 16 are stock DFlash2 rounds at width 8). It is an
  engine difference in SGLang's DFlash2 batch path (drafter forward, verify forward, or the host work
  between them), not in the tree; TensorRT-LLM's `iterLatencyMS` overlaps under its overlap scheduler
  and cannot split it. Locating it needs a kernel-level profile of both engines at batch 16 (next
  step, not started; estimate 2-4 GPU hours).

### Batch-16 profiling and acceptance

Capture stock SGLang, AX Engine, and TensorRT-LLM DFlash2 on one B200, in that
order for the first pass. Use the same target and drafter checkpoint revisions,
ordered prompts, greedy decoding, 64 requests, a 1,024-token limit, and GPU. Start with
NVFP4, then repeat with BF16. Keep graph replay and overlap scheduling enabled.
Use a separate output directory for each pass.

Export `PROMPT_FILE` with the shared prompt snapshot and set `OUT` to the root
for this pass. Point `HF_HOME` to a complete cache containing only the pinned
snapshot of each model used by the run, and set `HF_HUB_OFFLINE=1`.
[Checkpoint revisions](2026-11-17-rtx-development-validation.md#checkpoint-identities)

Run the SGLang lanes with SGLang 0.5.23 and the AX package in `AXSITE`:

```bash
MODEL=RadixArk/Qwen3.8-27B-NVFP4
DRAFT=incoai/Qwen3.8-27B-DFlash2
capacity="--max-running-requests 32 --cuda-graph-max-bs-decode 32"
export N_REQUESTS=64 MAX_TOKENS=1024

MODEL="$MODEL" DRAFT="$DRAFT" OUT="$OUT/sglang" SETS=chat TREE_CONCURRENCY=16 \
LANES="chain8n axtree16cfl_b16" CHAIN_ARGS="$capacity --attention-backend trtllm_mha" \
TREE_ARGS="$capacity --speculative-draft-attention-backend trtllm_mha --prefill-attention-backend trtllm_mha --decode-attention-backend triton --speculative-attention-mode decode" \
AX_CHAIN_WIDTH=8 AX_CHAIN_GRAPH_WIDTHS="16 8" AX_CHAIN_DRAFT_WIDTHS=8 AX_CHAIN_ATTN=trtllm_mha \
PROFILE=1 AXPROF_SYNC=0 AXPROF_NVTX=1 NSYS_TRACE=1 bash benchmarks/sglang/run_tree_plugin.sh

OUT="$OUT/trt" DRAFT="$DRAFT" MODELS=nvfp4 TRT_LANES=trt_dflash2 \
SETS=chat CONCURRENCY=16 NSYS_TRACE=1 bash benchmarks/sglang/run_trtllm_dflash2.sh
```

For BF16, use `MODEL=Qwen/Qwen3.8-27B` and `MODELS=bf16`. Both launchers
start capture after the server is ready and write one `decode.nsys-rep` per
lane. Check that each report contains the GPU work of the decode worker.
Inspect the steady-state decode window at 16 running requests. Compare draft
and verify, the KV commit, host API calls, synchronization, and gaps on the
GPU critical path. Use `NSYS_GRAPH_TRACE=node` on a separate pass if the
graph-level capture shows that time is inside a graph. In the SGLang runs
`axprof.json` records host wall time in this configuration. The Nsight
timeline identifies which host work overlaps the GPU.

Use the trace to choose the smallest change to the DFlash2 worker or the
backend. Measure throughput on separate unprofiled runs (`PROFILE=0`,
`NSYS_TRACE=0`, and iteration polling off), with the original and candidate
builds interleaved. The targets are NVFP4 chat c=16 at least 1.05× the matched
TensorRT-LLM run and BF16 chat c=16 at least 1.00×. Check greedy token IDs
against the original AX build and verify that single-request NVFP4 code
throughput and chat latency stay within the measured baseline variation.

### Lattice predecessor bookkeeping (2026-11-20)

DFlash2 lattice scoring uses the drafter's transition table. Only the n-gram
builder reads the token before the anchor. AX skips the index conversion,
gather and per-request storage on DFlash2 lattice chain rounds, and skips
the corresponding copy and lookup on lattice tree rounds.

On one B200 (driver 595.112.07, Python 3.12.3, SGLang 0.5.23,
PyTorch 2.14.1+cu130, FlashInfer 0.7.2), the isolated width-8 chain helper
at batch 16 took a median 55.34 µs before and 0.16 µs after across six
alternating 1,000-call measurements per build. The CUDA check matched
all six lattice outputs for three anchor IDs and verified that n-gram
chain rounds still saved the expected predecessor for every request.

The unprofiled serving comparison used `d899260` and `187f430`, the
same ordered prompts, 64 requests per row, greedy decoding, a 1,024-token
limit, and tree width 16 / chain width 8. The first pass ran the original
build first. The second ran the candidate first and warmed up each
workload at its measured concurrency before timing.

| NVFP4 workload | Pass | Aggregate tok/s, original → candidate | Median decode tok/s per request, original → candidate |
|---|---:|---:|---:|
| Chat, c=16 | 1 | 2,420.1 → 2,756.9 | 192.2 → 196.6 |
| Chat, c=16 | 2 | 2,697.8 → 2,769.0 | 194.5 → 191.6 |
| Code, c=1 | 1 | 884.3 → 1,010.4 | 1,727.3 → 1,698.2 |
| Code, c=1 | 2 | 927.9 → 1,015.4 | 1,716.5 → 1,697.5 |

The first 16 chat requests had median TTFT of 1.940 s on the original
build and 0.264 s on the candidate. For the 42 of 64 prompts with equal
output length, the median paired decode rate ratio was 1.011×. That
ratio was 1.003× across 48 of 64 chat prompts in the warmed pass.
The aggregate chat gain fell from 13.9% to 2.6% between passes. The two
passes do not establish a repeatable per-request decode improvement.
The greedy token-ID check matched 16/16 chat sequences and 1/1 code
sequence at a 128-token limit in each pass.

The stock SGLang CUDA-graph node trace recorded CUDA kernels and
draft, target verify and KV-append NVTX ranges during the client warm-up.
The scheduler stopped responding before the measured c=16 workload.
TensorRT-LLM 1.4.0rc3 loaded the same checkpoints and stalled during
autotuning at startup, before any timed requests. No new TensorRT-LLM
throughput or batch-16 engine trace was collected.

### B200 baseline recovery and GDN replay (2026-11-20)

The `track_l_attn/trtllm_mha/chain16` speed artifact is a single-request
SGLang 0.5.22 run with Qwen3.5-9B BF16 and a width-16 DFlash block. Its
`trtllm_mha` setting selects the SGLang attention backend. The historical
TensorRT-LLM target for this work is the Qwen3.8-27B result in section 17.

TensorRT-LLM 1.4.0rc3 completed the Qwen3.8-27B NVFP4 benchmark with
`TRT_AUTOTUNE=0` and eight HTTP frontends. The client measured 1,785.4
tok/s aggregate, 134.7 tok/s median decode per request and 413 ms median
TTFT. This run used one B200 (driver 595.112.07), DFlash2 width 8, 64
ordered chat prompts, c=16, greedy decoding, and a 1,024-token limit.

A second run completed TensorRT-LLM autotuning and served the historical
c=1, c=4, c=16 sequence. The autotuner took 183.7 s for its first prefill
shape and 606.2 s for its first decode shape. All 192 measured requests
completed and the client wrote all three rows before the timeout fired
during teardown.

| NVFP4 chat, c=16 | Aggregate tok/s | Median decode tok/s per request | Median TTFT |
|---|---:|---:|---:|
| TensorRT-LLM, untuned | 1,785.4 | 134.7 | 413 ms |
| TensorRT-LLM, tuned | 2,205.6 | 153.6 | 351 ms |
| Stock SGLang, GDN snapshots | 2,618.7 | 189.1 | 116 ms |
| AX Engine, warmed lattice change | 2,769.0 | 191.6 | 114 ms |
| Stock SGLang with experimental GDN replay | 3,219.1 | 233.4 | 134 ms |
| Stock SGLang with experimental GDN replay, TF32x3 | 2,704.3 | 191.6 | 95 ms |

The AX and tuned TensorRT-LLM prompt digests match. For the 33 of 64
requests with the same completion length, the median paired decode rate
ratio was 1.253×. The same-pod measurements put AX 25.5% ahead in
aggregate. The new TensorRT-LLM run is 34.9% below its historical 3,385.9
tok/s record. The source of that cross-session difference is unresolved,
and the historical result remains the original performance target.

Stock SGLang 0.5.23 with GDN snapshots reached 2,618.7 tok/s aggregate and
189.1 tok/s median decode per request on the same B200 and ordered prompts.
It averaged 3.19 output tokens per verify call and allocated 37.12 GB for
intermediate SSM states at 32 request slots. SGLang explicitly rejected
`--enable-linear-replayssm-spec` with DFlash on this GDN model. The
installed DFlash worker commits through a backend hook that supports KDA
replay and GDN snapshots. The generic commit path already supports GDN
replay for verified linear chains.

In an isolated SGLang 0.5.23 copy, DFlash reused that GDN commit for a
single-GPU, width-8 chain with radix caching disabled. The worker selected
the replay path, allocated no intermediate SSM snapshots, and allocated
about 0.78 GB for the replay ring. The warmed serving run gained 22.9%
aggregate throughput over stock. For the 31 of 64 requests with equal
completion length, median paired decode throughput rose 24.3%.

Replay matched 0/16 full greedy sequences from stock at a 128-token limit.
The earliest difference was the second generated token. Following AX Code
Qwen 4 Max's review, the parity check repeated the same c=16 request batch
on the replay worker after the first batch finished. The replay runs
matched 16/16 sequences. This candidate fails the stock parity gate and
remains an isolated experiment. The patch, raw timings, token IDs, and QA
are retained locally under `results/b200-dflash-hotpath/replayssm/`.

Two c=1 probes on fresh workers reproduced the second-token difference.
Both variants agreed on the first token and its log probability. On the
second token, stock favored its chosen token over the replay choice by
0.875 and 0.375 log-probability units. Replay reversed those preferences
by 0.125 and 0.375. Repeating the first probe in each worker gave the same
output and token scores. The discrepancy reproduces at c=1 with stable
inputs, before a saturated batch or repeated slot churn is required.

Increasing only the replay verify kernel's dot-product precision from
TF32 to TF32x3 brought throughput to 2,704.3 tok/s, 3.3% above the
snapshot baseline. It still matched 0/16 stock sequences and repeated
16/16 of its own sequences. Higher dot-product precision alone did not
recover the stock output.

### First DFlash verify comparison (2026-11-20)

Following the Claude Code Opus 5.5 review, we instrumented separate stock and
replay SGLang 0.5.23 workers on the same B200. The workers ran the same NVFP4
Qwen3.8-27B target and width-8 DFlash2 checkpoint with one live request and
CUDA graphs disabled. Both monitored prompts reproduced the second-token
divergence. The worker runs target verify and selects tokens before calling
the GDN commit.

For the first GDN layer, the eight verify token IDs, convolution state,
FP32 SSM checkpoint, post-convolution q/k/v, and gate inputs were bitwise
equal. The replay history was empty, with `write_pos`, `cache_base`, and
`is_flush` all zero. Reinvoking each SGLang kernel with the captured
stock tensors reproduced all 49,152 elements of its respective serving
output bitwise. The difference therefore appears in the verify calculation
on matching prefill inputs, before a replay commit or slot reuse is needed.

| First GDN layer, first verify position | Max absolute difference from stock | Max absolute error vs sequential FP64 reference |
|---|---:|---:|
| Stock recurrent verify | 0 | 0.0035424 |
| Replay TF32 | 0.0078125 | 0.0042701 |
| Replay TF32x3 | 0 | 0.0035424 |

In a separate TF32x3 serving trace, all 48 GDN layers had bitwise-equal
prefill checkpoints and convolution states. At verify position zero, layer
0 matched stock. Layer 1 used matching inputs and differed in 2 of 6,144
outputs, with a maximum absolute difference of 5.96 × 10⁻⁸. Position-zero
GDN inputs first differed at layer 10. The maximum position-zero output
difference across the trace was 0.0058594 at layer 56. TF32x3 matched the
generated tokens on the two short c=1 prompts while changing their scores.
Its c=16 parity failure still stands.

### Recurrent verification with accepted-prefix replay (2026-11-20)

The second prototype keeps SGLang's recurrent Triton GDN verifier and writes
compact inputs for each draft step. After acceptance, one all-layer kernel
replays the accepted prefix into the FP32 checkpoints. This uses the existing
SGLang fold-every-commit verifier and commit kernel. At 32 request slots,
intermediate SSM snapshots went from 37.12 GB to zero and the raw replay
windows occupied approximately 0.197 GB.

The captured layer fixture matched stock bitwise for all verify outputs
at batches 1 and 16. At batch 1, the committed FP32 state matched the
corresponding stock snapshot for every accepted length from 1 through 8.
At batch 16, all eight tests combined different accepted lengths across
requests and matched every state element.

The first unprofiled NVFP4 chat c=16 run reached 2,982.0 tok/s, 13.9% above
the earlier stock run. Median paired decode throughput increased 11.5%
across the 43 of 64 requests with matching completion lengths. It matched
all 16 stock greedy sequences and all 16 repeated sequences at a 128-token
limit.

The reverse-order pair ran replay first and warmed each worker with 16
concurrent requests. Both timed rows used the same ordered 64 prompts,
greedy decoding, and a 1,024-token limit. Replay reached 2,957.3 tok/s
versus stock's 2,685.6 tok/s, a 10.1% gain. Median paired decode throughput
rose 12.0% across the 43 equal-length requests. A separate 256-token check
matched all 64 stock sequences, and both workers repeated all 64 of their
own sequences. This covers the single-B200 NVFP4 stock-chain lane. The
historical 3,385.9 tok/s TensorRT-LLM target, BF16, and AX tree integration
remain outside the demonstrated result.

Apply the benchmark patch to a copy of the SGLang 0.5.23 build used in
this comparison. Use this overlay only for the tested stock width-8
DFlash2 setup with an FP32 GDN state, TP1/PP1, and radix cache off. The
overlay selects recurrent folding for replay-enabled GDN pools.
The following commands use the same serving entry point as the measurements.

```bash
(
set -euo pipefail
overlay=$(mktemp -d)
python - "$overlay" <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import shutil
import sys

assert version("sglang") == "0.5.23"
source, = find_spec("sglang").submodule_search_locations
shutil.copytree(source, Path(sys.argv[1]) / "sglang", ignore=shutil.ignore_patterns("__pycache__"))
PY
patch="$PWD/benchmarks/sglang/sglang-0.5.23-dflash-recurrent-replay.patch"
git -C "$overlay" apply --check "$patch"
git -C "$overlay" apply "$patch"
MODEL=RadixArk/Qwen3.8-27B-NVFP4 DRAFT=incoai/Qwen3.8-27B-DFlash2 \
  PYTHONPATH="$overlay" SGLANG_PLUGINS=none BLOCK=8 RADIX_CACHE=0 \
  EXTRA_ARGS="--attention-backend trtllm_mha --max-running-requests 32 --cuda-graph-max-bs-decode 32 --enable-linear-replayssm-spec" \
  bash benchmarks/sglang/serve_sglang.sh dflash
)
```
