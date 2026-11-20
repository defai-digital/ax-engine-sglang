# Jetson Thor: ax-engine-sglang and vLLM on 2026-11-19

ax-engine-sglang runs on NVIDIA Jetson Thor without code changes to its
tree path beyond the FlashInfer `fc1_plan` fix in `dec7f17`. On
every target that fits one Thor, the ax-engine-sglang tree is the fastest measured
route, ahead of stock SGLang DFlash2 chain decoding and of vLLM's DFlash and
MTP. Qwen3.8-Flash-Next NVFP4 does not fit one Thor and was run as a
two-host pipeline over 2.5 GbE (section 5); there MTP7 with the ax-engine-sglang QSA
ring is the fastest route for code and math.

All numbers are measured end-to-end output throughput at concurrency 1,
greedy, thinking off. Nothing in this report is projected.

## 1. Hardware and software

| Item | Value |
|---|---|
| Hosts | `df-thor-01` (192.168.2.61), `df-thor-02` (192.168.2.62) |
| GPU | NVIDIA Thor, SM 11.0, aarch64, 122 GB unified LPDDR5X (~273 GB/s) |
| OS / driver | L4T R39.2.0, kernel 6.8.12-1021-tegra, driver 595.78 |
| Link | 2.5 GbE (`enP2p1s0`), no InfiniBand / RoCE |
| SGLang | `lmsysorg/sglang:v0.5.23-cu130` (torch 2.14.1+cu130, Triton 3.8, FlashInfer 0.7.2, sgl_kernel 0.4.10) |
| vLLM | `nvcr.io/nvidia/vllm:26.11-py3` |
| ax-engine-sglang | working tree at `fcc5ee2` (ax-engine-sglang sources identical to `dec7f17`); section 9 re-checks `6b0df5d` and the fixes made after it |

Co-located document-labelling services were stopped on both hosts for the
whole campaign and restored afterwards. Page cache was dropped before every
server start (CUDA's free-memory figure on Jetson excludes page cache).

Benchmark: `benchmarks/baselines/bench_speed.py` through `benchmarks/sglang/run_baselines.sh` and
`benchmarks/sglang/run_tree_plugin.sh`; code (HumanEval), math (GSM8K) and chat prompt
sets, one request at a time. 9B and 27B NVFP4 use 12 prompts per set and
512 output tokens; 27B bf16 uses 8 prompts and 384 tokens (bf16 decoding at
~4.5 tok/s makes the full set impractical). Harness scripts are in
`results/thor_2026-11-19/sglang/scripts` and `results/thor_2026-11-19/vllm`.
They expect a host directory holding a copy of this repository (`repo27`), the
Hugging Face cache (`hf`), `data/ngram_counts.npz` for the DFlash v1 drafter
and, for Flash-Next, the per-host stage directory `fnstage` (section 6, item 5).

## 2. Qwen3.5-9B bf16 (DFlash drafter `z-lab/Qwen3.5-9B-DFlash`)

Output tokens/s, code / math / chat.

| Route | Engine | Attention | code | math | chat |
|---|---|---|---:|---:|---:|
| AR | SGLang | FlashInfer | 14.8 | 14.9 | 14.9 |
| AR | vLLM | — | 14.4 | 14.5 | 14.5 |
| MTP3 | vLLM | — | 36.4 | 34.6 | 26.5 |
| MTP5 | vLLM | — | 41.7 | 37.7 | 24.4 |
| DFlash 16 | vLLM | — | 116.3 | 79.6 | 33.6 |
| DFlash chain 16 | SGLang | FlashInfer | 118.7 | 78.0 | 31.3 |
| DFlash chain 16 | SGLang | Triton | 121.1 | 81.9 | 33.6 |
| ax-engine-sglang tree, launch defaults | SGLang | FlashInfer | 125.5 | 98.2 | 46.2 |
| **ax-engine-sglang tree `axtree32cf_b16`** | SGLang | Triton | **131.3** | **102.3** | **47.6** |

The first 9B ax-engine-sglang tree lane ran before the FlashInfer `fc1_plan` fix and
therefore only on Triton. The FlashInfer tree row was measured afterwards at
`6b0df5d` with the minimal command (launch defaults) and with the co-located
services running (section 9); it is 1.06× / 1.26× / 1.48× the FlashInfer chain.
The FlashInfer chain lane overlapped a model download and is slightly perturbed.

Teacher-forced check (`benchmarks/sglang/sglang_teacher_check.py`, fresh prefill on a
stock server, tie 0.5 nats, maximum near-tie rate 0.5%) over AR, ax-engine-sglang tree and
both chain lanes: **passed**. 144 outputs, 48,799 tokens, 0.36% mismatch
rate, none over the tie threshold, worst regret 0.25 nats. Mismatch rows per
lane: AR 61, ax-engine-sglang tree 31, chain FlashInfer 38, chain Triton 48.
The FlashInfer tree lane was checked separately and also **passed**: 36
outputs, 12,205 tokens, 47 mismatches (0.39%), none over the tie threshold,
worst regret 0.25 nats. Its greedy outputs equal the Triton tree's on 19 of 36
prompts (all 12 code prompts), the level bf16 near-ties give between attention
backends.

## 3. Qwen3.8-27B NVFP4 (`RadixArk/Qwen3.8-27B-NVFP4`, drafter `incoai/Qwen3.8-27B-DFlash2`)

| Route | Engine | Attention | code | math | chat |
|---|---|---|---:|---:|---:|
| AR | SGLang | FlashInfer | 13.4 | 13.5 | 13.5 |
| AR | vLLM | — | 13.0 | 13.1 | 13.1 |
| MTP3 | SGLang | FlashInfer | 36.2 | 35.2 | 27.0 |
| MTP3 | vLLM | — | 35.0 | 34.6 | 26.9 |
| DFlash 16 | vLLM | — | 89.3 | 67.1 | 31.9 |
| DFlash2 chain 16 | SGLang | FlashInfer | 93.0 | 73.5 | 30.6 |
| DFlash2 chain 16 | SGLang | Triton | 98.4 | 70.0 | 31.3 |
| **ax-engine-sglang tree `axtree32cfl_b16`** | SGLang | FlashInfer | 106.1 | **92.1** | 43.1 |
| **ax-engine-sglang tree `axtree32cfl_b16`** | SGLang | Triton | **112.0** | 90.0 | **45.2** |

On FlashInfer the ax-engine-sglang tree is 1.14× / 1.25× / 1.41× the stock chain and
7.9× / 6.8× / 3.2× AR; against vLLM's best route (DFlash) it is
1.19× / 1.37× / 1.35×. GDN tree verification (48/48 layers) was enabled
automatically.

Greedy identity against AR is low for every speculative lane, stock ones
included (code 5–9 of 12, math 0–1 of 12, chat 0 of 12), as expected for NVFP4
(see `2026-11-18-b200-27b-nvfp4.md`, Greedy follow-up). The teacher check
**fails for AR itself**: a fresh prefill disagrees with AR decoding on about
3% of tokens, so the check cannot qualify NVFP4 routes on Thor. Per lane
(216 outputs, 73,059 tokens in total):

| Lane | Mismatches | Over tie (≥0.5 nats) | Worst regret |
|---|---:|---:|---:|
| AR | 348 | 131 | 2.62 |
| MTP3 | 342 | 118 | 1.87 |
| Chain 16, FlashInfer | 365 | 145 | 3.87 |
| ax-engine-sglang tree, FlashInfer | 362 | 124 | 2.00 |
| Chain 16, Triton | 437 | 171 | 2.50 |
| ax-engine-sglang tree, Triton | 359 | 134 | 3.25 |

The ax-engine-sglang tree lanes sit at the AR baseline: NVFP4 prefill and decode numerics
differ on SM110, and that dominates. Use bf16 (next section) for the
correctness verdict, and AR when exact AR output is required.

## 4. Qwen3.8-27B bf16 (`Qwen/Qwen3.8-27B`)

| Route | Engine | Attention | code | math | chat |
|---|---|---|---:|---:|---:|
| AR | SGLang | FlashInfer | 4.6 | 4.6 | 4.6 |
| AR | vLLM | — | 4.4 | 4.4 | 4.4 |
| MTP3 | SGLang | FlashInfer | 13.5 | 12.9 | 9.9 |
| MTP3 | vLLM | — | 14.2 | 13.4 | 10.5 |
| DFlash 16 | vLLM | — | 43.9 | 30.1 | 13.5 |
| DFlash2 chain 16 | SGLang | FlashInfer | 43.1 | 29.6 | 12.5 |
| DFlash2 chain 16 | SGLang | Triton | 44.5 | 30.0 | 12.9 |
| **ax-engine-sglang tree `axtree32cfl_b16`** | SGLang | FlashInfer | 47.5 | 37.6 | **18.3** |
| **ax-engine-sglang tree `axtree32cfl_b16`** | SGLang | Triton | **48.0** | **38.5** | 18.2 |

ax-engine-sglang tree / stock chain on FlashInfer: 1.10× / 1.27× / 1.46×;
ax-engine-sglang tree / AR:
10.3× / 8.1× / 3.9×.

Teacher check over all six lanes: **passed**. 144 outputs, 40,152 tokens,
0.32% mismatch rate, none over the tie threshold, worst regret 0.125 nats.
Mismatch rows: AR 19, MTP3 22, chain FlashInfer 18, ax-engine-sglang tree FlashInfer 27,
chain Triton 23, ax-engine-sglang tree Triton 21. Every lane matched AR exactly on all 8 code
prompts; math and chat divergences start at a median of 40–140 tokens for
all lanes alike.

## 5. Qwen3.8-Flash-Next NVFP4 over two Thors

`RadixArk/Qwen3.8-Flash-Next-NVFP4` needs about 148 GB of weights (51 GB FP8
PLE n-gram table in layer 1, 68 GB NVFP4 experts, 16 GB bf16 dense), so it
was served with SGLang pipeline parallelism over the two hosts:

```text
--pp-size 2 --nnodes 2 --dist-init-addr 192.168.2.62:29500 --node-rank {0,1}
--max-running-requests 4 --chunked-prefill-size 2048 --mem-fraction-static 0.90
--moe-runner-backend flashinfer_cutlass --speculative-moe-runner-backend flashinfer_cutlass
SGLANG_PP_LAYER_PARTITION=4,44  NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=enP2p1s0  NCCL_IB_DISABLE=1
SGLANG_ENABLE_PP_SPEC=1   # MTP lanes only
```

Stage 0 (`df-thor-02`, layers 0–3 with the PLE table) and stage 1
(`df-thor-01`, layers 4–47, MTP head, LM head) each hold 74.0 GB of weights.
The first split, 10/38, left stage 0 only 6 GB and no KV cache. The ax-engine-sglang
lanes add `SGLANG_PLUGINS=ax-engine AX_QSA_RING=16` on both stages.
Coordinator: `results/thor_2026-11-19/sglang/scripts/run_fn_pp.sh`.

| Route | code | math | chat | Accepted tokens / step |
|---|---:|---:|---:|---|
| AR | 18.3 | 19.0 | 19.2 | — |
| MTP3, stock | 32.0 | 32.4 | 24.8 | 3.80 / 3.54 / 2.67 |
| MTP3 + ax-engine-sglang QSA ring 16 | 33.3 | 32.5 | 25.7 | 3.81 / 3.52 / 2.73 |
| **MTP7 + ax-engine-sglang QSA ring 16** | **43.1** | **37.5** | 22.8 | 7.00 / 5.60 / 3.22 |

MTP7 with the ax-engine-sglang QSA pending ring is 2.4× / 2.0× / 1.2× AR and the
fastest route for code and math; MTP3 remains better for chat, where
acceptance is low. Single-request decode crosses the 2.5 GbE link twice per
step, which caps AR at about 19 tok/s. There is no single-host Flash-Next
baseline on Thor to compare against. vLLM was not run for Flash-Next.
The first MTP3 attempt stopped on SGLang's pipeline-parallel speculation gate
(section 6, item 6), and the MTP3 + ax-engine-sglang attempt queued behind it was
killed 8 s after launch (exit 137) before serving anything; all MTP rows are from
the rerun with `SGLANG_ENABLE_PP_SPEC=1` (`scripts/fn_spec.log`).

Greedy identity against AR is at the same level for stock and ax-engine-sglang lanes
(code 10–11 of 12, math and chat 0 of 12, first divergence after a median of
18–27 tokens in math and chat), as expected for NVFP4. Decoded MTP7 outputs
are coherent. No teacher check was run: the 27B NVFP4 result shows it cannot
separate NVFP4 routes on SM110.

## 6. Problems found and fixed

1. **ax-engine-sglang tree assertion on FlashInfer** (`fast_prefill_plan` requires
   `uniform_verify_width`). Fixed in `ax_engine_sglang/fc1_plan.py` (`dec7f17`)
   and verified on Thor; both attention backends now run the ax-engine-sglang tree.
2. **NVFP4 MoE preflight skipped on Thor.** `tests/gpu_nvfp4_moe_check.py`
   only ran on SM120, but SGLang routes SM110 NVFP4 MoE through the same
   FlashInfer CUTLASS path (`is_blackwell`, not `is_sm100`). The test now runs
   on SM110 and SM120; on Thor it passes with maximum relative error
   1.5e-4, including changed-input CUDA graph replay.
3. **Automatic NVFP4 MoE backend fails on SM110** (`'FusedMoE' object has no
   attribute 'g1_scale_c'`): SGLang 0.5.23 `auto` resolves to FlashInfer
   TRT-LLM for any capability ≥ 10 while weights are prepared for another
   layout. Pass `--moe-runner-backend flashinfer_cutlass
   --speculative-moe-runner-backend flashinfer_cutlass`, as already
   documented for SM120; the README now names SM110 too.
4. **Memory sizing on a shared 122 GB unified-memory host.**
   `--mem-fraction-static` is a fraction of memory free at start-up, and the
   speculative / hybrid-Mamba reserve has a 10 GB floor on GPUs above 60 GB
   plus ~1.5 MB per chunked-prefill token. Working settings:
   `--mem-fraction-static 0.70 --chunked-prefill-size 2048` (9B, 27B NVFP4),
   `0.80` for 27B bf16, `0.90` for each Flash-Next stage, after dropping page
   cache. 27B bf16 MTP3 additionally needs `--max-running-requests 4`, or the
   Mamba state pool sized for the default request count leaves no KV cache.
5. **Partial checkpoints per host.** SGLang refuses a hub snapshot with
   missing files and a local directory whose `model.safetensors.index.json`
   names missing shards. Each Flash-Next host therefore serves a stage
   directory of symlinks plus an index filtered to its own shards.
6. **Pipeline parallelism with MTP is gated.** Without
   `SGLANG_ENABLE_PP_SPEC=1`, SGLang 0.5.23 accepts PP + speculative decoding
   only on disaggregated prefill nodes. With the flag (EAGLE, aggregate mode)
   stock MTP and the ax-engine-sglang QSA ring both run across two hosts.

7. **Root-owned output directories.** The lanes write results from inside the
   container as root, so a later host-side `mkdir` under `results/` fails for
   the unprivileged user. The harness tolerates this (the container creates the
   directory); copies made for a new run should go to a new directory rather
   than be removed with `sudo`.

Found in the follow-up review (section 9) and fixed after `6b0df5d`:

8. **A masked FlashInfer plan outside the tree shape asserted.** `fc1_plan`
   asserted on `FC1_TREE_N` values other than 16 / 32 / 64 and on any masked
   plan that did not match the recorded host length. It now falls back to
   SGLang's own plan, and `fc1_branch64` clears the host length as soon as the
   round's verify returns, so a later masked plan cannot pick up a stale value.
9. **A missing draft block length fell back to 24 tokens.** Without
   `--speculative-num-draft-tokens`, a drafter `block_size` or
   `FC1_DRAFT_BLOCK`, the worker drafted DForkTree's default of 24. The launch
   now stops with the reason. `FC1_DRAFT_BLOCK` above 32 with `FC1_TREE64=1` is
   refused (the tree's ancestor closure covers distance 31).
10. **Launch messages.** The recommended command drafts 16 tokens with DFlash2
    drafters whose config lists `block_size` 8; the note now says so instead of
    reading as a conflict, the `FC1_DRAFT_BLOCK` contradiction names the draft
    block rather than "the drafter block", and a stock speculative launch whose
    target config cannot be read says that the NVFP4 MoE backend and
    `AX_QSA_RING` defaults were not derived.
11. **Harness.** `benchmarks/sglang/serve_sglang.sh` now takes `EXTRA_ARGS_BASE` itself
    (the Thor scripts used to patch it in place, and the teacher and stage-1
    scripts relied on that patch without applying it). the archived `queue_teach.sh` listed a non-existent MTP3 ids
    path (the reported teacher files cover all six lanes, 216 and 144 outputs,
    so the reported numbers are not affected);
    `run_teacher.sh` and `run_lanes27.sh` now fail when a lane or the teacher
    fails; `run_vllm_lanes.sh` removes a lane's earlier speed file before
    serving, refuses a port that already answers and reports a failed
    benchmark; `run_fn_pp.sh` sets `SGLANG_ENABLE_PP_SPEC=1` for MTP lanes
    itself (it was passed through `EXTRA_ENV` in the reported run).

## 7. Not covered

- Sampled decoding (temperature > 0), `AX_TREE_SAMPLING` and
  `AX_DFLASH_JOINT_SAMPLING` were not run against a served model on Thor; the
  sampler kernel checks pass on SM110 (section 9), and sampling distribution
  qualification remains the B200 result.
- Concurrency above 1 (`AX_CHAIN_*` chain rounds).
- Flash-Next bf16 (about 360 GB) cannot be served on two Thors.
- Flash-Next was not re-run after `6b0df5d`; its launch defaults were checked
  by argument resolution only (section 9).
- `AX_QSA_STABLE_TOPK` on a served Flash-Next model.

## 8. Recommendation

For single-stream serving on Thor, SGLang with ax-engine-sglang is the faster
choice for Qwen3.5-9B and Qwen3.8-27B in both precisions: the ax-engine-sglang tree beats vLLM
DFlash by 8–38% on 9B (FlashInfer; 13–42% on Triton), 19–37% on 27B NVFP4 and
8–35% on 27B bf16 (FlashInfer lanes; Triton is within 6 points). vLLM and
stock SGLang are equivalent for AR and MTP. Prefer 27B NVFP4 on Thor (about
2.9× AR throughput of bf16 at equal memory bandwidth); qualify exact-output
requirements on bf16. Flash-Next NVFP4 is servable on two Thors over
2.5 GbE, at 43 tok/s for code with MTP7 and the ax-engine-sglang QSA ring, but the link
bounds it; for an interactive Flash-Next deployment one larger-memory GPU is
the better fit.

Raw results: `results/thor_2026-11-19/sglang/` and
`results/thor_2026-11-19/vllm/` (speed JSON, token ids, teacher reports,
server logs).

## 9. Follow-up verification at `6b0df5d` and after the fixes

The measurements above ran at `fcc5ee2` / `eb6e15a`. The same day, `6b0df5d`
and then the fixes of section 6 (items 8–11) were checked on both hosts with
the co-located services left running (about 59 GB free of 122 GB), so these
throughput figures are slightly perturbed and are not table entries except for
the 9B FlashInfer row.

| Check | Host | Result |
|---|---|---|
| CPU test suite (`ax-engine-sglang/tests`, aarch64 container) | `df-thor-02`, `df-thor-01` | `6b0df5d`: 105 passed; with fixes: 107 passed |
| All nine `tests/gpu_*.py` checks (no model weights) | `df-thor-01` | pass at `6b0df5d` and with fixes; NVFP4 CUTLASS maximum relative error 1.5e-4 |
| Launch-default resolution (27B NVFP4, 27B bf16, 9B, Flash-Next MTP) | `df-thor-02` | GDN shape 48 / 48 and 24 / 32, lattice for DFlash2, CUTLASS MoE and `AX_QSA_RING=16` for Flash-Next on SM110; an unreadable drafter config stops the launch |
| 27B NVFP4 tree, minimal command, `6b0df5d` | `df-thor-02` | 107.6 / 93.3 / 43.7 tok/s; 36 of 36 outputs token-identical to the `eb6e15a` run |
| 27B NVFP4 tree, with fixes, `AX_LATTICE=1` set by hand | `df-thor-02` | 109.1 / 94.5 / 44.3 tok/s; 36 of 36 identical |
| 9B tree on FlashInfer, minimal command, `6b0df5d` | `df-thor-02` | 125.5 / 98.2 / 46.2 tok/s; teacher check passed (section 2) |
| 9B tree on FlashInfer, with fixes | `df-thor-02` | 127.3 / 99.3 / 46.7 tok/s; 36 of 36 identical to the `6b0df5d` run |

With `--mem-fraction-static 0.70` of 59 GB free, the 27B NVFP4 tree server keeps
87,110 KV tokens and about 11 GB of headroom; the three document-labelling
services stayed active throughout.

Setting only `AX_LATTICE=1` installs the tree hooks in the launcher before the
defaults are resolved; the scheduler process starts with the resolved
environment, and the run above is token-identical, so this ordering has no
effect on serving.

The review used five independent readers (code, harness, and this report
against the raw files). Every throughput and teacher figure in sections 2–5
matched the raw results; the two rounding errors they found in the prose ratios
are corrected above.

Raw results: `results/thor_2026-11-19/sglang/{thor_27b_nvfp4_head,thor_27b_nvfp4_fix1,thor_9b_fi_head,thor_9b_fi_head_teacher,thor_9b_fi_fix1}`
and `results/thor_2026-11-19/sglang/followup` (scripts and logs).

## 10. HEAD `88baa07`: sampled decoding and several running requests

On 2026-11-20 `88baa07` ran on both hosts, again with the three document-labelling
services active (about 59 GB free at each server start). All lanes use the 9B
(`z-lab/Qwen3.5-9B-DFlash`, n-gram table, FlashInfer) unless noted, 12 prompts
per family, 512 tokens, greedy, minimal launch command plus the flags listed.

| Check | Result |
|---|---|
| CPU test suite, `df-thor-01` | 109 passed at `88baa07`; 128 passed with the start-up fix below |
| `tests/gpu_*.py`, `df-thor-01` | all ten pass; `gpu_nvfp4_parity_check.py` is SM120 only and skips on SM110 |
| 27B NVFP4 tree | 109.0 / 94.3 / 44.3 tok/s; 36 of 36 outputs identical to the section 9 run; mixed requests passed |
| 9B tree | 128.0 / 100.0 / 47.1 tok/s; 36 of 36 identical; mixed requests passed |
| Sampled tree verify, `AX_TREE_SAMPLING=1`, against stock AR with `--random-seed 1` | 4 prompts x 300 samples; 28 tests, none below p = 0.01 (0.3 expected by chance): `DIST_PASS` |
| `--max-running-requests 4`, default chain rounds | mixed requests passed; stress passed (2,000 requests, 178 cancelled, all seen as aborted, token and GDN-state pools back to idle, 24 reference outputs unchanged) |

Aggregate output tok/s for code / math / chat with `--max-running-requests 4`:

| Server | c = 1 | c = 2 | c = 4 |
|---|---|---|---|
| Plugin, default chain rounds | 124.1 / 99.2 / 46.7 | 190.7 / 138.5 / 58.2 | 297.3 / 232.8 / 93.0 |
| Plugin, width-8 chain rounds (`AX_CHAIN_WIDTH=8 AX_CHAIN_GRAPH_WIDTHS=8 AX_CHAIN_DRAFT_WIDTHS=8 AX_SSM_POSITIONS=8`) | 126.8 / 100.1 / 47.1 | 139.7 / 118.2 / 60.3 | 217.7 / 209.0 / 105.4 |
| Stock DFlash, block 16 | 117.1 / 78.5 / 32.8 | 193.5 / 141.6 / 57.6 | 300.9 / 234.5 / 91.9 |

With several requests the default plugin server runs at the stock DFlash rate
(within 3%); the tree's gain is at one request. Width-8 chain rounds lose 10–27%
on code and math at c = 2–4 and gain 13% on chat at c = 4, so the default
(chain rounds at the drafted block) is the better Thor setting for this model.

At c = 1 every multi-slot server produced the same 36 outputs as the
single-slot tree server. At c = 4, 19 of 36 (default chain) and 18 of 36
(width 8) outputs matched their own c = 1 outputs; stock DFlash matched 22 of
36. Batched rounds resolve greedy near-ties differently for stock SGLang as
well, so these divergences are not attributed to the plugin.

**Problem found and fixed.** The width-8 settings above stopped the launch at
`88baa07` with `AX_SSM_POSITIONS needs FC1_FUSED=1, AX_TREE=1 and AX_CHAIN_WIDTH`
unless `FC1_FUSED=1` was set by hand: the plugin validated the environment when
it loaded, before the launch-default resolver writes `FC1_FUSED=1`. The plugin
now validates a launcher configuration that requests the tree against the tree
defaults, marks it provisional, installs the tree hooks only after resolution,
and re-reads the environment after every resolution (`config.startup_config`,
`tests/test_startup_config.py`). Explicit settings still win, and without
resolution hooks (SGLang 0.5.22) or with `AX_AUTO_DEFAULTS=0` the settings must
be explicit as before. A launch that relies on a checkpoint-derived value at
start-up, such as `AX_CHAIN_WIDTH=32` with `--speculative-num-draft-tokens 32`
and no `FC1_DRAFT_BLOCK`, is still refused, as it was before the change. With
the fix the width-8 server started with `FC1_FUSED=1 [auto]` and gave the
results above.

Raw results: `results/thor_2026-11-19/sglang/head_88baa07`.
