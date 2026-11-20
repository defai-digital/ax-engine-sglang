# AX Engine for SGLang

`ax-engine-sglang` is the SGLang add-on for AX Engine's Linux CUDA edition.
Deploy it on Linux amd64 (x86_64) or arm64 (aarch64) with a CUDA-enabled
SGLang environment and NVIDIA GPU. SGLang 0.5.23 is the reference release.
[Platform and model support](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/README.md) · [Side-by-side throughput](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/performance.md)

The Python distribution is `ax-engine-sglang`, the import is
`ax_engine_sglang`, and the SGLang plugin name is `ax-engine`. The plugin
integrates with the installed SGLang worker for DFlash-family speculative
verification. Its tree worker is generated at startup from installed
SGLang code and validates the expected upstream integration points.
[DForkTree attribution](https://github.com/defai-digital/ax-engine-sglang/blob/main/ax-engine-sglang/NOTICE)

## Install and run

Use Python 3.11 or 3.12 and a CUDA 13-compatible NVIDIA driver on Linux. In a
fresh virtual environment, run `python -m pip install ax-engine-sglang==0.3.0`
to install the add-on and SGLang 0.5.23. Use
`python -m pip install ./ax-engine-sglang` from a source checkout. Add
`--no-deps` when you already have the pinned SGLang environment. The Jetson
Thor smoke test used the 0.2.1 wheel in `lmsysorg/sglang:v0.5.23-cu130`.

```bash
ax-engine-sglang serve RadixArk/Qwen3.8-27B-NVFP4 \
  --speculative-algorithm FC1_DFLASH_TREE \
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 16
```

The command enables the plugin, passes the server flags to SGLang and stops
startup if AX fails to load. An existing `SGLANG_PLUGINS` allowlist retains
its entries. To use the upstream launcher directly, set `SGLANG_PLUGINS=ax-engine`.
On Jetson Thor, the installed wheel served a 32-token request with six tree
rounds and rejected a checkpoint setting mismatch before loading weights.
[Smoke test and exact launch flags](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/README.md#021-installed-wheel-smoke-on-jetson-thor)

## B200 benchmark results

Tested on SGLang 0.5.23 (and 0.5.22 for the single-request tree), one B200, greedy unless noted,
64 prompts per family, 1,024 tokens. Full records:
[dual attention backend](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/decisions/2026-11-16-dual-attention-backend.md),
[Flash-Next](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/decisions/2026-11-17-qwen38-flash-next-feasibility.md).
Jetson Thor (SM110, aarch64) is measured separately with 12 prompts and 512 tokens; see
[Jetson Thor](#jetson-thor).

| Model | Mode | Result vs. stock SGLang |
|---|---|---|
| Qwen3.8-27B NVFP4 | tree at 1 request, chain width 8 / 16 above (32-64 slots) | c = 1: code 1.42x, math 1.33x, chat 1.14x; c = 2-64: 1.01x-1.21x |
| Qwen3.8-27B bf16 | same (16 slots) | c = 1: 1.09x-1.15x vs. the faster stock block; c = 2-16: 0.93x-1.09x |
| Qwen3.8-27B, both | vs. TensorRT-LLM 1.4 DFlash2 | c = 1: 1.07x-1.72x; c = 16 chat 0.81x-0.90x (engine round cost) |
| Qwen3.8-Flash-Next NVFP4 | stock MTP 7 + `AX_QSA_RING=16` vs. stock MTP 3 | code 1.22x-1.30x, math 1.05x-1.15x, chat 0.80x-0.86x |

Correctness checks: greedy token identity against the plugin's earlier builds (192 / 192), fp32
teacher audit at the stock mismatch rate, stress runs with aborts, sampled tree verify matching plain
decoding on 28 / 28 distribution tests (bf16), QSA ring consistency check (Flash-Next).

Limits:
- Tree rounds serve one greedy request without grammar or logprobs. Every other batch, and one
  sampled request unless `AX_TREE_SAMPLING=1`, runs a stock DFlash2 chain round.
- Tested targets: Qwen3.5-9B and Qwen3.8-27B (Gated-DeltaNet hybrids). On Qwen3.8-Flash-Next
  (Qwen4-Exp) only the QSA ring fix applies; SGLang does not expose its aux hidden states (needed by
  DFlash2) and the tree needs the PLE n-gram context.
- MTP 12 on Flash-Next needs `--max-running-requests 32` (the GDN state pool grows with the draft
  length). Wider MTP is slower on chat; choose the step count per workload.

## Launch defaults

On SGLang 0.5.23 the plugin fills in the settings below while SGLang resolves the server arguments,
so the command above runs the measured configuration. The launcher prints every value with its
origin (`auto` or `env`). Explicit values take priority. A mismatch with the
checkpoint stops the launch and reports the reason. `AX_AUTO_DEFAULTS=0` turns
off these defaults. For an existing SGLang 0.5.22 environment, use `--no-deps`
and supply the settings in the table below explicitly.

| Applies to | Default | Derived from |
|---|---|---|
| `FC1_DFLASH_TREE` | `AX_GDN_LAYERS`, `AX_GDN_HEADS` | the target's `layer_types` and `linear_num_value_heads` (48 / 48 for Qwen3.8-27B, 24 / 32 for Qwen3.5-9B) |
| `FC1_DFLASH_TREE` | `FC1_DRAFT_BLOCK` | `--speculative-num-draft-tokens`, else the drafter's `dflash_config.block_size` |
| `FC1_DFLASH_TREE` | `AX_LATTICE=1` | a DFlash2 drafter; a DFlash (v1) drafter needs `FC1_NGRAM_PATH` (checked at launch) |
| `FC1_DFLASH_TREE` | `FC1_TREE64=1 FC1_BRANCH_PILOT=1 FC1_OURS=1 FC1_FUSED=1 FC1_COMPILE_BUILD=1 AX_FAST=1 FC1_TREE_N=32` | the measured tree configuration |
| `FC1_DFLASH_TREE` | `--max-running-requests 1 --cuda-graph-max-bs-decode 1` | only when `--max-running-requests` is not given; set it (and `AX_CHAIN_*`) for multi-user chain rounds |
| Qwen4-Exp target with any speculative algorithm | `AX_QSA_RING=16` | the stock 4-slot ring is wrong for verify windows that cross a group and refuses windows above 4 |
| NVFP4 MoE target on SM110 (Jetson Thor) / SM120 | `--moe-runner-backend flashinfer_cutlass`, and `--speculative-moe-runner-backend` with speculative decoding | only when left at `auto`: SGLang 0.5.23 picks TRT-LLM (SM100 only) |

The tree is refused on Qwen4-Exp targets. If the target's or drafter's `config.json` cannot be read (offline, a wrong revision), a tree launch stops instead of guessing: set `AX_GDN_LAYERS` and `AX_GDN_HEADS`, or `AX_LATTICE` (`1` for DFlash2, `0` for DFlash v1), explicitly. It also stops when neither `--speculative-num-draft-tokens`, the drafter config nor `FC1_DRAFT_BLOCK` gives the draft block length. The published DFlash2 drafters list `block_size` 8; the measured lanes draft 16 tokens per round, so keep `--speculative-num-draft-tokens 16` (the launcher prints a note about the difference). For a stock speculative route (MTP) with an unreadable target config, the launcher notes that the NVFP4 MoE backend and `AX_QSA_RING` defaults were not derived.

| Variable | Meaning |
|---|---|
| `FC1_TREE_N` | nodes verified per round, root included (32 measured best on B200 and 64 on RTX PRO 6000 for single-request 27B NVFP4). 16, 32 and 64 use the plugin's host-length FlashInfer plan; other sizes use SGLang's own plan |
| `FC1_DRAFT_BLOCK` | tokens drafted per round (16 on the 9B and the 27B; at most 32 with `FC1_TREE64=1`) |
| `AX_LATTICE=1` | build the tree from the DFlash2 selector lattice (needs a DFlash2 drafter) |
| `AX_FAST=1` | per-round overhead fixes (no host readback of the prefix length) |
| `AX_GDN_LAYERS`, `AX_GDN_HEADS` | Gated-DeltaNet shape of the target: 48 / 48 for Qwen3.8-27B, 24 / 32 for Qwen3.5-9B |
| `FC1_FUSED=1` | fused tree verifier; `0` uses SGLang's own tree verify (slower, a reference) |
| `FC1_NGRAM_PATH` | n-gram table of the DForkTree release, used to score trees with a DFlash (v1) drafter; lattice trees (`AX_LATTICE=1`) do not use it. The measured runs set it, so the example does too |
| `AX_CHECK=1` | validation only: checks the mask and length bound every round (synchronous, slow) |
| `AX_QSA_RING=16` | Qwen4-Exp (Qwen3.8-Flash-Next) QSA sparse attention: widen the per-request pending index-key ring from 4 to 16 slots. Fixes compressed index keys built from overwritten ring members whenever a verify window crosses a 4-token group (stock SGLang 0.5.23, MTP 1-3 included) and allows verify windows up to 13 tokens (stock refuses more than 4). Works with stock MTP (`--speculative-algorithm EAGLE`); the tree is not supported on Qwen4-Exp |
| `AX_QSA_CHECK=1` | validation only: counts QSA groups compressed from ring members that do not hold their group's positions (eager, `--disable-cuda-graph`) |

B200 multi-user server for the 27B (tree at one request, chain rounds of width 16 / 8 with their own
CUDA graphs on `trtllm_mha`): see `benchmarks/sglang/run_stage1a.sh` for the full flag set
(`AX_CHAIN_ATTN=trtllm_mha AX_CHAIN_GRAPH_WIDTHS="16 8" AX_CHAIN_DRAFT_WIDTHS=8`, 16 slots bf16 or 32
NVFP4, prefill on `trtllm_mha`, tree verify on `triton`). Set `HYBRID=0` to use Triton
for the target, drafter and private chain runners.

Qwen3.8-Flash-Next with stock MTP and the ring fix (`AX_QSA_RING=16` is the launch default for
Qwen4-Exp; it is spelled out here because SGLang 0.5.22 needs it):

```bash
AX_QSA_RING=16 ax-engine-sglang serve RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-algorithm EAGLE --speculative-num-steps 7 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 8
```

The measurement lanes `axtree<spec>` (venv SGLang) and `axotree<spec>` (0.5.22 overlay) in
`benchmarks/sglang/run_tree_plugin.sh` run the standalone plugin's DFlash tree configuration.
The `axtree` lanes use `AXSITE` and the venv. Lattice lanes use DFlash2 scores
and can run without an n-gram table.

## RTX PRO 6000

For single-request Qwen3.8-27B NVFP4 serving, the fastest setting in the
128-token sweep used a 64-node tree, a draft block of 16 and verification
parity disabled. It reached **156.87 output tok/s** at **0.818 s median
request latency**. At eight serving slots, set `AX_CHAIN_WIDTH=8`,
`AX_CHAIN_GRAPH_WIDTHS=8`, `AX_CHAIN_DRAFT_WIDTHS=8` and `AX_SSM_POSITIONS=8`
to reproduce the measured cache allocation. Tree 64 matched packed AR exactly
on one of ten complete outputs per pass.

At concurrency eight, stock `DFLASH` with `--speculative-num-draft-tokens 8`
averaged **489.86 aggregate output tok/s** over two 40-request passes. Plugin
tree 32 averaged **484.78 tok/s** with width-eight chain rounds for grouped
requests. Packed AR reached **388.29 tok/s**.
[Launch commands and all ten measured configurations](https://github.com/defai-digital/ax-engine-sglang/blob/main/results/rtx_27b_nvfp4_speed/REPORT.md)

## Jetson Thor

Jetson Thor (SM 11.0, aarch64, 122 GB unified memory) runs the same commands as above on
`lmsysorg/sglang:v0.5.23-cu130`; the launch defaults detect SM110. Measured single-request output
throughput (greedy, 12 prompts, 512 tokens; 27B bf16: 8 prompts, 384 tokens), code / math / chat:

| Target | AR | Stock chain 16 | ax-engine-sglang tree |
|---|---|---|---|
| Qwen3.5-9B bf16 | 14.8 / 14.9 / 14.9 | 118.7 / 78.0 / 31.3 | 125.5 / 98.2 / 46.2 |
| Qwen3.8-27B NVFP4 | 13.4 / 13.5 / 13.5 | 93.0 / 73.5 / 30.6 | 106.1 / 92.1 / 43.1 |
| Qwen3.8-27B bf16 | 4.6 / 4.6 / 4.6 | 43.1 / 29.6 / 12.5 | 47.5 / 37.6 / 18.3 |

All three rows use FlashInfer attention. Full tables (Triton, vLLM, MTP, teacher checks) and the
two-host Flash-Next run are in the [Thor report](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-19-jetson-thor.md).

Settings that differ from a discrete GPU:

- `--mem-fraction-static` is a fraction of the memory free at start-up, and CUDA's free figure on
  Jetson excludes page cache. Drop the page cache before starting the server
  (`sync; echo 3 > /proc/sys/vm/drop_caches` as root) and pass
  `--mem-fraction-static 0.70 --chunked-prefill-size 2048` for the 9B and the 27B NVFP4, `0.80`
  for the 27B bf16. With other services holding about half of the memory, `0.70` still serves the
  9B and the 27B NVFP4 tree (about 87k KV tokens on the 27B NVFP4).
- Stock MTP on the 27B bf16 also needs `--max-running-requests 4`: the Gated-DeltaNet state pool
  for the default request count leaves no KV cache.
- Qwen3.8-Flash-Next NVFP4 (about 148 GB) does not fit one Thor. Two hosts serve it with SGLang
  pipeline parallelism (`--pp-size 2 --nnodes 2 --dist-init-addr <rank-0 host>:29500
  --node-rank {0,1}`, `SGLANG_PP_LAYER_PARTITION=4,44`, `--mem-fraction-static 0.90`,
  `--max-running-requests 4`, and `SGLANG_ENABLE_PP_SPEC=1` for MTP), each host loading a stage
  directory that lists only its own shards. Without InfiniBand, set `NCCL_IB_DISABLE=1` and
  `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` to the Ethernet interface. The report gives the
  coordinator script. Flash-Next bf16 does not fit two Thors.
- NVFP4 targets: a fresh prefill and AR decoding disagree on about 3% of tokens on SM110, for
  stock AR as well, so greedy identity and the teacher check cannot qualify NVFP4 routes on Thor.
  Qualify exact-output requirements on bf16.

Also measured on Thor with the 9B: sampled tree verify (`AX_TREE_SAMPLING=1`) matched stock AR
on 28 of 28 distribution tests, and a server with `--max-running-requests 4` passed the mixed
request and stress checks (2,000 requests, 178 cancelled) at the stock DFlash rate for two and
four requests. Not qualified on Thor: `AX_EAGLE_JOINT_SAMPLING` and `AX_DFLASH_JOINT_SAMPLING`
on a served model (their kernel-level checks pass on SM110), and `AX_QSA_STABLE_TOPK` on a
served model. All `tests/gpu_*.py` checks pass on SM110; the NVFP4 parity check is SM120 only and
skips.

## Maintenance and validation

Startup parses and validates the tree/chain configuration before installing
upstream hooks. QSA/RNG fixes can load without constructing a DFlash worker;
tree hooks load when their flags or the custom algorithm request them. Supported
SGLang versions are 0.5.22 and 0.5.23. `AX_ALLOW_UNTESTED_SGLANG=1` permits an
explicit porting experiment; exact source-anchor checks still apply.

Round policy is pure Python in `policy.py`. `runtime.py` keeps round decisions,
planning lengths and fused GDN buffers on each worker. Startup patches use
`compat.py` to restore replaced attributes on failure. SSM allocation and budget
hooks are compiled together before either replacement is published.
Tree rounds read fresh logits or a lattice from the drafter's sampler folded
into its CUDA graph. Eager drafts and draft graph misses use chain verification
for the round, including when the server was launched with the tree algorithm.
When chain verification keeps a prefix of a wider draft, it keeps the sampled
candidate IDs and proposal probabilities for those same positions.

The package owns `ax_fast.py` and `ax_lattice.py`; `benchmarks/sglang/dft_*.py` preserve the
older research imports. Legacy DForkTree overlays copy the package-owned files.
Shared commit fixtures and benchmark evidence schemas live in `../contracts/`;
see [maintenance boundaries](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/decisions/2026-11-17-maintenance-boundaries.md).

Run the CPU checks from the repository root. Install `setuptools>=77`, `wheel`,
`scipy` and `jsonschema` in the test environment, as in the CPU workflow.

```bash
python -m unittest discover -s ax-engine-sglang/tests -v
python scripts/check_engine_contracts.py --validate-schema
python scripts/check_engine_contracts.py --peer /path/to/ax-engine
python benchmarks/sglang/test_dft_lattice.py --plugin ax-engine-sglang/ax_engine_sglang
python benchmarks/sglang/test_paired_ci.py
```

CUDA checks include `benchmarks/sglang/test_dft_fast.py`, `benchmarks/sglang/test_tree_gdn.py`, and
`ax-engine-sglang/tests/gpu_runtime_check.py` (its header lists the flags).
`gpu_ple_offload_check.py` checks pinned-host PLE reads and CUDA replay, including
large offsets. `gpu_qsa_diagnostic_check.py` checks multimodal QSA diagnostics
against real ring-overwrite and RoPE-store negative controls.
`gpu_nvfp4_moe_check.py` checks SM110 (Jetson Thor) / SM120 CUTLASS weight preparation, nonzero MoE
execution and changed-input graph replay before loading Flash Next weights.
`scripts/check_model_regression.py --help` describes the bounded comparison
against an earlier checkout with local target/drafter checkpoints. It checks
greedy identity, mixed request types and post-request state reuse. Optional
`--stress-requests`, `--sampling-n`, `--sampling-seed` and `--long-context` controls
exercise cancellation, sampled distributions and longer contexts. The stress
check requires matching idle pool usage and an observed abort when cancellation
is enabled. Use
`--candidate-only --route ar|mtp3|mtp7|mtp12` for Flash Next, with a local model
and `--model-override` for pinned FP8 PLE on a 96 GB card. `--qsa-ring 16
--qsa-check --eager` enables the synchronous ring diagnostic. The runner refuses
Flash Next tree verification until PLE/QSA tree support exists. The RTX
development checks cover correctness and lifecycle behavior. See the
[RTX validation plan](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/decisions/2026-11-17-rtx-development-validation.md)
and [installed-wheel QA results](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-18-development-qa.md).
The independent RTX BF16 distribution control passes; the NVFP4 AR/chain
comparison fails and remains outside cross-route lossless qualification.

For Flash Next NVFP4 on SM110 (Jetson Thor) and SM120 the runner must be
`flashinfer_cutlass` for the target and the MTP draft. SGLang 0.5.23's automatic
NVFP4 MoE selection can prepare a different weight layout from the runner it
later chooses (`g1_scale_c` missing during graph warmup). The plugin's launch
defaults select CUTLASS when both are left at `auto`; benchmark harnesses still
pass the backends explicitly so they appear in the saved launch command.

Sampling defaults to the server RNG. Paired comparisons use `--server-seed` for
the baseline and the next seed for the candidate, and enforce
`sampling_dist_check.py compare A B --require-independent-seeds`. A single lane
uses the requested server seed unchanged and records
`sampling_comparison: not_run_single_lane`. Collection sidecars record the
effective seed mode, tokenized prompt hashes, and a hash of the completed sample file.
Request-seed collection requires `--candidate-only` and
`--enable-deterministic-inference`; correlated request-seed cohorts cannot be
used for the independent two-sample test. SGLang otherwise ignores that request
field. The distribution gate requires at least 10 samples per prompt in each
cohort and reports an inconclusive result when low expected counts force all
observed differences into one cell.

For failure diagnosis, `--repeat-reference` checks reference stability before
stress, and stress records retain the before/after token arrays.
`--candidate-only --keep-server-on-failure` keeps an owned failed server resident
and writes its process-group stop command to `retained-server.json`. Each model
run also saves a copy of the harness source for replay.

`AX_QSA_STABLE_TOPK=1` (runner: `--qsa-stable-topk`) selects a fixed-shape,
stable tensor-sort fallback for QSA. It supports CUDA graph replay on SM120,
where the native deterministic FilteredTopK path needs more shared memory than
the device provides. Equal scores select the smaller relative block index.
`gpu_qsa_topk_check.py` covers ragged rows, ties, replay and hook rollback (it also passes on
SM110; a served Flash-Next model with this option was not run on Jetson Thor).
This is an opt-in correctness control; its throughput is not qualified.

`AX_EAGLE_JOINT_SAMPLING=1` matches AR's FlashInfer top-k/top-p filter during
target-only EAGLE verification on SGLang 0.5.23. Both filters use the original
probabilities. The option applies when both cutoffs are active and min-p is
disabled. NVFP4 model logits can still differ between AR and block verification.
`gpu_eagle_sampling_check.py` checks the filter against FlashInfer's AR sampler.

`AX_DFLASH_JOINT_SAMPLING=1` (runner flag `--dflash-joint-sampling`) applies
FlashInfer AR's joint top-k/top-p filter to target probabilities in
`FC1_DFLASH_TREE` chain rounds on SGLang 0.5.23. It applies when both cutoffs
are active and min-p is disabled. Rejection sampling retains the drafter's
proposal probabilities, and sampling-mask capture uses the same target filter.
`gpu_dflash_sampling_check.py` checks the target support and frequencies against
the FlashInfer AR sampler on CUDA.

For the B200 Qwen3.8-27B NVFP4 sampled profile, use `--attention-backend
flashinfer --fp4-gemm-backend flashinfer_cutlass` with checkpoint-selected FP8
KV. Enable `AX_GDN_UNPACKED_DECODE=1` on the AR reference and
`AX_DFLASH_JOINT_SAMPLING=1` on the DFlash2 service with chain width 8. The
profile passed all 28 distribution checks at the specified cutoff. AR and
DFlash2 produced identical greedy token sequences for two of seven prompts.
[B200 retest and commands](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-18-b200-27b-nvfp4.md) ·
[Greedy token identity and logit example](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-18-b200-27b-nvfp4.md#greedy-follow-up)

The RTX PRO 6000 retest passed real NVFP4 serving and kernel checks. Recurrent
GDN tree and chain each matched four of seven greedy outputs against RTX AR.
[RTX PRO 6000 model and kernel results](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-19-rtx-27b-nvfp4.md)

For the dense Qwen3.8-27B NVFP4 checkpoint on RTX PRO 6000, the regression
runner accepts `--nvfp4-verify-parity` for AR, DFlash2 tree and chain.
It selects FlashInfer paged attention, CUTLASS NVFP4, unpacked GDN
and target projections in fixed groups of rows. The serving Python must be able
to import `cutlass.cute` from NVIDIA CUTLASS DSL. Use the option on both sides
of a greedy comparison, because it changes the AR numerical reference.
`--nvfp4-parity-rows` (`AX_NVFP4_PARITY_ROWS`, default 16) sets the group size
and must also match on both sides. A value of 1 runs verification one row at
a time. [Verification controls](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-19-rtx-nvfp4-parity.md) · [Row group results](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-20-rtx-nvfp4-parity-row-groups.md)

On the Flash Next AR reference server, `AX_GDN_UNPACKED_DECODE=1` selects
unpacked Triton Gated DeltaNet decode before CUDA graph capture. To reproduce
the AR profile through the plugin, set `SGLANG_PLUGINS=ax-engine` and
`AX_FI_OFFSET_FIX=0`. Both MTP7 configurations passed the 28 distribution checks
against the unpacked AR reference. The packed AR comparisons still fail.
[B200 sampling and throughput results](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-18-b200-eagle-sampling.md)

To avoid reloading unrelated target shards for Flash Next MTP, derive a compact
snapshot outside the repository with `scripts/prepare_flash_mtp.py --help`,
then pass it as `--mtp-draft`. The tool selects the loader's MTP/embedding names,
preserves model configuration and verifies every tensor's SHA-256 after writing.
It records the declared source revision in `derivation.json` and does not
requantize weights.

The [main deployment and logit report](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-18-main-deployment-and-nvfp4.md)
records clean-wheel GPU smokes and the unresolved NVFP4 AR/block-verification
differences on RTX PRO 6000, including native DFlash controls. The
[27B NVFP4 B200 retest](https://github.com/defai-digital/ax-engine-sglang/blob/main/docs/reports/2026-11-18-b200-27b-nvfp4.md) records
the follow-up attention and sampling controls. `scripts/nvfp4_forward_probe.py`
provides opt-in SGLang forward-hook factories and a matched-prefix analyzer.
Use eager execution and a single greedy request in flight; it rejects tree positions,
rejected speculative histories and incomplete/mismatched prompt sets. Its
synchronous replay probes preserve model outputs and are unsuitable for timing.
