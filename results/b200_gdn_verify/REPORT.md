# B200 · Qwen3.8-27B NVFP4: GDN verify tiles and projection views (2026-11-28)

Qwen3.8-27B NVFP4 (ModelOpt mixed precision) with the DFlash2 drafter on one B200 (SM100, 183,359 MiB,
driver 595.112.07). SGLang 0.5.23, torch 2.14.1+cu130, FlashInfer 0.7.2, Triton 3.8.0, FlashInfer cute-dsl
FP4 GEMMs, `trtllm_mha` attention, FlashInfer autotune on, 8,192-token prefill chunks, prefill CUDA graphs
on, the launch defaults of `results/b200_gdn_prefill` (one DFlash prefill admitted at a time, FlashInfer GDN
prefill). AX GDN replay 8 (`AX_GDN_CHAIN_REPLAY=1`, fp32 SSM state, SGLang's ReplaySSM ring verify and fold).
Stock vLLM 0.34.0rc1.dev1105+g2f170a0fe (DFlash2, 7 speculative tokens, FP32 Mamba state) on the same pod,
three servers.

Workload and pooling as in `results/h200_humming`: 128 greedy tokens, 8 prompts per workload at one client
and 24 at eight, two sweeps per server, the first sweep of every server dropped (both engines); n = servers.

## Where the eight-client round went

Torch-profiler traces of eight chat clients, per decode round (SGLang replay 8 against vLLM):

| Kernel group | SGLang | vLLM |
|---|---:|---:|
| GDN verify (ReplaySSM ring verify, Triton) | 1,878 µs (39 µs per layer) | 1,426 µs (CUDA MTP kernel) |
| GDN fold after acceptance | 376 µs | — |
| Verify conv1d with intermediate windows | 496 µs | 230 µs |
| GDN in_proj unpack copy (`fused_qkvzba_split_reshape_cat`) | 218 µs | — |
| `cache_indices >= 0` per layer (`compare_scalar<int>`) | 89 µs | — |

- **Ring verify.** With an fp32 state SGLang verifies through its generic Triton recurrent kernel on a
  32-wide value tile with one warp (255 registers). On its own at eight requests × eight draft tokens it
  ran 46 µs per layer; a 16-wide tile ran 33 µs. A Gluon kernel with an explicit K-per-thread layout
  (`gluon_verify.py`) reached 43 µs at eight requests and 9 µs at one, and a chunked closed form was
  either too slow in IEEE fp32 (80–700 µs) or too lossy in TF32 (relative error 3e-3); neither was kept.
- **Verify conv.** The deduplicated window view is written at a 20-byte channel stride. Reproduced in
  isolation at 11.8 µs; 64 channels on two warps ran 8.6 µs with bitwise-equal outputs, states and windows
  (`conv_bench2.py`).
- **Unpack copy and mask.** On target verify the in_proj output is copied into packed q/k/v, z, b and a,
  although every consumer on the fp32 Triton ReplaySSM route reads strides (prefill already uses views).
  The per-layer mask is evaluated eagerly and read only with pipeline-parallel stable rows. Codex traced
  both through the SGLang source; each consumer was checked against the source before the change.

## Changes

- `AX_GDN_VERIFY_TILES=1`: the ring verify on a 16-wide value tile, the window-saving verify conv on 64
  channels / two warps (SM100 only).
- `AX_GDN_VERIFY_VIEWS=1`: target verify reads strided views of the projection (only for the fp32-state
  GDN ReplaySSM fold) and the mask is built only when stable rows are on.
- `AX_NORM_FP4=1` (opt-in): the post-attention Gemma add-RMSNorm fused with gate_up's NVFP4 input
  quantization (vLLM's kernel, Apache-2.0). 3.6 µs per launch in the server, about 77 µs per round.

## Results

Output tok/s, code / math / chat. First token: p50 / p90.

| Eight clients | n | tok/s | Against vLLM | First token | Tokens per verify |
|---|---:|---:|---:|---:|---:|
| AX GDN replay 8, previous defaults | 3 | 2,540.0 / 2,705.6 / 1,570.3 | −5.5 / −6.7 / −7.2% | 0.064 / 0.089 s | 6.05 / 5.87 / 3.12 |
| + verify tiles + views (new defaults) | 4 | 2,757.7 / 2,796.6 / 1,724.9 | **+2.6** / −3.6 / **+2.0%** | 0.067 / 0.084 s | 6.03 / 5.84 / 3.07 |
| + tiles + views + `AX_NORM_FP4` | 6 | 2,708.8 / 2,755.4 / 1,745.6 | +0.7 / −5.0 / +3.2% | 0.066 / 0.081 s | 6.00 / 5.81 / 3.07 |
| + tiles + views + `AX_NORM_FP4`, `--min-free-slots-delay 2` | 2 | 2,832.1 / 2,653.1 / 1,835.4 | +5.3 / −8.6 / +8.5% | 0.070 / 0.118 s | 6.01 / 5.79 / 3.12 |
| + tiles + `AX_NORM_FP4`, `--enable-mixed-chunk` | 2 | 2,592.4 / 2,439.2 / 1,580.1 | −3.6 / −15.9 / −6.6% | 0.074 / 0.099 s | 6.33 / 5.88 / 3.16 |
| Stock vLLM DFlash2 | 3 | **2,688.8 / 2,901.2 / 1,691.4** | | 0.081 / 0.097 s | |

| One client | n | tok/s | Against vLLM |
|---|---:|---:|---:|
| AX GDN replay 8, previous defaults | 3 | 706.3 / 636.8 / 397.3 | −14.5 / −5.6 / −8.8% |
| + verify tiles + views (new defaults) | 4 | 755.0 / 667.1 / 428.6 | −8.6 / −1.1 / −1.6% |
| + tiles + views + `AX_NORM_FP4` | 6 | 783.8 / 688.4 / 438.0 | −5.1 / +2.1 / +0.6% |
| Stock vLLM DFlash2 | 3 | **826.2 / 674.3 / 435.5** | |

- **New defaults.** The tiles and views raised replay 8 by 9 / 3 / 10% at eight clients and 7 / 5 / 8% at
  one. Eight clients now lead vLLM on code and chat; math stays 3.6% behind.
- **Outputs.** The views and the conv tile did not change any one-client greedy output (24 of 24 equal
  with and without them). The ring tile sums over K in another order: 5 of 24 one-client outputs changed
  (2 math, 3 chat), tokens per verify 6.61 / 5.79 / 3.32 → 6.61 / 5.72 / 3.30. `AX_NORM_FP4` changed 18 of
  24 (it normalizes the fp32 sum, as vLLM does, where FlashInfer normalizes the rounded bf16 sum) and raised
  one-client code to 6.92 tokens per verify; at eight clients it did not move throughput beyond the
  server-to-server spread, so it stays opt-in.
- **Rejected.** `--enable-mixed-chunk` (SGLang allows it with DFlash, but running requests take a one-token
  decode in mixed steps) lost 11% on math and 7% on chat at eight clients against the same flags without it
  (three servers: 2,543.0 / 2,729.7 / 1,705.7). Admitting prefills in pairs (`--min-free-slots-delay 2`)
  moved math down and chat up, inside the spread, with a 0.12 s first-token p90. A full (unbroken) prefill
  CUDA graph needs fa3/fa4 prefill attention; with fa4 prefill and `trtllm_mha` verify the capture failed
  (`IndexError` in `PrefillCudaGraphRunner.capture_one_shape`).
- **Tree 32** (one server each, not pooled): one client +4 / +5 / +9% against vLLM with the flags, eight
  clients −1 / −17 / −9%, as before the change.
- **Spread.** Second-sweep eight-client code ranged 2,144–3,060 tok/s across these servers (the
  configurations differ by 2–10% when pooled); read the eight-client rows with that in mind.

## What remains: prefill steps

Plain decode rounds now cost the same as vLLM's (9.7 ms against 9.4 ms median span in a profile of 24 math
requests at eight clients), and so does the GPU time each prefill adds (10.2 ms against 9.7 ms,
`trace_pkern.py`). The difference is host time: each SGLang prefill step (EXTEND) ran 65 breakable-graph
segments and about 511 eager kernel launches, because SGLang's breakable prefill graph runs every
attention and linear-attention layer eagerly (`eager_on_graph`); a verify step is one graph launch. vLLM
folds the prefill into the next decode step. In the profile the prefill-containing segments spanned
695 ms for SGLang against 505 ms for vLLM over the same requests (both inflated by the profiler). Math has
the shortest requests (about 22 rounds against chat's 41), so it pays the most per token. Closing it needs
a graph-capturable GDN and attention prefill, not a launch setting.

## Launch defaults

On SM100, for a Gated-DeltaNet target with 128-dimension heads, the launcher now sets
`AX_GDN_VERIFY_TILES=1` for DFlash and tree launches and `AX_GDN_VERIFY_VIEWS=1` for DFlash launches (the
views were measured on the replay route only), when the user has not set them. Skipped with
`AX_AUTO_DEFAULTS=0` and with the parity profiles.

Files: `fin/` (passes, launch and environment JSON per server), `pg/*.log` (sweep and vLLM logs),
`pool.py`, `trace_prefill2.py`, `trace_pkern.py`, `conv_bench2.py`, `gluon_verify.py`.
