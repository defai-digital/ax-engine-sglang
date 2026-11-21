# RTX NVFP4 parity and speed profiles at a fixed 128 tokens

The parity and speed profiles ran on one RTX PRO 6000 with the speed sweep's
protocol: every request returned exactly 128 tokens (`ignore_eos=true`). Each
parity lane equalled parity AR on all ten inputs and equalled its own outputs at
concurrency one when eight ran together. No speed-profile lane did, including
SGLang's AR without the plugin.

With the plugin's parity defaults, tree 16 reached 118.09 tok/s alone and
493.03 tok/s at concurrency eight. Speed-profile tree 16 reached 126.59 and
449.86 tok/s, with features matched. One request alone, parity was 7% slower; at
concurrency eight it was 10% faster.

Tree 64 now runs with eight request slots under the parity profile. It reached
508.50 tok/s at concurrency eight; two slots had given 179.08 tok/s.

## Lanes

Every lane ran once, from a fresh server. Each pass sent the ten frozen inputs
at concurrency one (C1), then twice at concurrency eight (C8), then C1 again.
"Same as parity AR" compares the first C1 pass with parity AR's first C1 pass
in the same group. "C8 same as C1" counts requests in each C8 pass equal to
that lane's first C1 pass.

### Plugin parity defaults

Radix cache and FlashInfer autotune off, prefill graphs on. The launch command
had no disable flags; the defaults added them.

| Lane | C1 tok/s | C8 tok/s | Same as parity AR | C8 same as C1 |
|---|---:|---:|---:|---:|
| Parity AR | 51.79 | 226.78 | reference | 10, 10 |
| Parity tree 16 | 118.09 | 493.03 | 10/10 | 10, 10 |
| Parity tree 32 | 120.05 | 520.60 | 10/10 | 10, 10 |
| Parity chain 8 | 109.76 | 507.35 | 10/10 | 10, 10 |
| Parity tree 64, 8 slots, `AX_SSM_POSITIONS=8` | 102.34 | 508.50 | 10/10 | 10, 10 |

### Speed sweep flags

Radix cache, FlashInfer autotune and prefill graphs off. Speed-profile settings
as in the [published sweep](../../results/rtx_27b_nvfp4_speed/REPORT.md).

| Lane | C1 tok/s | C8 tok/s | Same as parity AR | C8 same as C1 |
|---|---:|---:|---:|---:|
| Parity AR | 50.29 | 222.79 | reference | 10, 10 |
| Parity tree 16 | 112.70 | 387.57 | 10/10 | 10, 10 |
| Parity tree 32 | 109.81 | 383.50 | 10/10 | 10, 10 |
| Parity chain 8 | 101.73 | 471.55 | 10/10 | 10, 10 |
| Parity tree 64, 2 slots | 96.88 | 179.08 | 10/10 | 10, 10 |
| Parity tree 64, 8 slots, `AX_SSM_POSITIONS=8` | 96.93 | 471.75 | 10/10 | 10, 10 |
| Parity tree 16, `AX_SSM_POSITIONS=8` | 111.76 | 410.94 | 10/10 | 10, 10 |
| Speed AR, plugin excluded | 52.31 | 234.11 | 0/10 | 0, 0 |
| Speed tree 16 | 126.59 | 449.86 | 1/10 | 0, 0 |
| Speed tree 64, `AX_SSM_POSITIONS=8` | 139.58 | 404.96 | 1/10 | 0, 0 |

The published sweep recorded speed tree 64 at 156.87 tok/s and AR at 54.41 on
another pod. Compare rates within one pod.

### Which SGLang feature breaks matched outputs

| Prefill graphs | FlashInfer autotune | Lane | C1 tok/s | C8 tok/s | C8 same as C1 |
|---|---|---|---:|---:|---:|
| on | off | Parity AR | 51.72 | 240.24 | 10, 10 |
| on | off | Parity tree 16 | 122.24 | 510.27 | 10, 10 |
| off | on | Parity AR | 58.80 | 256.11 | 8, 8 |
| off | on | Parity tree 16 | 114.51 | 467.33 | 8, 8 |
| on | on | Parity AR | 60.98 | 270.79 | 9, 9 |
| on | on | Parity tree 16 | 128.32 | 382.54 | 8, 8 |
| on | on | Parity tree 32 | 130.85 | 485.96 | 8, 9 |
| on | on | Parity chain 8 | 117.34 | 526.01 | 8, 8 |
| on | on | Speed AR | 65.56 | 283.21 | 0, 0 |
| on | on | Speed tree 64 | 149.73 | 523.29 | 1, 1 |

Autotune chooses GEMM tactics by row count, so a request batched with others can
take other kernels. Prefill graphs alone kept every output. The earlier report's
30/30 result with autotune on stopped at end-of-sequence and compared shorter
sequences; with a fixed 128 tokens, autotune broke one or two of ten.

## Changes

**Parity defaults.** `AX_NVFP4_VERIFY_PARITY=1` disables the radix cache and
FlashInfer autotune. Prefill graphs stay at SGLang's default.

**Tree 64 with eight slots.** With `AX_SSM_POSITIONS=P`, the GDN snapshot buffer
holds P positions for each slot plus one spare row. A tree round verifies one
request, and its snapshots are only read until that round commits. The round
now uses the whole buffer as one row: eight slots of eight positions give 72,
enough for 64 candidates. Verification and SGLang's commit scatter both read
this view. A round with several requests, or a tree larger than the buffer,
stops with an error. `AX_SSM_POSITIONS` is accepted with
`AX_GDN_RECURRENT_VERIFY=1` as well as `FC1_FUSED=1`.

The GPU check runs SGLang's verify kernel on a nine-by-eight buffer for trees
of 16 and 64 candidates, then commits one candidate with SGLang's scatter.
Outputs, snapshots and the committed state equalled independent unpacked decode
in FP32 and BF16.

## Where parity time goes

PyTorch profiler, one C1 request of 128 tokens after two warm-ups, prefill
graphs off. GPU kernel time per request:

| Kernel group | Parity tree 16 | Speed tree 16 | Parity tree 64 |
|---|---:|---:|---:|
| NVFP4 GEMM (CUTLASS) | 220.2 ms | 222.1 ms | 181.8 ms |
| GDN gated norm (`_layer_norm_fwd_1pass_kernel`) | 50.6 ms | fused | 151.8 ms |
| GDN verify (`fused_sigmoid_gating_delta_rule_update`) | 59.0 ms | tree kernels | 149.5 ms |
| BF16 GEMM, batch-invariant deep_gemm | 90.7 ms | 44.1 ms in cuBLAS | 61.1 ms |
| FP8 GEMM (`torch._scaled_mm`) | 101.2 ms | 106.6 ms | 99.2 ms |
| All GPU kernels | 602.3 ms | 463.6 ms | 748.7 ms |

NVFP4 and FP8 cost the same in both profiles. The difference is GDN: the
parity path reloads a 48×128×128 state per candidate and normalises every
candidate separately, while the speed path fuses these into its tree kernels.
Tree 64 spends 301 ms of 749 ms there. The candidate depth is not the limit:
a version that decoded all candidates at one tree depth per launch matched bit
for bit but took 57.7 ms per round across 48 layers, against 12.0 ms for the
one-launch kernel, and was dropped.

BF16 GEMM rows differ with the row count under cuBLAS on this GPU (draft
projections from 2 rows, `in_proj_ba` from 8). Batch-invariant mode keeps them
equal at the same speed for the large projections and 1.5× slower for
`in_proj_ba`.

## Conditions

- One NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), RunPod US-PA-1,
  SGLang 0.5.23, PyTorch 2.14.1+cu130, FlashInfer 0.7.2.
- Target RadixArk/Qwen3.8-27B-NVFP4 at `319f741cce68d7914884900c138a1fbb70a42f30`,
  BF16 DFlash2 draft at `015e795645c74b1a0eeef3b570031fb62e769bc5`, 16 draft tokens.
- FlashInfer attention, flashinfer_cutlass NVFP4 GEMM, 8 request slots unless
  noted, context 4,096, KV pool 16,384 tokens, memory fraction 0.8, greedy,
  server seed 42. Tree lanes set chain width 8 for grouped requests.
- Inputs: the ten frozen prompts in `parity-expanded-ids.json` (23–2,266 tokens).
  Warm-up: one request, then eight concurrent 32-token requests. Rates include
  prefill and local HTTP and exclude startup.
- Local results: `results/rtx-nvfp4-parity-forced-128/` (per-lane launch
  settings, server logs, output IDs, profiler summaries, and the drivers).
