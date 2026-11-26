# NVFP4 GEMM backend on the RTX PRO 6000: CUTLASS vs cute-dsl (2026-11-26)

Qwen3.8-27B NVFP4 (ModelOpt mixed precision: the MLP is NVFP4, attention and Gated-DeltaNet
projections FP8) with the DFlash2 drafter on one RTX PRO 6000 Blackwell Server Edition (SM120, 96 GB,
driver 595.112.07), SGLang 0.5.23, torch 2.14.1, FlashInfer 0.7.2, nvidia-cutlass-dsl 4.8.0,
ax-engine-sglang at 5cfd3c6. On SM120 SGLang's `--fp4-gemm-backend auto` resolves to
`flashinfer_cutlass` (only SM100 resolves to `flashinfer_cutedsl`).

## GEMM microbenchmark

`benchmarks/sglang/nvfp4_gemm_backends.py`: `flashinfer.mm_fp4` on the 27B MLP shapes, CUDA graphs of
50 GEMMs cycling through weight copies of four times the 128 MB L2 cache, median of 15. Microseconds
per GEMM; `trtllm`, `cutedsl_low_latency` and `cake` do not support SM120. Every backend's eager and
CUDA-graph outputs were bit-identical.

| Rows | gate+up CUTLASS | cuDNN | b12x | cute-dsl | down CUTLASS | cuDNN | b12x | cute-dsl |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 96.9 | 96.1 | 69.2 | **67.6** | 66.3 | 58.6 | 41.7 | **34.8** |
| 16 | 97.7 | 102.6 | 69.5 | **68.0** | 66.3 | 50.1 | 41.7 | **35.1** |
| 32 | 89.4 | 94.6 | **69.0** | 69.2 | 66.2 | 50.2 | 41.7 | **40.2** |
| 64 | 74.0 | 74.0 | **69.0** | 69.7 | 66.2 | 50.4 | 41.7 | **38.0** |
| 128 | 74.1 | 82.7 | 72.9 | **72.9** | 66.2 | 50.9 | **41.9** | 69.1 |

gate+up plus down, cute-dsl against CUTLASS: -37% at 8 and 16 rows, -30% at 32, -23% at 64, +1% at 128.
b12x is fastest at 128 rows (-18%) but SGLang 0.5.23 cannot select it. A first run that replayed one
weight copy kept it in L2 and reported cute-dsl above the card's memory bandwidth; those numbers were
discarded.

## Served A/B

`benchmarks/sglang/hopper_tree_sweep.py` with `AX_SWEEP_FP4_BACKEND`, the 128-token retest workload
(8 prompts per workload at C1, 24 at C8, two sweeps), lanes interleaved CUTLASS / cute-dsl per route.
Output tok/s, code / math / chat. C1 pools both sweeps; C8 is the second sweep (see cold start).

| Route | Clients | CUTLASS | cute-dsl | Change |
|---|---:|---:|---:|---:|
| AX tree 32 | 1 | 292.0 / 250.8 / 154.0 | **310.1 / 267.3 / 164.6** | +6.2 / +6.6 / +6.9% |
| AX tree 32 | 8 | 789.1 / 665.5 / 443.3 | **822.1 / 745.1 / 455.5** | +4.2 / +12.0 / +2.8% |
| AX GDN replay 8 | 1 | 216.4 / 180.0 / 121.8 | **249.8 / 208.9 / 143.6** | +15.4 / +16.1 / +17.9% |
| AX GDN replay 8 | 8 | 862.7 / **772.6** / 520.2 | **896.8** / 748.3 / **546.0** | +4.0 / -3.1 / +5.0% |
| Stock DFlash2 width 8 | 1 | 203.0 / 167.6 / 114.2 | **238.0 / 199.3 / 135.7** | +17.2 / +18.9 / +18.8% |
| Stock DFlash2 width 8 | 8 | 738.0 / 659.2 / 441.0 | **819.3 / 722.5 / 482.0** | +11.0 / +9.6 / +9.3% |

A one-client verify round took 25.4 → 21.1 ms (stock width 8), 24.1 → 19.9 ms (replay) and 25.9 → 23.5 ms
(tree 32), close to the microbenchmark's 3.7 ms (8 rows) and 3.0 ms (32 rows) per 64 MLP layers.
Tokens per verify were equal at one client. Greedy outputs were identical across the backends on
24/24 one-client requests of every route; at eight clients 59-66 of 72 matched (batch composition
changes GEMM shapes between runs, as with either backend alone).

**Cold start.** cute-dsl compiles kernels the first time it meets a shape. The first cute-dsl server
(tree 32) ran its first eight-client pass at 372.5 / 409.0 tok/s on chat / code with a 3.85 s worst
first-token time, against 0.5-1.0 s elsewhere; later servers reused the on-disk cache and showed no
such pass. Warm a new deployment before measuring or serving latency-sensitive traffic.

## RTX 5090

One GeForce RTX 5090 (SM120, 32 GB), same software, the add-on's working tree after 1cd283e.
`hopper_tree_sweep.py` with the 5090 server shape: two slots, 4,096 KV tokens, static memory fraction
0.94, `AX_SSM_POSITIONS=8`, `SGLANG_FLASHINFER_WORKSPACE_SIZE=67108864`
(`AX_SWEEP_SLOTS=2 AX_SWEEP_KV_TOKENS=4096 AX_SWEEP_MEM=0.94 AX_SWEEP_CONCURRENCY=1,2`). Same
128-token workload with 8 prompts per workload at one request and 24 at two. Two rounds of servers,
interleaved CUTLASS / cute-dsl per route, two sweeps per server. Cells pool every sweep except the
first sweep of the first round, where the first cute-dsl server compiled its kernels (tree 32 math at
one request: 190.3 tok/s in that sweep, 270-282 afterwards).

| Route | Requests | CUTLASS | cute-dsl | Change |
|---|---:|---:|---:|---:|
| AX tree 32 | 1 | 321.2 / 269.9 / 169.7 | **333.3 / 281.0 / 177.9** | +3.8 / +4.1 / +4.9% |
| AX tree 32 | 2 | 349.1 / 334.6 / 208.5 | **377.1 / 361.9 / 230.2** | +8.0 / +8.2 / +10.4% |
| Stock DFlash2 width 8 | 1 | 242.4 / 202.9 / 130.9 | **271.7 / 229.5 / 149.3** | +12.1 / +13.1 / +14.0% |
| Stock DFlash2 width 8 | 2 | 323.3 / 325.0 / 202.4 | **351.7 / 354.3 / 224.4** | +8.8 / +9.0 / +10.9% |

Greedy outputs were identical across the backends on 96/96 one-request requests of both routes,
288/288 two-request tree requests and 272/288 two-request stock requests. Both backends started in the
32 GB memory budget. Outside the first sweep the pass-to-pass spread was at most 3.6% per cell.

## Default

ax-engine-sglang now sets `--fp4-gemm-backend flashinfer_cutedsl` on SM12x for dense ModelOpt NVFP4
targets when the flag is unset or `auto`, outside `AX_NVFP4_VERIFY_PARITY` and
`AX_QUALITY_PROFILE=verified` (which keep the CUTLASS kernels they were matched on). MoE targets
(Flash Next) and SM110 were not measured and keep SGLang's choice. At 128 rows (C16 replay) the
microbenchmark shows no gain.
