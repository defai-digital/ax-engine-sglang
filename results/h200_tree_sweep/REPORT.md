# H200 tree size, GDN replay, lattice bypass and Marlin tuning (2026-11-26)

Qwen3.8-27B (official FP8 checkpoint, and the NVFP4 checkpoint through Marlin W4A16) with the DFlash2
drafter on one H200 SXM (SM90, 141 GB, driver 595.112.07), SGLang 0.5.23, torch 2.14.1, FlashInfer
0.7.2, ax-engine-sglang 0.5.1 at f6ca327 (`AX_TREE_TEMP=1.55` launch default). The workload is the
128-token retest of `results/rtx_pro_6000_retest`: greedy, the first 8 (C1) and 24 (C8) prompts per
workload, two sweeps, cells pool both. Runner: `benchmarks/sglang/hopper_tree_sweep.py`.

This was a second H200 pod. Stock DFlash2 width 8 ran 5-8% faster at one client (5-10% at eight) here than on the pod of the
cross-engine H200 section (`results/cross_engine_2026-11-25`), so compare lanes within this report.
Stock vLLM and TensorRT-LLM were not rerun.

## Lanes

Output tok/s, code / math / chat. Tokens per verify and milliseconds per verify round come from the
server's per-request `spec_verify_ct` and the decode time after the first token.

| Precision | Clients | Stock DFlash2 width 8 | AX tree 16 | AX tree 32 | AX chain 8 | AX GDN replay 8 |
|---|---:|---:|---:|---:|---:|---:|
| FP8 | 1 | 332.1 / 307.2 / 197.9 | **458.0** / 366.7 / 238.5 | 443.3 / **388.1 / 246.6** | 328.7 / 303.9 / 195.8 | 317.5 / 293.9 / 188.7 |
| FP8 | 8 | 1,182.2 / 995.8 / 709.4 | 1,247.0 / **1,045.7** / 742.2 | **1,261.1** / 1,025.5 / 739.7 | 1,222.3 / 990.4 / 685.4 | 1,248.3 / 959.8 / **753.7** |
| NVFP4 | 1 | 368.5 / 324.9 / 214.7 | **473.6** / 375.8 / 245.8 | 448.5 / **379.9 / 248.8** | 368.0 / 324.2 / 214.2 | 335.3 / 309.6 / 197.3 |
| NVFP4 | 8 | 1,166.4 / **988.6** / 609.2 | 1,208.1 / 924.8 / **667.7** | **1,212.6** / 971.0 / 666.6 | 1,168.2 / 960.2 / 628.4 | 1,193.6 / 955.7 / 658.4 |

At one client (tokens per verify, ms per round):

| Precision | Stock DFlash2 width 8 | AX tree 16 | AX tree 32 |
|---|---|---|---|
| FP8 | 6.36 / 5.66 / 3.27, 13.9 ms | 10.67 / 7.70 / 4.32, 14.5 ms | 10.78 / 8.68 / 4.74, 15.2 ms |
| NVFP4 | 6.83 / 5.72 / 3.34, 12.8 ms | 11.38 / 7.82 / 4.38, 14.2 ms | 11.38 / 8.83 / 4.92, 15.5 ms |

- **Tree size.** Tree 32 accepts more on math and chat, tree 16 is cheaper per round and wins code.
  Over the three workloads (geometric mean) tree 32 is 1.9% ahead on FP8 and 1.0% behind on NVFP4,
  inside the pass spread. The 32-node default stays on SM90; no capability-specific size.
- **Per-round cost.** A 32-node tree round costs 1.3 ms more than a stock width-8 round on FP8 and
  2.7 ms (+21%) more on NVFP4: Marlin W4A16 time grows with the row count much faster than the FP8
  GEMMs do (profile below).
- **Eight clients.** The tree routes serve chain rounds of width 8 above one request; they and GDN
  replay finish within the eight-client pass spread (up to 17% on FP8, 9.5% on NVFP4) of stock.
  Replay showed no reliable gain on SM90, unlike the RTX PRO 6000 (+16-27%).
- **AX_TREE_TEMP=1.55.** Against stock on the same pod, FP8 tree 32 rose from 1.32 / 1.22 / 1.20× (first
  pod, temperature 1.0) to 1.33 / 1.26 / 1.25×. Different pods; the per-pod ratio is the comparison.
- `tree8` (and any `FC1_TREE_N` other than 16, 32, 64) does not run: the fused tree kernels refuse it.

## Lattice bypass on chain-only servers

FP8 chain 8 (`AX_TREE=0`), before and after 5cfd3c6, same pod:

| Build | C1 | C8 | ms per C1 round |
|---|---:|---:|---:|
| f6ca327 | 328.7 / 303.9 / 195.8 | 1,222.3 / 990.4 / 685.4 | 14.0 |
| 5cfd3c6 (lattice skipped) | 333.0 / 307.9 / 198.2 | 1,239.4 / 1,009.8 / 737.0 | 13.9 |

+1.2-1.3% at one client with identical tokens per verify; greedy outputs equal on 48/48 C1
requests (135/144 at C8, where batch composition changes GEMM shapes between runs). The eight-client
gain is inside the first run's 14.9% pass spread. Chain 8 now matches stock DFlash2 at one client.

## Where a verify round goes

Torch-profiler traces of two C1 code requests per lane (`prof-*` lanes), kernel time per verify
round by category. Profiling inflates kernel times and the traces include the two prefills, so the
shares are what to read:

| Category | FP8 stock 8 | FP8 tree 32 | NVFP4 stock 8 | NVFP4 tree 32 |
|---|---:|---:|---:|---:|
| FP8 GEMM (DeepGEMM) | 53.4% | 52.6% | | |
| Marlin W4A16 (MLP) | | | 43.9% | 49.7% |
| cuBLAS / cuBLASLt GEMM (FP8 attention on NVFP4, BF16 heads, drafter) | 26.3% | 25.9% | 34.6% | 30.4% |
| Tree build and verify | | 6.3% | | 5.8% |
| Everything else | 20.3% | 15.2% | 21.5% | 14.1% |

GEMMs are about 80% of a verify round on SM90. Only the MLP of the NVFP4 checkpoint is FP4 (gate+up
34816x5120, down 5120x17408, 64 layers); its attention and Gated-DeltaNet projections are FP8.

## Marlin tile tuning

`benchmarks/sglang/marlin_fp4_tune.py` compiles SGLang's Marlin source with the tile (thread_k,
thread_n) and SM count forceable, prepares layer 0's (and layer 40's) MLP weights as SGLang does, and
times each configuration in CUDA graphs that cycle through weight copies of four times the L2 cache.
Microseconds per GEMM, automatic choice vs the best candidate:

| Rows | gate+up auto | best | down auto | best |
|---:|---:|---|---:|---|
| 8 | 46.4 | none faster | 26.8 | none faster |
| 32 | 55.3 | 54.9 (-0.7%, fp32 reduce off) | 39.6 | 37.9 (-4.4%, 64x256 tile, fp32 reduce off) |
| 64 | 80.2 | 79.5 (-0.8%, fp32 reduce off) | 57.5 | 53.2 (-7.5%, fp32 reduce off) |
| 128 | 147.2 | 146.0 (-0.8%, fp32 reduce off) | 82.2 | 80.1 (-2.7%, fp32 reduce off) |

No bit-identical candidate beat Marlin's automatic choice; fewer SMs (66, 96, 114 of 132) never
helped. The gains need `use_fp32_reduce=False` (partial sums reduced in BF16, outputs change) and
come to about 0.3 ms per eight-client round (1%), so SGLang's defaults stay. The kernel reaches about
1.9 TB/s on the gate+up GEMM at 8 rows, 40% of the H200's bandwidth: closing the NVFP4 gap to stock
vLLM on SM90 needs a faster W4A16 kernel, not a different Marlin tile.

## Reproduce

```bash
# work dir with prompts.json, models/{27b-fp8,27b-nvfp4,27b-dflash2}, stage.log ("ready <model>" lines)
python benchmarks/sglang/hopper_tree_sweep.py 27b-fp8 27b-nvfp4
python benchmarks/sglang/hopper_tree_sweep.py 27b-fp8:prof-tree32 27b-fp8:prof-stock8
python benchmarks/sglang/marlin_fp4_tune.py --model models/27b-nvfp4 --rows 1 8 16 32 48 64 128 --sms 66 96 114
```
