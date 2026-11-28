# Qwen3.8-27B NVFP4 on four GPUs: SGLang + AX against same-pod vLLM (2026-11-28)

Qwen3.8-27B NVFP4 (ModelOpt mixed precision) with the DFlash2 drafter. SGLang 0.5.23 (torch 2.14.1+cu130,
FlashInfer 0.7.2) with ax-engine-sglang and its launch defaults at 2ad90d1 (the lanes before that commit
passed the same settings explicitly). Stock vLLM 0.34.0rc1.dev1105+g2f170a0fe (torch 2.14.0, FlashInfer
0.7.4) on the same pod: DFlash2, 7 speculative tokens, 8 sequences (2 on the RTX 5090), 4,096-token
context, 8,192 batched tokens, memory utilization 0.90, prefix caching off, FP32 Mamba state.

Workload: the 128-token retest. 8 prompts per workload at one client, 24 at eight (or two),
greedy, exactly 128 output tokens, prefix cache off, prefill CUDA graphs on. There are two sweeps per server,
and the first sweep of every server is dropped for both engines. Cells are code / math / chat output tok/s, pooled over the
servers listed. "Stock SGLang" runs with the plugin off (`SGLANG_PLUGINS=none`).

| GPU | SGLang FP4 GEMM | Attention | AX launch defaults in effect |
|---|---|---|---|
| B200 (SM100) | cute-dsl (SGLang's SM100 `auto`) | `trtllm_mha`; tree verify `triton` with `AX_CHAIN_ATTN=trtllm_mha`; autotune on | admit 1 at 8 slots, FlashInfer GDN prefill |
| H200 (SM90) | AX routes: Humming (`AX_NVFP4_HUMMING`); stock: Marlin W4A16 | replay `fa3`; tree `triton` with `AX_CHAIN_ATTN=fa3` | admit 1 at 8 slots, Humming |
| RTX PRO 6000 (SM120) | cute-dsl (add-on SM12x default) | SGLang default | admit 1 at 8 slots |
| RTX 5090 (SM120, 32 GB, 2 slots) | cute-dsl | SGLang default | none apply at 2 slots |

## B200

| Route | Clients | Servers | tok/s | Against vLLM | First token p50 / p90 | Tokens per verify |
|---|---:|---:|---:|---:|---:|---:|
| Stock vLLM DFlash2 | 1 | 2 | 838.4 / 695.6 / 451.6 |  | 0.031 / 0.032 s |  |
| Stock vLLM DFlash2 | 8 | 2 | 2,660.1 / 2,851.6 / 1,819.6 |  | 0.084 / 0.111 s |  |
| Stock SGLang DFlash2 | 1 | 2 | 692.7 / 581.7 / 370.2 | −17.4% / −16.4% / −18.0% | 0.044 / 0.049 s | 7.01 / 5.63 / 3.29 |
| Stock SGLang DFlash2 | 8 | 2 | 2,271.3 / 1,976.9 / 1,329.0 | −14.6% / −30.7% / −27.0% | 0.087 / 0.178 s | 6.01 / 5.71 / 3.11 |
| AX tree 32 | 1 | 2 | 945.6 / 766.5 / 459.1 | +12.8% / +10.2% / +1.7% | 0.039 / 0.040 s | 11.64 / 8.75 / 4.68 |
| AX tree 32 | 8 | 2 | 2,634.3 / 2,551.6 / 1,538.8 | −1.0% / −10.5% / −15.4% | 0.066 / 0.082 s | 6.11 / 5.76 / 3.10 |
| AX GDN replay 8 | 1 | 2 | 717.1 / 645.7 / 406.2 | −14.5% / −7.2% / −10.1% | 0.036 / 0.037 s | 6.61 / 5.79 / 3.36 |
| AX GDN replay 8 | 8 | 2 | 2,613.2 / 2,656.8 / 1,707.0 | −1.8% / −6.8% / −6.2% | 0.061 / 0.070 s | 6.01 / 5.85 / 3.21 |

AX rows are the two servers launched with no explicit AX flags. Lanes with the settings passed by hand
are in `results/b200_gdn_prefill`.

## H200

| Route | Clients | Servers | tok/s | Against vLLM | First token p50 / p90 | Tokens per verify |
|---|---:|---:|---:|---:|---:|---:|
| Stock vLLM DFlash2 | 1 | 2 | 486.3 / 431.5 / 282.8 |  | 0.040 / 0.077 s |  |
| Stock vLLM DFlash2 | 8 | 2 | 1,570.1 / 1,574.0 / 1,063.5 |  | 0.115 / 0.170 s |  |
| Stock SGLang DFlash2 | 1 | 2 | 485.4 / 407.4 / 259.9 | −0.2% / −5.6% / −8.1% | 0.048 / 0.050 s | 6.92 / 5.63 / 3.37 |
| Stock SGLang DFlash2 | 8 | 2 | 1,514.0 / 1,387.4 / 876.1 | −3.6% / −11.9% / −17.6% | 0.135 / 0.215 s | 6.23 / 5.72 / 3.05 |
| AX tree 32 | 1 | 5 | 587.0 / 483.6 / 294.2 | +20.7% / +12.1% / +4.0% | 0.054 / 0.066 s | 11.43 / 8.75 / 4.89 |
| AX tree 32 | 8 | 5 | 1,714.0 / 1,720.7 / 1,074.9 | +9.2% / +9.3% / +1.1% | 0.075 / 0.113 s | 6.10 / 5.82 / 3.15 |
| AX GDN replay 8 | 1 | 5 | 463.2 / 396.0 / 252.0 | −4.8% / −8.2% / −10.9% | 0.051 / 0.063 s | 6.94 / 5.70 / 3.40 |
| AX GDN replay 8 | 8 | 5 | 1,658.4 / 1,687.0 / 1,075.7 | +5.6% / +7.2% / +1.1% | 0.080 / 0.121 s | 6.01 / 5.77 / 3.11 |

AX rows pool three servers launched with the flags and two launched with the defaults only.
vLLM pools the faster two of three servers. The third ran 12-27% slower at eight clients, and
`results/h200_humming` gives the comparison with all three.

## RTX PRO 6000

| Route | Clients | Servers | tok/s | Against vLLM | First token p50 / p90 | Tokens per verify |
|---|---:|---:|---:|---:|---:|---:|
| Stock vLLM DFlash2 | 1 | 2 | 294.1 / 283.4 / 179.5 |  | 0.049 / 0.117 s |  |
| Stock vLLM DFlash2 | 8 | 2 | 897.9 / 926.7 / 690.1 |  | 0.178 / 0.261 s |  |
| Stock SGLang DFlash2 | 1 | 2 | 318.4 / 258.6 / 168.8 | +8.2% / −8.8% / −6.0% | 0.053 / 0.053 s | 6.87 / 5.36 / 3.38 |
| Stock SGLang DFlash2 | 8 | 2 | 1,236.8 / 1,127.7 / 678.8 | +37.7% / +21.7% / −1.6% | 0.092 / 0.178 s | 6.30 / 5.81 / 3.06 |
| AX tree 32 | 1 | 2 | 431.4 / 363.4 / 209.5 | +46.7% / +28.2% / +16.8% | 0.055 / 0.058 s | 11.01 / 8.98 / 4.79 |
| AX tree 32 | 8 | 2 | 1,151.8 / 1,216.0 / 684.9 | +28.3% / +31.2% / −0.8% | 0.086 / 0.116 s | 6.07 / 5.85 / 2.97 |
| AX GDN replay 8 | 1 | 2 | 329.4 / 267.1 / 174.9 | +12.0% / −5.8% / −2.6% | 0.052 / 0.053 s | 6.87 / 5.36 / 3.38 |
| AX GDN replay 8 | 8 | 2 | 1,194.8 / 1,313.8 / 790.1 | +33.1% / +41.8% / +14.5% | 0.079 / 0.112 s | 5.96 / 5.72 / 3.05 |

Eight-client code varied by up to 13% between two servers of the same lane.

## RTX 5090

| Route | Clients | Servers | tok/s | Against vLLM | First token p50 / p90 | Tokens per verify |
|---|---:|---:|---:|---:|---:|---:|
| Stock vLLM DFlash2 | 1 | 2 | 245.3 / 213.6 / 173.1 |  | 0.157 / 0.242 s |  |
| Stock vLLM DFlash2 | 2 | 2 | 316.2 / 345.9 / 231.2 |  | 0.199 / 0.304 s |  |
| Stock SGLang DFlash2 | 1 | 2 | 294.9 / 237.2 / 155.1 | +20.2% / +11.1% / −10.4% | 0.075 / 0.115 s | 7.11 / 5.60 / 3.22 |
| Stock SGLang DFlash2 | 2 | 2 | 378.0 / 393.3 / 234.9 | +19.5% / +13.7% / +1.6% | 0.097 / 0.134 s | 5.89 / 5.65 / 3.06 |
| AX tree 32 | 1 | 2 | 389.4 / 318.0 / 186.4 | +58.8% / +48.9% / +7.7% | 0.077 / 0.106 s | 11.13 / 8.75 / 4.83 |
| AX tree 32 | 2 | 2 | 418.2 / 421.8 / 243.7 | +32.3% / +21.9% / +5.4% | 0.096 / 0.140 s | 6.19 / 5.91 / 3.12 |
| AX GDN replay 8 | 1 | 2 | 317.0 / 259.4 / 160.3 | +29.2% / +21.4% / −7.4% | 0.074 / 0.076 s | 7.11 / 5.60 / 3.22 |
| AX GDN replay 8 | 2 | 2 | 402.9 / 405.5 / 255.7 | +27.4% / +17.2% / +10.6% | 0.092 / 0.133 s | 5.89 / 5.65 / 3.06 |

## Not in these rows

- **B200, bf16 SSM state.** Stock DFlash2 with `--mamba-ssm-dtype bfloat16` and FlashInfer decode and
  verify (SGLang's CuTeDSL MTP kernel, no replay fold) ran eight clients at 2,739.4 / 2,779.5 / 1,723.6
  (two servers; the first alone read 2,977.7 / 2,784.5 / 1,765.8), +3.0 / −2.5 / −5.3% against vLLM and
  1-5% ahead of AX replay. One-client tokens per verify fell from 6.61 / 5.79 to 6.40 / 5.31 on code and
  math. vLLM ran an FP32 state, so it is not a like-for-like route.
- **The remaining B200 eight-client gap** is per decode round: the replay fold, the verify conv window
  writes, and the RMSNorm + FP4/FP8 quantization that vLLM fuses (`results/b200_gdn_prefill`).
