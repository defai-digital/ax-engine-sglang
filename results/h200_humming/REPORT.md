# H200 · Qwen3.8-27B NVFP4: Humming NVFP4 GEMMs and one-at-a-time DFlash admission (2026-11-28)

Qwen3.8-27B NVFP4 (ModelOpt mixed precision) with the DFlash2 drafter on one H200 SXM (SM90, 143,771 MiB,
driver 595.112.07). SGLang 0.5.23, torch 2.14.1+cu130, FlashInfer 0.7.2, humming-kernels 0.1.20,
ax-engine-sglang at 070dbf6 (flagged servers) and with this change's launch defaults (default servers). Stock vLLM
0.34.0rc1.dev1105+g2f170a0fe (torch 2.14.0, FlashInfer 0.7.4) on the same pod.

Workload and pooling as in `results/h200_prefill_graph`: 128-token retest (8 prompts per workload at one
client, 24 at eight, greedy, exactly 128 tokens), two sweeps per server, prefill CUDA graphs on, 8,192-token
prefill chunks. Cells pool the second sweep of every server (the first sweep of each server is dropped, for
both engines); "n" is the number of servers. vLLM: DFlash2, 7 speculative tokens, 8 sequences, 4,096-token
context, 8,192 batched tokens, memory utilization 0.90, prefix caching off, FP32 Mamba state.

## Why H200 trailed at eight clients

Exclusive (PDL-aware) GPU time in an eight-client math profile: SGLang's Marlin W4A16 NVFP4 GEMMs took
356 ms of a 768 ms span, vLLM's humming-kernels GEMMs 239 ms of 691 ms. The GEMM difference alone was
larger than the gap between the engines; the other kernel groups were even or favoured SGLang. On the
same H200, in CUDA graphs, Humming ran the 27B gate_up, down and lm_head shapes in 0.89-0.96x Marlin's
time at 1-8 rows and 0.59-0.65x at 128 rows (1.08-1.11x on gate_up and lm_head at 32 rows); both stayed
within 4-8e-3 of the FP32 dequantized reference.

The second cause was SGLang's admission rule for DFlash. With 8 or more running slots SGLang 0.5.23
holds each new prefill until `min(4, max(2, (slots + 5) // 6))` slots are free (2 at 8 slots,
`managers/min_free_slots_delayer.py`). With eight clients a finished request frees one slot, so the next
request waits for a second one to finish: the server ran seven requests most of the time, and the chat
first-token p90 reached 0.21-0.24 s against vLLM's 0.17 s. `--min-free-slots-delay 1` disables the hold.

## Results

Output tok/s, code / math / chat, against the first two vLLM servers (the faster baseline; see the note
on the third server below). First token: p50 / p90 over the cell's requests.

| Route, eight clients | n | tok/s | Against vLLM | First token | Tokens per verify |
|---|---:|---:|---:|---:|---:|
| Stock DFlash2 width 8, Marlin | 2 | 1,514.0 / 1,387.4 / 876.1 | −3.6 / −11.9 / −17.6% | 0.135 / 0.225 s | 6.23 / 5.72 / 3.05 |
| AX GDN replay 8, Marlin | 2 | 1,537.4 / 1,416.8 / 911.1 | −2.1 / −10.0 / −14.3% | 0.108 / 0.240 s | 6.27 / 5.74 / 3.09 |
| AX GDN replay 8, Humming | 3 | 1,718.3 / 1,617.5 / 999.2 | +9.4 / +2.8 / −6.0% | 0.089 / 0.177 s | 6.03 / 5.80 / 3.09 |
| AX GDN replay 8, Humming, admit 1 | 3 | 1,669.7 / 1,660.1 / 1,074.1 | **+6.3 / +5.5 / +1.0%** | 0.087 / 0.124 s | 5.99 / 5.74 / 3.10 |
| AX tree 32, Marlin | 2 | 1,524.5 / 1,418.6 / 916.0 | −2.9 / −9.9 / −13.9% | 0.123 / 0.187 s | 6.22 / 5.76 / 3.11 |
| AX tree 32, Humming | 2 | 1,738.3 / 1,629.7 / 1,023.8 | +10.7 / +3.5 / −3.7% | 0.077 / 0.207 s | 6.10 / 5.77 / 3.18 |
| AX tree 32, Humming, admit 1 | 3 | **1,803.9 / 1,775.6 / 1,086.8** | **+14.9 / +12.8 / +2.2%** | 0.074 / 0.108 s | 6.16 / 5.84 / 3.14 |
| AX GDN replay 8, launch defaults only | 2 | 1,641.7 / 1,729.0 / 1,078.1 | +4.6 / +9.8 / +1.4% | 0.075 / 0.119 s | 6.04 / 5.81 / 3.13 |
| AX tree 32, launch defaults only | 2 | 1,594.7 / 1,644.4 / 1,057.6 | +1.6 / +4.5 / −0.6% | 0.078 / 0.115 s | 6.01 / 5.81 / 3.17 |
| **AX GDN replay 8, Humming + admit 1, all servers** | 5 | 1,658.4 / 1,687.0 / 1,075.7 | **+5.6 / +7.2 / +1.1%** | 0.080 / 0.121 s | 6.01 / 5.77 / 3.11 |
| **AX tree 32, Humming + admit 1, all servers** | 5 | **1,714.0 / 1,720.7 / 1,074.9** | **+9.2 / +9.3 / +1.1%** | 0.075 / 0.113 s | 6.10 / 5.82 / 3.15 |
| Stock vLLM DFlash2 | 2 | 1,570.1 / 1,574.0 / 1,063.5 | | 0.115 / 0.170 s | |

| Route, one client | n | tok/s | Against vLLM |
|---|---:|---:|---:|
| AX GDN replay 8, Marlin | 2 | 455.3 / 381.9 / 242.9 | −6.4 / −11.5 / −14.1% |
| AX GDN replay 8, Humming, admit 1 | 3 | 461.1 / 394.0 / 250.7 | −5.2 / −8.7 / −11.4% |
| AX tree 32, Marlin | 2 | 591.1 / 482.6 / 298.6 | +21.6 / +11.8 / +5.6% |
| AX tree 32, Humming, admit 1 | 3 | 579.0 / 478.9 / 289.5 | +19.1 / +11.0 / +2.3% |
| AX GDN replay 8, Humming + admit 1, all servers | 5 | 463.2 / 396.0 / 252.0 | −4.8 / −8.2 / −10.9% |
| AX tree 32, Humming + admit 1, all servers | 5 | 587.0 / 483.6 / 294.2 | +20.7 / +12.1 / +4.0% |
| Stock vLLM DFlash2 | 2 | 486.3 / 431.5 / 282.8 | |

- **Eight clients now lead vLLM on every workload.** Humming took replay 8 and the tree from 10-14% behind
  on math and chat to within 6%; admitting one prefill at a time closed the rest and cut the
  first-token p90 from 0.18-0.21 s to 0.11-0.12 s. Tokens per verify did not move.
- **Launch defaults only.** Two servers per route ran with neither `AX_NVFP4_HUMMING` nor
  `--min-free-slots-delay` given; the launcher set both (logged as `auto`) and the scheduler loaded
  Humming. They are the same configuration as the three flagged servers, so the "all servers" rows
  pool the five. Single eight-client sweeps of that configuration ranged 1,509-1,822 tok/s on tree 32
  code; the two default servers drew the low end on code.
- **One client** stays within ±3% of the Marlin lanes (admission does nothing below eight slots); the
  tree keeps its lead and replay its deficit there.
- **The third vLLM server** ran eight clients at 1,383 / 1,387 / 910 and then 1,139 / 1,290 / 985 tok/s
  (first-token p90 0.31 s), 12-27% below the first two servers, with one client unchanged. The cause
  was not found. With all three servers pooled, vLLM reads 1,394.1 / 1,466.3 / 1,035.9 at eight clients
  and the tree with Humming and admission +29.4 / +21.1 / +4.9% (three flagged servers) or
  +22.9 / +17.3 / +3.8% (all five); the table uses the first two.
- **Outputs.** Humming's split-K is not run-to-run deterministic (it reduces through locks), and its
  greedy outputs differ from Marlin's; acceptance stayed within run variance. The parity and verified
  profiles refuse it.

## Launch defaults

The launcher now sets, when the user has not:
- `AX_NVFP4_HUMMING=1` and `--fp4-gemm-backend marlin` (the branch Humming replaces) for dense ModelOpt NVFP4
  DFlash and tree launches on SM90 when `humming-kernels` is importable. `AX_NVFP4_HUMMING=0` keeps Marlin.
- `--min-free-slots-delay 1` for DFlash and tree launches with exactly eight running slots, on every GPU.

Both are skipped with `AX_AUTO_DEFAULTS=0` and with the parity profiles. Install Humming into the SGLang
environment with `uv pip install --no-deps humming-kernels==0.1.20`.
