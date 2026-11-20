# Qwen3.8-Flash-Next on SGLang 0.5.23: QSA ring fix and wider MTP (2026-11-17)

Decision record and analysis: [docs/decisions/2026-11-17-qwen38-flash-next-feasibility.md](../../docs/decisions/2026-11-17-qwen38-flash-next-feasibility.md)
(sections 6-8). Raw outputs of every run below are in this directory (git-ignored).

## Setup

| Item | Value |
|---|---|
| Hardware | 1x NVIDIA B200 (RunPod), models loaded from `/dev/shm` |
| Engine | SGLang 0.5.23 (`/root/venv-sgl`), ax-engine-sglang at commits b3ca2ca-733c4d3 |
| Target | RadixArk/Qwen3.8-Flash-Next-NVFP4 (180B MoE, 12 QSA + 36 GDN layers, built-in MTP layer) |
| Speculation | stock SGLang MTP (`--speculative-algorithm EAGLE`, topk 1); `mtpN` = N steps, N + 1 tokens verified per round |
| Plugin switch | `AX_QSA_RING=16` (pending index-key ring of 16 slots per request instead of 4) |
| Benchmark | `benchmarks/baselines/bench_speed.py` via `benchmarks/sglang/run_baselines.sh`; greedy, 64 requests per set, 1,024 max tokens, c = 1 / 4 / 16 |
| Metric | aggregate output tok/s; "accept" = mean tokens emitted per round |

## Runs

| Run | What | Command (on the pod, repository root) |
|---|---|---|
| f0, f0r | AR and stock MTP 1-3 baselines (f0r: RadixArk build) | `MODES="ar mtp1 mtp2 mtp3" bash benchmarks/sglang/run_baselines.sh` |
| q1 | token identity vs. AR at 6,000-token contexts; first MTP 5 / 7 speed | `IDS_CONTEXT_TOKENS=6000 ...`, plugin overlay with the ring |
| q2 | ring consistency check, eager | `AX_QSA_CHECK=1 EXTRA_ARGS=--disable-cuda-graph`, ring 4 (MTP 1, 3) and ring 16 (MTP 3, 7) |
| q3 | speed MTP 3 / 5 / 7 with ring 16, after the plugin capture fix | `AX_QSA_RING=16 MODES="mtp3 mtp5 mtp7"` |
| q4, q4c | MTP 9 and MTP 12 (13 tokens), code and math | q4c adds `--max-running-requests 32` |
| q5 | second pass: stock MTP 3 (no plugin) vs. ring-16 MTP 7 | as q3 |

## Correctness: ring consistency (q2)

Groups compressed from ring members that do not hold the group's own positions (6,000-token
contexts, code + math, 24 requests x 256 tokens):

| Ring, mode | Target pool | MTP draft pool |
|---|---|---|
| 4 (stock), MTP 1 | 7,024 / 16,000 (44%) | overwritten members (e.g. 124 replaced by 128) |
| 4 (stock), MTP 3 | 10,636 / 16,000 (66%) | overwritten members |
| 16, MTP 3 | 48 / 16,000 | none |
| 16, MTP 7 | 72 / 20,000 | none |

The 48 / 72 left are one pattern present in all four runs and independent of the ring size; open.
Greedy token identity against AR cannot separate the variants on this MoE (batch-verify numerics
dominate).

## Speed

Stock MTP 3 vs. ring-16 MTP 7, both passes (tok/s; ratio from pass 2):

| Set | c | MTP 3 pass 1 / 2 | MTP 7 pass 1 / 2 | Ratio |
|---|---|---|---|---|
| code | 1 | 508 / 509 | 662 / 663 | **1.30x** |
| code | 4 | 1,216 / 1,201 | 1,461 / 1,460 | **1.22x** |
| code | 16 | 2,398 / 2,382 | 3,027 / 3,006 | **1.26x** |
| math | 1 | 493 / 494 | 567 / 567 | 1.15x |
| math | 4 | 1,290 / 1,292 | 1,362 / 1,356 | 1.05x |
| math | 16 | 2,559 / 2,553 | 2,771 / 2,757 | 1.08x |
| chat | 1 | 380 / 381 | 327 / 327 | 0.86x |
| chat | 4 | 1,021 / 1,021 | 820 / 819 | 0.80x |
| chat | 16 | 2,200 / 2,168 | 1,819 / 1,809 | 0.83x |

Step-count sweep, single pass (tok/s, accept at c = 1):

| Set, c | AR | MTP 3 | MTP 5 | MTP 7 | MTP 9 | MTP 12 |
|---|---|---|---|---|---|---|
| code 1 | 222 | 508 (3.87) | 596 (5.68) | 662 (7.27) | 669 (8.74) | 683 (10.58) |
| code 16 | 1,463 | 2,398 | 2,671 | **3,027** | 2,508 | 2,335 |
| math 1 | 227 | 493 (3.50) | 543 (4.69) | **567** (5.56) | 546 (6.31) | 517 (7.00) |
| math 16 | 1,644 | 2,559 | 2,703 | 2,771 | **2,798** | 2,457 |
| chat 1 | 229 | **380** (2.60) | 357 (2.94) | 327 (3.04) | - | - |

The ring itself costs nothing (ring-16 MTP 3 equals stock MTP 3 within run spread).

## Conclusions

- Reported upstream as sgl-project/sglang#51009.
- Stock SGLang 0.5.23 builds wrong compressed index keys whenever a verify window crosses a 4-token
  group, MTP 1-3 included; it shows on contexts past the 2,048-token selection budget. Ring 16 fixes
  it and allows verify windows up to 13 tokens.
- MTP 7 is the best single setting: code +22-30%, math +5-15% over stock MTP 3. Chat loses 14-20%
  because acceptance stops at about 3 tokens per round. The step count should follow the workload.
- Not measured: task accuracy, sampled decoding, contexts other than the benchmark prompts (the
  speed runs use short prompts, below the 2,048-token budget).
