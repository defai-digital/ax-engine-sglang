# Tree width, batched lattice trees and tree-shape tuning (RTX PRO 6000, 2026-11-25)

Qwen3.8-27B NVFP4 target, DFlash2 drafter, SGLang 0.5.23 with ax-engine-sglang, one RTX PRO 6000
(96 GB). Greedy decoding, 256 output tokens, `--disable-radix-cache`. Prompts: the 200-prompt mix
of `ids200.json` (HumanEval code, GSM8K math, UltraChat chat). Measured numbers are end-to-end
server throughput (output tokens per second over the pass wall time); simulated numbers are accepted
tokens per verify round from `simulate_lattice_tree.py` and are marked as projections.

## 1. Measured baselines

`ab_adaptive_width.py` lanes, 200 prompts, 256 tokens (`FC1_DFLASH_TREE` worker):

| lane | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| tree16 (fused GDN tree, one request; chain rounds above C1) | 235.2 (5.62 t/v) | 335.8 (4.89) | | |
| tree16, SGLang native GDN tree | 202.4 (5.75) | | | |
| plugin chain, width 16 | 185.6 (4.91) | 323.3 (4.85) | 517.8 (4.91) | 688.0 (4.97) |
| plugin chain, width 8 | 183.4 (4.32) | 315.7 (4.32) | 534.5 (4.32) | 758.0 (4.34) |

`bench_27b_nvfp4.py` lanes per prompt family (`tree_width_study.py replay`, the same 200 prompts:
60 code, 60 math, 80 chat; "all" pools the three families' tokens):

| C | family | replay16 | replay8 | stock16 |
|---|---|---|---|---|
| 1 | code | 278.3 (7.73) | 213.9 (5.45) | 265.0 (7.73) |
| 1 | math | 263.4 (7.20) | 226.3 (5.64) | 249.4 (7.20) |
| 1 | chat | 123.9 (3.17) | 130.0 (3.11) | 116.7 (3.17) |
| 1 | all | 183.6 (4.84) | 172.3 (4.22) | 173.5 (4.84) |
| 2 | all | 340.6 (4.87) | 318.8 (4.33) | 314.7 (4.86) |
| 4 | all | 599.5 (4.89) | 564.1 (4.30) | 514.0 (4.92) |
| 8 | all | 865.5 (4.90) | 880.5 (4.34) | 697.6 (4.90) |

`replay16` (`AX_GDN_CHAIN_REPLAY` at width 16, newly accepted by `defaults.py`) beats `stock16` at
every concurrency (+6% at C1, +17% at C4, +24% at C8) and `replay8` up to C4; at C8 width 8 is
ahead again (chat gains nothing from width 16). Greedy outputs differ between widths (NVFP4 GEMM
results depend on the verify shape): replay8 vs replay16 match exactly on 79/800 requests, stock16
vs replay16 on 579/800.

## 2. Lattice traces and the offline simulator (projections)

`AX_LATTICE_TRACE` recorded every round's DFlash2 lattice for tree16 at C1 and chain16 at C1 and C4.
The simulator re-derives each logged round before pricing other shapes: 0 mismatches over 7,188
chain drafts and their accept lengths, and 2,962 tree rounds (accept lengths of `tree:16`).

Tokens per round (accepted drafts + the target's token) at the traced anchors, all three traces:

| policy | code | math | chat |
|---|---|---|---|
| chain:16 | 8.83 | 8.22 | 3.09 |
| chain:8 | 5.56 | 5.85 | 2.99 |
| tree:8 | 5.60 | 5.95 | 3.26 |
| tree:16 | 9.01 (+2.0%) | 8.67 (+5.5%) | 3.74 (+21.0%) |
| tree:16, AX_TREE_TEMP=1.55 | 9.10 (+3.0%) | 8.78 (+6.8%) | 3.87 (+25.1%) |
| tree:32 | 9.60 (+8.7%) | 9.56 (+16.4%) | 4.11 (+32.9%) |
| tree:32, AX_TREE_TEMP=1.62 | 9.69 (+9.7%) | 9.74 (+18.5%) | 4.25 (+37.4%) |
| oracle:16 (upper bound) | 11.09 | 12.58 | 5.83 |

(percentages vs chain:16). The tree's gain is in chat, where the draft is least certain; code and
math already accept most of a width-16 chain.

## 3. Tree-shape tuning with Optuna (projection)

`tune_tree_optuna.py`: TPE, 300 trials, seed 0, all traces, requests split in halves by hash (tune
on one, report the other), objective the geometric mean over families of tokens per round vs the
default tree. At a fixed `FC1_TREE_N` the round time does not depend on the shape.

| nodes | best shape | tune half | holdout half |
|---|---|---|---|
| 16 | `AX_TREE_TEMP=1.544 AX_TREE_BRANCH_PENALTY=0.018` | +1.87% | +1.82% (chat +3.6%, code +0.9%, math +1.0%) |
| 32 | `AX_TREE_TEMP=1.619 AX_TREE_BRANCH_PENALTY=0.008 AX_TREE_LMAX=5` | +1.99% | +2.13% |

The top trials all sit at temperature 1.53-1.63 with a negligible branch penalty and any branch
limit: a flatter lattice distribution is the one lever. `notail` and the greedy spine
(`FC1_CHAIN_FIRST`) were worse. On the GPU `AX_TREE_TEMP=1.55` measured +3.9% at C1 (section 4).

## 4. Batched lattice trees (AX_BTREE), measured

`tree_width_study.py lanes`, the 200 prompts, 256 tokens, 8 slots, `FC1_TREE_N=16`, fused GDN
kernels (`FC1_FUSED=1`, `AX_FAST=1`), no `AX_SSM_POSITIONS` (one SSM snapshot per node). `btreeK`
serves 2..K requests with one lattice tree each (`AX_BTREE=K`); `n` = SGLang's native GDN tree
kernels (`AX_BTREE_NATIVE=1`); `s` = `AX_TREE_TEMP=1.55` (section 3). tree16 serves C2 and C4 with
width-16 chain rounds. Output tok/s (tokens per verify), median / p95 request seconds:

| lane | C1 | C2 | C4 |
|---|---|---|---|
| tree16 | 235.1 (5.63) 0.91 / 1.84 | 336.6 (4.91) 1.29 / 2.72 | 536.8 (4.92) 1.60 / 3.35 |
| tree16s | 244.3 (5.86) 0.88 / 1.76 | 333.9 (4.86) 1.29 / 2.64 | 534.8 (4.92) 1.62 / 3.34 |
| btree4 | 235.3 (5.63) 0.91 / 1.84 | 395.1 (5.66) 1.11 / 2.16 | 678.8 (5.70) 1.32 / 2.51 |
| **btree4s** | **244.1 (5.86)** 0.88 / 1.76 | **407.7 (5.87)** 1.09 / 2.06 | **689.0 (5.84)** 1.29 / 2.36 |
| btree4n | 234.9 (5.63) 0.91 / 1.85 | 336.8 (5.67) 1.33 / 2.56 | 548.0 (5.67) 1.64 / 3.05 |
| tree32 | 245.2 (6.17) 0.86 / 1.75 | | |
| tree32s (`AX_TREE_TEMP=1.62`) | 256.1 (6.44) 0.85 / 1.69 | | |
| AR | | | 208.3, 4.90 / 5.10 |

- **Batched trees:** btree4 vs tree16 is +17.4% at C2 and +26.5% at C4. Against the best chain
  lane of section 1, replay16, it is +16.0% at C2 and +13.2% at C4. These are different servers but
  the same prompts. The server stats count 6,494 btree rounds against 10 chain rounds in the btree4
  lane. The projection before the kernels were written was +20% at C2 and +10% at C4.
- **Native kernels:** btree4n (SGLang's native GDN tree verify) gains only 0-2%. Its tree accepts
  as much (5.67 t/v), but the native per-node state path costs the difference; the fused batched
  kernels (`ax_btree_kernels.py`) are what make batched trees pay.
- **Tree temperature:** `AX_TREE_TEMP` measured +3.9% at C1 (tree16s vs tree16, 5.86 vs 5.63
  t/v) and +3.2% at C2 for btree4s vs btree4. That is more than the simulator's +1.8% holdout; the
  traces were recorded on a subset of these prompts. Rounds served by the chain do not change.
- **Wider trees:** tree32 is +4.3% over tree16 at C1, and tree32s +8.9%. AX_BTREE requires
  `FC1_TREE_N` at most the drafter block (16), so batched trees stay at 16 nodes.
- **Kernel check:** `tests/gpu_btree_kernels.py` passed on this GPU and on the RTX 5090. It requires
  every request of a batched launch (conv, GDN outputs and saved keys/values/gains, commit
  snapshots, padded slot -1) to equal a batch-1 launch of the existing kernels bit for bit.

Exact greedy outputs. btree4 and btree4s at C1 equal tree16 and tree16s on 200/200 requests, the
same single-request path. Batched rounds change NVFP4 GEMM shapes, so outputs drift on near-ties as
batch composition changes. Against AR at C4, every speculative lane matches 15-21 of 200 requests
exactly, with a mean common prefix of 73-77 tokens:

| comparison | exact | mean common prefix (tokens) |
|---|---|---|
| tree16 C1 / C2 / C4 vs AR | 18 / 19 / 21 | 73.1 / 73.5 / 74.3 |
| btree4 C1 / C2 / C4 vs AR | 18 / 16 / 16 | 73.1 / 75.0 / 73.3 |
| btree4 vs btree4n, C2 / C4 | 17 / 16 | 80.0 / 80.8 |
| tree16 C2 vs C4 (chain only, batch differs) | 21 | 83.5 |

Batched trees drift from AR no more than the established tree and chain paths do. Earlier NVFP4
reports show the same: stock DFlash2 matched AR on 0/48 at C8 (`rtx_pro_6000_retest`).
Bit-identical greedy output needs `AX_NVFP4_VERIFY_PARITY`, which AX_BTREE does not support.

**AX_SSM_POSITIONS.** After these runs, the fused batched commit gained the one-request tree's
`AX_SSM_POSITIONS` mode. The accepted state is written in place into the slot's persistent state,
so the intermediate cache keeps only P positions per slot. That is what lets AX_BTREE fit on a 32 GB
RTX 5090 (section 5). The GPU kernel check covers the in-place mode.

## 5. RTX 5090 (low concurrency)

One GeForce RTX 5090 (32 GB), the same checkpoints, prompts (200: 60 code, 60 math, 80 chat), 256
greedy tokens and SGLang 0.5.23. The weights leave about 7 GB, so every tree lane ran with two slots,
4,096 KV tokens, `--mem-fraction-static 0.94`, `--chunked-prefill-size 512`, a 64 MiB FlashInfer
workspace, FlashInfer autotune off, a width-8 chain for chain rounds and `AX_SSM_POSITIONS=8` (the
in-place commit of section 4). Without the position cap the server could not size a KV cache.
Output tok/s (tokens per verify), median / p95 request seconds:

| lane | C1 | C2 |
|---|---|---|
| tree16 | 213.9 (5.66) 1.03 / 1.99 | 307.8 (4.38) 1.45 / 2.58 |
| tree16s | 220.4 (5.86) 1.03 / 1.93 | 307.7 (4.37) 1.44 / 2.67 |
| btree2 | 213.6 (5.66) 1.03 / 1.99 | 362.5 (5.65) 1.27 / 2.30 |
| **btree2s** | 220.2 (5.86) 1.03 / 1.93 | **370.6 (5.80) 1.23 / 2.23** |
| tree32 | 231.6 (6.18) 0.91 / 1.96 | 307.8 (4.38) 1.44 / 2.62 |
| **tree32s** | **237.8 (6.38) 0.93 / 1.80** | 310.6 (4.45) 1.44 / 2.64 |
| AR | | 116.7, 4.39 / 4.40 |

`s` is `AX_TREE_TEMP=1.55` (tree32s too). Stock DFlash2 width 16 with accepted-prefix GDN replay
(`tree_width_study.py replay`, per family, "all" pools the families' tokens):

| C | replay16 code / math / chat | replay16 all | replay8 all |
|---|---|---|---|
| 1 | 290.8 / 263.2 / 126.7 | 187.6 (4.90) | 181.8 (4.33) |
| 2 | 496.5 / 449.7 / 237.7 | 338.5 (4.88) | 309.9 (4.29) |

- **Two requests:** batched trees are +17.8% over tree16's width-8 chain rounds (btree2s +20.4%)
  and +7.1% over replay16 (btree2s +9.5%), at a lower p95 (2.23 s against 2.58 s). The server
  stats of the btree2 lane count btree rounds once two requests are live; one-request rounds stay
  on the single tree.
- **One request:** tree32s is the fastest lane, +11.2% over tree16 and +26.8% over replay16. The
  32-node tree fits on the 5090 because `AX_SSM_POSITIONS` caps the per-node state.
- **Tree temperature:** +3.0% at C1 (220.4 against 213.9), as on the RTX PRO 6000 (+3.9%).
- **Outputs:** btree2 and btree2s at C1 equal tree16 and tree16s on 200/200 requests. Against AR at
  C2 every lane matches 18-20 of 200 exactly, mean common prefix 81-86 tokens, the same NVFP4 batch
  drift as section 4.

For a low-concurrency RTX 5090 deployment the two best lanes cannot share one server: AX_BTREE
requires `FC1_TREE_N` at most the drafter block (16). Traffic that is mostly one request at a time
is served best by `FC1_TREE_N=32 AX_TREE_TEMP=1.55` (tree32s); traffic with two live requests by
`AX_BTREE=2 AX_TREE_TEMP=1.55` (btree2s), which gives up 7.4% at C1 for +19.3% at C2 against
tree32s. Both need `AX_SSM_POSITIONS=8` on 32 GB.

## Reproduce

```bash
# pod: models in /dev/shm/axm, prompt ids in ids200.json
python benchmarks/sglang/tree_width_study.py trace  --source SRC --models M --ids ids200.json --out OUT
python benchmarks/sglang/tree_width_study.py replay --source SRC --models M --ids ids200.json --out OUT --lanes replay8 replay16 stock16
python benchmarks/sglang/tree_width_study.py lanes  --source SRC --models M --ids ids200.json --out OUT \
    --lanes tree16 tree16s btree4 btree4s btree4n --concurrency 1 2 4 --shape-env AX_TREE_TEMP=1.55
python benchmarks/sglang/tree_width_study.py lanes  ... --lanes tree32 tree32s --concurrency 1 --shape-env AX_TREE_TEMP=1.62
python ax-engine-sglang/tests/gpu_btree_kernels.py   # with the env in its docstring
# offline
python benchmarks/sglang/simulate_lattice_tree.py OUT/study/trace-*/c1-chat.jsonl --policy chain:16 --policy tree:16
python benchmarks/sglang/tune_tree_optuna.py OUT/study/trace-*/*.jsonl --nodes 16 --trials 300
```
