# RTX 5090 · DFlash2 accepted-prefix GDN replay

## Two-slot stock and replay comparison

| Concurrency | Stock DFlash2 output tok/s | AX replay output tok/s | Exact 128-token matches |
|---:|---:|---:|---:|
| 1 | 123.21 | 124.39 | 40/40 |
| 2 | 178.67 | 181.19 | 160/160 |

The rates pool four passes per lane from two runs in opposite lane order.
AX replay measured 1.0% higher throughput at concurrency one and 1.4%
higher at concurrency two. The primary improvement is memory. The
speculative SSM snapshot allocation fell from **3.38 GiB to zero**.
The replay buffers used about **0.017 GiB**. The convolution window
allocation stayed at 0.03 GiB.

## Eight-slot capacity and state reuse

The replay lane served eight active requests on the 32 GB card. Two
concurrency-eight passes of 40 requests delivered 377.69 and 402.35
output tok/s, or **389.63 tok/s pooled**. The fold allocated about 0.055 GiB
of replay buffers and zero speculative SSM snapshots. The matched stock
eight-slot launch exhausted the GPU memory budget before KV allocation.

A separate eight-slot stress check sent 48 requests from 12 client workers
with variable output limits and cancellation enabled. Four requests
reported cancellation, all 48 finished without errors, and the pools
returned to their pre-stress idle usage. Six serialized references with a 64-token limit matched their repeats
before and after the stress run.

## Tree-width comparison

| Fused tree nodes | C1 output tok/s | C2 output tok/s |
|---:|---:|---:|
| 16 | 123.88 | 179.71 |
| 32 | **135.95** | **183.02** |

Tree 32 delivered **9.7%** more throughput at concurrency one on the
frozen prompts and remains the automatic default. Both lanes fell back to width-eight
chains for grouped requests.

## Measurement conditions

- One NVIDIA GeForce RTX 5090 (SM120, 32,607 MiB), driver 595.112.07, Ubuntu
  24.04 x86_64, SGLang 0.5.23, PyTorch 2.14.1+cu130 and FlashInfer 0.7.2.
  Target and draft revisions match the original 5090 study.
  [Checkpoints and frozen prompts](../rtx_5090_27b_nvfp4/REPORT.md#measurement-conditions)
- Native DFlash2 with block eight on both chain lanes, FP32 GDN state,
  Triton GDN verify, FlashInfer attention and FP8 target KV. The replay
  lane enables fold-after-accept through AX. The final reverse-order run
  sets the dtype and verify backend explicitly and disables AX's sampling
  offset patch in the replay lane. All requests in both runs are greedy.
- Context limit and KV pool of 4,096 tokens, 512-token chunked prefill,
  64 MiB FlashInfer workspace, static memory fraction 0.94, and server
  seed 42. Prefix caching, prefill graphs, and FlashInfer autotune are
  disabled. Decode graph replay is enabled.
- Each timed request generates exactly 128 tokens with EOS ignored. Each
  server runs 10, 40, 40, and 10 requests at concurrency 1, 2, 2, and 1,
  respectively. Rates divide total output tokens by summed wall-clock
  time across passes and include prefill and loopback HTTP.
- Tree 16 and tree 32 use the same 16-token drafter block and width-eight
  batched fallback and run with verification parity disabled.

## Verification

The CPU suite completed 145 tests with 3 skips. The wheel build included
the replay module and SGLang plugin entry point. On the installed SGLang
0.5.23 source, the three replay hooks installed, tolerated a second
installation, and restored the original methods after a simulated
startup failure. Grok CLI, Muse CLI, and AGY CLI reviewed the candidate.
The final replay launch isolates its option resolution and uses the same
FP32 and Triton settings as the stock control.

## Reproduce

Use the same setup and pinned checkpoints as the original 5090 benchmark.
Install this checkout's plugin alongside SGLang 0.5.23. Set `TARGET` and
`DRAFT` to the pinned local checkpoints and download the
`parity-expanded-ids.json` release asset from v0.4.0.
[Inputs and setup](../rtx_5090_27b_nvfp4/REPORT.md#reproduce-the-comparison)

```bash
python benchmarks/sglang/bench_27b_nvfp4.py \
  --target "$TARGET" --draft "$DRAFT" --input-ids parity-expanded-ids.json \
  --out results/rtx_5090_gdn_chain_replay/ab --lanes replay8 stock8 tree16 tree32

python benchmarks/sglang/bench_27b_nvfp4.py \
  --target "$TARGET" --draft "$DRAFT" --input-ids parity-expanded-ids.json \
  --out results/rtx_5090_gdn_chain_replay/ba --lanes stock8 replay8

python benchmarks/sglang/bench_27b_nvfp4.py \
  --target "$TARGET" --draft "$DRAFT" --input-ids parity-expanded-ids.json \
  --out results/rtx_5090_gdn_chain_replay/c8 --slots 8 --lanes replay8
```

The benchmark command includes serialized passes alongside the two
concurrency-eight passes. The capacity run timed the two concurrent
passes after the same warm-up.

The memory savings apply to the standalone chain. Tree and mixed tree/chain
serving retain their existing state layouts. The upstream fold replays the
accepted prefix into the FP32 checkpoint and rolls back the convolution
window. The native DFlash2 worker supplies the accepted length and request
rows in the same order as the snapshot commit.
[B200 stock-worker prototype](../../docs/decisions/2026-11-16-dual-attention-backend.md#recurrent-verification-with-accepted-prefix-replay-2026-11-20)
