# Two B200s · Flash Next BF16: prefill CUDA graphs close the gap to stock vLLM (2026-11-26)

`Qwen/Qwen3.8-Flash-Next` BF16 (revision of `results/cross_engine_2026-11-25`) on two B200s with
tensor parallelism of 2 (driver 595.112.07), PLE table in pinned host memory, SGLang 0.5.23, torch
2.14.1, FlashInfer 0.7.2, ax-engine-sglang 0.5.1, stock vLLM 0.34.0rc1.dev1105+g2f170a0fe. Same
client and prompts as the cross-engine two-B200 section: `benchmarks/baselines/bench_speed.py`, the
frozen B300 prompt file (SHA-256 `7f5953285f237f81eda3a502e3e2454719d9352106c824c67889c9d1f2855dd8`),
24 greedy prompts per workload, a 256-token limit, concurrency 1, 4 and 16. Every lane ran on one pod.

## Why SGLang trailed at 4 and 16 clients

In the cross-engine run stock SGLang MTP3 and stock vLLM MTP3 accepted the same number of tokens per
verify (3.93 / 3.50 / 2.64 against 3.92 / 3.52 / 2.62), and at one client a request decoded at the
same rate on both (564 against 567 tok/s per request on code). From four clients on, SGLang's
per-request rate fell behind (257 against 355 on code at C4), although its logged decode steps did
not: a C4 MTP3 round took 10.2-10.5 ms on both pods measured.

The time went to prefill. The SGLang lanes were launched with `--disable-prefill-cuda-graph`, which
the B300 harness had carried over. Eager Flash Next prefills of 80-220 tokens took 0.085 s on the
first pod and about 0.2 s on this one (first-token time), against 0.03-0.07 s for vLLM, and every
prefill stops the decode batch that is running. At four clients each request sees about three other
requests' prefills during its own decode, which accounts for the per-request gap.

## Lanes

Output tok/s, code / math / chat, pooled over all passes of a lane (total tokens over total wall
time). The SGLang lanes below use SGLang's own attention and MoE-autotune choices unless marked.

| Lane | Passes | C1 | C4 | C16 |
|---|---:|---:|---:|---:|
| Stock SGLang MTP3, eager prefill (cross-engine launch: Triton attention, no FlashInfer autotune) | 2 | 309.0 / 353.0 / 285.5 | 544.3 / 705.5 / 572.6 | 752.0 / 1,567.3 / 1,125.0 |
| Stock SGLang MTP3, eager prefill | 2 | 222.7 / 363.0 / 298.4 | 415.2 / 691.5 / 608.9 | 1,103.4 / 1,594.8 / 1,251.0 |
| Stock SGLang MTP3, prefill graphs (cross-engine attention and autotune) | 2 | 314.8 / 346.0 / 327.7 | 946.6 / 819.1 / 790.3 | 1,859.7 / 2,129.4 / 1,627.1 |
| **Stock SGLang MTP3, prefill graphs** | 4 | 418.2 / 409.3 / 319.2 | 969.9 / 948.2 / 781.1 | 1,932.9 / 2,127.4 / 1,565.9 |
| AX MTP7, eager prefill | 2 | 351.1 / 386.0 / 256.6 | 595.1 / 688.5 / 498.8 | 1,358.9 / 1,625.3 / 1,120.9 |
| AX MTP5, prefill graphs | 2 | 450.3 / 472.1 / 286.5 | 903.6 / 976.6 / 667.5 | 1,600.7 / 2,129.8 / 1,490.4 |
| **AX MTP7, prefill graphs** | 4 | **549.4 / 505.7** / 311.1 | **1,073.5** / 1,090.0 / 679.0 | 1,978.8 / **2,490.2** / 1,442.4 |
| Stock SGLang AR, prefill graphs | 2 | 214.0 / 166.5 / 191.6 | 484.1 / 628.2 / 606.7 | 995.5 / 1,361.1 / 1,252.5 |
| **Stock vLLM MTP3** | 4 | 364.7 / 440.1 / **314.6** | 884.3 / **1,126.1 / 777.3** | **2,096.2** / 2,432.1 / **1,679.6** |

- **Prefill graphs.** With SGLang's prefill CUDA graphs on (its default), first-token time at one and
  four clients fell from 0.20-0.33 s to 0.08-0.11 s and stock MTP3 rose 78 / 34 / 36% at four clients and 157 / 36 / 39% at sixteen. AX MTP7
  rose 80 / 58 / 36% and 46 / 53 / 29%. The Triton attention and disabled FlashInfer autotune of the
  cross-engine launch made no consistent difference once prefill graphs were on.
- **Against stock vLLM MTP3**, AX MTP7 with prefill graphs led code at one and four clients (1.51x and
  1.21x). At sixteen it was 6% behind pooled, but its pooled code cell includes a first pass that
  compiled kernels (1,362.5 tok/s); its other three C16 code passes ran at 2,040.7-2,507.5. Math was
  level (+15%, -3%, +2% at C1 / C4 / C16) and chat trailed (-1%, -13%, -14%), where seven draft tokens
  accept only 3.1 per verify. Stock SGLang MTP3 with prefill graphs was within 16% of stock vLLM MTP3
  in every cell, ahead on code at one and four clients.
- **Draft length by load.** On chat at sixteen clients stock MTP3 (1,565.9) beat AX MTP7 (1,442.4); MTP5
  sat between them on code and chat. A shorter draft at high load would help chat, as
  `AX_CHAIN_WIDTH_BY_BATCH` does for DFlash; it is not implemented for MTP.

## Noise

This pod was slower than the cross-engine pod for both engines: stock vLLM MTP3 reached 884.3 /
1,126.1 / 777.3 at four clients here against 1,260.3 / 1,270.7 / 945.0 there, with first-token times
of 0.05-0.10 s against 0.03-0.05 s. Pass-to-pass spread per cell reached 25-45% for some cells of
every engine (the largest on first passes, where kernels are compiled for new shapes: AX MTP7's first
C16 code pass ran at 1,362.5 tok/s, its later passes at 2,040.7-2,507.5). Compare lanes within this
report; the two key SGLang lanes and vLLM have four passes each (two servers each).

Bench outputs were not stored, so greedy outputs were not compared across lanes; mean completion
length, truncation and tokens per verify matched lane for lane. The prefill-graph lanes did not run
the mixed-request or multi-user stress checks of the cross-engine run.

## Launcher note

`ax-engine-sglang` now notes, for a speculative Qwen4-Exp launch that passes
`--disable-prefill-cuda-graph` outside the verified profile, that eager prefills stall the decode
batch and cost 26-45% of four-client throughput here.

The 27B DFlash2 harness (`benchmarks/sglang/bench_27b_nvfp4.py`, also used by
`hopper_tree_sweep.py`) passes `--disable-prefill-cuda-graph` as well; its prompts are shorter
relative to the 128-token outputs, and the effect there was not measured.

## Files

`benchmarks/sglang/flash_next_tp2_ab.py` (one SGLang server per lane, `bench_speed.py` passes), `fn_vllm.sh` (stock vLLM
`--gpu-memory-utilization 0.93`, 16 sequences, 8,192 batched tokens) and `stage_local.py` (copies the
checkpoint off the FUSE volume) were run from the pod; the SGLang launch line is the cross-engine
`run_matrix.py` one with the lane's changes.
