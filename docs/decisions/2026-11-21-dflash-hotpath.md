# DFlash2 batched decode hot path

The RTX PRO 6000 C8 NVFP4 chat retest measured 505.3 output tok/s for
stock DFlash2 and 580.8 for accepted-prefix GDN replay. Both routes emitted
3.07 tokens per verify. The reverse-order run measured an 11.2% replay gain
and matched 38 of 48 stock sequences. Use the replay route as the control
for further hot-path work. [RTX retest](../../results/rtx_pro_6000_retest/REPORT.md#chat-at-concurrency-eight)

## Source review

- Claude Code traced the separate snapshot and replay routes and proposed
  combining the C1 tree with the C8 replay worker. Pool allocation and
  tree-state commits are the main integration risks.
- Kimi Code identified the eager draft-context KV append as a cost that
  also survives on the replay route and proposed benchmarking the GDN
  fold tile separately from the existing parity-kernel tuning.
- AGY compared vLLM's full draft graph with SGLang's host-fed verify plan.
  Its proposed shortcut around AX runner swaps would bypass worker-state
  and buffer restoration in some configurations. Direct width-eight
  drafting already takes the `wid == block` branch and skips replanning.
- Grok CLI found the drafter's grouped convolution still used the initial
  block size during narrower graph capture and eager fallback. Its focused
  review also confirmed the redundant replay slot gather and integer casts.

## Implementation order

1. Remove redundant work from the current replay commit. Native SGLang's
   hybrid backend already consumes physical Mamba indices from the verify
   metadata. The AX replay hook can use the same live rows and keep its
   convolution step indices in the upstream scatter kernel's int32 format.
   Preserve the translated pool lookup for backends without the metadata.
   This keeps the existing experimental replay launch and fold kernel.
   [SGLang hybrid commit](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L1514-L1574)
2. Verify the narrower AX draft route before further tuning chain widths.
   Native SGLang uses the worker's position offsets in eager block prep.
   Swap offsets alongside the cached runner, sampler and scratch buffers
   so an eager or Triton-failure round uses the selected width. The pinned
   DFlash2 checkpoint also has two-tap grouped convolutions. Set their width
   before capture and keep the live drafter width in sync for eager rounds.
   Otherwise the convolution uses the wider request boundary on a narrower
   batch. [SGLang draft prep](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/speculative/dflash_worker_v2.py#L2504-L2571) ·
   [Grouped convolution](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/models/dflash.py#L519-L535) ·
   [Pinned drafter config](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/015e795645c74b1a0eeef3b570031fb62e769bc5/config.json)
3. Prototype a native SGLang worker enhancement for fixed-width DFlash2.
   Capture draft-context projection and KV writes at the existing
   post-verify commit point. Stage the current round's target hidden
   states, RoPE positions, physical draft slots and commit lengths on the
   worker stream. Run the same fused projection and prefix-valid writer
   inside a graph for each live batch size. Start with the TP1, PP1, DP1
   width-eight GDN replay lane and keep eager writes for prefill and
   unsupported inputs. Measure the staging and replay cost together.
   vLLM V1 includes context KV work inside its full draft graph. Moving
   SGLang's write to the next draft graph head would carry physical slots
   across scheduler cancellation, completion and slot reuse. That phase
   needs an explicit scheduler lifetime contract and validation of
   in-place attention metadata updates.
   [vLLM DFlash graph preparation](https://github.com/vllm-project/vllm/blob/c32251b5d192f0f72f9ca8c5595d1d43f1dd37b8/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L199-L247) ·
   [SGLang post-verify KV append](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/speculative/dflash_worker_v2.py#L2974-L3000)
4. Evaluate one server with the C1 tree and C8 replay if the workload needs
   both. The RTX retest used separate serving routes. A combined worker needs
   a direct tree-state commit and a linear-prefix replay commit over a
   compatible pool. Validate both transitions and slot reuse before changing
   the launch gates. [RTX throughput and route settings](../../results/rtx_pro_6000_retest/REPORT.md)

Replacing the native DFlash worker or scheduler is a larger option if the
trace shows that split graph replay and host-fed planning dominate. The
first native prototype can keep SGLang's scheduler and cache ownership.
FlashInfer's DFlash verify plan currently needs fresh host inputs at replay.
A graph-based metadata rewrite needs a backend that refreshes the plan on
device, with explicit slot padding and idle-rank handling. Keep this work
independent of the single-request tree scorer and the verified profile.
[SGLang graph guard](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L545-L556)

Flash Next has a separate MTP path. At C8, AX MTP3 led chat and MTP7 led
code. A batch-size table is worth testing on mixed loads after controlling
for the greedy repeat variation. vLLM's scheduler-side table is a reference
for that design. [Flash Next retest](../../results/rtx_pro_6000_retest/REPORT.md#qwen38-flash-next-nvfp4) ·
[vLLM dynamic speculative decoding](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md)

## Validation

Use CPU tests for state-slot and convolution-step selection and width
restoration. On CUDA, compare the accepted-prefix fold and convolution
state to the reference on graph replay, eager verify, live rows with
padding, and request-slot reuse.
Exercise mixed greedy and sampled batches plus the grammar and logprob
fallbacks. For speed, compare old and new builds on the same checkpoint,
prompts and warmed C8 and C16 chat runs in both lane orders. Record tok/s,
p95 latency and tokens per verify. Include BF16 and C1 controls and keep
profiling separate from the throughput passes.

For Nsight Systems, use `PROFILE=1 AXPROF_SYNC=0 AXPROF_NVTX=1 NSYS_TRACE=1`
and compare `verify_prepare`, `verify_accept`, `target_commit` and
`draft_kv_append`. Inspect the GPU timeline for overlap and graph replay
gaps.

### Draft-context KV graph

Run the CUDA fixture to compare the native eager and captured prefix
writes at batch sizes 1 and 8 with zero, partial and full commits,
changed RoPE positions, and reused physical slots.

```bash
python ax-engine-sglang/tests/gpu_dflash_kv_graph_check.py
```

Use the same checkpoint paths and input IDs for paired throughput runs.
Repeat with the two lanes in reverse order and a separate output
directory. [Frozen input IDs](https://github.com/defai-digital/ax-engine-sglang/releases/download/v0.4.0/parity-expanded-ids.json)

```bash
python benchmarks/sglang/bench_27b_nvfp4.py \
  --target "$TARGET" --draft "$DRAFT" --input-ids parity-expanded-ids.json \
  --lanes replay8 replay8-kv-graph --out results/kv-graph-forward \
  --slots 8 --concurrency 1 8 --repeat 4 --output-tokens 256 \
  --kv-tokens 16384 --context-length 4096 --mem-fraction-static 0.90
```

The harness reads scheduler counters before and after each timed replay
pass. It records optimized rounds and fallbacks by live batch size and
requires at least one round and zero fallbacks at the requested live
batch size. Smaller batches remain in the pass record. If a pass
encounters a first-time KV graph capture, the harness repeats it.
Compare output IDs, p95 request latency, tokens per verify, and output
tokens per second before deciding whether to enable this lane in a
serving configuration.
[RTX PRO 6000 measurements](../../results/rtx_pro_6000_kv_graph/REPORT.md)

### Direct writes after the stacked draft KV projection

- Claude Code compared verifies per second across the prior runs and
  identified the fixed query offsets as a candidate for host buffer reuse.
- Kimi Code traced the work done for rejected rows and found an existing
  native multi-tensor scatter for GDN convolution state.
- AGY traced the native per-layer KV writes and vLLM's in-graph attention
  metadata update. The pinned SGLang worker already sets the RoPE position
  hint during helper initialization.
- Grok CLI recommended direct BF16 stores as the first patch and identified
  physical slot mapping and graph buffer lifetimes as rewrite constraints.

1. Keep the native SGLang commit point and projection. Run one Triton
   kernel for all draft layers that checks the accepted prefix before
   reading projected K/V. Normalize and rotate K, then store K and V
   in the existing BF16 draft pool. This removes the per-layer writer
   launches and the temporary K/V round trip. The stacked GEMM still
   covers the full verify window. Use `AX_DFLASH_KV_DIRECT=1` with the
   width-eight GDN replay lane. Use the eager writer for prefill and
   incompatible pools or layouts.
   [Native projection and per-layer dispatch](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/kernels/ops/speculative/fused_kv_materialize.py#L389-L466)
2. Profile host planning and GDN commit separately if further changes
   are needed. Native FlashInfer already avoids device-to-host reads in
   the FA2 DFlash target verify plan. It still builds host query and
   KV offsets per replay. Reuse the fixed query offsets only after
   confirming the graph wrapper and any asynchronous host buffer
   consumers can share them. The existing GDN multi-convolution
   scatter is worth measuring when a model has multiple conv state
   tensors.
   [SGLang FlashInfer verify planning](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/layers/attention/flashinfer_backend.py#L2348-L2407)
3. Evaluate a focused SGLang draft worker and attention backend rewrite
   if the host planning trace shows enough cost to justify it. vLLM
   prepares context KV, updates attention metadata, and runs the draft
   forward in one graph within the same proposal call. SGLang would need
   a device-driven draft plan and an explicit slot lifetime contract
   before combining those phases across worker rounds. Keep target
   verify, acceptance, and scheduler interfaces stable during that work.
   [vLLM draft graph](https://github.com/vllm-project/vllm/blob/c32251b5d192f0f72f9ca8c5595d1d43f1dd37b8/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L199-L246)

Run `python ax-engine-sglang/tests/gpu_dflash_kv_graph_check.py --direct`
to compare complete cache contents against the native writer. For the
frozen-prompt comparison use the same settings as above with
`--lanes replay8 replay8-kv-direct`. The benchmark checks that the direct
writer ran at the measured batch size. Verify output agreement and
tokens per verify alongside throughput and p95 latency.
[Direct writer measurements](../../results/rtx_pro_6000_kv_direct/REPORT.md)

### Reuse SGLang FA2 replay inputs

Claude Code, Kimi Code, AGY, and Grok all identified redundant fixed
query offsets and a duplicate draft/verify host bound. The RTX PRO 6000
GDN fold sweep kept the existing one-warp reduction and matched checkpoint
state bits at value tiles 8, 16, 32, and 64. At batch eight, tiles 8, 16,
and 32 were all about 1.66 ms per fold and tile 64 took 2.02 ms. The
native tile remains 32.

The vLLM checkout at `~/code/ax-engine/.internal/reference/vllm` gathers
target block tables and constructs slot mappings into persistent GPU
buffers before target graph replay. Its DFlash FULL draft graph can
update draft attention metadata during replay when the backend supports
that operation. The MTPLX reference in the same directory keeps fixed
batch shapes using explicit dummy rows and tracks the real cohort
separately. The SGLang prototype preserves native padding and cache IDs.
[vLLM target preparation](https://github.com/vllm-project/vllm/blob/2f170a0fed20fd84dfcf6da2a3a8188a32f67e4e/vllm/v1/worker/gpu/model_runner.py#L1701-L1721) ·
[vLLM block-table outputs](https://github.com/vllm-project/vllm/blob/2f170a0fed20fd84dfcf6da2a3a8188a32f67e4e/vllm/v1/worker/gpu/block_table.py#L173-L218)

1. On the native width-eight, mask-free FA2 graph lane, cache each
   wrapper's immutable query offsets on the host and device. Pack the
   changing page-ID stream directly into its captured buffer through
   SGLang's existing `kv_indices_buf` argument. Keep the native planner's
   changing KV offsets and scheduling logic.
2. Share the prepared host sequence bound between draft and verify when
   the draft uses the full cache and both bounds have the same dtype.
   Compact-cache, capacity overflow, masked, ragged, and eager cases
   continue through the native metadata paths.
3. Compare `replay8` with `replay8-plan-reuse` in both launch orders.
   Check live graph activation, exact serialized outputs, tokens per
   verify, throughput, and p95. Use the same chat token IDs to measure the
   pinned vLLM build and record backend and cache-layout differences.

The larger rewrite would replace the draft planner with one that can
refresh attention state on the device and expose a slot lifetime contract.
The target planner remains a separate stage in both implementations.
The local vLLM reference also has a CUDA GDN speculative path that fuses
recurrent decode with output normalization and gating when the state
dtype, head ratio, and verify width qualify. Compare that eligibility
and memory traffic against SGLang on each GPU before moving either
engine's state commit boundary.
[vLLM fused GDN path](https://github.com/vllm-project/vllm/blob/2f170a0fed20fd84dfcf6da2a3a8188a32f67e4e/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L2587-L2670)

The RTX comparison measured 700.3 tok/s for vLLM CUDA GDN and 696.3
for its Triton GDN path at eight clients. Both were above the SGLang
GDN replay control at 584.3. Enabling FP8 draft KV in SGLang measured
570.7 in one check. Prioritize separate traces of draft replay, target
verify, and the recurrent commit on each GPU. Use those measurements
to rank a device-driven draft planner and model-layer fusion work.

Use `python ax-engine-sglang/tests/gpu_dflash_plan_check.py` on each
platform to check changing lengths and padded rows. Match the workload,
runtime, KV precision, cache budget, and graph sizes within each GPU's
A/B. Evaluate Thor with its unified-memory headroom recorded and a
dedicated serving window. Keep the RTX and Thor throughput decisions
independent. [Plan reuse and fold sweep results](../../results/rtx_pro_6000_plan_reuse/REPORT.md)

## Stability and profiling follow-up

1. Check the cache guard with a chain width equal to and above the number
   of allocated SSM positions per slot. On Thor-01, run the KV graph,
   direct KV, and FA2 plan fixtures. Use a short pinned-model run to
   check the timed counters on a live scheduler. With `FC1_TREE64=0`,
   check explicit and inferred DFlash block sizes during SGLang argument
   resolution, before any model or graph is initialized.
2. On RTX PRO 6000, compare stock DFlash2 and accepted-prefix replay on
   the same frozen prompt IDs at concurrency 1 and 8 in both launch
   orders. Run 128- and 1,024-token checks, compare greedy outputs and
   first differing tokens, and run `stress_multiuser.py` with cancellation
   and repeated reference requests. Use the verified width-eight chain
   and serialized AR for the exact-output check. The benchmark requires
   the AR lane when `--verified-profile` is selected. Reproduce the two C1
   tree divergences with the same decoded prefix and record the first
   differing layer or token scores.
3. Capture warmed C8 and C16 chat on the same server configuration. Use
   the timeline to attribute time to draft forward, target verify, GDN
   commit, draft KV append, and the gaps between them. Run throughput and
   p95 comparisons with profiling disabled. Compare the fixed `1:16,4:8`
   table with fixed-width controls under the same mixed request workload.
   Use the trace to decide whether to work on model kernels or graph
   scheduling.

Install this checkout into `AXSITE` and point `TARGET`, `DRAFT`, and
`PROMPT_FILE` to the pinned checkpoints and fixed speed prompt file.
The profiling harness uses the same server arguments for both lanes.

```bash
MODEL="$TARGET" DRAFT="$DRAFT" PROMPT_FILE="$PROMPT_FILE" \
AXSITE="$AXSITE" OUT=results/rtx-replay-profile LANES="chain8n axreplay8" \
SETS=chat TREE_CONCURRENCY="8 16" N_REQUESTS=64 MAX_TOKENS=1024 \
CHAIN_ARGS="--attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass --linear-attn-verify-backend triton --mamba-ssm-dtype float32 --disable-prefill-cuda-graph --max-running-requests 16 --cuda-graph-max-bs-decode 16" \
PROFILE=1 AXPROF_SYNC=0 AXPROF_NVTX=1 NSYS_TRACE=1 \
bash benchmarks/sglang/run_tree_plugin.sh
```

Repeat with `PROFILE=0 NSYS_TRACE=0`, the same prompts, and a fresh
output directory. Run the lanes in reverse order on a second pass.

For the width comparison, use held-out prompt IDs ordered the same way
for all lanes. Each wave finishes before the next one starts. The final
wave uses the remaining prompts, and the actual wave sizes are saved
with throughput, p95 latency, and tokens per verification.

```bash
python benchmarks/sglang/ab_adaptive_width.py --target "$TARGET" --draft "$DRAFT" \
  --input-ids "$WIDTH_PROMPTS" --out results/rtx-width-mixed \
  --lanes w16 w8 table --repeat 1 --mixed-concurrency 1 2 4 8 --env AX_TREE=0 \
  --server-args "--disable-radix-cache --disable-prefill-cuda-graph --attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass"
```

On Thor-02 (SM110, driver 595.78, SGLang 0.5.23), the three CUDA
fixtures matched the native draft KV cache or FA2 attention inputs.
The installed `/server_info` route returned the expected timed counter
deltas with a simulated worker.
The pinned-model launch reached KV-cache sizing, where SGLang rejected
`--mem-fraction-static 0.60` and reported a minimum above 0.643. A
separate GPU workload occupied the host before a higher-budget retry.

On Thor-01 (SM110, driver 595.78, SGLang 0.5.23), all three CUDA
fixtures passed. Both pinned-model lanes ran with two request slots,
a 4,096-token KV pool, and `--mem-fraction-static 0.82`. For the ten
frozen prompts and 64 greedy output tokens, the KV graph matched GDN
replay on all 30 paired outputs across the C1, C2, C1 passes. The C2
pass recorded 94 optimized two-request rounds and 14 optimized
single-request rounds in each lane, with zero fallbacks. The launch
records and token IDs are under `results/thor_01_dflash_hotpath_smoke/raw/`.

The Thor-01 width-table smoke test submitted waves of 1, 2, 4, 1, and 2
requests. After warmup, its round dump recorded 26 single-request and
13 two-request rounds at width 16, and seven four-request rounds at
width 8. This run used the round dump to check routing. Its outputs are
under `results/thor_01_dflash_hotpath_smoke/raw/mixed-table/`.

The SGLang 0.5.23 argument resolver on Thor-01 rejected a width-16
chain graph against an eight-position DFlash allocation with
`FC1_TREE64=0`. The rejection held with the token-count flag, the
DFlash block-size flag, and checkpoint inference. A width-eight graph
passed with the same allocation.

A second pinned-model smoke test exercised the stricter timed gate on
Thor-01. GDN replay and the KV graph matched all 30 paired 32-token
outputs across C1, C2, and C1. Each C2 pass recorded 35 optimized
two-request rounds and 14 optimized single-request rounds, with zero
fallbacks. The pass records are under
`results/thor_01_dflash_hotpath_smoke/strict-guards/`.

The RTX PRO 6000 follow-up completed the frozen-input replay comparisons,
cancellation test, warmed C8 and C16 throughput passes, Nsight captures,
mixed-width waves, and serialized tree score probe. Target verification
had the largest GPU span in replay. At C16 the block-scaled and FP8 GEMMs
accounted for 65% of the summed replay verify kernel time. Fixed width 8
led the mixed-wave workload. [RTX PRO 6000 follow-up results](../../results/rtx_pro_6000_hotpath_validation/REPORT.md)

## RTX PRO 6000 target-verification priorities

On the 128-token streamed chat workload, AX replay reached 584.3 output
tok/s at C8 and vLLM reached 700.3. Their p95 request latencies were 2.14 s
and 1.63 s, with about three output tokens per verification in each engine.
The newer AX run reached 772–783 tok/s on a 512-token chat workload. Use
that workload for a fresh matched engine comparison.
[Engine comparison](../../results/rtx_pro_6000_plan_reuse/REPORT.md#sglang-ax-engine-sglang-and-vllm-on-rtx-pro-6000)

1. Profile the remaining FP8 projections at actual graph-padded verify
   shapes. NVFP4 and FP8 GEMMs made up 65% of summed C16 replay verify
   kernel time. Width-eight verification can have 64 rows at C8 and 128
   at C16. The pinned FP8 specialization covers selected shapes with up
   to nine rows. Use width eight and BF16 draft KV for these controls.
   [SM120 FP8 dispatch](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/kernels/kda_kernels/sm120_fp8.py)
2. Use CUTLASS as the FP4 control and test FlashInfer autotuning as its own
   variable. Compare `flashinfer_cudnn` and opt-in `b12x` by shape and batch
   size. SGLang 0.5.23 resolves FP4 `auto` to CUTLASS on SM120. FlashInfer
   0.7.2 supports `b12x` for eligible SM120 layouts. Confirm the selected
   kernels in Nsight.
   [SGLang dispatch](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/layers/quantization/fp4_utils.py#L88-L151) · [FlashInfer implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/flashinfer/gemm/gemm_base.py#L7180-L7218)
3. Compare both engines under sustained load with identical prompts,
   output budgets, and streaming conditions. Only 84–93 of roughly 830
   replay rounds per C16 pass had 16 live requests. Report the live-batch
   histogram, throughput, time to first token, p95 inter-token latency,
   and tokens per verification. Require the serialized 1,024-token AR
   comparison before qualifying a backend for the verified profile.
   [RTX validation](../../results/rtx_pro_6000_hotpath_validation/REPORT.md)

### Lessons from AX Engine and AX Engine vLLM

- **Audit live FP8 operands.** AX Engine vLLM improved C8 throughput by
  3.43% with 64-row GDN QKVZ and attention QKV tactics and by 1.46% with
  a separate 48-row profile. The 64-row profile matched all 2,112
  same-input projection outputs. Capture SGLang's own activations,
  scales, and native kernels. Rotate layer weights during timing, match
  BF16 outputs in both eager and CUDA graph runs, then check live graph
  coverage. Key any accepted tactic to the GPU, checkpoint, shape,
  native tactic, and library versions.
  [vLLM 64-row audit](https://github.com/defai-digital/ax-engine-vllm/blob/464911d4a16a6eba0a0ad995a7caf7c0b38c3a7b/results/REPORT.md#sm120-fp8-tactics-on-real-c8-operands) · [48-row audit](https://github.com/defai-digital/ax-engine-vllm/blob/464911d4a16a6eba0a0ad995a7caf7c0b38c3a7b/results/REPORT.md#sm120-48-row-dflash2-fp8-tactics)
- **Keep a live numerical gate for FP4 experiments.** In vLLM's C8
  backend sweep, CUTLASS reached 700.91 tok/s, FlashInfer B12x 688.67,
  and cuDNN 687.07. All three matched the small projection fixture, while
  the serving outputs differed. A faster down-projection tactic passed
  a synthetic test and changed BF16 bits on 27 of 2,112 live projections.
  [vLLM backend sweep and live tactic checks](https://github.com/defai-digital/ax-engine-vllm/blob/464911d4a16a6eba0a0ad995a7caf7c0b38c3a7b/results/REPORT.md#nvfp4-gemm-backends)
- **Measure verify cost against rows.** AX Engine found that a 16-row
  packed verify on M5 Max cost 1.6× an 8-row verify on MLX 0.32.4 and
  about the same on MLX 0.32.5. Use the same row sweep method on SM120
  to locate its own cost transitions. A vLLM policy with five proposals
  at four to eight live requests gained 1.77% at C8. The equivalent
  SGLang verify width is six including the anchor. Test it after adding
  capacity and graph validation for replay widths beyond the current
  width-eight configuration.
  [AX Engine row sweep](https://github.com/defai-digital/ax-engine/blob/426ab2bd0872d77eb8fe885c183a90ccef874ace/docs/DFLASH2-TREE-VERIFY.md#verify-cost-by-row-count) · [vLLM width comparison](https://github.com/defai-digital/ax-engine-vllm/blob/464911d4a16a6eba0a0ad995a7caf7c0b38c3a7b/results/REPORT.md#five-dflash2-proposals-at-high-concurrency)
- **Measure commit overlap after projection work.** AX Engine vLLM
  overlapped GDN commit with drafting on a second stream for a 1.29%
  C8 gain. A SGLang prototype would need a separate graph pool, an event
  wait before the next target verify, and a drain path for skipped
  drafts and teardown.
  [vLLM overlap implementation and trace](https://github.com/defai-digital/ax-engine-vllm/blob/464911d4a16a6eba0a0ad995a7caf7c0b38c3a7b/results/REPORT.md#follow-up-experiment-on-gdn-and-draft-overlap)

### FP8 tactic benchmark

Capture QKVZ and QKV inputs and native outputs inside actual width-eight
target verification. Probe cuBLAS, cuDNN, and SM120 CUTLASS with rotating
layer weights. A replay profile requires bit-identical BF16 outputs in eager
and CUDA graphs and lower paired graph time. It binds the choice to the
checkpoint config and weight index, runtime, row shape, workspace size, and
native tactic. Serving measurements require graph coverage at the requested
live batch in both launch orders.

Start with 64 rows at C8, then repeat at 48, 96, and 128 rows. Compare
throughput, p95 request latency, and generated IDs on held-out chat prompts.
Before promoting a candidate to a verified serving profile, run the
serialized 1,024-token AR comparison.
[Benchmark commands](../../benchmarks/sglang/README.md#dflash2-fp8-verification-tactics) · [Measurement record](../../results/fp8_verify_tactics/REPORT.md)

### Splash DFlash2 follow-up

1. **Reduce draft vocabulary projection.** Splash uses a 59.8% Qwen
   vocabulary subset when the last 64 tokens of every request qualify.
   The AX prototype rounds the ID ranges outward to NVFP4 scale tiles,
   packs the weights once, and remaps draft proposals to global IDs.
   Thor-01 cut isolated projection and top-k time by 39.8–40.5% at
   7, 14, and 56 draft rows.
   [Thor serving and projection tests](../../results/dflash_draft_vocab/REPORT.md) · [Splash vocabulary policy](https://github.com/incoai/splash/blob/d40c250ff248938edebdb91fb6124514e32daeaa/docs/models.md#drafts)
2. **Score verified rows by expected committed tokens per millisecond.**
   Splash calibrates the cost of row counts and adds a term for context
   history. It estimates conditional acceptance by draft confidence and
   depth, then pools candidate rows across requests. AX already adapts
   chain width using round times and observed acceptance. Score the
   existing captured widths offline by live batch, padded rows, and
   context length. Compare against the fixed width-eight lane before
   adding graph shapes or changing recurrent state capacity.
   [Splash cost selection](https://github.com/incoai/splash/blob/d40c250ff248938edebdb91fb6124514e32daeaa/runtime/model/StepCosts.cpp#L274-L301) · [AX width policy](../../ax-engine-sglang/ax_engine_sglang/policy.py)
3. **Use prompt lookup to fill a fixed tree budget.** Splash deduplicates
   a request's own repeated suffix with the learned draft tree. Test a
   bounded lookup on C1 code and tool-call prompts, replacing low-ranked
   proposals at the same total row count. Record lookup hits, accepted
   lookup rows, and regressions on chat. Keep sampled decoding on the
   existing path until lookup proposals have a validated rejection rule.
   [Splash lookup proposal code](https://github.com/incoai/splash/blob/d40c250ff248938edebdb91fb6124514e32daeaa/runtime/model/Runtime.cpp#L2654-L2685)

Splash's Metal results combine several changes. The RTX PRO 6000 replay
trace puts 20.98 ms in target verify and 4.07 ms in draft forward at C8.
Draft KV append takes 0.35 ms at the median. Use the CUDA trace to choose
the first experiment and check its gain in both launch orders. Require
the serialized AR parity gate before changing the verified profile.
[RTX replay trace](../../results/rtx_pro_6000_hotpath_validation/REPORT.md#nsight-attribution) · [Splash release measurements](https://github.com/incoai/splash/blob/d40c250ff248938edebdb91fb6124514e32daeaa/docs/release-notes/1.7.0.md)

The ax-engine-vLLM audit found that C8 clients could yield at most three
live requests on an RTX 5090. Several faster NVFP4 kernels on synthetic
operands also changed live outputs. AX's draft vocabulary benchmark
now records live and padded CUDA graph rows for both draft and verify,
compares complete greedy outputs, and requires observed subset replay
at the requested concurrency. The current SGLang width-eight chain
produces 48 live rows at six requests. Its graph rows depend on the
configured capture sizes. vLLM's five-proposal lane produces 48 live
rows at eight requests.
[vLLM graph accounting change](https://github.com/defai-digital/ax-engine-vllm/commit/c0b548d65a631a7b212d462740c024d087817cf5) · [vLLM validation results](https://github.com/defai-digital/ax-engine-vllm/blob/baf2630/results/REPORT.md#48-row-nvfp4-tactic-sweep)

AX vLLM's five-proposal RTX PRO 6000 lane combined GDN replay with
asynchronous accepted-state commit. Throughput increased from 717.09
to 753.62 tok/s at C8 in that combined comparison. On SGLang, a
separate overlap experiment would need to trace the dependency from
accepted-state commit to the next target verify and check that the
draft graph and commit kernel use disjoint state buffers on every
replay. Compare it against the existing serial GDN replay lane at
the same verified width.
[vLLM overlap implementation](https://github.com/defai-digital/ax-engine-vllm/commit/a1133e422c8b958033067dc6559d29942b40f47d) · [Five-proposal serving retest](https://github.com/defai-digital/ax-engine-vllm/blob/baf2630/results/REPORT.md#rtx-pro-6000-and-b200-retest)
