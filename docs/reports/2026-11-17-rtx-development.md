# RTX development validation (2026-11-17)

Development, debugging and correctness work runs on the assigned NVIDIA RTX PRO
6000 Blackwell Server Edition (97,887 MiB reported). B200 availability is not a
prerequisite. This report extends the
[maintenance-refactor validation](2026-11-17-maintenance-refactor.md); it makes no
new throughput or B200 qualification claim.

The [development plan](../decisions/2026-11-17-rtx-development-validation.md)
records checkpoint revisions and the test sequence. Software: SGLang 0.5.23,
PyTorch 2.14.1+cu130, Triton 3.8.0, Python 3.12.3, driver 595.112.07.

## Completed dense-model controls

The baseline/candidate comparison was repeated with the pinned bf16 27B model,
in addition to the previously reported NVFP4 comparison. All six greedy outputs
matched before/after the refactor; both lanes passed 15 mixed requests and
reproduced the references afterward. Baseline is `959c27e`; the candidate runtime
is `95c7a58`. Evidence: `ax-refactor-model-check-bf16/result.json` on the pod.

The expanded runner exercised the following four cases with two server slots,
decode CUDA graphs enabled, seed 0, seven reference prompts and context limit
4096. The extra prompt actually encoded to 2,609 tokens, past QSA's 2,048-token
selection budget. Tree controls used 16 nodes, draft block 16, chain width 8,
private graph widths 16/8, draft width 8. Radix cache, FlashInfer autotune and
prefill graphs were disabled for all lanes.
`AX_TREE_SAMPLING` was unset: sampled requests on the tree/chain server use
SGLang's chain verifier. These results do not qualify the optional sampled-tree
path.

| Target | Route | Mixed requests | Stress requests | Lifecycle result |
|---|---|---|---|---|
| NVFP4 27B | AR | 15 passed | not requested | reference identity retained |
| NVFP4 27B | tree/chain | 15 passed | 32; five aborts sent, four observed | zero errors; pools returned; references retained |
| bf16 27B | AR | 15 passed | 32; five aborts sent, five observed | zero errors; pools returned; references retained |
| bf16 27B | tree/chain | 15 passed | 64; 14 aborts sent, 11 observed | zero errors; pools returned; references retained |

Stress used eight clients, a 0.2 abort probability and token budgets from
1/2/7/16/64/256/512. An abort sent after completion need not be observed.
This validates two active server slots; it does not qualify 16 active slots.
An additional NVFP4 pre/post-refactor stress control passed 64 requests per lane,
zero errors, unchanged references and returned pool usage.

Each of the four cases collected four prompts times 500 samples, six output
tokens, temperature 1, top-p 0.95, top-k 20 and server RNG seed 0. The collector
sent per-request seeds, but SGLang ignored those because deterministic inference
was disabled (see the seed correction below).
The existing 28-test distribution check used alpha 0.001:

- bf16 AR versus tree: pass; no tests below 0.01 or 0.001. All seven greedy
  references were also identical across routes.
- NVFP4 AR versus tree: fail; four tests below 0.01 and one below 0.001
  (prompt 0, position 3, p=0.000168). Only one of seven greedy references matched
  across routes. This is an unresolved route/precision difference, not a passed
  lossless qualification. Target-only verifier controls and bf16 controls passed.

Records are under `ax-rtx-development/` on the pod. The same-route pre/post
comparison under `ax-rtx-nvfp4-followup/tree-pre-post/` passed with server RNG seed
0: all seven greedy references and all 2,000 sampled token sequences
matched exactly between `959c27e` and the current plugin. Both lanes also passed
32 stress requests and returned pool usage. Its distribution check passed 28/28.
This provides no evidence of a maintenance-induced sampling regression.

The followup attempted to change the request seed base to 1,000,000, but its AR
sample file was byte-identical to the original seed-base-0 AR file. Inspection
confirmed that `SamplingBatchInfo` creates request seeds only with deterministic
inference enabled; these servers used the default FlashInfer sampler without
that flag. The claim that those cohorts are independent is withdrawn. Their
histogram differences remain observations, not independent-seed confirmation.
The unchanged same-route replay remains a valid maintenance regression control.

The collector now rejects ignored request seeds before generating samples and
records the effective RNG mode, server seed and sampling backend in a sidecar.
The runner exposes `--server-seed`, defaults request seeds to -1 (server RNG),
and rejects explicit request seeds without deterministic inference. A strict
`compare --require-independent-seeds` gate rejects missing metadata or reused
server RNG seeds. Fresh AR/chain cohorts use distinct, recorded server seeds and
the strict metadata/count gate. Each lane collects 2,000 requests (four prompts,
500 samples each). The NVFP4 AR/chain comparison fails six of 28 tests at alpha
0.001, with p-values from 6.74e-8 to 6.08e-4. Both individual lanes pass reference
replay and 15 mixed requests; those lifecycle passes do not override the failed
cross-route distribution gate. Only one of seven greedy references matches
across the NVFP4 AR/chain routes in this fresh campaign as well.

The actual NVFP4 server seeds are 314160 (AR) and 271829 (chain). These two runs
started with the pre-review harness, which incremented candidate-only seeds;
the launch records and sampler sidecars record the actual distinct seeds. The
BF16 runs use the corrected harness and preserve requested seeds 314159 and
271828. All 28 BF16 distribution checks pass, with none below 0.01; all seven
greedy references also match across routes. Both lanes pass their 15 mixed
requests and before/after replay. Every lane has exactly 2,000 complete six-token
samples, and each actual launch seed matches its collection metadata.

| Precision | Actual AR seed | Actual chain seed | Greedy references matching across routes | Distribution checks below 0.001 |
|---|---|---|---|---|
| BF16 | 314159 | 271828 | 7/7 | 0/28; pass |
| NVFP4 | 314160 | 271829 | 1/7 | 6/28; fail |

Records are under `ax-rtx-independent-cohorts/`. Attributing the NVFP4 difference
to GEMM batch shape still requires matched-row logit measurements.

## Flash Next controls and diagnostic repair

The complete Flash Next snapshot has 11 safetensors shards totaling
132,680,249,378 bytes. All sizes matched pinned Hub metadata, and all 299,545
tensor headers parsed. Dense snapshots were preserved with full SHA-256 readback
before freeing temporary RAM copies. No checkpoints live in Git.

The FP8 PLE table uses SGLang's pinned-host backend on the 96 GB device, with
explicit `text_config` overrides for `ple_offload_embedding=true`,
`ple_offload_backend=pinned`, and `ple_embedding_dtype=float8_e4m3fn`.

`gpu_ple_offload_check.py` passed two tests on the RTX, each covering bf16 and
FP8: host gather versus device reference, out-of-shard zeros, changed-input CUDA
graph replay, and table offsets crossing 2^31 elements.

The QSA checker previously assumed that RoPE coordinates equal logical sequence
offsets. That assumption fails for the image warmup used by multimodal models.
The earlier residual examples (logical group ending at 123, RoPE values 64..67)
occur during that warmup. The repaired diagnostic shadows the logical offsets
and RoPE values passed to the actual ring store. It checks both group membership
and the physical RoPE buffer, so it handles multimodal positions while retaining
overwrite detection. It only allocates diagnostic buffers under `AX_QSA_CHECK=1`.

`gpu_qsa_diagnostic_check.py` passed three RTX controls: multimodal coordinates
with a partial prefill group, a real ring-4 overwrite, and a deliberately damaged
physical RoPE write. The last two must be detected. The corrected full-model
eager diagnostic results are recorded below.

The first complete load succeeded in 1,221 seconds, but AR startup failed during
decode graph warmup: `FusedMoE` lacked `g1_scale_c`. SGLang 0.5.23's NVFP4 MoE
constructor observes global backend `auto` while its runner later resolves that
to TRT-LLM. The associated weight-preparation flag remains false, so the runner
requests a parameter that was never created. SM120 additionally needs a backend
with kernels for that architecture; FlashInfer's installed source explicitly
lists SM120 CUTLASS NVFP4 support, whereas TRT-LLM routed NVFP4 excludes SM120.

The development runner now accepts explicit target/draft MoE backend flags.
The retry uses `flashinfer_cutlass` for both, rather than patching site-packages
or inventing missing scales. A small nonzero NVFP4 MoE plus graph-replay control
passed after compiling the SM120 kernels: maximum relative error 0.00015343
against its analytical constant-weight reference (gate 0.05). This checks the
backend, not model quality. Approximately 79 GB of non-PLE shards have been
staged in RAM with streamed SHA-256 readback; the PLE shard remains on persistent
storage. The retained RAM subset was temporarily reduced to 48,945,990,576 bytes
to leave more host-memory headroom for PLE loading, then restored to all ten
non-PLE shards after the memory controls passed. Both stages use the verified
persistent copies and preserve model weights.

The explicit-backend AR retry generated all seven references and passed 15 mixed
requests. Its 32 stress requests had zero HTTP errors, five observed aborts and
returned pool usage, but four post-stress references changed (prompts 0/1/3/5).
That run failed its identity gate and is not counted as a pass.

A resident MTP3 control then repeated seven prompts three times **before** its
stress phase. The second round differed on prompts 2/6 and the third on 2/5/6.
Differences appeared at generated positions 0, 11, 14 and 20. Successful cache
flushes between requests did not eliminate the variation. Consequently, a changed
post-stress reference alone cannot establish a cancellation-induced state leak
on this configuration. An isolated random-weight, 512-expert, top-10 NVFP4 MoE
control replayed each of batch sizes 1/2/13 thirty times, with and without fused
finalization: all 180 replays were bit-identical. This narrows the investigation;
it does not establish a full-model cause.

These initial failures are retained under `ax-rtx-flash-development/`; the stable
configuration and completed graph matrix are described below. Flash Next tree
verification remains unsupported until the PLE/QSA tree path is implemented.

The deterministic AR control failed before serving: QSA's graph-safe
FilteredTopK needs at least 128 KiB shared memory, and its deterministic CUB
fallback is unavailable on the RTX for this call. A separate native QSA replay
control found another reproducible issue with fixed 1,024-column scores: output
order differed on all 30 replays; with tied scores, membership also differed on
all 30. The 16-column controls were stable. This is a concrete source of QSA
selection variability, but does not by itself explain every full-model difference.

An opt-in `AX_QSA_STABLE_TOPK=1` fallback uses fixed-shape tensor operations and
stable sorting, with smaller relative indices winning ties and -1 padding for
short rows. Three RTX tests passed: 90 dtype/shape/ragged/tie combinations,
changed-score/bound CUDA graph replay, and real SGLang alias installation followed
by transactional rollback without loading the tree worker. Full-model validation
of the combined deterministic-inference/stable-Top-K configuration then passed:
seven 64-token references, 15 mixed requests, 32 stress requests with five
observed aborts, exact repeated references before stress, exact references after
stress and returned pool usage. Decode CUDA graphs remained enabled. The record
is `ar-stable-topk-deterministic/result.json`. The ordinary eager AR control had
still varied before stress (prompts 4/5) and after stress (5/6), so disabling
graphs alone did not resolve the baseline variation.

A compact MTP checkpoint was derived outside Git from exactly the 3,102 original
MTP/embedding tensors consumed by the SGLang MTP loader (3,969,424,896 data bytes).
Every selected tensor's shape, dtype and SHA-256 matched after serialization.
It preserves checkpoint precision and values while avoiding repeated scans of
unrelated target-model shards. `scripts/prepare_flash_mtp.py` reproduces this
derivation from a pinned local snapshot. Its real 3,102-tensor output was checked
against the original derivation, including every tensor hash; the scalar hashing
case was also verified against a fixed byte representation.

MTP3 with the compact snapshot, deterministic inference, stable QSA Top-K, ring
16 and decode graphs passed all seven references, 15 mixed requests and 32 stress
requests (five sent/observed aborts). Its pre-stress repeat and post-stress
reference differences were both empty, and pool usage returned. Evidence:
`ax-rtx-flash-stable/mtp3-graph/result.json`.

MTP7 and MTP12 graph runs passed the same controls, including exact repeated
references, 15 mixed requests, 32 stress requests, five observed aborts and
returned pools. Records are `ax-rtx-flash-stable/mtp7-graph/result.json` and
`ax-rtx-flash-stable/mtp12-graph/result.json`. The first eager
diagnostic exposed an observer-only dtype error: real QSA logical positions are
int32, while the shadow buffer is int64. Both logical/RoPE shadow assignments now
convert to the destination dtype. GPU controls cover int32 and int64 inputs using
SGLang's real RoPE store, while retaining overwrite and damaged-store negative
controls. The corrected observer then passed both MTP3 and MTP7 eager runs:
seven 64-token references, 15 mixed requests and exact post-request replay in
each case. Each eager run's seven references also exactly match its graph run.
The last periodic QSA counters report 4,000 checked groups for MTP3 and 6,000
for MTP7, both with zero corrupt groups; neither complete log contains an
overwrite or physical-RoPE mismatch diagnostic. Records are under
`ax-rtx-eager-reviewed/{mtp3,mtp7}/`. These eager runs do not add stress requests;
the graph runs provide the cancellation and pool-return controls above.

All owned model servers were terminated after the campaign, GPU allocation
returned to zero and the cgroup reported no OOM or OOM kills. Compact evidence,
including raw independent samples, launch metadata, lifecycle records and eager
server logs, was stored in the sibling directory
`ax-engine-sglang-rtx-evidence-20261117/` and removed on 2026-11-19 after this
report was recorded on main. Checkpoints remain outside Git.

## Reusable tooling

`scripts/check_model_regression.py` now exposes AR/MTP, candidate-only, stress,
sampling seeds, slot count, long-context, model overrides and eager QSA controls.
It saves launch flags and plugin environment with results. It refuses unsupported
Flash Next tree verification before loading weights. Explicit token corpora let
the stress tool run without dataset downloads. Distribution comparisons now
reject empty/duplicate records, missing prompts or unequal per-prompt sample
counts instead of silently comparing their intersection. Category ordering is
deterministic.

Initial CPU tests passed on Python 3.10.22, 3.12.13 and 3.14.10: 25 tests per interpreter,
including independent wheel/sdist builds and malformed-input negative controls.
The reviewed QA integration passed 38 tests per interpreter with SciPy and no
skips; see the [branch review](../reviews/2026-11-17-deepseek-runtime-qa.md).
The repaired seed protocol was also exercised against the resident Flash Next
server: ignored request seeds were rejected before generation; a server-RNG
collection completed eight requests and recorded the actual seed 0, FlashInfer
backend and disabled deterministic mode.

The final main integration retains the follow-up metadata validation while
rejecting correlated request-seed statistical comparisons. Its 42-test CPU suite
passes on Python 3.10.22, 3.12.13 and 3.14.10 with SciPy and no skips. The final
comparator replay on the retained independent samples preserves the BF16 pass
and NVFP4 failure above; the integration changes no inference kernels.

The subsequent [main deployment and matched-prefix investigation](2026-11-18-main-deployment-and-nvfp4.md)
adds clean-wheel GPU smokes, native DFlash and BF16 controls, and real-weight
projection/MLP replay diagnostics. It retains the failed NVFP4 qualification and
documents an additional BF16 greedy near-tie on the expanded prompt set.
