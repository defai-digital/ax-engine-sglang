# Main deployment and NVFP4 target-logit investigation (2026-11-18)

This extends the [RTX correctness campaign](2026-11-17-rtx-development.md).
The assigned RTX PRO 6000 Blackwell Server Edition runs all model inference and
CUDA controls. SGLang 0.5.23, PyTorch 2.14.1+cu130, Python 3.12.3 and checkpoint
revisions are unchanged. These are correctness diagnostics, not throughput
measurements or B200 qualification. NVFP4's independent AR/chain distribution
gate remains failed; no new sampling cohort replaces that result.

## Complete main deployment

Main commit `ea282d9` was exported with `git archive`. Its plugin was built as a
wheel and force-installed without dependencies into the existing pinned GPU
environment, replacing the editable installation. Imports from `/tmp` resolve to
the environment's `site-packages`, with no source checkout on `PYTHONPATH`.
The installed package digest is
`a7ed7e11f1ddb29e7684c7157d9a47be08e85fe675b2348c856deed1067e6f86`,
identical to main's runtime sources and the earlier passed eager diagnostics.

| Installed-wheel case | Reference prompts | Mixed requests | Replay after mixed requests |
|---|---|---|---|
| NVFP4 27B AR | 7, up to 32 output tokens | 15 passed | identical |
| NVFP4 27B tree/chain | 7, up to 32 output tokens | 15 passed | identical |
| Flash Next MTP7 | 7, up to 32 output tokens | 15 passed | identical |

All cases have two server slots, a 4,096-token context limit and a long prompt
that actually encodes to 2,609 tokens. Decode graphs remain enabled. Flash Next
uses deterministic inference, stable QSA Top-K, ring 16, explicit CUTLASS MoE
backends, the verified compact MTP snapshot and pinned-host FP8 PLE. Each server
starts in `/tmp`; the plugin runs from the installed wheel. No research checkout
is required to resolve its worker modules. Launch commands and plugin settings
are saved with each result under `ax-main-validation-20261118/` on the pod.

After the Flash smoke passed, its 78,962,697,648-byte RAM shard cache was replaced
by links to the retained, same-size persistent checkpoint shards. The prior RAM
SHA-256 manifest and the cache-release record are preserved. No persistent
checkpoint was removed. This releases host RAM for dense-model precision controls.

## Compare identical conditioning histories

`scripts/nvfp4_forward_probe.py` uses SGLang's public `--forward-hooks` mechanism.
It captures target logits before sampling only while an explicit request marker
exists. Eager execution is required: Python hooks do not execute on graph replay.
The probe records logical positions and input token IDs. It compares a row only
when every earlier token in that verify window agrees with the committed output,
and the complete conditioning prefix also matches the reference run. A rejected
candidate invalidates later rows even if a later candidate happens to match.
Tree positions are rejected. All labeled prompts need a captured first prediction.

Six frozen prompts are used: the four distribution-test prompts plus the first
two regression prompts. Each request is greedy, with at most 32 output tokens,
one request in flight, server seed 314160 and radix cache disabled. AR and plugin
chain use the installed plugin; `AX_TREE=0` forces chain verification at width 8.
The original tree settings retain a draft block of 16 narrowed to 8. The stock
control uses native DFlash at width 8 with all plugin flags disabled. The models,
token inputs and attention backend are held fixed within each precision.

| Comparison | Identical prefixes compared | Different argmax choices | Largest raw-softmax total variation |
|---|---|---|---|
| NVFP4 AR vs plugin chain | 144 | 4 | 0.2720 |
| NVFP4 AR vs native DFlash | 124 | 5 | 0.4063 |
| BF16 AR vs plugin chain | 162 | 1 | 0.03119 |

For the first six positions of the four distribution prompts, the NVFP4
AR/plugin-chain comparison has maximum raw-softmax total variation 0.07871.
The largest logit difference over all compared prefixes is 5.4765625 in both
NVFP4 comparisons. These observations establish a difference before the sampler
on the tested prefixes. Native DFlash reproduces the class of difference; a
plugin-only explanation is insufficient. They do not establish the full cause
of every failed sampled-distribution test or qualify a different sampler.
The distances use raw softmax, without top-k/top-p truncation; they are not new
two-sample statistical verdicts.

All six prefill predictions are bit-identical within each comparison. The
differences appear in subsequent incremental/block verification. Comparing the
two NVFP4 speculative modes directly gives 172 identical prefixes with one
argmax difference and maximum logit difference 2.71875. Their different draft
block settings and resulting accepted chunk boundaries prevent attributing that
difference solely to plugin code; they are not bit-equivalent controls.

The expanded BF16 prompt set also exposes a near-tie at lighthouse prompt p0,
generated position 1: the chain choice has 0.125 logit regret in AR, and AR's
choice is tied for best in the chain logits. This does not invalidate the
previous independent 28-test BF16 distribution pass, but demonstrates that it
cannot be generalized into universal greedy token identity. The largest BF16
logit difference is 0.3125 over the compared prefixes.

An exported-tensor CPU audit checks all valid prediction rows from the seven
probe runs against their actual committed token IDs. All 1,015 committed tokens
have zero positive logit regret relative to that run's own target logits. This
finds no greedy accept/commit violation in these records; it does not validate
sampled acceptance or prove equivalence of recurrent states. The offline audit
uses CPU PyTorch 2.15.0 on saved tensors, without additional model inference.

## Localize arithmetic without changing the inference path

Real checkpoint projections in layers 0/7/23/47 are replayed with an exact input
row repeated at M=1/2/4/8/16/32/64/128. Each layer has three trials. The original
outputs remain the model's outputs; these extra calls do not run attention or
touch recurrent/KV state. Gate/up controls in all three NVFP4 serving modes are
bit-identical across the tested M values. A complete gate/activation/quantization/
down MLP control also gives bit-identical repeated-row results in four layers.
This does not reproduce a GEMM row-count defect on these inputs.

The final control additionally compares the single-row result to its original
mixed-input matrix and tests that matrix's exact M. It verifies that diagnostic
replays preserve the original output storage, rejecting reused-buffer corruption.
All 12 final MLP trials preserve the original output. Both the original mixed
matrix row and every replicated-row result match the single-row result exactly;
the 100 replicated-row measurements include the actual 26-row prefill size.
This does not reproduce an MLP batch-shape/content defect at the sampled inputs.
It does not isolate the remaining difference to a specific attention, GDN or
state-update kernel, or establish a safe runtime fix. No serving kernel is
changed to conceal the failed NVFP4 gate.

The final tool includes CPU tests for prefill/decode alignment, rejected draft
histories, output budgets and unsupported tree shapes. Five CUDA controls check
inactive/capture skipping, unchanged outputs, trial limits, complete/matching
prompt sets, dense MLP replay and detection of reused output storage. The runtime
package and serving kernels are unchanged by this diagnostic tooling.

The complete 46-test CPU suite passes with SciPy and no skips on Python
3.10.22, 3.12.13 and 3.14.10. All final comparisons were replayed through the
strict analyzer. All model servers were stopped, GPU allocation returned to
0 MiB, and no cgroup OOM or OOM kills occurred.

The sibling directory `ax-engine-sglang-rtx-evidence-20261118/` held
the wheel, launch metadata, exact diagnostic sources, summaries and raw traces.
It was removed on 2026-11-19 after this report was recorded on main.
`raw-logit-traces.tar.gz` contained 627 tensor files (867,561,147 compressed bytes);
its gzip CRC was checked and its SHA-256 is
`2139a771b4650cdd9f96ab85a1ecfd8b48a69624d7b5941e409eb8cca69471c2`.
The installed wheel's SHA-256 is
`6b6217269011cb7c6b2c8d427afe37f1ec31c2fe71b7f7140ac53097e3a3db3a`.
NVFP4 cross-route lossless qualification remains blocked; these measurements
justify better diagnostics and narrower claims, not an unproven runtime patch.
