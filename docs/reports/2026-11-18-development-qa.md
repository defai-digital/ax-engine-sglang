# Development QA on RTX PRO 6000 (2026-11-18)

The validation campaign exercised `9ed3bdb` on a single RTX PRO 6000 Blackwell
Server Edition (SM120, 97,887 MiB). The pod used driver 595.112.07, Python 3.12.3,
SGLang 0.5.23, PyTorch 2.14.1+cu130, Triton 3.8.0, and FlashInfer 0.7.2.
The serving subprocesses started in `/tmp` and loaded the plugin from the
installed wheel. Its runtime source digest matched `9ed3bdb`:
`e0bedf7fcf4b1c286b361498b4ede66f20302c289816374186b925cec2982f4d`.

## CPU and CUDA checks

The 85-test CPU suite passed on the pod and on macOS with Python 3.10, 3.11,
3.12, 3.13, and 3.14. The contract digests matched in both AX Engine
repositories, and the benchmark JSON schema passed validation. Python
compilation, Bash syntax, paired confidence intervals, lattice parity, and
tree and DFlash2 headroom checks passed. The Phase 1 tokenizer and Phase 2
drafter completed training and evaluation on synthetic CPU inputs. The
synthetic Phase 1 accuracy gate failed as expected for that fixture.

The ten CUDA check groups passed on the pod. They cover SM120 NVFP4 MoE,
pinned FP8 PLE gather with 64-bit offsets, QSA ring overwrite and RoPE-store
fault detection, stable Top-K replay, forward probes, plugin startup rollback,
tree masks, fused tree GDN and state commits, target-only sampling, and chain
reject sampling.

The mixed-request checker now rejects aborted generations, malformed response
choices, incorrectly typed JSON fields, and logprob counts that disagree with
the token count. Two regression tests cover the response-validation change.

## Installed-wheel serving

Each server used two active request slots, server seed 314160, a 4,096-token
context limit, and disabled radix caching. The seven greedy references used a
32-token output budget, with one input over the QSA selection budget of 2,048
tokens. The mixed suite sent five request types separately, then two of each
type concurrently. Request types were greedy, sampled, default sampling,
JSON-schema constrained, and logprobs.

| Serving route | Mixed requests passed | Greedy references reproduced | Stress requests | Observed aborts |
|---|---:|---:|---:|---:|
| Flash Next NVFP4 MTP7, CUDA graphs | 15/15 | 7/7 | 32 | 5 |
| Flash Next NVFP4 MTP7, eager | 15/15 | 7/7 | — | — |
| Flash Next NVFP4 AR, CUDA graphs | 15/15 | 7/7 | — | — |
| 27B NVFP4 DFlash2 tree and chain, CUDA graphs | 15/15 | 7/7 | 32 | 5 |

Both stress tests used eight clients and recorded zero request errors. Each
sent and observed five aborts. All reported pools returned to idle usage, and
all seven greedy outputs matched before and after the aborted requests.
The 27B server used the tree for single greedy requests and the chain for
sampled and concurrent requests. It captured private chain graphs for widths 8
and 16 and used the NVFP4 target and DFlash2 drafter pinned in the development plan.
[Checkpoint revisions](../decisions/2026-11-17-rtx-development-validation.md#checkpoint-identities)

Flash Next used the pinned NVFP4 target revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`, pinned-host FP8 PLE, explicit
`flashinfer_cutlass` MoE backends, deterministic inference, stable QSA Top-K, and
a QSA ring of 16 slots. MTP7 used the verified compact MTP snapshot. Ten target
shards were staged in RAM, and the PLE shard was read from persistent storage.

MTP7 graph and eager runs produced identical output tokens on all seven
references, including the 2,609-token input. The eager diagnostic reported
4,000 checked QSA compression groups with zero corrupt at its last periodic
count and emitted no overwrite or physical-RoPE mismatch messages.

The Flash Next AR control reproduced its own seven references after the mixed
requests. Three of seven references matched MTP7 across routes. The first
differing token appeared at zero-based position 6, 21, 20, and 11 for prompts
3 through 6 respectively. On the same inputs, the 32-token MTP7 outputs matched
the first 32 tokens of all seven archived RTX references for each of MTP3,
MTP7, and MTP12. The current replay establishes same-route repeatability on
these inputs. The cross-route Flash Next NVFP4 greedy gate remains open,
consistent with the earlier B200 comparisons of stock and ring-16 MTP against
AR. [Flash Next B200 results](../decisions/2026-11-17-qwen38-flash-next-feasibility.md)

## Replayed distribution checks

The current checker reproduced the archived 27B independent-cohort verdicts
at alpha 0.001. BF16 passed all 28 comparisons and NVFP4 failed six of 28.
These are offline checks of saved samples. The
[matched-prefix investigation](2026-11-18-main-deployment-and-nvfp4.md) records
the target-logit differences seen with both native DFlash and the plugin.
Cross-route sampled losslessness for NVFP4 remains unqualified.

## Evidence and scope

The pod matrix exited with status 0. After shutdown, the GPU reported 0 MiB
allocated and no compute processes, and the cgroup recorded zero OOM events.
The sibling directory `ax-engine-sglang-rtx-evidence-9ed3bdb/` held the
launch records, server logs, kernel checks, token and lifecycle results, and
the scripts used for this matrix. It was removed on 2026-11-19 after this
report was recorded on main. The checkpoints and staged copies stayed
on the pod.

The measured scope is two-slot serving correctness and lifecycle behavior on
one RTX PRO 6000. B200 throughput and H200 behavior were outside this campaign.
Flash Next tree verification still requires PLE and QSA tree support.
