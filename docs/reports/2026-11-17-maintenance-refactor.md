# Maintenance refactor validation (2026-11-17)

The implementation keeps the Rust/MLX engine and SGLang plugin in their existing
repositories. The [decision and plan](../decisions/2026-11-17-maintenance-boundaries.md)
describe the ownership and shared-data boundaries.

## Implemented behavior

- Validate plugin configuration before upstream mutations. QSA/RNG-only loading
  does not build the DFlash worker. Unknown SGLang versions require explicit opt-in.
- Roll back production startup attribute patches on failure. Compile SSM budget
  and allocation hooks together and set the installed marker only after success.
- Separate pure round policy from backend adaptation. Keep round state, planning
  lengths and fused buffers on the worker/model runner, including buffers captured
  before speculative-worker initialization.
- Make fast/lattice runtime modules package-owned while preserving research and
  legacy-overlay entry points.
- Put accepted-prefix and EOS/budget commit rules in the existing Rust core and
  use them in MLX linear/tree verification. Both repositories run the same pinned
  13-case conformance corpus and carry the benchmark evidence schema.
- Include versioned benchmark evidence in new token-identity sidecars, preserving
  existing fields and distinguishing unknown diagnostic state from measured claims.

## Software checks

Plugin tests passed on Python 3.10.22, 3.12.13 and 3.14.10: 20 tests on each
interpreter. Coverage includes startup rollback/retry, invalid option combinations,
route capability combinations, worker isolation, pre-capture state adoption,
the commit corpus, and independent wheel/sdist builds. The CI workflow runs the
same CPU checks on all three Python versions.

Existing lattice checks passed (3/3), paired-statistics tests passed, and the two
repositories' corpus/schema files match their SHA-256 manifests. The benchmark
schema accepted a valid evidence record and rejected an unsupported measurement
kind using a Draft 2020-12 validator.

AX Engine commit `8ace77c7b` passed all nine required software gates on Apple M2
Ultra 192 GiB, macOS 27.2.1, Rust 1.99.1, Python 3.12.13 and MLX 0.32.4:

- `cargo fmt --check` and Clippy with all targets/features and warnings denied.
- Workspace Rust tests: 4,703 passed, zero failed, 95 ignored by existing gates.
- `maturin develop` and Python tests: 344 passed, seven existing skips.
- `bash scripts/check-scripts.sh`.
- Both Qwen 27B / Flash Next qualification dry runs and primary-claims checks.

## CUDA development checks

Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB class (97,887 MiB
reported). Driver 595.112.07; Python 3.12.3; PyTorch 2.14.1+cu130; SGLang 0.5.23.
Synthetic kernel seed: 0.

| Check | Result |
|---|---|
| Device tree masks across prefix lengths and tree sizes | 90/90 pass |
| Alternating buffers and oversized-mask guard | pass |
| Tree GDN vs sequential recurrence, 9B/27B shapes, 16/32/64 nodes | 8/8 pass; maximum relative error 0.00410, gate 0.02 |
| Full-size 27B state-commit buffers across int32 offset boundaries | 4/4 pass; maximum relative error 2.53e-7 |
| Verify-to-commit recurrence seam, 32/64 nodes | 2/2 pass; maximum relative error 0.000666 |
| Real hook idempotence, worker buffer isolation, captured-buffer reuse, QSA ring wrap | 4/4 pass |
| QSA-only loading without DFlash worker generation | pass |
| Real SSM allocation/budget hook installation | pass |
| Late real-hook failure followed by rollback and successful retry | pass |
| Target-only sampled verifier, 100,000 trials per tree, vocab 128, seed 0 | valid trees max absolute z-score 3.0; duplicate-sibling negative control 65.9 |

These are development correctness controls. They do not qualify B200 performance.

## Model regression

Baseline: project commit `959c27e` (branding only, before this refactor).
Target: `RadixArk/Qwen3.8-27B-NVFP4` at
`319f741cce68d7914884900c138a1fbb70a42f30`. Drafter:
`incoai/Qwen3.8-27B-DFlash2` at `015e795645c74b1a0eeef3b570031fb62e769bc5`.

The reusable harness is `scripts/check_model_regression.py`. It uses local copies
of those snapshots, seed 0, a 16-node tree, draft block 16, chain width 8, private
chain graph widths 16/8, two running slots, context limit 4096 and 64-token
sequential references. Both lanes use identical flags, including disabled radix
cache, FlashInfer autotune and prefill graphs; decode graphs remain enabled.
Six code/math/chat references are compared before and after mixed greedy,
sampled, JSON-schema and logprob requests.

All six baseline/candidate greedy outputs were token-identical: lengths
35 / 64 / 64 / 64 / 52 / 64, 343 tokens total. Both lanes passed all 15 mixed
requests (five types alone, then ten concurrent requests) and reproduced their
six reference outputs afterward. The candidate package source SHA-256 was
`d0af26b3aecdedc8707c6eda654721e9954db022eadc9e4175b0860750fa101f`.

The run record is `/root/ax-refactor-model-check-6/result.json` on the development
pod. No throughput claim is made by this harness; B200 qualification remains a
separate campaign. Further RTX 27B/Flash Next development is recorded separately.
