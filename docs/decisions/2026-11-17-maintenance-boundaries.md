# Runtime maintenance boundaries (2026-11-17)

## Decision and implementation plan

Keep AX Engine's Rust/MLX runtime and the SGLang plugin in their existing
repositories. Do not create a third runtime repository. Backend schedulers,
memory pools, kernels, sampling implementations, cost models and release gates
remain with the backend that executes them.

1. Parse and validate the plugin's startup configuration before installing any
   upstream hooks. Separate optional QSA/RNG fixes from the DFlash tree worker.
   Register the custom algorithm independently and load its hooks when requested.
2. Make startup attribute patches reversible; compile both SSM allocation and
   budget replacements before publishing either one. A failure must not mark
   installation complete or leave half of a pair installed.
3. Separate pure tree/chain policy from SGLang object adaptation. Keep round
   counters, planning lengths and fused GDN buffers on the worker, selected by
   execution context during initialization and each round.
4. Move `ax_fast.py` and `ax_lattice.py` into the installable package. Preserve
   `benchmarks/sglang/dft_*.py` imports and legacy DForkTree overlays as compatibility paths.
5. Extract accepted-prefix and token-cut rules into AX Engine's existing Rust
   core. Reuse them from both MLX linear and tree verification. Leave platform
   build/diagnostic APIs in their existing locations to preserve that public API.
6. Share a versioned greedy-commit corpus and benchmark evidence schema as data,
   with SHA-256 manifests and an explicit peer consistency check. Each runtime
   retains its own implementation and hardware qualification.

## Contracts

Canonical assets are in `contracts/`. AX Engine mirrors them under
`crates/ax-engine-core/tests/fixtures/` and checks their hashes in its Rust suite.
Run `python scripts/check_engine_contracts.py --peer /path/to/ax-engine` after
updating either mirror. Change both mirrors and their manifests together; use a
new schema version for incompatible semantics. Neither repository imports the
other's runtime or requires a sibling checkout to build or run its tests.

The commit corpus supplies target-selected token IDs, not floating-point logits.
It checks the first mismatch, bonus token, EOS inclusion, budget cuts, retained
draft rows, and malformed prediction lengths. It is a host oracle and does not
replace SGLang's sampled verifier or certify cross-device numerical identity.
Cancellation, cache isolation and sampled distributions retain backend-specific
tests because their state representations differ.

`benchmarks/sglang/token_identity.py` adds an `ax.engine.benchmark.v1` evidence object to
new completion sidecars. Workload wall time is measured around the requests;
concurrent request latencies are not summed into throughput. Unknown precision,
model revision, or hardware remains null. Without a server diagnostic record,
the measurement is `unclassified`. Existing rows and sidecar fields stay readable.

## Validation

CPU gates: plugin unit tests, shared corpus/hash checks, paired statistics,
lattice tests, and standalone wheel/sdist builds. Rust gates: core contract and
property tests, existing MLX tree/commit tests, and the repository's standard
software gates. GPU gates: RTX PRO 6000 Blackwell development controls, tree GDN
and state-commit kernels, QSA ring addressing, worker buffer isolation and a
baseline-versus-candidate model run. B200 throughput qualification is separate.

Validation results and limitations are recorded in the implementation report.
