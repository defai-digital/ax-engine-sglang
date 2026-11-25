## Related issue

AX Engine generally does not accept unsolicited code PRs (see
[CONTRIBUTING.md](../CONTRIBUTING.md)) — this is especially true for
verification, draft trees, chain acceptance, QSA or MTP windows, CUDA or
Triton kernels, worker generation, launch defaults, and performance-tuning
changes. If this PR wasn't invited by a maintainer after an issue discussion,
please open an issue first instead.

Closes #

## What changed and why

## Local reproduction

- [ ] `python -m unittest discover -s ax-engine-sglang/tests -v`
- [ ] `python scripts/check_engine_contracts.py --validate-schema`
- [ ] `python -m compileall -q ax-engine-sglang/ax_engine_sglang`

## Performance-sensitive change?

If this touches verification, kernels, worker generation, or launch defaults:
attach raw benchmark artifacts (not just summary numbers) per CONTRIBUTING.md's
"Performance Change Policy".
