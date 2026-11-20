# Review of runtime QA branch (2026-11-17)

Reviewed commit: `69cea59` on `fix/deepseek-runtime-qa-2026-11-17`, based on
`3083f82`. Integration target: `work/rtx-validation-20261117` at `5850eb2`.
The branch changes the regression harness, sampling validation, tests and
documentation. It does not change inference kernels or model weights.

The source branch subsequently advanced to `d04ee98` during validation. That
follow-up was also reviewed. It adds metadata checks, but its new
`--require-request-seeds` gate requires matching request seed bases and then runs
the same unpaired chi-square test (`track_l/sampling_dist_check.py:167-202`).
Matching metadata does not remove correlation between the two cohorts. The
single-lane seed offset and missing SciPy CI dependency also remain in that head.
The final integration merges both source commits with these findings resolved.
It retains `d04ee98`'s metadata count validation and collector negative control,
and rejects paired request-seed statistical comparisons. It does not import the
follow-up's request-seed verdict path.

## Findings and integration fixes

1. **P2: single-lane seed changed silently.** `lane_server_seed` at reviewed
   line 53 increments every sampled candidate seed, including candidate-only
   runs. A requested seed of 41 launched seed 42. Increment only for a paired
   server-RNG comparison; preserve the requested seed for a single lane.
2. **P2: request-seed comparisons can use an invalid independence assumption.**
   Reviewed lines 111-129 apply the unpaired chi-square test to matching explicit
   request seeds without the independent-cohort gate. Shared seeds can correlate
   observations. The integration retains request-seed collection for a single
   deterministic server and rejects this mode for paired statistical comparison.
   Paired distribution comparisons use separate server RNG seeds and the strict
   metadata/count gate.
3. **P2: CI skips the new statistical verdict tests.** The two SciPy-dependent
   tests are skipped when SciPy is absent, and `plugin-cpu.yml` did not install
   it. Add SciPy to the CPU workflow so both pass/fail controls run in CI.

The integration preserves the RTX stable-QSA flag, compact MTP snapshot support,
failure-debug controls and the newer hardware results. New regressions cover
single-lane seed preservation and rejection of correlated request cohorts.

## Validation and retest scope

The reviewed branch itself passed all 35 tests with SciPy on Python 3.12.
Its later `d04ee98` head passed 43 tests with SciPy on Python 3.12.13; those
tests do not establish the unpaired test's independence assumption.
The corrected integration passed all 38 tests, with no skips, on Python
3.10.22, 3.12.13 and 3.14.10. Contract hashes and the Draft 2020-12 schema check
also passed against the AX Engine peer. Python compilation and shell syntax
checks passed.

The final integration of `d04ee98` passes all 42 tests, with no skips, on the
same three Python versions. Positive integer metadata counts are required;
boolean, float, zero, negative and mismatched counts are rejected. Direct
request-seed statistical comparisons cannot receive a pass. The peer contract
hash/schema check passes again. Replaying the retained independent GPU samples
through the final comparator preserves the BF16 pass and NVFP4 failure below;
no new model inference is needed for this metadata-only integration.

The inference package is unchanged by this merge. Eight AR/MTP launch controls
(AR and MTP3/7/12, eager and graphs, without distribution sampling) were compared
against `5850eb2` and are identical. Existing passed kernel, greedy identity and
request-lifecycle results therefore remain applicable. The changed sampling
protocol was rerun with new independent cohorts, actual server seed metadata
and `--require-independent-seeds`: BF16 AR/chain passes 28/28 checks, while
NVFP4 AR/chain fails six of 28. Each lane has 2,000 complete six-token samples.
These compare serving routes, not before/after versions of the QA merge; the
NVFP4 failure does not by itself attribute a regression to this branch. The
previously withdrawn request-seed cohorts are not independent evidence.

Both reviewed source heads are preserved in the final main integration history.
Completed local branches and temporary worktrees can be removed after the
integration, while retaining raw GPU evidence outside Git. See the
[RTX report](../reports/2026-11-17-rtx-development.md) for the GPU controls and
separate hardware findings. NVFP4 cross-route lossless qualification remains
blocked by its observed distribution and greedy failures.
