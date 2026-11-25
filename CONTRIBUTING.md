# Contributing to AX Engine for SGLang

AX Engine welcomes community input primarily through reports, requests, and
evidence. We are not looking for unsolicited coding contributions at this stage.

The most helpful contributions are:

- Issue tickets for bugs, regressions, confusing behavior, install problems, or
  compatibility gaps
- Wishlist requests for models, routes, serving behavior, or documentation
- Reproducible performance benchmark results, especially with raw artifacts
- Benchmark methodology feedback
- Documentation corrections or clarification requests

We generally do not accept unsolicited code pull requests. This is especially
important for verification, draft trees, chain acceptance, QSA or MTP windows,
CUDA or Triton kernels, worker generation, launch defaults, and
performance-tuning changes. A change that improves one benchmark row can easily
regress acceptance, another model, another GPU, or a multi-request route.

If you believe a code change is necessary, open an issue first. Maintainers may
invite a scoped pull request after the expected behavior, evidence, and
validation gate are agreed.

## What To Send

### Issue Tickets

For bugs or regressions, include:

- The package version or commit, and the SGLang version
- Host details: Linux distribution, CPU architecture (`x86_64` or `aarch64`),
  GPU model, driver, and CUDA version
- The exact command, request payload, target checkpoint, drafter checkpoint,
  and launch flags
- Expected behavior and actual behavior
- Logs or terminal output when available
- Whether the issue reproduces on stock SGLang without the add-on, and which
  add-on route you used: DFlash tree, DFlash chain, MTP, or autoregressive

### Wishlist Requests

For product or model requests, include:

- The user workflow you want to support
- The model family, precision, context length, and serving route you care about
- Why the request matters compared with the currently documented routes
- Any reference behavior from stock SGLang, another runtime, or a paper

### Benchmark Results

Benchmark reports are especially useful when they include:

- The raw artifact directory or attached JSON and log files
- The exact command used to run the benchmark
- Package version or commit, and the SGLang version
- Host details: GPU model, memory, driver, and CUDA version
- Target and drafter checkpoint IDs, precision, and tokenizer source
- Prompt tokens, generated tokens, concurrency, repetitions, and seed
- The route under test: DFlash tree, DFlash chain, MTP, or autoregressive
- A stock SGLang row for the same checkpoint and prompts when you are comparing
  speed

Published comparisons live in `docs/performance.md`. Result records follow
`contracts/benchmark-result-v1.schema.json`.

## Performance Change Policy

Performance work is evidence-first. Do not send isolated speed patches without a
shared issue and validation plan. Runtime changes must preserve the rule that
the target model decides every emitted token, keep tree, chain, and MTP routes
distinct, and stay reproducible against the published benchmarks.

The primary measured target is Qwen3.8-27B NVFP4 on one NVIDIA B200, with
SGLang 0.5.23. A speed claim for a verification or kernel change needs that
route, plus a second board when the change can affect other GPUs. RTX PRO 6000
and Jetson Thor numbers are separate measurements and do not replace the B200
result. See [Measured throughput](README.md#measured-throughput).

Maintainers may still use community reports to create focused internal patches or
invite a narrow PR, but the default public contribution path is to submit the
issue, wishlist, benchmark artifact, or reproduction evidence first.

## Local Reproduction

These commands are useful for maintainers and for invited changes. Run them
from the repository root with Python 3.11 or 3.12:

```bash
python -m pip install 'setuptools>=77' wheel jsonschema scipy
python -m unittest discover -s ax-engine-sglang/tests -v
python scripts/check_engine_contracts.py --validate-schema
python -m compileall -q ax-engine-sglang/ax_engine_sglang
```

GPU checks are separate and need a matching NVIDIA board. See
`ax-engine-sglang/README.md`.

## License

By submitting issues, benchmark artifacts, reproduction details, documentation
suggestions, or invited changes, you agree that maintainers may use that material
in AX Engine documentation, tests, benchmarks, and project work under the
Apache License, Version 2.0 unless you clearly state otherwise.

## Contact

- Website: <https://automatosx.com>
- Discord: <https://discord.gg/MECsKdA6zF>
- Email: <enquiry@defai.digital>
- Company: DEFAI Private Limited (<https://defai.digital>)
