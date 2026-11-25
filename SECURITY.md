# Security Policy

## Reporting a Vulnerability

Please report suspected security vulnerabilities privately — do not open a
public GitHub issue for a security report.

- Preferred: use [GitHub Security Advisories](https://github.com/defai-digital/ax-engine-sglang/security/advisories/new)
  for this repository.
- Alternative: email <enquiry@defai.digital> with a description of the issue,
  the affected version or commit, and reproduction steps if available.

We aim to acknowledge new reports within 5 business days and to keep you
updated as we investigate. Please give us reasonable time to investigate and
ship a fix before any public disclosure.

## Supported Versions

AX Engine for SGLang ships from a single, fast-moving line of development.
User-facing history lives on
[GitHub Releases](https://github.com/defai-digital/ax-engine-sglang/releases).
Security fixes are made against the latest release; we do not maintain parallel
long-term-support branches. If you are running an older version, please upgrade
before reporting an issue that may already be fixed.

## Scope

This repository is a Python add-on for SGLang on NVIDIA Linux. SGLang owns the
HTTP server, the scheduler, and CUDA execution. The add-on changes speculative
verification. Two boundaries are distinct from a conventional defect in this
package:

- **Model weights and prompts are untrusted input.** Loading arbitrary
  downloaded checkpoints, or serving arbitrary prompts, can exhaust memory or
  GPU time. Please note that distinction in your report.
- **The HTTP API is SGLang's.** Authentication, network exposure, and request
  limits follow the SGLang server you launch. A missing default control on that
  server is in scope here only when this package changes it.

Reports that show the add-on accepting tokens the target model did not decide,
or that show worker generation loading code other than the installed SGLang
sources, are in scope.
