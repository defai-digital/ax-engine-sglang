# RTX development validation (2026-11-17)

The user assigned an RTX PRO 6000 Blackwell Server Edition (96 GB class) for
Qwen 27B and Flash Next development, correctness testing, debugging and
improvements. B200 availability does not block this work. B200 performance
comparisons remain a separate final campaign.

## Sequence

1. Finish the maintenance refactor's pre/post comparison and software gates.
2. Check dense 27B NVFP4 and bf16 with identical checkpoints and seeds: plain
   decoding and plugin tree/chain serving, mixed request capabilities, aborted
   requests, slot reuse, long contexts, and sampled distributions.
3. Load Flash Next NVFP4 on the same RTX. Its 132.7 GB checkpoint includes an
   FP8 PLE table; use SGLang's pinned-host embedding support with explicit FP8
   storage instead of allocating that table on the 96 GB device. Check the real
   model with AR, MTP 3/7/12 and the QSA ring, including contexts past the sparse
   attention selection budget and eager ring diagnostics.
4. Attribute any failures to the plugin, engine, numerical contract or test
   harness. Fix reproducible implementation defects and rerun the affected
   controls. Keep unsuccessful runs; never turn an aborted pass into a result.
5. Measure any proposed tuning on the RTX under the same workload and precision.
   Do not transfer RTX throughput or precision evidence into a B200 claim.

The reusable runner is `scripts/check_model_regression.py`; its optional stress,
sampling, long-context, AR/MTP, QSA and eager controls use owned server process
groups. `benchmarks/sglang/stress_multiuser.py --input-ids` accepts an explicit token corpus
so lifecycle tests need no dataset download and can replay exactly the same inputs.

On SM120, set `--moe-runner-backend flashinfer_cutlass` and
`--speculative-moe-runner-backend flashinfer_cutlass` for Flash Next. The installed
SGLang 0.5.23 automatic NVFP4 MoE route failed at graph warmup with missing
`g1_scale_c`; explicit selection keeps weight preparation and execution aligned.
Run `gpu_nvfp4_moe_check.py` before another large checkpoint load. Retain failed
automatic-selection logs as evidence, alongside successful explicit-backend runs.

## Checkpoint identities

- Dense NVFP4: `RadixArk/Qwen3.8-27B-NVFP4` at
  `319f741cce68d7914884900c138a1fbb70a42f30`.
- Dense bf16: `Qwen/Qwen3.8-27B` at
  `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
- Dense drafter: `incoai/Qwen3.8-27B-DFlash2` at
  `015e795645c74b1a0eeef3b570031fb62e769bc5`.
- Flash Next NVFP4: `nvidia/Qwen3.8-Flash-Next-NVFP4` at
  `fc694b54fb0174e0913e6adf86691ef85a4ead47`.

Checkpoint shards stay outside the repository. Verify downloaded shard sizes
against Hub metadata; the existing network-volume cache contains empty files and
must not be treated as a complete snapshot. Do not include connection details in
records; use hardware descriptors and software/checkpoint versions.

## Interpretation

Sampled tests compare distributions, not seeded token identity. NVFP4 target
logits can vary with verify row count; a difference from one-token AR is not by
itself proof of a verifier implementation defect. Pair it with bf16 controls and
the target-only kernel tests. Flash Next tree verification still requires a PLE
and QSA tree port; the development runner rejects that unsupported route before
loading model weights instead of claiming that AR/MTP tests cover it.
