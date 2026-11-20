# Stage S code QA (gpt-6.1-sol, read-only) — 2026-11-14

Review of `track_a/spine/` before the first GPU run. Reviewer output is reproduced below; the
disposition of each finding comes first.

| # | Finding | Disposition |
|---|---|---|
| 1 | Evaluator could not load SpecForge training checkpoints (`training_state.pt`) | Fixed: `eval_accept.py` reads `draft_state_dict` |
| 2 | Generation context differed from the captured context (system prompt; end-of-turn appended to truncated answers) | Fixed: `gen_data.py` sends the template's system prompt; the evaluator reports `greedy_consistency`, drops a non-greedy tail and starts after the empty think block. Rendering checked against the serving prompt on the pod (prefix match) |
| 3 | Shuffled-concept control leaked future states and depended on batching | Fixed: donor is position `anchor // 2` of the same sequence (causal, deterministic) |
| 4 | Checkpoint selection and reporting used the same set | Fixed: separate `val` (selection) and `eval` (test) splits |
| 5 | Failed generation was marked complete | Fixed: no `meta.json` and non-zero exit on request errors or < 97% answered |
| 6 | Auxiliary loss trained on unused anchor slots and padding | Fixed: anchor 0 excluded; real lengths passed |
| 7 | Refinement paths skipped the final concept injection | Fixed: concept is added before each layer's adjacent injection (`pre_injection` includes it) |
| 8 | Cached runs ignored changed inputs | Fixed: every cached step (data, conversion, capture, warm-start variants, training, evaluation) stores a fingerprint of its inputs (`fingerprint.py`); a mismatch stops the run instead of reusing the output |
| 9 | Interrupted summary write counted as success | Fixed: atomic write; the driver parses the summary before skipping |
| 10 | Server survived cancellation | Fixed: EXIT/INT/TERM traps |

After the fixes: `track_a/test_concept.py` and `track_a/spine/test_patch.py` pass; SpecForge's DSpine
test files pass (265 passed, 11 skipped) on the re-patched checkout; the evaluator runs on CPU on
four hand-written rows.

---

1. **[P1] Trained checkpoints cannot be loaded by the evaluator** — [eval_accept.py:34](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/eval_accept.py:34). The driver passes SpecForge’s training checkpoint directly, but `find_weights()` accepts only `model.safetensors`. The published SpecForge writer saves `training_state.pt` containing `draft_state_dict`, so post-training evaluation raises `FileNotFoundError`. **Fix:** load that state dictionary using `--draft-config`, or explicitly export before evaluation. [SpecForge checkpoint writer](https://github.com/sgl-project/SpecForge/blob/main/specforge/training/controller.py).

2. **[P1] Capture changes the prompt and the tokens being verified** — [gen_data.py:74](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/gen_data.py:74), [run_stage_s.sh:84](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:84). Generation sends only a user message; capture’s `qwen` template inserts a system message. Re-rendering also adds assistant scaffolding and a closing terminator, including for length-truncated answers. Consequently, the stored sequence is not necessarily the verifier’s greedy continuation under the captured context. **Fix:** preserve original prompt IDs, generated IDs, generation boundaries, and finish reason; capture those exact IDs and score only actual generated tokens. [SpecForge template](https://github.com/sgl-project/SpecForge/blob/main/specforge/data/template.py), [parser](https://github.com/sgl-project/SpecForge/blob/main/specforge/data/parse.py).

3. **[P1] Shuffling leaks future target information across anchors** — [concept.py:115](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/concept.py:115). Evaluation uses `B=1` and batches many anchors. Permuting `B*A` concepts can give an early anchor a concept computed from later target states. Proposals also depend on `--chunk` and fresh permutations, so batched scoring cannot equal sequential decoding for this arm. **Fix:** use deterministic donor concepts from an independent bank, keyed per example/anchor and independent of evaluation batching.

4. **[P1] The reported evaluation set also selects checkpoints** — [run_stage_s.sh:115](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:115), [run_stage_s.sh:137](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:137). Training selects `*-best` using `feat_eval`, then reports acceptance on that same set. This biases comparisons against released weights and any final confidence intervals. **Fix:** create separate validation and test splits; select checkpoints/LRs on validation and report test acceptance once.

5. **[P1] Failed generation is permanently marked complete** — [gen_data.py:109](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/gen_data.py:109). Request errors and empty answers are swallowed, yet `main()` writes `meta.json` and exits successfully. The driver thereafter skips generation. I reproduced this with every request failing: generation returned normally with an empty output. **Fix:** require the expected IDs/counts before publishing a completion marker; exit nonzero while required rows remain missing.

6. **[P2] Auxiliary loss trains on invalid anchor slots** — [patch_specforge.py:51](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/patch_specforge.py:51). The patch passes neither anchor validity nor real sequence lengths. SpecForge fills unused anchor slots with position zero; `next_pred` therefore supervises those slots against the beginning of the prompt. Short responses paired with long responses can receive mostly this spurious auxiliary loss. **Fix:** propagate `keep` and real lengths, exclude invalid anchors, and normalize over valid auxiliary targets. [SpecForge anchor sampler](https://github.com/sgl-project/SpecForge/blob/main/specforge/algorithms/common/dflash_family_model.py).

7. **[P2] Refinement bypasses the final concept injection** — [patch_specforge.py:55](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/patch_specforge.py:55), [eval_accept.py:104](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/eval_accept.py:104). Concepts are injected after adjacent injection, but refined scoring reconstructs the final output from `pre_injection` without reapplying the final concept. Thus unary candidates include that concept contribution while refined scores omit it; training’s refinement objective has the same omission. **Fix:** apply the final concept contribution before normalization in both refinement paths. [DSpine forward/refinement](https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/dspine.py).

8. **[P2] Cached runs silently ignore changed experiment inputs** — [run_stage_s.sh:113](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:113). Run names contain only arm/LR/seed. Changing `STEPS`, features, initialization, or recipe still hits `DONE`; the summary cache can then return the previous result as well. **Fix:** store and validate an input/config fingerprint for features, training runs, and evaluation outputs before skipping.

9. **[P2] An interrupted summary write becomes successful evaluation on retry** — [eval_accept.py:188](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/eval_accept.py:188), [run_stage_s.sh:96](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:96). `write_text()` creates/truncates the summary before finishing. Interruption or disk exhaustion can leave invalid JSON, but the next invocation returns success based solely on file existence. **Fix:** write atomically and validate the summary plus row counts before accepting it.

10. **[P2] Cancellation leaves the detached server running** — [run_stage_s.sh:51](/Volumes/Ext16TR0/code/ax-engine-sglang/track_a/spine/run_stage_s.sh:51). Cleanup occurs only after generation returns. There is no signal/exit trap, and `setsid` lets the server survive interruption of the driver, retaining GPU memory and the port. **Fix:** register group cleanup immediately after launch for `EXIT`, `INT`, and `TERM`, and reap the launcher with `wait`.

The walk arithmetic itself passed 48 boundary cases: advancing by matching prefix plus one and clipping at `last` is consistent. Ordinary block masking excludes other blocks and target positions `>= anchor`; the concrete cross-anchor leak above is the shuffled arm. [Mask implementation](https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/dflash_mask.py).

Review limitation: no GPU execution; `c0f0a39` was unavailable locally and remotely. Dependency-specific findings were checked against published SpecForge `main`, so those require confirmation against the pinned checkout.
