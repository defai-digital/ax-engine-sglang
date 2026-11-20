# Code QA request: NCM-Draft training pipeline (before a multi-day 2x B200 run)

You are reviewing code read-only in the repository at /Volumes/Ext16TR0/code/ax-engine-sglang. Do not edit files.
Goal: find bugs that would waste GPU time or silently invalidate results. No style comments.

## What will run
`bash track_a/run_ncm.sh` on a pod with 2x B200 (183 GB each), vLLM 0.33, torch 2.14:
1. `phase2/gen_distill.py`: Qwen/Qwen3.8-27B greedy non-thinking self-distillation, 40k prompts per
   domain (code, math, chat) plus the benchmark prompts as val-only domains (bench_code/math/chat).
2. `track_a/blocks_from_distill.py`: 32-token blocks of the responses.
3. `phase1/train_tokenizer.py` + `phase1/vqvae.py` with `track_a/configs/tok_b32_k{6,5}.yaml`.
4. `phase2/train_drafter.py` + `phase2/drafter.py` + `phase2/distill_data.py` with
   `track_a/configs/{ncm_b32_k6,ncm_b32_k5,direct_b32,chunks_b32}.yaml`. The frozen 27B (HF
   AutoModelForCausalLM, bf16) runs online each step; hidden states at layers 5/19/33/47/61 are tapped
   with forward hooks and are the drafter's memory. `anchor: true` = serving convention: memory is
   valid only for positions before the last committed token; that token enters as an embedding.
5. `phase2/eval_drafter.py`: free-running draft, greedy verification by the same frozen 27B,
   accepted-prefix length and survival curve.
Design: `docs/decisions/2026-11-14-ncm-draft.md`.

## Check specifically
1. Information leak: can the drafter see any token at or after the block it predicts (memory mask,
   anchor off-by-one, hook outputs, context_view, block-diagonal masks, tokenizer codes at eval)?
   A leak gives fake acceptance. Also the opposite: is the last committed token really available?
2. Train/eval mismatch: does eval (`simulate_cached`, `draft_from_memory`) use exactly the
   conditioning used in training? Is the verifier's greedy token compared at the right offsets?
   Is the acceptance metric the standard "matching prefix, then +1 bonus token"?
3. NCM recipe: corruption schedule, clean labels, level weights, code dropout, null code, nested
   levels — any indexing error between levels (l_k sizes, upsampling, code offsets)?
4. Controls fairness: do direct_b32 / chunks_b32 get the same conditioning, capacity and data?
5. Tokenizer: nested dropout, commitment loss, dead codes, the 248k-way head memory/time, eval gate.
6. Data: prompt/response boundaries, padding, val split leakage (bench_* must never be trained on),
   mixing weights, windows that start mid-response, sequences longer than context_window.
7. Run script: resume logic, DONE markers, stage skipping, both lanes writing the same files,
   `set -uo pipefail` without -e, exit codes, GPU memory (27B bf16 55 GB + drafter + activations
   at windows=16 x context 2560), checkpoint size (must not save the frozen 27B), OOM risks,
   AMP / dtype mismatches between bf16 taps and the float32 drafter, NaNs.
8. Throughput traps: anything recomputed per block that should be once per window; hooks left
   registered; `torch.no_grad` missing around the frozen model; DataLoader on /dev/shm.

## Answer format
Numbered findings ordered by cost, each: file:line, what goes wrong, concrete fix. Mark each
CONFIRMED (you traced or reproduced it) or SUSPECTED. Then a short list "checked, correct".
Maximum 700 words.
