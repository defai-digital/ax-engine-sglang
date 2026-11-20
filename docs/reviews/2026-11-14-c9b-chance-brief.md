Odds review, no code. At most 450 words, numbered, blunt, give numbers. Do not run commands or modify files. You may read /Volumes/Ext16TR0/code/ax-engine-sglang/docs/decisions/2026-11-14-track-c-draftable-trunk.md, docs/reviews/2026-11-14-budget2000-review.md, results/track_b_decisive/REPORT.md, track_c/train_joint.py, and use the web.

The experiment now running (Track C, stage C1) on Qwen/Qwen3.5-9B:
- Trunk T = Qwen3.5-9B (hybrid linear/full attention, same family as the Qwen3.8-27B end target). Public block-16 drafter z-lab/Qwen3.5-9B-DFlash (1.29B parameters) as baseline B0 and as initialisation.
- Baseline B0 measured in our HF harness (greedy, thinking off, 512 new tokens), committed tokens per trunk pass: HumanEval 10.83 (164 prompts), GSM8K about 7.3 (first 51 prompts), relaxed acceptance ratio:0.1 gives 11.78 / 8.75. Teacher-forced accepted length on a held-out code/math/chat mix: 6.4. Chat alone is expected near 3-4.
- Data: 30k prompts (10k each code / math / chat), T's own greedy answers, about 18M tokens, 512-token answers.
- Arms, 2 epochs each, same drafter learning rate 2e-5: A0 drafter only (trunk frozen); A1 drafter + trunk LoRA rank 64 on attention and MLP (148M parameters, lr 1e-4), loss = block cross-entropy (position decay gamma 7, back-propagated through the trunk taps into the LoRA) + 1.0 * KL(T || T') on next-token distributions; A1k the same with KL weight 10. Decoding: drafter proposes 15 tokens, adapted trunk T' verifies exactly (free pass), quality = task accuracy of T' greedy vs T greedy.
- Gates fixed beforehand. C1: A1 >= 1.3x B0 tokens per pass (mean of code/math/chat) and >= 1.15x A0, task accuracy within 2% relative. C2 (<= 100 GPU-h in total, more data, agreement loss on T' own rollouts, corrupted prefixes, block 32): >= 1.8x B0 on average, chat >= 2x B0 and >= 6 tokens per pass, every quality metric >= 98% of T. Final bar at 27B: >= 3x measured tok/s over the best public drafter at <= 2% relative loss.

Questions.
1. Probability that C1 passes as specified. What is the most likely measured outcome (A0 vs B0, A1 vs A0, in tokens per pass and accuracy)? Name the mechanism that would make A1 beat A0, or say why there is none.
2. Probability that C2 passes at 9B, conditional on C1 passing and unconditional.
3. Probability of the final 27B bar given C2 passes at 9B, and unconditional.
4. Is anything in the C1 design likely to produce a false negative or a false positive (e.g. the block loss only reaching the trunk through the tap features, KL too weak or too strong, 18M tokens too few, exact verification against T' hiding quality drift, bf16 near-ties)? Give the single most valuable change if you had to change one thing before C2.
5. One line: the overall odds that Qwen3.5-9B ends with a result worth scaling to 27B, and what the most valuable fallback outcome is if it does not.
