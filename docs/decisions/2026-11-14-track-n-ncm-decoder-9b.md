# Track N: NCM block decoder on the Qwen3.5-9B trunk, and a more aggressive Track C — 2026-11-14

## Why

The owner asked for (a) a small NCM test in parallel with Track C and (b) a more aggressive NCM
design for Qwen3.5-9B. Five reviewers put the chance that the conservative Track C stage C1 passes
at 12-25% and the chance that 9B produces a result worth scaling at 2-10%
(`docs/reviews/2026-11-14-c9b-chance-review.md`): C1 keeps the public one-pass block-16 drafter and
only lets the trunk adapt through its tap features.

## What "aggressive" can and cannot mean

- With an LLM trunk in the loop, every round has to run the trunk over the committed tokens to
  get the states for the next block, and that pass yields the trunk's own next-token logits at
  every position. Verification is therefore free: dropping it saves no compute. It only allows
  committing tokens the trunk disagrees with, which Track B showed destroys accuracy.
- The speed lever is the number of tokens committed per trunk pass. Per-round cost is about one
  trunk pass plus the block decoder; a perfect 32-token block at the 27B round time is about 3.8x
  DFlash2, a perfect 64-token block about 7x.
- So the aggressive design is: long blocks (32, then 64), a decoder built for long blocks (NCM:
  coarse-to-fine code levels, corruption training, so that the tokens of a block are decided
  jointly instead of independently), a trunk trained to be predictable in blocks, and relaxed
  acceptance (the Track B `ratio` rule) instead of exact match.
- A from-scratch native NCM student (no LLM trunk) is not run: two outside reproductions already
  show a small NCM losing to a same-size AR model on text with a 99.8% tokenizer
  (`docs/reviews/2026-11-14-ncm-evidence-notes/reproductions.md`), and it would not inform the 9B
  design. The NCM generator is tested where it could matter: as the block decoder on the 9B trunk.

## Track N stage N1 (running on the H200 pod)

The NCM-Draft design of `docs/decisions/2026-11-14-ncm-draft.md`, which was never run on a GPU,
retargeted to Qwen3.5-9B (`.internal/research/coarse_to_fine_decoding/run_n1.sh`, configs in `.internal/research/coarse_to_fine_decoding/configs/`):

- data: the Track C answers of Qwen3.5-9B (30k prompts), converted by `.internal/research/coarse_to_fine_decoding/jsonl_to_distill.py`;
- block tokenizer: 32-token blocks, residual VQ levels [1, 2, 4, 8, 16, 32], plan mode (the last
  decoder pass predicts tokens through the trunk's LM head);
- decoder `ncm_b32_k6`: 6 code levels then tokens (7 passes per block), input-code corruption with
  mean rate 0.5 decreasing over levels, conditioned on the frozen trunk's states at the taps of the
  public drafter; control `direct_b32`: 32 tokens in one pass, no hierarchy;
- metric: simulated lossless decoding on held-out answers (`.internal/research/ncm/drafting/eval_drafter.py`): tokens per
  trunk pass at draft lengths 8 / 16 / 32, and decoder latency.

Reading of N1 (thresholds fixed now): the NCM decoder is worth combining with the trunk adaptation
if, at draft length 32, it commits more tokens per trunk pass than the public block-16 drafter does
on the same data (teacher-forced accepted length 6.4) **and** more than the `direct_b32` control by
at least 15%. If it stays below the public drafter, the NCM hierarchy is dropped and Track C
continues with the DFlash-style decoder at block 32.

## Track C changes

- C1 runs as started (it is cheap and gives the drafter-only and tap-only reference points).
- C2 is no longer "more data": it adds what all five reviewers asked for. The adapted trunk and
  the drafter are rolled out on the training prompts; where the drafter's token is rated at least
  r times the original trunk's best token, the adapted trunk is trained to choose it (agreement
  loss on its own next-token logits); the drafter is trained on these rollouts with corrupted
  prefixes; the block loss is flat over the block; KL to the original trunk stays. Decoding uses
  `ratio` acceptance. Block 32 with the better of the two decoders from N1.
- The C2 gates are unchanged, plus two checks against a false positive: a blind pairwise chat
  judgement as a hard stop, and verification of the adapted system's output by the original trunk
  (share of committed tokens the original trunk rates below r).

Budget: N1 at most 15 GPU-hours on the H200; the C stages as before (ceiling 100 GPU-hours at 9B).
