# Track M best-practice review (grok, muse, own source check)

Date: 2026-11-14. Question: does the Track M design (LM1B-recipe span tokenizer + Qwen3.5-9B hybrid
trunk as NCM generator) follow best practice, measured against the NCM paper/official code and
against other projects that converted pretrained autoregressive (AR) models to parallel decoding?

Reviewers: grok CLI and muse CLI (same brief, web search allowed), plus our own reading of the
primary sources. Claims below are marked **verified** only when we opened the source ourselves.

## Verdict

The trunk wiring (causal cached prefix, block-diagonal levels, restart of the recurrent layers from
the prefix state) matches the closest published systems. It was not yet best practice in four
places: the start state of the right-to-left scan (fixed in this commit), the missing shift /
auxiliary next-token loss, the planned adapter targets, and level conditioning/heads. No reviewed
project has converted pretrained causal recurrent layers to two-sided use with published benchmarks,
so that part remains our own experiment.

## What comparable projects do (verified)

| Project | Base | Conversion data | Recipe | Outcome |
|---|---|---|---|---|
| DiffuGPT / DiffuLLaMA (arXiv 2410.17891) | GPT-2 / LLaMA2-7B | 30B / ~65B tokens, full fine-tune | shift kept (state i predicts token i+1); mask annealing on GPT-2 only | no shift: GSM8K-symbolic 45.4 -> 33.5; no annealing: 45.4 -> 43.3; DiffuLLaMA far below LLaMA2 zero-shot (TriviaQA 18.5 vs 45.4) |
| Dream 7B (arXiv 2508.15487) | Qwen2.5-7B | ~580B tokens | shift kept; learning rate tuned so AR knowledge is not erased | HumanEval 57.9 vs 56.7, GSM8K 77.2 vs 78.9, MBPP 56.2 vs 63.6 |
| Fast-dLLM v2 (arXiv 2509.26328) | Qwen2.5-1.5B/7B | ~1-3B tokens, 2,500 steps (7B), lr 1e-5, batch 256 | block 32, two-sided inside the block, causal across blocks with a block cache, shift kept, complementary masks | average 60.3 vs 58.2 (NTP-tuned baseline); up to 2.5x throughput |
| BYOD (arXiv 2610.41574) | four 7-9B models | <=102.4M tokens, LoRA r=1024 on attention Q/V only, trunk frozen | - | average 49-51% vs 59-72% for the AR parents; no speedup claimed |
| BDLM Mamba-H (arXiv 2607.02805) | from scratch, 87M / 350M | 5B / 17.65B tokens | forward scan continues from the cached prefix state; reverse scan over the active block only, no prefix state; outputs summed; no timestep on the prefix | C4 perplexity 61.6 vs 83.7 for full-sequence two-sided Mamba and 76.5 for attention block diffusion (87M) |
| DiffuMamba (arXiv 2511.15927) | from scratch | - | separate weights per direction, outputs summed | - |
| Diffusion-Qwen3.5-9B (HF summerAIO) | Qwen3.5-9B | 20 QLoRA steps | `out = orig(h) + g * flip(orig(flip(h)))`, per-channel gate initialised to 0 | not benchmarked |
| Clover-35B-A3B-Preview (HF Diffuspace) | Qwen3.6-35B-A3B | not stated | extra Gated DeltaNet parameters (right-convolution weights, softmax controls), causal prefix, resolved blocks join the prefix | card claims best open diffusion LM; numbers only in its blog (not checked) |

Not verified by us: muse's "CLLM needs ~1M tokens" and "zero-init right-context branch" for Clover
(the card does not say so).

## Findings and rulings

| # | Topic | grok | muse | Ruling |
|---|---|---|---|---|
| 1 | Right-to-left scan started from the prefix-end state | wrong; start from zero or a learned state | wrong; same | **Changed**: empty state and empty conv tail (as BDLM Mamba-H eq. 12). 8/8 trunk checks pass. |
| 2 | Gate | keep shared weights + zero-init gate | per-channel gate; separate reverse conv weights | **Changed** to per layer and channel. Separate reverse conv weights are an M1 ablation, not the default. |
| 3 | RoPE positions (official: reset per level; ours: mapped to token positions) | mapped is right for a pretrained trunk | same, ablate | Keep mapped positions; add the official level embedding in M1; one reset-vs-mapped ablation. Levels never attend to each other, so equal positions across levels do not collide. |
| 4 | Causal prefix (paper: two-sided) | better practice here | right call | Keep. Same pattern as Fast-dLLM v2, BDLM Mamba-H and Clover. |
| 5 | Mask annealing | drop | add inside blocks | Not planned: the only ablation shows ~2 points on GPT-2 and the 7B conversion skipped it. Revisit only if M1 is unstable at the start. |
| 6 | Shift and AR loss | add shift readout on the finest level + next-token loss on the prefix | keep AR loss; distil from the AR teacher | **New for M1**: auxiliary next-token loss on the prefix (generated spans become causal prefix, so the prefix path must not drift); shift vs no-shift readout as the first M1 ablation. Largest verified effect of any recipe item (11 points). |
| 7 | Adapter targets | LoRA on 8 attention layers cannot teach 24 recurrent layers; adapt the DeltaNet projections/conv/gates | LoRA first, staged | LoRA on both layer types (attention q/k/v/o and DeltaNet in/out projections) first; full fine-tune of chosen layers only after the probe and smoke test. BYOD shows attention-only LoRA with ~100M tokens loses 10-20 points. |
| 8 | Data scale | ~1B tokens | staged | Plan ~1B tokens for M1 (we hold ~2.2B tokens of 128-token blocks). |
| 9 | Level 0 | last prefix state when l_0 = 1 | same | Adopt (the pretrained next-token state). |
| 10 | Block inputs and heads | projected cumulative z_hat + level embedding | same; tie heads to the codebook (12 untied heads = 12 x 8192 x 4096 = 403M parameters) | Adopt both. |
| 11 | Timestep conditioning | do not add | add level/timestep conditioning | Level embedding only; in BDLM Mamba-H a timestep made the prefix cache step-dependent and did not help. |
| 12 | Verification gaps | check that a committed span re-encoded causally matches a fresh forward pass; probe is ambiguous for the reverse scan | no evidence codes are predictable from the frozen trunk; no loss-decreases smoke test | Before M1: (a) frozen-trunk linear probes on all 12 levels against unigram baselines, (b) one-level LoRA smoke run, (c) reverse scan on/off and shift on/off measured by per-level code accuracy, (d) span-commit consistency check in the generation loop. A few GPU-hours in total. |
| 13 | Tokenizer | fix before M1 | same | Round trip is the gate. Step 4,000 of the LM1B run: token accuracy chat 99.3 / code 99.3 / math 99.9; GSM8K 86.3 -> 86.3; HumanEval 82.9 -> 65.9 (step 2,000: 39.0). Training continues to 25,957 steps. |

## Speed outlook (unchanged, stated again)

The best verified conversion of a comparable model reaches about 2.5x over AR (Fast-dLLM v2). One
128-token NCM span costs 12 trunk passes over 641 positions. No measured speedup exists for Track M
yet; any number before M1 is a projection.

## Code changed with this review

- `track_m/ncm_trunk.py`: right-to-left scan starts from an empty state and conv tail; gate per
  layer and channel.
- `track_m/test_ncm_trunk.py`: check 7 also verifies that a level run alone equals its slice of the
  all-levels pass with the gate on. Qwen3.5-9B on B200: 8/8 in float32 and bfloat16.

## Sources

- https://arxiv.org/html/2610.14162v1 and `models/next_concept.py`, `models/quant.py` at HazyResearch/ncm c702742
- https://arxiv.org/html/2410.17891v3
- https://arxiv.org/html/2508.15487v1
- https://arxiv.org/html/2509.26328v1
- https://arxiv.org/html/2610.41574v1
- https://arxiv.org/html/2607.02805v1
- https://arxiv.org/html/2511.15927v3
- https://huggingface.co/summerAIO/Diffusion-Qwen3.5-9B
- https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview
