# Verified profile gate and adaptive chain width on an RTX PRO 6000

`AX_QUALITY_PROFILE=verified` was started on a GPU for the first time, on one
RTX PRO 6000 (SM120, 97 GB) with Qwen3.8-27B NVFP4, the DFlash2 drafter and
SGLang 0.5.23. The chain route and AR equalled serialized AR on every sequence.
The tree route equalled it for 128 tokens but not for 1,024, so the profile now
refuses the tree route. The adaptive chain width did not beat the best fixed
width and stays experimental and off by default. A rerun with 200 distinct
prompts measured the chain width table and found no measurable cost in
switching widths between rounds, which a ten-prompt run had suggested.

Source: branch `v0.5.0-qa2` at `1c0df1d`. Raw outputs are in
`results/m2_v0.5.0/pro6000/` (not in Git).

## Exact output

`benchmarks/sglang/bench_27b_nvfp4.py --verified-profile` launches each lane
with `AX_QUALITY_PROFILE=verified` alone (plus `AX_TREE=0` and the draft block
for chain 8), so the launcher derives every other setting. Each pass sends the
ten frozen prompts (23 to 2,266 tokens) in order; a concurrent pass repeats the
list. Every sequence is compared with the AR lane's first pass at concurrency
one. Requests are greedy with `ignore_eos`.

| Run | Lane | C1 | C2 | C8 |
|---|---|---:|---:|---:|
| 2 slots, 128 tokens | ar, tree 16, chain 8 | 20/20 each | 80/80 each | – |
| 8 slots, 128 tokens | ar | 20/20 | 490/490 | 490/490 |
| | tree 16 | 20/20 | 490/490 | 490/490 |
| | chain 8 | 20/20 | 490/490 | 490/490 |
| 8 slots, 1,024 tokens | ar | 20/20 | 50/50 | 50/50 |
| | tree 16 | **16/20** | 50/50 | 50/50 |
| | chain 8 | 20/20 | 50/50 | 50/50 |

The sequences are ten distinct prompts repeated, so 1,000 sequences cover the
concurrency and scheduling combinations, not 1,000 inputs. The rows stay
`exact_measured`.

### The tree route

Two of the ten prompts left the AR sequence in the tree lane at concurrency
one, at output tokens 329 and 492, the same way in both passes. A tree server
verifies a single request with tree rounds and two or more with chain rounds,
which is why its C2 and C8 passes matched. All earlier parity evidence compared
128 tokens.

A fresh AR server given the prompt and the AR output up to each point reported
these top candidates:

| Prompt | Token | AR chose | Tree chose | AR log-probabilities of the two |
|---|---:|---:|---:|---|
| 1 | 329 | 5816 | 258 | 258: −0.753, 5816: −1.003 |
| 3 | 492 | 4338 | 2579 | 4338: −0.681, 2579: −0.806 |

The two candidates are 0.25 and 0.125 apart, and at token 329 the probe (which
prefilled the prefix) ranked them the other way round than the AR run (which
decoded it). The tree's verification pass therefore differs numerically from
AR's by enough to reorder close candidates. The cause was not investigated.

### Throughput of the verified lanes (output tok/s)

| Lane | C1 | C2 | C8, 128 tokens | C8, 1,024 tokens |
|---|---:|---:|---:|---:|
| ar | 52–54 | 105–109 | 362 | 356 |
| chain 8 | 109–113 | 199–207 | 489 | 561 |
| tree 16 (not admitted) | 126–132 | 203–220 | 499 | 592 |

### A launch default that did nothing

The first start printed `--disable-prefill-cuda-graph True [auto]` and then
captured 50 prefill graphs: SGLang builds `cuda_graph_config` before the
plugin's resolution hook runs. Fixed in `1c0df1d`; every result above is from
after the fix.

## Adaptive chain width

`benchmarks/sglang/ab_adaptive_width.py` starts one server per lane with the
launch defaults (no exact-output claim) and the same captured chain graphs
(`AX_CHAIN_GRAPH_WIDTHS="16 8"`): fixed width 16, fixed width 8, and
`AX_POLICY_MODE=adaptive`. 40 requests of 256 tokens per pass, two sweeps.
Output tok/s, first / second sweep:

| Concurrency | Width 16 | Width 8 | Adaptive (32 samples) | Adaptive (16 samples) |
|---|---:|---:|---:|---:|
| 2 | 285.8 / 286.5 | 262.9 / 267.1 | 268.7 / 269.2 | 260.9 / 272.4 |
| 4 | 467.8 / 468.2 | 468.0 / 468.2 | 460.1 / 464.0 | 474.9 / 449.6 |
| 8 | 583.9 / 584.9 | 698.6 / 699.4 | 678.2 / 706.3 | 665.2 / 714.2 |

These lanes ran with `--disable-radix-cache` (see below). The 40 requests are
the ten frozen prompts repeated, and the launch defaults make no exact-output
claim, so each lane generated different text; the 200-prompt rerun below
replaces the conclusions that depend on small differences.

- The best fixed width depends on the batch: width 16 is 8% faster with two
  requests, width 8 is 20% faster with eight (4.9% and 9.8% on 200 prompts).
- At concurrency eight the adaptive policy was 3% below fixed width 8 in its
  first sweep and 1% above in its second; the 16-sample lane spread from 665 to
  714 tok/s, so the lanes are not separable there.
- At concurrency two it stayed on width 16 (no switch in that batch-size group)
  and used about as many rounds as fixed width 16 (about 960 for 10,240
  tokens), yet ran 6% below it: about 2.4 ms more per round. Exploration
  rounds (6% of rounds, each about 3 ms slower) account for about 0.2 ms. The
  rest was not explained; on 200 prompts the gap was 2.3%.
- It switched 7 times in the three-to-four request group, where the two widths
  are equal: its score estimates crossed the 8% margin from noise.
- `AX_ADAPTIVE_MIN_SAMPLES=16` was not better than 32. The default stays 32.
- Round times above 0.5 s, which the policy drops, occurred once per run.

The release condition (5% above the best fixed configuration, no group more
than 5% slower) is not met.

### Review after the run

Two reviewers (gpt-6.2-sol and Qwen4 Max, given the code and the round
statistics) found the switch rule biased rather than only noisy: another
width's cost is one sampled round divided by the current width's last two
rounds, a bucket mixes batch sizes three and four, and four such comparisons
allowed a switch.

Changes after this run:

- `AX_CHAIN_WIDTH_BY_BATCH` selects the chain width by batch size from a fixed
  table (`1:16,5:8` for this checkpoint and GPU). vLLM offers the same form as
  `num_speculative_tokens_per_batch_size`. Measured below.
- The accept counter uses `scatter_add_` instead of `torch.bincount`, which on
  CUDA reads the input's range back to the host every round. The reviewers named
  it as the cause of the per-round loss. Its effect was not measured on its own
  (the rerun has no `bincount` lane); on ten prompts a similar loss appeared in a
  lane that runs no accept counter, and on 200 prompts the adaptive lane was
  2.3% below width 16 with two requests.
- A switch needs `AX_ADAPTIVE_MIN_SAMPLES` comparisons instead of four. The
  sampling every fourth round that collects them stops after four times the
  samples a clean warm-up needs, then falls back to the normal rate.
- The worker's `replay` statistics counted each target verify they saw twice:
  SGLang 0.5.23 asks `can_run_graph` in `prepare_for_verify` and again in
  `ModelRunner._forward_raw` with the same batch. They now count it once. No
  figure in this report was taken from them.

### Rerun with 200 prompts

Source: branch `chain-width-table` at `20e1492`, on another RTX PRO 6000 pod
with the same model, drafter and SGLang. Raw outputs are in
`results/m3_width_table/pro6000/` (not in Git).

A first rerun with the ten prompts (four times each) disagreed with the run
above: width 8 beat width 16 with two, three and four requests, and a fixed
schedule alternating 31 rounds of width 16 with one of width 8
(`AX_CHAIN_SCHEDULE=16x31,8x1`, no adaptive policy and no accept counter) was 4%
to 9% below width 16 with two requests (252 and 267 against about 278 tok/s).
Diagnostics on that loss:

- Round times did not change after a switch (25.9 ms at width 16, 24.1 ms at
  width 8). The loss was in accepted tokens per round, spread over all rounds of
  the schedule lane, not concentrated after switches.
- SGLang's coarse overlap barrier (`SGLANG_FORCE_COARSE_WAR_BARRIER=1`) left it
  unchanged.
- The greedy outputs differed between lanes and between repeats of one prompt
  within a lane. At the same text position the drafts of the schedule lane
  matched width 16's in 98% of cases before its first width-8 round and 35%
  after it, while accept lengths stayed equal in 94%. This fits a width-8 verify
  that differs numerically from a width-16 one, which changes the drafter's
  context and then the text; it was not traced further.

The comparison was therefore rerun on 200 distinct chat-templated prompts (80
UltraChat, 60 GSM8K, 60 HumanEval; up to 1,089 tokens), 256 output tokens each,
radix cache off, two sweeps. Every pass produced all 51,200 tokens. Output
tok/s, mean of the two sweeps, change against width 16:

| Lane | C2 | C4 | C8 |
|---|---:|---:|---:|
| width 16 | 346.3 | 565.9 | 782.5 |
| width 8 | 330.2 (−4.7%) | 582.8 (+3.0%) | 859.1 (+9.8%) |
| schedule `16x31,8x1` | 344.2 (−0.6%) | 562.6 (−0.6%) | 789.6 (+0.9%) |
| table `1:16,5:8` | 351.0 (+1.4%) | 567.9 (+0.3%) | 865.7 (+10.6%) |
| adaptive | 338.4 (−2.3%) | 590.1 (+4.3%) | 853.7 (+9.1%) |

- With two switches every 32 rounds, switching has no measurable cost: the
  schedule lane is within 1% of width 16 at every concurrency. The ten-prompt
  loss did not reproduce; the differences in generated text found above are
  the likely cause, but this was not isolated.
- The table matches the better fixed width with two and eight requests. With
  four it runs width 16, 2.6% below width 8; a split at three or four requests
  was not measured.
- The adaptive policy is 2.3% below width 16 with two requests, 1.2% above
  width 8 with four (sweeps 582.0 and 598.2) and 0.6% below width 8 with eight.
  It does not meet the release condition.
- The two concurrency-two sweeps of a lane differ by up to 3.4%: in every lane
  the first pass after the server starts is the slower one.

The comparison of the design with vLLM's dynamic speculative decoding (which
picks the token count per batch size in the scheduler, before the step) found no
reason to move the decision: the cost it would avoid was not measured.

Two gaps found in the review of these changes are older than them and remain
open: with `AX_SSM_POSITIONS` and `FC1_TREE64=1`, a chain width above the tree
width passes the launch check although the state pool is sized for the tree
width; and with `AX_CHAIN_DRAFT_WIDTHS`, a narrow draft round keeps full-width
position offsets if SGLang falls back from its Triton block preparation.

### Fewer running requests than asked

The first comparison ran with the radix cache on. SGLang then logged
`max_running_requests is capped to 3 by the mamba state cache
(max_mamba_cache_size=19, 5 state slots per request)`: with
`--max-running-requests 8` and `--mem-fraction-static 0.82` the server ran three
requests at a time. That run's C4 and C8 numbers are not concurrency 4 and 8
and are not reported here. The verified lanes turn the radix cache off and ran
eight requests.

## Jetson Thor (SM110, arm64)

Qwen3.5-9B with the DFlash v1 drafter, four slots, adaptive widths 16 and 8,
`AX_CHAIN_CHECK=1`, production services left running. This checks wiring and
state, not speed.

- `stress_multiuser.py`: 1,200 requests, no errors; 120 cancellations sent,
  119 seen; 24 reference prompts identical before and after the stress and
  between repeats.
- 45,728 state-allocation checks, none short or shrunk; 9 width switches
  (5,119 rounds at width 16, 5,441 at width 8).
- `ax-engine-sglang doctor` with the verified profile reports `unsupported`.

The CPU suite was not run in full on arm64: the uploaded source archive lacked
`scripts/` and the report files, and 20 tests failed on the missing files.
