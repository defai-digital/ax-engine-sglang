# Verified profile gate and adaptive chain width on an RTX PRO 6000

`AX_QUALITY_PROFILE=verified` was started on a GPU for the first time, on one
RTX PRO 6000 (SM120, 97 GB) with Qwen3.8-27B NVFP4, the DFlash2 drafter and
SGLang 0.5.23. The chain route and AR equalled serialized AR on every sequence.
The tree route equalled it for 128 tokens but not for 1,024, so the profile now
refuses the tree route. The adaptive chain width did not beat the best fixed
width and stays experimental and off by default.

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

These lanes ran with `--disable-radix-cache` (see below).

- The best fixed width depends on the batch: width 16 is 8% faster with two
  requests, width 8 is 20% faster with eight.
- The adaptive policy reached width 8's rate at concurrency eight in its second
  sweep. At concurrency two it stayed on width 16 (no switch in that batch-size
  group) yet ran 6% below fixed width 16. That loss was not explained.
- It switched 7 times in the three-to-four request group, where the two widths
  are equal: its score estimates crossed the 8% margin from noise.
- `AX_ADAPTIVE_MIN_SAMPLES=16` was not better than 32. The default stays 32.
- Round times above 0.5 s, which the policy drops, occurred once per run.

The release condition (5% above the best fixed configuration, no group more
than 5% slower) is not met.

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
