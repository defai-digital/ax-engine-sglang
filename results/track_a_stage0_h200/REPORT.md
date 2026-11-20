# Speed and quality report: `/dev/shm/ax/results/track_a_stage0_h200`

## Environment (env_gpu0.json)

```
{
  "gpu": "NVIDIA H200, 143771 MiB, 595.91.07",
  "sglang": "0.5.22",
  "torch": "2.14.1+cu130 13.0",
  "python": "3.12.3",
  "model": "Qwen/Qwen3.8-27B",
  "precision": "bf16 weights, default KV cache dtype",
  "lane": {
    "MODES": "dspine dflash mtp7 mtp5 dspark ar",
    "AUTOTUNE": "",
    "RADIX_CACHE": "1",
    "BLOCK": "",
    "DRAFT": "",
    "EXTRA_ARGS": ""
  }
}
```

## Speed (greedy)

| mode | set | conc | out tok/s | per-req tok/s p50 | TTFT p50 (s) | accept len | vs ar | vs best MTP |
|---|---|---|---|---|---|---|---|---|
| dflash | code | 1 | 351.2 | 432.8 | 0.069 | 7.41 | - | 1.29x |
| dflash | code | 8 | 1492.9 | 231.5 | 0.095 | 7.55 | - | 1.24x |
| dflash | code | 32 | 2227.2 | 233.6 | 1.258 | 7.58 | - | 1.77x |
| dflash | math | 1 | 323.5 | 350.0 | 0.069 | 6.17 | - | 1.33x |
| dflash | math | 8 | 1494.1 | 236.7 | 0.099 | 6.10 | - | 1.14x |
| dflash | math | 32 | 2374.1 | 219.5 | 2.908 | 6.23 | - | 1.36x |
| dflash | chat | 1 | 178.0 | 183.8 | 0.065 | 3.28 | - | 1.29x |
| dflash | chat | 8 | 952.9 | 136.8 | 0.092 | 3.27 | - | 1.22x |
| dflash | chat | 32 | 1397.7 | 119.1 | 9.015 | 3.25 | - | 1.22x |
| dspine | code | 1 | 511.5 | 762.8 | 0.070 | 12.40 | - | 1.87x |
| dspine | code | 8 | 1035.8 | 377.0 | 0.101 | 12.24 | - | 0.86x |
| dspine | code | 32 | 1823.4 | 141.8 | 0.221 | 12.46 | - | 1.45x |
| dspine | math | 1 | 363.5 | 397.2 | 0.070 | 7.36 | - | 1.50x |
| dspine | math | 8 | 1952.0 | 282.6 | 0.072 | 7.33 | - | 1.49x |
| dspine | math | 32 | 2363.1 | 99.3 | 0.225 | 7.30 | - | 1.36x |
| dspine | chat | 1 | 154.6 | 159.5 | 0.067 | 2.99 | - | 1.12x |
| dspine | chat | 8 | 895.2 | 122.5 | 0.088 | 3.00 | - | 1.15x |
| dspine | chat | 32 | 1247.0 | 47.6 | 0.197 | 3.00 | - | 1.09x |
| mtp7 | code | 1 | 272.9 | 330.4 | 0.075 | 6.99 | - | 1.00x |
| mtp7 | code | 8 | 1201.2 | 188.0 | 0.116 | 6.93 | - | 1.00x |
| mtp7 | code | 32 | 1261.3 | 152.0 | 1.625 | 6.87 | - | 1.00x |
| mtp7 | math | 1 | 242.7 | 262.6 | 0.075 | 5.67 | - | 1.00x |
| mtp7 | math | 8 | 1312.2 | 188.6 | 0.104 | 5.69 | - | 1.00x |
| mtp7 | math | 32 | 1743.0 | 152.0 | 3.770 | 5.69 | - | 1.00x |
| mtp7 | chat | 1 | 138.0 | 143.5 | 0.072 | 3.16 | - | 1.00x |
| mtp7 | chat | 8 | 780.7 | 113.1 | 0.103 | 3.16 | - | 1.00x |
| mtp7 | chat | 32 | 1144.8 | 95.8 | 10.544 | 3.13 | - | 1.00x |

## Speed (qwen38)

| mode | set | conc | out tok/s | per-req tok/s p50 | TTFT p50 (s) | accept len | vs ar | vs best MTP |
|---|---|---|---|---|---|---|---|---|
| dflash | code | 1 | 222.5 | 262.8 | 0.070 | 4.16 | - | 1.54x |
| dflash | code | 8 | 1090.9 | 193.0 | 0.091 | 4.34 | - | 1.42x |
| dflash | code | 32 | 1416.8 | 172.1 | 6.093 | 4.21 | - | 1.47x |
| dflash | math | 1 | 248.9 | 284.1 | 0.069 | 4.73 | - | 1.53x |
| dflash | math | 8 | 935.6 | 194.5 | 0.090 | 4.70 | - | 1.38x |
| dflash | math | 32 | 1176.2 | 177.0 | 3.243 | 4.76 | - | 1.53x |
| dflash | chat | 1 | 170.2 | 183.0 | 0.071 | 3.15 | - | 1.49x |
| dflash | chat | 8 | 932.9 | 136.1 | 0.092 | 3.15 | - | 1.45x |
| dflash | chat | 32 | 1241.7 | 116.3 | 31.585 | 3.16 | - | 1.36x |
| dspine | code | 1 | 234.5 | 292.6 | 0.072 | 4.64 | - | 1.63x |
| dspine | code | 8 | 1147.0 | 223.8 | 0.073 | 4.66 | - | 1.49x |
| dspine | code | 32 | 1456.3 | 84.4 | 0.251 | 4.86 | - | 1.51x |
| dspine | math | 1 | 245.3 | 295.5 | 0.072 | 4.93 | - | 1.51x |
| dspine | math | 8 | 1056.4 | 218.2 | 0.073 | 5.05 | - | 1.56x |
| dspine | math | 32 | 1136.7 | 83.3 | 0.195 | 5.25 | - | 1.48x |
| dspine | chat | 1 | 158.9 | 165.4 | 0.073 | 3.11 | - | 1.39x |
| dspine | chat | 8 | 878.5 | 127.0 | 0.071 | 3.11 | - | 1.36x |
| dspine | chat | 32 | 1258.1 | 47.8 | 0.211 | 3.15 | - | 1.38x |
| mtp7 | code | 1 | 144.1 | 174.4 | 0.076 | 3.36 | - | 1.00x |
| mtp7 | code | 8 | 768.4 | 128.5 | 0.104 | 3.32 | - | 1.00x |
| mtp7 | code | 32 | 965.3 | 114.5 | 8.506 | 3.39 | - | 1.00x |
| mtp7 | math | 1 | 162.2 | 189.0 | 0.075 | 3.79 | - | 1.00x |
| mtp7 | math | 8 | 675.7 | 136.2 | 0.103 | 3.97 | - | 1.00x |
| mtp7 | math | 32 | 766.4 | 112.2 | 4.175 | 3.80 | - | 1.00x |
| mtp7 | chat | 1 | 114.0 | 121.1 | 0.077 | 2.61 | - | 1.00x |
| mtp7 | chat | 8 | 644.7 | 95.5 | 0.104 | 2.60 | - | 1.00x |
| mtp7 | chat | 32 | 914.0 | 79.5 | 39.767 | 2.57 | - | 1.00x |


## Reading (same GPU: one H200, SGLang 0.5.22 with the DSpine plugin, radix cache on for every mode)

The lane was stopped by the operator after MTP7; MTP5, DSpark and AR were not run on this GPU
(they exist on B200, `results/track_a_baselines/REPORT.md`), so the "vs ar" column is empty and
"vs best MTP" compares with MTP7 only. All numbers are measured end to end.

Greedy, concurrency 1, tok/s (accepted length):

| set | MTP7 | DFlash2 (block 8) | DSpine (block 16) | DSpine vs DFlash2 |
|---|---|---|---|---|
| code | 272.9 (6.99) | 351.2 (7.41) | 511.5 (12.40) | 1.46x |
| math | 242.7 (5.67) | 323.5 (6.17) | 363.5 (7.36) | 1.12x |
| chat | 138.0 (3.16) | 178.0 (3.28) | 154.6 (2.99) | 0.87x |

- DSpine is the best public drafter on code at concurrency 1 (1.46x DFlash2, 1.87x MTP7): its
  block of 16 is used (12.4 accepted), where DFlash2 is capped by its block of 8.
- On chat every method accepts about 3 tokens per round; DSpine's longer block buys nothing and
  its heavier round makes it slower than DFlash2.
- With sampling and thinking (`qwen38` preset) the accepted length falls to 4.2-5.3 on code and
  math for both drafters and they are within 5% of each other at concurrency 1.
- At concurrency 32 DSpine's per-request speed drops sharply (code 141.8 vs DFlash2 233.6 tok/s
  p50) although its time to first token stays low; total throughput is lower than DFlash2 on code
  and chat.
- DFlash2 on this H200 is slower than on B200 (code 351 vs 489 tok/s at concurrency 1); numbers
  from the two GPUs must not be mixed.

Consequence for the project: the best lossless drafter gains over the previous one where the
target's continuation is predictable (code) and not at all on chat. Tokens per round on chat,
about 3 for every public method, is the number any larger speed-up has to move.
