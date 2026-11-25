# TensorRT Edge-LLM on one Jetson Thor

Run on November 25, 2026 on Thor-02. Every lane used TensorRT Edge-LLM 0.12.0 and
`trtedge serve`, its OpenAI-compatible server. The same harness, prompts and
output limits as the earlier SGLang Thor runs drove the load
(`benchmarks/baselines/bench_speed.py`, greedy, thinking off). Edge-LLM has no
ax-engine-sglang integration. Each Edge-LLM lane is compared with the stock
SGLang 0.5.23 lane of the same method, and with the SGLang lane that uses
ax-engine-sglang.

| Item | Value |
|---|---|
| Host | Thor-02, SM110, 122 GB unified memory, L4T R39.2.0 |
| Engine | `tensorrt-edgellm[runtime]==0.12.0`, TensorRT 10.16.2.10, `jetson-thor-jp72-cu13-sm110` payload |
| 27B NVFP4 | `RadixArk/Qwen3.8-27B-NVFP4` at `319f741c`, the checkpoint used by the SGLang Thor runs |
| 27B BF16 | `Qwen/Qwen3.8-27B` at `37561816` |
| DFlash2 drafter | `z-lab/Qwen3.8-27B-DFlash2`, byte-identical to `incoai/Qwen3.8-27B-DFlash2` at `015e7956` used by SGLang |
| Flash Next NVFP4 | `nvidia/Qwen3.8-Flash-Next-NVFP4` at `fc694b54`; the SGLang and vLLM rows used `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| Driver | [`benchmarks/sglang/run_edgellm_thor.sh`](../../benchmarks/sglang/run_edgellm_thor.sh) |

The driver drops the page cache, starts one server per lane, waits for
`/v1/models`, runs the benchmark, and stops the server. The first start of each
engine profile built its TensorRT engines into the cache, and the server took
741 to 1,772 seconds to become ready. Timing starts after that.

## Qwen3.8-27B, concurrency 1

Output tok/s for code / math / chat. NVFP4 used 12 prompts per family and a
512-token limit. BF16 used 8 prompts and a 384-token limit. The SGLang numbers
come from the [Thor report](../../docs/reports/2026-11-19-jetson-thor.md)
(FlashInfer lanes).

| Route | NVFP4 | BF16 |
|---|---:|---:|
| Stock SGLang AR | 13.4 / 13.5 / 13.5 | 4.6 / 4.6 / 4.6 |
| Edge-LLM AR | 13.3 / 13.3 / 13.3 | 4.7 / 4.7 / 4.7 |
| Stock SGLang MTP3 | 36.2 / 35.2 / 27.0 | 13.5 / 12.9 / 9.9 |
| Edge-LLM MTP3 | 39.1 / 37.3 / 28.4 | 13.8 / 12.9 / 10.2 |
| Stock SGLang DFlash2 chain 16 | 93.0 / 73.5 / 30.6 | 43.1 / 29.6 / 12.5 |
| Edge-LLM DFlash2 chain 8 | 73.0 / 65.7 / 34.5 | 30.1 / 25.1 / 14.0 |
| Edge-LLM DDTree 32 (`draft_top_k` 4) | 71.4 / 68.8 / **44.2** | 30.1 / 28.8 / 18.2 |
| SGLang + ax-engine-sglang tree 32 | **106.1 / 92.1** / 43.1 | **47.5 / 37.6 / 18.3** |

- AR runs at the same speed on both engines. Edge-LLM's MTP3 is 0–8% faster
  than stock SGLang's.
- Edge-LLM's DFlash2 chain verifies 8 positions per round. It uses the
  drafter's `block_size` of 8, and the server reports `verify_tokens=None` for
  the chain. Stock SGLang verifies a chain of 16. The longer chain wins on code
  and math, and Edge-LLM's chain wins on chat.
- ax-engine-sglang tree 32 against Edge-LLM DDTree 32: 1.49× / 1.34× / 0.98×
  on NVFP4 and 1.58× / 1.31× / 1.01× on BF16. The two trees tie on chat.

## Qwen3.8-Flash-Next NVFP4, concurrency 1 and 4

Output tok/s at C1 / C4. 48 prompts per workload, 256-token limit, in-flight
batching with `--max-batch-size 4`, PLE table with `--ple-residency cached
--ple-cache-slots 4097`. The SGLang and vLLM rows come from the
[single-Thor report](../../docs/reports/2026-11-21-flash-next-single-thor.md).

| Configuration | Code | Math | Chat | Geomean |
|---|---:|---:|---:|---:|
| Edge-LLM AR | 17.3 / 49.1 | 17.2 / 51.5 | 17.5 / 50.1 | 29.5 |
| Edge-LLM MTP3 | 37.2 / 89.8 | 34.7 / 90.3 | 26.4 / 67.0 | 51.4 |
| Stock SGLang MTP3 | 42.3 / 96.2 | 38.8 / 92.2 | 28.9 / 65.4 | 54.9 |
| Stock vLLM MTP3 | 42.4 / 101.0 | 41.2 / 101.5 | 31.3 / 75.8 | 59.1 |
| SGLang adaptive MTP + ax-engine-sglang | 59.8 / 123.3 | 49.1 / 103.1 | 33.6 / 70.5 | 66.8 |
| vLLM adaptive K + ax-engine-vllm | **62.4 / 131.4** | **53.3 / 117.2** | **36.9 / 82.7** | **73.4** |

- Edge-LLM AR ran at 17.3 tok/s on code at C1. Stock SGLang AR ran at 20.6 in
  run 6 of the single-Thor report, which used 24 prompts.
- Edge-LLM MTP3 was 6% below stock SGLang MTP3 on the geometric mean. It led
  only on chat at C4.
- The two NVFP4 checkpoints differ, so this table compares serving stacks, not
  one set of weights.

## Failed and untested lanes

- `RadixArk/Qwen3.8-Flash-Next-NVFP4` did not start on Edge-LLM, either for AR
  or MTP3. The engines built. Loading the weights then failed with CUDA
  `misaligned address` in `checkpointReader.cpp`, and the server reported
  `--enable-in-flight-batching is not supported for this deployment`. This
  checkpoint stores experts in per-layer shards. NVIDIA's checkpoint loaded and
  served with in-flight batching.
- Flash Next BF16 was not run. Its weights are about 360 GB, and one Thor has
  122 GB of memory.
- The harness's Prometheus scrape fails on `trtedge serve`, so acceptance
  lengths were not recorded. Mean completion lengths match across the lanes of
  each family: 27B code 170 / 170 / 170 / 172 tokens, Flash Next code 160 / 159.
  The harness does not save generated text.

## Reproduce

```bash
python3 -m venv ~/ax-edgellm/.venv --system-site-packages
~/ax-edgellm/.venv/bin/pip install "tensorrt-edgellm[runtime]==0.12.0"
python3 -m venv ~/ax-edgellm/.venv-client && ~/ax-edgellm/.venv-client/bin/pip install openai datasets
# copy benchmarks/baselines to ~/ax-edgellm/repo/benchmarks/baselines, download the checkpoints to ~/ax-edgellm/models
TARGET_FN=~/ax-edgellm/models/fn-nvfp4-nvidia \
  LANES="27b-ar 27b-mtp3 27b-dflash2 27b-ddtree32 bf16-ar bf16-mtp3 bf16-dflash2 bf16-ddtree32 fn-ar fn-mtp3" \
  bash benchmarks/sglang/run_edgellm_thor.sh
```

The benchmark client needs its own virtual environment. With the system site
packages, the Ubuntu pandas build fails against newer numpy.
