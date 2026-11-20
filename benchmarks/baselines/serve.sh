#!/usr/bin/env bash
# Launch vLLM for Qwen3.6-35B-A3B in one decoding mode.
#   benchmarks/baselines/serve.sh ar      # plain autoregressive decoding
#   benchmarks/baselines/serve.sh mtp2    # built-in MTP speculative decoding, 2 draft tokens (mtp1..mtp9)
# Env overrides: MODEL, SERVED_MODEL, TP, PORT, MAX_LEN, GPU_UTIL, MTP_METHOD, EXTRA_ARGS
set -euo pipefail

MODE=${1:-ar}
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
TP=${TP:-1}                 # bf16 weights ~72 GB fit on one H200/B200
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-32768}   # benchmark contexts are short; keeps KV memory for concurrency
GPU_UTIL=${GPU_UTIL:-0.90}
MTP_METHOD=${MTP_METHOD:-qwen3_next_mtp}   # name from the Qwen3.6 model card; newer vLLM may use "mtp"

args=(serve "$MODEL"
  --served-model-name "${SERVED_MODEL:-qwen}"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --reasoning-parser qwen3
  --language-model-only
  --seed 0)

case "$MODE" in
  ar) ;;
  mtp[1-9])
    n=${MODE#mtp}
    args+=(--speculative-config "{\"method\":\"$MTP_METHOD\",\"num_speculative_tokens\":$n}")
    ;;
  *) echo "unknown mode: $MODE (use ar or mtp1..mtp9)" >&2; exit 2 ;;
esac

# shellcheck disable=SC2086
exec vllm "${args[@]}" ${EXTRA_ARGS:-}
