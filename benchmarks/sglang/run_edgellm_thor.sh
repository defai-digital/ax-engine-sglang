#!/usr/bin/env bash
# TensorRT Edge-LLM lanes on one Jetson Thor, measured with the shared speed harness.
#
# Runs each lane in LANES back to back: drop the page cache, start `trtedge serve`
# (the first start of a profile builds and caches its engines), wait for /v1/models,
# run benchmarks/baselines/bench_speed.py, then stop the server. A failed lane is
# recorded and the queue continues.
#
# Env: ROOT (work dir holding .venv, .venv-client, repo/, models/), OUT, LANES,
#      TARGET_27B, TARGET_BF16, DRAFT_27B, TARGET_FN, READY_TIMEOUT (seconds, default 14400), PORT.
# Lanes:
#   27b-ar | 27b-mtp3 | 27b-dflash2 | 27b-ddtree32   Qwen3.8-27B NVFP4, C1, 12 prompts/set, 512 tokens
#   bf16-ar | bf16-mtp3 | bf16-dflash2 | bf16-ddtree32  Qwen3.8-27B BF16, C1, 8 prompts/set, 384 tokens
#   fn-ar  | fn-mtp3                                  Qwen3.8-Flash-Next NVFP4, C1 and C4, 48 prompts/set, 256 tokens
#   fn-ar-blocking | fn-mtp3-blocking                 the same without in-flight batching (one request at a time)
set -euo pipefail

ROOT=${ROOT:-$HOME/ax-edgellm}
OUT=${OUT:-$ROOT/out}
PORT=${PORT:-8000}
READY_TIMEOUT=${READY_TIMEOUT:-14400}
TARGET_27B=${TARGET_27B:-$ROOT/models/qwen38-27b-nvfp4}
DRAFT_27B=${DRAFT_27B:-$ROOT/models/zlab-dflash2}
TARGET_BF16=${TARGET_BF16:-$ROOT/models/qwen38-27b-bf16}
TARGET_FN=${TARGET_FN:?set TARGET_FN to the Flash Next NVFP4 checkpoint directory}
LANES=${LANES:-"27b-ar 27b-mtp3 27b-dflash2 27b-ddtree32 fn-ar fn-mtp3"}
CACHE=$ROOT/cache
mkdir -p "$OUT" "$CACHE"

lane_args() {
  local fn_args=(--max-batch-size 4 --enable-in-flight-batching --ple-residency cached --ple-cache-slots 4097)
  local fn_blocking=(--ple-residency cached --ple-cache-slots 4097)
  case "$1" in
    27b-ar)       MODEL=$TARGET_27B; ARGS=(); BENCH=(--n-requests 12 --max-tokens 512 --concurrency 1) ;;
    27b-mtp3)     MODEL=$TARGET_27B; ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
                  BENCH=(--n-requests 12 --max-tokens 512 --concurrency 1) ;;
    27b-dflash2)  MODEL=$TARGET_27B; ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_27B\"}")
                  BENCH=(--n-requests 12 --max-tokens 512 --concurrency 1) ;;
    27b-ddtree32) MODEL=$TARGET_27B
                  ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_27B\",\"draft_top_k\":4,\"verify_tokens\":32}")
                  BENCH=(--n-requests 12 --max-tokens 512 --concurrency 1) ;;
    bf16-ar)      MODEL=$TARGET_BF16; ARGS=(); BENCH=(--n-requests 8 --max-tokens 384 --concurrency 1) ;;
    bf16-mtp3)    MODEL=$TARGET_BF16; ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
                  BENCH=(--n-requests 8 --max-tokens 384 --concurrency 1) ;;
    bf16-dflash2) MODEL=$TARGET_BF16; ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_27B\"}")
                  BENCH=(--n-requests 8 --max-tokens 384 --concurrency 1) ;;
    bf16-ddtree32) MODEL=$TARGET_BF16
                  ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_27B\",\"draft_top_k\":4,\"verify_tokens\":32}")
                  BENCH=(--n-requests 8 --max-tokens 384 --concurrency 1) ;;
    fn-ar)        MODEL=$TARGET_FN; ARGS=("${fn_args[@]}"); BENCH=(--n-requests 48 --max-tokens 256 --concurrency 1 4) ;;
    fn-mtp3)      MODEL=$TARGET_FN; ARGS=("${fn_args[@]}" --speculative-config '{"method":"mtp","num_speculative_tokens":3}')
                  BENCH=(--n-requests 48 --max-tokens 256 --concurrency 1 4) ;;
    fn-ar-blocking)   MODEL=$TARGET_FN; ARGS=("${fn_blocking[@]}"); BENCH=(--n-requests 48 --max-tokens 256 --concurrency 1 4) ;;
    fn-mtp3-blocking) MODEL=$TARGET_FN; ARGS=("${fn_blocking[@]}" --speculative-config '{"method":"mtp","num_speculative_tokens":3}')
                  BENCH=(--n-requests 48 --max-tokens 256 --concurrency 1 4) ;;
    *) echo "unknown lane $1" >&2; return 1 ;;
  esac
}

wait_ready() {
  local pid=$1 deadline=$((SECONDS + READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    kill -0 "$pid" 2>/dev/null || return 1
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && return 0
    sleep 10
  done
  return 1
}

for lane in $LANES; do
  dest=$OUT/$lane
  if [[ -f $dest/DONE ]]; then echo "skip $lane (done)"; continue; fi
  rm -rf "$dest"; mkdir -p "$dest"
  lane_args "$lane"
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || true
  cmd=("$ROOT/.venv/bin/trtedge" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name qwen
       --cache-dir "$CACHE" --engine-cache-max-size-gb 300 "${ARGS[@]}")
  printf '%q ' "${cmd[@]}" > "$dest/command.txt"; echo >> "$dest/command.txt"
  echo "$(date -Is) start $lane"
  started=$SECONDS
  setsid "${cmd[@]}" > "$dest/server.log" 2>&1 &
  pid=$!
  if ! wait_ready "$pid"; then
    echo "$(date -Is) FAILED $lane: server not ready (see $dest/server.log)"
    kill -TERM -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    continue
  fi
  echo "ready_s=$((SECONDS - started))" > "$dest/ready.txt"
  echo "$(date -Is) ready $lane after $((SECONDS - started)) s"
  if (cd "$ROOT/repo" && HF_HOME=$ROOT/hf BASE_URL=http://127.0.0.1:$PORT SERVED_MODEL=qwen \
        "$ROOT/.venv-client/bin/python" benchmarks/baselines/bench_speed.py --mode "edgellm-$lane" \
        --out "$dest/bench" --preset greedy --sets code math chat "${BENCH[@]}") > "$dest/bench.log" 2>&1; then
    touch "$dest/DONE"; echo "$(date -Is) done $lane"
  else
    echo "$(date -Is) FAILED $lane: benchmark (see $dest/bench.log)"
  fi
  kill -TERM -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  sleep 5
done
echo "$(date -Is) ALL-DONE"
