#!/usr/bin/env bash
# Track A step 1: speed baselines for Qwen3.8-27B on SGLang, one GPU per lane.
#   GPU=0 PORT=30000 MODES="ar mtp3 mtp7" benchmarks/sglang/run_baselines.sh
#   GPU=1 PORT=30001 MODES="dflash dspark" benchmarks/sglang/run_baselines.sh
# Two settings per mode: greedy without thinking (parity setting) and the model card sampling with
# thinking on (the setting used by the public DFlash2 / DSpark numbers). Every mode sees the same
# request list (prompt i with seed i) at every concurrency level.
# Env: MODES, OUT, GPU, PORT, CONCURRENCY, N_REQUESTS, MAX_TOKENS, MAX_TOKENS_THINK, PRESETS,
#   SETS (greedy preset only: subset of code math chat), PROMPT_FILE (ordered speed prompts),
#   BENCHMARK_PRECISION (label for weights and KV cache), TORCH_PROFILE_STEPS / TORCH_PROFILE_START,
#   NSYS_TRACE=1 (CUDA/NVTX timeline), NSYS_GRAPH_TRACE=graph|node (default graph)
#      ("greedy qwen38" by default; "greedy" skips the sampled thinking setting; "ids" saves the
#      greedy output token ids with benchmarks/sglang/token_identity.py), and the
#      serve_sglang.sh variables (AUTOTUNE, RADIX_CACHE, BLOCK, DRAFT, EXTRA_ARGS)
set -uo pipefail
cd "$(dirname "$0")/../.."
source benchmarks/sglang/nsys_trace.sh
nsys_trace_init || exit 2
if [[ "${NSYS_TRACE:-0}" == 1 && -n "${TORCH_PROFILE_STEPS:-}" ]]; then
  echo "Run Nsight Systems and the SGLang torch profiler in separate passes" >&2
  exit 2
fi

MODES=${MODES:-"ar mtp3 dflash dspark"}
OUT=${OUT:-results/track_a_baselines}
GPU=${GPU:-0}
PORT=${PORT:-30000}
CONCURRENCY=${CONCURRENCY:-"1 8 32"}
N_REQUESTS=${N_REQUESTS:-64}
MAX_TOKENS=${MAX_TOKENS:-1024}
MAX_TOKENS_THINK=${MAX_TOKENS_THINK:-4096}
PRESETS=${PRESETS:-"greedy qwen38"}
export BASE_URL="http://127.0.0.1:${PORT}" SERVED_MODEL=qwen CUDA_VISIBLE_DEVICES=$GPU PORT
mkdir -p "$OUT"

python - "$OUT/env_gpu$GPU.json" <<'PY'
import json, subprocess, sys, platform, os
sys.path.insert(0, "benchmarks/baselines")
from common import DIAGNOSTIC_ENV_KEYS

def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return str(e)
tmp = sys.argv[1] + ".tmp"
json.dump({
  "gpu": sh("nvidia-smi -i %s --query-gpu=name,memory.total,driver_version --format=csv,noheader" % os.environ["CUDA_VISIBLE_DEVICES"]),
  "sglang": sh("python -c 'import sglang; print(sglang.__version__)'"),
  "torch": sh("python -c 'import torch; print(torch.__version__, torch.version.cuda)'"),
  "python": platform.python_version(),
  "model": os.environ.get("MODEL", "Qwen/Qwen3.8-27B"),
  "precision": os.environ.get("BENCHMARK_PRECISION") or None,
  "lane": {k: os.environ.get(k, "") for k in (*DIAGNOSTIC_ENV_KEYS, "MODES", "AUTOTUNE", "RADIX_CACHE", "BLOCK", "DRAFT", "EXTRA_ARGS",
           "PRESETS", "PROMPT_FILE", "PYTHONPATH", "AX_SITE_SHA", "AXPROF_SYNC", "AXPROF_NVTX", "NSYS_GRAPH_TRACE", "NSYS_VERSION",
           "AX_FAST", "AX_LATTICE", "AX_GDN_LAYERS", "AX_GDN_HEADS", "FC1_TREE64",
           "FC1_TREE_N", "FC1_DRAFT_BLOCK", "FC1_COMPILE_BUILD", "FC1_FUSED", "FC1_BRANCH_PILOT", "FC1_OURS",
           "FC1_NGRAM_PATH", "AX_TREE", "AX_CHAIN_WIDTH", "AX_CHAIN_WIDTH_BY_BATCH", "AX_POLICY_MODE", "AX_CHAIN_SCHEDULE", "AX_CHAIN_GRAPH_WIDTHS", "AX_CHAIN_DRAFT_WIDTHS", "AX_CHAIN_ATTN", "AX_FP8_PROBE", "AX_TREE_SAMPLING", "AX_FI_OFFSET_FIX", "AX_SSM_POSITIONS", "IDS_CONCURRENCY", "AX_QSA_RING", "IDS_CONTEXT_TOKENS")},
}, open(tmp, "w"), indent=2)
os.replace(tmp, sys.argv[1])
PY

SERVER_PID=""
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  nsys_trace_cleanup
  kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$SERVER_PID" 2>/dev/null || true
  # Scheduler children hold the GPU memory; kill those left on this GPU, but only if they belong to
  # the process group this lane started (setsid made SERVER_PID the group leader).
  for p in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader); do
    [[ "$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')" == "$SERVER_PID" ]] && kill -9 "$p" 2>/dev/null
  done
  SERVER_PID=""
  sleep 10
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Speculative algorithm a mode's server must report (empty = plain decoding).
expected_algorithm() {
  case "$1" in
    ar) echo "" ;; mtp*) echo "EAGLE" ;; dspark) echo "DSPARK" ;; dflash) echo "DFLASH" ;;
    dftree) echo "FC1_DFLASH_TREE" ;; dspine) echo "DSPINE" ;; *) echo "?" ;;
  esac
}

# Wait until the server we started answers /health; fail fast if it died. The answering server must
# also be ours: still alive after /health succeeds and reporting this mode's algorithm.
wait_ready() {
  for _ in $(seq 360); do
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE_URL/health")" == "200" ]]; then
      sleep 2
      kill -0 "$SERVER_PID" 2>/dev/null || { echo "a server answers on $PORT but ours died"; return 1; }
      got=$(curl -s -m 10 "$BASE_URL/get_server_info" | python -c 'import json,sys; print(json.load(sys.stdin).get("speculative_algorithm") or "")' 2>/dev/null || echo "?")
      [[ "$got" == "$(expected_algorithm "$mode")" ]] && return 0
      echo "server on $PORT reports algorithm '$got', expected '$(expected_algorithm "$mode")'"; return 1
    fi
    sleep 5
  done
  return 1
}

# DSpine cannot run without the radix prefix cache; the benchmark repeats prompts across
# concurrency levels, so a sweep that includes it must give every mode the same cache policy.
if [[ " $MODES " == *" dspine "* && "$MODES" != "dspine" && "${RADIX_CACHE:-0}" != "1" ]]; then
  echo "MODES contains dspine next to other modes: set RADIX_CACHE=1 so all modes use the radix cache" >&2
  exit 2
fi
export RADIX_CACHE=${RADIX_CACHE:-0}

failed=0
lane_index=0
prompt_args=()
if [[ -n "${PROMPT_FILE:-}" ]]; then prompt_args=(--prompt-file "$PROMPT_FILE"); fi
for mode in $MODES; do
  lane_index=$((lane_index + 1))
  echo "=== mode: $mode (GPU $GPU) $(date -u +%H:%M:%S) ==="
  # invalidate this mode's previous outputs before anything can fail, so a failed rerun never
  # leaves an older pass that looks complete (token_identity.py writes ids.meta.json last)
  mkdir -p "$OUT/$mode"
  rm -f "$OUT/$mode"/speed_*.json "$OUT/$mode"/ids*.jsonl "$OUT/$mode"/ids*.meta.json "$OUT/$mode"/stress.json
  if [[ " $PRESETS " == *" dist "* ]]; then
    rm -f "$OUT/$mode/dist.jsonl.metadata.json" "$OUT/$mode/dist.jsonl"
  fi
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE_URL/health")" == "200" ]]; then
    echo "SERVER_FAILED $mode: port $PORT already serves something"; failed=1; continue
  fi
  if ! nsys_trace_prepare "$lane_index" "$OUT/$mode"; then failed=1; continue; fi
  setsid "${nsys_launch[@]}" benchmarks/sglang/serve_sglang.sh "$mode" > "$OUT/$mode/server.log" 2>&1 &
  SERVER_PID=$!
  if ! wait_ready; then
    echo "SERVER_FAILED $mode"; grep -iE "error|traceback" "$OUT/$mode/server.log" | tail -5
    tail -5 "$OUT/$mode/server.log"; failed=1; stop_server; continue
  fi
  if ! nsys_trace_start; then
    echo "PROFILE_FAILED $mode: could not start Nsight Systems" >&2
    failed=1
    stop_server
    continue
  fi
  if [[ -n "${TORCH_PROFILE_STEPS:-}" ]]; then
    # SGLang torch profiler over TORCH_PROFILE_STEPS scheduler steps from step TORCH_PROFILE_START
    # (past warm-up); kernel times only, tok/s of this run is not a speed result
    curl -s -m 30 -X POST "$BASE_URL/start_profile" -H 'Content-Type: application/json' \
      -d "{\"output_dir\": \"$OUT/$mode/torch_profile\", \"start_step\": ${TORCH_PROFILE_START:-300}, \"num_steps\": $TORCH_PROFILE_STEPS, \"activities\": [\"CPU\", \"GPU\"]}" \
      || echo "TORCH_PROFILE request failed"
    echo
  fi
  if [[ " $PRESETS " == *" greedy "* ]]; then
    # shellcheck disable=SC2086
    python benchmarks/baselines/bench_speed.py --mode "$mode" --out "$OUT" --preset greedy --max-tokens "$MAX_TOKENS" \
        --server-env "$OUT/env_gpu$GPU.json" "${prompt_args[@]}" \
        --n-requests "$N_REQUESTS" --concurrency $CONCURRENCY ${SETS:+--sets $SETS} || { echo "BENCH_FAILED $mode greedy"; failed=1; }
  fi
  if [[ " $PRESETS " == *" ids "* ]]; then
    # IDS_CONCURRENCY: requests in flight per pass (default 1); c > 1 writes ids_c<c>.jsonl (identity
    # checks under load, not speed)
    for ic in ${IDS_CONCURRENCY:-1}; do
      f="$OUT/$mode/ids.jsonl"; [[ "$ic" == 1 ]] || f="$OUT/$mode/ids_c$ic.jsonl"
      python benchmarks/sglang/token_identity.py collect --out "$f" --n "$N_REQUESTS" --concurrency "$ic" \
          --max-tokens "$MAX_TOKENS" --expect-algorithm "$(expected_algorithm "$mode")" \
          --server-env "$OUT/env_gpu$GPU.json" --context-tokens "${IDS_CONTEXT_TOKENS:-0}" ${SETS:+--sets $SETS} \
          || { echo "BENCH_FAILED $mode ids c=$ic"; failed=1; }
    done
  fi
  if [[ " $PRESETS " == *" stress "* ]]; then
    # multi-user edge cases and pool accounting (benchmarks/sglang/stress_multiuser.py)
    python benchmarks/sglang/stress_multiuser.py --base-url "$BASE_URL" --server-log "$OUT/$mode/server.log" --model "$MODEL" \
        --out "$OUT/$mode/stress.json" || { echo "BENCH_FAILED $mode stress"; failed=1; }
  fi
  if [[ " $PRESETS " == *" mixed "* ]]; then
    # greedy / sampled / JSON-schema / logprob requests alone and together (benchmarks/sglang/mixed_requests_check.py)
    python benchmarks/sglang/mixed_requests_check.py --base-url "$BASE_URL" --out "$OUT/$mode/mixed.json" \
        || { echo "BENCH_FAILED $mode mixed"; failed=1; }
  fi
  if [[ " $PRESETS " == *" dist "* ]]; then
    # sampled output distribution, one request in flight (benchmarks/sglang/sampling_dist_check.py collect)
    python benchmarks/sglang/sampling_dist_check.py collect --base-url "$BASE_URL" --model "${TOKENIZER:-$MODEL}" \
        --out "$OUT/$mode/dist.jsonl" --n "${DIST_N:-500}" --seed0 "${DIST_SEED0:--1}" || { echo "BENCH_FAILED $mode dist"; failed=1; }
  fi
  if [[ " $PRESETS " == *" teacher "* ]]; then
    # teacher-forced check of saved outputs on this (stock, radix-cache-off) server:
    # TEACHER_IDS="lane/ids.jsonl ...", TEACHER_OUT=file (benchmarks/sglang/sglang_teacher_check.py)
    # shellcheck disable=SC2086
    python benchmarks/sglang/sglang_teacher_check.py --base-url "$BASE_URL" --server-log "$OUT/$mode/server.log" \
        --model "${TOKENIZER:-$MODEL}" --ids $TEACHER_IDS --out "$TEACHER_OUT" \
        || { echo "BENCH_FAILED $mode teacher"; failed=1; }
  fi
  if [[ " $PRESETS " == *" qwen38 "* ]]; then
    # shellcheck disable=SC2086
    python benchmarks/baselines/bench_speed.py --mode "$mode" --out "$OUT" --preset qwen38 --thinking \
        --server-env "$OUT/env_gpu$GPU.json" "${prompt_args[@]}" \
        --max-tokens "$MAX_TOKENS_THINK" --n-requests "$N_REQUESTS" --concurrency $CONCURRENCY \
        || { echo "BENCH_FAILED $mode qwen38"; failed=1; }
  fi
  nsys_trace_stop || failed=1
  stop_server
done
echo "LANE_DONE gpu=$GPU failed=$failed $(date -u)"
exit $failed
