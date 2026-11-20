#!/usr/bin/env bash
# TensorRT-LLM DFlash2 baseline for Qwen3.8-27B on one GPU, measured with the same client as our lanes.
#
#   OUT=/dev/shm/ax/results/trt bash benchmarks/sglang/run_trtllm_dflash2.sh
#
# Serves the target with trtllm-serve (PyTorch backend) as plain decoding ("trt_ar") and with
# DFlashDecodingConfig ("trt_dflash2": the incoai DFlash2 drafter, max_draft_len 7 for its 8-position
# block, drafter cross-attention on the TRTLLM backend), then runs benchmarks/baselines/bench_speed.py with the
# greedy preset. TensorRT-LLM reports no acceptance through /metrics, so accept_len stays empty.
#   TRT_VENV   venv holding tensorrt_llm (default /root/venv-trt)
#   MODELS     "bf16 nvfp4" (default) - Qwen/Qwen3.8-27B and RadixArk/Qwen3.8-27B-NVFP4
#   TRT_LANES  "trt_ar trt_dflash2" (default)
#   CONCURRENCY "1 4 16", N_REQUESTS 64, MAX_TOKENS 1024, SETS (subset of code math chat),
#   PROMPT_FILE (ordered speed prompts, shared with run_baselines.sh)
#   TRT_MAX_BS 32, TRT_KV_FRACTION 0.85, PORT 30000, GPU 0
#   TRT_ITER_STATS=1  enable per-iteration stats and save every /metrics scrape during the benchmark to
#                     <lane>/iter_stats.jsonl (speculative acceptance; slightly perturbs timing). /metrics
#                     only answers with one HTTP frontend, so it implies TRT_FRONTENDS=1
#   TRT_FRONTENDS     --num_serve_frontends (TensorRT-LLM's default here is 8 HTTP frontend processes;
#                     SGLang serves through one tokenizer manager)
#   TRT_AUTOTUNE      0 skips autotuning, 1 enables it, unset keeps the engine default
#   NSYS_TRACE=1      capture the CUDA/NVTX timeline, NSYS_GRAPH_TRACE=graph|node
set -euo pipefail
cd "$(dirname "$0")/../.."
source benchmarks/sglang/nsys_trace.sh
nsys_trace_init || exit 2
case "${TRT_AUTOTUNE:-}" in
  0) trt_autotune=false ;;
  1) trt_autotune=true ;;
  "") trt_autotune= ;;
  *) echo "TRT_AUTOTUNE must be 0 or 1" >&2; exit 2 ;;
esac
OUT=${OUT:?set OUT}
TRT_VENV=${TRT_VENV:-/root/venv-trt}
MODELS=${MODELS:-"bf16 nvfp4"}
TRT_LANES=${TRT_LANES:-"trt_ar trt_dflash2"}
CONCURRENCY=${CONCURRENCY:-"1 4 16"}
N_REQUESTS=${N_REQUESTS:-64}
MAX_TOKENS=${MAX_TOKENS:-1024}
TRT_MAX_BS=${TRT_MAX_BS:-32}
PORT=${PORT:-30000}
TRT_FRONTENDS=${TRT_FRONTENDS:-$([[ ${TRT_ITER_STATS:-0} == 1 ]] && echo 1 || echo)}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# the pip OpenMPI refuses to run as root without these
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
DRAFT=${DRAFT:-incoai/Qwen3.8-27B-DFlash2}
prompt_args=()
if [[ -n "${PROMPT_FILE:-}" ]]; then prompt_args=(--prompt-file "$PROMPT_FILE"); fi
nvtx_env=()
if [[ "${NSYS_TRACE:-0}" == 1 ]]; then
  export TLLM_LLMAPI_ENABLE_NVTX=1
  nvtx_env=(-x TLLM_LLMAPI_ENABLE_NVTX)
fi
# local snapshot directory (snapshot_download refuses caches without non-weight files such as .gitattributes)
snap() {
  local d
  d=$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${1//\//--}"/snapshots/*/ 2> /dev/null | head -1)
  [[ -n $d && -f $d/config.json ]] || { echo "no local snapshot of $1" >&2; return 1; }
  echo "${d%/}"
}

failed=0
lane_index=0
pid=
poll=
stop_server() {
  if [[ -n "$poll" ]]; then
    kill "$poll" 2>/dev/null || true
    wait "$poll" 2>/dev/null || true
    poll=
  fi
  nsys_trace_cleanup
  if [[ -n "$pid" ]]; then
    kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    pid=
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for m in $MODELS; do
  model=$([[ $m == nvfp4 ]] && echo RadixArk/Qwen3.8-27B-NVFP4 || echo Qwen/Qwen3.8-27B)
  mpath=$(snap "$model")
  for lane in $TRT_LANES; do
    lane_index=$((lane_index + 1))
    dir="$OUT/${m}_$lane"
    mkdir -p "$dir"
    rm -f "$dir/$lane/speed_greedy.json" "$dir/iter_stats.jsonl"
    cfg="$dir/extra_llm_api_options.yaml"
    cat > "$cfg" <<EOF
max_batch_size: $TRT_MAX_BS
kv_cache_config:
  free_gpu_memory_fraction: ${TRT_KV_FRACTION:-0.85}
  enable_block_reuse: false
cuda_graph_config:
  enable_padding: true
  max_batch_size: $TRT_MAX_BS
EOF
    if [[ -n "$trt_autotune" ]]; then
      printf 'enable_autotuner: %s\n' "$trt_autotune" >> "$cfg"
    fi
    if [[ ${TRT_ITER_STATS:-0} == 1 ]]; then
      printf 'enable_iter_perf_stats: true\niter_stats_max_iterations: 100000\n' >> "$cfg"
    fi
    if [[ $lane == trt_dflash2 ]]; then
      cat >> "$cfg" <<EOF
speculative_config:
  decoding_type: DFlash
  max_draft_len: 7
  speculative_model: $(snap "$DRAFT")
  attention_backend: TRTLLM
EOF
    fi
    python3 - "$dir/lane.json" "$TRT_VENV" "$model" "$mpath" "$lane" "$cfg" "$CONCURRENCY" "$N_REQUESTS" "$MAX_TOKENS" "${TRT_FRONTENDS:-default}" <<'PY'
import json, os, subprocess, sys
from pathlib import Path

out, venv, model, model_path, lane, cfg, concurrency, requests, max_tokens, frontends = sys.argv[1:]
versions = subprocess.run([
    f"{venv}/bin/python", "-c",
    "import importlib.metadata as m, json; print(json.dumps({n: m.version(n) for n in ('tensorrt-llm', 'torch')}))",
], check=True, capture_output=True, text=True)
versions = json.loads(versions.stdout)
gpu = subprocess.run([
    "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
    "--query-gpu=name,driver_version", "--format=csv,noheader",
], capture_output=True, text=True).stdout.strip()
Path(out).write_text(json.dumps({
    "engine": "tensorrt_llm", "version": versions["tensorrt-llm"], "torch": versions["torch"],
    "gpu": gpu, "model": model, "model_path": model_path, "lane": lane,
    "config": Path(cfg).read_text(), "concurrency": concurrency, "n_requests": int(requests),
    "max_tokens": int(max_tokens), "frontends": frontends,
}, indent=1))
# lane.json keeps its scalar lane name for consumers of existing artifacts.
Path(out).with_name("benchmark_env.json").write_text(json.dumps({
    "lane": {name: os.environ.get(name, "") for name in (
        "NSYS_TRACE", "NSYS_GRAPH_TRACE", "NSYS_VERSION", "TRT_ITER_STATS", "TLLM_LLMAPI_ENABLE_NVTX",
    )},
}, indent=1))
PY
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; then
      echo "SERVER_FAILED $m $lane: port $PORT already serves something" >&2
      failed=1
      continue
    fi
    if ! nsys_trace_prepare "$lane_index" "$dir"; then failed=1; continue; fi
    # under mpirun the worker is already an MPI rank, so trtllm-serve does not need MPI Spawn (which the
    # pip OpenMPI 5 cannot do as a singleton in this container)
    PATH="$TRT_VENV/bin:$PATH" setsid "${nsys_launch[@]}" "$TRT_VENV/bin/mpirun" -n 1 --allow-run-as-root --oversubscribe -x CUDA_VISIBLE_DEVICES -x HF_HOME \
      -x HF_HUB_OFFLINE -x PATH "${nvtx_env[@]}" "$TRT_VENV/bin/trtllm-llmapi-launch" \
      "$TRT_VENV/bin/trtllm-serve" serve "$mpath" --host 127.0.0.1 --port "$PORT" \
      --max_seq_len 4096 --max_batch_size "$TRT_MAX_BS" --served_model_name qwen ${TRT_FRONTENDS:+--num_serve_frontends $TRT_FRONTENDS} \
      --extra_llm_api_options "$cfg" > "$dir/server.log" 2>&1 &
    pid=$!
    ok=0
    for _ in $(seq 1 360); do
      kill -0 "$pid" 2> /dev/null || break
      if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; then ok=1; break; fi
      sleep 5
    done
    if [[ $ok == 1 ]]; then
      if ! nsys_trace_start; then
        echo "PROFILE_FAILED $m $lane: could not start Nsight Systems" >&2
        failed=1
        stop_server
        continue
      fi
      if [[ ${TRT_ITER_STATS:-0} == 1 ]]; then
        curl -sf -m 5 "http://127.0.0.1:$PORT/metrics" > /dev/null || true      # drop warm-up iterations
        ( while true; do curl -sf -m 5 "http://127.0.0.1:$PORT/metrics" >> "$dir/iter_stats.jsonl" && echo >> "$dir/iter_stats.jsonl"; sleep 2; done ) &
        poll=$!
      fi
      SERVED_MODEL=qwen python benchmarks/baselines/bench_speed.py --mode "$lane" --out "$dir" --base-url "http://127.0.0.1:$PORT" \
        --preset greedy --max-tokens "$MAX_TOKENS" --n-requests "$N_REQUESTS" --concurrency $CONCURRENCY \
        --server-env "$dir/benchmark_env.json" "${prompt_args[@]}" \
        ${SETS:+--sets $SETS} || { echo "BENCH_FAILED $m $lane"; failed=1; }
      nsys_trace_stop || failed=1
      [[ -n $poll ]] && { sleep 3; curl -sf -m 5 "http://127.0.0.1:$PORT/metrics" >> "$dir/iter_stats.jsonl" || true; }
    else
      echo "SERVER_FAILED $m $lane"; tail -20 "$dir/server.log"; failed=1
    fi
    stop_server
    sleep 10
  done
done
exit $failed
