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
#   CONCURRENCY "1 4 16", N_REQUESTS 64, MAX_TOKENS 1024, SETS (subset of code math chat)
#   TRT_MAX_BS 32, TRT_KV_FRACTION 0.85, PORT 30000, GPU 0
#   TRT_ITER_STATS=1  enable per-iteration stats and save every /metrics scrape during the benchmark to
#                     <lane>/iter_stats.jsonl (speculative acceptance; slightly perturbs timing). /metrics
#                     only answers with one HTTP frontend, so it implies TRT_FRONTENDS=1
#   TRT_FRONTENDS     --num_serve_frontends (TensorRT-LLM's default here is 8 HTTP frontend processes;
#                     SGLang serves through one tokenizer manager)
set -euo pipefail
cd "$(dirname "$0")/../.."
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
# local snapshot directory (snapshot_download refuses caches without non-weight files such as .gitattributes)
snap() {
  local d
  d=$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${1//\//--}"/snapshots/*/ 2> /dev/null | head -1)
  [[ -n $d && -f $d/config.json ]] || { echo "no local snapshot of $1" >&2; return 1; }
  echo "${d%/}"
}

failed=0
for m in $MODELS; do
  model=$([[ $m == nvfp4 ]] && echo RadixArk/Qwen3.8-27B-NVFP4 || echo Qwen/Qwen3.8-27B)
  mpath=$(snap "$model")
  for lane in $TRT_LANES; do
    dir="$OUT/${m}_$lane"
    mkdir -p "$dir"
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
    python3 - "$dir/lane.json" <<PY
import json, subprocess, sys
v = subprocess.run(["$TRT_VENV/bin/python", "-c", "import tensorrt_llm; print(tensorrt_llm.__version__)"],
                   capture_output=True, text=True).stdout.strip().splitlines()
json.dump({"engine": "tensorrt_llm", "version": v[-1] if v else None, "model": "$model", "lane": "$lane",
           "config": open("$cfg").read(), "concurrency": "$CONCURRENCY", "n_requests": $N_REQUESTS,
           "max_tokens": $MAX_TOKENS, "frontends": "${TRT_FRONTENDS:-default}"}, open(sys.argv[1], "w"), indent=1)
PY
    # under mpirun the worker is already an MPI rank, so trtllm-serve does not need MPI Spawn (which the
    # pip OpenMPI 5 cannot do as a singleton in this container)
    PATH="$TRT_VENV/bin:$PATH" setsid "$TRT_VENV/bin/mpirun" -n 1 --allow-run-as-root --oversubscribe -x CUDA_VISIBLE_DEVICES -x HF_HOME \
      -x HF_HUB_OFFLINE -x PATH "$TRT_VENV/bin/trtllm-llmapi-launch" \
      "$TRT_VENV/bin/trtllm-serve" serve "$mpath" --host 127.0.0.1 --port "$PORT" \
      --max_seq_len 4096 --max_batch_size "$TRT_MAX_BS" --served_model_name qwen ${TRT_FRONTENDS:+--num_serve_frontends $TRT_FRONTENDS} \
      --extra_llm_api_options "$cfg" > "$dir/server.log" 2>&1 &
    pid=$!
    ok=0
    for _ in $(seq 1 360); do
      if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then ok=1; break; fi
      kill -0 "$pid" 2> /dev/null || break
      sleep 5
    done
    if [[ $ok == 1 ]]; then
      poll=
      if [[ ${TRT_ITER_STATS:-0} == 1 ]]; then
        curl -sf "http://127.0.0.1:$PORT/metrics" > /dev/null || true      # drop warm-up iterations
        ( while true; do curl -sf "http://127.0.0.1:$PORT/metrics" >> "$dir/iter_stats.jsonl" && echo >> "$dir/iter_stats.jsonl"; sleep 2; done ) &
        poll=$!
      fi
      SERVED_MODEL=qwen python benchmarks/baselines/bench_speed.py --mode "$lane" --out "$dir" --base-url "http://127.0.0.1:$PORT" \
        --preset greedy --max-tokens "$MAX_TOKENS" --n-requests "$N_REQUESTS" --concurrency $CONCURRENCY \
        ${SETS:+--sets $SETS} || { echo "BENCH_FAILED $m $lane"; failed=1; }
      [[ -n $poll ]] && { sleep 3; curl -sf "http://127.0.0.1:$PORT/metrics" >> "$dir/iter_stats.jsonl" || true; kill "$poll" 2> /dev/null || true; }
    else
      echo "SERVER_FAILED $m $lane"; tail -20 "$dir/server.log"; failed=1
    fi
    kill -TERM -- "-$pid" 2> /dev/null || true
    wait "$pid" 2> /dev/null || true
    sleep 10
  done
done
exit $failed
