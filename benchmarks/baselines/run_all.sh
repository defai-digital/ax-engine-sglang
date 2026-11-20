#!/usr/bin/env bash
# Run the full Phase 0 sweep: for each mode, start vLLM, benchmark speed + quality, stop it.
#   benchmarks/baselines/run_all.sh                       # modes: ar mtp1 mtp2 mtp3
#   MODES="ar mtp2" LIMIT=100 benchmarks/baselines/run_all.sh
# Env: MODES, OUT, LIMIT (items per quality task; empty = full sets), CONCURRENCY, SKIP_QUALITY,
#      SAMPLED (1 = also run speed with the model card's instruct sampling)
set -euo pipefail
cd "$(dirname "$0")/../.."

MODES=${MODES:-"ar mtp1 mtp2 mtp3"}
OUT=${OUT:-results/$(date +%Y%m%d-%H%M%S)}
LIMIT=${LIMIT:-}
CONCURRENCY=${CONCURRENCY:-"1 4 16 64"}
PORT=${PORT:-8000}
export BASE_URL="http://127.0.0.1:${PORT}"
mkdir -p "$(dirname "$OUT")"
if ! mkdir "$OUT"; then
  echo "Choose a new OUT directory for each sweep: $OUT" >&2
  exit 2
fi

python - "$OUT/env.json" <<'EOF'
import json, subprocess, sys, platform
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return str(e)
json.dump({
  "gpus": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
  "vllm": sh("python -c 'import vllm; print(vllm.__version__)'"),
  "torch": sh("python -c 'import torch; print(torch.__version__, torch.version.cuda)'"),
  "python": platform.python_version(),
  "model": __import__("os").environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B"),
}, open(sys.argv[1], "w"), indent=2)
EOF

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

limit_arg=()
[[ -n "$LIMIT" ]] && limit_arg=(--limit "$LIMIT")
failed=0

for mode in $MODES; do
  echo "=== mode: $mode ==="
  if ! python -c "import sys; sys.path.insert(0,'benchmarks/baselines'); from common import check_loopback_port; check_loopback_port(int(sys.argv[1]))" "$PORT"; then
    echo "server port unavailable for $mode: $PORT" >&2
    failed=1
    continue
  fi
  mkdir -p "$OUT/$mode"
  export SERVED_MODEL="qwen-phase0-$$-$mode"
  benchmarks/baselines/serve.sh "$mode" > "$OUT/$mode/server.log" 2>&1 &
  SERVER_PID=$!
  python -c "import sys; sys.path.insert(0,'benchmarks/baselines'); from common import SERVED_MODEL, wait_for_server; wait_for_server(pid=int(sys.argv[1]), model=SERVED_MODEL)" "$SERVER_PID" \
    || { echo "server failed for $mode; see $OUT/$mode/server.log"; tail -50 "$OUT/$mode/server.log"; cleanup; SERVER_PID=""; failed=1; continue; }

  # shellcheck disable=SC2086
  python benchmarks/baselines/bench_speed.py --mode "$mode" --out "$OUT" --preset greedy --concurrency $CONCURRENCY
  if [[ "${SAMPLED:-1}" == "1" ]]; then
    # shellcheck disable=SC2086
    python benchmarks/baselines/bench_speed.py --mode "$mode" --out "$OUT" --preset instruct --concurrency $CONCURRENCY
  fi
  if [[ -z "${SKIP_QUALITY:-}" ]]; then
    python benchmarks/baselines/eval_quality.py --mode "$mode" --out "$OUT" --preset greedy "${limit_arg[@]}"
  fi

  cleanup; SERVER_PID=""
  sleep 10   # let GPU memory free before the next server
done

python benchmarks/baselines/summarize.py "$OUT" > "$OUT/.REPORT.md.tmp"
mv "$OUT/.REPORT.md.tmp" "$OUT/REPORT.md"
echo "Report: $OUT/REPORT.md"
exit "$failed"
