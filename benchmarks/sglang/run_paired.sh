#!/usr/bin/env bash
# Track L: paired, interleaved c=1 measurement for gate GL0 and the token-identity part of GL1.
#   GPU=0 PORT=30000 REPEATS="1 3" OUT=/dev/shm/ax/results/track_l_paired bash benchmarks/sglang/run_paired.sh
# Every repeat is one complete pass over all lanes on this GPU, in an order rotated by the repeat
# number, each lane saving per-request token ids and wall time (benchmarks/sglang/token_identity.py, greedy,
# thinking off, N prompts per family, MAX_TOKENS new tokens, WARMUP unrecorded requests first).
# Statistics: benchmarks/sglang/paired_ci.py over the files of all repeats. $OUT/manifest.txt lists every lane
# pass and whether it succeeded; the script exits non-zero if any lane failed.
# Env: GPU, PORT, OUT, REPEATS, LANES (see benchmarks/sglang/run_tree_plugin.sh), N (prompts per family),
#      MAX_TOKENS, plus everything run_tree_plugin.sh reads.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-results/track_l_paired}
REPEATS=${REPEATS:-"1 2 3"}
read -r -a L <<< "${LANES:-tree32cf chain8 chain16 chain24 chain8n chain16n ar}"
export N_REQUESTS=${N:-128} MAX_TOKENS=${MAX_TOKENS:-2048} PRESETS=ids
failed=()
for r in $REPEATS; do
  k=$(( r % ${#L[@]} ))
  order=("${L[@]:k}" "${L[@]:0:k}")
  echo "repeat $r order: ${order[*]} $(date -u +%T)"
  for lane in "${order[@]}"; do
    rm -rf "${OUT:?}/r$r/$lane"                 # a retried lane must not inherit an older pass
    if LANES="$lane" OUT="$OUT/r$r" bash benchmarks/sglang/run_tree_plugin.sh; then
      echo "r$r $lane ok $(date -u +%T)" >> "$OUT/manifest.txt"
    else
      echo "LANE_FAILED r$r $lane"; failed+=("r$r/$lane")
      echo "r$r $lane FAILED $(date -u +%T)" >> "$OUT/manifest.txt"
    fi
  done
done
echo "PAIRED_DONE failed=${#failed[@]} ${failed[*]:-} $(date -u)"
[[ ${#failed[@]} -eq 0 ]]
