#!/usr/bin/env bash
# Stage 1a of docs/decisions/2026-11-16-dual-attention-backend.md: tree rounds on triton, chain
# rounds and prefill on trtllm_mha (AX_CHAIN_ATTN, hybrid attention), one server. Correctness first, then speed.
#   GPU=0 PORT=30000 REF_GPU=1 REF_PORT=30001 MODEL=Qwen/Qwen3.8-27B OUT=/dev/shm/ax/results/track_l_s1a \
#     AXSITE=/root/ax-site-s1a bash benchmarks/sglang/run_stage1a.sh [correct|speed|all]
# correct: identity sets (c = 1, 4, 16) with round dumps for the mixed tree lane (mtree) and the
#   forced width-switching lane with the tree (msw), the allocation check on; the HF accept oracle
#   (bf16 only); the teacher-forced fresh-prefill check on a stock server without radix cache on
#   REF_GPU (also the NVFP4 oracle); the stress test; FP8 write probe at start-up when the KV is FP8.
#   NVFP4 interim gate (until the teacher check is trusted): mtree0 = mixed server with the tree off at
#   c = 1, to compare with the all-trtllm_mha lane with the tree off.
# speed: mixed server (tree + width 8 drafted at 8, and tree + width 16), c = 1 2 4 8 16 (32 for
#   NVFP4), 64 prompts per family, 1,024 tokens; compare with track_l_s0k / s0p / s0p_r2 (bf16) or
#   track_n_nv7 / nv8 (NVFP4). Run speed twice (OUT=..._r2) for the paired two-pass gate.
set -euo pipefail
cd "$(dirname "$0")/../.."
STEP=${1:-all}
case "$STEP" in correct|speed|all) ;; *) echo "usage: $0 [correct|speed|all]" >&2; exit 2 ;; esac
GPU=${GPU:-0} PORT=${PORT:-30000} REF_GPU=${REF_GPU:-1} REF_PORT=${REF_PORT:-30001}
MODEL=${MODEL:-Qwen/Qwen3.8-27B} DRAFT=${DRAFT:-incoai/Qwen3.8-27B-DFlash2}
OUT=${OUT:-/dev/shm/ax/results/track_l_s1a}
nvfp4=0; [[ "${MODEL:-}" == *NVFP4* ]] && nvfp4=1
SLOTS=${SLOTS:-$([[ $nvfp4 == 1 ]] && echo 32 || echo 16)}
CONC=${CONC:-"1 2 4 8 16"}
FAIL=0
fail() { echo "GATE_FAILED $*"; FAIL=1; }
export MODEL DRAFT AXSITE=${AXSITE:-/root/ax-site} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# prefill on trtllm_mha as stock (section 11c: a triton prefill makes greedy near-ties resolve differently
# from stock, so the two servers generate different text); tree verify on triton (decode backend with
# --speculative-attention-mode decode); chain rounds on trtllm_mha (AX_CHAIN_ATTN). HYBRID=0: all triton.
export AX_CHAIN_GRAPH_WIDTHS="16 8" AX_CHAIN_DRAFT_WIDTHS="8"
G="--max-running-requests $SLOTS --cuda-graph-max-bs-decode $SLOTS"
if [[ "${HYBRID:-1}" == 1 ]]; then
  export ATTN= AX_CHAIN_ATTN=trtllm_mha
  G="$G --prefill-attention-backend trtllm_mha --decode-attention-backend triton --speculative-attention-mode decode"
else
  export ATTN=triton AX_CHAIN_ATTN=triton
fi
G="$G --speculative-draft-attention-backend $AX_CHAIN_ATTN"
lane() {   # name tree width presets extra-env...
  local name=$1 tree=$2 wid=$3 presets=$4; shift 4
  mkdir -p "$OUT/$name"
  env "$@" PRESETS="$presets" AX_TREE="$tree" AX_CHAIN_WIDTH="$wid" TREE_ARGS="$G" GPU="$GPU" PORT="$PORT" \
    LANES="axtree16cfl_b16" OUT="$OUT/$name" bash benchmarks/sglang/run_tree_plugin.sh || fail "lane $name"
}

if [[ "$STEP" == correct || "$STEP" == all ]]; then
  rm -f "$OUT"/*/rounds.jsonl   # dumps are appended to; never mix passes
  ids_env=(IDS_CONCURRENCY="1 4 16" N_REQUESTS=16 MAX_TOKENS=512 AX_CHAIN_CHECK=1 AX_FP8_PROBE=$nvfp4)
  lane mtree 1 16 ids "${ids_env[@]}" AX_ROUND_DUMP="$OUT/mtree/rounds.jsonl"
  lane msw 1 "" ids "${ids_env[@]}" AX_CHAIN_SCHEDULE="16x5,8x3" AX_ROUND_DUMP="$OUT/msw/rounds.jsonl"
  [[ $nvfp4 == 1 ]] && lane mtree0 0 16 ids IDS_CONCURRENCY=1 N_REQUESTS=16 MAX_TOKENS=512 AX_ROUND_DUMP="$OUT/mtree0/rounds.jsonl"
  lane mstress 1 16 stress AX_CHAIN_SCHEDULE="16x5,8x3"
  for name in mtree msw; do
    I="$OUT/$name/axtree16cfl_b16/dftree"
    if [[ $nvfp4 == 0 ]]; then
      CUDA_VISIBLE_DEVICES=$GPU python benchmarks/sglang/accept_oracle.py --rounds "$OUT/$name/rounds.jsonl" \
        --ids "$I/ids.jsonl" "$I/ids_c4.jsonl" "$I/ids_c16.jsonl" --model "$MODEL" --out "$OUT/$name/oracle.json" \
        > "$OUT/$name/oracle.log" 2>&1 || fail "oracle $name"
    fi
  done
  # fresh prefill on a stock AR server (radix cache off) on REF_GPU, all saved outputs of both lanes
  ids=$(ls "$OUT"/mtree/axtree16cfl_b16/dftree/ids*.jsonl "$OUT"/msw/axtree16cfl_b16/dftree/ids*.jsonl)
  env -u PYTHONPATH SGLANG_PLUGINS=none RADIX_CACHE=0 EXTRA_ARGS="--chunked-prefill-size 16384" MODES=ar \
    GPU="$REF_GPU" PORT="$REF_PORT" PRESETS=teacher TEACHER_IDS="$ids" TEACHER_OUT="$OUT/teacher.json" \
    OUT="$OUT/ref" bash benchmarks/sglang/run_baselines.sh || [[ $nvfp4 == 1 && -s "$OUT/teacher.json" ]] || fail "teacher (fresh prefill)"
  if [[ $nvfp4 == 0 ]]; then
    grep -q '"passed": true' "$OUT/teacher.json" 2>/dev/null || fail "teacher result"
  else
    # NVFP4 cannot meet the bf16 bound (static activation scales make outputs depend on the batch);
    # gate: no worse than stock NVFP4 under the same check (control s1a_nvfp4_ctl: mismatches
    # 2.9-3.3%, over the tie 1.06-1.30%, worst 6.9). Bounds overridable.
    python3 - "$OUT/teacher.json" "${NV_MAX_RATE:-0.035}" "${NV_MAX_OVER:-0.013}" "${NV_MAX_WORST:-7.0}" <<'PY' || fail "teacher result (NVFP4 vs stock control)"
import json, sys
d = json.load(open(sys.argv[1])); r, o, w = (float(x) for x in sys.argv[2:5])
over = d["over_tie"] / d["tokens"]
print(f"NVFP4 teacher: mismatches {d['mismatch_rate']:.4f} (max {r}), over tie {over:.4f} (max {o}), worst {d['worst_regret']:.2f} (max {w})")
sys.exit(0 if d["mismatch_rate"] <= r and over <= o and d["worst_regret"] <= w else 1)
PY
  fi
  grep -q '"passed": true' "$OUT/mstress/axtree16cfl_b16/dftree/stress.json" 2>/dev/null || fail "stress result"
  # round dumps: every chain round on trtllm_mha and replayed from a graph (batches <= SLOTS), every
  # tree round on triton, and some requests that cross tree <-> chain (the KV handoff)
  python3 - "$OUT" "$AX_CHAIN_ATTN" <<'PY' || fail "round dump checks"
import glob, json, sys
from collections import Counter, defaultdict
out, chain_backend, ok = sys.argv[1], sys.argv[2], True
for f in sorted(glob.glob(f"{out}/*/rounds.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    c = Counter((r["kind"], r.get("backend"), r.get("graph")) for r in rows)
    kinds = defaultdict(set)
    for r in rows:
        kinds[r["rid"]].add(r["kind"])
    crossing = sum(1 for k in kinds.values() if k == {"tree", "chain"})
    print(f, dict(c), "requests crossing tree<->chain:", crossing)
    bad = [k for k in c if (k[0] == "chain" and (k[1] != chain_backend or k[2] is not True))
           or (k[0] == "tree" and (k[1] != "triton" or k[2] is not True))]
    if not rows or bad:
        print("BAD_ROUNDS", f, bad); ok = False
    if f.endswith("/mtree/rounds.jsonl") or f.endswith("/msw/rounds.jsonl"):
        if crossing == 0:
            print("NO_HANDOFF", f); ok = False
sys.exit(0 if ok else 1)
PY
  if [[ $FAIL != 0 && "$STEP" == all ]]; then
    echo "STAGE1A_DONE correct FAILED: speed not run $(date -u)"; exit 1
  fi
fi

if [[ "$STEP" == speed || "$STEP" == all ]]; then
  [[ $nvfp4 == 1 ]] && CONC=${CONC_NVFP4:-"1 2 4 8 16 32"}
  lane sd8 1 8 greedy TREE_CONCURRENCY="$CONC" N_REQUESTS="${N_SPEED:-64}" MAX_TOKENS=1024
  lane sp16 1 16 greedy TREE_CONCURRENCY="$CONC" N_REQUESTS="${N_SPEED:-64}" MAX_TOKENS=1024
fi
echo "STAGE1A_DONE $STEP fail=$FAIL $(date -u)"
exit $FAIL
