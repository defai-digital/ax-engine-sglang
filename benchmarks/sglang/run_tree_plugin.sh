#!/usr/bin/env bash
# Track L stage L0(d): measured end-to-end speed of the released DForkTree SGLang plugin
# (github.com/Perseus037/DForkTree, Apache-2.0) against the DFlash chain on the same verifier.
#   GPU=0 PORT=30000 OUT=/dev/shm/ax/results/track_l_tree9b bash benchmarks/sglang/run_tree_plugin.sh
# The plugin pins SGLang 0.5.22 and asserts batch size 1, so every lane here runs at concurrency 1
# from an overlay directory (SITE) that holds SGLang 0.5.22, its pinned kernel packages and the
# plugin, placed ahead of the venv on PYTHONPATH. Set up once with:
#   uv pip install --no-deps --target $SITE /root/DForkTree sglang==0.5.22 sglang-kernel==0.4.9 \
#       flashinfer_python==0.7.1 sgl-deep-gemm==0.2.2 cuda-tile==1.6.0rc5 tokenspeed_mla==0.1.8
#   python benchmarks/sglang/patch_dforktree.py $SITE/dforktree --release <pristine>   # see that file
# Lanes (LANES): ar (plain decoding on the same overlay), chain16n chain8n (stock chain on the
# venv's newer SGLang, no overlay), chain16 chain24 (stock DFLASH, plugin not loaded), tree32 tree64 (plugin, block 24),
# tree32c tree64c (the same with the compiled tree builder), tree32c_b16 (drafter block 16),
# tree32cfl_b16 (DFlash2 lattice scoring), tree32cfln_b16 (the same on SGLang's own tree verify
# instead of the plugin's fused Gated-DeltaNet verifier; a reference for correctness, slower).
# axotree<spec> / axtree<spec>: the same tree lanes through the standalone ax-engine-sglang plugin
# (AXSITE, `uv pip install --no-deps --target $AXSITE ax-engine-sglang/`): axotree on the 0.5.22
# overlay (parity with the patched DForkTree plugin), axtree on the venv's own SGLang, no overlay.
# axreplay8: native width-eight DFlash2 with AX accepted-prefix GDN replay, using AXSITE.
# PROFILE=1 also loads benchmarks/sglang/axprof.py (round-time breakdown, written to <lane>/axprof.json);
# AXPROF_SYNC=0 preserves overlap. AXPROF_NVTX=1 adds ranges to an Nsight Systems capture.
# Profiled runs are diagnostic. NSYS_TRACE=1 enables capture in run_baselines.sh.
# The plugin's fused verifier asserts one request per batch even while SGLang captures its CUDA
# graphs, so the tree lanes are limited to one running request (TREE_ARGS).
# Env: GPU, PORT, OUT, SITE, NGRAM (n-gram table from the plugin's v0.1.0 release), LANES, PRESETS,
#      N_REQUESTS, MAX_TOKENS, MODEL, DRAFT, TREE_ARGS, ATTN (attention backend for every lane;
#      unset = engine default, which on B200 with SGLang 0.5.22 resolves to triton while 0.5.23
#      picks trtllm_mha; the plugin's fast attention plan only engages with flashinfer), RELEASE (pristine plugin
#      package). The compiled tree builder (0.8 ms instead of 2.7 ms per round on B200) is chosen
#      only by the lane name (tree32c), so a lane's name always says what ran.
#      GDN_DEBUG=1 / LATTICE_DEBUG=N turn on the diagnostics of patch_dforktree.py / dft_lattice.py
#      (they sync every round; such a run prints a NOTE and is not a speed result). AX_GDN_DEBUG and
#      AX_LATTICE_DEBUG inherited from the shell are ignored.
set -euo pipefail
cd "$(dirname "$0")/../.."

SITE=${SITE:-/root/dft-site}
AXSITE=${AXSITE:-/root/ax-site}
NGRAM=${NGRAM:-/dev/shm/ax/l9b/dft/ngram_counts.npz}
RELEASE=${RELEASE:-/root/DForkTree/dforktree}   # pristine v0.1.0 plugin package (patches restore from it)
OUT=${OUT:-results/track_l_tree9b}
LANES=${LANES:-"chain16 chain24 tree32 tree64"}
export MODEL=${MODEL:-Qwen/Qwen3.5-9B} DRAFT=${DRAFT:-z-lab/Qwen3.5-9B-DFlash}
export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
TREE_ARGS=${TREE_ARGS:-"--max-running-requests 1 --cuda-graph-max-bs-decode 1"}
# TREE_CONCURRENCY: concurrency levels of the speed presets (default 1; multi-user lanes of ax-engine-sglang)
export CONCURRENCY=${TREE_CONCURRENCY:-1} PRESETS=${PRESETS:-greedy}
# invalidate every requested lane's previous pass first: a run that fails below (missing table,
# patch refused) must not leave an older pass that still looks complete
use_dforktree=0
for lane in $LANES; do
  rm -f "$OUT/$lane"/*/ids.jsonl "$OUT/$lane"/*/ids.meta.json
  [[ "$lane" == tree* ]] && use_dforktree=1
done
# the patched plugin files' hashes and the hashes of the sources they were built from go next to
# the results, so a number can be tied to the exact code that produced it
mkdir -p "$OUT"
if [[ "$use_dforktree" == 1 ]]; then
  [[ -d "$SITE/dforktree" ]] || { echo "missing $SITE/dforktree" >&2; exit 2; }
  python benchmarks/sglang/patch_dforktree.py "$SITE/dforktree" --release "$RELEASE"
fi
{
  sha256sum benchmarks/sglang/patch_dforktree.py ax-engine-sglang/ax_engine_sglang/ax_fast.py ax-engine-sglang/ax_engine_sglang/ax_lattice.py benchmarks/sglang/run_tree_plugin.sh benchmarks/sglang/run_baselines.sh benchmarks/sglang/serve_sglang.sh benchmarks/sglang/axprof.py benchmarks/sglang/nsys_trace.sh
  echo "git $(git rev-parse HEAD 2>/dev/null || echo none) $(date -u +%FT%TZ)"; } | tee -a "$OUT/plugin_patch.txt"
# identity of the overlay's code (patched plugin + SGLang version), recorded in every lane's
# metadata: two GPUs use two overlay directories, and paired_ci.py compares this instead of the path
# (the axtree lanes need no 0.5.22 overlay, so its sglang/version.py may be absent)
AX_SITE_SHA=$({ cat "$SITE"/dforktree/*.py "$SITE"/sglang/version.py 2>/dev/null || true; } | sha256sum | cut -c1-16)
export AX_SITE_SHA

ATTN_ARGS=${ATTN:+--attention-backend $ATTN}
# Gated-DeltaNet shape of the target for the fused tree verifier (9B: 24 layers x 32 value heads)
case "$MODEL" in
  *Qwen3.8-27B*) GDN_LAYERS=${GDN_LAYERS:-48}; GDN_HEADS=${GDN_HEADS:-48} ;;
  *) GDN_LAYERS=${GDN_LAYERS:-24}; GDN_HEADS=${GDN_HEADS:-32} ;;
esac
PROF=""
if [[ "${PROFILE:-0}" == "1" ]]; then
  PROF=",axprof"
  if [[ "${NSYS_TRACE:-0}" == 1 ]]; then
    export AXPROF_SYNC=${AXPROF_SYNC:-0} AXPROF_NVTX=${AXPROF_NVTX:-1}
  fi
fi
install_profiler() {
  [[ -n "$PROF" ]] || return 0
  local target=$1
  # register the profiler as an SGLang plugin through a hand-written dist-info in the overlay
  mkdir -p "$target/axprof-0.0.0.dist-info"
  cp benchmarks/sglang/axprof.py "$target/axprof.py"
  printf 'Metadata-Version: 2.1\nName: axprof\nVersion: 0.0.0\n' > "$target/axprof-0.0.0.dist-info/METADATA"
  printf '[sglang.srt.plugins]\naxprof = axprof:register\n' > "$target/axprof-0.0.0.dist-info/entry_points.txt"
}

GDN_DEBUG=${GDN_DEBUG:-0}; LATTICE_DEBUG=${LATTICE_DEBUG:-0}
[[ "$GDN_DEBUG" == "0" || "$GDN_DEBUG" == "1" ]] || { echo "GDN_DEBUG must be 0 or 1" >&2; exit 2; }
[[ "$LATTICE_DEBUG" =~ ^[0-9]+$ ]] || { echo "LATTICE_DEBUG must be a number of rounds" >&2; exit 2; }
[[ "${CHECK:-0}" == "1" || "${PROFILE:-0}" == "1" || "${NSYS_TRACE:-0}" == "1" || "$GDN_DEBUG" == "1" || "$LATTICE_DEBUG" != "0" ]] \
  && echo "NOTE: CHECK/PROFILE/DEBUG run, tok/s of these lanes is not a speed result"
failed=0
for lane in $LANES; do
  case "$lane" in
    ar)
      mkdir -p "$OUT/$lane"
      install_profiler "$SITE"
      env SGLANG_PLUGINS="none$PROF" AXPROF_OUT="$OUT/$lane/axprof.json" EXTRA_ARGS="$ATTN_ARGS" MODES=ar OUT="$OUT/$lane" \
        bash benchmarks/sglang/run_baselines.sh || failed=1
      ;;
    chain*n)
      # stock DFLASH chain on the venv's own (newer) SGLang, overlay removed from the path
      b=${lane#chain}; b=${b%n}
      mkdir -p "$OUT/$lane"
      # CHAIN_ARGS: extra server flags for these lanes, e.g. --max-running-requests 8 (27B DFlash2 at
      # block >= 16 reserves per-draft-token recurrent state for every request slot and otherwise
      # finds no room for the KV cache)
      stock_env=(env -u PYTHONPATH)
      if [[ -n "$PROF" ]]; then
        # Load only the profiler, never the SGLang version in SITE.
        profiler_dir="$OUT/$lane/profiler"
        install_profiler "$profiler_dir"
        stock_env+=("PYTHONPATH=$profiler_dir")
      fi
      "${stock_env[@]}" AX_SITE_SHA= SGLANG_PLUGINS="none$PROF" AXPROF_OUT="$OUT/$lane/axprof.json" EXTRA_ARGS="${CHAIN_ARGS:-} $ATTN_ARGS" MODES=dflash BLOCK="$b" OUT="$OUT/$lane" \
        bash benchmarks/sglang/run_baselines.sh || failed=1
      ;;
    chain*)
      # SGLANG_PLUGINS names a plugin that does not exist, so the tree plugin is not loaded.
      mkdir -p "$OUT/$lane"
      install_profiler "$SITE"
      env SGLANG_PLUGINS="none$PROF" AXPROF_OUT="$OUT/$lane/axprof.json" EXTRA_ARGS="${CHAIN_ARGS:-} $ATTN_ARGS" MODES=dflash BLOCK="${lane#chain}" OUT="$OUT/$lane" \
        bash benchmarks/sglang/run_baselines.sh || failed=1
      ;;
    axreplay8)
      [[ -d "$AXSITE/ax_engine_sglang" ]] || { echo "missing $AXSITE/ax_engine_sglang" >&2; exit 2; }
      mkdir -p "$OUT/$lane"
      profiler_dir="$OUT/$lane/profiler"
      install_profiler "$profiler_dir"
      env PYTHONPATH="$AXSITE:$profiler_dir" SGLANG_PLUGINS="ax-engine$PROF" \
        AX_GDN_CHAIN_REPLAY=1 AX_FI_OFFSET_FIX=0 AXPROF_OUT="$OUT/$lane/axprof.json" \
        EXTRA_ARGS="${CHAIN_ARGS:-} $ATTN_ARGS" MODES=dflash BLOCK=8 OUT="$OUT/$lane" \
        bash benchmarks/sglang/run_baselines.sh || failed=1
      ;;
    tree*|axtree*|axotree*)
      # plugin and SGLang: tree = patched DForkTree on the overlay, axotree = ax-engine-sglang on the
      # overlay, axtree = ax-engine-sglang on the venv's SGLang
      plugin=dforktree; lane_path="$PYTHONPATH"; site_sha="$AX_SITE_SHA"; spec=${lane#tree}; profiler_site=$SITE
      case "$lane" in
        axotree*) spec=${lane#axotree}; plugin=ax-engine; lane_path="$AXSITE:$SITE" ;;
        axtree*) spec=${lane#axtree}; plugin=ax-engine; lane_path="$AXSITE" ;;
      esac
      if [[ "$plugin" == ax-engine ]]; then
        [[ -d "$AXSITE/ax_engine_sglang" ]] || { echo "missing $AXSITE/ax_engine_sglang" >&2; exit 2; }
        site_sha="ax$(cat "$AXSITE"/ax_engine_sglang/*.py | sha256sum | cut -c1-12)-$(PYTHONPATH="$lane_path" python -c 'import sglang; print(sglang.__version__)')"
        profiler_site=$AXSITE
      fi
      # lane name tree<N>[c][f][s][l][n][_b<B>]: N nodes, c = compiled builder, f = AX_FAST=1 overhead fixes
      # (benchmarks/sglang/dft_fast.py), s = AX_FORCE_SYNC=1 ablation (one GPU sync back per round),
      # B = drafter block (default 24)
      tb=24
      if [[ "$spec" == *_b* ]]; then tb=${spec#*_b}; spec=${spec%_b*}; fi
      # l = DFlash2 lattice scoring (benchmarks/sglang/dft_lattice.py, needs a DFlash2 drafter)
      # n = native: SGLang's own tree verify instead of the plugin's fused GDN verifier
      fused=1; [[ "$spec" == *n ]] && { fused=0; spec=${spec%n}; }
      [[ "$fused" == "0" && "$GDN_DEBUG" == "1" ]] && echo "NOTE: $lane uses SGLang's own verify, GDN_DEBUG prints nothing there"
      lat=0; [[ "$spec" == *l ]] && { lat=1; spec=${spec%l}; }
      sync=0; [[ "$spec" == *s ]] && { sync=1; spec=${spec%s}; }
      fast=0; [[ "$spec" == *f ]] && { fast=1; spec=${spec%f}; }
      compile=0; [[ "$spec" == *c ]] && { compile=1; spec=${spec%c}; }   # from the lane name only
      n=$spec
      [[ "$n" =~ ^[0-9]+$ ]] || { echo "bad tree lane name: $lane" >&2; exit 2; }
      [[ "$lat" == 1 || -s "$NGRAM" ]] || { echo "missing n-gram table $NGRAM for $lane" >&2; exit 2; }
      install_profiler "$profiler_site"
      mkdir -p "$OUT/$lane"
      # every behaviour switch is set here, so nothing inherited from the shell changes a lane;
      # CHECK=1 (AX_CHECK, synchronous mask/bound validation) is for correctness runs, not speed
      # several-request switches of ax-engine-sglang (ax_chain.py): AX_TREE, AX_CHAIN_WIDTH, AX_CHAIN_SCHEDULE,
      # AX_ROUND_DUMP, AX_CHAIN_GRAPH_WIDTHS, AX_CHAIN_CHECK pass through explicitly (defaults: tree on, width =
      # drafter block, no schedule, no dump, eager chain rounds, no allocation check)
      env -u FC1_CHAIN_FIRST PYTHONPATH="$lane_path" AX_SITE_SHA="$site_sha" SGLANG_PLUGINS="$plugin$PROF" \
          AX_TREE="${AX_TREE:-1}" AX_CHAIN_WIDTH="${AX_CHAIN_WIDTH:-}" AX_CHAIN_SCHEDULE="${AX_CHAIN_SCHEDULE:-}" AX_ROUND_DUMP="${AX_ROUND_DUMP:-}" AX_CHAIN_GRAPH_WIDTHS="${AX_CHAIN_GRAPH_WIDTHS:-}" AX_CHAIN_DRAFT_WIDTHS="${AX_CHAIN_DRAFT_WIDTHS:-}" AX_CHAIN_ATTN="${AX_CHAIN_ATTN:-}" AX_SSM_POSITIONS="${AX_SSM_POSITIONS:-}" AX_FP8_PROBE="${AX_FP8_PROBE:-0}" AX_CHAIN_CHECK="${AX_CHAIN_CHECK:-0}" AX_TREE_SAMPLING="${AX_TREE_SAMPLING:-0}" AX_FI_OFFSET_FIX="${AX_FI_OFFSET_FIX:-1}" AXPROF_OUT="$OUT/$lane/axprof.json" AX_FAST="$fast" AX_FORCE_SYNC="$sync" AX_CHECK="${CHECK:-0}" AX_LATTICE="$lat" AX_GDN_DEBUG="$GDN_DEBUG" AX_LATTICE_DEBUG="$LATTICE_DEBUG" AX_GDN_LAYERS="$GDN_LAYERS" AX_GDN_HEADS="$GDN_HEADS" FC1_TREE64=1 FC1_TREE_N="$n" FC1_DRAFT_BLOCK="$tb" FC1_COMPILE_BUILD="$compile" \
          FC1_FUSED="$fused" FC1_BRANCH_PILOT=1 FC1_OURS=1 FC1_NGRAM_PATH="$NGRAM" \
          EXTRA_ARGS="$TREE_ARGS $ATTN_ARGS" MODES=dftree BLOCK="$tb" OUT="$OUT/$lane" bash benchmarks/sglang/run_baselines.sh || failed=1
      ;;
    *) echo "unknown lane $lane" >&2; exit 2 ;;
  esac
done
echo "TREE_PLUGIN_DONE failed=$failed $(date -u)"
exit $failed
