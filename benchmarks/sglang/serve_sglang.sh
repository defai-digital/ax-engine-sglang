#!/usr/bin/env bash
# Launch SGLang for Qwen3.8-27B in one decoding mode (Track A baselines).
#   benchmarks/sglang/serve_sglang.sh ar          # plain autoregressive decoding
#   benchmarks/sglang/serve_sglang.sh mtp3        # built-in MTP head, 3 draft steps (mtp1..mtp9)
#   benchmarks/sglang/serve_sglang.sh dspark      # RadixArk DSpark draft model
#   benchmarks/sglang/serve_sglang.sh dflash      # DFlash2 draft model, block 8
#   benchmarks/sglang/serve_sglang.sh dftree      # DFlash draft tree through the DForkTree plugin (Track L)
#   benchmarks/sglang/serve_sglang.sh dspine      # Tencent DSpine draft model, block 16 (needs the
#                                       # sglang-dspine plugin: its own venv with SGLang 0.5.22)
# Env overrides: MODEL, PORT, DRAFT (draft model path for dspark/dflash and our own checkpoints),
#   BLOCK (dflash / dspine draft tokens, dspark block size), AUTOTUNE, RADIX_CACHE, EXTRA_ARGS,
#   DSPINE_ARGS (replaces the DSpine card's serving flags), DSPINE_ATTN (attention backend for
#   DSpine: fa3 as on the card, or flashinfer on B200)
# Every mode runs with the same kernel policy. FlashInfer autotune is off by default because it
# hangs while capturing the DFlash draft graph on B200 (SGLang 0.5.23); AUTOTUNE=1 turns it on.
# The radix prefix cache is off by default so repeated prompts are not served from cache.
set -euo pipefail

MODE=${1:-ar}
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
PORT=${PORT:-30000}

args=(-m sglang.launch_server
  --model-path "$MODEL"
  --served-model-name qwen
  --host 127.0.0.1 --port "$PORT"
  --reasoning-parser qwen3
  --enable-metrics
  --random-seed 0)
[[ "${AUTOTUNE:-0}" == "1" ]] || args+=(--disable-flashinfer-autotune)
# DSpine needs the radix cache (its prefix cache lives there) and refuses to start without it.
# For a like-for-like comparison with DSpine run the other modes with RADIX_CACHE=1 as well.
[[ "${RADIX_CACHE:-0}" == "1" || "$MODE" == dspine ]] || args+=(--disable-radix-cache)

case "$MODE" in
  ar) ;;
  mtp[1-9]|mtp[1-9][0-9])
    n=${MODE#mtp}
    args+=(--speculative-algorithm EAGLE --speculative-num-steps "$n" --speculative-eagle-topk 1
           --speculative-num-draft-tokens $((n + 1)))
    ;;
  dspark)
    args+=(--speculative-algorithm DSPARK
           --speculative-draft-model-path "${DRAFT:-RadixArk/Qwen3.8-27B-DSpark}"
           --speculative-dspark-block-size "${BLOCK:-7}")
    ;;
  dflash)
    args+=(--speculative-algorithm DFLASH
           --speculative-draft-model-path "${DRAFT:-incoai/Qwen3.8-27B-DFlash2}"
           --speculative-num-draft-tokens "${BLOCK:-8}")
    ;;
  dftree)
    # DForkTree plugin (Apache-2.0, pins SGLang 0.5.22; single request only). The caller puts the
    # plugin overlay on PYTHONPATH and sets the FC1_* switches, see benchmarks/sglang/run_tree_plugin.sh.
    args+=(--speculative-algorithm FC1_DFLASH_TREE
           --speculative-draft-model-path "${DRAFT:-z-lab/Qwen3.5-9B-DFlash}"
           --speculative-num-draft-tokens "${BLOCK:-24}")
    ;;
  dspine)
    # Flags from the model card (fa3 attention, ReplaySSM state replay for the DeltaNet layers).
    # shellcheck disable=SC2206
    # fa3 only exists for SM80..SM90 (H100/H200); on B200 (SM100) set DSPINE_ATTN=flashinfer.
    attn=${DSPINE_ATTN:-fa3}
    dspine_args=(${DSPINE_ARGS:---attention-backend $attn --speculative-draft-attention-backend $attn
      --linear-attn-backend triton --enable-linear-replayssm-spec --mamba-ssm-dtype float32
      --mamba-radix-cache-strategy extra_buffer --sampling-backend flashinfer})
    args+=(--speculative-algorithm DSPINE
           --speculative-draft-model-path "${DRAFT:-tencent/Qwen3.8-27B-DSpine-b16}"
           --speculative-num-draft-tokens "${BLOCK:-16}" --dtype bfloat16 "${dspine_args[@]}")
    ;;
  *) echo "unknown mode: $MODE (use ar, mtp1..mtp99, dspark, dflash, dftree, dspine)" >&2; exit 2 ;;
esac

# EXTRA_ARGS_BASE: flags a harness adds to every lane (memory sizing on a shared host), kept apart
# from the per-lane EXTRA_ARGS.
# shellcheck disable=SC2086
exec python "${args[@]}" ${EXTRA_ARGS:-} ${EXTRA_ARGS_BASE:-}
