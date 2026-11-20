#!/usr/bin/env bash
# Optional Nsight Systems capture shared by the two serving benchmark launchers.

nsys_trace_init() {
  nsys_launch=()
  nsys_session=
  nsys_report=
  nsys_capturing=0
  if [[ "${NSYS_TRACE:-0}" == 1 ]]; then
    command -v nsys >/dev/null || { echo "NSYS_TRACE=1 requires nsys on PATH" >&2; return 1; }
    case "${NSYS_GRAPH_TRACE:-graph}" in
      graph|node) ;;
      *) echo "NSYS_GRAPH_TRACE must be graph or node" >&2; return 1 ;;
    esac
    NSYS_VERSION=$(nsys --version) || return 1
    export NSYS_VERSION
  fi
}

nsys_trace_prepare() {
  [[ "${NSYS_TRACE:-0}" == 1 ]] || return 0
  # The PID and lane index keep simultaneous benchmarks in distinct sessions.
  nsys_session="ax-$$-${1}-$RANDOM"
  nsys_report="$(cd "$2" && pwd -P)/decode" || return 1
  rm -f "${nsys_report}.nsys-rep" || return 1
  nsys_launch=(nsys launch --session-new="$nsys_session" --wait=primary --trace=cuda,nvtx,osrt
    "--cuda-graph-trace=${NSYS_GRAPH_TRACE:-graph}")
}

nsys_trace_start() {
  [[ -n "$nsys_session" ]] || return 0
  nsys start --session="$nsys_session" --output="$nsys_report" --sample=none --cpuctxsw=none || return 1
  nsys_capturing=1
}

nsys_trace_stop() {
  [[ "$nsys_capturing" == 1 ]] || return 0
  nsys_capturing=0
  nsys stop --session="$nsys_session" || return 1
  if [[ ! -s "${nsys_report}.nsys-rep" ]]; then
    echo "Nsight Systems did not write ${nsys_report}.nsys-rep" >&2
    return 1
  fi
}

nsys_trace_cleanup() {
  [[ -n "$nsys_session" ]] || return 0
  nsys_trace_stop || true
  nsys shutdown --session="$nsys_session" --kill=sigterm >/dev/null 2>&1 || true
  nsys_session=
  nsys_capturing=0
}
