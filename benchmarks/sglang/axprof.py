"""Round-time breakdown for SGLang speculative decoding (Track L root-cause tool).

Loaded as an SGLang general plugin (entry point group `sglang.srt.plugins`, name `axprof`); see
benchmarks/sglang/run_tree_plugin.sh (PROFILE=1). It wraps the phases of one speculative round with timers:

  round            the spec worker's forward_batch_generation (draft + build + verify + commit)
  interval         wall time between the starts of consecutive rounds (round + scheduler, sampling
                   bookkeeping, detokeniser hand-off); only gaps under 50 ms are counted
  fwd.<who>.<mode> every ModelRunner.forward, by worker (draft / target) and forward mode
  verify_prepare   DFlash input binding and attention metadata preparation
  verify_accept    DFlash token acceptance after the target forward
  target_commit    DFlash recurrent state and convolution rollback
  tree_build       DForkTree Builder call (n-gram re-scoring + beam tree, CUDA graph replay)
  verify_branch    DForkTree tree verification including the target pass and the state commit
  eagle_verify     run_eagle_verify inside it
  draft_kv_append  writing the accepted target hidden states into the drafter's cache

With AXPROF_SYNC=1 (default) every timer is bracketed by torch.cuda.synchronize(), which attributes GPU time
to the right phase but removes any overlap between phases, so the phase sums are an upper bound
and tok/s measured with profiling on must not be reported as speed. Use AXPROF_SYNC=0 and AXPROF_NVTX=1
to mark host ranges for Nsight Systems without synchronization. With sync disabled the JSON timings
are host wall times, including any time the engine itself spends waiting. Means are written to
AXPROF_OUT (JSON) every AXPROF_EVERY rounds.
"""
import atexit
import functools
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext

SYNC = os.environ.get("AXPROF_SYNC", "1") == "1"
NVTX = os.environ.get("AXPROF_NVTX", "0") == "1"
OUT = os.environ.get("AXPROF_OUT", "/tmp/axprof.json")
EVERY = int(os.environ.get("AXPROF_EVERY", "500"))
acc = defaultdict(lambda: [0, 0.0])
state = {"last_start": None, "rounds": 0}


def _sync():
    if SYNC:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def timed(label_fn):
    if NVTX:
        from torch.cuda.nvtx import range as phase_range
    else:
        phase_range = nullcontext

    def deco(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            label = label_fn(*a, **kw) if callable(label_fn) else label_fn
            _sync()
            t = time.perf_counter()
            try:
                with phase_range(label):
                    return f(*a, **kw)
            finally:
                _sync()
                rec = acc[label]
                rec[0] += 1
                rec[1] += time.perf_counter() - t
        return wrapper
    return deco


def dump():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        state["last_dump"] = time.perf_counter()
        json.dump({"sync": SYNC, "nvtx": NVTX, "pid": os.getpid(), "rounds": state["rounds"],
                   "phases": {k: {"calls": n, "mean_ms": 1e3 * s / max(n, 1)} for k, (n, s) in sorted(acc.items())}},
                  fh, indent=2)
    os.replace(tmp, OUT)


# The runner stops servers with SIGTERM then SIGKILL, which can skip exit handlers, so the file is
# also rewritten at least every AXPROF_SECONDS; atexit only covers a clean exit. A profile can thus
# miss at most the last AXPROF_SECONDS of rounds; it reports the rounds it covers.
SECONDS = float(os.environ.get("AXPROF_SECONDS", "5"))
atexit.register(lambda: dump() if acc else None)


def wrap_round(cls):
    inner = timed("round")(cls.forward_batch_generation)

    @functools.wraps(inner)
    def wrapper(self, *a, **kw):
        now = time.perf_counter()
        last = state["last_start"]
        if last is not None and now - last < 0.05:
            rec = acc["interval"]
            rec[0] += 1
            rec[1] += now - last
        state["last_start"] = now
        out = inner(self, *a, **kw)
        state["rounds"] += 1
        if state["rounds"] % EVERY == 0 or time.perf_counter() - state.get("last_dump", 0.0) > SECONDS:
            dump()
        return out
    cls.forward_batch_generation = wrapper


def wrap_dflash_worker(cls):
    wrap_round(cls)
    for method, phase in (
        ("_accept_block", "verify_accept"),
        ("_update_target_mamba_state_after_verify", "target_commit"),
        ("_append_target_hidden_to_draft_kv_by_loc", "draft_kv_append"),
    ):
        impl = getattr(cls, method, None)
        if impl is not None:
            setattr(cls, method, timed(phase)(impl))


def register():
    import sys
    from pathlib import Path

    plugins = {name.strip() for name in os.environ.get("SGLANG_PLUGINS", "").split(",")}
    if "ax-engine" in plugins:
        # SGLang executes entry points in discovery order. Install the AX
        # replay commit before timing it, even when axprof is discovered first.
        from ax_engine_sglang.plugin import register as register_ax_engine

        register_ax_engine()

    from sglang.srt.model_executor.model_runner import ModelRunner

    def fwd_label(self, forward_batch, *a, **kw):
        who = "draft" if getattr(self, "is_draft_worker", False) else "target"
        return f"fwd.{who}.{getattr(forward_batch.forward_mode, 'name', forward_batch.forward_mode)}"
    ModelRunner.forward = timed(fwd_label)(ModelRunner.forward)

    # graph.<who>.w<width>.<replay|eager>: how often each decode graph runner could replay its graph
    # (calls only; the time column of these entries is zero)
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    can_run = DecodeCudaGraphRunner.can_run_graph

    @functools.wraps(can_run)
    def can_run_graph(self, forward_batch):
        ok = can_run(self, forward_batch)
        who = "draft" if getattr(self.model_runner, "is_draft_worker", False) else "target"
        acc[f"graph.{who}.w{getattr(self, 'captured_req_width', '?')}.{'replay' if ok else 'eager'}"][0] += 1
        return ok
    DecodeCudaGraphRunner.can_run_graph = can_run_graph

    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    DFlashVerifyInput.prepare_for_verify = timed("verify_prepare")(DFlashVerifyInput.prepare_for_verify)
    wrap_dflash_worker(DFlashWorkerV2)
    if "ax-engine" in plugins:
        # The resolved scheduler config installs the generated tree worker.
        if "fc1_dflash_worker" not in sys.modules:
            print("[axprof] AX tree worker not loaded; profiling the native worker", flush=True)
            return
    elif "dforktree" in plugins:
        here = [p for p in sys.path if (Path(p) / "dforktree" / "fc1_plugin.py").exists()]
        if not here:
            return
        sys.path.insert(0, str(Path(here[0]) / "dforktree"))
    else:
        return
    import fc1_branch64
    import fc1_builder
    import fc1_dflash_worker
    wrap_dflash_worker(fc1_dflash_worker.DFlashTreeWorkerV2)
    fc1_builder.Builder.__call__ = timed("tree_build")(fc1_builder.Builder.__call__)
    fc1_branch64.run_eagle_verify = timed("eagle_verify")(fc1_branch64.run_eagle_verify)
    fc1_branch64.verify_branch = timed("verify_branch")(fc1_branch64.verify_branch)
