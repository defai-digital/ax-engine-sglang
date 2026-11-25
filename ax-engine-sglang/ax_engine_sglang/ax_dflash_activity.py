"""Opt-in, scheduler-local counters for single-GPU DFlash benchmark passes."""

from collections import Counter, defaultdict
from functools import wraps
import os

from .compat import patch_attribute


class Activity:
    def __init__(self):
        self.rounds = Counter()
        self.optimized = defaultdict(Counter)
        self.graphs = defaultdict(Counter)
        self.paths = set()
        self.capture_attempts = 0

    def wrap_round(self, original):
        @wraps(original)
        def forward(worker, batch, *args, **kwargs):
            mode = batch.forward_mode
            if mode.is_idle() or mode.is_extend() or batch.is_extend_in_batch:
                return original(worker, batch, *args, **kwargs)

            size = str(len(batch.seq_lens))  # live requests, before any verify graph padding
            self.paths.clear()
            output = original(worker, batch, *args, **kwargs)
            self.rounds[size] += 1
            for path in self.paths:
                self.optimized[path][size] += 1
            return output

        return forward

    def wrap_state(self, original):
        @wraps(original)
        def get_internal_state(controller, request):
            reply = original(controller, request)
            reply.internal_state["ax_dflash_activity"] = {
                "rounds": dict(self.rounds),
                "optimized": {path: dict(counts) for path, counts in self.optimized.items()},
                "graphs": {phase: dict(counts) for phase, counts in self.graphs.items()},
                "capture_attempts": self.capture_attempts,
            }
            return reply

        return get_internal_state

    def wrap_graph(self, original):
        @wraps(original)
        def execute(runner, forward_batch, *args, **kwargs):
            output = original(runner, forward_batch, *args, **kwargs)
            if forward_batch.forward_mode.is_target_verify() and not runner.ragged_verify_mode:
                phase = "draft" if runner.model_runner.is_draft_worker else "verify"
                key = runner._replay_graph_key
                shape = f"{forward_batch.batch_size}:{key.size}:{runner.captured_req_width}"
                self.graphs[phase][shape] += 1
            return output

        return execute


activity = Activity() if os.environ.get("AX_DFLASH_BENCH_ACTIVITY") == "1" else None


def install():
    if activity is None:
        return
    from sglang.srt.managers.scheduler_components.internal_state import SchedulerInternalStateController
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    patch_attribute(DFlashWorkerV2, "forward_batch_generation",
                    activity.wrap_round(DFlashWorkerV2.forward_batch_generation))
    patch_attribute(SchedulerInternalStateController, "get_internal_state",
                    activity.wrap_state(SchedulerInternalStateController.get_internal_state))
    patch_attribute(DecodeCudaGraphRunner, "execute", activity.wrap_graph(DecodeCudaGraphRunner.execute))
