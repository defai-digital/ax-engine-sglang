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
        # AX_DFLASH_BENCH_GRAPH_TIMING=1 times each verify-mode graph replay
        # with CUDA events, keyed by phase, live batch and graph variant.
        self.timing = os.environ.get("AX_DFLASH_BENCH_GRAPH_TIMING") == "1"
        self.graph_us = defaultdict(lambda: [0, 0.0])
        self.host_us = defaultdict(lambda: [0, 0.0])
        self._pending = []

    def add_host(self, name, seconds):
        row = self.host_us[name]
        row[0] += 1
        row[1] += seconds * 1e6

    def _collect(self, wait=False):
        keep = []
        for key, start, end in self._pending:
            if wait:
                end.synchronize()
            elif not end.query():
                keep.append((key, start, end))
                continue
            row = self.graph_us[key]
            row[0] += 1
            row[1] += start.elapsed_time(end) * 1e3
        self._pending = keep

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
            timing = {}
            if self.timing:
                self._collect(wait=True)
                timing = {"graph_us": {key: list(row) for key, row in self.graph_us.items()},
                          "host_us": {key: list(row) for key, row in self.host_us.items()}}
            reply.internal_state["ax_dflash_activity"] = {
                **timing,
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
            counted = forward_batch.forward_mode.is_target_verify() and not runner.ragged_verify_mode
            start = None
            if counted and self.timing:
                import torch

                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            output = original(runner, forward_batch, *args, **kwargs)
            if counted:
                phase = "draft" if runner.model_runner.is_draft_worker else "verify"
                key = runner._replay_graph_key
                shape = f"{forward_batch.batch_size}:{key.size}:{runner.captured_req_width}"
                self.graphs[phase][shape] += 1
                if start is not None:
                    end.record()
                    variant = key.variant_label or "native"
                    self._pending.append((f"{phase}:{forward_batch.batch_size}:{variant}", start, end))
                    if len(self._pending) > 64:
                        self._collect()
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
