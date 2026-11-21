"""AX_POLICY_MODE=adaptive in ax_chain: width choice, round times and accept counts, with a stand-in torch."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from ax_engine_sglang import runtime
from ax_engine_sglang.config import get_config

ENV = {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8", "FC1_DRAFT_BLOCK": "16", "FC1_FUSED": "1",
       "AX_ADAPTIVE_MIN_SAMPLES": "8", "AX_ADAPTIVE_COOLDOWN": "4", "AX_ADAPTIVE_EXPLORE": "8"}


class Tensor:
    """The few tensor operations the accept counter uses."""
    readbacks = 0

    def __init__(self, values):
        self.values = list(values)

    def reshape(self, *_):
        return self

    def to(self, _):
        return self

    def clamp(self, low, high):
        return Tensor(min(max(v, low), high) for v in self.values)

    def __add__(self, other):
        return Tensor(a + b for a, b in zip(self.values, other.values))

    def tolist(self):
        Tensor.readbacks += 1
        return list(self.values)


def bincount(tensor, minlength):
    counts = [0] * minlength
    for value in tensor.values:
        counts[value] += 1
    return Tensor(counts)


class AdaptiveChainTests(unittest.TestCase):
    def setUp(self):
        torch = ModuleType("torch")
        torch.int64, torch.bincount = "int64", bincount
        self.enterContext(patch.dict(sys.modules, {"torch": torch}))
        self.enterContext(patch.dict(os.environ, ENV, clear=True))
        get_config.cache_clear()
        self.addCleanup(get_config.cache_clear)
        spec = importlib.util.spec_from_file_location("ax_chain_adaptive_test", PACKAGE / "ax_engine_sglang/ax_chain.py")
        self.chain = importlib.util.module_from_spec(spec)
        with patch("atexit.register", lambda function: function):
            spec.loader.exec_module(self.chain)
        self.clock = 0.0
        self.enterContext(patch.object(self.chain.time, "perf_counter", lambda: self.clock))
        self.worker = SimpleNamespace(block_size=16, _ax_draft_sets={})
        Tensor.readbacks = 0
        self.addCleanup(runtime._active.reset, runtime._active.set(None))

    def round(self, accepted, seconds, requests=1):
        """One chain decode round: begin, choose the width as `narrow` does, accept, then time passes."""
        chain = self.chain
        chain.begin_round(self.worker)
        state = chain.current_runtime()
        state.kind = "chain"
        width = chain.width(16, requests)
        accept = Tensor([min(accepted, width - 1)] * requests)
        with patch.object(chain, "_TREE", False):
            chain.record_accept(self.worker, SimpleNamespace(reqs=[]), None, accept)
        self.clock += seconds[width]
        return width

    def test_candidates_are_the_graph_widths(self):
        self.assertEqual(self.chain._configured_widths(16), [8, 16])

    def test_rounds_feed_the_controller_and_it_narrows(self):
        widths = [self.round(3, {8: 0.010, 16: 0.014}) for _ in range(200)]
        state = self.chain.current_runtime()
        self.assertEqual(state.controller.buckets[0].width, 8)
        self.assertGreater(widths[-32:].count(8), 24)
        self.assertEqual(state.stats["adaptive"]["buckets"][0]["width"], 8)
        arms = state.controller.buckets[0].arms
        self.assertAlmostEqual(arms[8].cost, 0.010, places=4)
        self.assertAlmostEqual(arms[16].cost, 0.014, places=4)
        self.assertAlmostEqual(arms[8].tokens, 4.0)

    def test_accept_counts_are_read_back_in_groups(self):
        for _ in range(64):
            self.round(15, {8: 0.010, 16: 0.014})
        # 64 rounds, read back every 16; warm-up rounds use both widths, so at most two tensors per read.
        self.assertLessEqual(Tensor.readbacks, 8)
        self.assertGreaterEqual(Tensor.readbacks, 4)

    def test_accept_lengths_outside_the_width_are_clamped(self):
        chain = self.chain
        chain.begin_round(self.worker)
        state = chain.current_runtime()
        state.kind = "chain"
        state.controller = chain.WidthController((8, 16), 8)
        state.pending = (0, 8)
        with patch.object(chain, "_TREE", False):
            chain.record_accept(self.worker, SimpleNamespace(reqs=[]), None, Tensor([-3, 2, 7, 40]))
        self.assertEqual(state.accept_counts[(0, 8)].values, [1, 0, 1, 0, 0, 0, 0, 2] + [0] * 8)

    def test_batch_sizes_use_their_own_bucket(self):
        for _ in range(64):
            self.round(15, {8: 0.020, 16: 0.022}, requests=8)
        state = self.chain.current_runtime()
        self.assertEqual(list(state.controller.buckets), [3])
        self.assertEqual(state.controller.buckets[3].arms[16].token_samples % 8, 0)

    def test_prefill_and_idle_time_are_not_charged_to_a_width(self):
        self.round(15, {8: 0.010, 16: 0.014})
        self.chain.begin_round(self.worker)   # the next batch: settles the chain round
        state = self.chain.current_runtime()
        self.assertEqual(state.controller.buckets[0].arms[16].cost_samples, 1)
        self.assertIsNone(state.pending)
        self.clock += 30.0                    # a prefill, then an idle server
        self.chain.begin_round(self.worker)
        self.assertEqual(state.controller.buckets[0].arms[16].cost_samples, 1)

    def test_static_mode_keeps_the_configured_width(self):
        self.chain.begin_round(self.worker)   # this worker's own round state
        with patch.object(self.chain, "_ADAPTIVE", False), patch.object(self.chain, "_CONFIG", SimpleNamespace(
                chain_schedule=(), chain_width=8)):
            self.assertEqual(self.chain.width(16, 4), 8)
        self.assertIsNone(self.chain.current_runtime().controller)


if __name__ == "__main__":
    unittest.main()
