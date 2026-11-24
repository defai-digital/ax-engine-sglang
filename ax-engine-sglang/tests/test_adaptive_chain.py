"""AX_POLICY_MODE=adaptive in ax_chain: width choice, round times and accept counts, with a stand-in torch."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from ax_engine_sglang import runtime
from ax_engine_sglang.config import get_config

ENV = {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8", "FC1_DRAFT_BLOCK": "16", "FC1_FUSED": "1",
       "AX_ADAPTIVE_MIN_SAMPLES": "8", "AX_ADAPTIVE_COOLDOWN": "4", "AX_ADAPTIVE_EXPLORE": "8"}


class Tensor:
    """The few tensor operations the accept counter uses."""
    readbacks = 0

    def __init__(self, values, ndim=1):
        self.values = list(values)
        self.ndim = ndim

    def reshape(self, *shape):
        return Tensor(self.values, ndim=len(shape))

    def to(self, _):
        return self

    def clamp(self, low, high):
        return Tensor((min(max(v, low), high) for v in self.values), self.ndim)

    device = "gpu"

    def scatter_add_(self, dim, index, source):
        if dim != 0 or index.ndim != 1:     # as torch: the counter has one dimension
            raise RuntimeError(f"scatter_add_ on a 1-D counter with dim={dim}, index of {index.ndim} dimensions")
        for position, value in zip(index.values, source.values):
            self.values[position] += value
        return self

    def tolist(self):
        Tensor.readbacks += 1
        return list(self.values)


def zeros(size, dtype, device):
    return Tensor([0] * size)


def ones_like(tensor):
    return Tensor([1] * len(tensor.values), tensor.ndim)


def bincount(tensor, minlength):
    raise AssertionError("torch.bincount reads back to the host on CUDA")


class AdaptiveChainTests(unittest.TestCase):
    def setUp(self):
        torch = ModuleType("torch")
        torch.int64, torch.bincount, torch.zeros, torch.ones_like = "int64", bincount, zeros, ones_like
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
                chain_schedule=(), chain_width=8, chain_width_table=())):
            self.assertEqual(self.chain.width(16, 4), 8)
        self.assertIsNone(self.chain.current_runtime().controller)

    def test_a_width_table_follows_the_batch_size(self):
        table = SimpleNamespace(chain_schedule=(), chain_width=None, chain_width_table=((1, 16), (5, 8)))
        with patch.object(self.chain, "_ADAPTIVE", False), patch.object(self.chain, "_CONFIG", table):
            self.assertEqual([self.chain.width(16, n) for n in (1, 2, 4, 5, 8, 64)], [16, 16, 16, 8, 8, 8])
            self.assertEqual(self.chain._configured_widths(16), [8, 16])

    def test_draft_width_switch_restores_position_offsets(self):
        chain = self.chain
        wide_offsets = tuple(range(16))
        narrow_offsets = wide_offsets[:8]
        wide_runner = SimpleNamespace(attn_backend=object())
        narrow_runner = SimpleNamespace(attn_backend=object())
        dmr = SimpleNamespace(attn_backend=wide_runner.attn_backend, decode_cuda_graph_runner=wide_runner)
        worker = self.worker
        worker.draft_model_runner = dmr
        worker._draft_sampler = object()
        worker._draft_block_spec_info = object()
        worker._block_pos_offsets = wide_offsets
        worker.draft_model = SimpleNamespace(block_size=16)
        worker.draft_model.set_block_size = Mock(
            side_effect=lambda width: setattr(worker.draft_model, "block_size", width)
        )
        worker._ax_draft_sets[8] = {
            "runner": narrow_runner, "sampler": object(), "spec_info": object(),
            "buffers": None, "position_offsets": narrow_offsets,
        }

        decode = SimpleNamespace(
            reqs=[object(), object()], forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
        )
        with patch.object(chain, "width", return_value=8):
            chain.begin_round(worker, decode)
            chain.begin_round(worker, decode)
        self.assertEqual(worker.block_size, 8)
        self.assertEqual(worker.draft_model.block_size, 8)
        self.assertIs(worker._block_pos_offsets, narrow_offsets)
        self.assertIs(dmr.decode_cuda_graph_runner, narrow_runner)

        chain.begin_round(worker)
        self.assertEqual(worker.block_size, 16)
        self.assertEqual(worker.draft_model.block_size, 16)
        self.assertIs(worker._block_pos_offsets, wide_offsets)
        self.assertIs(dmr.decode_cuda_graph_runner, wide_runner)
        self.assertEqual(worker.draft_model.set_block_size.call_args_list, [call(8), call(16)])
        self.assertIsNone(self.chain.current_runtime().controller)

    def test_accept_lengths_of_any_shape_are_counted(self):
        self.chain.begin_round(self.worker)
        width = self.chain.width(16, 2)
        self.chain._count_accepts(self.chain.current_runtime(), Tensor([2, 3], ndim=2))   # one column per request
        self.assertEqual(self.chain.current_runtime().accept_counts[(1, width)].values[2:4], [1, 1])

    def test_accept_counting_does_not_call_bincount(self):
        # The stand-in bincount raises: on CUDA the real one waits for the GPU every round.
        for _ in range(3):
            self.round(4, {16: 0.02, 8: 0.01}, requests=2)
        counts = self.chain.current_runtime().accept_counts
        self.assertEqual(sum(sum(held.values) for held in counts.values()), 6)

    def test_a_verify_asked_twice_is_counted_once(self):
        # SGLang asks can_run_graph in prepare_for_verify and again in _forward_raw, same batch.
        answers = iter([True, True, True, False])
        runner = SimpleNamespace(can_run_graph=lambda forward_batch: next(answers))
        self.chain._count_replays(runner, 8, "fa3")
        self.chain.begin_round(self.worker)
        first, second = SimpleNamespace(batch_size=2), SimpleNamespace(batch_size=2)
        for batch in (first, first, second, second):
            runner.can_run_graph(batch)
        # the second verify's later answer (eager) is the one _forward_raw acted on
        self.assertEqual(self.chain.current_runtime().stats["replay"], {"8:2:graph": 1, "8:2:eager": 1})
        self.assertFalse(self.chain.current_runtime().round_info["graph"])


if __name__ == "__main__":
    unittest.main()
