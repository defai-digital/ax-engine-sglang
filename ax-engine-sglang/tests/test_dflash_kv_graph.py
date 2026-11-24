from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang import ax_dflash_kv_graph as kv_graph
from ax_engine_sglang.ax_dflash_activity import Activity


class Tensor:
    device = SimpleNamespace(type="cuda")
    dtype = "bf16"

    def __init__(self, *shape):
        self.shape = shape
        self.ndim = len(shape)

    def is_contiguous(self):
        return True


class DraftKVGraphTests(unittest.TestCase):
    def setUp(self):
        self.pool = SimpleNamespace(use_hnd=False, v_row_dim=128, row_dim=128, dtype=Tensor.dtype,
                                    store_dtype=Tensor.dtype, k_buffer=[object()], v_buffer=[object()])
        rope = SimpleNamespace(cos_sin_cache=Tensor(512, 64))
        helper = SimpleNamespace(rotary_emb=rope, max_position_hint=255)
        draft = SimpleNamespace(fc=SimpleNamespace(in_features=128, weight=Tensor(128, 128)),
                                hidden_norm=SimpleNamespace(weight=Tensor(128)))
        self.worker = SimpleNamespace(block_size=8, lilicorr=None, draft_model=draft, _fused_kv_helper=helper,
                                      _use_fused_kv_materialize=True,
                                      draft_model_runner=SimpleNamespace(token_to_kv_pool=self.pool))
        self.cuda = SimpleNamespace(current_stream=Mock(return_value=object()),
                                    is_current_stream_capturing=Mock(return_value=False))
        self.builder = self.enterContext(patch.object(kv_graph, "_AppendGraph"))
        self.original = Mock()
        self.append = kv_graph._wrap_append(
            self.original, SimpleNamespace(cuda=self.cuda, inference_mode=nullcontext),
            type(self.pool), nullcontext,
        )

    def append_batch(self, bs=8, **overrides):
        args = dict(target_hidden=Tensor(bs * 8, 128), cache_loc=Tensor(bs * 8), positions=Tensor(bs * 8),
                    cache_loc_2d=Tensor(bs, 8), commit_lens=Tensor(bs))
        args.update(overrides)
        self.append(self.worker, **args)
        return args

    def test_replays_each_live_shape_with_fresh_inputs(self):
        graph8, graph1 = Mock(), Mock()
        self.builder.side_effect = [graph8, graph1]
        self.append_batch()
        self.append_batch(1)
        last = self.append_batch()
        self.assertEqual(self.builder.call_count, 2)
        self.assertEqual(graph8.replay.call_count, 2)
        graph8.replay.assert_called_with(tuple(last[name] for name in
                                             ("target_hidden", "positions", "cache_loc_2d", "commit_lens")))
        graph1.replay.assert_called_once()
        self.original.assert_not_called()

    def test_prefill_and_unsupported_inputs_keep_native_append(self):
        self.append_batch(cache_loc_2d=None, commit_lens=None, extend_lens=Tensor(8))
        self.append_batch(target_hidden=Tensor(65, 128))
        self.append_batch(bs=17)
        self.cuda.is_current_stream_capturing.return_value = True
        self.append_batch()
        self.assertEqual(self.original.call_count, 4)
        self.builder.assert_not_called()

    def test_replaced_cache_and_other_stream_use_eager(self):
        self.append_batch()
        stream = self.cuda.current_stream.return_value
        self.cuda.current_stream.return_value = object()
        self.append_batch()
        self.cuda.current_stream.return_value = stream
        self.pool.k_buffer[0] = object()
        self.append_batch()
        self.builder.assert_called_once()
        self.builder.return_value.replay.assert_called_once()
        self.assertEqual(self.original.call_count, 2)

    def test_failed_capture_uses_eager_once_per_call_without_retrying_capture(self):
        self.builder.side_effect = RuntimeError("capture failed")
        with self.assertLogs(kv_graph.logger, level="WARNING"):
            self.append_batch()
        self.append_batch()
        self.builder.assert_called_once()
        self.assertEqual(self.original.call_count, 2)

    def test_failed_live_replay_does_not_repeat_the_write_in_eager(self):
        self.builder.return_value.replay.side_effect = RuntimeError("replay failed")
        with self.assertRaisesRegex(RuntimeError, "replay failed"):
            self.append_batch()
        self.original.assert_not_called()

    def test_activity_counts_live_decode_rounds_and_does_not_reuse_a_marker(self):
        activity = Activity()
        mode = SimpleNamespace(is_extend=lambda: False, is_idle=lambda: False)
        batch = SimpleNamespace(forward_mode=mode, is_extend_in_batch=False, seq_lens=[32, 48])
        forward = activity.wrap_round(lambda worker, batch: self.append_batch(bs=2))
        snapshot = activity.wrap_state(lambda controller, request: SimpleNamespace(internal_state={}))

        with patch.object(kv_graph, "activity", activity):
            forward(self.worker, batch)
            before = snapshot(None, None).internal_state["ax_dflash_activity"]
            self.cuda.is_current_stream_capturing.return_value = True
            forward(self.worker, batch)
            batch.is_extend_in_batch = True
            forward(self.worker, batch)

        self.assertEqual(before["rounds"], {"2": 1})
        self.assertEqual(snapshot(None, None).internal_state["ax_dflash_activity"], {
            "rounds": {"2": 2}, "optimized": {"kv_graph": {"2": 1}}, "capture_attempts": 1,
        })


if __name__ == "__main__":
    unittest.main()
