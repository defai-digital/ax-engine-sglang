"""Check GDN candidate ancestry, scratch writes and the verify parent table without a CUDA runtime."""

import importlib.util
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_gdn_recurrent import one_row, replay, verify
from ax_engine_sglang.config import PluginConfig


class RecurrentVerifyTests(unittest.TestCase):
    def test_flag_requires_native_tree_commit(self):
        env = {"AX_GDN_RECURRENT_VERIFY": "1"}
        self.assertTrue(PluginConfig.from_env(env).gdn_recurrent_verify)
        with self.assertRaisesRegex(ValueError, "requires FC1_FUSED=0"):
            PluginConfig.from_env({**env, "FC1_FUSED": "1"})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for the tensor check")
    def test_replay_uses_parent_snapshots_and_skips_padding(self):
        import torch

        def decode(q, k, v, a, b, *, ssm_states, cache_indices, query_start_loc, **kwargs):
            self.assertEqual(query_start_loc.tolist(), [0, 1, 2, 3])
            incoming = ssm_states[cache_indices.clamp_min(0)]
            updated = incoming + v[0]
            valid = cache_indices >= 0
            ssm_states[cache_indices[valid]] = updated[valid]
            return updated.masked_fill(~valid[:, None, None], 0).unsqueeze(0)

        def scatter(dst, src, dst_rows, steps, src_rows):
            for target, origin, step in zip(dst_rows.tolist(), src_rows.tolist(), steps.tolist()):
                if step >= 0:
                    dst[0, target].copy_(src[0, origin, step])

        committed = torch.tensor([100, 200, 300], dtype=torch.float32).reshape(3, 1, 1)
        inputs = torch.tensor([1, 2, 4, 8, 10, 20, 40, 80, 7, 7, 7, 7], dtype=torch.float32)
        qkv = inputs.reshape(1, 12, 1, 1)
        gates = torch.zeros(12, 1)
        cases = (
            (None, [301, 303, 307, 315], [110, 130, 170, 250]),
            (torch.tensor([[-1, 0, 0, 1], [-1, 0, 1, 0], [-1, 99, 99, 99]]),
             [301, 303, 305, 311], [110, 130, 170, 190]),
        )
        for parents, first, second in cases:
            with self.subTest(tree=parents is not None):
                snapshots = torch.full((3, 5, 1, 1), -10.0)
                actual = replay(
                    decode, scatter, None, None, qkv, qkv, qkv, gates, gates,
                    # A padded row can carry a slot placeholder and share a scratch row.
                    ssm_states=committed, cache_indices=torch.tensor([2, 0, 1]),
                    query_start_loc=torch.tensor([0, 4, 8, 8]), intermediate_states_buffer=snapshots,
                    intermediate_state_indices=torch.tensor([1, 0, 0]),
                    cache_steps=4, retrieve_parent_token=parents,
                )
                self.assertEqual(actual.flatten().tolist(), first + second + [0] * 4)
                self.assertEqual(snapshots[:, :, 0, 0].tolist(),
                                 [second + [-10], first + [-10], [-10] * 5])
                self.assertEqual(committed.flatten().tolist(), [100, 200, 300])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for the tensor check")
class VerifyParentTableTests(unittest.TestCase):
    def call(self, batch, width, parents):
        import torch

        seen = {}

        def native(*args, **kwargs):
            seen.update(kwargs)
            return "output"

        q = torch.zeros(1, batch * width, 2, 4)
        offsets = torch.arange(0, (batch + 1) * width, width, dtype=torch.int32)
        result = verify(native, None, None, q, q, q, None, None, query_start_loc=offsets,
                        cache_steps=width, retrieve_parent_token=parents, ssm_states="states",
                        intermediate_states_buffer=torch.zeros(batch + 1, width, 1),
                        intermediate_state_indices=torch.arange(batch, dtype=torch.int32))
        self.assertEqual(result, "output")
        self.assertEqual(seen["ssm_states"], "states")
        return seen["retrieve_parent_token"]

    def test_chain_gets_a_linear_parent_table(self):
        import torch

        table = self.call(batch=2, width=4, parents=None)
        self.assertEqual(table.dtype, torch.int32)
        self.assertEqual(table.tolist(), [[-1, 0, 1, 2]] * 2)

    def test_tree_parents_pass_through(self):
        import torch

        parents = torch.tensor([[-1, 0, 0, 1]], dtype=torch.int32)
        self.assertIs(self.call(batch=1, width=4, parents=parents), parents)

    def test_ragged_windows_and_wrong_tables_are_refused(self):
        import torch

        with self.assertRaisesRegex(ValueError, "uniform verification window"):
            verify(None, None, None, torch.zeros(1, 7, 2, 4), None, None, None, None,
                   query_start_loc=torch.tensor([0, 4, 7]), cache_steps=4,
                   intermediate_states_buffer=None, intermediate_state_indices=None)
        with self.assertRaisesRegex(ValueError, "one local parent index per candidate"):
            self.call(batch=2, width=4, parents=torch.zeros(1, 4, dtype=torch.int32))

    def test_one_request_uses_a_capped_buffer_as_one_row(self):
        import torch

        seen = {}
        q = torch.zeros(1, 64, 2, 4)
        capped = torch.arange(9 * 8 * 3, dtype=torch.float32).reshape(9, 8, 3)  # 9 rows of 8 positions
        verify(lambda *args, **kwargs: seen.update(kwargs), None, None, q, q, q, None, None,
               query_start_loc=torch.tensor([0, 64]), cache_steps=64, intermediate_states_buffer=capped,
               intermediate_state_indices=torch.tensor([5], dtype=torch.int32))
        row = seen["intermediate_states_buffer"]
        self.assertEqual(tuple(row.shape), (1, 72, 3))
        self.assertEqual(row.data_ptr(), capped.data_ptr())  # a view: the commit reads what verify wrote
        self.assertEqual(seen["intermediate_state_indices"].tolist(), [0])
        self.assertEqual(tuple(one_row(torch.zeros(48, 9, 8, 3), layers=1).shape), (48, 1, 72, 3))

    def test_capped_buffer_refuses_several_requests_and_oversized_trees(self):
        import torch

        for batch, rows in ((2, 9), (1, 7)):  # two requests; 7 rows of 8 hold 56 < 64
            q = torch.zeros(1, batch * 64, 2, 4)
            with self.subTest(batch=batch, rows=rows), self.assertRaisesRegex(ValueError, "needs 64 snapshot positions"):
                verify(None, None, None, q, q, q, None, None,
                       query_start_loc=torch.arange(0, (batch + 1) * 64, 64), cache_steps=64,
                       intermediate_states_buffer=torch.zeros(rows, 8, 3),
                       intermediate_state_indices=torch.zeros(batch, dtype=torch.int32))
