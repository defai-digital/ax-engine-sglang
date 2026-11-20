"""Check GDN candidate ancestry and scratch writes without a CUDA runtime."""

import importlib.util
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_gdn_recurrent import replay
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
