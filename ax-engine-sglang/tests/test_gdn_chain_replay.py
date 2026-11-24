from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_gdn_chain_replay import _replay_commit


class IndexVector(tuple):
    """Exercise view selection and integer subtraction without a GPU."""

    @property
    def shape(self):
        return (len(self),)

    def numel(self):
        return len(self)

    def __sub__(self, value):
        return IndexVector(item - value for item in self)


class GdnChainReplayTests(unittest.TestCase):
    def setUp(self):
        self.pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(replayssm_spec_fold=True, replayssm_is_kda=False),
            get_mamba_indices=Mock(return_value=IndexVector((18, 7))),
            translate_mamba_indices=Mock(return_value=IndexVector((4, 1))),
            get_speculative_mamba2_params_all_layers=Mock(return_value=object()),
        )
        self.metadata = SimpleNamespace(mamba_cache_indices=IndexVector((4, 1, -1, -1)))
        backend = SimpleNamespace(linear_attn_backend=SimpleNamespace(forward_metadata=self.metadata))
        runner = SimpleNamespace(req_to_token_pool=self.pool, attn_backend=backend)
        self.worker = SimpleNamespace(target_worker=SimpleNamespace(model_runner=runner))
        self.batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            mamba_track_indices=None,
            req_pool_indices=IndexVector((9, 3, 0, 0)),
        )
        self.original = Mock()
        self.fold = Mock()
        self.commit = _replay_commit(self.original, self.fold)

    def run_commit(self, lengths):
        self.commit(self.worker, batch=self.batch, seq_lens_pre_verify=None,
                    seq_lens_post_verify=None, commit_lens=lengths)

    def test_reuses_only_live_physical_slots(self):
        lengths = IndexVector((1, 8))
        self.run_commit(lengths)
        self.original.assert_not_called()
        self.pool.get_mamba_indices.assert_not_called()
        self.pool.translate_mamba_indices.assert_not_called()
        self.fold.assert_called_once()
        args = self.fold.call_args.kwargs
        self.assertEqual(args["state_batch_indices"], (4, 1))
        self.assertIs(args["accept_lens"], lengths)
        self.assertIsInstance(args["last_correct_step_indices"], IndexVector)
        self.assertEqual(args["last_correct_step_indices"], (0, 7))

    def test_fallback_translates_pool_slots(self):
        self.metadata.mamba_cache_indices = None
        self.run_commit(IndexVector((2, 5)))
        self.pool.get_mamba_indices.assert_called_once_with((9, 3))
        self.pool.translate_mamba_indices.assert_called_once_with((18, 7))
        self.assertEqual(self.fold.call_args.kwargs["state_batch_indices"], (4, 1))

    def test_idle_and_empty_rounds_skip_commit(self):
        self.run_commit(IndexVector())
        self.batch.forward_mode.is_idle = lambda: True
        self.run_commit(IndexVector((1, 8)))
        self.fold.assert_not_called()
        self.pool.get_mamba_indices.assert_not_called()

    def test_tracking_is_rejected_in_replay_mode(self):
        self.batch.mamba_track_indices = IndexVector((7, 8))
        with self.assertRaisesRegex(RuntimeError, "requires radix caching to be disabled"):
            self.run_commit(IndexVector((1, 8)))
        self.fold.assert_not_called()

    def test_kda_commit_keeps_upstream_path(self):
        self.pool.mamba_pool.replayssm_is_kda = True
        lengths = IndexVector((1, 8))
        self.run_commit(lengths)
        self.original.assert_called_once_with(
            self.worker, batch=self.batch, seq_lens_pre_verify=None,
            seq_lens_post_verify=None, commit_lens=lengths,
        )
        self.fold.assert_not_called()
