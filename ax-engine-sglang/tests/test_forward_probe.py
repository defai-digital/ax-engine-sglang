"""Reject logit comparisons after the speculative history has diverged."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "nvfp4_forward_probe", Path(__file__).resolve().parents[2] / "scripts/nvfp4_forward_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ForwardProbeTests(unittest.TestCase):
    def test_prefill_and_decode_have_the_same_conditioning_prefix(self):
        expected = [(0, 0, (10, 20, 30))]
        self.assertEqual(probe.aligned_rows([10, 20, 30], [0, 1, 2], 1, [10, 20, 30], [40]), expected)
        self.assertEqual(probe.aligned_rows([30], [2], 1, [10, 20, 30], [40]), expected)

    def test_rejected_candidate_invalidates_later_rows_even_after_token_rejoins(self):
        # 99 is rejected in favor of 50; the later matching 60 cannot restore history.
        rows = probe.aligned_rows([30, 40, 99, 60], [2, 3, 4, 5], 4, [10, 20, 30], [40, 50, 60, 70])
        self.assertEqual(rows, [(0, 0, (10, 20, 30)), (1, 1, (10, 20, 30, 40))])

    def test_rows_past_output_budget_are_excluded(self):
        self.assertEqual(probe.aligned_rows([30, 40, 50], [2, 3, 4], 3, [10, 20, 30], [40]),
                         [(0, 0, (10, 20, 30))])

    def test_unsupported_tree_positions_and_shapes_are_rejected(self):
        for ids, positions, count in (([30, 40], [2, 2], 2), ([30], [2, 3], 1), ([30], [2], 2)):
            with self.subTest(positions=positions, count=count), self.assertRaises(ValueError):
                probe.aligned_rows(ids, positions, count, [10, 20, 30], [40, 50])
