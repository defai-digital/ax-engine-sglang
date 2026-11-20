"""The paged decode of a branch may read its prefix and ancestors only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_verify_attention import branch_pages

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for page index tests")
class VerifyPagesTests(unittest.TestCase):
    def test_two_chains_with_different_prefix_lengths(self):
        indices = torch.tensor([10, 11, 12, 20, 21, 22, 23], dtype=torch.int32)
        bounds, pages = branch_pages(indices, [0, 3, 7], 2, None)
        self.assertEqual(bounds.tolist(), [0, 2, 5, 8, 12])
        self.assertEqual(pages.tolist(), [10, 11, 10, 11, 12, 20, 21, 22, 20, 21, 22, 23])

    def test_sibling_is_excluded_from_masked_history(self):
        indices = torch.tensor([19, 33, 28, 94], dtype=torch.int32)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]], dtype=torch.uint8)
        bounds, pages = branch_pages(indices, [0, 4], 3, mask.flatten())
        self.assertEqual(bounds.tolist(), [0, 2, 5, 8])
        self.assertEqual(pages.tolist(), [19, 33, 19, 33, 28, 19, 33, 94])
