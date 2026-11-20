"""Fixed projection row groups keep AR decode and verification on one kernel shape."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_nvfp4_verify import grouped
from ax_engine_sglang.config import PluginConfig

PARITY = {"AX_NVFP4_VERIFY_PARITY": "1", "AX_GDN_UNPACKED_DECODE": "1"}


class ParityRowsConfigTests(unittest.TestCase):
    def test_default_covers_a_sixteen_node_tree_in_one_group(self):
        self.assertEqual(PluginConfig.from_env(PARITY).nvfp4_parity_rows, 16)
        self.assertEqual(PluginConfig.from_env({**PARITY, "AX_NVFP4_PARITY_ROWS": "1"}).nvfp4_parity_rows, 1)
        PluginConfig.from_env({})  # the default must not require the parity profile

    def test_row_groups_need_the_parity_profile_and_a_supported_size(self):
        with self.assertRaisesRegex(ValueError, "needs AX_NVFP4_VERIFY_PARITY=1"):
            PluginConfig.from_env({"AX_NVFP4_PARITY_ROWS": "8"})
        for rows in ("0", "65"):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "between 1 and 64"):
                PluginConfig.from_env({**PARITY, "AX_NVFP4_PARITY_ROWS": rows})


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for the tensor check")
class GroupedProjectionTests(unittest.TestCase):
    def check(self, count, rows, expected_calls):
        import torch

        x = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) + 1
        calls = []

        def project(group):
            self.assertTrue(group.is_contiguous())
            calls.append(group.clone())
            return group * 2

        torch.testing.assert_close(grouped(project, x, rows), x * 2, atol=0, rtol=0)
        self.assertEqual([len(group) for group in calls], [rows] * expected_calls)
        padding = torch.cat(calls)[count:]
        self.assertEqual(padding.count_nonzero().item(), 0)

    def test_single_ar_row_is_padded_to_one_group(self):
        self.check(count=1, rows=8, expected_calls=1)

    def test_full_chain_window_is_one_call(self):
        self.check(count=8, rows=8, expected_calls=1)

    def test_tree_window_splits_and_pads_the_last_group(self):
        self.check(count=16, rows=8, expected_calls=2)
        self.check(count=11, rows=8, expected_calls=2)

    def test_empty_input_reaches_the_projection_unchanged(self):
        import torch

        result = grouped(lambda group: group.new_zeros(len(group), 5), torch.zeros(0, 3), 8)
        self.assertEqual(result.shape, (0, 5))

    def test_one_row_groups_run_each_candidate_alone(self):
        self.check(count=5, rows=1, expected_calls=5)


if __name__ == "__main__":
    unittest.main()
