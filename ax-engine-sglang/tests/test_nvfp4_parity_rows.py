"""Fixed projection row groups keep AR decode and verification on one kernel shape."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_nvfp4_verify import grouped
from ax_engine_sglang.config import PluginConfig

PARITY = {"AX_NVFP4_VERIFY_PARITY": "1", "AX_GDN_UNPACKED_DECODE": "1"}
HAS_SGLANG = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("sglang"))


class ParityRowsConfigTests(unittest.TestCase):
    def test_default_covers_a_sixteen_node_tree_in_one_group(self):
        self.assertEqual(PluginConfig.from_env(PARITY).nvfp4_parity_rows, 0)
        self.assertEqual(PluginConfig.from_env({**PARITY, "AX_NVFP4_PARITY_ROWS": "1"}).nvfp4_parity_rows, 1)
        PluginConfig.from_env({})  # the default must not require the parity profile

    def test_row_groups_need_the_parity_profile_and_a_supported_size(self):
        with self.assertRaisesRegex(ValueError, "needs AX_NVFP4_VERIFY_PARITY=1"):
            PluginConfig.from_env({"AX_NVFP4_PARITY_ROWS": "8"})
        for rows in ("-1", "65"):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "between 0 and 64"):
                PluginConfig.from_env({**PARITY, "AX_NVFP4_PARITY_ROWS": rows})

    def test_snapshot_cap_works_with_recurrent_verify(self):
        chain = {"AX_GDN_RECURRENT_VERIFY": "1", "FC1_FUSED": "0", "AX_TREE": "1", "AX_CHAIN_WIDTH": "8",
                 "AX_CHAIN_GRAPH_WIDTHS": "8", "AX_SSM_POSITIONS": "8"}
        self.assertEqual(PluginConfig.from_env({**PARITY, **chain}).ssm_positions, 8)
        with self.assertRaisesRegex(ValueError, "AX_SSM_POSITIONS needs"):
            PluginConfig.from_env({**chain, "AX_GDN_RECURRENT_VERIFY": "0"})

    def test_chain_attention_override_is_outside_the_parity_patches(self):
        chain = {"AX_GDN_RECURRENT_VERIFY": "1", "FC1_FUSED": "0", "AX_CHAIN_WIDTH": "8",
                 "AX_CHAIN_GRAPH_WIDTHS": "8"}
        PluginConfig.from_env({**PARITY, **chain})
        with self.assertRaisesRegex(ValueError, "does not support AX_CHAIN_ATTN"):
            PluginConfig.from_env({**PARITY, **chain, "AX_CHAIN_ATTN": "trtllm_mha"})


@unittest.skipUnless(HAS_SGLANG, "SGLang is required to install the projection wrapper")
class ForwardModeTests(unittest.TestCase):
    """Which forward modes regroup their projections, with a CPU stand-in for the kernel."""

    def calls(self, rows, mode, width):
        from types import SimpleNamespace
        from unittest.mock import patch

        import torch
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
        from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

        from ax_engine_sglang import ax_nvfp4_verify
        from ax_engine_sglang.compat import installation

        seen = []

        def apply(self, layer, x, bias=None):
            seen.append(x.shape[0])
            return x * 2

        def forward(self, x, positions, batch):
            return ModelOptFp8LinearMethod.apply(None, None, x)

        batch = SimpleNamespace(forward_mode=SimpleNamespace(
            is_target_verify=lambda: mode == "verify", is_decode=lambda: mode == "decode"))
        x = torch.arange(width * 3, dtype=torch.float32).reshape(width, 3)
        with (patch.object(ModelOptFp8LinearMethod, "apply", apply),
              patch.object(Qwen3_5ForConditionalGeneration, "forward", forward),
              self.assertRaisesRegex(RuntimeError, "end patch scope"), installation()):
            ax_nvfp4_verify.install(rows)
            result = Qwen3_5ForConditionalGeneration.forward(None, x, None, batch)
            torch.testing.assert_close(result, x * 2, atol=0, rtol=0)
            raise RuntimeError("end patch scope")
        return seen

    def test_decode_and_verify_use_the_same_groups(self):
        for rows, width, expected in ((16, 1, [16]), (16, 2, [16]), (16, 16, [16]), (8, 16, [8, 8]),
                                      (1, 2, [1, 1]), (1, 1, [1])):
            for mode in ("decode", "verify"):
                with self.subTest(rows=rows, width=width, mode=mode):
                    self.assertEqual(self.calls(rows, mode, width), expected)

    def test_invariant_mode_leaves_row_counts_alone(self):
        # rows 0 replaces the FP8 kernel instead of regrouping; the stand-in here is not FP8-capable.
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
        import torch
        from ax_engine_sglang import ax_nvfp4_verify
        from ax_engine_sglang.compat import installation

        with self.assertRaisesRegex(RuntimeError, "needs SGLang's generic FP8 GEMM"), installation():
            with patch.dict(os.environ):
                ax_nvfp4_verify.install(0)
                self.assertEqual(os.environ["SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE"], "0")
            method = SimpleNamespace(use_marlin=True, cutlass_fp8_supported=True)
            ModelOptFp8LinearMethod.apply(method, None, torch.zeros(2, 3))

    def test_prefill_is_not_regrouped(self):
        self.assertEqual(self.calls(16, "prefill", 5), [5])
        self.assertEqual(self.calls(1, "prefill", 5), [5])


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
