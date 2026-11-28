from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ax_engine_sglang.ax_nvfp4_b12x import _route
from ax_engine_sglang.config import PluginConfig


class B12xRouteTests(unittest.TestCase):
    def setUp(self):
        self.original = Mock(return_value="original")
        self.direct = Mock(return_value="direct")
        self.backend = SimpleNamespace(is_flashinfer_cutedsl=Mock(return_value=True))
        self.gemm = _route(self.original, self.direct, lambda: self.backend)

    def call(self, rows=128, weight=(8704, 5120), quant_mode="w4a4"):
        return self.gemm(SimpleNamespace(shape=(rows, 1)), SimpleNamespace(shape=weight),
                         "input_sf", "weight_sf", "alpha", "dtype", 5120, quant_mode)

    def test_only_the_128_row_down_projection_uses_b12x(self):
        self.assertEqual(self.call(), "direct")
        self.original.assert_not_called()
        args = self.direct.call_args.args
        self.assertEqual(args[2:], ("input_sf", "weight_sf", "alpha", "dtype"))
        self.assertEqual(self.direct.call_args.kwargs, {"backend": "b12x"})
        self.call()
        self.assertEqual(self.backend.is_flashinfer_cutedsl.call_count, 1)

    def test_every_other_shape_and_mode_keeps_the_original_path(self):
        cases = (
            {"rows": 64},
            {"weight": (2560, 34816)},
            {"quant_mode": "w4a16"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                self.assertEqual(self.call(**changes), "original")
        self.direct.assert_not_called()
        self.assertEqual(self.original.call_count, len(cases))

    def test_the_fast_path_requires_cutedsl(self):
        self.backend.is_flashinfer_cutedsl = lambda: False
        with self.assertRaisesRegex(RuntimeError, "flashinfer_cutedsl"):
            self.call()
        self.direct.assert_not_called()

    def test_the_feature_is_opt_in_and_requires_replay(self):
        self.assertFalse(PluginConfig.from_env({}).nvfp4_b12x_down128)
        with self.assertRaisesRegex(ValueError, "requires AX_GDN_CHAIN_REPLAY=1"):
            PluginConfig.from_env({"AX_NVFP4_B12X_DOWN128": "1"})
        config = PluginConfig.from_env({"AX_NVFP4_B12X_DOWN128": "1", "AX_GDN_CHAIN_REPLAY": "1"})
        self.assertTrue(config.nvfp4_b12x_down128)
        with self.assertRaisesRegex(ValueError, "standalone AX_GDN_CHAIN_REPLAY=1"):
            PluginConfig.from_env({"AX_NVFP4_B12X_DOWN128": "1", "AX_GDN_CHAIN_REPLAY": "1",
                                   "AX_DRAFT_MODEL": "ax-draft"})


if __name__ == "__main__":
    unittest.main()
