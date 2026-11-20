"""Plugin start-up validation before SGLang has resolved the launch defaults (offline)."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.config import get_config, startup_config

# The width-eight multi-user settings from the README, on the minimal launch command.
CHAIN8 = {"AX_CHAIN_WIDTH": "8", "AX_CHAIN_GRAPH_WIDTHS": "8", "AX_CHAIN_DRAFT_WIDTHS": "8",
          "AX_SSM_POSITIONS": "8"}


class StartupConfigTests(unittest.TestCase):
    def test_chain_settings_wait_for_the_tree_defaults(self):
        config, provisional = startup_config(dict(CHAIN8), hooks_available=True)
        self.assertTrue(provisional)
        self.assertTrue(config.fused)
        self.assertEqual(config.ssm_positions, 8)

    def test_a_complete_configuration_is_final(self):
        config, provisional = startup_config({**CHAIN8, "FC1_FUSED": "1"}, hooks_available=True)
        self.assertFalse(provisional)
        self.assertTrue(config.fused)

    def test_explicit_settings_still_win_over_the_defaults(self):
        with self.assertRaisesRegex(ValueError, "AX_SSM_POSITIONS needs FC1_FUSED=1"):
            startup_config({**CHAIN8, "FC1_FUSED": "0"}, hooks_available=True)

    def test_without_resolution_hooks_the_settings_must_be_explicit(self):
        with self.assertRaisesRegex(ValueError, "AX_SSM_POSITIONS needs FC1_FUSED=1"):
            startup_config(dict(CHAIN8), hooks_available=False)
        with self.assertRaisesRegex(ValueError, "AX_SSM_POSITIONS needs FC1_FUSED=1"):
            startup_config({**CHAIN8, "AX_AUTO_DEFAULTS": "0"}, hooks_available=True)

    def test_get_config_is_final_after_the_resolver_clears_it(self):
        with patch.dict(os.environ, CHAIN8, clear=True), \
                patch("ax_engine_sglang.defaults.resolution_hooks_available", return_value=True):
            get_config.cache_clear()
            self.addCleanup(get_config.cache_clear)
            self.assertTrue(startup_config()[1])
            self.assertTrue(get_config().fused)   # install-time readers such as ax_qsa see the defaults
            os.environ["FC1_FUSED"] = "0"
            get_config.cache_clear()
            with self.assertRaisesRegex(ValueError, "AX_SSM_POSITIONS needs FC1_FUSED=1"):
                get_config()

    def test_defaults_do_not_hide_an_invalid_value(self):
        with self.assertRaisesRegex(ValueError, "chain graph widths exceed AX_SSM_POSITIONS"):
            startup_config({**CHAIN8, "AX_SSM_POSITIONS": "4"}, hooks_available=True)
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            startup_config({"FC1_FUSED": "yes"}, hooks_available=True)


if __name__ == "__main__":
    unittest.main()
