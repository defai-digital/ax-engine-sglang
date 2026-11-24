"""Plugin start-up validation before SGLang has resolved the launch defaults (offline)."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.config import get_config, server_args_error, startup_config

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

    def test_parity_alone_waits_for_its_defaults(self):
        config, provisional = startup_config({"AX_NVFP4_VERIFY_PARITY": "1"}, hooks_available=True)
        self.assertTrue(provisional)
        self.assertTrue(config.gdn_unpacked_decode)
        self.assertFalse(config.tree_requested)
        with self.assertRaisesRegex(ValueError, "requires unpacked GDN decode"):
            startup_config({"AX_NVFP4_VERIFY_PARITY": "1"}, hooks_available=False)

    def test_parity_chain_settings_take_the_parity_tree_defaults(self):
        environ = {"AX_NVFP4_VERIFY_PARITY": "1", "AX_CHAIN_WIDTH": "8", "AX_CHAIN_GRAPH_WIDTHS": "8"}
        config, provisional = startup_config(environ, hooks_available=True)
        self.assertTrue(provisional)
        self.assertFalse(config.fused)
        self.assertTrue(config.gdn_recurrent_verify)
        self.assertEqual(config.tree_nodes, 16)

    def test_defaults_do_not_hide_an_invalid_value(self):
        with self.assertRaisesRegex(ValueError, "chain graph widths exceed AX_SSM_POSITIONS"):
            startup_config({**CHAIN8, "AX_SSM_POSITIONS": "4"}, hooks_available=True)
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            startup_config({"FC1_FUSED": "yes"}, hooks_available=True)

    def test_tree_node_count_also_limits_the_allocated_chain_width(self):
        base = {"FC1_TREE64": "1", "FC1_TREE_N": "16", "FC1_DRAFT_BLOCK": "32",
                "FC1_FUSED": "1", "AX_SSM_POSITIONS": "24"}
        for policy in (
            {"AX_CHAIN_WIDTH": "24"},
            {"AX_CHAIN_WIDTH_BY_BATCH": "1:24,4:8", "AX_CHAIN_GRAPH_WIDTHS": "24 8"},
            {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "24 8"},
        ):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "chain width 24 exceeds the 16 SSM positions"):
                startup_config({**base, **policy}, hooks_available=True)
        config, _ = startup_config({**base, "AX_CHAIN_WIDTH": "16"}, hooks_available=True)
        self.assertEqual(config.chain_width, 16)

    def test_resolved_dflash_block_limits_all_legacy_chain_graphs(self):
        base = {"FC1_TREE64": "0", "FC1_DRAFT_BLOCK": "32", "FC1_FUSED": "1", "AX_SSM_POSITIONS": "16"}
        args = SimpleNamespace(disable_radix_cache=True, speculative_num_draft_tokens=8)
        for policy in (
            {"AX_CHAIN_WIDTH": "16"},
            {"AX_CHAIN_WIDTH": "8", "AX_CHAIN_GRAPH_WIDTHS": "8 16"},
            {"AX_CHAIN_WIDTH_BY_BATCH": "1:8", "AX_CHAIN_GRAPH_WIDTHS": "8 16"},
            {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "8 16"},
        ):
            with self.subTest(policy=policy):
                config, _ = startup_config({**base, **policy}, hooks_available=True)
                self.assertEqual(server_args_error(config, args),
                                 "chain width 16 exceeds the 8 SSM positions allocated per slot by DFlash")
        config, _ = startup_config({**base, "AX_CHAIN_WIDTH": "8", "AX_CHAIN_GRAPH_WIDTHS": "8"},
                                   hooks_available=True)
        self.assertIsNone(server_args_error(config, args))

    def test_doctor_checks_legacy_capacity_with_checkpoint_or_dflash_block_flag(self):
        from ax_engine_sglang.doctor import diagnose, parse

        env = {"AX_AUTO_DEFAULTS": "0", "FC1_TREE64": "0", "FC1_DRAFT_BLOCK": "32", "FC1_FUSED": "1",
               "AX_SSM_POSITIONS": "16", "AX_CHAIN_WIDTH": "8", "AX_CHAIN_GRAPH_WIDTHS": "8 16"}
        command = ["target", "--speculative-algorithm", "FC1_DFLASH_TREE", "--speculative-draft-model-path", "draft",
                   "--disable-radix-cache"]
        for arguments, draft in (([], {"block_size": 8}),
                                 (["--speculative-dflash-block-size", "8"], {"dflash_config": {"block_size": 16}})):
            with self.subTest(arguments=arguments):
                def load(path, filename, _revision):
                    return draft if path == "draft" and filename == "config.json" else {}

                report = diagnose(parse(command + arguments), env, load=load, capability=(11, 0), sglang="0.5.23")
                self.assertIn("chain width 16 exceeds the 8 SSM positions allocated per slot by DFlash", report["errors"])


class FusedTreeShapeTests(unittest.TestCase):
    def test_fused_tree_rejects_width_without_inverse_kernel(self):
        from ax_engine_sglang.config import PluginConfig

        for nodes in (16, 32, 64):
            with self.subTest(nodes=nodes):
                self.assertEqual(PluginConfig.from_env({"FC1_FUSED": "1", "FC1_TREE_N": str(nodes)}).tree_nodes,
                                 nodes)
        with self.assertRaisesRegex(ValueError, "16, 32 or 64"):
            PluginConfig.from_env({"FC1_FUSED": "1", "FC1_TREE_N": "24"})

    def test_replay_refuses_tree_or_parity_options(self):
        from ax_engine_sglang.config import PluginConfig

        for option in ("FC1_TREE64", "AX_GDN_RECURRENT_VERIFY", "AX_NVFP4_VERIFY_PARITY", "AX_GDN_UNPACKED_DECODE"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "stock width-eight DFLASH"):
                PluginConfig.from_env({"AX_GDN_CHAIN_REPLAY": "1", option: "1"})

    def test_dflash_kv_graph_requires_native_gdn_replay(self):
        from ax_engine_sglang.config import PluginConfig

        self.assertFalse(PluginConfig.from_env({}).dflash_kv_graph)
        with self.assertRaisesRegex(ValueError, "requires AX_GDN_CHAIN_REPLAY=1"):
            PluginConfig.from_env({"AX_DFLASH_KV_GRAPH": "1"})
        self.assertTrue(PluginConfig.from_env(
            {"AX_DFLASH_KV_GRAPH": "1", "AX_GDN_CHAIN_REPLAY": "1"},
        ).dflash_kv_graph)

    def test_direct_kv_requires_its_own_gdn_replay_lane(self):
        from ax_engine_sglang.config import PluginConfig

        with self.assertRaisesRegex(ValueError, "requires AX_GDN_CHAIN_REPLAY=1"):
            PluginConfig.from_env({"AX_DFLASH_KV_DIRECT": "1"})
        options = {"AX_DFLASH_KV_DIRECT": "1", "AX_GDN_CHAIN_REPLAY": "1"}
        self.assertTrue(PluginConfig.from_env(options).dflash_kv_direct)
        with self.assertRaisesRegex(ValueError, "separate experiments"):
            PluginConfig.from_env({**options, "AX_DFLASH_KV_GRAPH": "1"})


if __name__ == "__main__":
    unittest.main()
