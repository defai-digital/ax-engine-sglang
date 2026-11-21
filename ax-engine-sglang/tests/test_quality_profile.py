"""AX_QUALITY_PROFILE, the qualification rows and `ax-engine-sglang doctor`, offline."""
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang import cli, doctor
from ax_engine_sglang.config import PluginConfig, startup_config
from ax_engine_sglang.defaults import LaunchConfigError, plan
from ax_engine_sglang.qualification import EXACT_MEASURED, UNSUPPORTED, lookup, target_key

QWEN38_27B = {"architectures": ["Qwen3_5ForConditionalGeneration"],
              "text_config": {"model_type": "qwen3_5_text", "linear_num_value_heads": 48,
                              "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16}}
QWEN38_27B_NVFP4 = {**QWEN38_27B, "quantization_config": {"quant_method": "modelopt",
                                                          "quant_algo": "MIXED_PRECISION"}}
DFLASH2 = {"architectures": ["DFlash2DraftModel"], "dflash_config": {"block_size": 8}}
TREE = "FC1_DFLASH_TREE"
VERIFIED = {"AX_QUALITY_PROFILE": "verified"}
RTX, B200 = (12, 0), (10, 0)


def args(algorithm=TREE, **values):
    base = dict(speculative_algorithm=algorithm, speculative_num_draft_tokens=16,
                moe_runner_backend="auto", speculative_moe_runner_backend=None,
                max_running_requests=None, cuda_graph_max_bs_decode=None)
    base.update(values)
    return SimpleNamespace(**base)


class QualificationTests(unittest.TestCase):
    def test_only_the_nvfp4_27b_has_rows(self):
        self.assertEqual(target_key(QWEN38_27B_NVFP4), "qwen3.8-27b-nvfp4")
        self.assertEqual(target_key(QWEN38_27B, {"quantization": {"quant_algo": "NVFP4"}}), "qwen3.8-27b-nvfp4")
        self.assertIsNone(target_key(QWEN38_27B))
        self.assertIsNone(target_key(None))

    def test_rows_are_measured_not_gate_verified(self):
        for route in ("ar", "tree", "chain"):
            row = lookup(RTX, "qwen3.8-27b-nvfp4", route)
            self.assertEqual(row.status, EXACT_MEASURED)
            self.assertTrue(row.admits_verified)
            self.assertTrue(all((Path(__file__).resolve().parents[2] / path.split("#")[0]).is_file()
                                for path in row.evidence))

    def test_unmeasured_combinations_are_unsupported(self):
        for capability, target in ((B200, "qwen3.8-27b-nvfp4"), ((11, 0), "qwen3.8-27b-nvfp4"),
                                   (RTX, None), (None, "qwen3.8-27b-nvfp4")):
            row = lookup(capability, target, "tree")
            self.assertEqual(row.status, UNSUPPORTED)
            self.assertFalse(row.admits_verified)


class VerifiedProfileTests(unittest.TestCase):
    def test_verified_turns_on_the_parity_path_on_a_measured_gpu(self):
        result = plan(args(), VERIFIED, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        self.assertEqual(result.env["AX_NVFP4_VERIFY_PARITY"], ("1", "AX_QUALITY_PROFILE=verified"))
        self.assertEqual(result.env["AX_GDN_RECURRENT_VERIFY"][0], "1")
        self.assertEqual(result.args["attention_backend"][0], "flashinfer")
        self.assertTrue(any("exact_measured" in note for note in result.notes))
        self.assertTrue(any("prefill chunk" in note for note in result.notes))

    def test_verified_covers_autoregressive_serving(self):
        result = plan(args(None), VERIFIED, QWEN38_27B_NVFP4, None, capability=RTX)
        self.assertEqual(result.env["AX_NVFP4_VERIFY_PARITY"][0], "1")

    def test_verified_refuses_unmeasured_launches(self):
        cases = (
            (dict(capability=B200), QWEN38_27B_NVFP4, {}, "SM100"),
            (dict(capability=None), QWEN38_27B_NVFP4, {}, "no CUDA GPU"),
            (dict(capability=RTX), QWEN38_27B, {}, "not a measured checkpoint"),
            (dict(capability=RTX), None, {}, "not a measured checkpoint"),
            (dict(capability=RTX), QWEN38_27B_NVFP4, {"AX_NVFP4_VERIFY_PARITY": "0"}, "contradicts"),
            (dict(capability=RTX), QWEN38_27B_NVFP4, {"AX_AUTO_DEFAULTS": "0"}, "AX_AUTO_DEFAULTS"),
            (dict(capability=RTX), QWEN38_27B_NVFP4, {"AX_GDN_CHAIN_REPLAY": "1"}, "AX_GDN_CHAIN_REPLAY"),
        )
        for keywords, target, extra, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(LaunchConfigError, message):
                plan(args(), {**VERIFIED, **extra}, target, DFLASH2, **keywords)

    def test_verified_refuses_stock_speculative_algorithms(self):
        with self.assertRaisesRegex(LaunchConfigError, "DFLASH verifies"):
            plan(args("DFLASH"), VERIFIED, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)

    def test_parity_without_the_profile_is_unchanged(self):
        result = plan(args(), {"AX_NVFP4_VERIFY_PARITY": "1"}, QWEN38_27B_NVFP4, DFLASH2, capability=B200)
        self.assertNotIn("AX_NVFP4_VERIFY_PARITY", result.env)
        self.assertFalse(any("verified profile" in note for note in result.notes))

    def test_startup_waits_for_the_verified_defaults(self):
        config, provisional = startup_config(dict(VERIFIED), hooks_available=True)
        self.assertTrue(provisional)
        self.assertTrue(config.nvfp4_verify_parity)
        self.assertEqual(config.quality_profile, "verified")

    def test_verified_needs_the_resolution_hooks(self):
        explicit = {**VERIFIED, "AX_NVFP4_VERIFY_PARITY": "1", "AX_GDN_UNPACKED_DECODE": "1"}
        for environ, hooks in ((dict(VERIFIED), False), (explicit, False),
                               ({**explicit, "AX_AUTO_DEFAULTS": "0"}, True)):
            with self.subTest(environ=environ, hooks=hooks), self.assertRaises(ValueError):
                startup_config(environ, hooks_available=hooks)
        config, provisional = startup_config(explicit, hooks_available=True)
        self.assertFalse(provisional)

    def test_profiles_reject_contradictions(self):
        with self.assertRaisesRegex(ValueError, "contradicts"):
            PluginConfig.from_env({"AX_QUALITY_PROFILE": "fast", "AX_NVFP4_VERIFY_PARITY": "1",
                                   "AX_GDN_UNPACKED_DECODE": "1"})
        with self.assertRaisesRegex(ValueError, "verified or fast"):
            PluginConfig.from_env({"AX_QUALITY_PROFILE": "exact"})
        self.assertEqual(PluginConfig.from_env({"AX_QUALITY_PROFILE": "fast"}).quality_profile, "fast")


class DoctorTests(unittest.TestCase):
    def report(self, argv, environ=None, capability=RTX, sglang="0.5.23", target=QWEN38_27B_NVFP4):
        files = {("target", "config.json"): target, ("draft", "config.json"): DFLASH2}
        options = doctor.parser().parse_args(argv)
        return doctor.diagnose(options, environ or {}, load=lambda path, name, revision=None: files.get((path, name)),
                               capability=capability, sglang=sglang)

    TREE_ARGV = ["target", "--speculative-algorithm", TREE, "--speculative-draft-model-path", "draft",
                 "--speculative-num-draft-tokens", "16"]

    def test_a_measured_tree_launch_is_supported(self):
        report = self.report(self.TREE_ARGV)
        self.assertEqual((report["status"], report["errors"], report["route"]), ("supported", [], "tree"))
        self.assertEqual(report["derived"]["env"]["AX_GDN_LAYERS"]["value"], "48")
        self.assertEqual(report["quality"]["exact_output"]["status"], EXACT_MEASURED)
        self.assertEqual(report["target"]["qualified_as"], "qwen3.8-27b-nvfp4")
        self.assertFalse(report["gdn_chain_replay"]["eligible"])
        json.dumps(report)

    def test_verified_on_an_unmeasured_gpu_is_unsupported(self):
        report = self.report(self.TREE_ARGV, VERIFIED, capability=B200)
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("SM100", report["errors"][0])
        self.assertEqual(report["quality"]["exact_output"]["status"], UNSUPPORTED)

    def test_experimental_features_and_untested_sglang_are_flagged(self):
        adaptive = self.report(self.TREE_ARGV + ["--max-running-requests", "8"],
                               {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8"})
        self.assertEqual((adaptive["status"], adaptive["policy_mode"]), ("experimental", "adaptive"))
        self.assertEqual(adaptive["experimental_features"], ["AX_POLICY_MODE=adaptive"])
        self.assertEqual(self.report(self.TREE_ARGV, sglang="0.6.0")["status"], "experimental")
        self.assertEqual(self.report(self.TREE_ARGV, sglang=None)["status"], "experimental")
        self.assertEqual(self.report(self.TREE_ARGV, capability=None)["status"], "experimental")

    def test_replay_eligibility_names_the_missing_requirement(self):
        argv = ["target", "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", "draft",
                "--speculative-num-draft-tokens", "8"]
        missing = self.report(argv, {"AX_GDN_CHAIN_REPLAY": "1"})
        self.assertEqual(missing["status"], "unsupported")
        self.assertIn("--disable-radix-cache", missing["errors"][0])
        ready = self.report(argv + ["--disable-radix-cache"], {"AX_GDN_CHAIN_REPLAY": "1"})
        self.assertEqual((ready["status"], ready["gdn_chain_replay"]["eligible"]), ("experimental", True))

    def test_invalid_settings_are_reported_once(self):
        report = self.report(self.TREE_ARGV, {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16"})
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("at least two widths", report["errors"][0])
        both = self.report(self.TREE_ARGV + ["--speculative-adaptive"],
                           {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8"})
        self.assertIn("--speculative-adaptive", both["errors"][0])

    def test_an_unreadable_target_is_reported(self):
        report = self.report(["missing", "--speculative-algorithm", TREE, "--speculative-num-draft-tokens", "16"])
        self.assertEqual(report["status"], "unsupported")
        self.assertIn("could not read config.json of missing", report["warnings"])

    def test_cli_prints_json_and_fails_on_unsupported(self):
        report = self.report(self.TREE_ARGV, VERIFIED, capability=B200)
        output = io.StringIO()
        with patch.object(doctor, "diagnose", return_value=report), contextlib.redirect_stdout(output), \
                patch.object(sys, "argv", ["ax-engine-sglang", "doctor", "target", "--sm", "10.0", "--json"]), \
                self.assertRaises(SystemExit) as stopped:
            cli.main()
        self.assertEqual(stopped.exception.code, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "unsupported")

    def test_text_report_lists_the_refusal(self):
        text = doctor.render(self.report(self.TREE_ARGV, VERIFIED, capability=B200))
        self.assertIn("status: unsupported", text)
        self.assertIn("GPU: SM100", text)
        self.assertIn("error: ax-engine-sglang: AX_QUALITY_PROFILE=verified refused", text)


if __name__ == "__main__":
    unittest.main()
