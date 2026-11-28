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
from ax_engine_sglang.qualification import EXACT_MEASURED, EXPERIMENTAL, UNSUPPORTED, lookup, target_key

QWEN38_27B = {"architectures": ["Qwen3_5ForConditionalGeneration"],
              "text_config": {"model_type": "qwen3_5_text", "linear_num_value_heads": 48,
                              "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16}}
QWEN38_27B_NVFP4 = {**QWEN38_27B, "quantization_config": {"quant_method": "modelopt",
                                                          "quant_algo": "MIXED_PRECISION"}}
DFLASH2 = {"architectures": ["DFlash2DraftModel"], "dflash_config": {"block_size": 8}}
TREE = "FC1_DFLASH_TREE"
VERIFIED = {"AX_QUALITY_PROFILE": "verified"}
CHAIN = {**VERIFIED, "AX_TREE": "0"}      # the speculative route the verified profile admits
RTX, B200 = (12, 0), (10, 0)


def chain_args(**values):
    return args(**{"speculative_num_draft_tokens": 8, **values})


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
            # Tree rounds differed from AR in 1,024-token outputs (RTX PRO 6000, 2026-11-21).
            self.assertEqual(row.status, EXPERIMENTAL if route == "tree" else EXACT_MEASURED)
            self.assertEqual(row.admits_verified, route != "tree")
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
        result = plan(chain_args(), CHAIN, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        self.assertEqual(result.env["AX_NVFP4_VERIFY_PARITY"], ("1", "AX_QUALITY_PROFILE=verified"))
        self.assertEqual(result.env["AX_GDN_RECURRENT_VERIFY"][0], "1")
        self.assertEqual(result.args["attention_backend"][0], "flashinfer")
        self.assertTrue(any("exact_measured" in note for note in result.notes))
        self.assertTrue(any("prefill chunk" in note for note in result.notes))

    def test_verified_covers_autoregressive_serving(self):
        result = plan(args(None), VERIFIED, QWEN38_27B_NVFP4, None, capability=RTX)
        self.assertEqual(result.env["AX_NVFP4_VERIFY_PARITY"][0], "1")
        self.assertFalse(any(name.startswith(("AX_CHAIN", "FC1_", "AX_SSM")) for name in result.env))

    def test_verified_refuses_unmeasured_launches(self):
        cases = (
            (dict(capability=B200), QWEN38_27B_NVFP4, {}, "SM100"),
            (dict(capability=None), QWEN38_27B_NVFP4, {}, "no CUDA GPU"),
            (dict(capability=RTX), QWEN38_27B, {}, "not a measured checkpoint"),
            (dict(capability=RTX), None, {}, "not a measured checkpoint"),
            (dict(capability=RTX), QWEN38_27B_NVFP4, {"AX_NVFP4_VERIFY_PARITY": "0"}, "contradicts"),
            (dict(capability=RTX), QWEN38_27B_NVFP4, {"AX_AUTO_DEFAULTS": "0"}, "AX_AUTO_DEFAULTS"),
            (dict(capability=RTX), QWEN38_27B_NVFP4, {"AX_GDN_CHAIN_REPLAY": "1"}, "AX_GDN_CHAIN_REPLAY"),
            (dict(capability=RTX), QWEN38_27B_NVFP4, {"AX_TREE": "1"}, "tree rounds .* Use the chain route: AX_TREE=0"),
        )
        for keywords, target, extra, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(LaunchConfigError, message):
                plan(chain_args(), {**CHAIN, **extra}, target, DFLASH2, **keywords)
        with self.assertRaisesRegex(LaunchConfigError, "Use the chain route"):     # the tree is the default route
            plan(args(), VERIFIED, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)

    def test_verified_keeps_the_measured_chain_sizes(self):
        chain = plan(chain_args(), CHAIN, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        self.assertEqual({name: chain.env[name][0] for name in ("AX_CHAIN_WIDTH", "FC1_TREE_N", "FC1_DRAFT_BLOCK")},
                         {"AX_CHAIN_WIDTH": "8", "FC1_TREE_N": "8", "FC1_DRAFT_BLOCK": "8"})
        self.assertNotIn("AX_CHAIN_GRAPH_WIDTHS", chain.env)
        # The settings the launcher derives make a valid final configuration.
        config = PluginConfig.from_env({**CHAIN, **{name: value for name, (value, _) in chain.env.items()}})
        self.assertEqual((config.chain_width, config.tree_nodes, config.tree), (8, 8, False))
        for block in (16, None):
            with self.subTest(block=block), self.assertRaisesRegex(LaunchConfigError, f"draft-tokens 8, got {block}"):
                plan(args(speculative_num_draft_tokens=block), CHAIN, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        for extra, message in (({"AX_CHAIN_WIDTH": "4"}, "AX_CHAIN_WIDTH=8"),
                               ({"FC1_TREE_N": "16"}, "FC1_TREE_N=8"),
                               ({"AX_CHAIN_GRAPH_WIDTHS": "8"}, "own graph"),
                               ({"AX_CHAIN_SCHEDULE": "8x2,4x2"}, "AX_CHAIN_SCHEDULE"),
                               ({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8"}, "unset AX_CHAIN_WIDTH_BY_BATCH"),
                               ({"AX_TREE_SAMPLING": "1"}, "greedy")):
            with self.subTest(extra=extra), self.assertRaisesRegex(LaunchConfigError, message):
                plan(chain_args(), {**CHAIN, **extra}, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        explicit = plan(chain_args(), {**CHAIN, "FC1_TREE_N": "8", "AX_CHAIN_WIDTH": "8"},
                        QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        self.assertNotIn("AX_CHAIN_WIDTH", explicit.env)

    def test_verified_prefills_each_prompt_in_one_pass_without_graphs(self):
        result = plan(chain_args(), CHAIN, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        self.assertEqual(result.args["chunked_prefill_size"][0], 4096)
        self.assertTrue(result.args["disable_prefill_cuda_graph"][0])
        given = plan(chain_args(chunked_prefill_size=8192, disable_prefill_cuda_graph=True), CHAIN,
                     QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        self.assertFalse({"chunked_prefill_size", "disable_prefill_cuda_graph"} & set(given.args))
        self.assertNotIn("chunked_prefill_size", plan(chain_args(chunked_prefill_size=-1), CHAIN, QWEN38_27B_NVFP4,
                                                      DFLASH2, capability=RTX).args)
        with self.assertRaisesRegex(LaunchConfigError, "--chunked-prefill-size 4096 or more, got 512"):
            plan(chain_args(chunked_prefill_size=512), CHAIN, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        parity = plan(args(), {"AX_NVFP4_VERIFY_PARITY": "1"}, QWEN38_27B_NVFP4, DFLASH2, capability=RTX)
        self.assertFalse({"chunked_prefill_size", "disable_prefill_cuda_graph"} & set(parity.args))

    def test_measured_settings_given_explicitly_pass_the_first_validation(self):
        # The plugin validates before plan() runs; the verified lane's other settings are filled in.
        for environ, width in (({"FC1_TREE_N": "8", "AX_CHAIN_WIDTH": "8"}, 8), ({}, None)):
            with self.subTest(environ=environ):
                config, provisional = startup_config({**CHAIN, **environ}, hooks_available=True)
                self.assertTrue(provisional)
                self.assertEqual(config.chain_width, width)

    def test_graph_settings_reach_the_record_sglang_already_built(self):
        # SGLang builds cuda_graph_config before the plugin's hook runs; a late flag alone is ignored.
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from ax_engine_sglang import defaults

        declared, hooks = {}, {}
        built = {"prefill": {"backend": "breakable"}, "decode": {"max_bs": 8}}
        fake = {name: ModuleType(name) for name in (
            "sglang", "sglang.srt", "sglang.srt.arg_groups", "sglang.srt.arg_groups.overrides",
            "sglang.srt.arg_groups.resolution_hooks", "sglang.srt.model_executor",
            "sglang.srt.model_executor.cuda_graph_config")}
        overrides = fake["sglang.srt.arg_groups.overrides"]
        overrides.declare_resolution = lambda server_args, source, **fields: declared.update(fields)
        overrides.resolving_view = lambda server_args: SimpleNamespace(cuda_graph_config=built)
        fake["sglang.srt.arg_groups.resolution_hooks"].register_resolution_hook = (
            lambda name: lambda hook: hooks.setdefault(name, hook))
        graph = fake["sglang.srt.model_executor.cuda_graph_config"]
        graph.Backend, graph.Phase = SimpleNamespace(DISABLED="disabled"), SimpleNamespace(PREFILL="prefill",
                                                                                         DECODE="decode")
        graph.with_phase = lambda config, phase, **changes: {**config, phase: {**config[phase], **changes}}
        result = defaults.Plan()
        result.args.update(disable_prefill_cuda_graph=(True, ""), cuda_graph_max_bs_decode=(1, ""))
        with patch.dict(sys.modules, fake), patch.object(defaults, "plan", return_value=result), \
                patch.object(defaults, "read_json", return_value=None), \
                patch.object(defaults, "device_capability", return_value=RTX), \
                patch.object(defaults, "describe", return_value=""), patch.dict("os.environ", {}, clear=False):
            self.assertTrue(defaults.install())
            hooks["handle_moe_kernel_config"](SimpleNamespace(), lambda server_args: None)
        self.assertEqual(declared["cuda_graph_config"],
                         {"prefill": {"backend": "disabled"}, "decode": {"max_bs": 1}})
        self.assertTrue(declared["disable_prefill_cuda_graph"])
        self.assertEqual(built["prefill"]["backend"], "breakable")   # SGLang's record is replaced, not edited

    def test_the_resolver_does_not_skip_a_verified_launch(self):
        from ax_engine_sglang.defaults import resolve
        with self.assertRaisesRegex(LaunchConfigError, "AX_AUTO_DEFAULTS"):
            resolve(args(), environ={**VERIFIED, "AX_AUTO_DEFAULTS": "0"}, capability=RTX,
                    load=lambda *_: QWEN38_27B_NVFP4)

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
        options = doctor.parse(argv)
        return doctor.diagnose(options, environ or {}, load=lambda path, name, revision=None: files.get((path, name)),
                               capability=capability, sglang=sglang)

    TREE_ARGV = ["target", "--speculative-algorithm", TREE, "--speculative-draft-model-path", "draft",
                 "--speculative-num-draft-tokens", "16"]

    CHAIN_ARGV = TREE_ARGV[:-1] + ["8"]

    def test_a_tree_launch_is_supported_without_an_exact_output_claim(self):
        report = self.report(self.TREE_ARGV)
        self.assertEqual((report["status"], report["errors"], report["route"]), ("supported", [], "tree"))
        self.assertEqual(report["derived"]["env"]["AX_GDN_LAYERS"]["value"], "48")
        self.assertEqual(report["quality"]["exact_output"]["status"], EXPERIMENTAL)
        self.assertFalse(report["quality"]["claims_exact_output"])
        chain = self.report(self.CHAIN_ARGV, CHAIN)
        self.assertEqual((chain["status"], chain["errors"], chain["route"]), ("supported", [], "chain"))
        self.assertEqual(chain["quality"]["exact_output"]["status"], EXACT_MEASURED)
        self.assertTrue(chain["quality"]["claims_exact_output"])
        refused = self.report(self.TREE_ARGV, VERIFIED)
        self.assertEqual(refused["status"], "unsupported")
        self.assertIn("Use the chain route", refused["errors"][0])
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
        table = self.report(self.TREE_ARGV + ["--max-running-requests", "8"],
                            {"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8", "AX_CHAIN_GRAPH_WIDTHS": "16 8"})
        self.assertEqual((table["status"], table["errors"]), ("experimental", []))
        self.assertEqual(table["experimental_features"], ["AX_CHAIN_WIDTH_BY_BATCH"])
        single = self.report(self.TREE_ARGV, {"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8", "AX_CHAIN_GRAPH_WIDTHS": "16 8"})
        self.assertTrue(any("no effect with one running request" in note for note in single["derived"]["notes"]), single)
        untested = self.report(self.TREE_ARGV, sglang="0.6.0")
        self.assertEqual(untested["status"], "unsupported")
        self.assertIn("AX_ALLOW_UNTESTED_SGLANG=1", untested["errors"][0])
        allowed = self.report(self.TREE_ARGV, {"AX_ALLOW_UNTESTED_SGLANG": "1"}, sglang="0.6.0")
        self.assertEqual((allowed["status"], allowed["errors"]), ("experimental", []))
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

    def test_b12x_down128_is_experimental_on_a_qualified_replay_launch(self):
        argv = ["target", "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", "draft",
                "--speculative-num-draft-tokens", "8", "--disable-radix-cache"]
        report = self.report(argv, {"AX_GDN_CHAIN_REPLAY": "1", "AX_NVFP4_B12X_DOWN128": "1"})
        self.assertEqual(report["status"], "experimental")
        self.assertEqual(report["experimental_features"], ["AX_GDN_CHAIN_REPLAY=1", "AX_NVFP4_B12X_DOWN128=1"])
        self.assertEqual(report["derived"]["args"]["fp4_gemm_runner_backend"]["value"], "flashinfer_cutedsl")

    def test_invalid_settings_are_reported_once(self):
        report = self.report(self.TREE_ARGV, {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16"})
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("at least two widths", report["errors"][0])
        both = self.report(self.TREE_ARGV + ["--speculative-adaptive"],
                           {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8"})
        self.assertIn("--speculative-adaptive", both["errors"][0])

    def test_settings_refused_when_the_plugin_loads_are_refused_here(self):
        # The plugin validates before SGLang resolves --speculative-num-draft-tokens (block 24 until then).
        argv = ["target", "--speculative-algorithm", TREE, "--speculative-draft-model-path", "draft",
                "--speculative-num-draft-tokens", "32"]
        report = self.report(argv, {"AX_CHAIN_WIDTH": "32"})
        self.assertEqual(report["status"], "unsupported")
        self.assertIn("chain width 32 outside 2..24", report["errors"][0])
        explicit = self.report(argv, {"AX_CHAIN_WIDTH": "32", "FC1_DRAFT_BLOCK": "32"})
        self.assertEqual((explicit["status"], explicit["errors"]), ("supported", []))

    def test_adaptive_launch_rules_match_the_plugin(self):
        environ = {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8"}
        head = ["target", "--speculative-algorithm", TREE, "--speculative-draft-model-path", "draft"]
        block16 = ["--speculative-num-draft-tokens", "16"]
        # SGLang 0.5.22 derives no FC1_DRAFT_BLOCK: the worker drafts --speculative-num-draft-tokens.
        for argv, version, message in ((head + block16 + ["--speculative-adaptive"], "0.5.23", "--speculative-adaptive"),
                                       (head + block16 + ["--tp-size", "2"], "0.5.23", "one GPU"),
                                       (head + ["--speculative-num-draft-tokens", "8"], "0.5.22",
                                        "exceeds the draft block 8")):
            with self.subTest(message=message):
                report = self.report(argv, environ, sglang=version)
                self.assertEqual(report["status"], "unsupported")
                self.assertIn(message, report["errors"][0])
        self.assertEqual(self.report(head + block16, environ, sglang="0.5.22")["errors"], [])

    def test_server_argument_rules(self):
        from ax_engine_sglang.config import server_args_error
        adaptive = PluginConfig.from_env({"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8"})
        tree64 = PluginConfig.from_env({"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8",
                                        "FC1_TREE64": "1", "FC1_DRAFT_BLOCK": "16"})
        self.assertIsNone(server_args_error(adaptive, SimpleNamespace(speculative_num_draft_tokens=16)))
        self.assertIsNone(server_args_error(adaptive, SimpleNamespace(tp_size=None)))
        self.assertIn("draft block 8", server_args_error(adaptive, SimpleNamespace(speculative_num_draft_tokens=8)))
        # With FC1_TREE64=1 the plugin replaces the flag with the tree size; the block is FC1_DRAFT_BLOCK.
        self.assertIsNone(server_args_error(tree64, SimpleNamespace(speculative_num_draft_tokens=8)))
        self.assertIn("one GPU", server_args_error(adaptive, SimpleNamespace(pp_size=2)))
        self.assertIsNone(server_args_error(PluginConfig.from_env({}), SimpleNamespace(tp_size=2)))

    def test_the_ssm_cap_needs_the_radix_cache_off(self):
        environ = {"AX_SSM_POSITIONS": "8", "AX_CHAIN_WIDTH": "8"}
        report = self.report(self.TREE_ARGV, environ)
        self.assertEqual(report["errors"], ["AX_SSM_POSITIONS requires --disable-radix-cache"])
        self.assertEqual(self.report(self.TREE_ARGV + ["--disable-radix-cache"], environ)["errors"], [])

    def test_the_ssm_cap_sees_the_radix_cache_the_launcher_turns_off(self):
        environ = {"AX_NVFP4_VERIFY_PARITY": "1", "AX_SSM_POSITIONS": "8", "AX_CHAIN_WIDTH": "8"}
        report = self.report(self.TREE_ARGV, environ)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["derived"]["args"]["disable_radix_cache"]["value"])

    def test_backend_flags_reach_the_launch_rules(self):
        report = self.report(self.TREE_ARGV + ["--attention-backend", "triton"], {"AX_NVFP4_VERIFY_PARITY": "1"})
        self.assertEqual(report["status"], "unsupported")
        self.assertIn("--attention-backend flashinfer, got triton", report["errors"][0])

    def test_flags_outside_the_rules_are_named(self):
        report = self.report(self.TREE_ARGV + ["--port", "30001"])
        self.assertEqual((report["status"], report["unchecked_flags"]), ("supported", ["--port", "30001"]))
        self.assertIn("not checked (flags outside the launch rules): --port 30001", doctor.render(report))

    def test_the_model_comes_first_as_with_serve(self):
        late = self.report(["--port", "30001", "target", "--speculative-algorithm", TREE])
        self.assertIn("no target model", late["errors"][0])     # 30001 is --port's value, not the model
        self.assertEqual(late["unchecked_flags"], ["--port", "30001", "target"])
        named = self.report(["--port", "30001", "--model-path", "target", *self.TREE_ARGV[1:]])
        self.assertEqual((named["status"], named["target"]["path"]), ("supported", "target"))

    def test_every_server_argument_the_rules_read_is_a_flag(self):
        import re
        source = (Path(__file__).resolve().parents[1] / "ax_engine_sglang/defaults.py").read_text()
        read = set(re.findall(r'getattr\((?:args|server_args), "([a-z_0-9]+)"', source)) - {"_ax_defaults_resolved"}
        self.assertEqual(read - {field for _, field, _, _ in doctor.SERVER_FLAGS}, set())

    def test_replay_rules_see_the_parallelism_flags(self):
        argv = ["target", "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", "draft",
                "--speculative-num-draft-tokens", "8", "--disable-radix-cache"]
        self.assertTrue(self.report(argv)["gdn_chain_replay"]["eligible"])
        for extra in (["--tp-size", "2"], ["--speculative-eagle-topk", "2"], ["--enable-page-major-kv-layout"]):
            with self.subTest(extra=extra):
                self.assertFalse(self.report(argv + extra)["gdn_chain_replay"]["eligible"])

    def test_only_the_verified_profile_claims_exact_output(self):
        cases = (
            (self.TREE_ARGV, {"AX_QUALITY_PROFILE": "fast"}, "supported", False),
            (self.TREE_ARGV, {}, "supported", False),
            (self.CHAIN_ARGV, CHAIN, "supported", True),
            (self.TREE_ARGV, VERIFIED, "unsupported", False),
        )
        for argv, environ, status, claimed in cases:
            with self.subTest(environ=environ, status=status):
                report = self.report(argv, environ)
                self.assertEqual(report["status"], status)
                self.assertEqual(report["quality"]["claims_exact_output"], claimed)
                self.assertEqual("not claimed" in doctor.render(report), not claimed)
        fast = self.report(self.TREE_ARGV, {"AX_QUALITY_PROFILE": "fast"})
        self.assertIn("fast profile: this launch makes no exact-output claim", fast["derived"]["notes"])

    def test_a_malformed_capability_is_a_usage_error(self):
        for value in ("x", "12"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as stopped:
                doctor.parse(["target", "--sm", value])
            self.assertEqual(stopped.exception.code, 2)
        self.assertEqual(doctor.parse(["target", "--sm", "12.0"]).sm, (12, 0))

    def test_sglang_0_5_22_derives_nothing(self):
        report = self.report(self.TREE_ARGV, VERIFIED, sglang="0.5.22")
        self.assertEqual(report["status"], "unsupported")
        self.assertIn("needs the launch defaults", report["errors"][0])
        plain = self.report(self.TREE_ARGV, sglang="0.5.22")
        self.assertEqual((plain["errors"], plain["derived"]["env"]), ([], {}))

    def test_a_launch_needs_a_target(self):
        report = self.report(["--speculative-algorithm", TREE, "--speculative-num-draft-tokens", "16"])
        self.assertEqual(report["status"], "unsupported")
        self.assertIn("no target model", report["errors"][0])

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
