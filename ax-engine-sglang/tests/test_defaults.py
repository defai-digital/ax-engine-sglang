"""Launch defaults derived from the checkpoint and GPU (ax_engine_sglang/defaults.py), offline."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.defaults import LaunchConfigError, plan, resolve

# Excerpts of the published config.json files.
QWEN35_9B = {"architectures": ["Qwen3_5ForConditionalGeneration"],
             "text_config": {"model_type": "qwen3_5_text", "linear_num_value_heads": 32,
                             "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8}}
QWEN38_27B = {"architectures": ["Qwen3_5ForConditionalGeneration"],
              "text_config": {"model_type": "qwen3_5_text", "linear_num_value_heads": 48,
                              "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16}}
QWEN38_27B_NVFP4 = {**QWEN38_27B, "quantization_config": {"quant_method": "modelopt",
                                                          "quant_algo": "MIXED_PRECISION"}}
FLASH_NEXT = {"architectures": ["Qwen4ExpForConditionalGeneration"],
              "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
              "text_config": {"model_type": "qwen4_exp_text", "num_experts": 512,
                              "linear_num_value_heads": 48,
                              "layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12}}
DFLASH = {"architectures": ["DFlashDraftModel"], "dflash_config": {"block_size": 16}}
DFLASH2 = {"architectures": ["DFlash2DraftModel"], "dflash_config": {"block_size": 8}}


def args(algorithm="FC1_DFLASH_TREE", **values):
    base = dict(speculative_algorithm=algorithm, speculative_num_draft_tokens=16,
                moe_runner_backend="auto", speculative_moe_runner_backend=None,
                max_running_requests=None, cuda_graph_max_bs_decode=None)
    base.update(values)
    return SimpleNamespace(**base)


def env_of(result):
    return {name: value for name, (value, _) in result.env.items()}


class TreeDefaultsTests(unittest.TestCase):
    def test_27b_dflash2_gets_measured_tree_and_target_shape(self):
        result = plan(args(), {}, QWEN38_27B, DFLASH2)
        env = env_of(result)
        self.assertEqual((env["AX_GDN_LAYERS"], env["AX_GDN_HEADS"]), ("48", "48"))
        self.assertEqual(env["FC1_DRAFT_BLOCK"], "16")   # the launch flag, not the drafter's 8
        self.assertEqual(env["AX_LATTICE"], "1")
        for name, value in dict(FC1_TREE64="1", FC1_BRANCH_PILOT="1", FC1_OURS="1", FC1_FUSED="1",
                                AX_FAST="1", FC1_TREE_N="32", FC1_COMPILE_BUILD="1").items():
            self.assertEqual(env[name], value)
        self.assertEqual(result.args["max_running_requests"][0], 1)
        self.assertEqual(result.args["cuda_graph_max_bs_decode"][0], 1)
        self.assertNotIn("moe_runner_backend", result.args)   # dense target

    def test_lattice_tree_gets_the_measured_tree_temperature(self):
        result = plan(args(), {}, QWEN38_27B, DFLASH2)
        env = env_of(result)
        self.assertEqual(env["AX_TREE_TEMP"], "1.55")
        from ax_engine_sglang.config import PluginConfig
        self.assertEqual(PluginConfig.from_env(env).tree_temp, 1.55)   # the defaults pass validation
        for capability in ((9, 0), (10, 0), (12, 0)):   # a drafter property: the same on every GPU
            self.assertEqual(env_of(plan(args(), {}, QWEN38_27B, DFLASH2, capability=capability))["AX_TREE_TEMP"],
                             "1.55")
        self.assertNotIn("AX_TREE_TEMP", env_of(plan(args(), {"AX_TREE_TEMP": "1"}, QWEN38_27B, DFLASH2)))
        self.assertNotIn("AX_TREE_TEMP", env_of(plan(args(algorithm="DFLASH"), {}, QWEN38_27B, DFLASH2)))
        self.assertNotIn("AX_TREE_TEMP", env_of(plan(args(), {"AX_TREE": "0"}, QWEN38_27B, DFLASH2)))  # chain only
        with tempfile.NamedTemporaryFile(suffix=".npz") as table:   # DFlash (v1): n-gram trees, no lattice
            env = env_of(plan(args(), {"FC1_NGRAM_PATH": table.name}, QWEN35_9B, DFLASH))
        self.assertNotIn("AX_TREE_TEMP", env)

    def test_parity_tree_keeps_the_unshaped_lattice(self):
        environ = {"AX_NVFP4_VERIFY_PARITY": "1"}
        result = plan(args(attention_backend=None, fp4_gemm_runner_backend="auto", linear_attn_verify_backend=None),
                      environ, QWEN38_27B_NVFP4, DFLASH2)
        self.assertNotIn("AX_TREE_TEMP", env_of(result))

    def test_explicit_settings_win_and_are_not_reported_as_auto(self):
        environ = {"FC1_TREE_N": "64", "AX_FAST": "0", "AX_GDN_LAYERS": "48"}
        result = plan(args(max_running_requests=16), environ, QWEN38_27B, DFLASH2)
        env = env_of(result)
        self.assertNotIn("FC1_TREE_N", env)
        self.assertNotIn("AX_FAST", env)
        self.assertNotIn("AX_GDN_LAYERS", env)
        self.assertNotIn("max_running_requests", result.args)

    def test_contradicting_gdn_shape_or_block_stops_launch(self):
        with self.assertRaisesRegex(LaunchConfigError, "AX_GDN_LAYERS=24 but the target has 48"):
            plan(args(), {"AX_GDN_LAYERS": "24"}, QWEN38_27B, DFLASH2)
        with self.assertRaisesRegex(LaunchConfigError, "FC1_DRAFT_BLOCK=24 but the draft block is 16"):
            plan(args(), {"FC1_DRAFT_BLOCK": "24"}, QWEN38_27B, DFLASH2)

    def test_block_falls_back_to_the_drafter_config(self):
        result = plan(args(speculative_num_draft_tokens=None), {}, QWEN38_27B, DFLASH2)
        self.assertEqual(env_of(result)["FC1_DRAFT_BLOCK"], "8")

    def test_dflash_v1_needs_the_ngram_table(self):
        with self.assertRaisesRegex(LaunchConfigError, "ngram_counts.npz"):
            plan(args(), {"FC1_NGRAM_PATH": "/nonexistent/ngram.npz"}, QWEN35_9B, DFLASH)
        with self.assertRaisesRegex(LaunchConfigError, "needs a DFlash2 drafter"):
            plan(args(), {"AX_LATTICE": "1"}, QWEN35_9B, DFLASH)
        with tempfile.NamedTemporaryFile(suffix=".npz") as table:
            env = env_of(plan(args(), {"FC1_NGRAM_PATH": table.name}, QWEN35_9B, DFLASH))
        self.assertEqual((env["AX_GDN_LAYERS"], env["AX_GDN_HEADS"]), ("24", "32"))
        self.assertNotIn("AX_LATTICE", env)

    def test_tree_refuses_qwen4_exp(self):
        with self.assertRaisesRegex(LaunchConfigError, "does not support Qwen4-Exp"):
            plan(args(), {}, FLASH_NEXT, DFLASH2)

    def test_unreadable_target_needs_an_explicit_gdn_shape(self):
        with self.assertRaisesRegex(LaunchConfigError, "set AX_GDN_LAYERS and AX_GDN_HEADS"):
            plan(args(), {}, None, DFLASH2)
        with self.assertRaisesRegex(LaunchConfigError, "set AX_GDN_LAYERS and AX_GDN_HEADS"):
            plan(args(), {"AX_GDN_LAYERS": "48"}, None, DFLASH2)
        result = plan(args(), {"AX_GDN_LAYERS": "48", "AX_GDN_HEADS": "48"}, None, DFLASH2)
        self.assertNotIn("AX_GDN_LAYERS", env_of(result))   # the explicit shape is kept
        self.assertEqual(env_of(result)["AX_LATTICE"], "1")

    def test_unreadable_drafter_needs_an_explicit_lattice_choice(self):
        with self.assertRaisesRegex(LaunchConfigError, "could not tell the drafter kind"):
            plan(args(), {}, QWEN38_27B, None)
        self.assertNotIn("AX_LATTICE", env_of(plan(args(), {"AX_LATTICE": "1"}, QWEN38_27B, None)))

    def test_flag_block_differing_from_the_drafter_is_noted(self):
        result = plan(args(), {}, QWEN38_27B, DFLASH2)
        self.assertTrue(any("drafter config lists block_size 8" in n for n in result.notes))

    def test_missing_draft_block_stops_launch(self):
        bare = {"architectures": ["DFlash2DraftModel"]}   # no dflash_config.block_size
        with self.assertRaisesRegex(LaunchConfigError, "no draft block length"):
            plan(args(speculative_num_draft_tokens=None), {}, QWEN38_27B, bare)
        explicit = plan(args(speculative_num_draft_tokens=None), {"FC1_DRAFT_BLOCK": "16"}, QWEN38_27B, bare)
        self.assertNotIn("FC1_DRAFT_BLOCK", env_of(explicit))


class StockRouteDefaultsTests(unittest.TestCase):
    def test_gdn_chain_replay_resolves_with_auto_defaults_disabled(self):
        launch = args("DFLASH", speculative_num_draft_tokens=8, disable_radix_cache=True,
                      model_path="target", speculative_draft_model_path="draft")
        env = {"AX_GDN_CHAIN_REPLAY": "1", "AX_AUTO_DEFAULTS": "0"}
        configs = {("target", "config.json"): QWEN38_27B, ("draft", "config.json"): DFLASH2}
        declare = Mock()
        result = resolve(launch, environ=env, capability=(12, 0), declare=declare,
                         load=lambda path, name, revision: configs.get((path, name)))
        self.assertEqual(result.env, {})
        self.assertEqual(plan(launch, {"AX_GDN_CHAIN_REPLAY": "1"}, QWEN38_27B, DFLASH2,
                              capability=(12, 0)).args, result.args)
        declare.assert_called_once_with(
            launch, "ax-engine-sglang defaults", enable_linear_replayssm_spec=True,
            mamba_ssm_dtype="float32", linear_attn_verify_backend="triton",
        )
        self.assertEqual(plan(launch, {}, QWEN38_27B, DFLASH2, capability=(12, 0)).args, {})

    def test_gdn_chain_replay_accepts_the_experimental_width_sixteen(self):
        launch = args("DFLASH", speculative_num_draft_tokens=16, disable_radix_cache=True)
        result = plan(launch, {"AX_GDN_CHAIN_REPLAY": "1"}, QWEN38_27B, DFLASH2, capability=(12, 0))
        self.assertEqual(result.args["enable_linear_replayssm_spec"][0], True)

    def test_gdn_chain_replay_rejects_unqualified_launches(self):
        cases = (
            ({"speculative_algorithm": "FC1_DFLASH_TREE"}, "algorithm DFLASH"),
            ({"speculative_num_draft_tokens": 12}, "tokens 8"),
            ({"pp_size": 2}, "TP1, PP1 and DP1"),
            ({"disable_radix_cache": False}, "disable-radix-cache"),
            ({"mamba_ssm_dtype": "bfloat16"}, "mamba-ssm-dtype float32"),
            ({"linear_attn_verify_backend": "flashinfer"}, "linear-attn-verify-backend triton"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(LaunchConfigError, message):
                values = {"speculative_num_draft_tokens": 8, "disable_radix_cache": True, **overrides}
                algorithm = values.pop("speculative_algorithm", "DFLASH")
                plan(args(algorithm, **values), {"AX_GDN_CHAIN_REPLAY": "1"},
                     QWEN38_27B, DFLASH2, capability=(12, 0))

    def test_parity_tree_gets_the_measured_parity_settings(self):
        result = plan(args(attention_backend=None, fp4_gemm_runner_backend="auto", linear_attn_verify_backend=None),
                      {"AX_NVFP4_VERIFY_PARITY": "1"}, QWEN38_27B_NVFP4, DFLASH2)
        env = env_of(result)
        for name, value in (("FC1_FUSED", "0"), ("FC1_TREE_N", "16"), ("AX_GDN_UNPACKED_DECODE", "1"),
                            ("AX_GDN_RECURRENT_VERIFY", "1"), ("AX_DFLASH_JOINT_SAMPLING", "1"),
                            ("FC1_TREE64", "1")):
            self.assertEqual(env[name], value, name)
        self.assertEqual(result.args["attention_backend"][0], "flashinfer")
        self.assertEqual(result.args["fp4_gemm_runner_backend"][0], "flashinfer_cutlass")
        self.assertEqual(result.args["linear_attn_verify_backend"][0], "triton")
        for name in ("disable_radix_cache", "disable_flashinfer_autotune"):
            self.assertIs(result.args[name][0], True, name)
        self.assertNotIn("disable_prefill_cuda_graph", result.args)  # measured with matched outputs
        from ax_engine_sglang.config import PluginConfig
        PluginConfig.from_env({"AX_NVFP4_VERIFY_PARITY": "1", **env})  # the defaults pass validation

    def test_parity_ar_needs_no_tree_settings(self):
        result = plan(args(algorithm=None, attention_backend="flashinfer", fp4_gemm_runner_backend="flashinfer_cutlass"),
                      {"AX_NVFP4_VERIFY_PARITY": "1"}, QWEN38_27B_NVFP4, None)
        self.assertEqual(env_of(result), {"AX_GDN_UNPACKED_DECODE": "1"})
        self.assertEqual(sorted(result.args), ["disable_flashinfer_autotune", "disable_radix_cache"])
        explicit = plan(args(algorithm=None, attention_backend="flashinfer", fp4_gemm_runner_backend="flashinfer_cutlass",
                             disable_radix_cache=True, disable_flashinfer_autotune=True),
                        {"AX_NVFP4_VERIFY_PARITY": "1"}, QWEN38_27B_NVFP4, None)
        self.assertEqual(explicit.args, {})
        self.assertNotIn("linear_attn_verify_backend", result.args)

    def test_parity_refuses_stock_algorithms_and_other_backends(self):
        environ = {"AX_NVFP4_VERIFY_PARITY": "1"}
        with self.assertRaisesRegex(LaunchConfigError, "supports AR and --speculative-algorithm"):
            plan(args(algorithm="DFLASH"), environ, QWEN38_27B_NVFP4, DFLASH2)
        with self.assertRaisesRegex(LaunchConfigError, "needs --attention-backend flashinfer, got triton"):
            plan(args(attention_backend="triton"), environ, QWEN38_27B_NVFP4, DFLASH2)
        with self.assertRaisesRegex(LaunchConfigError, "needs --fp4-gemm-backend flashinfer_cutlass, got marlin"):
            plan(args(fp4_gemm_runner_backend="marlin"), environ, QWEN38_27B_NVFP4, DFLASH2)
        with self.assertRaisesRegex(LaunchConfigError, "needs --linear-attn-verify-backend triton"):
            plan(args(linear_attn_verify_backend="nv_cutedsl"), environ, QWEN38_27B_NVFP4, DFLASH2)
        plan(args(algorithm="DFLASH"), {}, QWEN38_27B_NVFP4, DFLASH2)  # without parity, stock is fine

    def test_explicit_parity_tree_settings_win(self):
        environ = {"AX_NVFP4_VERIFY_PARITY": "1", "FC1_TREE_N": "32"}
        self.assertNotIn("FC1_TREE_N", env_of(plan(args(), environ, QWEN38_27B_NVFP4, DFLASH2)))

    def test_flash_next_mtp_gets_the_qsa_ring(self):
        result = plan(args("EAGLE", speculative_num_draft_tokens=None), {}, FLASH_NEXT, None)
        self.assertEqual(env_of(result)["AX_QSA_RING"], "16")
        self.assertEqual(result.args, {})   # no GPU capability given: MoE backend untouched
        self.assertNotIn("AX_QSA_RING", env_of(plan(args("EAGLE"), {"AX_QSA_RING": "0"},
                                                    FLASH_NEXT, None)))
        self.assertNotIn("AX_QSA_RING", env_of(plan(args(None), {}, FLASH_NEXT, None)))

    def test_nvfp4_moe_uses_cutlass_on_sm110_and_sm120_only(self):
        for capability in ((11, 0), (12, 0)):
            result = plan(args("EAGLE"), {}, FLASH_NEXT, None, capability=capability)
            self.assertEqual(result.args["moe_runner_backend"][0], "flashinfer_cutlass")
            self.assertEqual(result.args["speculative_moe_runner_backend"][0], "flashinfer_cutlass")
        ar = plan(args(None), {}, FLASH_NEXT, None, capability=(11, 0))
        self.assertNotIn("speculative_moe_runner_backend", ar.args)
        self.assertEqual(plan(args("EAGLE"), {}, FLASH_NEXT, None, capability=(10, 0)).args, {})
        explicit = plan(args("EAGLE", moe_runner_backend="flashinfer_trtllm"), {}, FLASH_NEXT, None,
                        capability=(11, 0))
        self.assertNotIn("moe_runner_backend", explicit.args)
        dense = plan(args(None), {}, QWEN38_27B_NVFP4, None, capability=(11, 0))
        self.assertEqual(dense.args, {})
        # ModelOpt's hf_quant_config.json (no quant_method) next to a config.json that has one
        modelopt = {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4"}}
        both = plan(args(None), {}, FLASH_NEXT, None, modelopt, capability=(11, 0))
        self.assertEqual(both.args["moe_runner_backend"][0], "flashinfer_cutlass")
        bare = {**FLASH_NEXT, "quantization_config": None}
        only_file = plan(args(None), {}, bare, None, modelopt, capability=(12, 0))
        self.assertEqual(only_file.args["moe_runner_backend"][0], "flashinfer_cutlass")

    def test_unreadable_target_is_noted_for_stock_speculation(self):
        result = plan(args("EAGLE"), {}, None, None, capability=(11, 0))
        self.assertEqual((result.env, result.args), ({}, {}))
        self.assertTrue(any("could not read the target's config.json" in n for n in result.notes))
        self.assertEqual(plan(args(None), {}, None, None, capability=(11, 0)).notes, [])

    def test_opt_out(self):
        self.assertEqual(env_of(plan(args(), {"AX_AUTO_DEFAULTS": "0"}, QWEN38_27B, DFLASH2)), {})


class ResolveTests(unittest.TestCase):
    def test_resolve_uses_the_selected_cuda_device_for_moe_defaults(self):
        cutlass = {"moe_runner_backend": "flashinfer_cutlass",
                   "speculative_moe_runner_backend": "flashinfer_cutlass"}
        for device, gpu_id, capabilities, expected in (
            ("cuda", 1, [(10, 0), (12, 0)], cutlass),
            ("cuda", 1, [(12, 0), (10, 0)], {}),
            (None, None, [(11, 0)], cutlass),
        ):
            with self.subTest(gpu_id=gpu_id, capabilities=capabilities):
                probe = Mock(side_effect=lambda gpu_id=0: capabilities[gpu_id])
                cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=probe)
                declared = {}
                server_args = args("EAGLE", device=device, base_gpu_id=gpu_id, model_path="target")
                with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
                    resolve(server_args, environ={},
                            load=lambda path, name, revision: FLASH_NEXT if name == "config.json" else None,
                            declare=lambda a, source, **flags: declared.update(flags))
                self.assertEqual(declared, expected)
                probe.assert_called_once_with(0 if gpu_id is None else gpu_id)

    def test_resolve_skips_cuda_on_an_explicit_cpu_launch(self):
        cuda_available = Mock(return_value=True)
        probe = Mock(return_value=(12, 0))
        cuda = SimpleNamespace(is_available=cuda_available, get_device_capability=probe)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            result = resolve(args("EAGLE", device="cpu", model_path="target"), environ={},
                             load=lambda path, name, revision: FLASH_NEXT if name == "config.json" else None)
        self.assertEqual(result.args, {})
        cuda_available.assert_not_called()
        probe.assert_not_called()

    def test_resolve_writes_env_declares_args_once(self):
        configs = {("target", "config.json"): QWEN38_27B, ("draft", "config.json"): DFLASH2}
        declared, cleared, environ = [], [], {}
        server_args = args(model_path="target", speculative_draft_model_path="draft")
        for _ in range(2):   # both resolution hooks call resolve()
            resolve(server_args, environ=environ, load=lambda p, f, r=None: configs.get((p, f)),
                    capability=(11, 0), declare=lambda a, source, **f: declared.append(f),
                    on_env_change=lambda: cleared.append(1))
        self.assertEqual(environ["AX_GDN_LAYERS"], "48")
        self.assertEqual(declared, [{"max_running_requests": 1, "cuda_graph_max_bs_decode": 1}])
        self.assertEqual(cleared, [1])


if __name__ == "__main__":
    unittest.main()
