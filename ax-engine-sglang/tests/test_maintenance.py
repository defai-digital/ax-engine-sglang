"""CPU regression checks for startup failures, route policy and worker isolation."""
from contextvars import Context
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.compat import installation, patch_attribute, patch_item, check_sglang
from ax_engine_sglang.config import PluginConfig, get_config
from ax_engine_sglang.policy import RoundFeatures, chain_width, use_tree
from ax_engine_sglang.runtime import activate_worker, current_runtime


class PredecessorTests(unittest.TestCase):
    def test_chain_predecessors_are_only_saved_for_the_ngram_tree(self):
        fake_torch = SimpleNamespace(int64=object())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            from ax_engine_sglang import ax_chain

        worker = SimpleNamespace()
        state = SimpleNamespace(kind="chain")
        req = SimpleNamespace()
        batch = SimpleNamespace(reqs=[req])
        candidates, accepted = MagicMock(), MagicMock()
        with (patch.object(ax_chain, "current_runtime", return_value=state),
              patch.object(ax_chain, "torch", fake_torch),
              patch.object(ax_chain, "_TREE", True), patch.object(ax_chain, "_DUMP", "oracle")):
            with patch.object(ax_chain, "_CONFIG", PluginConfig(lattice=True)), patch.object(ax_chain, "_dump") as dump:
                ax_chain.record_accept(worker, batch, candidates, accepted)
                candidates.gather.assert_not_called()
                accepted.to.assert_not_called()
                self.assertIsNone(ax_chain.pre1_for(req))
                dump.assert_called_once_with(batch, "chain", candidates, accepted)

            with patch.object(ax_chain, "_CONFIG", PluginConfig(lattice=False)), patch.object(ax_chain, "_dump"):
                ax_chain.record_accept(worker, batch, candidates, accepted)
                self.assertIsNotNone(ax_chain.pre1_for(req))
                candidates.gather.assert_called_once()

    def test_lattice_builder_can_replay_without_an_ngram_predecessor(self):
        with patch.dict(sys.modules, {"torch": ModuleType("torch")}):
            from ax_engine_sglang import ax_lattice

        builder = ax_lattice.LatticeBuilder.__new__(ax_lattice.LatticeBuilder)
        builder.sampler = SimpleNamespace(fc1_cand=["candidates"], fc1_lat=["scores"], fc1_tail=["tail"])
        builder.cand, builder.lat, builder.tail, builder.root, builder.graph = (MagicMock() for _ in range(5))
        builder.tokens = MagicMock()
        builder.depth, builder.parent, builder.anc, builder.nxt, builder.sib = range(5)
        with patch.object(ax_lattice, "_DEBUG", 0), patch.object(ax_lattice, "_DUPCHECK", False):
            tree = builder(None, "anchor", None)
        builder.root.copy_.assert_called_once_with("anchor")
        builder.graph.replay.assert_called_once_with()
        self.assertEqual(tree[1:], tuple(range(5)))


class ConfigTests(unittest.TestCase):
    def test_defaults_and_ring_only(self):
        self.assertFalse(PluginConfig.from_env({}).tree_requested)
        self.assertFalse(PluginConfig.from_env({"AX_QSA_RING": "16"}).tree_requested)
        self.assertTrue(PluginConfig.from_env({"FC1_TREE64": "1"}).tree_requested)
        stable = PluginConfig.from_env({"AX_QSA_STABLE_TOPK": "1"})
        self.assertTrue(stable.qsa_stable_topk)
        self.assertFalse(stable.tree_requested)
        joint = PluginConfig.from_env({"AX_EAGLE_JOINT_SAMPLING": "1"})
        self.assertTrue(joint.eagle_joint_sampling)
        self.assertFalse(joint.tree_requested)
        dflash = PluginConfig.from_env({"AX_DFLASH_JOINT_SAMPLING": "1"})
        self.assertTrue(dflash.dflash_joint_sampling)
        self.assertFalse(PluginConfig.from_env({}).dflash_joint_sampling)
        unpacked = PluginConfig.from_env({"AX_GDN_UNPACKED_DECODE": "1"})
        self.assertTrue(unpacked.gdn_unpacked_decode)
        self.assertFalse(unpacked.tree_requested)

    def test_multiuser_configuration(self):
        config = PluginConfig.from_env({
            "FC1_TREE64": "1", "FC1_DRAFT_BLOCK": "16", "FC1_FUSED": "1",
            "AX_CHAIN_WIDTH": "8", "AX_CHAIN_GRAPH_WIDTHS": "16 8",
            "AX_CHAIN_DRAFT_WIDTHS": "8", "AX_CHAIN_ATTN": "trtllm_mha",
            "AX_SSM_POSITIONS": "16",
        })
        self.assertEqual(chain_width(config, 16, 0), 8)

    def test_invalid_inputs_fail_before_importing_runtime(self):
        invalid = (
            {"AX_QSA_RING": "6"}, {"AX_QSA_RING": "-4"}, {"AX_TREE": "yes"},
            {"FC1_DRAFT_BLOCK": "1"}, {"AX_CHAIN_WIDTH": "0"},
            {"AX_CHAIN_WIDTH": "999"}, {"AX_CHAIN_GRAPH_WIDTHS": "8 8"},
            {"AX_CHAIN_GRAPH_WIDTHS": "8"}, {"AX_CHAIN_DRAFT_WIDTHS": "8"},
            {"AX_CHAIN_ATTN": "triton"}, {"AX_CHAIN_SCHEDULE": "8x0"},
            {"AX_CHAIN_SCHEDULE": "8x10001"}, {"AX_CHAIN_SCHEDULE": "8xbad"},
            {"AX_SSM_CONV_SHARE": "nan"}, {"AX_SSM_CONV_SHARE": "1.5"},
            {"AX_SSM_POSITIONS": "8"},
            {"FC1_FUSED": "1", "AX_CHAIN_WIDTH": "16", "AX_SSM_POSITIONS": "8"},
            {"FC1_TREE64": "1", "FC1_DRAFT_BLOCK": "33"},
        )
        for env in invalid:
            with self.subTest(env=env), self.assertRaises(ValueError):
                PluginConfig.from_env(env)

    def test_schedule_precedence_and_round_index(self):
        config = PluginConfig.from_env({"AX_CHAIN_WIDTH": "4", "AX_CHAIN_SCHEDULE": "8x2,16x1"})
        self.assertEqual([chain_width(config, 24, i) for i in range(5)], [8, 8, 16, 8, 8])
        with self.assertRaises(ValueError):
            chain_width(config, 4, 0)

    def test_policy_capability_matrix(self):
        for sampled in (False, True):
            config = PluginConfig(tree_sampling=sampled)
            for requests in (0, 1, 2):
                for grammar in (False, True):
                    for logprobs in (False, True):
                        for custom in (False, True):
                            for greedy in (False, True):
                                for min_p in (False, True):
                                    features = RoundFeatures(requests, grammar, logprobs, custom, greedy, min_p)
                                    expected = (requests == 1 and not grammar and not logprobs and not custom
                                                and (greedy or (sampled and not min_p)))
                                    self.assertEqual(use_tree(config, features), expected)
        self.assertFalse(use_tree(PluginConfig(tree=False), RoundFeatures(1)))


class RuntimeTests(unittest.TestCase):
    def test_worker_state_does_not_leak(self):
        first, second = SimpleNamespace(), SimpleNamespace()
        state = activate_worker(first)
        state.kind, state.host_kv_len, state.fused_context = "tree", 128, object()
        state.round_info["width"] = 16
        state.stats["tree"] = 3
        activate_worker(second)
        self.assertIsNone(current_runtime().kind)
        self.assertIsNone(current_runtime().host_kv_len)
        self.assertIsNone(current_runtime().fused_context)
        self.assertIsNone(current_runtime().round_info["width"])
        self.assertEqual(current_runtime().stats["tree"], 0)
        self.assertIs(activate_worker(first), state)
        self.assertEqual(state.stats["tree"], 3)

    def test_independent_execution_contexts(self):
        left, right = Context(), Context()
        self.assertIsNot(left.run(current_runtime), right.run(current_runtime))

    def test_worker_adopts_buffers_captured_before_its_initialization(self):
        runner = SimpleNamespace()
        captured = activate_worker(runner)
        captured.fused_context = object()
        captured.host_kv_len = 77
        worker = SimpleNamespace(target_worker=SimpleNamespace(model_runner=runner))
        self.assertIs(activate_worker(worker), captured)
        self.assertIs(worker._ax_runtime, runner._ax_runtime)
        self.assertEqual(current_runtime().host_kv_len, 77)


class PatchTests(unittest.TestCase):
    def test_rollback_restores_inherited_methods_and_module_entries(self):
        class Base:
            def method(self):
                return "base"
        class Child(Base):
            pass
        entries = {"existing": 1}
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with installation():
                patch_attribute(Child, "method", lambda self: "patched")
                patch_attribute(Child, "marker", True)
                patch_item(entries, "existing", 2)
                with installation():
                    patch_item(entries, "new", 3)
                raise RuntimeError("injected failure")
        self.assertNotIn("method", vars(Child))
        self.assertFalse(hasattr(Child, "marker"))
        self.assertEqual(Child().method(), "base")
        self.assertEqual(entries, {"existing": 1})

    def test_success_keeps_patches(self):
        target = SimpleNamespace(value=1)
        with installation():
            patch_attribute(target, "value", 2)
        self.assertEqual(target.value, 2)

    def test_failed_assignment_preserves_original_error_and_restores_prior_patch(self):
        class ReadOnly:
            @property
            def value(self):
                return 1
        target = SimpleNamespace(value=1)
        with self.assertRaises(AttributeError):
            with installation():
                patch_attribute(target, "value", 2)
                patch_attribute(ReadOnly(), "value", 3)
        self.assertEqual(target.value, 1)

    def test_unsupported_version_requires_explicit_override(self):
        with patch.dict(sys.modules, {"sglang": SimpleNamespace(__version__="0.5.99")}):
            with self.assertRaisesRegex(RuntimeError, "untested SGLang"):
                check_sglang(PluginConfig())
            self.assertEqual(check_sglang(PluginConfig(allow_untested_sglang=True)), "0.5.99")


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        get_config.cache_clear()
        self.addCleanup(self.env.stop)
        self.addCleanup(get_config.cache_clear)

    def load_plugin(self):
        registry = {}
        class Algorithm:
            @staticmethod
            def register(name, **kwargs):
                def decorate(factory):
                    registry[name] = factory
                    return factory
                return decorate
        modules = {
            "sglang": SimpleNamespace(__version__="0.5.23"),
            "sglang.srt.speculative.spec_info": SimpleNamespace(SpeculativeAlgorithm=Algorithm),
            "sglang.srt.speculative.spec_registry": SimpleNamespace(CustomSpecAlgo=object),
        }
        context = patch.dict(sys.modules, modules)
        context.start()
        self.addCleanup(context.stop)
        name = "ax_engine_sglang._startup_test_plugin"
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "ax_engine_sglang/plugin.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(module)
        return module, registry

    def test_ring_only_does_not_import_or_install_tree_worker(self):
        os.environ["AX_QSA_RING"] = "16"
        plugin, registry = self.load_plugin()
        calls = []
        fixes = {name: SimpleNamespace(install=lambda n=name: calls.append(n))
                 for name in ("ax_rng", "ax_qsa")}
        with patch.dict(sys.modules, fixes), patch.object(plugin, "_install_tree_hooks") as tree, patch.object(sys, "path", list(sys.path)):
            plugin.register()
            plugin.register()
        self.assertEqual(calls, ["ax_rng", "ax_qsa"])
        tree.assert_not_called()
        self.assertIn("FC1_DFLASH_TREE", registry)
        self.assertFalse(plugin._tree_installed)

    def test_tree_initialization_failure_rolls_back_and_can_retry(self):
        os.environ["FC1_TREE64"] = "1"
        plugin, _ = self.load_plugin()
        target = SimpleNamespace(value="original")
        def fail(config):
            patch_attribute(target, "value", "patched")
            raise RuntimeError("missing upstream anchor")
        with patch.object(plugin, "_install_fixes"), patch.object(plugin, "_install_tree_hooks", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "missing upstream anchor"):
                plugin.register()
        self.assertEqual(target.value, "original")
        self.assertFalse(plugin._tree_installed)
        self.assertFalse(plugin._registered)
        with patch.object(plugin, "_install_fixes"), patch.object(plugin, "_install_tree_hooks"):
            plugin.register()
        self.assertTrue(plugin._tree_installed)

    def test_invalid_configuration_precedes_any_upstream_changes(self):
        os.environ["AX_SSM_POSITIONS"] = "8"
        plugin, _ = self.load_plugin()
        with patch.object(plugin, "_install_fixes") as fixes:
            with self.assertRaises(ValueError):
                plugin.register()
        fixes.assert_not_called()

    def test_registration_failure_rolls_back_installed_hooks(self):
        os.environ["FC1_TREE64"] = "1"
        plugin, _ = self.load_plugin()
        target = SimpleNamespace(value="original")

        def install(config):
            patch_attribute(target, "value", "patched")

        with patch.object(plugin, "_install_fixes"), patch.object(plugin, "_install_tree_hooks", install):
            with patch.object(plugin.SpeculativeAlgorithm, "register", side_effect=RuntimeError("registry error")):
                with self.assertRaisesRegex(RuntimeError, "registry error"):
                    plugin.register()
            self.assertEqual(target.value, "original")
            self.assertFalse(plugin._tree_installed)
            plugin.register()
        self.assertEqual(target.value, "patched")

    def test_capture_routes_only_tree_runners_to_tree_metadata(self):
        os.environ["FC1_BRANCH_PILOT"] = "1"
        plugin, _ = self.load_plugin()
        from ax_engine_sglang import runtime

        class Runner:
            def __init__(self, algorithm, draft=False):
                self.model_runner = SimpleNamespace(
                    is_draft_worker=draft, server_args=SimpleNamespace(speculative_algorithm=algorithm))
                self.buffers = SimpleNamespace(custom_mask=None)
                self.speculative_num_draft_tokens = 16
                self.capture_bs = self.compile_bs = [1, 2]

            def capture(self):
                return chain.is_tree(), self.get_spec_info(16), self.capture_bs

            def get_spec_info(self, num_tokens):
                return "stock"

            def can_run_graph(self, forward_batch):
                return True

        modules = {
            "torch": ModuleType("torch"),
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner": SimpleNamespace(DecodeCudaGraphRunner=Runner),
            "sglang.srt.speculative.dflash_info": SimpleNamespace(DFlashVerifyInput=type("Verify", (), {})),
            "sglang.srt.speculative.eagle_info": SimpleNamespace(EagleVerifyInput=lambda **kwargs: "tree"),
            "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(CaptureHiddenMode=SimpleNamespace(FULL=1)),
        }
        for name in ("ax_state_cap", "worker_gen", "ax_lattice"):
            modules[name] = SimpleNamespace(install=lambda: None)

        with patch.dict(sys.modules, modules):
            from ax_engine_sglang import ax_chain as chain
            with patch.dict(sys.modules, {"ax_chain": chain}), patch.object(chain, "install", chain._install_main_only):
                runtime.activate_worker(SimpleNamespace())
                plugin._install_tree_hooks(get_config())
                self.assertEqual(Runner("EAGLE").capture(), (False, "stock", [1, 2]))
                self.assertEqual(Runner("FC1_DFLASH_TREE").capture(), (True, "tree", [1]))
                current_runtime().kind = "tree"
                chain_runner = Runner("FC1_DFLASH_TREE")
                chain_runner._ax_chain_width = 8
                self.assertEqual(chain_runner.get_spec_info(8), "stock")
                self.assertEqual(Runner("FC1_DFLASH_TREE", draft=True).capture(), (False, "stock", [1, 2]))
                self.assertEqual(current_runtime().kind, "tree")
                current_runtime().kind = "chain"
                self.assertTrue(Runner("EAGLE").can_run_graph(None))
                self.assertFalse(Runner("FC1_DFLASH_TREE").can_run_graph(None))
                worker = SimpleNamespace(block_size=16)
                state = activate_worker(worker)
                state.kind, state.host_kv_len = "tree", 128
                chain.begin_round(worker)
                self.assertFalse(chain.is_tree())
                self.assertIsNone(state.host_kv_len)

    def test_kv_trace_can_retry_after_a_partial_install(self):
        from ax_engine_sglang import ax_nxt_debug

        class Draft:
            def prepare_for_decode(self, batch):
                pass

        original = Draft.prepare_for_decode
        modules = {"sglang.srt.speculative.dflash_info_v2": SimpleNamespace(DFlashDraftInputV2=Draft)}
        with patch.dict(sys.modules, modules), patch.object(ax_nxt_debug, "_installed", False):
            with self.assertRaises(AttributeError):
                ax_nxt_debug.install(SimpleNamespace())
            self.assertIs(Draft.prepare_for_decode, original)
            self.assertFalse(ax_nxt_debug._installed)

    def test_state_cap_import_failure_does_not_mark_installed(self):
        from ax_engine_sglang import ax_state_cap
        with patch.object(ax_state_cap, "POS", 8), patch.object(ax_state_cap, "_installed", False), patch.dict(sys.modules, {"sglang.srt.mem_cache.kv_cache_configurator": None}):
            for _ in range(2):
                with self.assertRaises(ImportError):
                    ax_state_cap.install()
                self.assertFalse(ax_state_cap._installed)

    def test_state_cap_failed_second_patch_leaves_first_untouched(self):
        from ax_engine_sglang import ax_state_cap
        class Pool:
            pass
        class Budget:
            pass
        modules = {
            "sglang.srt.mem_cache.memory_pool": SimpleNamespace(MambaPool=Pool),
            "sglang.srt.mem_cache.kv_cache_configurator": SimpleNamespace(KVCacheConfigurator=Budget),
        }
        original = Pool.__init__
        with patch.object(ax_state_cap, "POS", 8), patch.object(ax_state_cap, "_installed", False), patch.dict(sys.modules, modules), patch.object(ax_state_cap, "_rebuild", side_effect=[lambda self: None, RuntimeError("budget anchor")]):
            with self.assertRaisesRegex(RuntimeError, "budget anchor"):
                ax_state_cap.install()
            self.assertIs(Pool.__init__, original)
            self.assertFalse(ax_state_cap._installed)


if __name__ == "__main__":
    unittest.main()
