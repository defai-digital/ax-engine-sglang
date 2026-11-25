"""AX_BTREE: the batched lattice tree, its round routing, graph capture and accept bookkeeping."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE), str(PACKAGE / "ax_engine_sglang")]

try:
    import torch
except ModuleNotFoundError:   # the CPU workflow may run without torch
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class BatchedTreeTests(unittest.TestCase):
    def setUp(self):
        import tree_ops
        from ax_engine_sglang import ax_btree
        from ax_engine_sglang.ax_lattice import lattice_logc
        self.tree_ops, self.btree, self.lattice_logc = tree_ops, ax_btree, lattice_logc

    def lattices(self, seed, batch, positions=15, top_k=16):
        rng = torch.Generator().manual_seed(seed)
        lat = torch.randn(batch, positions, top_k, top_k, generator=rng) * 3
        tail = torch.randn(batch, positions, generator=rng)
        cand = torch.stack([torch.stack([torch.randperm(5000, generator=rng)[:top_k] for _ in range(positions)])
                            for _ in range(batch)])
        return cand, lat, tail

    def single(self, cand, lat, tail, anchor, nodes, spine):
        """ax_lattice.LatticeBuilder.run for one request, without the CUDA graph."""
        c0, logc = self.lattice_logc(lat, tail)
        ss, ci, parent, _ = self.tree_ops.beam_tree(c0, logc, nodes, 99, spine, None, S=cand.shape[0])
        parent_full = torch.cat((torch.tensor([-1]), parent + 1))
        anc, depth, cidx = self.tree_ops.tree_struct(parent_full, 4)
        tokens = torch.cat((anchor.reshape(1), cand[ss, ci]))
        return tokens, parent_full, anc, depth, cidx

    def test_each_request_gets_its_batch_one_tree(self):
        for nodes, spine, batch in ((16, False, 4), (8, False, 3), (16, True, 2), (12, False, 8)):
            cand, lat, tail = self.lattices(nodes + batch, batch)
            anchors = torch.arange(batch) + 7
            logs = [self.lattice_logc(lat[b], tail[b]) for b in range(batch)]
            c0 = torch.stack([log[0] for log in logs])
            logc = torch.stack([log[1] for log in logs])
            tokens, parent, anc, depth, nxt, sib, cidx = self.btree.build_trees(
                cand, c0, logc, anchors, nodes, chain_first=spine)
            for b in range(batch):
                with self.subTest(nodes=nodes, spine=spine, request=b):
                    t1, p1, a1, d1, c1 = self.single(cand[b], lat[b], tail[b], anchors[b], nodes, spine)
                    self.assertTrue(torch.equal(cidx[b], c1))
                    self.assertTrue(torch.equal(tokens[b], t1))
                    self.assertTrue(torch.equal(parent[b], p1))
                    self.assertTrue(torch.equal(anc[b], a1))
                    self.assertTrue(torch.equal(depth[b], d1))
                    for node in range(nodes):
                        kids = [c for c in range(1, nodes) if int(p1[c]) == node]
                        self.assertEqual(int(nxt[b, node]), kids[0] if kids else -1)
                        later = [c for c in range(node + 1, nodes) if c > 0 and int(p1[c]) == int(p1[node])]
                        self.assertEqual(int(sib[b, node]), later[0] if node > 0 and later else -1)

    def test_mask_blocks_follow_each_prefix(self):
        cand, lat, tail = self.lattices(1, 2)
        logs = [self.lattice_logc(lat[b], tail[b]) for b in range(2)]
        _, _, anc, *_ = self.btree.build_trees(cand, torch.stack([x[0] for x in logs]),
                                               torch.stack([x[1] for x in logs]), torch.tensor([1, 2]), 8)
        mask = self.btree.tree_mask_reference(torch.tensor([3, 5]), anc)
        self.assertEqual(mask.numel(), 8 * (3 + 8) + 8 * (5 + 8))
        first = mask[:8 * 11].view(8, 11)
        self.assertTrue(first[:, :3].all())
        self.assertTrue(torch.equal(first[:, 3:], anc[0].bool()))
        second = mask[8 * 11:].view(8, 13)
        self.assertTrue(torch.equal(second[:, 5:], anc[1].bool()))


BTREE_ENV = {"AX_LATTICE": "1", "FC1_TREE64": "1", "FC1_BRANCH_PILOT": "1", "FC1_TREE_N": "16",
             "FC1_DRAFT_BLOCK": "16", "AX_BTREE": "4"}


class ConfigTests(unittest.TestCase):
    def test_btree_settings(self):
        from ax_engine_sglang.config import PluginConfig
        self.assertEqual(PluginConfig.from_env(BTREE_ENV).btree, 4)
        capped = PluginConfig.from_env({**BTREE_ENV, "FC1_FUSED": "1", "AX_SSM_POSITIONS": "8", "AX_CHAIN_WIDTH": "8"})
        self.assertEqual((capped.btree, capped.ssm_positions), (4, 8))   # fused rounds commit directly
        for change, message in (({"AX_BTREE": "9"}, "0 \\(off\\) or 2..8"), ({"AX_LATTICE": "0"}, "AX_LATTICE=1"),
                                ({"AX_SSM_POSITIONS": "8", "FC1_FUSED": "0", "AX_CHAIN_WIDTH": "8"}, "fused kernels"),
                                ({"AX_SSM_POSITIONS": "8", "FC1_FUSED": "1", "AX_BTREE_NATIVE": "1",
                                  "AX_CHAIN_WIDTH": "8"}, "fused kernels"),
                                ({"FC1_TREE_N": "32"}, "at most FC1_DRAFT_BLOCK")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                PluginConfig.from_env({**BTREE_ENV, **change})

    def test_policy(self):
        from ax_engine_sglang.config import PluginConfig
        from ax_engine_sglang.policy import RoundFeatures, use_btree, use_tree
        config = PluginConfig.from_env(BTREE_ENV)
        self.assertTrue(use_btree(config, RoundFeatures(requests=2)))
        self.assertTrue(use_btree(config, RoundFeatures(requests=4)))
        self.assertFalse(use_btree(config, RoundFeatures(requests=5)))
        self.assertFalse(use_btree(config, RoundFeatures(requests=1)))
        self.assertFalse(use_btree(config, RoundFeatures(requests=2, greedy=False)))
        self.assertFalse(use_btree(config, RoundFeatures(requests=2, logprobs=True)))
        self.assertFalse(use_tree(config, RoundFeatures(requests=2)))
        self.assertFalse(use_btree(PluginConfig.from_env({**BTREE_ENV, "AX_BTREE": "0"}), RoundFeatures(requests=2)))


@unittest.skipIf(torch is None, "needs torch")
class RoutingTests(unittest.TestCase):
    def setUp(self):
        from ax_engine_sglang import runtime
        from ax_engine_sglang.config import get_config
        self.enterContext(patch.dict(os.environ, BTREE_ENV, clear=True))
        get_config.cache_clear()
        self.addCleanup(get_config.cache_clear)
        spec = importlib.util.spec_from_file_location("ax_chain_btree_test", PACKAGE / "ax_engine_sglang/ax_chain.py")
        self.chain = importlib.util.module_from_spec(spec)
        with patch("atexit.register", lambda function: function):
            spec.loader.exec_module(self.chain)
        self.addCleanup(runtime._active.reset, runtime._active.set(None))

    def batch(self, requests, **kw):
        info = SimpleNamespace(has_custom_logit_processor=False, is_all_greedy=kw.pop("greedy", True),
                               need_min_p_sampling=False)
        values = dict(reqs=[SimpleNamespace(rid=str(i)) for i in range(requests)], has_grammar=False,
                      return_logprob=False, sampling_info=info)
        values.update(kw)
        return SimpleNamespace(**values)

    def test_round_kind(self):
        chain = self.chain
        worker = SimpleNamespace(target_worker=SimpleNamespace(model_runner=SimpleNamespace(attn_backend=None)))
        for requests, ready, kw, kind in ((1, True, {}, "tree"), (2, True, {}, "btree"), (4, True, {}, "btree"),
                                          (5, True, {}, "chain"), (2, False, {}, "chain"),
                                          (3, True, {"greedy": False}, "chain"),
                                          (2, True, {"has_grammar": True}, "chain")):
            with self.subTest(requests=requests, ready=ready, kw=kw):
                self.assertEqual(chain.choose(worker, self.batch(requests, **kw), draft_features_ready=ready), kind)
        stats = chain.current_runtime().stats
        self.assertEqual((stats["tree"], stats["btree"], stats["chain"]), (1, 2, 4))
        chain.current_runtime().kind = "btree"
        self.assertFalse(chain.is_tree())     # the fused one-request hooks decline a batched round

    def test_capture_kind_per_shape(self):
        chain = self.chain
        seen = []

        class Runner:
            def capture_one_shape(self, size, forward):
                seen.append((size, chain.current_runtime().kind))

        chain._install_shape_kind(Runner)
        runner = Runner()
        runner._ax_main_tree = True
        chain.current_runtime().kind = "tree"
        for size in (4, 2, 1):
            runner.capture_one_shape(size, None)
        other = Runner()
        other.capture_one_shape(3, None)          # a chain or drafter runner keeps the capture's kind
        self.assertEqual(seen, [(4, "btree"), (2, "btree"), (1, "tree"), (3, "tree")])
        self.assertEqual(chain.current_runtime().kind, "tree")

    def test_tree_accept_path_uses_each_request_nodes(self):
        chain = self.chain
        tokens = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        # flat node indices: request 1's nodes are 4..7; -1 pads a shorter path
        accept_index = torch.tensor([[0, 2, -1], [4, 5, 7]])
        recorded = []
        with patch.object(chain, "_DUMP", "x"), patch.object(chain, "_dump",
                                                              lambda batch, kind, path, accept, nodes: recorded.append(path)):
            chain.current_runtime().round_info["backend"] = "fi"
            chain.record_tree(self.batch(2), None, tokens, accept_index, torch.tensor([2, 3]))
        self.assertEqual(recorded[0].tolist(), [[10, 12, 10], [20, 21, 23]])


if __name__ == "__main__":
    unittest.main()
