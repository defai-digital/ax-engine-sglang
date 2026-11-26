"""AX_LATTICE_TRACE and the offline tree simulator: encoding, the trace writer and the tree rule."""
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
sys.path[:0] = [str(PACKAGE), str(ROOT / "benchmarks/sglang")]
from ax_engine_sglang import runtime   # noqa: E402
from ax_engine_sglang.config import PluginConfig, get_config   # noqa: E402
import simulate_lattice_tree as sim   # noqa: E402

try:
    import torch
except ModuleNotFoundError:   # the CPU workflow may run without torch
    torch = None


def random_lattice(rng, positions=6, top_k=5, spread=3.0):
    cand = tuple(tuple(rng.sample(range(100, 200), top_k)) for _ in range(positions))
    lat = tuple(tuple(tuple(rng.gauss(0.0, spread) for _ in range(top_k)) for _ in range(top_k))
                for _ in range(positions))
    tail = tuple(rng.gauss(0.0, 1.0) for _ in range(positions))
    return sim.Lattice(positions, top_k, cand, lat, tail)


def paths(nodes):
    """Each node as the tuple of candidate indices from the anchor."""
    out = []
    for node in range(1, len(nodes)):
        path, at = [], node
        while at > 0:
            path.append(nodes[at][2])
            at = nodes[at][0]
        out.append(tuple(reversed(path)))
    return sorted(out)


class SimulatorTests(unittest.TestCase):
    def test_chain_and_tree_accept_follow_the_continuation(self):
        rng = random.Random(1)
        lattice = random_lattice(rng)
        walk = [lattice.cand[s][c] for s, c in enumerate(sim.greedy_walk(lattice))]
        self.assertEqual(sim.chain_accept(lattice, 7, walk), 6)
        self.assertEqual(sim.chain_accept(lattice, 3, walk), 2)
        wrong = walk[:2] + [-1] + walk[3:]
        self.assertEqual(sim.chain_accept(lattice, 7, wrong), 2)
        # A runner-up at the first position: the chain accepts nothing, a tree holding it accepts one.
        second = sorted(range(lattice.top_k), key=lambda j: -lattice.lat[0][0][j])[1]
        truth = [lattice.cand[0][second], -1, -1, -1, -1, -1]
        self.assertEqual(sim.chain_accept(lattice, 7, truth), 0)
        nodes = sim.build_tree(lattice, 8)
        self.assertIn((second,), paths(nodes))
        self.assertEqual(sim.tree_accept(lattice, nodes, truth), 1)
        self.assertEqual(sim.oracle_accept(lattice, walk), 6)
        self.assertEqual(sim.oracle_accept(lattice, walk, rows=4), 3)

    def test_children_cap_and_spine(self):
        rng = random.Random(2)
        for _ in range(20):
            lattice = random_lattice(rng)
            capped = sim.build_tree(lattice, 12, children=2)
            counts = {}
            for parent, _, _ in capped[1:]:
                counts[parent] = counts.get(parent, 0) + 1
            self.assertLessEqual(max(counts.values()), 2)
            spine = sim.build_tree(lattice, 12, spine=True)
            walk = sim.greedy_walk(lattice)
            self.assertIn(tuple(walk), [p for p in paths(spine) if len(p) == len(walk)])
            self.assertEqual(len(spine), 12)
            self.assertEqual(len(set(paths(spine))), 11)

    @unittest.skipIf(torch is None, "needs torch")
    def test_best_first_selects_the_same_tree_as_beam_tree(self):
        sys.path.insert(0, str(PACKAGE / "ax_engine_sglang"))
        self.addCleanup(sys.path.remove, str(PACKAGE / "ax_engine_sglang"))
        import tree_ops
        from ax_engine_sglang.ax_lattice import lattice_logc

        rng = random.Random(3)
        for trial in range(40):
            lattice = random_lattice(rng, positions=7, top_k=6)
            lat = torch.tensor(lattice.lat, dtype=torch.float32)
            tail = torch.tensor(lattice.tail, dtype=torch.float32)
            for rows, spine in ((8, False), (16, False), (12, True)):
                c0, logc = lattice_logc(lat, tail)
                ss, cand, parent, _ = tree_ops.beam_tree(c0, logc, rows, 99, spine, None, S=lattice.positions)
                expected = []
                for i in range(rows - 1):
                    path, at = [], i
                    while at >= 0:
                        path.append(int(cand[at]))
                        at = int(parent[at])
                    expected.append(tuple(reversed(path)))
                with self.subTest(trial=trial, rows=rows, spine=spine):
                    self.assertEqual(paths(sim.build_tree(lattice, rows, spine=spine)), sorted(expected))

    def test_an_all_masked_row_has_no_likely_child(self):
        inf = float("inf")
        self.assertEqual(sim.log_softmax((-inf, -inf)), [-inf, -inf])
        self.assertEqual(sim.log_softmax((-inf, -inf), -inf), [-inf, -inf])
        self.assertAlmostEqual(sum(__import__("math").exp(v) for v in sim.log_softmax((0.0, 1.0), 0.5)) < 1, True)

    @unittest.skipIf(torch is None, "needs torch")
    def test_shaped_search_selects_the_same_tree_as_beam_tree(self):
        """AX_TREE_TEMP / AX_TREE_BRANCH_PENALTY / AX_TREE_TAIL / AX_TREE_LMAX: the plugin's search and
        the simulator's `t=`, `b=`, `notail` and `l=` select the same nodes."""
        sys.path.insert(0, str(PACKAGE / "ax_engine_sglang"))
        self.addCleanup(sys.path.remove, str(PACKAGE / "ax_engine_sglang"))
        import tree_ops
        from ax_engine_sglang.ax_lattice import lattice_logc

        rng = random.Random(5)
        shapes = ((0.7, 0.0, True, 99), (1.6, 0.0, True, 99), (1.0, 1.5, True, 99), (1.0, 0.0, False, 99),
                  (1.0, 0.0, True, 2), (1.3, 0.8, False, 3), (1.0, 0.4, True, 1))
        for trial in range(25):
            lattice = random_lattice(rng, positions=7, top_k=6)
            lat = torch.tensor(lattice.lat, dtype=torch.float32)
            tail = torch.tensor(lattice.tail, dtype=torch.float32)
            for (temp, penalty, use_tail, lmax), rows in ((shape, rows) for shape in shapes for rows in (8, 16)):
                c0, logc = lattice_logc(lat, tail, temp, penalty, use_tail)
                ss, cand, parent, _ = tree_ops.beam_tree(c0, logc, rows, lmax, False, None, S=lattice.positions)
                expected = []
                for i in range(rows - 1):
                    path, at = [], i
                    while at >= 0:
                        path.append(int(cand[at]))
                        at = int(parent[at])
                    expected.append(tuple(reversed(path)))
                got = sim.build_tree(lattice, rows, tail=use_tail, temp=temp, penalty=penalty,
                                     lmax=None if lmax == 99 else lmax)
                # With a small lmax the reachable set can be smaller than the rows; beam_tree then
                # pads with impossible nodes, which the simulator does not create.
                with self.subTest(trial=trial, shape=(temp, penalty, use_tail, lmax), rows=rows):
                    self.assertTrue(set(paths(got)) <= set(expected))
                    if len(got) == rows:
                        self.assertEqual(paths(got), sorted(expected))

    @unittest.skipIf(torch is None, "needs torch")
    def test_the_batched_search_uses_the_same_shape(self):
        sys.path.insert(0, str(PACKAGE / "ax_engine_sglang"))
        self.addCleanup(sys.path.remove, str(PACKAGE / "ax_engine_sglang"))
        from ax_engine_sglang.ax_btree import lattice_logc_batched
        from ax_engine_sglang.ax_lattice import lattice_logc

        lat, tail = torch.randn(3, 5, 6, 6), torch.randn(3, 5)
        c0, logc = lattice_logc_batched(lat, tail, 1.4, 0.6, False)
        for b in range(3):
            one = lattice_logc(lat[b], tail[b], 1.4, 0.6, False)
            self.assertTrue(torch.allclose(c0[b], one[0]) and torch.allclose(logc[b], one[1]))

    def test_policy_names(self):
        self.assertEqual(sim.parse_policy("tree:8:4:notail"),
                         {"name": "tree:8:4:notail", "kind": "tree", "rows": 8, "children": 4, "tail": False,
                          "spine": False, "temp": 1.0, "penalty": 0.0, "lmax": None})
        self.assertTrue(sim.parse_policy("tree:16:spine")["spine"])
        shaped = sim.parse_policy("tree:16:t=1.25:b=0.5:l=3")
        self.assertEqual((shaped["temp"], shaped["penalty"], shaped["lmax"]), (1.25, 0.5, 3))
        self.assertEqual(sim.parse_policy("oracle:8")["rows"], 8)
        for bad in ("tree:1", "chain:x", "tree:8:4:4", "oracle:1", "tree:8:notail:notail", "beam:8",
                    "tree:8:t=0", "tree:8:b=-1", "tree:8:l=0", "tree:8:x=1", "tree:8:t=1:t=2"):
            with self.subTest(policy=bad), self.assertRaises(Exception):
                sim.parse_policy(bad)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


@unittest.skipIf(torch is None, "needs torch")
class TraceWriterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "trace.jsonl"
        env = {"AX_LATTICE": "1", "AX_LATTICE_TRACE": str(self.path), "FC1_DRAFT_BLOCK": "16",
               "AX_TREE": "0", "AX_CHAIN_WIDTH": "16"}
        self.enterContext(patch.dict(os.environ, env, clear=True))
        get_config.cache_clear()
        self.addCleanup(get_config.cache_clear)
        spec = importlib.util.spec_from_file_location("ax_chain_trace_test", PACKAGE / "ax_engine_sglang/ax_chain.py")
        self.chain = importlib.util.module_from_spec(spec)
        with patch("atexit.register", lambda function: function):
            spec.loader.exec_module(self.chain)
        sys.path.insert(0, str(PACKAGE / "ax_engine_sglang"))
        self.addCleanup(sys.path.remove, str(PACKAGE / "ax_engine_sglang"))
        self.addCleanup(runtime._active.reset, runtime._active.set(None))

    def sampler(self, rng, bs, positions=15, top_k=16):
        lat = torch.randn(bs, positions, top_k, top_k, generator=rng)
        cand = torch.stack([torch.stack([torch.randperm(1000, generator=rng)[:top_k] for _ in range(positions)])
                            for _ in range(bs)])
        return SimpleNamespace(fc1_cand=cand, fc1_lat=lat, fc1_tail=torch.randn(bs, positions, generator=rng))

    def chain_round(self, worker, rids, tokens, accept, graph=True):
        state = self.chain.current_runtime()
        state.kind = "chain"
        state.round_info.update(width=16, draft_graph=graph)
        batch = SimpleNamespace(reqs=[SimpleNamespace(rid=r) for r in rids])
        self.chain.record_accept(worker, batch, torch.tensor(tokens), torch.tensor(accept))

    def test_rounds_round_trip_through_the_simulator(self):
        rng = torch.Generator().manual_seed(0)
        sampler = self.sampler(rng, bs=3)
        worker = SimpleNamespace(_draft_sampler=sampler)
        lattices = [sim.decode_lattice(row) for row in __import__("ax_lattice").trace_rows(sampler, 2)]
        for i, lattice in enumerate(lattices):
            self.assertEqual(lattice.cand, tuple(tuple(r) for r in sampler.fc1_cand[i].tolist()))
            self.assertAlmostEqual(lattice.lat[3][2][5], float(sampler.fc1_lat[i, 3, 2, 5]), places=6)
            self.assertAlmostEqual(lattice.tail[7], float(sampler.fc1_tail[i, 7]), places=6)
        # Two requests drafted the greedy walk; "a" accepts 3 drafts then 2 in a round drafted eager
        # (no fresh lattice), "b" 0, then 0 eager, then 1 alone (batch row 0's lattice).
        walks = [[lattices[i].cand[s][c] for s, c in enumerate(sim.greedy_walk(lattices[i]))] for i in range(2)]
        first = [[7, *walks[0]], [9, *walks[1]]]
        self.chain_round(worker, ["a", "b"], first, [3, 0])
        self.chain_round(worker, ["a", "b"], [[50] + [1] * 15, [51] + [2] * 15], [2, 0], graph=False)
        self.chain_round(worker, ["b"], [[52, *walks[0]]], [1])
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual([(r["rid"], r["accept"], "lat" in r) for r in rows],
                         [("a", 3, True), ("b", 0, True), ("a", 2, False), ("b", 0, False), ("b", 1, True)])
        self.assertEqual(self.chain.current_runtime().stats["trace_lattice"], 2)
        self.assertEqual(self.chain.current_runtime().stats["trace_without_lattice"], 1)
        requests = sim.load_requests([self.path])
        a = requests[(str(self.path), "a")]
        self.assertEqual(a[0]["continuation"], walks[0][:3] + [50, 1, 1])
        self.assertEqual(a[1]["offset"], 4)
        b = requests[(str(self.path), "b")]
        self.assertEqual(b[0]["continuation"], [51, 52, walks[0][0]])
        summary = sim.simulate(requests, [sim.parse_policy("chain:16")], include_cut=True)
        self.assertEqual(summary["checks"]["chain_rounds_checked"], 3)
        self.assertEqual(summary["checks"]["chain_draft_mismatch"], 0)
        self.assertEqual(summary["checks"]["recorded_accept_mismatch"], 0)
        self.assertEqual(summary["checks"]["without_lattice"], 2)
        cut = sim.simulate(requests, [sim.parse_policy("chain:16")])
        self.assertEqual(cut["checks"]["cut"], 3)

    def test_every_batch_size_writes_the_same_lattice_buffers(self):
        ax_lattice = __import__("ax_lattice")
        rng = torch.Generator().manual_seed(1)
        sampler = SimpleNamespace(candidate_out=torch.zeros(4, 15, 16, dtype=torch.int64))
        calls = []
        for bs in (4, 1, 2):           # captured largest first, then a smaller graph replays
            source = self.sampler(rng, bs)
            ax_lattice._store_lattice(sampler, source.fc1_cand, source.fc1_lat, source.fc1_tail)
            calls.append(source)
        self.assertEqual(sampler.fc1_cand.shape[0], 4)
        self.assertTrue(torch.equal(sampler.fc1_cand[:2], calls[2].fc1_cand))
        self.assertTrue(torch.equal(sampler.fc1_lat[2:], calls[0].fc1_lat[2:]))
        rows = ax_lattice.trace_rows(sampler, 2)
        self.assertEqual(sim.decode_lattice(rows[1]).cand, tuple(tuple(r) for r in calls[2].fc1_cand[1].tolist()))

    def test_a_sampler_without_a_lattice_is_written_without_one(self):
        worker = SimpleNamespace(_draft_sampler=SimpleNamespace())
        self.chain_round(worker, ["a"], [[5] + [1] * 15], [0])
        row = json.loads(self.path.read_text())
        self.assertNotIn("cand", row)
        self.assertEqual(row["tokens"][:1], [5])


class ChainOnlyTests(unittest.TestCase):
    def test_chain_only_servers_do_not_read_the_lattice(self):
        cases = (({"AX_LATTICE": "1", "AX_TREE": "0"}, False),
                 ({"AX_LATTICE": "1"}, True),                                    # tree rounds read it
                 ({"AX_LATTICE": "1", "AX_TREE": "0", "AX_LATTICE_TRACE": "/tmp/x"}, True),  # the trace too
                 ({"AX_TREE": "1"}, False))                                      # no lattice at all
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertIs(PluginConfig.from_env(env).lattice_read, expected)

    @unittest.skipIf(torch is None, "needs torch")
    def test_install_leaves_the_sampler_alone_on_a_chain_only_server(self):
        with patch.object(sys, "path", [str(PACKAGE / "ax_engine_sglang"), *sys.path]):
            ax_lattice = __import__("ax_lattice")
        config = PluginConfig.from_env({"AX_LATTICE": "1", "AX_TREE": "0"})
        with patch.dict(os.environ, {"AX_LATTICE": "1"}), patch.object(ax_lattice, "_installed", False), \
                patch("ax_engine_sglang.config.get_config", return_value=config), \
                patch.dict(sys.modules, {"fc1_dflash_worker": None}):   # importing it would raise
            ax_lattice.install()
            self.assertFalse(ax_lattice._installed)


class ConfigTests(unittest.TestCase):
    def test_the_trace_needs_the_lattice(self):
        with self.assertRaisesRegex(ValueError, "AX_LATTICE_TRACE"):
            PluginConfig.from_env({"AX_LATTICE_TRACE": "/tmp/x"})
        self.assertEqual(PluginConfig.from_env({"AX_LATTICE_TRACE": "/tmp/x", "AX_LATTICE": "1"}).lattice_trace,
                         "/tmp/x")

    def test_tree_shape_settings(self):
        config = PluginConfig.from_env({"AX_LATTICE": "1", "AX_TREE_TEMP": "1.3", "AX_TREE_BRANCH_PENALTY": "0.5",
                                        "AX_TREE_TAIL": "0", "AX_TREE_LMAX": "4"})
        self.assertEqual((config.tree_temp, config.tree_branch_penalty, config.tree_tail, config.tree_lmax),
                         (1.3, 0.5, False, 4))
        default = PluginConfig.from_env({})
        self.assertEqual((default.tree_temp, default.tree_branch_penalty, default.tree_tail, default.tree_lmax),
                         (1.0, 0.0, True, 99))
        with self.assertRaisesRegex(ValueError, "need AX_LATTICE"):
            PluginConfig.from_env({"AX_TREE_TEMP": "1.3"})
        for env in ({"AX_TREE_TEMP": "0"}, {"AX_TREE_BRANCH_PENALTY": "-1"}, {"AX_TREE_LMAX": "0"},
                    {"AX_TREE_TEMP": "nan"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                PluginConfig.from_env({"AX_LATTICE": "1", **env})


if __name__ == "__main__":
    unittest.main()
