"""Check observational hook behavior on CUDA without model checkpoints."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "nvfp4_forward_probe", Path(__file__).resolve().parents[2] / "scripts/nvfp4_forward_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ForwardProbeGpuTests(unittest.TestCase):
    def test_comparison_requires_all_labeled_prompts_and_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                lane = root / name
                (lane / "passes").mkdir(parents=True)
                (lane / "requests.json").write_text(json.dumps({
                    "p0": {"input_ids": [0, 2], "output_ids": [1]}}))
                torch.save({"label": "p0", "mode": "EXTEND", "input_ids": torch.tensor([0, 2]),
                            "positions": torch.tensor([0, 1]), "logits": torch.tensor([[0., 2., 1.]])},
                           lane / "passes/0.pt")
            result = probe.compare(root / "a", root / "b")
            self.assertEqual((result["matched_prefixes"], result["argmax_mismatches"]), (1, 0))
            self.assertEqual(result["records"][0]["softmax_total_variation"], 0.0)
            requests = {"p0": {"input_ids": [1, 2], "output_ids": [1]}}
            (root / "b/requests.json").write_text(json.dumps(requests))
            record = torch.load(root / "b/passes/0.pt", weights_only=True)
            record["input_ids"][0] = 1
            torch.save(record, root / "b/passes/0.pt")
            with self.assertRaisesRegex(ValueError, "same labeled prompts"):
                probe.compare(root / "a", root / "b")
            requests["missing"] = requests["p0"]
            (root / "b/requests.json").write_text(json.dumps(requests))
            with self.assertRaisesRegex(ValueError, "first prediction"):
                probe.load_run(root / "b")

    def test_logits_hook_preserves_output_and_skips_inactive_and_graph_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            callback = probe.logits_hook({"run_dir": directory})
            ids = torch.tensor([3], device="cuda")
            batch = SimpleNamespace(batch_size=1, positions=torch.tensor([2], device="cuda"),
                                    forward_mode=SimpleNamespace(name="DECODE"))
            output = SimpleNamespace(next_token_logits=torch.randn(1, 16, device="cuda"))
            original = output.next_token_logits.clone()
            inputs = (ids, None, None, batch)
            callback(None, inputs, output)
            self.assertEqual(list((root / "passes").glob("*.pt")), [])
            (root / "active.json").write_text(json.dumps({"label": "p0"}))
            self.assertIsNone(callback(None, inputs, output))
            paths = list((root / "passes").glob("*.pt"))
            self.assertEqual(len(paths), 1)
            saved = torch.load(paths[0], weights_only=True)
            self.assertTrue(torch.equal(saved["logits"], original.cpu()))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = output.next_token_logits + 0
                callback(None, inputs, output)
            graph.replay()
            self.assertTrue(torch.equal(graph_output, original))
            self.assertEqual(len(list((root / "passes").glob("*.pt"))), 1)
            self.assertTrue(torch.equal(output.next_token_logits, original))

    def test_linear_hook_preserves_output_and_respects_trial_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "active.json").write_text(json.dumps({"label": "p0"}))
            layer = torch.nn.Linear(16, 32, device="cuda", dtype=torch.bfloat16)
            layer.quant_method = SimpleNamespace()
            values = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
            expected = layer(values).detach().clone()
            handle = layer.register_forward_hook(probe.linear_hook({
                "run_dir": directory, "name": "control", "trials": 1}))
            with torch.inference_mode():
                self.assertTrue(torch.equal(layer(values), expected))
                self.assertTrue(torch.equal(layer(values), expected))
            handle.remove()
            files = list((root / "linear").glob("*.json"))
            self.assertEqual(len(files), 1)
            result = json.loads(files[0].read_text())
            self.assertTrue(result["repeat_equal"])
            self.assertTrue(result["original_output_preserved"])
            self.assertEqual([row["rows"] for row in result["measurements"]], [1, 2, 4, 8, 16, 32, 64, 128])

    def test_logits_hooks_skip_draft_prefill_and_keep_separate_capture_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "active.json").write_text(json.dumps({"label": "p0"}))
            target = probe.logits_hook({"run_dir": directory})
            other = probe.logits_hook({"run_dir": directory})
            batch = SimpleNamespace(batch_size=1, positions=torch.tensor([0], device="cuda"),
                                    forward_mode=SimpleNamespace(name="EXTEND"),
                                    spec_info=SimpleNamespace(is_draft_input=lambda: True))
            inputs = (torch.tensor([3], device="cuda"), None, None, batch)
            output = SimpleNamespace(next_token_logits=torch.ones(1, 8, device="cuda"))
            other(None, inputs, output)
            self.assertEqual(list((root / "passes").glob("*.pt")), [])
            batch.spec_info = None
            target(None, inputs, output)
            other(None, inputs, output)
            self.assertEqual(len(list((root / "passes").glob("*.pt"))), 2)

    def test_mlp_probe_preserves_the_live_module(self):
        class Mlp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = torch.nn.Linear(16, 32, device="cuda", dtype=torch.bfloat16)
                self.gate_up_proj.quant_method = SimpleNamespace()
                self.down = torch.nn.Linear(32, 16, device="cuda", dtype=torch.bfloat16)

            def forward(self, values):
                return self.down(torch.nn.functional.silu(self.gate_up_proj(values)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "active.json").write_text(json.dumps({"label": "p0"}))
            model = Mlp()
            names = set(vars(model))
            values = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                expected = model(values).clone()
                handle = model.register_forward_hook(probe.mlp_hook({
                    "run_dir": directory, "name": "mlp", "trials": 1}))
                actual = model(values)
            handle.remove()
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(names, set(vars(model)))
            self.assertTrue(json.loads((root / "linear/mlp-000.json").read_text())["repeat_equal"])

    def test_reused_output_storage_invalidates_the_probe(self):
        class Reused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant_method = SimpleNamespace()
                self.buffer = torch.empty(128, 16, device="cuda")

            def forward(self, values):
                self.buffer[:len(values)].copy_(values)
                return self.buffer[:len(values)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "active.json").write_text(json.dumps({"label": "p0"}))
            layer = Reused()
            handle = layer.register_forward_hook(probe.linear_hook({
                "run_dir": directory, "name": "reused", "trials": 1}))
            with self.assertRaisesRegex(RuntimeError, "overwrote the original output"):
                layer(torch.arange(128, device="cuda", dtype=torch.float32).reshape(8, 16))
            handle.remove()


if __name__ == "__main__":
    unittest.main()
