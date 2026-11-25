import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "sglang"))
import ax_fp8_verify as fp8


class CublasFp8GemmRunner:
    pass


class FP8VerifyBenchmarkTests(unittest.TestCase):
    def setUp(self):
        config = object()
        native = CublasFp8GemmRunner()
        self.candidate = SimpleNamespace(get_valid_tactics=lambda inputs, profile: [2])
        self.capturing = False
        self.activity = SimpleNamespace(paths=set())
        self.inputs = [SimpleNamespace(device="cuda:0"), None, None, None, None,
                       SimpleNamespace(numel=lambda: 4)]

        class Tuner:
            is_tuning_mode = False

            def choose_one(self, name, runners, tuning_config, inputs, tactic):
                return native, tactic

        tuner = Tuner()
        utils = SimpleNamespace(_raw_flashinfer_bmm_fp8=lambda inputs, tactic:
                                tuner.choose_one("fp8_gemm", [native], config, inputs, tactic=tactic))

        class Model:
            def forward(self, input_ids, positions, forward_batch):
                return utils._raw_flashinfer_bmm_fp8(input_ids, positions)

        test = self

        class Graph:
            ragged_verify_mode = False
            captured_req_width = 8
            model_runner = SimpleNamespace(is_draft_worker=False)

            def capture_one_shape(self, size, forward, stream_idx, variant_label, attention_variant):
                forward()
                test.capturing = True
                try:
                    return forward()
                finally:
                    test.capturing = False

            def _make_graph_key(self, *args):
                return args

            def execute(self, forward_batch):
                self._replay_graph_key = (8, None, None, forward_batch.variant)
                return "output"

        self.model = Model()
        self.graph = Graph()
        torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: self.capturing))
        projection = patch.object(fp8, "projection_for", return_value="gdn_qkvz")
        projection.start()
        self.addCleanup(projection.stop)
        fp8.install(Tuner, config, utils, Model, Graph, torch, 64, None,
                    {"gdn_qkvz": {"backend": "cublas", "native": [-1], "tactic": 2, "workspace_bytes": [4]}},
                    {"cublas": lambda: self.candidate}, self.activity)

    def batch(self, size=8, verify=True, variant=None):
        return SimpleNamespace(batch_size=size, variant=variant,
                               forward_mode=SimpleNamespace(is_target_verify=lambda: verify))

    def test_scope_and_workspace_fallback(self):
        self.assertIs(self.model.forward(self.inputs, -1, self.batch())[0], self.candidate)
        self.assertIsInstance(self.model.forward(self.inputs, -1, self.batch(verify=False))[0], CublasFp8GemmRunner)
        self.inputs[-1] = SimpleNamespace(numel=lambda: 8)
        self.assertIsInstance(self.model.forward(self.inputs, -1, self.batch())[0], CublasFp8GemmRunner)

    def test_reports_covered_graph_and_live_batch_only(self):
        def forward():
            self.model.forward(self.inputs, -1, self.batch())
            self.model.forward(self.inputs, -1, self.batch())

        self.graph.capture_one_shape(8, forward, attention_variant="window")
        self.graph.execute(forward_batch=self.batch())
        self.graph.execute(forward_batch=self.batch(7, variant="window"))
        self.assertEqual(self.activity.paths, set())
        self.assertEqual(self.graph.execute(forward_batch=self.batch(variant="window")), "output")
        self.assertEqual(self.activity.paths, {fp8.BENCH_PATH})

    def test_single_native_fallback_invalidates_graph_coverage(self):
        def forward():
            self.model.forward(self.inputs, -1, self.batch())
            return self.model.forward(self.inputs, 0, self.batch())

        self.assertIsInstance(self.graph.capture_one_shape(8, forward)[0], CublasFp8GemmRunner)
        self.graph.execute(self.batch())
        self.assertEqual(self.activity.paths, set())

    def test_profile_rejects_wrong_identity_and_rows(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            entry = {"backend": "cudnn", "tactic": [9, [[12, 228]]], "native": [-1],
                     "samples": 2, "workspace_bytes": [4]}
            path.write_text(json.dumps({"identity": {"gpu": "test"}, "rows": 64,
                                        "projections": {"attention_qkv": entry}}))
            for identity, rows in (({"gpu": "other"}, 64), ({"gpu": "test"}, 48)):
                with self.assertRaisesRegex(ValueError, "checkpoint, runtime and verification rows"):
                    fp8.read_profile(path, identity, rows)
            entries = fp8.read_profile(path, {"gpu": "test"}, 64)
            self.assertEqual(entries["attention_qkv"]["tactic"], (9, ((12, 228),)))
