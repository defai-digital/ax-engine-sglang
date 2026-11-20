"""CPU checks for benchmark provenance and server lifecycle handling."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / name) for name in ("benchmarks/baselines", "benchmarks/sglang")]

import common
import sampling_dist_check as sampling


class SamplingRecoveryTests(unittest.TestCase):
    def collect(self, path, post):
        args = SimpleNamespace(out=path, base_url="http://unused", model="fake", seed0=-1, n=1, tokens=2)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"enable_deterministic_inference":false,"random_seed":55}')
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [5, 6])
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: tokenizer))
        with (patch.dict(sys.modules, {"transformers": transformers}),
              patch.object(sampling.urllib.request, "urlopen", return_value=response),
              patch.object(sampling, "post", side_effect=post),
              redirect_stdout(io.StringIO())):
            sampling.collect(args)

    @staticmethod
    def complete(*args):
        return {"output_ids": [7, 8], "meta_info": {"finish_reason": {"type": "length"}}}

    def test_interrupted_rerun_leaves_no_old_or_partial_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            self.collect(path, self.complete)
            with self.assertRaisesRegex(SystemExit, "incomplete sample"):
                self.collect(path, [self.complete(), {"output_ids": []}])
            self.assertFalse(path.exists())
            self.assertFalse(sampling.metadata_path(path).exists())

    def test_replaced_samples_fail_the_sidecar_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            self.collect(path, self.complete)
            sidecar = sampling.load_metadata(path, required=True)
            digest = sidecar["samples_sha256"]
            self.assertEqual(len(sampling.load_samples(path, digest)), len(sampling.PROMPTS))
            self.assertEqual(len(sidecar["prompt_sha256"]), len(sampling.PROMPTS))
            path.write_text(path.read_text().replace("[7, 8]", "[7, 9]"))
            with self.assertRaisesRegex(ValueError, "sample file hash differs"):
                sampling.load_samples(path, digest)


class SweepRecoveryTests(unittest.TestCase):
    def test_occupied_port_cannot_be_reused(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            with self.assertRaises(OSError):
                common.check_loopback_port(server.getsockname()[1])

    def test_healthy_server_with_wrong_model_cannot_pass(self):
        healthy = MagicMock()
        healthy.__enter__.return_value.status = 200
        models = MagicMock()
        models.__enter__.return_value.read.return_value = b'{"data":[{"id":"another-sweep"}]}'
        with patch.object(common.urllib.request, "urlopen", side_effect=[healthy, models]):
            with self.assertRaisesRegex(RuntimeError, "does not serve this run"):
                common.wait_for_server("http://unused", timeout_s=1, model="this-sweep")

    def test_exited_server_fails_before_health_probe(self):
        with (patch.object(common.os, "kill", side_effect=ProcessLookupError),
              patch.object(common.urllib.request, "urlopen") as request):
            with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                common.wait_for_server("http://unused", timeout_s=1, pid=1)
        request.assert_not_called()

    def test_sweep_refuses_existing_results_before_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "REPORT.md"
            old.write_text("recorded evidence")
            result = subprocess.run(["bash", str(ROOT / "benchmarks/baselines/run_all.sh")],
                                    env={**os.environ, "OUT": str(root)}, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Choose a new OUT", result.stderr)
            self.assertEqual(old.read_text(), "recorded evidence")
