"""Argument checks of benchmarks/sglang/bench_27b_nvfp4.py (no server is started)."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench_27b_nvfp4", ROOT / "benchmarks/sglang/bench_27b_nvfp4.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

# Lengths of the published parity prompts.
PROMPTS = [[0] * length for length in (2266, 44, 35, 27, 27, 26, 26, 24, 24, 23)]


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.prompts = directory / "ids.json"
        self.prompts.write_text(json.dumps(PROMPTS))
        self.enterContext(patch.object(bench, "PROMPT_SHA256", hashlib.sha256(self.prompts.read_bytes()).hexdigest()))
        self.lanes = []
        self.enterContext(patch.object(bench, "run_lane", self.run_lane))
        self.base = ["bench", "--target", "t", "--draft", "d", "--input-ids", str(self.prompts),
                     "--out", str(directory / "out"), "--lanes", "ar"]

    def run_lane(self, args, lane, prompts):
        self.lanes.append(args)
        return [{"concurrency": 1, "rows": [{"output_ids": [1]} for _ in prompts]}]

    def run_main(self, *extra):
        error = io.StringIO()
        with patch.object(sys, "argv", [*self.base, *extra]), contextlib.redirect_stderr(error), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                bench.main()
            except SystemExit as stopped:
                return stopped.code, error.getvalue()
        return 0, error.getvalue()

    def test_defaults_keep_the_published_passes(self):
        self.assertEqual(self.run_main()[0], 0)
        args = self.lanes[0]
        self.assertEqual((args.concurrent_passes, args.output_tokens, args.kv_tokens), ([(2, 4), (2, 4)], 128, 4096))

    def test_the_gate_run_needs_a_pool_for_every_concurrent_request(self):
        gate = ("--slots", "8", "--concurrency", "2", "8", "--repeat", "49", "--output-tokens", "1024")
        code, message = self.run_main(*gate)
        self.assertEqual(code, 2)
        self.assertIn("--kv-tokens must be at least 10667", message)   # 2,475 prompt tokens + 8 x 1,024
        self.assertEqual(self.run_main(*gate, "--kv-tokens", "10667")[0], 0)
        self.assertEqual(self.lanes[0].concurrent_passes, [(2, 49), (8, 49)])

    def test_the_context_covers_the_longest_prompt_and_its_output(self):
        code, message = self.run_main("--output-tokens", "2048", "--kv-tokens", "65536")
        self.assertEqual(code, 2)
        self.assertIn("--context-length must cover", message)
        self.assertEqual(self.run_main("--output-tokens", "2048", "--kv-tokens", "65536",
                                       "--context-length", "4314")[0], 0)

    def test_the_verified_profile_lanes_leave_the_settings_to_the_launcher(self):
        self.base[-1:] = ["ar", "tree16", "chain8"]
        self.assertEqual(self.run_main("--verified-profile")[0], 0)
        args = self.lanes[0]
        for lane, block, tree in (("ar", None, None), ("tree16", "16", None), ("chain8", "8", "0")):
            command, flags, algorithm = bench.launch_config(args, lane)
            self.assertEqual({name: value for name, value in flags.items() if name.startswith(("AX_", "FC1_"))},
                             {"AX_QUALITY_PROFILE": "verified", **({"AX_TREE": tree} if tree else {})})
            self.assertFalse({"--chunked-prefill-size", "--disable-radix-cache", "--attention-backend"} & set(command))
            self.assertEqual(algorithm, None if lane == "ar" else "FC1_DFLASH_TREE")
            if block:
                self.assertEqual(command[command.index("--speculative-num-draft-tokens") + 1], block)
        self.assertIn("separate runs", self.run_main("--verified-profile", "--verify-parity")[1])

    def test_concurrency_stays_within_the_slots(self):
        self.assertIn("between 1 and --slots", self.run_main("--concurrency", "4")[1])

    def test_other_prompt_files_are_refused(self):
        with patch.object(bench, "PROMPT_SHA256", "0" * 64):
            self.assertIn("published parity-expanded-ids.json", self.run_main()[1])


if __name__ == "__main__":
    unittest.main()
