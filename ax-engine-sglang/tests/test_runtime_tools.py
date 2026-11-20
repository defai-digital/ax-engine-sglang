"""Offline test inputs must be rejected before dependency loads or HTTP requests."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_regression_module():
    path = ROOT / "scripts/check_model_regression.py"
    spec = importlib.util.spec_from_file_location("check_model_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regression_args(**overrides):
    values = dict(
        python=Path("/usr/bin/python3"), baseline=Path("/baseline"), candidate=Path("/candidate"),
        model=Path("/model"), draft=Path("/draft"), out=Path("/out"), port=30095, route="ar",
        slots=2, mem_fraction=0.8, context_limit=4096, max_tokens=64, model_override=None,
        attention_backend="triton", kv_cache_dtype=None, fp4_gemm_backend=None,
        gdn_recurrent_verify=False, baseline_route=None,
        mtp_draft=None, moe_runner_backend=None, speculative_moe_runner_backend=None,
        eager=False, enable_deterministic_inference=False, prefetch_checkpoints=False,
        server_seed=41, sampling_n=0, sampling_seed=-1, candidate_only=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def option(command, name):
    return command[command.index(name) + 1]


HAVE_SCIPY = importlib.util.find_spec("scipy") is not None


def write_distribution_cohorts(directory, a_ids, b_ids, samples=10, a_metadata=None, b_metadata=None):
    paths = []
    for name, ids, seed, extra in (("a.jsonl", a_ids, 11, a_metadata),
                                  ("b.jsonl", b_ids, 12, b_metadata)):
        path = Path(directory) / name
        path.write_text("".join(json.dumps({"p": 0, "i": i, "ids": list(ids)}) + "\n"
                                for i in range(samples)))
        metadata = {
            "seed_mode": "server_rng", "server_random_seed": seed,
            "prompts": 1, "samples_per_prompt": samples,
            "enable_deterministic_inference": False, "tokens": len(ids),
        }
        metadata.update(extra or {})
        Path(str(path) + ".metadata.json").write_text(json.dumps(metadata))
        paths.append(path)
    return paths


class RuntimeToolTests(unittest.TestCase):
    def test_relative_paths_keep_pointing_to_caller_files_from_the_lane_checkout(self):
        module = load_regression_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "model/config.json").write_text('{"model_type": "qwen3"}')
            (root / "candidate").mkdir()
            (root / "python").symlink_to(sys.executable)
            paths = {name: Path(os.path.relpath(root / name)) for name in
                     ("python", "baseline", "candidate", "model", "draft", "out")}
            argv = ["regression", "--candidate-only", "--route", "ar"]
            for name, path in paths.items():
                argv.extend((f"--{name}", str(path)))

            class Tokenizer:
                def apply_chat_template(self, *args, **kwargs):
                    return "prompt"

                def __call__(self, *args, **kwargs):
                    return SimpleNamespace(input_ids=[7])

            def run_lane(args, name, repo, encoded):
                for key, path in paths.items():
                    self.assertEqual(getattr(args, key), path.absolute())
                self.assertTrue(args.python.is_symlink())
                launched = subprocess.run([str(args.python), "-V"], cwd=repo, capture_output=True, check=True)
                self.assertIn(b"Python", launched.stdout)
                self.assertTrue(args.model.joinpath("config.json").is_file())
                return {"tokens": [[1]]}

            tokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
            with (patch.object(sys, "argv", argv), patch.object(module, "run_lane", run_lane),
                  patch.object(module.socket, "socket"),
                  patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=tokenizer)})):
                module.main()
            report = json.loads((root / "out/result.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["greedy_comparison"], "not_run_single_lane")

    def test_saved_greedy_comparison_retains_first_shared_prefixes_and_endings(self):
        script = ROOT / "scripts/check_model_regression.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, candidate = root / "reference", root / "candidate"
            for lane, outputs in ((reference, [[11, 12], [21], [31]]),
                                  (candidate, [[11, 13], [21, 22], [31]])):
                lane.mkdir()
                (lane / "input_ids.json").write_text(json.dumps([[1, 2], [3], [4]]))
                (lane / "tokens.json").write_text(json.dumps(outputs))
            out = root / "greedy.json"
            command = [sys.executable, str(script), "compare-greedy", "--reference", str(reference),
                       "--candidate", str(candidate), "--out", str(out)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(out.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["identical"], 1)
            self.assertEqual(report["differences"], [
                {"prompt_index": 0, "output_position": 1, "prefix_ids": [1, 2, 11],
                 "reference_token_id": 12, "candidate_token_id": 13},
                {"prompt_index": 1, "output_position": 1, "prefix_ids": [3, 21],
                 "reference_token_id": None, "candidate_token_id": 22},
            ])
            (candidate / "tokens.json").write_text((reference / "tokens.json").read_text())
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(out.read_text())["passed"])
            (candidate / "input_ids.json").write_text(json.dumps([[True, 2], [3], [4]]))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("different input token IDs", result.stderr)

    def test_server_rng_sampling_lanes_get_distinct_recorded_seeds(self):
        module = load_regression_module()
        args = regression_args(sampling_n=8, sampling_seed=-1, server_seed=41)
        baseline = module.lane_sampling(args, "baseline")
        candidate = module.lane_sampling(args, "candidate")
        self.assertEqual(baseline["seed_mode"], "server_rng")
        self.assertEqual(candidate["seed_mode"], "server_rng")
        self.assertEqual((baseline["server_random_seed"], candidate["server_random_seed"]), (41, 42))
        self.assertEqual(option(module.launch_command(args, "baseline"), "--random-seed"), "41")
        self.assertEqual(option(module.launch_command(args, "candidate"), "--random-seed"), "42")

    def test_benchmark_helpers_support_older_checkouts(self):
        module = load_regression_module()
        filename = "mixed_requests_check.py"
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory)
            helper = baseline / "track_l" / filename
            helper.parent.mkdir()
            helper.touch()
            self.assertEqual(module.benchmark_script(baseline, filename), helper)
        self.assertEqual(module.benchmark_script(ROOT, filename), ROOT / "benchmarks/sglang" / filename)

    def test_server_rng_sampling_enforces_independence_gate(self):
        module = load_regression_module()
        args = regression_args(sampling_n=8, sampling_seed=-1)
        command = module.sampling_compare_command(args)
        self.assertEqual(command[-1], "--require-independent-seeds")
        collect = module.sampling_collect_command(args, "http://127.0.0.1:30095", args.out / "baseline")
        self.assertEqual(option(collect, "--seed0"), "-1")

    def test_single_lane_server_rng_preserves_the_requested_seed(self):
        module = load_regression_module()
        args = regression_args(sampling_n=8, sampling_seed=-1, server_seed=41, candidate_only=True)
        self.assertEqual(module.lane_sampling(args, "candidate")["server_random_seed"], 41)
        self.assertEqual(option(module.launch_command(args, "candidate"), "--random-seed"), "41")

    def test_model_regression_forwards_precision_and_attention_settings(self):
        module = load_regression_module()
        args = regression_args(attention_backend="flashinfer", kv_cache_dtype="fp8_e4m3",
                               fp4_gemm_backend="flashinfer_cutlass")
        command = module.launch_command(args, "candidate")
        self.assertEqual(option(command, "--attention-backend"), "flashinfer")
        self.assertEqual(option(command, "--kv-cache-dtype"), "fp8_e4m3")
        self.assertEqual(option(command, "--fp4-gemm-backend"), "flashinfer_cutlass")

    def test_recurrent_gdn_comparison_uses_triton_only_on_the_dflash_lane(self):
        module = load_regression_module()
        for route in ("tree", "chain"):
            with self.subTest(route=route):
                args = regression_args(baseline_route="ar", route=route, gdn_recurrent_verify=True)
                baseline = module.launch_command(args, "baseline")
                candidate = module.launch_command(args, "candidate")
                self.assertNotIn("--speculative-algorithm", baseline)
                self.assertNotIn("--linear-attn-verify-backend", baseline)
                self.assertEqual(option(candidate, "--speculative-algorithm"), "FC1_DFLASH_TREE")
                self.assertEqual(option(candidate, "--linear-attn-verify-backend"), "triton")

    def test_cross_route_run_preserves_saved_prompts_and_rejects_greedy_divergence(self):
        module = load_regression_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type": "qwen3"}')
            prompts = [[5, 6, 7], [8, 9]]
            inputs = root / "saved.json"
            inputs.write_text(json.dumps(prompts))
            argv = ["regression", "--baseline-route", "ar", "--route", "chain",
                    "--input-ids", str(inputs), "--gdn-recurrent-verify", "--gdn-unpacked-decode"]
            for option_name in ("python", "baseline", "candidate", "model", "draft", "out"):
                argv.extend((f"--{option_name}", str(root / "out" if option_name == "out" else root)))
            routes = []

            def run_lane(args, name, repo, encoded):
                self.assertEqual(encoded, prompts)
                routes.append(module.lane_route(args, name))
                return {"tokens": [[11], [12 if name == "baseline" else 14]]}

            with (patch.object(sys, "argv", argv), patch.object(module, "run_lane", run_lane),
                  patch.object(module.socket, "socket"),
                  patch.object(module, "tokenize_prompts", side_effect=AssertionError("retokenized")),
                  self.assertRaisesRegex(SystemExit, "greedy tokens differ")):
                module.main()
            self.assertEqual(routes, ["ar", "chain"])
            report = json.loads((root / "out/greedy_comparison.json").read_text())
            self.assertEqual(report["identical"], 1)
            self.assertEqual(report["differences"][0]["prefix_ids"], [8, 9])

    def test_request_seed_mode_is_collection_only(self):
        module = load_regression_module()
        args = regression_args(sampling_n=8, sampling_seed=1000000, server_seed=41,
                               enable_deterministic_inference=True, candidate_only=True)
        self.assertEqual(module.lane_server_seed(args, "baseline"),
                         module.lane_server_seed(args, "candidate"))
        self.assertEqual(module.lane_sampling(args, "candidate")["seed_mode"], "request")
        with self.assertRaisesRegex(ValueError, "correlated"):
            module.sampling_compare_command(args)
        for name in ("baseline", "candidate"):
            collect = module.sampling_collect_command(args, "http://127.0.0.1:30095", args.out / name)
            self.assertEqual(option(collect, "--seed0"), "1000000")

    def test_correlated_request_seed_cohorts_cannot_receive_a_statistical_pass(self):
        module = load_regression_module()
        args = regression_args(sampling_n=8, sampling_seed=1000000,
                               enable_deterministic_inference=True)
        with patch.object(module.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "correlated"):
                module.compare_sampling_cohorts(args)
        run.assert_not_called()

    def test_correlated_request_seed_cohorts_are_rejected_before_launch(self):
        script = ROOT / "scripts/check_model_regression.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            result = subprocess.run([
                sys.executable, str(script), "--python", sys.executable,
                "--baseline", str(root), "--candidate", str(root),
                "--model", str(root), "--draft", str(root), "--out", str(output),
                "--sampling-n", "4", "--sampling-seed", "1000000",
                "--enable-deterministic-inference",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("paired request-seed cohorts are correlated", result.stderr)
            self.assertFalse(output.exists())

    def test_server_rng_sampling_rejects_deterministic_inference_before_launch(self):
        script = ROOT / "scripts/check_model_regression.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
            output = root / "out"
            result = subprocess.run([
                sys.executable, str(script), "--python", sys.executable,
                "--baseline", str(root), "--candidate", str(root),
                "--model", str(root), "--draft", str(root), "--out", str(output),
                "--sampling-n", "4", "--sampling-seed", "-1", "--enable-deterministic-inference",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("server RNG sampling cohorts require", result.stderr)
            self.assertFalse(output.exists())

    @unittest.skipUnless(HAVE_SCIPY, "scipy is required for chi-square verdicts")
    def test_identical_distinct_seed_cohorts_pass_statistical_verdict(self):
        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        with tempfile.TemporaryDirectory() as directory:
            a, b = write_distribution_cohorts(directory, [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
            result = subprocess.run([
                sys.executable, str(script), "compare", str(a), str(b),
                "--require-independent-seeds",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("DIST_PASS", result.stdout)

    @unittest.skipUnless(HAVE_SCIPY, "scipy is required for chi-square verdicts")
    def test_divergent_distinct_seed_cohorts_fail_statistical_verdict(self):
        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        with tempfile.TemporaryDirectory() as directory:
            a, b = write_distribution_cohorts(directory, [1] * 6, [2] * 6)
            result = subprocess.run([
                sys.executable, str(script), "compare", str(a), str(b),
                "--require-independent-seeds",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("DIST_FAIL", result.stdout)

    @unittest.skipUnless(HAVE_SCIPY, "scipy is required for chi-square verdicts")
    def test_early_stopping_cannot_hide_differences_at_later_positions(self):
        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        with tempfile.TemporaryDirectory() as directory:
            a, b = write_distribution_cohorts(directory, [1], [1, 2, 3], samples=20,
                                               a_metadata={"tokens": 3})
            result = subprocess.run([
                sys.executable, str(script), "compare", str(a), str(b), "--require-independent-seeds",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("position 2", result.stdout)

    def test_independence_gate_rejects_different_sampling_controls(self):
        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        for metadata, message in (({"top_p": 0.8}, "different top_p"),
                                  ({"tokens": 1}, "exceeds the recorded token budget"),
                                  ({"enable_deterministic_inference": True}, "deterministic inference disabled")):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                a, b = write_distribution_cohorts(directory, [1, 2], [1, 2], b_metadata=metadata)
                result = subprocess.run([
                    sys.executable, str(script), "compare", str(a), str(b), "--require-independent-seeds",
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(message, result.stdout)

    def test_single_sample_cohorts_cannot_pass(self):
        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        with tempfile.TemporaryDirectory() as directory:
            a, b = write_distribution_cohorts(directory, [1, 1], [2, 2], samples=1)
            result = subprocess.run([
                sys.executable, str(script), "compare", str(a), str(b), "--require-independent-seeds",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("DIST_INCONCLUSIVE", result.stdout)

    @unittest.skipUnless(HAVE_SCIPY, "scipy is required for chi-square verdicts")
    def test_fully_pooled_differences_are_inconclusive(self):
        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        with tempfile.TemporaryDirectory() as directory:
            a, b = write_distribution_cohorts(directory, [1], [2], samples=10)
            for path, offset in ((a, 0), (b, 10)):
                path.write_text("".join(json.dumps({"p": 0, "i": i, "ids": [offset + i]}) + "\n"
                                        for i in range(10)))
            result = subprocess.run([
                sys.executable, str(script), "compare", str(a), str(b), "--require-independent-seeds",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("low expected counts", result.stdout)
            self.assertIn("DIST_INCONCLUSIVE", result.stdout)

    def test_sampling_comparison_not_requested_status(self):
        module = load_regression_module()
        args = regression_args(sampling_n=0)
        with patch.object(module.subprocess, "run") as run:
            self.assertEqual(module.compare_sampling_cohorts(args), "not_requested")
        run.assert_not_called()

    def test_correlated_request_metadata_cannot_pass_the_independence_gate(self):
        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        metadata = {"seed_mode": "request", "enable_deterministic_inference": True,
                    "requested_seed_base": 1000}
        with tempfile.TemporaryDirectory() as directory:
            a, b = write_distribution_cohorts(directory, [1, 2], [1, 2],
                                               a_metadata=metadata, b_metadata=metadata)
            for gate in ("--require-independent-seeds", "--require-request-seeds"):
                with self.subTest(gate=gate):
                    result = subprocess.run([sys.executable, str(script), "compare", str(a), str(b), gate],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertNotIn("DIST_PASS", result.stdout)

    def test_independence_gate_rejects_invalid_metadata_counts(self):
        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        for key in ("prompts", "samples_per_prompt"):
            for value in (True, 1.0, 0, -1):
                with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                    a, b = write_distribution_cohorts(directory, [1, 2], [1, 2], samples=1,
                                                       a_metadata={key: value})
                    result = subprocess.run([
                        sys.executable, str(script), "compare", str(a), str(b),
                        "--require-independent-seeds",
                    ], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("positive integer", result.stdout)

    def test_independence_gate_rejects_metadata_count_mismatch(self):
        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        for metadata in ({"prompts": 2}, {"samples_per_prompt": 99}):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                a, b = write_distribution_cohorts(directory, [1, 2], [1, 2], a_metadata=metadata)
                result = subprocess.run([
                    sys.executable, str(script), "compare", str(a), str(b),
                    "--require-independent-seeds",
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("sample counts do not match the collection metadata", result.stdout)

    def test_server_rng_collection_rejects_deterministic_inference_before_output(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"server_args": {
                    "enable_deterministic_inference": True, "random_seed": 0}}).encode())

            def log_message(self, *args):
                pass

        script = ROOT / "benchmarks/sglang/sampling_dist_check.py"
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as directory:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            output = Path(directory) / "samples.jsonl"
            try:
                result = subprocess.run([
                    sys.executable, str(script), "collect", "--model", "unused",
                    "--base-url", f"http://127.0.0.1:{server.server_port}",
                    "--seed0", "-1", "--out", str(output),
                ], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("server RNG collection requires deterministic inference disabled", result.stderr)
                self.assertFalse(output.exists())
                self.assertFalse(Path(str(output) + ".metadata.json").exists())
            finally:
                server.shutdown()
                thread.join()

    def test_single_lane_sampling_reports_not_compared(self):
        module = load_regression_module()
        args = regression_args(sampling_n=8, candidate_only=True)
        with patch.object(module.subprocess, "run") as run:
            self.assertEqual(module.compare_sampling_cohorts(args), "not_run_single_lane")
        run.assert_not_called()

    def test_sampling_comparison_passed_only_after_successful_compare(self):
        module = load_regression_module()
        args = regression_args(sampling_n=8)
        with patch.object(module.subprocess, "run") as run:
            self.assertEqual(module.compare_sampling_cohorts(args), "passed")
        run.assert_called_once()
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertIn("--require-independent-seeds", run.call_args.args[0])

    def test_sampling_comparison_failure_does_not_return_passed(self):
        module = load_regression_module()
        args = regression_args(sampling_n=8)
        failure = module.subprocess.CalledProcessError(1, "compare")
        with patch.object(module.subprocess, "run", side_effect=failure):
            with self.assertRaises(module.subprocess.CalledProcessError):
                module.compare_sampling_cohorts(args)

    def test_same_server_rng_seed_cannot_pass_independence_gate(self):
        script = Path(__file__).resolve().parents[2] / "benchmarks/sglang/sampling_dist_check.py"
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("a.jsonl", "b.jsonl")]
            for path in paths:
                path.write_text(json.dumps({"p": 0, "i": 0, "ids": [1, 2]}) + "\n")
                Path(str(path) + ".metadata.json").write_text(json.dumps({
                    "seed_mode": "server_rng", "server_random_seed": 0}))
            result = subprocess.run([
                sys.executable, str(script), "compare", *map(str, paths),
                "--require-independent-seeds",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("require different recorded server RNG seeds", result.stdout)

    def test_ignored_request_seeds_are_rejected_before_sample_collection(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"server_args": {
                    "enable_deterministic_inference": False, "random_seed": 0}}).encode())

            def log_message(self, *args):
                pass

        script = Path(__file__).resolve().parents[2] / "benchmarks/sglang/sampling_dist_check.py"
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as directory:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            output = Path(directory) / "samples.jsonl"
            try:
                result = subprocess.run([
                    sys.executable, str(script), "collect", "--model", "unused",
                    "--base-url", f"http://127.0.0.1:{server.server_port}",
                    "--seed0", "1000000", "--out", str(output),
                ], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--seed0 requires", result.stderr)
                self.assertFalse(output.exists())
            finally:
                server.shutdown()
                thread.join()

    def test_incomplete_distribution_samples_cannot_pass(self):
        script = Path(__file__).resolve().parents[2] / "benchmarks/sglang/sampling_dist_check.py"
        row = {"p": 0, "i": 0, "ids": [1, 2]}
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl"
            a.write_text(json.dumps(row) + "\n")
            for rows in ([], [{**row, "p": 1}], [row, row],
                         [{**row, "ids": []}], [row, {**row, "i": 1}]):
                with self.subTest(rows=rows):
                    b.write_text("".join(json.dumps(item) + "\n" for item in rows))
                    result = subprocess.run([
                        sys.executable, str(script), "compare", str(a), str(b),
                    ], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("DIST_FAIL invalid samples", result.stdout)

    def test_flash_tree_route_is_refused_before_loading_dependencies_or_weights(self):
        script = Path(__file__).resolve().parents[2] / "scripts/check_model_regression.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            output = root / "out"
            result = subprocess.run([
                sys.executable, str(script), "--python", sys.executable,
                "--baseline", str(root), "--candidate", str(root),
                "--model", str(root), "--draft", str(root), "--out", str(output),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Flash Next DFlash2 verification needs PLE/QSA support", result.stderr)
            self.assertFalse(output.exists())

    def test_invalid_stress_token_inputs_fail_without_network(self):
        script = Path(__file__).resolve().parents[2] / "benchmarks/sglang/stress_multiuser.py"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for value in ([], [[]], [[-1]], [[True]], [[1.5]], "text", {}):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    result = subprocess.run([
                        sys.executable, str(script), "--input-ids", str(path),
                        "--base-url", "http://127.0.0.1:1", "--server-log", str(path),
                        "--out", str(Path(directory) / "result.json"),
                    ], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("--input-ids must contain", result.stderr)


if __name__ == "__main__":
    unittest.main()
