"""CPU checks for the fixed-output NVFP4 benchmark and its comparison tools."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench_27b_nvfp4", ROOT / "benchmarks/sglang/bench_27b_nvfp4.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
mixed_spec = importlib.util.spec_from_file_location("ab_adaptive_width", ROOT / "benchmarks/sglang/ab_adaptive_width.py")
mixed = importlib.util.module_from_spec(mixed_spec)
mixed_spec.loader.exec_module(mixed)
vocab_spec = importlib.util.spec_from_file_location(
    "ab_dflash_draft_vocab", ROOT / "benchmarks/sglang/ab_dflash_draft_vocab.py",
)
vocab = importlib.util.module_from_spec(vocab_spec)
vocab_spec.loader.exec_module(vocab)

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
        self.base[-1:] = ["ar", "tree16"]
        self.assertIn("refuses the tree route", self.run_main("--verified-profile")[1])
        self.base[-1:] = ["ar", "chain8"]
        self.assertEqual(self.run_main("--verified-profile")[0], 0)
        args = self.lanes[0]
        for lane, block, tree in (("ar", None, None), ("chain8", "8", "0")):
            command, flags, algorithm = bench.launch_config(args, lane)
            self.assertEqual({name: value for name, value in flags.items() if name.startswith(("AX_", "FC1_"))},
                             {"AX_QUALITY_PROFILE": "verified", **({"AX_TREE": tree} if tree else {})})
            self.assertFalse({"--chunked-prefill-size", "--disable-radix-cache", "--attention-backend"} & set(command))
            self.assertEqual(algorithm, None if lane == "ar" else "FC1_DFLASH_TREE")
            if block:
                self.assertEqual(command[command.index("--speculative-num-draft-tokens") + 1], block)
        self.assertIn("separate runs", self.run_main("--verified-profile", "--verify-parity")[1])

    def test_verified_profile_requires_serialized_ar_before_launch(self):
        self.base[-1:] = ["chain8"]
        code, error = self.run_main("--verified-profile")
        self.assertEqual(code, 2)
        self.assertIn("requires the ar lane as the serialized reference", error)
        self.assertEqual(self.lanes, [])

    def test_concurrency_stays_within_the_slots(self):
        self.assertIn("between 1 and --slots", self.run_main("--concurrency", "4")[1])

    def test_kv_graph_lane_changes_only_the_append_flag(self):
        self.base[-1:] = ["replay8", "replay8-kv-graph", "replay8-kv-direct", "replay8-plan-reuse"]
        self.assertEqual(self.run_main()[0], 0)
        args = self.lanes[0]
        control_cmd, control_env, control_algo = bench.launch_config(args, "replay8")
        self.assertEqual(control_env["AX_DFLASH_BENCH_ACTIVITY"], "1")
        for lane, flag in (("replay8-kv-graph", "AX_DFLASH_KV_GRAPH"),
                           ("replay8-kv-direct", "AX_DFLASH_KV_DIRECT"),
                           ("replay8-plan-reuse", "AX_DFLASH_PLAN_REUSE")):
            command, env, algo = bench.launch_config(args, lane)
            self.assertEqual((command, algo), (control_cmd, control_algo))
            self.assertEqual(env, {**control_env, flag: "1"})

    def test_other_prompt_files_are_refused(self):
        with patch.object(bench, "PROMPT_SHA256", "0" * 64):
            self.assertIn("published parity-expanded-ids.json", self.run_main()[1])


class HotpathPassTests(unittest.TestCase):
    def test_parity_report_identifies_the_earliest_divergent_token(self):
        reference = [[1, 2], [3, 4], [5, 6]]
        rows = [{"output_ids": [1, 9]}, {"output_ids": [8, 4]}, {"output_ids": [5, 6]}]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(bench.report_matches("replay against stock", reference, rows), 1)
        self.assertIn("1/3 exact, earliest difference at output token 1 of request 2 (3 vs 8)", output.getvalue())

    def test_warmup_activity_does_not_count_in_the_timed_pass(self):
        before = {"rounds": {"8": 10}, "optimized": {"kv_direct": {"8": 10}}, "capture_attempts": 0}
        after = {**before, "rounds": {"8": 14}}
        with patch.object(bench, "hotpath_snapshot", side_effect=[before, after]), \
                self.assertRaisesRegex(RuntimeError, "no optimized rounds at live batch size 8"):
            bench.hotpath_pass("", "kv_direct", 8, dict)

    def test_first_capture_pass_is_discarded_and_smaller_batch_fallbacks_are_counted(self):
        initial = {"rounds": {}, "optimized": {}, "capture_attempts": 0}
        captured = {"rounds": {"8": 4}, "optimized": {"kv_graph": {"8": 4}}, "capture_attempts": 1}
        after = {"rounds": {"8": 10, "2": 2}, "optimized": {"kv_graph": {"8": 10, "2": 1}},
                 "capture_attempts": 1}
        measure = Mock(side_effect=[{"wall_s": 10}, {"wall_s": 1}])
        with patch.object(bench, "hotpath_snapshot", side_effect=[initial, captured, captured, after]):
            result = bench.hotpath_pass("", "kv_graph", 8, measure)
        self.assertEqual(measure.call_count, 2)
        self.assertEqual(result["wall_s"], 1)
        self.assertEqual(result["hotpath"]["rounds_by_live_batch"], {
            "2": {"optimized": 1, "fallback": 1}, "8": {"optimized": 6, "fallback": 0},
        })

    def test_a_timed_pass_with_requested_batch_fallbacks_is_rejected(self):
        before = {"rounds": {}, "optimized": {}, "capture_attempts": 0}
        after = {"rounds": {"8": 100}, "optimized": {"kv_graph": {"8": 1}}, "capture_attempts": 0}
        with patch.object(bench, "hotpath_snapshot", side_effect=[before, after]), \
                self.assertRaisesRegex(RuntimeError, "fell back at live batch size 8.*'fallback': 99"):
            bench.hotpath_pass("", "kv_graph", 8, dict)


class MixedWidthPassTests(unittest.TestCase):
    def test_cli_uses_the_same_frozen_schedule_for_all_width_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory) / "inputs.json"
            payload = json.dumps(PROMPTS).encode()
            inputs.write_bytes(payload)
            out = Path(directory) / "out"
            out.mkdir()
            argv = ["ab_adaptive_width", "--target", "t", "--draft", "d", "--input-ids", str(inputs),
                    "--out", str(out), "--lanes", "w16", "w8", "table", "--repeat", "1",
                    "--mixed-concurrency", "1", "2", "4"]
            with patch.object(sys, "argv", argv), patch.object(mixed, "run_lane", return_value=[]) as run:
                mixed.main()
            self.assertEqual([call.args[1] for call in run.call_args_list], ["w16", "w8", "table"])
            for call in run.call_args_list:
                settings, _, prompts = call.args
                self.assertEqual(settings.prompt_sha256, hashlib.sha256(payload).hexdigest())
                self.assertEqual(settings.mixed_concurrency, [1, 2, 4])
                self.assertEqual(prompts, PROMPTS)
            self.assertEqual(json.loads((out / "summary.json").read_text()), {"w16": [], "w8": [], "table": []})

    def test_wave_schedule_repeats_and_finishes_with_a_partial_wave(self):
        def generate(_url, prompt, length, speculative):
            self.assertEqual(length, 8)
            self.assertTrue(speculative)
            return {"output_ids": prompt, "wall_s": prompt[0] + 1, "spec_verify_ct": 2}

        prompts = [[number] for number in range(13)]
        with patch.object(mixed.bench, "generate", side_effect=generate):
            result = mixed.run_mixed_pass("", prompts, [1, 2, 4], 8)
        self.assertEqual(result["wave_sizes"], [1, 2, 4, 1, 2, 3])
        self.assertEqual(result["concurrency"], 4)
        self.assertEqual([row["output_ids"] for row in result["rows"]], prompts)
        self.assertEqual(result["output_tokens"], 104)
        self.assertEqual(result["tokens_per_verify"], 4)
        self.assertEqual(result["p95_request_s"], 13)


class DraftVocabularyPassTests(unittest.TestCase):
    def test_batched_pass_submits_each_wave_in_one_request(self):
        def generate(request, timeout):
            payload = json.loads(request.data)
            wave = payload["input_ids"]
            self.assertTrue(payload["sampling_params"]["ignore_eos"])
            waves.append(wave)
            return io.StringIO(json.dumps([
                {"output_ids": prompt, "meta_info": {"finish_reason": {"type": "length"}, "spec_verify_ct": 1}}
                for prompt in wave
            ]))

        prompts = [[i, i] for i in range(5)]
        waves = []
        with patch.object(vocab.urllib.request, "urlopen", side_effect=generate):
            result = vocab.run_batched_pass("http://localhost", prompts, 2, 2, True)
        self.assertEqual(waves, [prompts[:2], prompts[2:4], prompts[4:]])
        self.assertEqual([row["output_ids"] for row in result["rows"]], prompts)
        self.assertEqual(result["tokens_per_verify"], 2)

    def test_decode_profile_distinguishes_actual_batch_width_from_clients(self):
        graphs = {"8": {"total": 2, "subset": 1}, "4": {"total": 2, "subset": 0}}
        self.assertEqual(vocab.decode_profile(graphs, 8),
                         {"rounds": 4, "mean_live_requests": 6.0, "full_batch_rounds": 2, "subset_rounds": 1})


if __name__ == "__main__":
    unittest.main()
