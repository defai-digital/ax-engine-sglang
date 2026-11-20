"""CPU regressions for streamed timing, diagnostic provenance and round audits."""
import hashlib
import importlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / name) for name in ("benchmarks/baselines", "benchmarks/sglang", "ax-engine-sglang")]

from accept_oracle import audit_round, load_rounds, match_responses
from common import is_diagnostic
from paired_speed import decode_seconds, group, main as paired_speed, points
from summarize import main as summarize
import token_identity

with patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=None)}):
    bench_speed = importlib.import_module("bench_speed")


def chunk(content=None, finish=None, tokens=None):
    choices = [] if tokens is not None else [
        SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)]
    usage = None if tokens is None else SimpleNamespace(completion_tokens=tokens)
    return SimpleNamespace(choices=choices, usage=usage)


def stream_client(*chunks):
    async def stream():
        for item in chunks:
            yield item

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=lambda **kwargs: stream()))))


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_chunk_has_no_decode_rate_and_ignores_usage_delay(self):
        client = stream_client(chunk("multiple tokens"), chunk(finish="length"), chunk(tokens=3))
        with patch.object(bench_speed.time, "perf_counter", side_effect=[2.0, 3.0, 10.0]):
            row = await bench_speed.one_request(client, "prompt", {})
        self.assertEqual(row["ttft_s"], 1.0)
        self.assertEqual(row["total_s"], 8.0)
        self.assertIsNone(row["decode_s"])
        self.assertIsNone(row["decode_tok_s"])

    async def test_decode_interval_ends_at_last_content_chunk(self):
        client = stream_client(chunk("first"), chunk("last"), chunk(finish="stop"), chunk(tokens=3))
        with patch.object(bench_speed.time, "perf_counter", side_effect=[2.0, 3.0, 5.0, 10.0]):
            row = await bench_speed.one_request(client, "prompt", {})
        self.assertEqual(row["decode_s"], 2.0)
        self.assertEqual(row["decode_tok_s"], 1.0)

    async def test_aborted_stream_does_not_produce_a_measurement(self):
        client = stream_client(chunk("partial"), chunk(finish="abort"), chunk(tokens=2))
        with self.assertRaisesRegex(RuntimeError, "generation did not complete"):
            await bench_speed.one_request(client, "prompt", {})

    async def test_small_benchmark_warms_up_with_separate_prompts(self):
        client = stream_client(chunk("answer"), chunk(finish="stop"), chunk(tokens=2))
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(base_url="http://unused", mode="ar", preset="greedy", thinking=False,
                                   max_tokens=2, n_requests=1, concurrency=[1], sets=["code"], out=directory,
                                   server_env=Path(directory) / "env.json", prompt_file=Path(directory) / "prompts.json")
            args.server_env.write_text(json.dumps({"lane": {"AX_QSA_CHECK": "1"}}))
            args.prompt_file.write_text(json.dumps({"code": ["measure", "warmup"]}))
            with (patch.object(bench_speed, "AsyncOpenAI", return_value=client),
                  patch.object(bench_speed, "load_prompt_sets", side_effect=AssertionError("snapshot must be used")),
                  patch.object(bench_speed, "scrape_spec_metrics", return_value={})):
                await bench_speed.main_async(args)
            calls = client.chat.completions.create.call_args_list
            self.assertEqual([call.kwargs["messages"][0]["content"] for call in calls], ["warmup", "measure"])
            report = json.loads((Path(directory) / "ar/speed_greedy.json").read_text())
            self.assertEqual(report["results"][0]["request_decode_seconds"], [None])
            self.assertEqual(report["results"][0]["prompt_sha256"], hashlib.sha256(b'["measure"]').hexdigest())
            self.assertTrue(report["diagnostic"])


class BenchmarkToolTests(unittest.TestCase):
    def test_paired_speed_rejects_different_ordered_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = (Path(directory) / f"{side}.json" for side in ("left", "right"))
            for path, digest in ((left, "a" * 64), (right, "b" * 64)):
                path.write_text(json.dumps({"results": [{"prompt_set": "code", "concurrency": 1,
                                                        "prompt_sha256": digest}]}))
            with (patch.object(sys, "argv", ["paired_speed.py", str(left), str(right)]),
                  redirect_stdout(io.StringIO()),
                  self.assertRaisesRegex(SystemExit, "ordered prompt digests must match")):
                paired_speed()

    def test_paired_statistics_exclude_the_same_unmeasurable_pairs(self):
        rows = [[0.0, 1.0, 3.0, 5], [0.0, 1.0, 100.0, 5], [0.0, 1.0, 20.0, 1]]
        a = decode_seconds({"requests": rows, "request_decode_seconds": [0.5, None, 1.0]})
        b = decode_seconds({"requests": rows})
        self.assertEqual(group(rows, rows, a, b, [0, 1, 2]), (4.0, 4.0))

    def test_server_side_diagnostic_switches_disqualify_speed_results(self):
        active_flags = ("AX_FORCE_SYNC", "AX_CHAIN_CHECK", "AX_LATTICE_DUPCHECK", "AX_NXT_DEBUG",
                        "AX_QSA_CHECK", "AX_ROUND_DUMP", "TORCH_PROFILE_STEPS")
        for name in active_flags:
            with self.subTest(flag=name):
                self.assertTrue(is_diagnostic({name: "1"}))
        self.assertFalse(is_diagnostic({"AX_LATTICE_DEBUG": "0", "AX_CHAIN_CHECK": "0", "AX_QSA_RING": "16"}))

    def test_diagnostic_baseline_is_excluded_from_speed_comparisons(self):
        row = dict(prompt_set="code", concurrency=1, output_tok_s=30.0,
                   per_req_decode_tok_s_p50=20.0, ttft_s_p50=0.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, diagnostic in (("ar", True), ("mtp2", False)):
                path = root / mode / "speed_greedy.json"
                path.parent.mkdir()
                path.write_text(json.dumps({"diagnostic": diagnostic, "results": [row]}))
            with self.assertRaisesRegex(SystemExit, "diagnostic timings"):
                points(root / "ar/speed_greedy.json")
            with redirect_stdout(io.StringIO()) as out:
                summarize(root)
            self.assertIn("| ar (diagnostic) | code | 1 | 30.0 | 20.0 | 0.500 | - | - | - |", out.getvalue())
            self.assertIn("| mtp2 | code | 1 | 30.0 | 20.0 | 0.500 | - | - | 1.00x |", out.getvalue())

    def test_identity_collection_refuses_a_short_prompt_set_before_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=str(Path(directory) / "ids.jsonl"), n=2, warmup=1,
                                   base_url="http://unused", expect_algorithm=None, sets=["math"])
            modules = {"transformers": SimpleNamespace(AutoTokenizer=None),
                       "prompts": SimpleNamespace(load_prompt_sets=lambda n: {"math": ["one"]})}
            with (patch.dict(sys.modules, modules), patch.object(token_identity, "server_info",
                  return_value={"model_path": "test"}), patch.object(token_identity, "post") as post):
                with self.assertRaisesRegex(SystemExit, "math has 1 prompts, requested 2"):
                    token_identity.collect(args)
            post.assert_not_called()
            self.assertFalse(Path(args.out).exists())

    def test_oracle_checks_bonus_from_saved_output(self):
        row = {"kind": "chain", "tokens": [10, 20], "accept": 1}
        count, accepted, bonus = audit_round(row, 0, [10, 20, 30], [10, 20, 31])
        self.assertEqual(count, 2)
        self.assertEqual(accepted, [])
        self.assertEqual(bonus, {"pos": 2, "server_token": 30, "bonus": True})
        self.assertEqual(audit_round(row, 0, [10], [10]), (0, [], None))

    def test_oracle_does_not_follow_rejected_drafts_past_bonus(self):
        row = {"kind": "chain", "tokens": [10, 20, 30], "accept": 0}
        count, (accepted,), bonus = audit_round(row, 0, [10, 21, 30], [10, 20, 30])
        self.assertEqual(count, 1)
        self.assertEqual(accepted["oracle_accept"], 1)
        self.assertTrue(accepted["oracle_accept_is_lower_bound"])
        self.assertEqual(accepted["pos"], bonus["pos"])
        _, (accepted,), bonus = audit_round(row, 0, [10, 20, 30], [10, 20, 30])
        self.assertTrue(accepted["structural"])
        self.assertIsNone(bonus)

    def test_oracle_checks_later_accept_errors_on_their_committed_prefixes(self):
        row = {"kind": "tree", "tokens": [10, 21, 31], "accept": 2}
        count, accepted, bonus = audit_round(row, 0, [10, 21, 31], [10, 20, 30])
        self.assertEqual(count, 2)
        self.assertEqual([error["pos"] for error in accepted], [1, 2])
        self.assertIsNone(bonus)

    def test_oracle_requires_a_dump_for_each_repeated_response(self):
        row = {"kind": "chain", "tokens": [10, 20], "accept": 1}
        saved = [(("code", 0), [10, 20, 30], "sha")] * 2
        with self.assertRaisesRegex(ValueError, "1 saved responses have no matching round dump"):
            match_responses({"pass1": [row]}, saved)
        rounds = {"pass1": [row], "pass2": [row], "warmup": [{**row, "tokens": [5, 6]}]}
        matches = match_responses(rounds, saved)
        self.assertEqual([match[0] for match in matches], ["pass1", "pass2"])

    def test_oracle_rejects_duplicate_or_reordered_round_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rounds.jsonl"
            for index in (5, 4):
                rows = [{"rid": "first", "round": 5}, {"rid": "second", "round": 5},
                        {"rid": "first", "round": index}]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, "non-increasing"):
                    load_rounds(path)
