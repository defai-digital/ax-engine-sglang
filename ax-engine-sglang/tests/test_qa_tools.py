"""Regression tests for the CPU sides of the serving gates."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "sglang"))

import mixed_requests_check as mixed
import sglang_teacher_check as teacher
import stress_multiuser as stress


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.meta = {"prompt_tokens": 3, "finish_reason": {"type": "length"},
                     "input_token_logprobs": [None, [-0.75, 7, ""]],
                     "input_top_logprobs": [None, [[-0.75, 7, ""], [-0.25, 8, ""]]]}

    def test_teacher_uses_highest_score_even_when_entries_are_out_of_order(self):
        self.assertEqual(teacher.score_response(self.meta, [7], 3, "test"), [
            {"pos": 0, "committed": 7, "best": 8, "regret": 0.5}])

    def test_teacher_rejects_invalid_scores_or_incomplete_responses(self):
        cases = [
            ({"input_token_logprobs": [None, [float("nan"), 7]]}, "non-finite"),
            ({"input_top_logprobs": [None, [[float("inf"), 7]]]}, "non-finite"),
            ({"input_top_logprobs": [None, [[0.1, 7]]]}, "positive"),
            ({"input_top_logprobs": [None, [[-0.9, 8]]]}, "exceeds"),
            ({"finish_reason": {"type": "abort"}}, "did not finish"),
            ({"prompt_tokens": 2}, "full prompt"),
        ]
        for patch_values, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                teacher.score_response({**self.meta, **patch_values}, [7], 3, "test")

    def test_failed_teacher_rerun_invalidates_an_old_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "out.json"
            out.write_text('{"passed": true}')
            argv = ["teacher", "--base-url", "http://unused", "--server-log", str(root / "log"),
                    "--ids", str(root / "missing"), "--out", str(out)]
            modules = {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: None)),
                       "prompts": SimpleNamespace(load_prompt_sets=None)}
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, modules):
                with self.assertRaises(FileNotFoundError):
                    teacher.main()
            self.assertFalse(out.exists())


class MixedRequestTests(unittest.TestCase):
    def response(self, content, finish="stop", **choice_fields):
        return {"choices": [{"message": {"content": content}, "finish_reason": finish, **choice_fields}]}

    def test_incomplete_or_malformed_success_responses_fail(self):
        for response in ({}, self.response("partial", "abort"), self.response(" ")):
            with self.subTest(response=response):
                self.assertIsNotNone(mixed.check("greedy", 200, response))

    def test_schema_requires_types_and_logprobs_require_usage(self):
        self.assertIsNone(mixed.check("json_schema", 200, self.response('{"name": "Python", "year": 1991}')))
        for invalid in ('["name", "year"]', '{"name": 1, "year": 1991}', '{"name": "Python", "year": true}'):
            with self.subTest(invalid=invalid):
                self.assertIsNotNone(mixed.check("json_schema", 200, self.response(invalid)))
        response = self.response("Python", logprobs={"content": [{}]})
        self.assertIsNotNone(mixed.check("logprobs", 200, response))
        response["usage"] = {"completion_tokens": 1}
        self.assertIsNone(mixed.check("logprobs", 200, response))


class StressTests(unittest.TestCase):
    def run_stress(self, post, abort=0, usage=(0, 0.1, 0.1)):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, log, out = (root / name for name in ("ids.json", "server.log", "out.json"))
            inputs.write_text("[[10]]")
            log.touch()
            argv = ["stress", "--input-ids", str(inputs), "--server-log", str(log), "--out", str(out),
                    "--n-ref", "1", "--n-stress", "1", "--abort", str(abort)]
            with (patch.object(sys, "argv", argv), patch.object(stress, "post", post),
                  patch.object(stress, "last_usage", return_value=usage), patch.object(stress.time, "sleep"),
                  patch.object(stress.random.Random, "uniform", return_value=0), redirect_stdout(io.StringIO())):
                with self.assertRaises(SystemExit) as finished:
                    stress.main()
            return finished.exception.code, json.loads(out.read_text())

    def test_unsolicited_abort_fails_the_stress_gate(self):
        def post(url, body, **kwargs):
            finish = "abort" if "rid" in body else "stop"
            return {"output_ids": [12], "meta_info": {"finish_reason": {"type": finish}}}

        code, report = self.run_stress(post)
        self.assertEqual(code, 1)
        self.assertEqual(report["stress"]["errors"], 1)

    def test_matching_non_idle_usage_cannot_pass(self):
        def post(url, body, **kwargs):
            return {"output_ids": [12], "meta_info": {"finish_reason": {"type": "stop"}}}

        code, report = self.run_stress(post, usage=(3, 0.1, 0.1))
        self.assertEqual(code, 1)
        self.assertFalse(report["usage_returned"])

    def test_cancellation_must_be_observed_when_requested(self):
        attempted = threading.Event()

        def post(url, body, **kwargs):
            if url.endswith("/abort_request"):
                attempted.set()
                return {}
            if "rid" in body:
                self.assertTrue(attempted.wait(2))
            return {"output_ids": [12], "meta_info": {"finish_reason": {"type": "stop"}}}

        code, report = self.run_stress(post, abort=1)
        self.assertEqual(code, 1)
        self.assertEqual(report["stress"]["aborted_sent"], 1)
        self.assertFalse(report["cancellation_exercised"])

    def test_successful_cancellation_and_idle_usage_pass(self):
        attempted = threading.Event()

        def post(url, body, **kwargs):
            if url.endswith("/abort_request"):
                attempted.set()
                return {}
            if "rid" in body:
                self.assertTrue(attempted.wait(2))
                return {"output_ids": [], "meta_info": {"finish_reason": {"type": "abort"}}}
            return {"output_ids": [12], "meta_info": {"finish_reason": {"type": "stop"}}}

        code, report = self.run_stress(post, abort=1)
        self.assertEqual(code, 0)
        self.assertTrue(report["usage_returned"])
        self.assertTrue(report["cancellation_exercised"])
