"""Check that the CUDA teacher recording matches AX Draft's anchor convention."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/sglang/capture_ax_draft_teacher.py"
spec = importlib.util.spec_from_file_location("capture_ax_draft_teacher", SCRIPT)
teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher)


class TeacherRecordingTests(unittest.TestCase):
    def output(self):
        return {
            "output_ids": [7, 8, 9],
            "meta_info": {
                "finish_reason": {"type": "length"},
                "hidden_states": [[[10, 11], [20, 21]], [30, 31], [40, 41]],
            },
        }

    def test_last_prefill_feature_is_paired_with_first_output_token(self):
        tokens, features = teacher.recording(self.output(), [1, 2], 3, 2, 10)
        self.assertEqual(tokens, [7, 8, 9])
        np.testing.assert_array_equal(features, [[20, 21], [30, 31], [40, 41]])

    def test_chunked_prefill_is_rejected_without_publishing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher"
            result = self.output()
            result["meta_info"]["hidden_states"][0] = [[20, 21]]
            with patch.object(teacher, "generate", return_value=result):
                with self.assertRaisesRegex(ValueError, "unchunked prefill"):
                    teacher.capture("", [{"id": "p", "input_ids": [1, 2]}], output, 3, 2, 10)
            self.assertFalse(output.exists())

    def test_recording_cannot_replace_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                teacher.capture("", [], output, 3, 2, 10)

    def test_batched_recording_preserves_prompt_order_and_row_offsets(self):
        import json

        prompts = [{"id": "first", "input_ids": [1, 2]}, {"id": "second", "input_ids": [3, 4]}]
        responses = [self.output(), self.output()]
        responses[1]["output_ids"] = [6, 5, 4]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher"
            with patch.object(teacher, "generate", return_value=responses) as generate:
                teacher.capture("", prompts, output, 3, 2, 10, batch_size=2)
                generate.assert_called_once_with("", [[1, 2], [3, 4]], 3)
            rows = [json.loads(line) for line in (output / "sequences.jsonl").read_text().splitlines()]
            self.assertEqual(rows, [{"id": "first", "row": 0, "tokens": [7, 8, 9]},
                                    {"id": "second", "row": 3, "tokens": [6, 5, 4]}])
            self.assertEqual((output / "states.f32").stat().st_size, 48)
