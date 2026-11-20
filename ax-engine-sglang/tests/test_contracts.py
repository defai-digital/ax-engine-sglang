import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.contracts import benchmark_record, plan_greedy_commit

ROOT = Path(__file__).resolve().parents[2]


class ContractTests(unittest.TestCase):
    def test_versioned_commit_corpus(self):
        fixture = json.loads((ROOT / "contracts/speculation-v1.json").read_text())
        self.assertEqual(fixture["schema_version"], "ax.engine.speculation.v1")
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                args = [case[k] for k in ("drafts", "predicted", "budget", "eos")]
                if "error" in case:
                    with self.assertRaisesRegex(ValueError, case["error"]):
                        plan_greedy_commit(*args)
                else:
                    actual = plan_greedy_commit(*args)
                    for key in ("accepted_drafts", "emitted", "retained_draft_rows"):
                        self.assertEqual(getattr(actual, key), case[key])

    def test_benchmark_uses_workload_wall_time_and_preserves_unknowns(self):
        rows = [{"set": "code", "i": i, "prompt_sha1": str(i), "n_out": 10, "wall_s": 4.0}
                for i in range(2)]
        record = benchmark_record(rows=rows, wall_seconds=4.5, server={}, model="target",
                                  concurrency=2, max_tokens=10, diagnostic=None)
        self.assertEqual(record["measurement"]["wall_seconds"], 4.5)
        self.assertEqual(record["measurement"]["output_tokens"], 20)
        self.assertEqual(record["measurement"]["kind"], "unclassified")
        self.assertIsNone(record["model"]["revision"])
        self.assertIsNone(record["precision"])


if __name__ == "__main__":
    unittest.main()
