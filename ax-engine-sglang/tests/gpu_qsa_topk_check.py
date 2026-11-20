"""Stable QSA selection and changed-bound CUDA graph replay on the CUDA host.

    PYTHONPATH=ax-engine-sglang python ax-engine-sglang/tests/gpu_qsa_topk_check.py
"""
import os
import sys
import unittest
from unittest.mock import patch

import torch

from ax_engine_sglang.qsa_topk import stable_topk


def reference(scores, starts, ends, count):
    matrix = scores.cpu().tolist()
    result = []
    for row, start, end in zip(matrix, starts.cpu().tolist(), ends.cpu().tolist()):
        indices = sorted(range(end - start), key=lambda i: (-row[start + i], i))[:count]
        result.append(indices + [-1] * (count - len(indices)))
    return torch.tensor(result, dtype=torch.int32, device=scores.device).reshape(scores.shape[0], count)


class QsaTopkTests(unittest.TestCase):
    def test_ragged_rows_ties_and_short_inputs(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            for columns in (0, 1, 16, 513, 1024):
                for rows in (0, 1, 4):
                    scores = torch.randint(0, 8, (rows, columns), device="cuda").to(dtype)
                    if columns:
                        scores[:, ::7] = float("-inf")
                    starts = torch.tensor([min(i * 3, columns) for i in range(rows)], device="cuda")
                    ends = torch.tensor([min(columns, start + i * 257)
                                         for i, start in enumerate(starts.tolist())], device="cuda")
                    for count in (1, 4, 512):
                        with self.subTest(dtype=dtype, columns=columns, rows=rows, count=count):
                            actual = stable_topk(scores, starts, ends, count)
                            self.assertTrue(torch.equal(actual, reference(scores, starts, ends, count)))

    def test_graph_replay_uses_current_scores_and_bounds(self):
        scores = torch.randint(0, 8, (4, 1024), device="cuda").float()
        starts = torch.tensor([0, 3, 15, 256], device="cuda")
        ends = torch.tensor([0, 4, 32, 1024], device="cuda")
        stable_topk(scores, starts, ends, 512)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = stable_topk(scores, starts, ends, 512)
        starts.copy_(torch.tensor([64, 0, 2, 8], device="cuda"))
        ends.copy_(torch.tensor([664, 0, 502, 1024], device="cuda"))
        scores.copy_(torch.randint(0, 8, scores.shape, device="cuda"))
        scores[:, ::5] = float("-inf")
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual, reference(scores, starts, ends, 512)))

    def test_real_qsa_aliases_install_and_roll_back_without_tree_worker(self):
        from ax_engine_sglang import ax_qsa
        from ax_engine_sglang.compat import installation
        from ax_engine_sglang.config import get_config
        from sglang.srt.layers.attention.qsa import kernel, qsa_indexer
        original = qsa_indexer.qsa_fast_topk
        try:
            with self.assertRaisesRegex(RuntimeError, "rollback control"):
                with installation(), patch.dict(os.environ, {"AX_QSA_STABLE_TOPK": "1"}):
                    get_config.cache_clear()
                    ax_qsa.install()
                    self.assertIs(kernel.qsa_fast_topk, stable_topk)
                    self.assertIs(qsa_indexer.qsa_fast_topk, stable_topk)
                    self.assertNotIn("fc1_dflash_worker", sys.modules)
                    raise RuntimeError("rollback control")
            self.assertIs(qsa_indexer.qsa_fast_topk, original)
            self.assertIs(kernel.qsa_fast_topk, original)
        finally:
            get_config.cache_clear()


if __name__ == "__main__":
    unittest.main()
