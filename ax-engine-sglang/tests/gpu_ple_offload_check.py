"""RTX control for Flash Next's pinned-host PLE gather, including CUDA replay.

    python ax-engine-sglang/tests/gpu_ple_offload_check.py

Uses synthetic checkpoint-dtype rows; no model weights or distributed groups.
"""
import unittest

import torch
import triton
from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel


class PleOffloadTests(unittest.TestCase):
    def test_large_table_uses_64_bit_offsets(self):
        width = 320
        boundary = 2 ** 31 // width
        rows = boundary + 3
        positions = [0, boundary - 1, boundary + 1, rows - 1]
        values = [0.5, -0.5, 1.0, 2.0]
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                table = torch.empty((rows, width), dtype=dtype, pin_memory=True)
                for row, value in zip(positions, values):
                    table[row].fill_(value)
                ids = torch.tensor(positions, device="cuda", dtype=torch.int64)
                output = torch.empty((len(positions), width), device="cuda", dtype=torch.bfloat16)
                _gather_ple_embedding_from_pinned_kernel[(len(positions),)](
                    table.data_ptr(), ids, output, embedding_dim=width,
                    tp_vocab_start=0, tp_vocab_end=rows,
                    is_fp8=dtype == torch.float8_e4m3fn, BLOCK_D=triton.next_power_of_2(width))
                expected = torch.tensor(values, device="cuda", dtype=torch.bfloat16)[:, None].expand_as(output)
                self.assertTrue(torch.equal(output, expected))

    def test_pinned_gather_and_replay_match_device_reference(self):
        torch.manual_seed(0)
        rows, width, start = 65536, 320, 128
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                table = torch.empty((rows, width), dtype=dtype, pin_memory=True)
                table.copy_(torch.randn(rows, width) * 0.5)
                reference = table.to(dtype=torch.bfloat16).cuda()
                ids = torch.randint(0, rows + 256, (4096,), device="cuda", dtype=torch.int64)
                output = torch.empty((ids.numel(), width), device="cuda", dtype=torch.bfloat16)
                def gather():
                    _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                        table.data_ptr(), ids, output, embedding_dim=width,
                        tp_vocab_start=start, tp_vocab_end=start + rows,
                        is_fp8=dtype == torch.float8_e4m3fn, BLOCK_D=triton.next_power_of_2(width))
                def expected():
                    valid = (ids >= start) & (ids < start + rows)
                    return torch.where(valid[:, None], reference[(ids - start).clamp(0, rows - 1)], 0)
                gather()
                self.assertTrue(torch.equal(output, expected()))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    gather()
                ids.copy_(torch.randint(0, rows + 256, ids.shape, device="cuda", dtype=ids.dtype))
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(output, expected()))


if __name__ == "__main__":
    unittest.main()
