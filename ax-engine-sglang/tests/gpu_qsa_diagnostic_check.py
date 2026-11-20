"""Check QSA diagnostics against multimodal coordinates and real overwritten slots.

    PYTHONPATH=ax-engine-sglang python ax-engine-sglang/tests/gpu_qsa_diagnostic_check.py

No model weights are required. Run on the development CUDA host.
"""
from types import SimpleNamespace
import unittest

import torch
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

from ax_engine_sglang import ax_qsa


class QsaDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.original_ring = ax_qsa.RING
        ax_qsa.CHECK.clear()

    def tearDown(self):
        ax_qsa.RING = self.original_ring
        ax_qsa.CHECK.clear()

    def run_case(self, ring, damage_rope=False, input_dtype=torch.int64):
        ax_qsa.CHECK.clear()
        ax_qsa.RING = ring
        pool = SimpleNamespace(qsa_rope_position_buffer=torch.full(
            (ring * 2, 3), -1, device="cuda", dtype=torch.int64))
        request = torch.tensor([1], device="cuda")
        indexer = SimpleNamespace(layer_id=3)
        # A 122-token image warmup ends in an incomplete group. The following
        # verify includes positions 122..125; group 120..123 must survive.
        for extend, positions in ((True, torch.arange(122, device="cuda", dtype=input_dtype)),
                                  (False, torch.arange(122, 126, device="cuda", dtype=input_dtype))):
            slots = ax_qsa.build_pending_ring_slots(
                token_to_batch_idx=torch.zeros_like(positions), req_pool_indices=request,
                sequence_lengths=torch.tensor([122 if extend else 126], device="cuda"),
                logical_positions=positions, compress_ratio=4, is_extend=extend)
            # MRoPE axes may differ from logical offsets and from one another.
            rope = torch.stack((positions - 56, positions // 2, positions * 2))
            QSATokenToKVPool.set_qsa_rope_position_buffer(pool, slots, rope)
            metadata = SimpleNamespace(
                is_cuda_graph=False, token_to_kv_pool=pool, req_pool_indices=request,
                compress_member_rows=torch.tensor([0], device="cuda") if extend else None,
                compress_group_positions=torch.tensor([123], device="cuda"),
                compress_sequence_ids=torch.tensor([0], device="cuda"),
                compress_group_ring_locs=ax_qsa.build_group_ring_slots(
                    req_pool_indices=request, group_end_positions=torch.tensor([123], device="cuda"),
                    sequence_ids=torch.tensor([0], device="cuda"), compress_ratio=4),
                write_locs=torch.tensor([1], device="cuda"))
            ax_qsa._record_check_positions(indexer, metadata, positions, rope, slots)
        if damage_rope:
            pool.qsa_rope_position_buffer[ring + 122 % ring, 1] += 1
        ax_qsa._check_groups(indexer, metadata)
        return ax_qsa.CHECK[id(pool)]

    def test_multimodal_coordinates_do_not_report_false_overwrites(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                counts = self.run_case(16, input_dtype=dtype)
                self.assertEqual((counts["groups"], counts["bad"]), (1, 0))

    def test_overwritten_ring_members_are_still_detected(self):
        self.assertEqual(self.run_case(4)["bad"], 1)

    def test_physical_rope_store_corruption_is_still_detected(self):
        self.assertEqual(self.run_case(16, damage_rope=True)["bad"], 1)


if __name__ == "__main__":
    unittest.main()
