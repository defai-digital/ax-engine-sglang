"""Run after installing the plugin on a CUDA host; no model weights required.

    AX_QSA_RING=16 FC1_TREE64=1 FC1_BRANCH_PILOT=1 FC1_FUSED=1 \
    FC1_TREE_N=16 FC1_DRAFT_BLOCK=16 AX_CHAIN_WIDTH=8 AX_SSM_POSITIONS=8 \
    python ax-engine-sglang/tests/gpu_runtime_check.py
"""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

from ax_engine_sglang.plugin import register
from ax_engine_sglang.runtime import activate_worker


class GpuRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ax_engine_sglang import plugin
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
        original_verify, original_capture = TritonGDNKernel.target_verify, DecodeCudaGraphRunner.capture
        original_packed_decode = TritonGDNKernel.supports_packed_decode
        install_hooks = plugin._install_tree_hooks
        def fail_after_hooks(config):
            install_hooks(config)
            raise RuntimeError("injected late startup failure")
        with patch.object(plugin, "_install_tree_hooks", side_effect=fail_after_hooks):
            try:
                register()
            except RuntimeError as error:
                if str(error) != "injected late startup failure":
                    raise
            else:
                raise AssertionError("startup failure was not injected")
        assert TritonGDNKernel.target_verify is original_verify
        assert DecodeCudaGraphRunner.capture is original_capture
        assert TritonGDNKernel.supports_packed_decode == original_packed_decode
        assert "fc1_dflash_worker" not in sys.modules
        assert not sys.modules["ax_state_cap"]._installed
        assert not plugin._registered and not plugin._tree_installed
        register()
        if plugin.get_config().gdn_unpacked_decode:
            assert TritonGDNKernel.supports_packed_decode is False

    def test_real_hooks_are_idempotent(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
        original = TritonGDNKernel.target_verify
        register()
        self.assertIs(TritonGDNKernel.target_verify, original)

    def test_fused_buffers_and_round_decisions_belong_to_worker(self):
        import ax_chain
        import fc1_fused
        def worker():
            backend = SimpleNamespace(topk=2)
            return SimpleNamespace(block_size=16, target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(attn_backend=backend)))
        first, second = worker(), worker()
        ax_chain.begin_round(first)
        first_context = fc1_fused.ensure_context()
        first_context.keys.fill_(7)
        batch = SimpleNamespace(reqs=[SimpleNamespace()], has_grammar=False,
                                return_logprob=False, sampling_info=None)
        self.assertEqual(ax_chain.choose(first, batch, draft_features_ready=True), "tree")
        first._ax_runtime.host_kv_len = 123
        ax_chain.begin_round(second)
        second_context = fc1_fused.ensure_context()
        self.assertNotEqual(first_context.keys.data_ptr(), second_context.keys.data_ptr())
        batch.reqs.append(SimpleNamespace())
        self.assertEqual(ax_chain.choose(second, batch, draft_features_ready=True), "chain")
        self.assertIsNone(second._ax_runtime.host_kv_len)
        activate_worker(first)
        self.assertIs(fc1_fused.context(), first_context)
        self.assertEqual(first._ax_runtime.host_kv_len, 123)
        self.assertEqual(first._ax_runtime.kind, "tree")
        self.assertTrue(bool((first_context.keys == 7).all()))

    def test_a_draft_graph_miss_uses_chain_verification(self):
        import ax_chain

        backend = SimpleNamespace(topk=2)
        worker = SimpleNamespace(block_size=16, target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=backend)))
        batch = SimpleNamespace(reqs=[SimpleNamespace()], has_grammar=False,
                                return_logprob=False, sampling_info=None)
        ax_chain.begin_round(worker)
        self.assertEqual(ax_chain.choose(worker, batch, draft_features_ready=True), "tree")
        worker._ax_tree_round = (id(batch), True)
        worker._ax_runtime.host_kv_len = 123
        self.assertEqual(ax_chain.choose(worker, batch, draft_features_ready=False), "chain")
        self.assertIsNone(worker._ax_runtime.host_kv_len)
        self.assertEqual(backend.topk, 1)

    def test_shorter_chain_keeps_the_proposal_rows_for_the_verified_tokens(self):
        import ax_chain

        translator = SimpleNamespace(plan=lambda **kwargs: kwargs)
        worker = SimpleNamespace(block_size=16, target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(kv_index_translator=translator)))
        ax_chain.begin_round(worker)
        worker._ax_round_width = 8
        tokens = torch.arange(32, device="cuda").view(2, 16)
        candidate_ids = torch.stack((tokens[:, 1:], tokens[:, 1:] + 32), dim=-1)
        probabilities = torch.rand_like(candidate_ids, dtype=torch.float32)
        worker._selector_sample = candidate_ids, probabilities
        batch = SimpleNamespace(spec_info=None, req_pool_indices=None, seq_lens_cpu=torch.zeros(2))
        narrowed = ax_chain.narrow(worker, batch, tokens, tokens.flatten(), tokens, None, None, 16)
        self.assertEqual(narrowed[-1], 8)
        torch.testing.assert_close(narrowed[0], tokens[:, :8])
        torch.testing.assert_close(worker._selector_sample[0], candidate_ids[:, :7])
        torch.testing.assert_close(worker._selector_sample[1], probabilities[:, :7])

    def test_qsa_ring_preserves_all_completed_groups_across_wrap(self):
        import ax_qsa
        self.assertEqual(ax_qsa.RING, 16)
        ratio, request = 4, 2
        for start in range(16, 32):
            for window in range(1, ax_qsa.max_window(ratio) + 1):
                positions = torch.arange(start, start + window, device="cuda")
                slots = ax_qsa.build_pending_ring_slots(
                    token_to_batch_idx=torch.zeros(window, device="cuda", dtype=torch.int64),
                    req_pool_indices=torch.tensor([request], device="cuda"),
                    sequence_lengths=torch.tensor([start + window], device="cuda"),
                    logical_positions=positions, compress_ratio=ratio, is_extend=False)
                ring = {request * 16 + p % 16: p for p in range(start - start % ratio, start)}
                ring.update(zip(slots.tolist(), positions.tolist()))
                ends = [p for p in range(start, start + window) if p % ratio == ratio - 1]
                if not ends:
                    continue
                ends_tensor = torch.tensor(ends, device="cuda")
                members = ax_qsa.build_group_ring_slots(
                    req_pool_indices=torch.tensor([request], device="cuda"),
                    group_end_positions=ends_tensor,
                    sequence_ids=torch.zeros(len(ends), device="cuda", dtype=torch.int64),
                    compress_ratio=ratio)
                for end, group in zip(ends, members.tolist()):
                    self.assertEqual([ring[s] for s in group], list(range(end - 3, end + 1)))

    def test_worker_reuses_the_buffers_bound_to_an_earlier_cuda_graph(self):
        import fc1_fused
        runner = SimpleNamespace()
        activate_worker(runner)
        captured = fc1_fused.ensure_context()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured.keys.fill_(5)
        worker = SimpleNamespace(target_worker=SimpleNamespace(model_runner=runner))
        activate_worker(worker)
        self.assertIs(fc1_fused.context(), captured)
        fc1_fused.context().keys.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(bool((fc1_fused.context().keys == 5).all()))


if __name__ == "__main__":
    unittest.main()
