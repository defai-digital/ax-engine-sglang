"""Compare the CUDA AX Draft selector with its eager reference."""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class AxDraftSelectorTests(unittest.TestCase):
    def test_selector_matches_eager_for_shortlists_and_ties(self):
        from ax_engine_sglang.ax_draft_kernels import select_chain

        generator = torch.Generator(device="cuda").manual_seed(17)
        for batch in (1, 8):
            for depth in (3, 7):
                for k in (4, 5, 8):
                    with self.subTest(batch=batch, depth=depth, k=k):
                        scores = torch.randint(-3, 4, (batch, depth, k), device="cuda", generator=generator).float()
                        edges = torch.randint(-3, 4, (batch, depth - 1, k, k), device="cuda", generator=generator).float()
                        candidates = torch.randint(0, 200, scores.shape, device="cuda", generator=generator)
                        vocabulary = torch.arange(200, device="cuda") * 3
                        lanes = torch.arange(batch, device="cuda")
                        picks = [scores[:, 0].argmax(-1)]
                        for pos in range(1, depth):
                            picks.append((scores[:, pos] + edges[lanes, pos - 1, picks[-1]]).argmax(-1))
                        selected = candidates.gather(2, torch.stack(picks, dim=1).unsqueeze(-1)).squeeze(-1)
                        torch.testing.assert_close(select_chain(scores, candidates, edges, vocabulary),
                                                   vocabulary[selected], rtol=0, atol=0)

    def test_selector_handles_nonfinite_scores_and_empty_batches(self):
        from ax_engine_sglang.ax_draft_kernels import select_chain

        scores = torch.tensor([[[float("inf"), 0.0, float("nan"), float("nan"), 3.0],
                                [float("-inf")] * 5]], device="cuda")
        candidates = torch.arange(10, device="cuda").reshape(1, 2, 5)
        edges = torch.zeros((1, 1, 5, 5), device="cuda")
        vocab = torch.arange(10, device="cuda")
        torch.testing.assert_close(select_chain(scores, candidates, edges, vocab),
                                   torch.tensor([[2, 5]], device="cuda"))
        self.assertEqual(select_chain(scores[:0], candidates[:0], edges[:0], vocab).shape, (0, 2))
