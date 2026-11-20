"""Check EAGLE target support against the real FlashInfer AR sampler."""
import unittest

from flashinfer.sampling import top_k_top_p_sampling_from_probs
from sglang.srt.speculative import eagle_utils, eagle_worker_common
import torch

from ax_engine_sglang.ax_eagle_sampling import install, joint_probs
from ax_engine_sglang.compat import installation


class EagleSamplingTests(unittest.TestCase):
    def test_joint_top_k_and_top_p_on_the_same_probabilities(self):
        probs = torch.tensor([[0.4, 0.27, 0.18, 0.15]], device="cuda")
        top_ks = torch.tensor([2], dtype=torch.int32, device="cuda")
        top_ps = torch.tensor([0.55], dtype=torch.float32, device="cuda")
        top_k = eagle_utils.top_k_renorm_prob
        top_p = eagle_utils.top_p_renorm_prob
        old = top_p(top_k(probs, top_ks), top_ps)
        fixed = joint_probs(probs, top_ks, top_ps, top_k, top_p)
        torch.testing.assert_close(old, torch.tensor([[1., 0., 0., 0.]], device="cuda"))
        expected = torch.tensor([[0.4 / 0.67, 0.27 / 0.67, 0., 0.]], device="cuda")
        torch.testing.assert_close(fixed, expected)
        trials = 40000
        drawn = top_k_top_p_sampling_from_probs(
            probs.expand(trials, -1).contiguous(), top_ks.expand(trials).contiguous(),
            top_ps.expand(trials).contiguous(), filter_apply_order="joint",
        )
        frequency = drawn.long().bincount(minlength=4).float() / trials
        torch.testing.assert_close(frequency, expected[0], atol=0.015, rtol=0)
        self.assertEqual(int(frequency[2:].count_nonzero()), 0)

    def test_runtime_import_alias_and_rollback(self):
        original = eagle_utils.eagle_sample
        self.assertIs(eagle_worker_common.eagle_sample, original)
        with self.assertRaisesRegex(RuntimeError, "rollback sentinel"), installation():
            install()
            self.assertIs(eagle_worker_common.eagle_sample, eagle_utils.eagle_sample)
            self.assertTrue(eagle_utils.eagle_sample._ax_joint_sampling)
            raise RuntimeError("rollback sentinel")
        self.assertIs(eagle_utils.eagle_sample, original)
        self.assertIs(eagle_worker_common.eagle_sample, original)


if __name__ == "__main__":
    unittest.main()
