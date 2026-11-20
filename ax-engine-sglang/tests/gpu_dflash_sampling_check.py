"""Check DFlash2 verifier support against the real FlashInfer AR sampler."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flashinfer.sampling import top_k_top_p_sampling_from_probs
from sglang.kernels.ops.speculative.dspark import dspark_accept
from sglang.srt import runtime_context
from sglang.srt.speculative import dflash_utils, spec_sampling_mask
import torch

from ax_engine_sglang.ax_dflash_sampling import install
from ax_engine_sglang.compat import installation


class DFlashSamplingTests(unittest.TestCase):
    def test_target_policy_and_runtime_aliases(self):
        device = "cuda"
        probs = torch.tensor([[0.4, 0.27, 0.18, 0.15],
                              [0.41, 0.28, 0.17, 0.14]], device=device)
        info = SimpleNamespace(
            temperatures=torch.tensor([[0.8], [1.2]], device=device),
            top_ks=torch.tensor([2, 3], dtype=torch.int32, device=device),
            top_ps=torch.tensor([0.55, 0.8], device=device),
            need_top_k_sampling=True, need_top_p_sampling=True, need_min_p_sampling=False,
        )
        logits = (probs.log() * info.temperatures).repeat_interleave(2, dim=0)
        inputs = dict(next_token_logits=logits, sampling_info=info, draft_token_num=2, bs=2)
        original = dflash_utils.build_speculative_verify_target_probs
        stock = original(**inputs)
        ar_backend = SimpleNamespace(kernel=SimpleNamespace(sampling_backend="flashinfer"))
        spec = SimpleNamespace(speculative_algorithm="FC1_DFLASH_TREE")
        with (patch.object(runtime_context, "get_exec", return_value=ar_backend),
              patch.object(runtime_context, "get_spec", return_value=spec),
              self.assertRaisesRegex(RuntimeError, "rollback sentinel"), installation()):
            install()
            build = dflash_utils.build_speculative_verify_target_probs
            self.assertIs(dspark_accept.build_speculative_verify_target_probs, build)
            self.assertIs(spec_sampling_mask.build_speculative_verify_target_probs, build)
            expected = probs.clone()
            expected[0, 2:] = 0
            expected[1, 3:] = 0
            expected /= expected.sum(dim=-1, keepdim=True)
            actual = build(**inputs)
            torch.testing.assert_close(actual, expected[:, None].expand(-1, 2, -1))
            self.assertEqual(stock[0, 0, 1].item(), 0.0)
            draws_per_row = 40000
            draws = top_k_top_p_sampling_from_probs(
                probs.repeat_interleave(draws_per_row, dim=0),
                info.top_ks.repeat_interleave(draws_per_row),
                info.top_ps.repeat_interleave(draws_per_row),
                filter_apply_order="joint", seed=3827,
            ).view(2, draws_per_row)
            frequency = torch.stack([row.long().bincount(minlength=4) for row in draws])
            self.assertEqual(int(frequency[expected == 0].sum()), 0)
            torch.testing.assert_close(frequency.float() / draws_per_row, expected, atol=0.015, rtol=0)
            spec.speculative_algorithm = "DFLASH"
            torch.testing.assert_close(build(**inputs), stock)
            spec.speculative_algorithm = "FC1_DFLASH_TREE"
            ar_backend.kernel.sampling_backend = "pytorch"
            torch.testing.assert_close(build(**inputs), stock)
            ar_backend.kernel.sampling_backend = "flashinfer"
            info.need_min_p_sampling = True
            torch.testing.assert_close(build(**inputs), stock)
            raise RuntimeError("rollback sentinel")
        for module in (dflash_utils, dspark_accept, spec_sampling_mask):
            self.assertIs(module.build_speculative_verify_target_probs, original)


if __name__ == "__main__":
    unittest.main()
