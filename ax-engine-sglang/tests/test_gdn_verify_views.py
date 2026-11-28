"""AX_GDN_VERIFY_VIEWS: projection views only for the fp32 GDN ReplaySSM fold; edits compile, offline."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_gdn_verify_views import MASK_EDIT, VIEW_EDIT, _verify_views, views_supported
from ax_engine_sglang.compat import rebuild_function
from ax_engine_sglang.config import PluginConfig

torch = SimpleNamespace(float32="fp32", bfloat16="bf16")


def backend(dtype="fp32", fold=True, kda=False, rawv=object()):
    cache = SimpleNamespace(replayssm_rawv=rawv, temporal=SimpleNamespace(dtype=dtype))
    pool = SimpleNamespace(mamba_cache=cache, replayssm_spec_fold=fold, replayssm_is_kda=kda)
    return SimpleNamespace(req_to_token_pool=SimpleNamespace(mamba_pool=pool))


def batch(verify):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: verify))


class Source:
    def forward(self, forward_batch, cache_indices, query_start_loc):
        use_strided_prefill_z = (
                True
                and forward_batch.logical_forward_mode.is_extend_without_speculative()
            )
        valid = select(
                cache_indices[: query_start_loc.shape[0] - 1] >= 0,
        )
        return use_strided_prefill_z, valid


class Views(unittest.TestCase):
    def test_only_the_fp32_gdn_fold(self):
        self.assertTrue(views_supported(backend(), torch))
        for other in (backend("bf16"), backend(fold=False), backend(kda=True), backend(rawv=None), SimpleNamespace()):
            self.assertFalse(views_supported(other, torch))

    def test_only_on_target_verify_and_through_the_hybrid_wrapper(self):
        hybrid = SimpleNamespace(linear_attn_backend=backend())
        check = _verify_views(lambda: hybrid, torch)
        self.assertTrue(check(batch(True)))
        self.assertFalse(check(batch(False)))
        self.assertFalse(_verify_views(lambda: SimpleNamespace(linear_attn_backend=backend("bf16")), torch)(batch(True)))

    def test_edits_compile_and_keep_the_mask_lazy(self):
        forward = rebuild_function(Source, "forward", (VIEW_EDIT, MASK_EDIT), feature="test")
        globals_ = forward.__globals__
        calls = []
        globals_.update(select=lambda valid: valid, pp_spec_stable_rows_enabled=lambda: False,
                        _ax_gdn_verify_views=lambda fb: calls.append(fb) or True)
        try:
            fb = SimpleNamespace(logical_forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False))
            self.assertEqual(forward(None, fb, None, None), (True, None))
            self.assertEqual(calls, [fb])
        finally:
            for name in ("select", "pp_spec_stable_rows_enabled", "_ax_gdn_verify_views"):
                globals_.pop(name)

    def test_flag(self):
        self.assertTrue(PluginConfig.from_env({"AX_GDN_VERIFY_VIEWS": "1"}).gdn_verify_views)
        self.assertFalse(PluginConfig.from_env({}).gdn_verify_views)


if __name__ == "__main__":
    unittest.main()
