"""Accepted-prefix GDN replay for the native width-eight DFlash2 worker."""

from functools import wraps
import sys

from .compat import check_sglang, installation, patch_attribute, rebuild_function
from .config import get_config

FEATURE = "AX_GDN_CHAIN_REPLAY"
COMMIT = "_update_target_mamba_state_after_verify"
FOLD = """self.replayssm_spec_fold = bool(
        enable_linear_replayssm_spec and cache_params.is_kda
    )"""
_installed = False


def install():
    if _installed:
        return
    if check_sglang(get_config()).split("+")[0] != "0.5.23":
        raise RuntimeError(f"{FEATURE} requires SGLang 0.5.23")

    import torch

    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold import (
        commit_gdn_replayssm_fold_after_verify,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.mem_cache.memory_pool import MambaPool
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    # Same recurrent verify and fold buffers as the measured SGLang 0.5.23 overlay.
    pool_init = rebuild_function(MambaPool, "__init__", (
        (FOLD, "self.replayssm_spec_fold = enable_linear_replayssm_spec", 1),
        ("if enable_linear_replayssm_spec and not cache_params.is_kda",
         "if enable_linear_replayssm_spec and not self.replayssm_spec_fold", 1),
        ("if enable_linear_replayssm_spec and cache_params.is_kda:\n"
         "                if cache_params.is_kda or not self.replayssm_spec_fold:",
         "if enable_linear_replayssm_spec and self.replayssm_spec_fold:\n"
         "                if cache_params.is_kda:", 1),
    ), feature=FEATURE)
    build_pool = rebuild_function(KVCacheConfigurator, "_build_hybrid_req_pool", (
        ("and self.hybrid_kda_config is None\n    ):",
         'and self.hybrid_kda_config is None\n'
         '        and not (_algo == "DFLASH" and self.hybrid_gdn_config is not None)\n    ):', 1),
    ), feature=FEATURE)

    original = getattr(DFlashWorkerV2, COMMIT)

    @wraps(original)
    def commit(self, *, batch, seq_lens_pre_verify, seq_lens_post_verify, commit_lens):
        pool = self.target_worker.model_runner.req_to_token_pool
        mamba_pool = getattr(pool, "mamba_pool", None)
        if (not getattr(mamba_pool, "replayssm_spec_fold", False)
                or getattr(mamba_pool, "replayssm_is_kda", False)):
            return original(self, batch=batch, seq_lens_pre_verify=seq_lens_pre_verify,
                            seq_lens_post_verify=seq_lens_post_verify, commit_lens=commit_lens)
        if batch.forward_mode.is_idle() or commit_lens.numel() == 0:
            return
        if batch.mamba_track_indices is not None:
            raise RuntimeError(f"{FEATURE} requires radix caching to be disabled")
        # Native DFlash uses the same leading request rows and accepted length minus one.
        # Its sequence-length inputs are only used to locate radix-cache tracking points.
        commit_gdn_replayssm_fold_after_verify(
            spec_state=pool.get_speculative_mamba2_params_all_layers(),
            state_batch_indices=pool.get_mamba_indices(batch.req_pool_indices[:commit_lens.shape[0]]),
            accept_lens=commit_lens,
            last_correct_step_indices=commit_lens.to(torch.int64) - 1,
            mamba_track_indices=None,
            mamba_steps_to_track=None,
            src_indices_raw=None,
            null_block_id=-1,
        )

    with installation():
        patch_attribute(MambaPool, "__init__", pool_init)
        patch_attribute(KVCacheConfigurator, "_build_hybrid_req_pool", build_pool)
        patch_attribute(DFlashWorkerV2, COMMIT, commit)
        patch_attribute(sys.modules[__name__], "_installed", True)
