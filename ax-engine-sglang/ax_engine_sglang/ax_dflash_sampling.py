"""Match the FlashInfer AR target distribution in DFlash2 chain verification."""
from functools import wraps

from .ax_eagle_sampling import joint_probs
from .compat import patch_attribute


def install():
    import torch
    from sglang.kernels.ops.speculative.dspark import dspark_accept
    from sglang.srt.runtime_context import get_exec, get_spec
    from sglang.srt.speculative import dflash_utils, spec_sampling_mask

    original = dflash_utils.build_speculative_verify_target_probs
    if getattr(original, "_ax_joint_sampling", False):
        return

    @wraps(original)
    def build(*, next_token_logits, sampling_info, draft_token_num, bs, **kwargs):
        # The draft proposal q is left intact. Rejection and support capture
        # both need the same AR target p, including when the target has ties.
        if (next_token_logits.is_cuda
                and sampling_info.need_top_k_sampling and sampling_info.need_top_p_sampling
                and not sampling_info.need_min_p_sampling
                and get_spec().speculative_algorithm == "FC1_DFLASH_TREE"
                and get_exec().kernel.sampling_backend == "flashinfer"):
            def repeat(values):
                return torch.repeat_interleave(values, draft_token_num, dim=0)

            probs = torch.softmax(next_token_logits / repeat(sampling_info.temperatures), dim=-1)
            filtered = joint_probs(
                probs, repeat(sampling_info.top_ks), repeat(sampling_info.top_ps),
                dflash_utils._dflash_top_k_renorm_prob, dflash_utils._dflash_top_p_renorm_prob,
            )
            return filtered.view(bs, draft_token_num, -1).contiguous()
        return original(next_token_logits=next_token_logits, sampling_info=sampling_info,
                        draft_token_num=draft_token_num, bs=bs, **kwargs)

    build._ax_joint_sampling = True
    for module in (dflash_utils, dspark_accept, spec_sampling_mask):
        if module.build_speculative_verify_target_probs is not original:
            raise RuntimeError(f"ax-engine-sglang: unexpected target filter in {module.__name__}")
        patch_attribute(module, "build_speculative_verify_target_probs", build)
