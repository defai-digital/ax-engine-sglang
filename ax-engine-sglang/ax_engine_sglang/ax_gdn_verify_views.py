"""Feed SGLang's fp32 ReplaySSM DFlash verify from the GDN projection itself (AX_GDN_VERIFY_VIEWS).

Two per-layer launches on SGLang 0.5.23's Qwen3.5 Gated-DeltaNet target verify do no useful work:

- fused_qkvzba_split_reshape_cat_contiguous copies in_proj_qkvz / in_proj_ba into packed mixed_qkv, z, b
  and a. Ordinary prefill already takes strided views of the projection (qwen3_5_gdn_prefill_projection_views),
  and every consumer on the fp32 Triton ReplaySSM verify route reads strides: the verify conv
  (causal_conv1d_update takes all three input strides and allocates a dense output), the ring verify
  (fused_sigmoid_gating_delta_rule_update reads the token strides of a and b) and the gated norm (3-D z
  with token and head strides, as on prefill). 4.7 us per layer in a B200 eight-client profile.
- GDNAttnBackend.forward_extend evaluates `cache_indices[:batch] >= 0` for every layer and passes it to
  select_verify_intermediate_state_indices, which reads it only with pipeline-parallel stable rows. 1.9 us
  per layer.

Views carry the same values as the copies, so the outputs do not change. The views are taken only for a
GDN ReplaySSM fold with an fp32 state (the CuTe DSL fold, bf16, assumes packed inputs).
"""

import sys

from .compat import patch_attribute, rebuild_function

FEATURE = "AX_GDN_VERIFY_VIEWS"
_installed = False

# rebuild_function edits the dedented method source (four spaces less than the file).
VIEW_EDIT = (
    "            and forward_batch.logical_forward_mode.is_extend_without_speculative()\n        )\n",
    "            and (forward_batch.logical_forward_mode.is_extend_without_speculative()\n"
    "                 or _ax_gdn_verify_views(forward_batch))\n        )\n",
    1,
)
MASK_EDIT = (
    "            cache_indices[: query_start_loc.shape[0] - 1] >= 0,\n",
    "            (cache_indices[: query_start_loc.shape[0] - 1] >= 0) if pp_spec_stable_rows_enabled() else None,\n",
    1,
)


def views_supported(backend, torch) -> bool:
    """Whether this linear-attention backend verifies through the fp32-state GDN ReplaySSM fold."""
    pool = getattr(getattr(backend, "req_to_token_pool", None), "mamba_pool", None)
    cache = getattr(pool, "mamba_cache", None)
    temporal = getattr(cache, "temporal", None)
    return bool(cache is not None and getattr(cache, "replayssm_rawv", None) is not None
                and getattr(pool, "replayssm_spec_fold", False) and not getattr(pool, "replayssm_is_kda", False)
                and temporal is not None and temporal.dtype == torch.float32)


def _verify_views(get_attn_backend, torch):
    def _ax_gdn_verify_views(forward_batch) -> bool:
        if not forward_batch.forward_mode.is_target_verify():
            return False
        backend = get_attn_backend()
        return views_supported(getattr(backend, "linear_attn_backend", backend), torch)

    return _ax_gdn_verify_views


def install():
    global _installed
    if _installed:
        return
    import torch
    from sglang.srt.layers.attention.linear import gdn_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.models import qwen3_5

    forward = rebuild_function(qwen3_5.Qwen3_5GatedDeltaNet, "forward", (VIEW_EDIT,), feature=FEATURE)
    forward_extend = rebuild_function(gdn_backend.GDNAttnBackend, "forward_extend", (MASK_EDIT,), feature=FEATURE)
    patch_attribute(qwen3_5, "_ax_gdn_verify_views", _verify_views(get_attn_backend, torch))
    patch_attribute(qwen3_5.Qwen3_5GatedDeltaNet, "forward", forward)
    patch_attribute(gdn_backend.GDNAttnBackend, "forward_extend", forward_extend)
    patch_attribute(sys.modules[__name__], "_installed", True)
    print(f"ax-engine-sglang: fp32 GDN ReplaySSM verify reads projection views ({FEATURE}=1)", flush=True)
