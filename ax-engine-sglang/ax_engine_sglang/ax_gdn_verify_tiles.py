"""Narrower launch tiles for SGLang's Gated-DeltaNet DFlash verify kernels on SM100 (AX_GDN_VERIFY_TILES).

Two Triton kernels of SGLang 0.5.23's fp32-state ReplaySSM verify are launched with tiles sized for
other paths, and on a B200 both are latency-bound at eight requests x eight draft tokens:

- The ring verify (fused_sigmoid_gating_delta_rule_update with cache_ring=True; the CuTe DSL kernel covers
  bf16 states only) runs a 32-wide value tile on one warp: a [128, 32] fp32 state tile per thread block at
  255 registers. A 16-wide tile ran 33 us against 46 us (CUDA graph).
- The verify conv (causal_conv1d_update with intermediate windows) runs 256 channels on four warps and
  writes the deduplicated window view at a 20-byte channel stride. 64 channels on two warps ran 8.6 us
  against 11.8 us.

The conv tile only repartitions channels: outputs, conv states and windows were bitwise equal to the
stock launch, and served greedy outputs did not change. The ring tile writes the same ring entries (the
fold that commits the state reads only those), but it reduces over K in another order, so the verify
outputs differ in fp32 rounding: in a B200 serving retest 5 of 24 one-client greedy outputs changed, with
tokens per verify within 1% (results/b200_gdn_verify). A Qwen3.8-27B eight-client profile spent 1.9 ms of
a 12.4 ms round in the ring verify and 0.5 ms in the verify conv.
"""

from functools import wraps
import sys

from .compat import patch_attribute

FEATURE = "AX_GDN_VERIFY_TILES"
RING_BLOCK_V = 16
CONV_TILE = {"BLOCK_N": 64, "num_warps": 2}
_installed = False


def ring_verify_call(kwargs, torch) -> bool:
    """Whether this fused_sigmoid_gating_delta_rule_update call is a GDN fp32-state ring verify."""
    state = kwargs.get("initial_state_source")
    v = kwargs.get("v")
    return bool(kwargs.get("cache_ring") and not kwargs.get("is_kda") and not kwargs.get("match_cutedsl_decode")
                and kwargs.get("intermediate_states_buffer") is None and state is not None
                and state.dtype == torch.float32 and v is not None and v.shape[-1] >= RING_BLOCK_V)


def _wrap_ring(original, module, torch):
    @wraps(original)
    def fused_sigmoid_gating_delta_rule_update(*args, **kwargs):
        if args or not ring_verify_call(kwargs, torch):
            return original(*args, **kwargs)
        select = module._select_recurrent_launch_config
        module._select_recurrent_launch_config = lambda *a, **k: (RING_BLOCK_V, 1)
        try:
            return original(**kwargs)
        finally:
            module._select_recurrent_launch_config = select

    return fused_sigmoid_gating_delta_rule_update


class ConvUpdateKernel:
    """Stands in for _causal_conv1d_update_kernel: launches that save intermediate windows take CONV_TILE."""

    def __init__(self, kernel, tile=CONV_TILE):
        self.kernel, self.tile = kernel, tile

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            if kwargs.get("SAVE_INTERMEDIATE"):
                kwargs = {**kwargs, **self.tile}
            return launch(*args, **kwargs)

        return run

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def install():
    global _installed
    if _installed:
        return
    import torch
    from sglang.kernels.ops.attention.linear import fused_sigmoid_gating_recurrent as recurrent
    from sglang.kernels.ops.mamba import causal_conv1d_triton as conv

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError(f"{FEATURE} is measured on SM100 only")
    patch_attribute(recurrent, "fused_sigmoid_gating_delta_rule_update",
                    _wrap_ring(recurrent.fused_sigmoid_gating_delta_rule_update, recurrent, torch))
    patch_attribute(conv, "_causal_conv1d_update_kernel", ConvUpdateKernel(conv._causal_conv1d_update_kernel))
    patch_attribute(sys.modules[__name__], "_installed", True)
    print(f"ax-engine-sglang: GDN verify on a {RING_BLOCK_V}-wide ring tile and a {CONV_TILE['BLOCK_N']}-channel "
          f"conv tile ({FEATURE}=1)", flush=True)
