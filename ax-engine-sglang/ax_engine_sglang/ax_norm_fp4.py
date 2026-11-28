"""Fuse the Qwen3.5 dense MLP's NVFP4 input quantization into the post-attention add-RMSNorm (AX_NORM_FP4).

SGLang 0.5.23 runs FlashInfer's Gemma fused add-RMSNorm and then a separate FP4 quantize kernel at the
head of gate_up_proj, in every layer of every forward. vLLM fuses the two (models/qwen3_5/nvidia/ops/
fused_norm_quant.py). The Triton kernel below is that kernel's NVFP4 epilogue (vLLM, Apache-2.0): the
residual add in fp32, the norm on the fp32 sum, the normed row rounded to bf16, then scaled_fp4_quant's
scale arithmetic on the rounded row, block scales written in the swizzled 128x4 layout.

The norm still returns its bf16 output, so every layer-boundary path in SGLang sees what it saw before.
The fused (fp4, scale) pair is left on gate_up_proj next to that output; gate_up_proj takes it only
when it is handed that very tensor, and otherwise quantizes as before.
"""

from functools import wraps
import sys

from .compat import patch_attribute

FEATURE = "AX_NORM_FP4"
STASH = "_ax_norm_fp4"
_installed = False

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU test environments
    triton = tl = None


if triton is not None:
    @triton.jit
    def _rcp_approx_ftz(x):
        return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def _cvt_e2m1x2(lo, hi):
        return tl.inline_asm_elementwise(
            """
            {
            .reg .b8 b0, b1, b2, b3;
            cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4;
            mov.b32 $0, {b0, b1, b2, b3};
            }
            """,
            "=r,r,r,r,r,r,r,r,r", [lo, hi], dtype=tl.uint8, is_pure=True, pack=4)

    @triton.jit
    def _gemma_add_rms_norm_fp4_kernel(x_ptr, res_ptr, w_ptr, scale_ptr, q_ptr, sf_ptr, y_ptr, eps,
                                       H: tl.constexpr, G: tl.constexpr):
        # The row as [G, 8, 2]: G NVFP4 blocks of 16 (G = next_power_of_2(H) / 16), each 8 (lo, hi) pairs.
        row = tl.program_id(0).to(tl.int64)
        g = tl.arange(0, G)
        pair = tl.arange(0, 8)
        cols = g[:, None, None] * 16 + pair[None, :, None] * 2 + tl.arange(0, 2)[None, None, :]
        mask = cols < H
        x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0)
        r = tl.load(res_ptr + row * H + cols, mask=mask, other=0.0)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        scale = tl.load(scale_ptr)
        h = x.to(tl.float32) + r.to(tl.float32)
        tl.store(res_ptr + row * H + cols, h, mask=mask)
        rstd = tl.rsqrt(tl.sum(h * h) / H + eps)
        y = (h * rstd * (1.0 + w.to(tl.float32))).to(x_ptr.dtype.element_ty)
        tl.store(y_ptr + row * H + cols, y, mask=mask)
        y32 = y.to(tl.float32)
        vmax = tl.max(tl.max(tl.abs(y32), axis=2), axis=1)
        sf = scale * (vmax * _rcp_approx_ftz(tl.full((G,), 6.0, tl.float32)))
        sf_fp8 = tl.minimum(sf, 448.0).to(tl.float8e4nv)
        sf32 = sf_fp8.to(tl.float32)
        out_scale = tl.where(sf32 != 0.0, _rcp_approx_ftz(sf32 * _rcp_approx_ftz(scale)), 0.0)
        lo, hi = tl.split(y32 * out_scale[:, None, None])
        pcols = g[:, None] * 8 + pair[None, :]
        tl.store(q_ptr + row * (H // 2) + pcols, _cvt_e2m1x2(lo, hi), mask=pcols < H // 2)
        num_k_tiles: tl.constexpr = (H // 16 + 3) // 4
        sf_off = ((row >> 7) * num_k_tiles + (g >> 2)) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (g & 3)
        tl.store(sf_ptr + sf_off, sf_fp8, mask=g < H // 16)


def gemma_add_rms_norm_fp4(x, residual, weight, eps, scale):
    """Gemma RMSNorm of x + residual (residual updated in place) and its NVFP4 quantization.

    Returns (y, q, sf): the bf16 normed rows, the packed FP4 rows (uint8, [M, H/2]) and the block scales
    (uint8, [round_up(M, 128), H/16] in the swizzled 128x4 layout; rows >= M are not written)."""
    import torch

    m, h = x.shape
    q = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty(((m + 127) // 128 * 128, h // 16 // 4), dtype=torch.int32, device=x.device).view(torch.float8_e4m3fn)
    y = torch.empty_like(x)
    # 16 warps up to 128 rows and 4 above: vLLM's sweep at H=5120 on GB300.
    _gemma_add_rms_norm_fp4_kernel[(m,)](x, residual, weight, scale, q, sf, y, eps, H=h, G=triton.next_power_of_2(h) // 16,
                                         num_warps=16 if m <= 128 else 4)
    return y, q, sf.view(torch.uint8)


def eligible(x, residual, post_residual_addition, weight, scale) -> bool:
    import torch

    return (scale is not None and residual is not None and post_residual_addition is None
            and x.dim() == 2 and x.shape[0] > 0 and x.shape[1] % 64 == 0
            and x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
            and residual.shape == x.shape and x.is_contiguous() and residual.is_contiguous()
            and weight.is_contiguous() and weight.shape == (x.shape[1],)
            and scale.numel() == 1 and scale.dtype == torch.float32)


def _norm_forward(norm, linear, fused=gemma_add_rms_norm_fp4, check=eligible):
    original = norm.forward

    def forward(x, residual=None, post_residual_addition=None):
        scale = getattr(linear, "input_scale_inv", None)
        if getattr(linear, "ax_humming", None) is not None or not check(
                x, residual, post_residual_addition, norm.weight, scale):
            linear.__dict__[STASH] = None
            return original(x, residual, post_residual_addition)
        y, q, sf = fused(x, residual, norm.weight.data, norm.variance_epsilon, scale)
        linear.__dict__[STASH] = (y, q, sf)
        return y, residual

    return forward


def _apply(original, is_marlin):
    @wraps(original)
    def apply(self, layer, x, bias=None):
        stash = layer.__dict__.get(STASH)
        if stash is None:
            return original(self, layer, x, bias)
        layer.__dict__[STASH] = None
        if stash[0] is not x or self.quant_mode != "w4a4" or is_marlin():
            return original(self, layer, x, bias)
        return original(self, layer, (stash[1], stash[2]), bias)

    return apply


def attach(layer, fp4_method) -> bool:
    """Fuse one decoder layer's post-attention norm with its gate_up_proj input quantization."""
    mlp = getattr(layer, "mlp", None)
    linear = getattr(mlp, "gate_up_proj", None)
    method = getattr(linear, "quant_method", None)
    if not (isinstance(method, fp4_method) and method.quant_mode == "w4a4"
            and not getattr(method.quant_config, "is_awq", False)
            and not getattr(linear, "_interleave_for_swiglu_fusion", False)):
        return False
    norm = layer.post_attention_layernorm
    norm.forward = _norm_forward(norm, linear)
    linear._accepts_prequantized_fp4 = True
    return True


def _layer_init(original, fp4_method, counter):
    @wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        counter[0] += attach(self, fp4_method)   # MoE layers have no gate_up_proj and are skipped

    return __init__


def install():
    global _installed
    if _installed:
        return
    from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4LinearMethod
    from sglang.srt.models import qwen3_5

    is_marlin = lambda: get_fp4_gemm_runner_backend().is_marlin()  # noqa: E731
    counter = [0]
    for cls in (qwen3_5.Qwen3_5LinearDecoderLayer, qwen3_5.Qwen3_5AttentionDecoderLayer):
        patch_attribute(cls, "__init__", _layer_init(cls.__init__, ModelOptFp4LinearMethod, counter))
    patch_attribute(ModelOptFp4LinearMethod, "apply", _apply(ModelOptFp4LinearMethod.apply, is_marlin))
    patch_attribute(sys.modules[__name__], "_installed", True)
    print(f"ax-engine-sglang: post-attention norm fused with the NVFP4 gate_up quantization ({FEATURE}=1)",
          flush=True)
