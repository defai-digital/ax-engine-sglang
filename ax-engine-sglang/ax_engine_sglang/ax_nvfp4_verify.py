"""Give dense Qwen NVFP4 decode and target verification the same projection arithmetic.

SGLang selects a different FP8 kernel for each row count, and the SM120 small-row
kernels and the FlashInfer BMM give a row different bits at different row counts.
By default (rows 0), the checkpoint's per-tensor FP8 projections use one Triton
tile shape in prefill, decode and verification. The tile fixes the accumulation
order even when cuBLAS would choose a different algorithm for another row count.

A positive row count keeps SGLang's kernel choice instead and pads AR decode and
verification to fixed groups of that many rows. Prefill is then not matched.
"""
from contextvars import ContextVar
import os
from functools import wraps

from .compat import patch_attribute, patch_item
from .defaults import parity_algorithm_error


def grouped(project, x, rows):
    """Apply project to fixed groups of rows, padding the last group with zeros."""
    import torch

    count = x.shape[0]
    if not count:
        return project(x)
    if count % rows:
        x = torch.cat([x, x.new_zeros(-count % rows, *x.shape[1:])])
    if x.shape[0] == rows:
        return project(x.contiguous())[:count]
    return torch.cat([project(group.contiguous()) for group in x.split(rows)])[:count]


def install(rows=0):
    import torch
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4LinearMethod, ModelOptFp8LinearMethod
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models import qwen3_5
    from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    patch_attribute(qwen3_5, "_gdn_decode_fused_proj_conv", False)
    patch_attribute(qwen3_5, "_maybe_enable_silu_fp4_quant_fusion", lambda mlp: None)

    # Keep request RNG and the FlashInfer sampler available for independent cohorts.
    # SGLang's deterministic serving flag also changes both sampling settings.
    invariant = ModelRunner.maybe_enable_batch_invariant_mode

    @wraps(invariant)
    def enable_invariant(self):
        if not self.is_draft_worker:
            config = self.model_config
            if (str(self.device).split(":")[0] != "cuda" or torch.cuda.get_device_capability(self.device)[0] != 12
                    or config.quantization != "modelopt_mixed"
                    or "Qwen3_5ForConditionalGeneration" not in config.hf_config.architectures):
                raise ValueError("AX_NVFP4_VERIFY_PARITY requires a dense Qwen ModelOpt mixed target on SM12x")
            # The launch resolver reports this too; it is skipped with AX_AUTO_DEFAULTS=0.
            error = parity_algorithm_error(getattr(self.server_args, "speculative_algorithm", None))
            if error:
                raise ValueError(error)
            try:
                import cutlass.cute  # SGLang can silently select different numerics without the DSL.
            except ImportError as error:
                raise RuntimeError("AX_NVFP4_VERIFY_PARITY requires an importable NVIDIA CUTLASS DSL "
                                   "(cutlass.cute) in the serving environment") from error
        invariant(self)
        enable_batch_invariant_mode()

    patch_attribute(ModelRunner, "maybe_enable_batch_invariant_mode", enable_invariant)
    # Decode is regrouped as well: AR must select the kernels of verification, also
    # when several requests decode together. A lone row in one-row groups is unchanged.
    regrouping = ContextVar("ax_nvfp4_regroup_rows", default=False)
    forward = Qwen3_5ForConditionalGeneration.forward

    @wraps(forward)
    def forward_target(self, input_ids, positions, forward_batch, *args, **kwargs):
        mode = forward_batch.forward_mode
        scope = regrouping.set(rows > 0 and (mode.is_target_verify() or mode.is_decode()))
        try:
            return forward(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            regrouping.reset(scope)

    def wrap_linear(apply):
        @wraps(apply)
        def project(self, layer, x, *args, **kwargs):
            # A full group still goes through grouped(): the SM120 dispatchers fall
            # back to another kernel for a non-contiguous input.
            if (regrouping.get() and isinstance(x, torch.Tensor) and x.ndim == 2
                    and (rows > 1 or x.shape[0] > 1)):
                return grouped(lambda group: apply(self, layer, group, *args, **kwargs), x, rows)
            return apply(self, layer, x, *args, **kwargs)
        return project

    patch_attribute(Qwen3_5ForConditionalGeneration, "forward", forward_target)
    for method in (ModelOptFp4LinearMethod, ModelOptFp8LinearMethod, UnquantizedLinearMethod):
        patch_attribute(method, "apply", wrap_linear(method.apply))

    if rows == 0:
        from sglang.kernels.ops.gemm.fp8_kernel import triton_scaled_mm
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, static_quant_fp8

        # Keep upstream's generic fallback for other scale layouts and empty inputs.
        patch_item(os.environ, "SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE", "0")
        fp8_apply = ModelOptFp8LinearMethod.apply

        @wraps(fp8_apply)
        def invariant_fp8(self, layer, x, bias=None):
            if self.use_marlin or not self.cutlass_fp8_supported or not isinstance(x, torch.Tensor):
                raise RuntimeError("AX_NVFP4_PARITY_ROWS=0 needs SGLang's generic FP8 GEMM, not Marlin")
            scale = layer.input_scale
            if scale is None or scale.numel() != 1 or scale.ndim >= 2 or layer.weight_scale.numel() != 1 or not x.numel():
                return apply_fp8_linear(input=x, weight=layer.weight, weight_scale=layer.weight_scale,
                                        input_scale=scale, bias=bias, cutlass_fp8_supported=True)

            flat = x.reshape(-1, x.shape[-1]).contiguous()
            if flat.dtype == fp8_dtype:
                # Match apply_fp8_linear's default for prequantized activations.
                quantized, out_dtype = flat, torch.bfloat16
            else:
                quantized, scale = static_quant_fp8(flat, scale)
                out_dtype = flat.dtype
            output = triton_scaled_mm(quantized, layer.weight, scale, layer.weight_scale, out_dtype, bias,
                                      block_size_m=64, block_size_n=128, block_size_k=256, use_heuristic=False)
            return output.view(*x.shape[:-1], layer.weight.shape[1])

        patch_attribute(ModelOptFp8LinearMethod, "apply", invariant_fp8)

    compute_head = LogitsProcessor._compute_lm_head

    @wraps(compute_head)
    def project_head(self, hidden_states, lm_head, embedding_bias=None, *, out=None):
        if regrouping.get() and hidden_states.ndim == 2 and (rows > 1 or hidden_states.shape[0] > 1):
            # rows == 0 never sets the scope: the head keeps its native full-width call.
            logits = grouped(lambda group: compute_head(self, group, lm_head, embedding_bias),
                             hidden_states, rows)
            if out is not None and out.shape == logits.shape and out.dtype == logits.dtype:
                return out.copy_(logits)
            return logits
        return compute_head(self, hidden_states, lm_head, embedding_bias, out=out)

    patch_attribute(LogitsProcessor, "_compute_lm_head", project_head)
