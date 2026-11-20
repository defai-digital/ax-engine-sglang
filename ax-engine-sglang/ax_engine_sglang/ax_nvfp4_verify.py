"""Give dense Qwen NVFP4 decode and target verification the same projection arithmetic.

SGLang selects a different FP8 or NVFP4 kernel for each row count. With one row
per call, verification repeats the kernels of single-request AR. With a wider
group, AR decode and verification both pad to that row count, so each call
selects the same kernel and reads the weights once per group.
"""
from contextvars import ContextVar
from functools import wraps

from .compat import patch_attribute


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


def install(rows=16):
    import torch
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.layers.logits_processor import LogitsProcessor
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
            try:
                import cutlass.cute  # SGLang can silently select different numerics without the DSL.
            except ImportError as error:
                raise RuntimeError("AX_NVFP4_VERIFY_PARITY requires an importable NVIDIA CUTLASS DSL "
                                   "(cutlass.cute) in the serving environment") from error
        invariant(self)
        enable_batch_invariant_mode()

    patch_attribute(ModelRunner, "maybe_enable_batch_invariant_mode", enable_invariant)
    # One row per call leaves AR decode on its native single-row kernels.
    # A wider group has to cover decode as well, or AR would select other kernels.
    regrouping = ContextVar("ax_nvfp4_regroup_rows", default=False)
    forward = Qwen3_5ForConditionalGeneration.forward

    @wraps(forward)
    def forward_target(self, input_ids, positions, forward_batch, *args, **kwargs):
        mode = forward_batch.forward_mode
        scope = regrouping.set(mode.is_target_verify() or (rows > 1 and mode.is_decode()))
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

    compute_head = LogitsProcessor._compute_lm_head

    @wraps(compute_head)
    def project_head(self, hidden_states, lm_head, embedding_bias=None, *, out=None):
        if regrouping.get() and hidden_states.ndim == 2 and (rows > 1 or hidden_states.shape[0] > 1):
            logits = grouped(lambda group: compute_head(self, group, lm_head, embedding_bias),
                             hidden_states, rows)
            if out is not None and out.shape == logits.shape and out.dtype == logits.dtype:
                return out.copy_(logits)
            return logits
        return compute_head(self, hidden_states, lm_head, embedding_bias, out=out)

    patch_attribute(LogitsProcessor, "_compute_lm_head", project_head)
