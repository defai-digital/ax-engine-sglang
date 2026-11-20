"""Use single-token projection arithmetic during dense Qwen NVFP4 target verification."""
from contextvars import ContextVar
from functools import wraps

from .compat import patch_attribute


def install():
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
    verifying = ContextVar("ax_nvfp4_target_verify", default=False)
    forward = Qwen3_5ForConditionalGeneration.forward

    @wraps(forward)
    def forward_target(self, input_ids, positions, forward_batch, *args, **kwargs):
        scope = verifying.set(forward_batch.forward_mode.is_target_verify())
        try:
            return forward(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            verifying.reset(scope)

    def wrap_linear(apply):
        @wraps(apply)
        def project(self, layer, x, *args, **kwargs):
            if verifying.get() and isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] > 1:
                return torch.cat([apply(self, layer, row.contiguous(), *args, **kwargs)
                                  for row in x.split(1)])
            return apply(self, layer, x, *args, **kwargs)
        return project

    patch_attribute(Qwen3_5ForConditionalGeneration, "forward", forward_target)
    for method in (ModelOptFp4LinearMethod, ModelOptFp8LinearMethod, UnquantizedLinearMethod):
        patch_attribute(method, "apply", wrap_linear(method.apply))

    compute_head = LogitsProcessor._compute_lm_head

    @wraps(compute_head)
    def project_head(self, hidden_states, lm_head, embedding_bias=None, *, out=None):
        if verifying.get() and hidden_states.shape[0] > 1:
            rows = [compute_head(self, row, lm_head, embedding_bias) for row in hidden_states.split(1)]
            logits = torch.cat(rows)
            if out is not None and out.shape == logits.shape and out.dtype == logits.dtype:
                return out.copy_(logits)
            return logits
        return compute_head(self, hidden_states, lm_head, embedding_bias, out=out)

    patch_attribute(LogitsProcessor, "_compute_lm_head", project_head)
