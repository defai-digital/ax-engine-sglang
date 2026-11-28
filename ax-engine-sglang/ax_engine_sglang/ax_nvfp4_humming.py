"""Run the SM90 NVFP4 linears with Humming instead of SGLang's Marlin fallback (AX_NVFP4_HUMMING=1).

On Hopper SGLang serves ModelOpt NVFP4 weights with Marlin W4A16; vLLM serves the same checkpoints
with humming-kernels (TMA + WGMMA, HummingNvFp4LinearKernel). On an H200, with the Qwen3.8-27B
gate_up, down and lm_head weights inside CUDA graphs, Humming took 0.89-0.96x Marlin's time at 1-8
rows and 0.59-0.65x at 128 rows (0.89-0.90x on gate_up and lm_head at 32 rows was the exception:
1.08-1.11x). In an eight-client profile Marlin took 356 ms of a 768 ms span and Humming 239 ms of
vLLM's 691 ms: the NVFP4 GEMM alone exceeded the whole gap between the engines.

The weights are converted at load time as vLLM converts them: per-shard weight_scale_2 (Marlin
applies their maximum to every shard), 16-bit activations, N padded to 256 and K to 128. The
converted layer keeps int32 packed weights and the attributes of SGLang's Marlin layer, so the
target and DFlash lm_head gates (should_apply_lm_head_quant_method) admit it unchanged.
Outputs are not bitwise equal to Marlin's; both stay within the FP32 dequantized reference.
"""

from functools import wraps
import json
import sys

from .compat import check_sglang, patch_attribute
from .config import get_config

FEATURE = "AX_NVFP4_HUMMING"
# vLLM's get_humming_linear_compute_config() without batch invariance or FP16 accumulation.
COMPUTE_CONFIG = json.dumps({"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "dense"})
_installed = False


class HummingOps:
    """The humming-kernels entry points, imported once (and replaced in the CPU tests)."""

    def __init__(self):
        import torch
        from humming.forward import humming_forward
        from humming.schema.humming import HummingInputSchema
        from humming.schema.modelopt import ModeloptNvfp4WeightSchema
        from humming.transform import prepare_and_transform_humming_tensors

        self.torch = torch
        self.forward = humming_forward
        self.weight_schema = ModeloptNvfp4WeightSchema
        self.input_schema = lambda dtype: HummingInputSchema().to_humming_schema(dtype)
        self.transform = prepare_and_transform_humming_tensors

    def parameter(self, tensor):
        return self.torch.nn.Parameter(tensor, requires_grad=False)

    def locks(self, device):
        count = self.torch.cuda.get_device_properties(device).multi_processor_count * 8
        return self.torch.zeros(count, dtype=self.torch.int32, device=device)


def convert(layer, ops) -> None:
    """Replace a loaded ModelOpt NVFP4 layer's weights with Humming's layout, in place."""
    weight = layer.weight.data
    n, k = int(weight.shape[0]), int(layer.input_size_per_partition)
    widths = [int(width) for width in (getattr(layer, "logical_widths", None) or [n])]
    if sum(widths) != n or int(weight.shape[1]) * 2 != k:
        raise RuntimeError(f"{FEATURE}: weight {tuple(weight.shape)} does not match N={widths} K={k}")
    if layer.quant_config.group_size != 16:
        raise RuntimeError(f"{FEATURE} requires group_size=16, got {layer.quant_config.group_size}")
    scale_2 = layer.weight_scale_2.data.float().reshape(-1)
    if scale_2.numel() != len(widths):
        raise RuntimeError(f"{FEATURE}: {scale_2.numel()} weight_scale_2 values for {len(widths)} shards")
    dtype = layer.params_dtype
    schema, tensors = ops.weight_schema().convert_humming(
        tensors={"weight": weight, "weight_scale": layer.weight_scale.data, "weight_scale_2": scale_2},
        shape_n_stacks=widths, shape_k_stacks=[k], param_dtype=dtype)
    config, tensors = ops.transform(
        tensors=tensors, shape_n=n, shape_k=k, weight_schema=schema, input_schema=ops.input_schema(dtype),
        pad_n_to_multiple=256, pad_k_to_multiple=128, has_bias=False, torch_dtype=dtype, device=weight.device)
    layer.weight = ops.parameter(tensors["weight"])
    layer.weight_scale = ops.parameter(tensors["weight_scale"])
    layer.weight_scale_2 = ops.parameter(tensors["weight_scale_2"])
    # The Marlin attributes the lm_head gates look for; Humming reads neither.
    layer.weight_global_scale = ops.parameter(scale_2.max())
    layer.workspace = ops.locks(weight.device)
    layer.output_size_per_partition = n
    layer.weights_padding_cols = 0
    layer.ax_humming = config


def _process(original, backend, ops):
    @wraps(original)
    def process_weights_after_loading(self, layer):
        if self.quant_mode != "w4a4" or not backend().is_marlin():
            return original(self, layer)
        convert(layer, ops())
        return None

    return process_weights_after_loading


def _apply(original, ops):
    @wraps(original)
    def apply(self, layer, x, bias=None):
        config = getattr(layer, "ax_humming", None)
        if config is None:
            return original(self, layer, x, bias)
        out = ops().forward(config, inputs=x.reshape(-1, x.shape[-1]), weight=layer.weight,
                            weight_scale=layer.weight_scale, weight_scale_2=layer.weight_scale_2,
                            locks=layer.workspace, compute_config=COMPUTE_CONFIG)
        n = layer.output_size_per_partition
        if out.shape[-1] != n:
            out = out[:, :n]
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], n)

    return apply


def _cached(factory):
    made = []

    def get():
        if not made:
            made.append(factory())
        return made[0]

    return get


def install():
    global _installed
    if _installed:
        return
    if check_sglang(get_config()).split("+")[0] != "0.5.23":
        raise RuntimeError(f"{FEATURE} requires SGLang 0.5.23")
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError(f"{FEATURE} requires an SM90 GPU")
    try:
        import humming  # noqa: F401
    except ImportError as error:
        raise RuntimeError(f"{FEATURE} requires humming-kernels (pip install humming-kernels==0.1.20)") from error
    from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4LinearMethod

    ops = _cached(HummingOps)
    patch_attribute(ModelOptFp4LinearMethod, "process_weights_after_loading",
                    _process(ModelOptFp4LinearMethod.process_weights_after_loading, get_fp4_gemm_runner_backend, ops))
    patch_attribute(ModelOptFp4LinearMethod, "apply", _apply(ModelOptFp4LinearMethod.apply, ops))
    patch_attribute(sys.modules[__name__], "_installed", True)
    print("ax-engine-sglang: Humming runs the NVFP4 linears in place of Marlin (AX_NVFP4_HUMMING=1)", flush=True)
