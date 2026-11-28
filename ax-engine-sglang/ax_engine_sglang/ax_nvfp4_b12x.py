"""Use b12x for the 27B NVFP4 down projection at 128 rows (AX_NVFP4_B12X_DOWN128=1)."""

from functools import wraps
import sys

from .compat import check_sglang, patch_attribute
from .config import get_config

FEATURE = "AX_NVFP4_B12X_DOWN128"
_installed = False


def _route(original, direct, backend):
    validated = False

    @wraps(original)
    def gemm(input, weight, input_sf, weight_sf, alpha, out_dtype, out_features, quant_mode="w4a4"):
        nonlocal validated
        if quant_mode == "w4a4" and input.shape[0] == 128 and weight.shape == (8704, 5120):
            if not validated:
                if not backend().is_flashinfer_cutedsl():
                    raise RuntimeError(f"{FEATURE} requires --fp4-gemm-backend flashinfer_cutedsl")
                validated = True
            return direct(input, weight, input_sf, weight_sf, alpha, out_dtype, backend="b12x")
        return original(input, weight, input_sf, weight_sf, alpha, out_dtype, out_features, quant_mode)

    return gemm


def install():
    global _installed
    if _installed:
        return
    if check_sglang(get_config()).split("+")[0] != "0.5.23":
        raise RuntimeError(f"{FEATURE} requires SGLang 0.5.23")
    from sglang.srt.layers.quantization import modelopt_quant
    from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend

    patch_attribute(modelopt_quant, "fp4_gemm", _route(
        modelopt_quant.fp4_gemm, modelopt_quant.flashinfer_fp4_gemm, get_fp4_gemm_runner_backend))
    patch_attribute(sys.modules[__name__], "_installed", True)
    print("ax-engine-sglang: b12x enabled only for the 27B NVFP4 down projection at 128 rows", flush=True)
