"""Exercise the real SM120 ModelOpt FP8 dispatch inside a target verification scope."""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from ax_engine_sglang import ax_nvfp4_verify
from ax_engine_sglang.compat import installation


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
                     "requires the SM120 FP8 GEMV and skinny GEMM dispatchers")
class VerifyProjectionTests(unittest.TestCase):
    def test_missing_cutlass_dsl_rejects_the_reference_profile(self):
        config = SimpleNamespace(quantization="modelopt_mixed",
                                 hf_config=SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"]))
        runner = SimpleNamespace(is_draft_worker=False, device=torch.device("cuda:0"), model_config=config)
        with self.assertRaisesRegex(RuntimeError, "requires an importable NVIDIA CUTLASS DSL"), installation():
            ax_nvfp4_verify.install()
            with patch.dict(sys.modules, {"cutlass.cute": None}):
                ModelRunner.maybe_enable_batch_invariant_mode(runner)

    def test_eight_fp8_projections_match_singleton_decode(self):
        generator = torch.Generator(device="cuda").manual_seed(4109)
        x = torch.randn(8, 5120, dtype=torch.bfloat16, device="cuda", generator=generator)
        weight = torch.randn(8192, 5120, dtype=torch.bfloat16, device="cuda", generator=generator)
        layer = SimpleNamespace(weight=weight.to(torch.float8_e4m3fn).T,
                                input_scale=torch.tensor([0.02], device="cuda"),
                                sm120_fp8_alpha=torch.tensor([0.02], device="cuda"))
        method = object.__new__(ModelOptFp8LinearMethod)
        method.use_marlin = False
        method.use_sm120_fp8 = True
        native = ModelOptFp8LinearMethod.apply
        expected = torch.cat([native(method, layer, row) for row in x.split(1)])
        rows = []

        def project(self, layer, inputs, *args, **kwargs):
            rows.append(inputs.shape[0])
            return native(self, layer, inputs, *args, **kwargs)

        def forward(self, inputs, positions, batch):
            return method.apply(layer, inputs)

        def batch(verify):
            return SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: verify))

        with (patch.object(ModelOptFp8LinearMethod, "apply", project),
              patch.object(Qwen3_5ForConditionalGeneration, "forward", forward),
              self.assertRaisesRegex(RuntimeError, "end patch scope"), installation()):
            ax_nvfp4_verify.install()
            result = Qwen3_5ForConditionalGeneration.forward(None, x, None, batch(True))
            self.assertEqual(rows, [1] * len(x))
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
            rows.clear()
            Qwen3_5ForConditionalGeneration.forward(None, x, None, batch(False))
            self.assertEqual(rows, [len(x)])
            raise RuntimeError("end patch scope")


if __name__ == "__main__":
    unittest.main()
