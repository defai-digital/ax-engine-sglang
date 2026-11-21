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
        runner = SimpleNamespace(is_draft_worker=False, device=torch.device("cuda:0"), model_config=config,
                                 server_args=SimpleNamespace(speculative_algorithm="FC1_DFLASH_TREE"))
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

        def batch(mode):
            return SimpleNamespace(forward_mode=SimpleNamespace(
                is_target_verify=lambda: mode == "verify", is_decode=lambda: mode == "decode"))

        with (patch.object(ModelOptFp8LinearMethod, "apply", project),
              patch.object(Qwen3_5ForConditionalGeneration, "forward", forward),
              self.assertRaisesRegex(RuntimeError, "end patch scope"), installation()):
            ax_nvfp4_verify.install(1)
            for mode in ("verify", "decode"):  # decode: several requests in one AR step
                rows.clear()
                result = Qwen3_5ForConditionalGeneration.forward(None, x, None, batch(mode))
                self.assertEqual(rows, [1] * len(x))
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
            rows.clear()
            Qwen3_5ForConditionalGeneration.forward(None, x, None, batch("prefill"))
            self.assertEqual(rows, [len(x)])
            raise RuntimeError("end patch scope")

    def test_invariant_gemm_gives_a_row_the_same_bits_at_every_row_count(self):
        generator = torch.Generator(device="cuda").manual_seed(4111)
        method = object.__new__(ModelOptFp8LinearMethod)
        method.use_marlin = False
        method.use_sm120_fp8 = True
        method.cutlass_fp8_supported = True
        with self.assertRaisesRegex(RuntimeError, "end patch scope"), installation():
            ax_nvfp4_verify.install(0)
            for inputs, outputs in ((5120, 8192), (5120, 16384), (6144, 5120), (5120, 5120), (5120, 34816)):
                weight = torch.randn(outputs, inputs, dtype=torch.bfloat16, device="cuda", generator=generator)
                layer = SimpleNamespace(weight=weight.to(torch.float8_e4m3fn).T,
                                        input_scale=torch.tensor([0.02], device="cuda"),
                                        sm120_fp8_alpha=torch.tensor([0.02], device="cuda"),
                                        weight_scale=torch.tensor([1.0], device="cuda"),
                                        use_flashinfer_bmm=True)
                x = torch.randn(512, inputs, dtype=torch.bfloat16, device="cuda", generator=generator)
                alone = torch.cat([method.apply(layer, row) for row in x.split(1)])
                for count in (2, 3, 4, 8, 9, 16, 17, 32, 64, 65, 129, 512):
                    with self.subTest(inputs=inputs, outputs=outputs, rows=count):
                        torch.testing.assert_close(method.apply(layer, x[:count]), alone[:count], atol=0, rtol=0)
            raise RuntimeError("end patch scope")

    def test_invariant_gemm_accepts_prequantized_inputs_in_a_cuda_graph(self):
        from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

        generator = torch.Generator(device="cuda").manual_seed(4112)
        x = torch.randn(8, 2, 5120, dtype=torch.bfloat16, device="cuda", generator=generator).transpose(0, 1)
        weight = torch.randn(8192, 5120, dtype=torch.bfloat16, device="cuda", generator=generator)
        bias = torch.randn(8192, dtype=torch.bfloat16, device="cuda", generator=generator)
        scale = torch.tensor([0.02], device="cuda")
        layer = SimpleNamespace(weight=weight.to(torch.float8_e4m3fn).T,
                                input_scale=scale, weight_scale=torch.ones_like(scale))
        method = SimpleNamespace(use_marlin=False, cutlass_fp8_supported=True)
        quantized, _ = static_quant_fp8(x.reshape(-1, x.shape[-1]), scale)
        quantized = quantized.reshape(x.shape)

        with self.assertRaisesRegex(RuntimeError, "end patch scope"), installation():
            ax_nvfp4_verify.install(0)
            apply = ModelOptFp8LinearMethod.apply
            expected = apply(method, layer, x, bias)
            torch.testing.assert_close(apply(method, layer, quantized, bias), expected, atol=0, rtol=0)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = apply(method, layer, quantized, bias)
            graph.replay()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            raise RuntimeError("end patch scope")

    def test_fixed_row_groups_match_between_decode_and_verify(self):
        # One group size must give a row the same bits beside zero padding (AR decode)
        # and beside other candidates (verification), at every position in the group.
        generator = torch.Generator(device="cuda").manual_seed(4110)
        native = ModelOptFp8LinearMethod.apply
        method = object.__new__(ModelOptFp8LinearMethod)
        method.use_marlin = False
        method.use_sm120_fp8 = True
        method.cutlass_fp8_supported = True
        calls = []

        def project(self, layer, inputs, *args, **kwargs):
            calls.append(inputs.shape[0])
            return native(self, layer, inputs, *args, **kwargs)

        def batch(mode):
            return SimpleNamespace(forward_mode=SimpleNamespace(
                is_target_verify=lambda: mode == "verify", is_decode=lambda: mode == "decode"))

        # Qwen3.8-27B FP8 projection shapes as (inputs, outputs).
        for inputs, outputs in ((5120, 8192), (5120, 16384), (6144, 5120), (5120, 5120)):
            weight = torch.randn(outputs, inputs, dtype=torch.bfloat16, device="cuda", generator=generator)
            layer = SimpleNamespace(weight=weight.to(torch.float8_e4m3fn).T,
                                    input_scale=torch.tensor([0.02], device="cuda"),
                                    sm120_fp8_alpha=torch.tensor([0.02], device="cuda"),
                                    # Sixteen rows leave the SM120 table for the generic FP8 path.
                                    weight_scale=torch.tensor([1.0], device="cuda"),
                                    use_flashinfer_bmm=False)

            def forward(self, x, positions, batch):
                return method.apply(layer, x)

            for rows in (16, 8, 4):
                for width in (rows, 16, 11, 64):
                    x = torch.randn(width, inputs, dtype=torch.bfloat16, device="cuda", generator=generator)
                    with (self.subTest(shape=(inputs, outputs), rows=rows, width=width),
                          patch.object(ModelOptFp8LinearMethod, "apply", project),
                          patch.object(Qwen3_5ForConditionalGeneration, "forward", forward),
                          self.assertRaisesRegex(RuntimeError, "end patch scope"), installation()):
                        ax_nvfp4_verify.install(rows)
                        calls.clear()
                        verified = Qwen3_5ForConditionalGeneration.forward(None, x, None, batch("verify"))
                        self.assertEqual(calls, [rows] * -(-width // rows))
                        decoded = torch.cat([
                            Qwen3_5ForConditionalGeneration.forward(None, row.contiguous(), None, batch("decode"))
                            for row in x.split(1)])
                        self.assertEqual(set(calls), {rows})
                        torch.testing.assert_close(verified, decoded, atol=0, rtol=0)
                        raise RuntimeError("end patch scope")


if __name__ == "__main__":
    unittest.main()
