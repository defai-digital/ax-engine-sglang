"""SM110 (Jetson Thor) / SM120 Flash Next NVFP4 MoE backend preflight, without checkpoint weights.

    python ax-engine-sglang/tests/gpu_nvfp4_moe_check.py

Uses SGLang's real weight preparation and FlashInfer's actual CUTLASS kernel at
the model's hidden/intermediate dimensions. This is a backend smoke control,
not a model quality or distribution qualification.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from sglang.srt.layers import moe
from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization import modelopt_quant


class Nvfp4MoeTests(unittest.TestCase):
    def test_cutlass_weight_preparation_execution_and_graph_replay(self):
        # SGLang routes NVFP4 MoE on SM110 (Thor) and SM120/121 through FlashInfer CUTLASS
        # (is_blackwell, not is_sm100); SM100 uses its own kernels and is qualified on B200.
        major = torch.cuda.get_device_capability()[0]
        if major not in (11, 12):
            self.skipTest("SM110/SM120/121 development control")
        backend = MoeRunnerBackend.FLASHINFER_CUTLASS
        layer = torch.nn.Module()
        layer.num_local_experts = layer.num_experts = 4
        layer.moe_ep_rank = 0
        layer._mega_moe_nvfp4 = False
        layer.moe_runner_config = MoeRunnerConfig(
            num_experts=4, num_local_experts=4, hidden_size=2560,
            intermediate_size_per_partition=640, top_k=2)
        layer.dispatcher = SimpleNamespace(set_quant_config=lambda value: None)
        # Supply the backend context without starting a distributed model server.
        # All weight transforms, scale calculations and GEMMs below are real.
        with patch.object(modelopt_quant, "get_moe_runner_backend", return_value=backend), \
                patch.object(moe, "get_moe_runner_backend", return_value=backend), torch.device("cuda"):
            config = modelopt_quant.ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=16,
                use_per_token_activation=False)
            method = modelopt_quant.ModelOptNvFp4FusedMoEMethod(config)
            method.moe_runner_config = layer.moe_runner_config
            method._moe_runner_backend = backend
            method.create_weights(layer, 4, 2560, 640, torch.bfloat16)
            gate = torch.tensor(2.5)
            activated = torch.nn.functional.silu(gate) * gate
            with torch.no_grad():
                for name, value in layer.named_parameters():
                    if value.dtype == torch.uint8:
                        value.fill_(0x22)  # two FP4 values of 1.0
                    else:
                        value.fill_(0.0625 if name in ("w13_weight_scale", "w2_weight_scale") else 1)
                # Calibrate constant activations to FP4 * FP8 maximum (6 * 448).
                # A scale of 1 makes these small values subnormal in the block SF.
                layer.w13_input_scale.fill_(0.015625 / (448 * 6))
                layer.w2_input_scale.fill_(activated.item() / (448 * 6))
            method.process_weights_after_loading(layer)
            method.runner = SimpleNamespace(run=lambda dispatch, payload: payload)
            quant = method.apply(layer, None)
            x = torch.full((8, 2560), 0.015625, dtype=torch.bfloat16)
            ids = torch.tensor([[0, 1]] * 8, dtype=torch.int32)
            scores = torch.full((8, 2), 0.5, dtype=torch.float32)

            def forward():
                # Match the actual SGLang runner's packed-weight/scale ABI.
                scales = [value.view(torch.int32) if i in (1, 4) else value
                          for i, value in enumerate(quant.quant_scales)]
                return cutlass_fused_moe(
                    input=x, token_selected_experts=ids, token_final_scales=scores,
                    fc1_expert_weights=quant.w13_weight.view(torch.long),
                    fc2_expert_weights=quant.w2_weight.view(torch.long),
                    output_dtype=torch.bfloat16, quant_scales=scales,
                    tune_max_num_tokens=8)[0]

            result = forward()
            torch.cuda.synchronize()
            expected = (activated * 640 * 0.0625).item()
            error = (result.float() - expected).abs().max().item() / expected
            self.assertTrue(bool(torch.isfinite(result).all()))
            self.assertLess(error, 0.05)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = forward()
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(result, replayed))
            x.zero_()
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(int(torch.count_nonzero(replayed)), 0)
            print(f"NVFP4 SM{major}x CUTLASS maximum relative error: {error:.8f}", flush=True)


if __name__ == "__main__":
    unittest.main()
