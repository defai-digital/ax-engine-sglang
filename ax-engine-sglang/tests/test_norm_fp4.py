"""AX_NORM_FP4: the post-attention norm hands gate_up_proj its fused NVFP4 input, offline."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_norm_fp4 import STASH, _apply, _norm_forward, attach
from ax_engine_sglang.config import PluginConfig


class Fp4Method:
    def __init__(self, mode="w4a4", awq=False):
        self.quant_mode, self.quant_config = mode, SimpleNamespace(is_awq=awq)


class Norm:
    def __init__(self):
        self.weight = SimpleNamespace(data="w")
        self.variance_epsilon = 1e-6
        self.calls = []

    def forward(self, x, residual=None, post_residual_addition=None):
        self.calls.append((x, residual, post_residual_addition))
        return f"norm({x})", residual


def layer(method=None, **linear):
    gate_up = SimpleNamespace(quant_method=method or Fp4Method(), input_scale_inv="gs", **linear)
    return SimpleNamespace(mlp=SimpleNamespace(gate_up_proj=gate_up), post_attention_layernorm=Norm())


class NormForward(unittest.TestCase):
    def setUp(self):
        self.norm, self.linear = Norm(), SimpleNamespace(input_scale_inv="gs")
        self.fused_calls = []

        def fused(x, residual, weight, eps, scale):
            self.fused_calls.append((x, residual, weight, eps, scale))
            return "y", "q", "sf"

        self.fused = fused

    def test_eligible_rows_run_the_fused_kernel_and_leave_the_pair(self):
        forward = _norm_forward(self.norm, self.linear, self.fused, check=lambda *a: True)
        self.assertEqual(forward("x", "res"), ("y", "res"))
        self.assertEqual(self.fused_calls, [("x", "res", "w", 1e-6, "gs")])
        self.assertEqual(self.linear.__dict__[STASH], ("y", "q", "sf"))
        self.assertEqual(self.norm.calls, [])

    def test_ineligible_rows_keep_the_original_norm_and_clear_the_pair(self):
        self.linear.__dict__[STASH] = ("old", "q", "sf")
        forward = _norm_forward(self.norm, self.linear, self.fused, check=lambda *a: False)
        self.assertEqual(forward("x", "res", "post"), ("norm(x)", "res"))
        self.assertIsNone(self.linear.__dict__[STASH])
        self.assertEqual(self.fused_calls, [])

    def test_humming_layers_keep_the_original_norm(self):
        self.linear.ax_humming = "config"
        forward = _norm_forward(self.norm, self.linear, self.fused, check=lambda *a: True)
        self.assertEqual(forward("x", "res"), ("norm(x)", "res"))
        self.assertEqual(self.fused_calls, [])


class Apply(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.apply = _apply(lambda method, layer, x, bias=None: self.calls.append((x, bias)) or "out", lambda: False)
        self.method, self.layer, self.y = Fp4Method(), SimpleNamespace(), object()

    def test_the_norm_output_takes_the_prequantized_pair_once(self):
        self.layer.__dict__[STASH] = (self.y, "q", "sf")
        self.assertEqual(self.apply(self.method, self.layer, self.y, "b"), "out")
        self.assertEqual(self.calls, [(("q", "sf"), "b")])
        self.assertIsNone(self.layer.__dict__[STASH])
        self.apply(self.method, self.layer, self.y)
        self.assertIs(self.calls[-1][0], self.y)

    def test_another_input_quantizes_as_before(self):
        self.layer.__dict__[STASH] = (self.y, "q", "sf")
        other = object()
        self.apply(self.method, self.layer, other)
        self.assertIs(self.calls[-1][0], other)
        self.assertIsNone(self.layer.__dict__[STASH])

    def test_marlin_and_w4a16_ignore_the_pair(self):
        marlin = _apply(lambda method, layer, x, bias=None: self.calls.append((x, bias)), lambda: True)
        self.layer.__dict__[STASH] = (self.y, "q", "sf")
        marlin(self.method, self.layer, self.y)
        self.layer.__dict__[STASH] = (self.y, "q", "sf")
        self.apply(Fp4Method("w4a16"), self.layer, self.y)
        self.assertEqual([c[0] for c in self.calls], [self.y, self.y])


class Attach(unittest.TestCase):
    def test_dense_w4a4_gate_up_is_fused(self):
        l = layer()
        self.assertTrue(attach(l, Fp4Method))
        self.assertTrue(l.mlp.gate_up_proj._accepts_prequantized_fp4)
        self.assertIn("forward", vars(l.post_attention_layernorm))

    def test_other_layers_are_left_alone(self):
        for l in (layer(method=SimpleNamespace(quant_mode="w4a4")), layer(Fp4Method("w4a16")),
                  layer(Fp4Method(awq=True)), layer(_interleave_for_swiglu_fusion=True),
                  SimpleNamespace(mlp=SimpleNamespace(), post_attention_layernorm=Norm())):
            self.assertFalse(attach(l, Fp4Method))
            self.assertNotIn("forward", vars(l.post_attention_layernorm))

    def test_flag_and_parity_conflict(self):
        self.assertTrue(PluginConfig.from_env({"AX_NORM_FP4": "1"}).norm_fp4)
        parity = {"AX_NVFP4_VERIFY_PARITY": "1", "AX_GDN_UNPACKED_DECODE": "1", "AX_GDN_RECURRENT_VERIFY": "1"}
        PluginConfig.from_env(parity)
        with self.assertRaisesRegex(ValueError, "AX_NORM_FP4"):
            PluginConfig.from_env({"AX_NORM_FP4": "1", **parity})


if __name__ == "__main__":
    unittest.main()
