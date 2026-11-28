"""AX_NVFP4_HUMMING: the load-time conversion and the dispatch around SGLang's Marlin NVFP4 path, offline."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_nvfp4_humming import COMPUTE_CONFIG, _apply, _process, convert
from ax_engine_sglang.config import PluginConfig
from ax_engine_sglang.defaults import LaunchConfigError, plan
from test_defaults import DFLASH2, QWEN38_27B, QWEN38_27B_NVFP4, args


class Tensor:
    """Just enough of a tensor for the conversion's bookkeeping."""

    def __init__(self, shape, values=None, name=""):
        self.shape, self.values, self.name = tuple(shape), values, name
        self.data, self.device = self, "cuda:0"

    def float(self):
        return self

    def reshape(self, *shape):
        if shape == (-1,):
            return Tensor((len(self.values),), self.values, self.name)
        return Tensor(shape, self.values, self.name)

    def numel(self):
        return len(self.values)

    def max(self):
        return max(self.values)

    def __getitem__(self, index):
        return Tensor((self.shape[0], index[1].stop), self.values, self.name + "[:, :n]")

    def __add__(self, other):
        return Tensor(self.shape, self.values, f"{self.name}+{other}")


class Ops:
    def __init__(self):
        self.schema = Mock()
        self.schema.convert_humming.return_value = ("schema", {"converted": True})
        self.weight_schema = Mock(return_value=self.schema)
        self.input_schema = Mock(return_value="a16")
        self.transform = Mock(return_value=("layer-config", {
            "weight": "packed", "weight_scale": "scales", "weight_scale_2": "scale-2"}))
        self.forward = Mock(return_value=Tensor((6, 34816), name="out"))

    def parameter(self, tensor):
        return ("param", tensor)

    def locks(self, device):
        return f"locks@{device}"


def gate_up(scales=(2.0, 3.0)):
    return SimpleNamespace(
        weight=Tensor((34816, 2560)), weight_scale=Tensor((34816, 320)),
        weight_scale_2=Tensor((len(scales),), list(scales)), input_size_per_partition=5120,
        logical_widths=[17408, 17408], params_dtype="bf16", quant_config=SimpleNamespace(group_size=16))


class ConvertTests(unittest.TestCase):
    def test_each_shard_keeps_its_own_global_scale(self):
        layer, ops = gate_up(), Ops()
        convert(layer, ops)
        call = ops.schema.convert_humming.call_args.kwargs
        self.assertEqual(call["shape_n_stacks"], [17408, 17408])
        self.assertEqual(call["shape_k_stacks"], [5120])
        self.assertEqual(call["tensors"]["weight_scale_2"].values, [2.0, 3.0])
        options = ops.transform.call_args.kwargs
        self.assertEqual((options["shape_n"], options["shape_k"]), (34816, 5120))
        self.assertEqual((options["pad_n_to_multiple"], options["pad_k_to_multiple"]), (256, 128))
        self.assertEqual(options["input_schema"], "a16")
        self.assertEqual(ops.input_schema.call_args.args, ("bf16",))

    def test_the_converted_layer_passes_the_marlin_lm_head_gate(self):
        layer, ops = gate_up(), Ops()
        convert(layer, ops)
        self.assertEqual(layer.weight, ("param", "packed"))   # Humming packs int32, as Marlin does
        self.assertEqual(layer.weight_scale_2, ("param", "scale-2"))
        self.assertEqual(layer.weight_global_scale, ("param", 3.0))
        self.assertEqual(layer.workspace, "locks@cuda:0")
        self.assertEqual(layer.output_size_per_partition, 34816)
        self.assertEqual(layer.ax_humming, "layer-config")

    def test_layouts_it_does_not_understand_are_refused(self):
        cases = (
            (dict(weight_scale_2=Tensor((1,), [2.0])), "1 weight_scale_2 values for 2 shards"),
            (dict(logical_widths=[17408]), "does not match"),
            (dict(quant_config=SimpleNamespace(group_size=32)), "group_size=16"),
        )
        for changes, message in cases:
            layer = gate_up()
            vars(layer).update(changes)
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                convert(layer, Ops())


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.ops = Ops()
        self.marlin = SimpleNamespace(is_marlin=lambda: True)

    def test_only_the_marlin_w4a4_branch_is_converted(self):
        original = Mock(return_value="stock")
        cutedsl = SimpleNamespace(is_marlin=lambda: False)
        for mode, backend in (("w4a16", self.marlin), ("w4a4", cutedsl)):
            with self.subTest(mode=mode):
                process = _process(original, lambda: backend, lambda: self.ops)
                self.assertEqual(process(SimpleNamespace(quant_mode=mode), gate_up()), "stock")
        self.ops.transform.assert_not_called()
        layer = gate_up()
        _process(original, lambda: self.marlin, lambda: self.ops)(SimpleNamespace(quant_mode="w4a4"), layer)
        self.assertEqual(layer.ax_humming, "layer-config")
        self.assertEqual(original.call_count, 2)

    def test_apply_flattens_adds_the_bias_once_and_restores_the_leading_dims(self):
        original = Mock(return_value="marlin")
        apply = _apply(original, lambda: self.ops)
        self.assertEqual(apply("method", SimpleNamespace(), "x"), "marlin")
        layer = SimpleNamespace(ax_humming="cfg", weight="w", weight_scale="s", weight_scale_2="s2",
                                workspace="locks", output_size_per_partition=34816)
        out = apply("method", layer, Tensor((2, 3, 5120)), "bias")
        call = self.ops.forward.call_args
        self.assertEqual(call.args, ("cfg",))
        self.assertEqual(call.kwargs["inputs"].shape, (-1, 5120))
        self.assertEqual(call.kwargs["compute_config"], COMPUTE_CONFIG)
        self.assertEqual(call.kwargs["locks"], "locks")
        self.assertEqual(out.name, "out+bias")
        self.assertEqual(out.shape, (2, 3, 34816))

    def test_padded_output_columns_are_dropped(self):
        self.ops.forward.return_value = Tensor((6, 248576), name="out")
        layer = SimpleNamespace(ax_humming="cfg", weight="w", weight_scale="s", weight_scale_2="s2",
                                workspace="locks", output_size_per_partition=248320)
        out = _apply(Mock(), lambda: self.ops)("method", layer, Tensor((6, 5120)))
        self.assertEqual((out.name, out.shape), ("out[:, :n]", (6, 248320)))


class LaunchTests(unittest.TestCase):
    def test_the_feature_is_opt_in_and_not_a_parity_path(self):
        self.assertFalse(PluginConfig.from_env({}).nvfp4_humming)
        self.assertTrue(PluginConfig.from_env({"AX_NVFP4_HUMMING": "1"}).nvfp4_humming)
        self.assertTrue(PluginConfig.from_env({"AX_NVFP4_HUMMING": "1", "AX_GDN_CHAIN_REPLAY": "1"}).nvfp4_humming)

    def test_the_launch_pins_marlin_on_sm90_nvfp4_only(self):
        environ = {"AX_NVFP4_HUMMING": "1"}
        for route in (args("DFLASH", speculative_num_draft_tokens=8), args()):
            result = plan(route, environ, QWEN38_27B_NVFP4, DFLASH2, capability=(9, 0))
            self.assertEqual(result.args["fp4_gemm_runner_backend"], ("marlin", "AX_NVFP4_HUMMING"))
        self.assertNotIn("fp4_gemm_runner_backend",
                         plan(args("DFLASH"), {}, QWEN38_27B_NVFP4, DFLASH2, capability=(9, 0)).args)
        launch = args("DFLASH", speculative_num_draft_tokens=8)
        cases = (
            (launch, environ, QWEN38_27B_NVFP4, (10, 0), "SM90"),
            (launch, environ, QWEN38_27B, (9, 0), "ModelOpt NVFP4"),
            (args("DFLASH", fp4_gemm_runner_backend="flashinfer_cutlass"), environ, QWEN38_27B_NVFP4, (9, 0),
             "Marlin FP4 backend"),
            (launch, {**environ, "AX_QUALITY_PROFILE": "verified"}, QWEN38_27B_NVFP4, (9, 0),
             "parity or verified"),
        )
        for route, options, target, capability, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(LaunchConfigError, message):
                plan(route, options, target, DFLASH2, capability=capability)


if __name__ == "__main__":
    unittest.main()
