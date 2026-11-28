"""AX_GDN_VERIFY_TILES: only fp32-state GDN ring verifies and window-saving verify convs change tiles, offline."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_gdn_verify_tiles import CONV_TILE, RING_BLOCK_V, ConvUpdateKernel, _wrap_ring, ring_verify_call
from ax_engine_sglang.config import PluginConfig

torch = SimpleNamespace(float32="fp32", bfloat16="bf16")


def tensor(dtype="fp32", shape=(1, 64, 48, 128)):
    return SimpleNamespace(dtype=dtype, shape=shape)


def ring_kwargs(**extra):
    return {"cache_ring": True, "initial_state_source": tensor(), "v": tensor(), "is_kda": False, **extra}


class RingVerify(unittest.TestCase):
    def setUp(self):
        self.module = SimpleNamespace(_select_recurrent_launch_config=lambda *a, **k: (32, 1))
        self.seen = []

        def original(*args, **kwargs):
            self.seen.append(self.module._select_recurrent_launch_config(8, 16, 48, 128, 128, False))
            return "o"

        self.call = _wrap_ring(original, self.module, torch)

    def test_fp32_ring_verify_runs_the_narrow_tile_then_restores_the_selector(self):
        self.assertEqual(self.call(**ring_kwargs()), "o")
        self.assertEqual(self.seen, [(RING_BLOCK_V, 1)])
        self.assertEqual(self.module._select_recurrent_launch_config(), (32, 1))

    def test_other_calls_keep_sglang_selection(self):
        for kwargs in (ring_kwargs(cache_ring=False), ring_kwargs(is_kda=True),
                       ring_kwargs(initial_state_source=tensor("bf16")), ring_kwargs(intermediate_states_buffer=object()),
                       ring_kwargs(match_cutedsl_decode=True)):
            self.assertFalse(ring_verify_call(kwargs, torch))
            self.call(**kwargs)
        self.call("positional")
        self.assertEqual(self.seen, [(32, 1)] * 6)

    def test_selector_restored_after_an_error(self):
        def failing(**kwargs):
            raise RuntimeError("launch")

        call = _wrap_ring(failing, self.module, torch)
        with self.assertRaises(RuntimeError):
            call(**ring_kwargs())
        self.assertEqual(self.module._select_recurrent_launch_config(), (32, 1))

    def test_flag(self):
        self.assertTrue(PluginConfig.from_env({"AX_GDN_VERIFY_TILES": "1"}).gdn_verify_tiles)
        self.assertFalse(PluginConfig.from_env({}).gdn_verify_tiles)


class Kernel:
    def __init__(self):
        self.calls, self.fn = [], "jit"

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


class ConvTile(unittest.TestCase):
    def test_window_saving_launch_takes_the_tile(self):
        kernel = Kernel()
        ConvUpdateKernel(kernel)["grid"]("x", SAVE_INTERMEDIATE=True, BLOCK_N=256)
        self.assertEqual(kernel.calls, [("grid", ("x",), {"SAVE_INTERMEDIATE": True, **CONV_TILE})])

    def test_other_launches_unchanged_and_attributes_forwarded(self):
        kernel = Kernel()
        proxy = ConvUpdateKernel(kernel)
        proxy["grid"]("x", SAVE_INTERMEDIATE=False, BLOCK_N=256)
        self.assertEqual(kernel.calls, [("grid", ("x",), {"SAVE_INTERMEDIATE": False, "BLOCK_N": 256})])
        self.assertEqual(proxy.fn, "jit")


if __name__ == "__main__":
    unittest.main()
