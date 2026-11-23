"""Rank counting for features that must stay off in multi-process servers."""
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_ranks import server_ranks  # noqa: E402


def server_args_module(get):
    module = ModuleType("sglang.srt.server_args")
    module.get_global_server_args = get
    return module


class ServerRanksTests(unittest.TestCase):
    def ranks(self, get):
        with patch.dict(sys.modules, {"sglang.srt.server_args": server_args_module(get)}):
            return server_ranks()

    def test_data_parallel_replicas_count(self):
        # each DP replica has a one-rank process group, so the world size alone says 1
        self.assertEqual(self.ranks(lambda: SimpleNamespace(dp_size=2)), 2)

    def test_single_server_is_one(self):
        self.assertEqual(self.ranks(lambda: SimpleNamespace(dp_size=1)), 1)
        self.assertEqual(self.ranks(lambda: SimpleNamespace(dp_size=None)), 1)

    def test_unset_server_args_fall_back_to_one(self):
        def unset():
            raise RuntimeError("runtime context not initialised")
        self.assertEqual(self.ranks(unset), 1)
        with patch.dict(sys.modules, {"sglang.srt.server_args": None}):
            self.assertEqual(server_ranks(), 1)


if __name__ == "__main__":
    unittest.main()
