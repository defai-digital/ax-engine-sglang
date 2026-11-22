"""Fresh sparse PLE table file per boot (offline, with stand-in SGLang modules)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang import ax_ple


def table_module(directory, calls):
    module = ModuleType("sglang.srt.models.qwen4_exp_ple_table")
    module.envs = SimpleNamespace(SGLANG_QWEN4_PLE_FILE_DIR=SimpleNamespace(get=lambda: directory))
    module.ple_table_file_name = lambda shape, dtype, tag: f"table_{shape[0]}x{shape[1]}_{dtype}_{tag}.bin"

    def allocate_ple_host_table(shape, dtype, backend="pinned", table_dir=None, tag=None):
        path = Path(table_dir or directory) / module.ple_table_file_name(shape, dtype, tag)
        calls.append((backend, path.exists()))
        return "table"

    module.allocate_ple_host_table = allocate_ple_host_table
    return module


class FreshPleFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.calls = []
        self.table = table_module(self.dir.name, self.calls)
        self.modules = {"sglang": ModuleType("sglang"), "sglang.srt": ModuleType("sglang.srt"),
                        "sglang.srt.models": ModuleType("sglang.srt.models"),
                        "sglang.srt.models.qwen4_exp_ple_table": self.table}
        self.modules["sglang.srt.models"].qwen4_exp_ple_table = self.table

    def tearDown(self):
        self.dir.cleanup()

    def install(self, env=None, extra=None):
        with patch.dict(sys.modules, {**self.modules, **(extra or {})}), \
             patch.dict(os.environ, env or {}, clear=True):
            ax_ple.install()

    def existing_file(self, tag="rows0-8"):
        path = Path(self.dir.name) / self.table.ple_table_file_name((8, 4), "fp8", tag)
        path.write_bytes(b"x" * 32)
        return path

    def test_file_backend_starts_from_a_new_file(self):
        path = self.existing_file()
        old_inode = path.stat().st_ino
        held = open(path, "rb")   # another server still mapping the old file keeps its inode
        self.addCleanup(held.close)
        self.install()
        self.assertEqual(self.table.allocate_ple_host_table((8, 4), "fp8", backend="file", tag="rows0-8"), "table")
        self.assertEqual(self.calls, [("file", False)])
        self.assertEqual(os.fstat(held.fileno()).st_ino, old_inode)
        self.assertEqual(held.read(), b"x" * 32)

    def test_explicit_table_dir_and_missing_file(self):
        with tempfile.TemporaryDirectory() as other:
            self.install()
            self.table.allocate_ple_host_table((8, 4), "fp8", backend="file", table_dir=other, tag="t")
        self.assertEqual(self.calls, [("file", False)])

    def test_pinned_backend_is_untouched(self):
        path = self.existing_file()
        self.install()
        self.table.allocate_ple_host_table((8, 4), "fp8", backend="pinned", tag="rows0-8")
        self.assertTrue(path.exists())
        self.assertEqual(self.calls, [("pinned", True)])

    def test_disabled_by_environment(self):
        original = self.table.allocate_ple_host_table
        self.install({"AX_PLE_FRESH_FILE": "0"})
        self.assertIs(self.table.allocate_ple_host_table, original)

    def test_loaded_model_module_binding_is_patched_once(self):
        model = ModuleType("sglang.srt.models.qwen4_exp")
        model.allocate_ple_host_table = self.table.allocate_ple_host_table
        self.install(extra={"sglang.srt.models.qwen4_exp": model})
        self.assertIs(model.allocate_ple_host_table, self.table.allocate_ple_host_table)
        fresh = self.table.allocate_ple_host_table
        self.install(extra={"sglang.srt.models.qwen4_exp": model})
        self.assertIs(self.table.allocate_ple_host_table, fresh)

    def test_without_the_ple_module_nothing_happens(self):
        with patch.dict(sys.modules, {"sglang.srt.models.qwen4_exp_ple_table": None}):
            ax_ple.install()


if __name__ == "__main__":
    unittest.main()
