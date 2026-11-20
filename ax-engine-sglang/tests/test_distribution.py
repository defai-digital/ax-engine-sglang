"""Build source-only distributions without a sibling checkout or network access."""
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


class DistributionTests(unittest.TestCase):
    def test_wheel_and_sdist_are_independent_of_research_tree(self):
        with tempfile.TemporaryDirectory(prefix="ax-engine-dist-") as directory:
            root = Path(directory)
            package = root / "package"
            shutil.copytree(Path(__file__).resolve().parents[1], package,
                            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "dist"))
            def build(path, method):
                result = subprocess.run([
                    sys.executable, "-c", f'import setuptools.build_meta as b; b.{method}("dist")',
                ], cwd=path, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            def check_wheel(path):
                wheel = next((path / "dist").glob("*.whl"))
                with zipfile.ZipFile(wheel) as archive:
                    files = archive.namelist()
                    for name in ("ax_fast.py", "ax_lattice.py", "config.py", "policy.py", "runtime.py", "compat.py", "contracts.py", "qsa_topk.py"):
                        self.assertIn(f"ax_engine_sglang/{name}", files)
                    dist_info = Path(next(name for name in files if name.endswith(".dist-info/METADATA"))).parent
                    for license_file in ("LICENSE", "NOTICE", "licenses/FLASH_LINEAR_ATTENTION-LICENSE"):
                        self.assertIn((dist_info / "licenses" / license_file).as_posix(), files)
                result = subprocess.run([
                    sys.executable, "-c",
                    'from importlib.metadata import distribution; '
                    'from ax_engine_sglang import __version__; '
                    'from ax_engine_sglang.config import PluginConfig; '
                    'from ax_engine_sglang.contracts import plan_greedy_commit; '
                    'dist = distribution("ax-engine-sglang"); '
                    'assert dist.version == __version__; '
                    'assert dist.entry_points["ax-engine-sglang"].load().__name__ == "main"; '
                    'assert dist.entry_points["ax-engine"].value == "ax_engine_sglang.plugin:register"; '
                    'assert not PluginConfig.from_env({}).tree_requested; '
                    'assert plan_greedy_commit([1], [1, 2], 8, []).emitted == [1, 2]',
                ], cwd=root, env={**__import__("os").environ, "PYTHONPATH": str(wheel)},
                   capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            build(package, "build_wheel")
            check_wheel(package)
            build(package, "build_sdist")
            with tarfile.open(next((package / "dist").glob("*.tar.gz"))) as archive:
                archive.extractall(root / "sdist", filter="data")
            standalone = next((root / "sdist").iterdir())
            build(standalone, "build_wheel")
            check_wheel(standalone)


if __name__ == "__main__":
    unittest.main()
