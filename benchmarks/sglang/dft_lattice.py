"""Compatibility import for the package-owned ax_lattice implementation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ax-engine-sglang"))
from ax_engine_sglang import ax_lattice as _implementation

# Keep one module instance, including diagnostic flags used by older callers.
_implementation = sys.modules.setdefault("ax_lattice", _implementation)
sys.modules[__name__] = _implementation
