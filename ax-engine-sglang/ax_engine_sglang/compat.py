"""Compatibility checks and reversible startup patches to upstream Python objects."""
from contextlib import contextmanager
from contextvars import ContextVar
import importlib


TESTED_SGLANG = ("0.5.22", "0.5.23")
_missing = object()
_active: ContextVar[object] = ContextVar("ax_engine_sglang_patches", default=None)


class PatchSet:
    def __init__(self):
        self.undo = []

    def attribute(self, target, name, value):
        # Preserve inherited descriptors by deleting the override on rollback.
        old = vars(target).get(name, _missing)
        setattr(target, name, value)
        self.undo.append((target, name, old, False))

    def item(self, target, name, value):
        self.undo.append((target, name, target.get(name, _missing), True))
        target[name] = value

    def rollback(self):
        for target, name, old, mapping in reversed(self.undo):
            if mapping:
                if old is _missing:
                    target.pop(name, None)
                else:
                    target[name] = old
            elif old is _missing:
                delattr(target, name)
            else:
                setattr(target, name, old)
        self.undo.clear()


@contextmanager
def installation():
    existing = _active.get()
    if existing is not None:
        yield existing
        return
    patches = PatchSet()
    token = _active.set(patches)
    try:
        yield patches
    except BaseException:
        patches.rollback()
        raise
    finally:
        _active.reset(token)


def patch_attribute(target, name, value):
    patches = _active.get()
    if patches is None:
        setattr(target, name, value)
    else:
        patches.attribute(target, name, value)


def patch_item(target, name, value):
    patches = _active.get()
    if patches is None:
        target[name] = value
    else:
        patches.item(target, name, value)


def check_sglang(config):
    version = getattr(importlib.import_module("sglang"), "__version__", "?")
    if version.split("+")[0] not in TESTED_SGLANG and not config.allow_untested_sglang:
        raise RuntimeError(
            f"ax-engine-sglang: untested SGLang {version}; tested: {', '.join(TESTED_SGLANG)}. "
            "Port and validate the hooks, or explicitly set AX_ALLOW_UNTESTED_SGLANG=1."
        )
    return version
