"""Compatibility checks and reversible startup patches to upstream Python objects."""
import __future__
from contextlib import contextmanager
from contextvars import ContextVar
import importlib
import inspect
import sys
import textwrap


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


def rebuild_function(cls, name, edits, *, feature):
    """Compile a version-checked replacement without changing the live class."""
    fn = getattr(cls, name)
    src = textwrap.dedent(inspect.getsource(fn))
    for old, new, count in edits:
        found = src.count(old)
        if found != count:
            raise RuntimeError(f"ax-engine {feature}: {cls.__name__}.{name} has {found} matches "
                               f"for {old!r}, expected {count} (SGLang changed?)")
        src = src.replace(old, new)
    mod = sys.modules[fn.__module__]
    ns = {}
    filename = f"<ax-engine {feature} {fn.__module__}.{cls.__name__}.{name}>"
    flags = fn.__code__.co_flags & __future__.annotations.compiler_flag
    exec(compile(src, filename, "exec", flags=flags, dont_inherit=True), mod.__dict__, ns)
    new_fn = ns[name]
    new_fn.__qualname__ = fn.__qualname__
    return new_fn
