"""Debug trace for the host-side KV-length mirror of the DFlash draft input (AX_NXT_DEBUG=1).

With no attention backend asking for host sequence lengths (triton target, trtllm_mha drafter),
SGLang drops batch.seq_lens_cpu and the draft step reads `draft_input.nxt_kv_lens_cpu` instead,
which `prepare_for_decode`, `filter_batch` and `merge_batch` keep one entry per request. This
records every one of those calls and every forward, and when a forward sees a mirror whose length
differs from the batch it prints the recent history with the stack of the forward and raises.
Not for speed runs.
"""
from __future__ import annotations

import collections
import sys
import traceback

from ax_engine_sglang.compat import installation, patch_attribute

_LOG = collections.deque(maxlen=64)
_installed = False


def _n(x):
    v = getattr(x, "nxt_kv_lens_cpu", None)
    return None if v is None else int(v.shape[0])


def _f(x):
    v = getattr(x, "future_indices", None)
    return None if v is None else int(v.shape[0])


def install(worker_cls):
    if _installed:
        return
    with installation():
        _install(worker_cls)
        patch_attribute(sys.modules[__name__], "_installed", True)


def _install(worker_cls):
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2 as D

    def wrap(name):
        orig = getattr(D, name)

        def f(self, *a, **k):
            before = _n(self)
            out = orig(self, *a, **k)
            extra = ""
            if name == "prepare_for_decode":
                extra = f" reqs={len(a[0].reqs)}"
            elif name == "merge_batch":
                extra = f" other={id(a[0]) if a else id(k.get('spec_info'))} other_n={_n(a[0] if a else k.get('spec_info'))}"
            elif name == "filter_batch":
                idx = k.get("new_indices_cpu")
                extra = f" keep={None if idx is None else len(idx)}"
            _LOG.append(f"{name} obj={id(self)} n {before}->{_n(self)} fut={_f(self)}{extra}")
            return out
        patch_attribute(D, name, f)

    for name in ("prepare_for_decode", "filter_batch", "merge_batch"):
        wrap(name)

    from sglang.srt.managers.schedule_batch import ScheduleBatch as SB

    def wrap_sb(name):
        orig = getattr(SB, name)

        def f(self, *a, **k):
            b = (len(self.reqs), None if self.seq_lens is None else len(self.seq_lens), _n(self.spec_info), _f(self.spec_info))
            out = orig(self, *a, **k)
            o = a[0] if (name == "merge_batch" and a) else None
            extra = "" if o is None else f" other reqs={len(o.reqs)} other_spec={type(o.spec_info).__name__} other_n={_n(o.spec_info)} other_fut={_f(o.spec_info)}"
            _LOG.append(f"SB.{name} id={id(self)} spec={id(self.spec_info)} (reqs,seq_lens,n,fut) {b}->"
                        f"{(len(self.reqs), None if self.seq_lens is None else len(self.seq_lens), _n(self.spec_info), _f(self.spec_info))}{extra}")
            return out
        patch_attribute(SB, name, f)

    for name in ("filter_batch", "merge_batch", "prepare_for_decode"):
        wrap_sb(name)

    orig_fwd = worker_cls.forward_batch_generation

    def forward_batch_generation(self, batch, *a, **k):
        di = getattr(batch, "spec_info", None)
        mode = getattr(batch.forward_mode, "name", batch.forward_mode)
        n = _n(di)
        bs = len(batch.reqs) if batch.reqs is not None else None
        cpu = None if batch.seq_lens_cpu is None else int(batch.seq_lens_cpu.shape[0])
        _LOG.append(f"forward mode={mode} obj={id(di)} type={type(di).__name__} n={n} fut={_f(di)} reqs={bs} "
                    f"seq_lens={len(batch.seq_lens)} seq_lens_cpu={cpu}")
        if (n is not None and batch.seq_lens is not None and not batch.forward_mode.is_extend()
                and len({n, len(batch.seq_lens), bs}) > 1):
            print("[ax_nxt_debug] mirror out of step; history:\n  " + "\n  ".join(_LOG), flush=True)
            traceback.print_stack()
            raise RuntimeError(f"ax-engine: nxt_kv_lens_cpu has {n} entries for {len(batch.seq_lens)} requests")
        res = orig_fwd(self, batch, *a, **k)
        nd = getattr(res, "next_draft_input", None)
        _LOG.append(f"result next_draft_input obj={id(nd)} n={_n(nd)}")
        return res

    patch_attribute(worker_cls, "forward_batch_generation", forward_batch_generation)
    print("[ax_nxt_debug] tracing the draft-input KV-length mirror", flush=True)
