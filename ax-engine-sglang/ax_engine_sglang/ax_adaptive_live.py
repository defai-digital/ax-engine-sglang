# SPDX-License-Identifier: Apache-2.0
"""Reload SGLang's adaptive speculative parameters while the server runs.

`--speculative-adaptive` reads its per-batch-size policy (candidate steps, EMA alpha, update
interval, warmup, hysteresis, ceiling) once at startup. Tuning it then needs a fresh boot per
candidate, which on a Jetson Thor with Qwen3.8-Flash-Next takes about 20 minutes. With
AX_ADAPTIVE_LIVE=<path> the policy is re-read from that JSON file (same format as
--speculative-adaptive-config, plus an optional "version") whenever its mtime changes, checked at
most every AX_ADAPTIVE_LIVE_POLL_S seconds (default 0.5) from the verify path. Every slot restarts
its EMA and warmup, so each setting is measured from the same state.

The runtime states (CUDA graphs) exist only for the candidate steps of the boot config, and each
step's graphs only for the CUDA-graph batch sizes whose boot slot listed it. A live file must keep
every CUDA-graph batch size on steps captured for it; any other file is rejected and the running
policy kept, as is a file whose numbers are out of range (ema_alpha in (0, 1], integer
update_interval >= 1 and warmup_batches >= 0, finite hysteresis and ceiling). After each reload
attempt `<path>.ack` records the version, whether it applied, and how many verify batches ran at
each step under the previous policy.

The verify batch that finds a new file still feeds the policy it ran under. With the overlap
scheduler the next batch was already launched under the old policy too, so the first verify after
a reload is neither fed to the new slots nor counted; they start from their own EMA after that.

Each scheduler process polls the file on its own clock, so with more than one rank (tensor, data or
pipeline parallel) ranks could switch at different verify batches and run forwards of different
widths, and replicas would race on one ack file. The reload is therefore disabled when the
torch.distributed world size or --dp-size is above one.
"""
import bisect
import json
import os
import sys
import time

from ax_engine_sglang.ax_ranks import server_ranks
from ax_engine_sglang.compat import patch_attribute


class LiveState:
    def __init__(self, path, poll_s):
        self.path = path
        self.poll_s = poll_s
        self.next_check = 0.0
        self.mtime = None
        self.step_counts = {}
        self.graph_steps = None   # CUDA-graph batch size -> steps captured for it, read at first reload
        self.enabled = None       # decided on the first verify batch
        self.skip = 0             # verify results still in flight from the replaced policy

    def due(self, now):
        if now < self.next_check:
            return False
        self.next_check = now + self.poll_s
        return True


def _write_ack(path, payload):
    # The ack is advisory: a failed write must never raise on the verify path.
    tmp = f"{path}.ack.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, f"{path}.ack")
    except OSError as exc:
        print(f"ax-engine-sglang: adaptive live ack not written: {exc}", file=sys.stderr)


def _closest(bs_list, batch_size):
    # Same rule as AdaptiveSpeculativeParams._find_closest_bs.
    return bs_list[max(0, bisect.bisect_right(bs_list, batch_size) - 1)]


def graph_steps(params):
    """CUDA-graph batch size -> the steps whose runtime state captured a graph for it at boot.

    {} when CUDA graphs are off (`_cuda_graph_bs` None). An SGLang without the attribute cannot be
    checked, so live reloads are refused there.
    """
    if not hasattr(params, "_cuda_graph_bs"):
        raise ValueError("this SGLang does not expose the CUDA-graph batch sizes "
                         "(AdaptiveSpeculativeParams._cuda_graph_bs); live reload is unsafe")
    graph_bs = params._cuda_graph_bs
    if not graph_bs:
        return {}
    return {bs: set(params._slots[_closest(params._bs_list, bs)].candidate_steps) for bs in graph_bs}


def _check_numbers(bs, cfg):
    # Absent keys take SGLang's defaults; present ones (null included) must be usable numbers.
    def real(key):
        if key not in cfg:
            return None
        value = cfg[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not -1e6 <= value <= 1e6):   # also rejects nan, inf and huge integers
            raise ValueError(f"BS {bs}: {key} must be a number in [-1e6, 1e6], got {value!r}")
        return value

    def count(key, minimum):
        if key in cfg and (type(cfg[key]) is not int or not minimum <= cfg[key] <= 10**9):
            raise ValueError(f"BS {bs}: {key} must be an integer in [{minimum}, 1e9], got {cfg[key]!r}")

    alpha = real("ema_alpha")
    if alpha is not None and not 0 < alpha <= 1:
        raise ValueError(f"BS {bs}: ema_alpha must be in (0, 1], got {alpha!r}")
    for key in ("down_hysteresis", "up_hysteresis", "ceiling_coeff"):
        real(key)
    count("update_interval", 1)
    count("warmup_batches", 0)


def build_slots(module, cfg, allowed_steps, current_steps, captured=None):
    """Return (bs_list, slots) for `cfg`, or raise ValueError when it needs uncaptured steps."""
    if not isinstance(cfg, dict):
        raise ValueError("live config must be a JSON object")
    bs_entries = {int(k): v for k, v in cfg.items() if k.isdigit()}
    if not bs_entries:
        raise ValueError("live config has no batch-size entries")
    shared = {k: v for k, v in cfg.items() if not k.isdigit() and k != "version"}
    slots = {}
    for bs, entry in sorted(bs_entries.items()):
        if not isinstance(entry, dict):
            raise ValueError(f"BS {bs}: entry must be a JSON object")
        steps = entry.get("candidate_steps")
        if (not isinstance(steps, list) or not steps or any(type(s) is not int for s in steps)
                or not set(steps) <= set(allowed_steps)):
            raise ValueError(f"BS {bs}: candidate_steps {steps!r} must be a non-empty subset of the "
                             f"boot candidates {sorted(allowed_steps)}")
        _check_numbers(bs, {**shared, **entry})
        # A start step outside the candidates falls back to one of them (AdaptiveStepSlot).
        slot = module.AdaptiveStepSlot(initial_steps=current_steps, cfg={**shared, **entry})
        slots[bs] = slot
    bs_list = sorted(slots)
    for graph_bs, steps in sorted((captured or {}).items()):
        routed = _closest(bs_list, graph_bs)
        if not set(slots[routed].candidate_steps) <= steps:
            raise ValueError(f"CUDA-graph batch size {graph_bs} would route to BS {routed} with steps "
                             f"{sorted(slots[routed].candidate_steps)}, but only {sorted(steps)} were "
                             "captured for it at boot")
    return bs_list, slots


def _reload(params, module, state):
    """Apply a changed live file; return True when the slots were replaced."""
    try:
        mtime = os.stat(state.path).st_mtime_ns
    except OSError:   # missing or unreadable: keep the running policy (polled again later)
        return False
    if mtime == state.mtime:
        return False
    state.mtime = mtime
    counts, state.step_counts = state.step_counts, {}
    cfg = None
    try:
        if state.graph_steps is None:
            state.graph_steps = graph_steps(params)   # still the boot slots: nothing swapped yet
        with open(state.path) as f:
            cfg = json.load(f)
        current = params._slots[params._bs_list[0]].current_steps
        bs_list, slots = build_slots(module, cfg, params.candidate_steps, current, state.graph_steps)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        version = cfg.get("version") if isinstance(cfg, dict) else None
        _write_ack(state.path, {"version": version, "applied": False, "error": str(exc),
                                "step_counts": counts})
        print(f"ax-engine-sglang: adaptive live config rejected: {exc}", file=sys.stderr)
        return False
    params._bs_list, params._slots = bs_list, slots
    _write_ack(state.path, {"version": cfg.get("version"), "applied": True, "step_counts": counts})
    print(f"ax-engine-sglang: adaptive live config version {cfg.get('version')} applied "
          f"({', '.join(f'{bs}:{slots[bs].candidate_steps}' for bs in bs_list)})", file=sys.stderr)
    return True


def _wrap(on_verify_complete, module, state, clock=time.monotonic, world_size=server_ranks):
    def wrapped(self, num_correct_drafts_per_req, batch_size):
        if state.enabled is None:
            ranks = world_size()
            state.enabled = ranks == 1
            if not state.enabled:
                print(f"ax-engine-sglang: AX_ADAPTIVE_LIVE ignored with {ranks} ranks (ranks would "
                      "switch steps at different batches); the boot policy stays", file=sys.stderr)
        if not state.enabled:
            return on_verify_complete(self, num_correct_drafts_per_req, batch_size)
        if state.skip:
            state.skip -= 1   # launched before the reload, under the replaced policy
            return None
        # This batch ran under the policy in force now: count it and feed it to that policy first.
        step = self._route(batch_size).current_steps
        state.step_counts[str(step)] = state.step_counts.get(str(step), 0) + 1
        new_step = on_verify_complete(self, num_correct_drafts_per_req, batch_size)
        if state.due(clock()) and _reload(self, module, state):
            state.skip = 1
            # The old slot's verdict no longer applies; run the new slot's step from now on.
            return self._route(batch_size).current_steps
        return new_step
    wrapped._ax_live = True
    return wrapped


def install():
    path = os.environ.get("AX_ADAPTIVE_LIVE")
    if not path:
        return
    import sglang.srt.speculative.adaptive_spec_params as module
    cls = module.AdaptiveSpeculativeParams
    if getattr(cls.on_verify_complete, "_ax_live", False):
        return
    state = LiveState(path, float(os.environ.get("AX_ADAPTIVE_LIVE_POLL_S", "0.5")))
    patch_attribute(cls, "on_verify_complete", _wrap(cls.on_verify_complete, module, state))
    print(f"ax-engine-sglang: adaptive policy reloads from {path}", file=sys.stderr)
