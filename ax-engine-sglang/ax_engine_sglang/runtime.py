"""Worker-owned round state, selected in the current Python execution context."""
from contextvars import ContextVar
from dataclasses import dataclass, field
from weakref import WeakSet


@dataclass(eq=False)
class WorkerRuntime:
    kind: str | None = None
    round_index: int = 0
    draft_width_override: int | None = None
    main_backend: str | None = None
    host_kv_len: int | None = None
    fused_context: object | None = None
    round_info: dict = field(default_factory=lambda: {"backend": None, "graph": None, "width": None})
    alloc_seen: dict = field(default_factory=dict)
    # AX_POLICY_MODE=adaptive: the worker's WidthController, the (bucket, width) of the chain round
    # in flight, when that round began, and accept-length counts not yet read back from the GPU.
    controller: object | None = None
    pending: tuple[int, int] | None = None
    round_started: float | None = None
    accept_counts: dict = field(default_factory=dict)
    accept_rounds: int = 0
    stats: dict = field(default_factory=lambda: {
        "tree": 0, "btree": 0, "chain": 0, "chain_width": {}, "hook_tree": 0, "hook_btree": 0, "hook_chain": 0,
        "alloc_checked": 0, "alloc_short": 0, "alloc_shrunk": 0, "replay": {},
    })


_active: ContextVar[WorkerRuntime | None] = ContextVar("ax_engine_sglang_worker", default=None)
_runtimes: WeakSet = WeakSet()


def current_runtime() -> WorkerRuntime:
    state = _active.get()
    if state is None:
        state = WorkerRuntime()
        _runtimes.add(state)
        _active.set(state)
    return state


def activate_worker(worker) -> WorkerRuntime:
    # SGLang can capture target graphs before initializing the speculative worker.
    # Adopt the model runner's capture state so replay sees the very same buffers.
    owner = getattr(getattr(worker, "target_worker", None), "model_runner", worker)
    state = getattr(owner, "_ax_runtime", None) or getattr(worker, "_ax_runtime", None)
    if state is None:
        state = WorkerRuntime()
        _runtimes.add(state)
    owner._ax_runtime = worker._ax_runtime = state
    _active.set(state)
    return state


def runtime_statistics():
    return [state.stats for state in _runtimes
            if state.stats["tree"] or state.stats["btree"] or state.stats["chain"]]
