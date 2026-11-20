# SPDX-License-Identifier: Apache-2.0
"""Several requests at once: chain rounds next to the one-request tree (multi-user Stage 0).

Design: docs/decisions/2026-11-15-multiuser-width-policy.md. One server, drafter block B
(`FC1_DRAFT_BLOCK`, 16) and `speculative_num_draft_tokens` = tree nodes (`FC1_TREE_N`, 16). Each
decode round is one of two kinds, decided here before the target verify and kept in `current_runtime().kind` so the
fused tree hooks (fc1_fused, fc1_conv, fc1_plan) run only in tree rounds:

  tree   one running request and the tree enabled: the fused tree verifier (fc1_branch64)
  chain  otherwise: SGLang's own DFlash chain verify on the first W drafted tokens

Stage 0 has no width rule. The width is pinned with `AX_CHAIN_WIDTH` (8 or 16, default B), or a
forced schedule `AX_CHAIN_SCHEDULE="16x5,8x3"` (width x rounds, cycled) for the switching checks.
`AX_TREE=0` turns tree rounds off (every round is a chain round, also with one request).

A chain round of width W = B runs the stock code unchanged. For W < B, `narrow` cuts the drafted
block, its positions and its KV slots to W columns, plans a W-slot verify window (the stock worker
plans one B-slot window for draft and verify), and sets the worker's `block_size` to W for the
rest of the round, so every stock read after the draft step (accept reshaping, logit adjustments,
logprob stride, hidden-state view, the result's `speculative_num_draft_tokens`) uses W.
`begin_round` puts B back at the start of the next call. The accept/bonus buffers are sized by
`block_size` when first used, so each width keeps its own set (`_swap_accept_buffers`).

`record_accept` (after the stock accept) keeps, per request, the token before the next anchor, which
the tree builder conditions on when the request is alone again (`pre1_for`), and appends one line
per round to `AX_ROUND_DUMP` (if set) for the offline accept oracle: request id, kind, width, the
verified tokens and the accepted length.

Two attention backends (docs/decisions/2026-11-16-dual-attention-backend.md): with
`AX_CHAIN_ATTN=<backend>` (for example trtllm_mha) the private chain runners are built with that
backend, in that backend's role, while the main runner (tree rounds, prefill) keeps the server's
`--attention-backend` (triton, which has the tree mask). Every chain round must then run on a
private runner: the mixed mode needs CUDA graphs and a graph runner for every chain width, and a
chain round that would run on the main backend stops the server. The round dump records, per
round, the backend that ran it and whether the graph replayed; `[ax_chain]` stats count graph and
eager rounds per (width, batch size).

Tree rounds serve one greedy request without grammar or logprobs (the tree verify accepts the
argmax path only). Any other decode batch, including a single sampled request, runs a chain round,
which is SGLang's own DFlash verify: sampled (selector sampling), grammar-constrained and logprob
requests get stock behaviour there instead of an error. (Until 2026-11-16 such batches raised in
the scheduler and stopped the server; with the model card's default temperature 1.0 that was any
two concurrent requests that did not set temperature 0.)
"""
from __future__ import annotations

import json
import os

import torch

from ax_engine_sglang.config import get_config
from ax_engine_sglang.policy import RoundFeatures, chain_width, use_tree
from ax_engine_sglang.runtime import activate_worker, current_runtime, runtime_statistics
from ax_engine_sglang.compat import patch_attribute

_CONFIG = get_config()
_TREE = _CONFIG.tree
_PINNED = _CONFIG.chain_width
_SCHED = _CONFIG.chain_schedule
_DUMP = _CONFIG.round_dump
_GRAPH_WIDTHS = _CONFIG.graph_widths
_DRAFT_WIDTHS = _CONFIG.draft_widths
_CHECK = _CONFIG.chain_check
_CHAIN_ATTN = _CONFIG.chain_attention


def is_tree():
    state = current_runtime()
    if state.kind is not None:
        state.stats["hook_tree" if state.kind == "tree" else "hook_chain"] += 1
    return state.kind == "tree"


def _check_alloc(batch):
    """AX_CHAIN_CHECK=1, sampled when the round's kind is chosen (after prepare_for_decode and the
    draft step): every running request has this round's 16-slot window allocated beyond its committed
    length (prepare reserves 2 x 16 = 32; with overlap scheduling the committed length can advance by
    one round's commit before this sample, so the gap seen here is 16..32), and its allocation never
    shrinks while it runs. The smallest gap seen is reported."""
    for req in batch.reqs:
        alloc, committed = int(req.kv.kv_allocated_len), int(req.kv.kv_committed_len)
        current_runtime().stats["alloc_checked"] += 1
        gap = alloc - committed
        current_runtime().stats["alloc_min_gap"] = min(current_runtime().stats.get("alloc_min_gap", gap), gap)
        if gap < 16:
            current_runtime().stats["alloc_short"] += 1
        if alloc < current_runtime().alloc_seen.get(req.rid, 0):
            current_runtime().stats["alloc_shrunk"] += 1
        current_runtime().alloc_seen[req.rid] = alloc
        if req.finished():
            current_runtime().alloc_seen.pop(req.rid, None)


def width(block):
    """Verify width of this chain round (anchor included)."""
    return chain_width(_CONFIG, block, current_runtime().round_index)


def _configured_widths(block):
    """Chain widths this server can choose (Stage 0: pinned, schedule or the block)."""
    if _SCHED:
        return sorted(set(_SCHED))
    if _PINNED:
        return [int(_PINNED)]
    return [block]


def tree_round(batch):
    """Translate SGLang sampling capabilities into the pure round policy."""
    if not _CONFIG.tree or len(batch.reqs) != 1 or batch.has_grammar or batch.return_logprob:
        return False
    info = batch.sampling_info
    return use_tree(_CONFIG, RoundFeatures(
        requests=len(batch.reqs), grammar=batch.has_grammar, logprobs=batch.return_logprob,
        custom_processor=bool(info is not None and info.has_custom_logit_processor),
        greedy=info is None or info.is_all_greedy,
        min_p=bool(info is not None and not info.is_all_greedy and info.need_min_p_sampling),
    ))


def begin_round(w, batch=None):
    """Start of forward_batch_generation: restore the drafter block after a narrowed chain round,
    the main attention backend and graph runner after a chain round on a private pair, and the
    drafter's state after a round drafted at a narrower width. Then, for a decode round that will be
    a chain round drafted at width W (AX_CHAIN_DRAFT_WIDTHS), switch the drafter to W."""
    state = activate_worker(w)
    _leave_draft_width(w)
    block = getattr(w, "_ax_block", None)
    if block is None:
        block = w._ax_block = int(w.block_size)
    w.block_size = block
    saved = getattr(w, "_ax_saved_runner", None)
    if saved is not None:
        mr = w.target_worker.model_runner
        mr.attn_backend, mr.decode_cuda_graph_runner = saved
        w._ax_saved_runner = None
    w._ax_round_width = None
    w._ax_tree_round = None
    state.kind = None
    state.host_kv_len = None
    state.round_info.update(backend=None, graph=None, width=None)
    sets = getattr(w, "_ax_draft_sets", None)
    if not sets or batch is None or batch.forward_mode.is_extend() or batch.is_extend_in_batch:
        return
    w._ax_tree_round = (id(batch), tree_round(batch))
    if w._ax_tree_round[1]:
        return
    wid = width(block)
    w._ax_round_width = wid
    if wid in sets:
        _enter_draft_width(w, wid)


def install():
    """With CUDA graphs for one request only (tree rounds), SGLang's in-graph draft sampler has
    buffers for one request, but the stock draft step stages its sampling parameters for every
    batch. A batch larger than the sampler runs the drafter eager and never reads the sampler, so the
    staging is skipped there."""
    import fc1_dflash_worker as W
    cls = getattr(W, "_SelectorDraftSampler", None)
    if cls is None or getattr(cls, "_ax_chain_guard", False):
        return
    original = cls.stage_sampling_params

    def stage_sampling_params(self, *, bs, sampling_info):
        if bs > self.greedy_mask.shape[0]:
            return None
        return original(self, bs=bs, sampling_info=sampling_info)

    patch_attribute(cls, 'stage_sampling_params', stage_sampling_params)
    patch_attribute(cls, '_ax_chain_guard', True)
    if _CHAIN_ATTN and not _GRAPH_WIDTHS:
        raise RuntimeError(f"ax-engine: AX_CHAIN_ATTN={_CHAIN_ATTN} needs AX_CHAIN_GRAPH_WIDTHS (a graph runner per "
                           "chain width); without them every chain round would run on the main backend")
    if _GRAPH_WIDTHS:
        _install_graphs(W)
    else:
        _install_main_only()
    if not getattr(cls_w := W.DFlashTreeWorkerV2, "_ax_checked", False):
        init0 = cls_w.init_cuda_graphs

        def init_cuda_graphs(self):
            activate_worker(self)
            return init0(self)
        patch_attribute(cls_w, 'init_cuda_graphs', init_cuda_graphs)
        patch_attribute(cls_w, '_ax_checked', True)
    if os.environ.get("AX_NXT_DEBUG") == "1":
        import ax_nxt_debug   # loaded next to ax_chain (the plugin puts this directory on sys.path)
        ax_nxt_debug.install(W.DFlashTreeWorkerV2)


def is_tree_runner(runner):
    """True for graph runners of a server running the tree algorithm. The capture patches below are
    global (SGLang's DecodeCudaGraphRunner), so a server that loads the plugin for something else (stock
    EAGLE / MTP, e.g. with AX_QSA_RING) must keep SGLang's own batch sizes."""
    algo = getattr(getattr(runner.model_runner, "server_args", None), "speculative_algorithm", None)
    return str(algo or "").upper() == "FC1_DFLASH_TREE"


def _install_main_only():
    """Servers without private chain runners (AX_CHAIN_GRAPH_WIDTHS unset): SGLang's main target runner
    is the only graph runner.

    - Tree on: it captures batch 1 only, with the tree verify input (the fused tree hooks handle one
      request; a capture at a larger batch with the tree input fails), and every chain round runs
      eager (it would otherwise be admitted to the tree graph at the tree's width).
    - Tree off: it captures the stock chain input at the block width, and a chain round narrowed
      below that width runs eager (the graph holds block-width buffers).
    """
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner as R
    if getattr(R, "_ax_main_only", False):
        return
    original_capture = R.capture

    def capture(self):

        if not is_tree_runner(self):
            return original_capture(self)
        if not self.model_runner.is_draft_worker:
            activate_worker(self.model_runner)
        previous_kind = current_runtime().kind
        main = not self.model_runner.is_draft_worker
        if main and _TREE:
            current_runtime().kind = "tree"
            self.capture_bs = [b for b in self.capture_bs if b == 1]
            self.compile_bs = [b for b in (self.compile_bs or []) if b == 1]
        else:
            current_runtime().kind = "chain"           # stock capture input (plugin.tree_capture_info)
        try:
            return original_capture(self)
        finally:
            current_runtime().kind = previous_kind
            if main and _TREE:
                self.max_bs = 1

    can_run = R.can_run_graph

    def can_run_graph(self, forward_batch):
        if is_tree_runner(self) and current_runtime().kind == "chain" and not self.model_runner.is_draft_worker:
            captured = getattr(self, "speculative_num_draft_tokens", None)
            if _TREE or captured is None or current_runtime().round_info.get("width") not in (None, captured):
                return False
        return can_run(self, forward_batch)

    patch_attribute(R, 'capture', capture)
    patch_attribute(R, 'can_run_graph', can_run_graph)
    patch_attribute(R, '_ax_main_only', True)


def _install_graphs(W):
    """CUDA graphs per round kind and width (design, Stage 0 item 3).

    - The main target runner (SGLang's) captures only batch 1, with the tree verify input when the
      tree is on: the tree graph.
    - Each chain width W in AX_CHAIN_GRAPH_WIDTHS gets its own runner with its own attention backend,
      built with speculative_eagle_topk = 1 and speculative_num_draft_tokens = W (as SGLang's UNO
      worker builds its private runner), capturing the stock DFlash chain verify at every batch size.
      A private backend keeps the main runner's captured buffers untouched (a second
      init_cuda_graph_state on a shared backend would reallocate them under the captured tree graph).
    - A chain round swaps model_runner.attn_backend and decode_cuda_graph_runner to its width's pair
      for the whole round (verify and the recurrent-state commit, which reads the backend's forward
      metadata); begin_round restores them at the next call.
    """
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner as R
    if getattr(R, "_ax_chain_graphs", False):
        return
    original_capture = R.capture

    def capture(self):

        if not is_tree_runner(self):
            return original_capture(self)
        if not self.model_runner.is_draft_worker:
            activate_worker(self.model_runner)
        previous_kind = current_runtime().kind
        target = not self.model_runner.is_draft_worker
        width = getattr(self, "_ax_chain_width", None)
        main = target and width is None
        if main:
            # one request only: the tree graph (tree on) or an unused batch-1 chain graph (tree off;
            # every chain round runs on a private runner)
            self.capture_bs = [b for b in self.capture_bs if b == 1]
            self.compile_bs = [b for b in (self.compile_bs or []) if b == 1]
        current_runtime().kind = "tree" if main and _TREE else "chain"
        try:
            return original_capture(self)
        finally:
            current_runtime().kind = previous_kind
            if main:
                self.max_bs = 1

    patch_attribute(R, 'capture', capture)
    patch_attribute(R, '_ax_chain_graphs', True)

    cls = W.DFlashTreeWorkerV2
    original_init = cls.init_cuda_graphs

    def init_cuda_graphs(self):
        out = original_init(self)
        _build_chain_runners(self, R)

        mr = self.target_worker.model_runner
        current_runtime().main_backend = _backend_name(mr, main=True)
        if mr.decode_cuda_graph_runner is not None:
            _count_replays(mr.decode_cuda_graph_runner, None, _backend_name(mr, main=True))
        if os.environ.get("AX_FP8_PROBE") == "1":
            import ax_fp8_probe
            ax_fp8_probe.run(mr)
        return out

    patch_attribute(cls, 'init_cuda_graphs', init_cuda_graphs)


def _backend_name(mr, main=False):
    """Name of the backend a target round runs on: the main runner's decode backend, or AX_CHAIN_ATTN."""
    if main or not _CHAIN_ATTN:
        return getattr(mr, "decode_attention_backend_str", None)
    return _CHAIN_ATTN


def _count_replays(runner, wid, name):
    """Record, per target verify on this graph runner, whether the captured graph replays (SGLang
    asks can_run_graph once per forward) or the round runs eager."""
    original = runner.can_run_graph

    def can_run_graph(forward_batch):
        ok = bool(original(forward_batch))
        key = f"{'tree' if wid is None else wid}:{forward_batch.batch_size}:{'graph' if ok else 'eager'}"
        current_runtime().stats["replay"][key] = current_runtime().stats["replay"].get(key, 0) + 1
        current_runtime().round_info["graph"] = ok
        return ok

    runner.can_run_graph = can_run_graph
    runner._ax_backend_name = name


def _build_backend(mr, wid):
    """Private attention backend of a chain runner. With AX_CHAIN_ATTN it is built by name in that
    backend's role: trtllm_mha decides its KV access checks and native-FP4 setup from the runner's
    prefill/decode role strings (trtllm_mha_backend.py:218-240) and the Gated-DeltaNet wrapper
    asserts on them (attention_registry.py:471-475), so both are set for the construction (as in an
    all-trtllm_mha server) and restored; SGLang records the pair on the backend object too."""
    if not _CHAIN_ATTN:
        return mr._get_attention_backend(init_new_workspace=True)
    from sglang.srt.model_executor.model_runner_components.attention_backend_setup import _build_backend_from_str
    roles = (mr.prefill_attention_backend_str, mr.decode_attention_backend_str)
    try:
        mr.prefill_attention_backend_str = mr.decode_attention_backend_str = _CHAIN_ATTN
        backend = _build_backend_from_str(model_runner=mr, backend_str=_CHAIN_ATTN, init_new_workspace=True)
    finally:
        mr.prefill_attention_backend_str, mr.decode_attention_backend_str = roles
    backend.prefill_attention_backend_str = backend.decode_attention_backend_str = _CHAIN_ATTN
    return backend


def _build_chain_runners(w, R):
    from sglang.srt.runtime_context import get_spec
    mr = w.target_worker.model_runner
    w._ax_chain_runners = {}
    if mr.decode_cuda_graph_runner is None:
        if _CHAIN_ATTN:
            raise RuntimeError("ax-engine: AX_CHAIN_ATTN needs CUDA graphs (no private chain runner is built without them)")
        return
    missing = [x for x in _configured_widths(int(w.block_size)) if x not in _GRAPH_WIDTHS]
    if missing:
        # a chain round of a width without a private runner would fail in _use_chain_runner
        raise RuntimeError(f"ax-engine: chain widths {missing} have no graph runner "
                           f"(AX_CHAIN_GRAPH_WIDTHS={_GRAPH_WIDTHS}); add them or unset AX_CHAIN_GRAPH_WIDTHS")
    for wid in _GRAPH_WIDTHS:
        free0 = torch.cuda.mem_get_info()[0]
        flag = mr.init_new_workspace
        try:
            with get_spec().override(speculative_eagle_topk=1, speculative_num_draft_tokens=wid):
                backend = _build_backend(mr, wid)
                runner = R.__new__(R)
                runner._ax_chain_width = wid
                R.__init__(runner, mr, attn_backend=backend, speculative_num_draft_tokens=wid)
        finally:
            mr.init_new_workspace = flag
        _count_replays(runner, wid, _backend_name(mr))
        w._ax_chain_runners[wid] = runner
        used = (free0 - torch.cuda.mem_get_info()[0]) / 2 ** 30
        print(f"[ax_chain] chain graph runner width {wid}: backend {_backend_name(mr)} "
              f"({type(backend).__name__}), batch sizes {runner.capture_bs}, {used:.2f} GB", flush=True)
    _build_draft_runners(w, R)


# every buffer `_ensure_draft_block_buffers` (re)allocates together: it sizes them all from the capacity
# of _draft_block_ids_buf, so a set that swapped only some of them would mix capacities (a width-16 host
# length buffer for 14 requests next to width-8 buffers for 15 crashed the stress run)
_DRAFT_BUFFERS = ("_draft_block_ids_buf", "_draft_block_positions_buf", "_draft_block_tokens_buf",
                  "_draft_verify_out_cache_loc_buf", "_draft_block_end_buf", "_draft_seq_lens_cpu_buf")


def _check_buffer_set(w, where):
    caps = {n: int(getattr(w, n).shape[0]) for n in _DRAFT_BUFFERS if getattr(w, n, None) is not None}
    if caps and (len(set(caps.values())) > 1 or len(caps) != len(_DRAFT_BUFFERS)):
        raise RuntimeError(f"ax-engine: draft buffers out of step {where}: {caps}")


def _build_draft_runners(w, R):
    """Per drafting width W < B: a W-token sampler folded into a private drafter graph runner (own
    draft attention backend), as SGLang builds the drafter graph for a block-W server."""
    import fc1_dflash_worker as W
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.draft_worker_common import make_draft_sampler_capture_hook

    w._ax_draft_sets = {}
    dmr = w.draft_model_runner
    if not _DRAFT_WIDTHS or dmr.decode_cuda_graph_runner is None or w._draft_sampler is None:
        return
    block = int(w.block_size)
    for wid in _DRAFT_WIDTHS:
        if wid not in _GRAPH_WIDTHS or not 2 <= wid < block:
            raise ValueError(f"ax-engine: drafting width {wid} needs a chain graph width below the block {block}")
        free0 = torch.cuda.mem_get_info()[0]
        hooks, flag = dmr.capture_tail_hooks, dmr.init_new_workspace
        try:
            w.block_size = wid
            sampler = w._maybe_build_draft_sampler()
            if sampler is None:
                raise RuntimeError("ax-engine: no in-graph draft sampler for a narrower drafting width")
            sampler.ax_chain_only = True     # ax_lattice: chain rounds only, no tree lattice needed
            dmr.capture_tail_hooks = [make_draft_sampler_capture_hook(sampler)]
            current_runtime().draft_width_override = wid
            with W.draft_tp_context(w.draft_parallel_layout), get_spec().override(speculative_num_draft_tokens=wid):
                backend = dmr._get_attention_backend(init_new_workspace=True)
                runner = R.__new__(R)
                runner._ax_chain_width = wid
                R.__init__(runner, dmr, attn_backend=backend, speculative_num_draft_tokens=wid)
        finally:
            current_runtime().draft_width_override = None
            dmr.capture_tail_hooks, dmr.init_new_workspace = hooks, flag
            w.block_size = block
        spec_info = W.make_draft_block_spec_info(draft_token_num=wid, device=w.device)
        w._ax_draft_sets[wid] = {"runner": runner, "sampler": sampler, "buffers": None, "spec_info": spec_info}
        used = (free0 - torch.cuda.mem_get_info()[0]) / 2 ** 30
        print(f"[ax_chain] drafter graph runner width {wid}: batch sizes {runner.capture_bs}, {used:.2f} GB", flush=True)


def _enter_draft_width(w, wid):
    """Before the draft step of a chain round that drafts at width W: swap block size, drafter graph
    runner and attention backend, in-graph sampler and the block-sized draft buffers."""
    d = w._ax_draft_sets[wid]
    dmr = w.draft_model_runner
    w._ax_draft_saved = (dmr.attn_backend, dmr.decode_cuda_graph_runner, w._draft_sampler,
                         w._draft_block_spec_info, {n: getattr(w, n, None) for n in _DRAFT_BUFFERS})
    dmr.attn_backend, dmr.decode_cuda_graph_runner = d["runner"].attn_backend, d["runner"]
    w._draft_sampler = d["sampler"]
    w._draft_block_spec_info = d["spec_info"]
    bufs = d["buffers"] or {n: None for n in _DRAFT_BUFFERS}   # None: _ensure_draft_block_buffers allocates at W
    for n, v in bufs.items():
        setattr(w, n, v)
    _check_buffer_set(w, f"entering width {wid}")
    w.block_size = wid
    w._ax_draft_width = wid


def _leave_draft_width(w):
    wid = getattr(w, "_ax_draft_width", None)
    if wid is None:
        return
    d = w._ax_draft_sets[wid]
    d["buffers"] = {n: getattr(w, n, None) for n in _DRAFT_BUFFERS}
    dmr = w.draft_model_runner
    dmr.attn_backend, dmr.decode_cuda_graph_runner, w._draft_sampler, w._draft_block_spec_info, bufs = w._ax_draft_saved
    for n, v in bufs.items():
        setattr(w, n, v)
    _check_buffer_set(w, f"leaving width {wid}")
    w._ax_draft_width = None


def _use_chain_runner(w, wid):
    runners = getattr(w, "_ax_chain_runners", None)
    if not runners:
        return
    runner = runners.get(wid)
    if runner is None:
        raise ValueError(f"ax-engine: no chain graph runner for width {wid} (AX_CHAIN_GRAPH_WIDTHS)")
    mr = w.target_worker.model_runner
    w._ax_saved_runner = (mr.attn_backend, mr.decode_cuda_graph_runner)
    mr.attn_backend, mr.decode_cuda_graph_runner = runner.attn_backend, runner
    current_runtime().round_info["backend"] = runner._ax_backend_name


def _topk_backends(w):
    """Target attention backends that read `topk` at run time. The plugin declares
    speculative_eagle_topk = 2 (a tree server), and SGLang's hybrid linear-attention backend then
    takes its tree verify path (parent links from spec_info) for every target verify; a chain round
    must take the chain path (topk 1) to run the same kernels as stock DFlash."""
    found = getattr(w, "_ax_topk_backends", None)
    if found is None:
        found, seen, todo = [], set(), [w.target_worker.model_runner.attn_backend]
        while todo:
            b = todo.pop()
            if b is None or id(b) in seen:
                continue
            seen.add(id(b))
            if isinstance(getattr(b, "topk", None), int):
                found.append((b, b.topk))
            for name in ("full_attn_backend", "linear_attn_backend", "attn_backend_list", "primary", "decode_backend", "prefill_backend"):
                v = getattr(b, name, None)
                todo.extend(v if isinstance(v, (list, tuple)) else [v])
        w._ax_topk_backends = found
    return found


def _set_topk(w, chain):
    for b, tree_topk in _topk_backends(w):
        b.topk = 1 if chain else tree_topk


def choose(w, batch, *, draft_features_ready):
    """Kind of this decode round; resets the single-request tree plan state for chain rounds."""

    import fc1_plan
    decided = getattr(w, "_ax_tree_round", None)
    wants_tree = decided[1] if decided is not None and decided[0] == id(batch) else tree_round(batch)
    # The builder reads this round's logits or lattice from the folded sampler.
    is_tree = wants_tree and draft_features_ready
    if is_tree and getattr(w, "_ax_draft_width", None) is not None:
        raise RuntimeError("ax-engine: tree round after drafting at a chain width")   # begin_round disagreed
    if is_tree:
        current_runtime().kind = "tree"
        current_runtime().stats["tree"] += 1
    else:
        current_runtime().kind = "chain"
        current_runtime().stats["chain"] += 1
        current_runtime().host_kv_len = None
    _set_topk(w, current_runtime().kind == "chain")
    if _CHECK:
        _check_alloc(batch)
    current_runtime().round_index += 1
    if current_runtime().round_index % 1000 == 0:
        print(f"[ax_chain] round {current_runtime().round_index} {current_runtime().stats}", flush=True)
    return current_runtime().kind


def _swap_accept_buffers(w, wid):
    """Keep one set of stock accept/bonus buffers per width (they are sized by block_size)."""
    names = ("_accept_bonus_buffer_cap", "_accept_len_buf", "_commit_lens_bufs", "_bonus_id_bufs",
             "_out_tokens_bufs", "_new_seq_lens_bufs", "_accept_bonus_buffer_slot")
    cur = getattr(w, "_ax_accept_width", None)
    if cur == wid:
        return
    store = w.__dict__.setdefault("_ax_accept_sets", {})
    if cur is not None:
        store[cur] = {n: getattr(w, n, None) for n in names}
    saved = store.get(wid)
    if saved is not None:
        for n, v in saved.items():
            setattr(w, n, v)
    else:
        w._accept_bonus_buffer_cap = 0          # _ensure_accept_bonus_buffers allocates at this width
        w._accept_bonus_buffer_slot = 0
    w._ax_accept_width = wid


def narrow(w, batch, draft_tokens, positions, cache_loc_2d, prefix_lens, kv_loc_plan, block):
    """Cut the round to its chain width. Returns the stock locals the verify uses next."""
    bs = draft_tokens.shape[0]
    wid = w._ax_round_width or width(block)   # decided in begin_round when the round drafted at its width
    current_runtime().stats["chain_width"][wid] = current_runtime().stats["chain_width"].get(wid, 0) + 1
    current_runtime().round_info["width"] = wid
    _swap_accept_buffers(w, wid)
    _use_chain_runner(w, wid)
    if _CHAIN_ATTN and current_runtime().round_info["backend"] != _CHAIN_ATTN:
        current_runtime().stats["chain_on_main"] = current_runtime().stats.get("chain_on_main", 0) + 1
        raise RuntimeError(f"ax-engine: chain round of width {wid} would run on the main backend, not {_CHAIN_ATTN}")
    if wid == block:
        w.block_size = block
        return draft_tokens, positions, cache_loc_2d.reshape(-1), cache_loc_2d, kv_loc_plan, block
    tokens = draft_tokens[:, :wid].contiguous()
    if w._selector_sample is not None:
        candidate_ids, proposal_probs = w._selector_sample
        # The first token is the anchor. Keep the original proposal law for each retained token.
        w._selector_sample = (candidate_ids[:, :wid - 1], proposal_probs[:, :wid - 1])
    pos = positions.view(bs, block)[:, :wid].contiguous().reshape(-1)
    loc2d = cache_loc_2d[:, :wid].contiguous()
    loc = loc2d.reshape(-1)
    draft_input = batch.spec_info
    plan = w.target_worker.model_runner.kv_index_translator.plan(
        req_pool_indices=batch.req_pool_indices,
        seq_lens=prefix_lens,
        seq_lens_cpu=batch.seq_lens_cpu if batch.seq_lens_cpu is not None else draft_input.nxt_kv_lens_cpu,
        write_virtual=loc,
        read_extent=wid,
    )
    w.block_size = wid
    return tokens, pos, loc, loc2d, plan, wid


def record_accept(w, batch, candidates, accept_len):
    """After the stock accept of a chain round: token before the next anchor, per request; oracle dump."""
    if current_runtime().kind != "chain":
        return
    before = candidates.gather(1, accept_len.to(torch.int64)[:, None])[:, 0]
    for i, req in enumerate(batch.reqs):
        req._ax_pre1 = before[i:i + 1]
    if _DUMP:
        _dump(batch, "chain", candidates, accept_len)


def record_tree(batch, pre1, tree_tokens, accept_index, accept_lens):
    """After a tree round: same bookkeeping as record_accept."""
    batch.reqs[0]._ax_pre1 = pre1
    if current_runtime().round_info["backend"] is None:
        current_runtime().round_info["backend"] = current_runtime().main_backend
    if _DUMP:
        path = tree_tokens.gather(1, accept_index.clamp(min=0).long())
        _dump(batch, "tree", path, accept_lens.to(torch.int64) - 1, nodes=tree_tokens)


def pre1_for(req):
    """Token before the anchor for the tree builder: kept from the request's last round of either
    kind; on its first decode round the last prompt token."""
    return getattr(req, "_ax_pre1", None)


def _dump(batch, kind, tokens, accept_len, nodes=None):
    toks = tokens.tolist()
    acc = accept_len.tolist()
    with open(_DUMP, "a") as f:
        for i, req in enumerate(batch.reqs):
            row = {"round": current_runtime().round_index, "rid": req.rid, "kind": kind, "bs": len(batch.reqs),
                   "tokens": toks[i], "accept": int(acc[i]),
                   "backend": current_runtime().round_info["backend"], "graph": current_runtime().round_info["graph"]}
            if nodes is not None:
                row["nodes"] = nodes.tolist()[i]
            f.write(json.dumps(row) + "\n")


import atexit   # noqa: E402


@atexit.register
def _report():
    for stats in runtime_statistics():
        print(f"[ax_chain] rounds {stats}", flush=True)
