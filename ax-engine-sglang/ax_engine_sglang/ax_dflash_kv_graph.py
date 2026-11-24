"""Capture the native DFlash2 accepted-prefix KV commit at its current call site."""

from copy import copy
from functools import wraps
import logging
import sys

from .ax_dflash_activity import activity
from .compat import installation, patch_attribute


MAX_BATCH_SIZE = 16
logger = logging.getLogger(__name__)
_installed = False


class _AppendGraph:
    def __init__(self, torch, draft_tp_context, worker, helper, pool, inputs, stream):
        hidden, positions, slots, lengths = inputs
        draft = worker.draft_model
        # The source helper's scratch can grow when an eager append runs.
        self.helper = copy(helper)
        self.helper._workspace_capacity = 0
        self.hidden = torch.zeros_like(hidden)
        self.positions = torch.zeros(positions.shape, device=hidden.device, dtype=torch.int64)
        self.slots = torch.zeros(slots.shape, device=hidden.device, dtype=torch.int64)
        self.lengths = torch.zeros(lengths.shape, device=hidden.device, dtype=torch.int32)
        self.graph = torch.cuda.CUDAGraph()

        def write_layer(layer_idx, key, value):
            attn = draft.layers[layer_idx].self_attn.attn
            pool.set_kv_buffer_prefix_valid(
                attn, self.slots, self.lengths, key, value, attn.k_scale, attn.v_scale,
            )

        def materialize():
            self.helper.materialize(
                ctx_hidden=draft.project_target_hidden(self.hidden),
                positions=self.positions,
                write_layer_kv=write_layer,
            )

        capture_stream = torch.cuda.Stream(device=hidden.device)
        capture_stream.wait_stream(stream)
        try:
            with torch.cuda.stream(capture_stream), draft_tp_context(worker.draft_parallel_layout):
                # Zero commit lengths make warmup and capture safe for live cache slots.
                materialize()
                self.graph.capture_begin(capture_error_mode="thread_local")
                try:
                    materialize()
                finally:
                    self.graph.capture_end()
        finally:
            stream.wait_stream(capture_stream)

    def replay(self, inputs):
        hidden, positions, slots, lengths = inputs
        self.hidden.copy_(hidden)
        self.positions.copy_(positions)
        self.slots.copy_(slots)
        self.lengths.copy_(lengths)
        self.graph.replay()


def _wrap_append(original, torch, pool_type, draft_tp_context):
    @wraps(original)
    def append(self, *, target_hidden, cache_loc, positions, cache_loc_2d=None,
               commit_lens=None, extend_lens=None):
        inputs = target_hidden, positions, cache_loc_2d, commit_lens
        if extend_lens is None and _try_replay(self, inputs, cache_loc, torch, pool_type, draft_tp_context):
            return
        return original(self, target_hidden=target_hidden, cache_loc=cache_loc, positions=positions,
                        cache_loc_2d=cache_loc_2d, commit_lens=commit_lens, extend_lens=extend_lens)

    return append


def _try_replay(worker, inputs, cache_loc, torch, pool_type, draft_tp_context):
    hidden, positions, slots, lengths = inputs
    if slots is None or lengths is None or hidden is None or hidden.device.type != "cuda":
        return False
    helper = worker._fused_kv_helper
    if (worker.block_size != 8 or worker.lilicorr is not None or helper is None
            or not worker._use_fused_kv_materialize or slots.ndim != 2):
        return False

    bs, width = slots.shape
    if not 1 <= bs <= MAX_BATCH_SIZE or width != worker.block_size:
        return False
    draft = worker.draft_model
    features = getattr(draft.fc, "in_features", None)
    if (hidden.shape != (bs * width, features) or not hidden.is_contiguous()
            or cache_loc.shape != (bs * width,) or positions.shape != (bs * width,)
            or lengths.shape != (bs,) or any(x.device != hidden.device for x in inputs)
            or hidden.dtype != draft.fc.weight.dtype):
        return False

    pool = worker.draft_model_runner.token_to_kv_pool
    rope_cache = helper.rotary_emb.cos_sin_cache
    if (type(pool) is not pool_type or pool.use_hnd or pool.v_row_dim != pool.row_dim
            or pool.dtype != hidden.dtype or pool.store_dtype != pool.dtype
            or helper.max_position_hint is None or rope_cache.shape[0] <= helper.max_position_hint
            or rope_cache.device != hidden.device or torch.cuda.is_current_stream_capturing()):
        return False

    stream = torch.cuda.current_stream(hidden.device)
    # Retain the original tensors, and fall back if SGLang rebinds any captured buffers.
    bindings = (helper, pool, rope_cache, draft.fc.weight, draft.hidden_norm.weight,
                *pool.k_buffer, *pool.v_buffer)
    state = getattr(worker, "_ax_dflash_kv_graph_state", None)
    if state is None:
        graphs = {}
        worker._ax_dflash_kv_graph_state = (stream, bindings, graphs)
    else:
        captured_stream, captured_bindings, graphs = state
        if (stream != captured_stream or len(captured_bindings) != len(bindings)
                or any(old is not new for old, new in zip(captured_bindings, bindings))):
            return False

    with torch.inference_mode():
        first_replay = bs not in graphs
        if first_replay:
            if activity is not None:
                activity.capture_attempts += 1
            try:
                graphs[bs] = _AppendGraph(torch, draft_tp_context, worker, helper, pool, inputs, stream)
            except Exception:
                graphs[bs] = None
                logger.warning("DFlash KV graph capture failed for batch size %d", bs, exc_info=True)
        append_graph = graphs[bs]
        if append_graph is None:
            return False
        append_graph.replay(inputs)
        if activity is not None:
            activity.paths.add("kv_graph")
        if first_replay:
            logger.info("DFlash KV graph replay active for batch size %d", bs)
    return True


def install():
    if _installed:
        return

    import torch
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    from sglang.srt.speculative.spec_utils import draft_tp_context

    name = "_append_target_hidden_to_draft_kv_by_loc"
    original = getattr(DFlashWorkerV2, name)
    with installation():
        patch_attribute(DFlashWorkerV2, name, _wrap_append(original, torch, MHATokenToKVPool, draft_tp_context))
        patch_attribute(sys.modules[__name__], "_installed", True)
