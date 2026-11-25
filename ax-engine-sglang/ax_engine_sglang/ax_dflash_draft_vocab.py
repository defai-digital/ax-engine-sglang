"""Use a smaller draft-only vocabulary for eligible native DFlash2 batches."""

from contextvars import ContextVar
from functools import wraps
import logging
import time
from types import SimpleNamespace

from .ax_dflash_activity import activity
from .compat import patch_attribute


FEATURE = "AX_DFLASH_DRAFT_VOCAB"
PATH = "draft_vocab"
GRAPH_VARIANT = "ax-dflash-draft-vocab"
# Round the Qwen leading and special-token ranges outward to NVFP4 scale tiles.
VOCAB_SIZE = 248_320
LEADING_ROWS = 148_352
TAIL_START = 247_936
SUBSET_SIZE = LEADING_ROWS + VOCAB_SIZE - TAIL_START
HISTORY = 64
RECHECK = 8

logger = logging.getLogger(__name__)
_capture_subset = ContextVar("ax_dflash_vocab_capture", default=None)
_replay_subset = ContextVar("ax_dflash_vocab_replay", default=None)


def in_subset(token):
    return 0 <= token < LEADING_ROWS or TAIL_START <= token < VOCAB_SIZE


def history_in_subset(prompt_ids, output_ids):
    remaining = max(0, HISTORY - len(output_ids))
    recent_prompt = prompt_ids[-remaining:] if remaining else ()
    return bool(recent_prompt or output_ids) and all(
        in_subset(token) for part in (recent_prompt, output_ids[-HISTORY:]) for token in part
    )


class History:
    """Position of a request's latest excluded token, updated from new outputs only."""

    __slots__ = ("prompt", "outputs", "scanned", "last_excluded")

    def __init__(self, prompt_ids, output_ids):
        self.prompt, self.outputs, self.scanned = prompt_ids, output_ids, 0
        self.last_excluded = -HISTORY - 1
        start = max(0, len(prompt_ids) - HISTORY)
        for index in range(len(prompt_ids) - 1, start - 1, -1):
            if not in_subset(prompt_ids[index]):
                self.last_excluded = index
                break

    def eligible(self, prompt_ids, output_ids):
        count = len(output_ids)
        if prompt_ids is not self.prompt or output_ids is not self.outputs or count < self.scanned:
            self.__init__(prompt_ids, output_ids)
        # Recheck the latest committed slots in case a pop was refilled between
        # checks. Tokens older than the window cannot change the result.
        for index in range(max(min(self.scanned, count) - RECHECK, count - HISTORY, 0), count):
            if not in_subset(output_ids[index]):
                self.last_excluded = len(prompt_ids) + index
        self.scanned = count
        total = len(prompt_ids) + count
        return total > 0 and self.last_excluded < total - HISTORY


def request_in_subset(req):
    history = getattr(req, "_ax_dflash_vocab_history", None)
    if history is None:
        history = History(req.origin_input_ids, req.output_ids)
        req._ax_dflash_vocab_history = history
    return history.eligible(req.origin_input_ids, req.output_ids)


def use_subset(batch):
    mode = batch.forward_mode
    sampling = batch.sampling_info
    if (mode.is_idle() or mode.is_extend() or batch.is_extend_in_batch or batch.has_grammar
            or (sampling is not None and not sampling.is_all_greedy)):
        return False
    return bool(batch.reqs) and all(request_in_subset(req) for req in batch.reqs)


def pack_weight(weight):
    import torch

    return torch.cat((weight[:LEADING_ROWS], weight[TAIL_START:VOCAB_SIZE]), dim=0)


def subset_topk(hidden, head, k, topk):
    import torch

    if head.quant_method is None:
        logits = torch.matmul(hidden.to(head.weight.dtype), head.weight.T)
    else:
        logits = head.quant_method.apply(head, hidden, None)
    values, indexes = topk(logits, k)
    indexes = indexes.long()
    # CandidateSelector uses global IDs to index its transition tables.
    ids = torch.where(indexes < LEADING_ROWS, indexes, indexes + TAIL_START - LEADING_ROWS)
    return ids, values


def pack_nvfp4_head(head):
    import torch

    # CUTLASS interleaves scales within groups of 128 output rows. Each
    # selected interval starts and ends on a group boundary.
    scale = head.weight_scale_interleaved.detach()
    return SimpleNamespace(
        quant_method=head.quant_method, weight=pack_weight(head.weight.detach()),
        weight_scale_interleaved=pack_weight(scale.view(torch.uint8)).view(scale.dtype),
        output_size_per_partition=SUBSET_SIZE, weights_padding_cols=head.weights_padding_cols,
        alpha=head.alpha, input_scale_inv=head.input_scale_inv,
    )


def install():
    import torch
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4LinearMethod
    from sglang.srt.models import dflash
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2, _SelectorDraftSampler

    build_sampler = DFlashWorkerV2._maybe_build_draft_sampler
    forward_batch = DFlashWorkerV2.forward_batch_generation
    compute_candidates = dflash.DFlash2DraftModel.compute_candidates
    capture_shape = DecodeCudaGraphRunner.capture_one_shape
    create_prefill_wrappers = FlashInferAttnBackend._create_prefill_wrappers
    graph_key = DecodeCudaGraphRunner._make_graph_key
    execute = DecodeCudaGraphRunner.execute

    @wraps(build_sampler)
    def build(worker):
        sampler = build_sampler(worker)
        if not isinstance(sampler, _SelectorDraftSampler):
            return sampler
        model = sampler.draft_model
        head = model.lm_head
        weight = getattr(head, "weight", None)
        args = worker.draft_model_runner.server_args
        if (args.enable_lora or args.enable_pdmux or weight is None or not weight.is_cuda
                or weight.ndim != 2 or weight.shape[0] < VOCAB_SIZE
                or int(getattr(head, "org_vocab_size", 0)) != VOCAB_SIZE):
            logger.info("%s kept the native draft head for this model or runner", FEATURE)
            return sampler

        method = getattr(head, "quant_method", None)
        if dflash.should_apply_lm_head_quant_method(head, method):
            if (type(method) is not ModelOptFp4LinearMethod or method.quant_mode != "w4a4"
                    or method.quant_config.group_size != 16 or method.quant_config.is_awq
                    or not get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
                    or weight.dtype != torch.uint8 or not weight.is_contiguous()):
                logger.info("%s kept the native draft head for this quantized layout", FEATURE)
                return sampler
            scales = head.weight_scale_interleaved
            if (not scales.is_contiguous() or scales.ndim != 2 or scales.shape[0] != weight.shape[0]
                    or scales.dtype != torch.float8_e4m3fn):
                logger.info("%s kept the native draft head for this block-scale layout", FEATURE)
                return sampler
            packed = pack_nvfp4_head(head)
        elif weight.dtype in (torch.bfloat16, torch.float16):
            packed = SimpleNamespace(quant_method=None, weight=pack_weight(weight.detach()))
        else:
            return sampler
        model._ax_dflash_vocab_head = packed
        logger.info("%s packed %d draft vocabulary rows (%s)", FEATURE, SUBSET_SIZE, packed.weight.dtype)
        return sampler

    @wraps(compute_candidates)
    def candidates(model, hidden):
        capture = _capture_subset.get()
        if capture is None or capture[0] is not model:
            return compute_candidates(model, hidden)
        ids, values = subset_topk(hidden, model._ax_dflash_vocab_head,
                                  model.candidate_selector.top_k, dflash._radix_topk)
        return ids, model._transform_unary_logits(values)

    @wraps(create_prefill_wrappers)
    def prefill_wrappers(attention, bs, use_custom_mask=False):
        capture = _capture_subset.get()
        if capture is not None and capture[1] is attention:
            # Both draft variants use the same attention geometry. Keep the
            # FlashInfer plan buffers referenced by the native graph alive,
            # and refresh those same buffers before replay of either variant.
            return attention.prefill_cuda_graph_metadata[bs]
        return create_prefill_wrappers(attention, bs, use_custom_mask)

    @wraps(capture_shape)
    def capture(runner, size, forward, stream_idx=None, variant_label=None, attention_variant=None):
        capture_shape(runner, size, forward, stream_idx, variant_label, attention_variant)
        model = runner.model_runner.model
        backend = runner.backend
        if (not runner.model_runner.is_draft_worker or runner.ragged_verify_mode
                or not isinstance(backend, FullCudaGraphBackend)
                or not isinstance(runner.attn_backend, FlashInferAttnBackend)
                or runner.attention_graph_variants is not None
                or not runner.capture_forward_mode.is_target_verify()
                or variant_label is not None or not hasattr(model, "_ax_dflash_vocab_head")):
            return
        # Full and subset graphs can alternate within one request. Isolate
        # their allocations even when they share input and sampler buffers.
        if not hasattr(runner, "_ax_dflash_vocab_pool"):
            runner._ax_dflash_vocab_pool = torch.cuda.graph_pool_handle()
        native_pool = backend._pool
        token = _capture_subset.set((model, runner.attn_backend))
        try:
            backend._pool = runner._ax_dflash_vocab_pool
            capture_shape(runner, size, forward, stream_idx, GRAPH_VARIANT, attention_variant)
        finally:
            backend._pool = native_pool
            _capture_subset.reset(token)
        model._ax_dflash_vocab_captured = True

    @wraps(graph_key)
    def key(runner, size, stream_idx=None, variant_label=None, attention_variant=None):
        selected = _replay_subset.get()
        if selected is not None and runner.model_runner.is_draft_worker and runner.model_runner.model is selected:
            variant_label = GRAPH_VARIANT
        return graph_key(runner, size, stream_idx, variant_label, attention_variant)

    @wraps(forward_batch)
    def forward(worker, batch, *args, **kwargs):
        model = worker.draft_model
        if not getattr(model, "_ax_dflash_vocab_captured", False):
            return forward_batch(worker, batch, *args, **kwargs)
        if activity is not None and activity.timing:
            started = time.perf_counter()
            selected = use_subset(batch)
            activity.add_host("draft_vocab_check", time.perf_counter() - started)
        else:
            selected = use_subset(batch)
        if not selected:
            return forward_batch(worker, batch, *args, **kwargs)
        token = _replay_subset.set(model)
        try:
            return forward_batch(worker, batch, *args, **kwargs)
        finally:
            _replay_subset.reset(token)

    @wraps(execute)
    def replay(runner, forward_batch, *args, **kwargs):
        result = execute(runner, forward_batch, *args, **kwargs)
        if activity is not None and runner._replay_graph_key.variant_label == GRAPH_VARIANT:
            activity.paths.add(PATH)
        return result

    patch_attribute(DFlashWorkerV2, "_maybe_build_draft_sampler", build)
    patch_attribute(DFlashWorkerV2, "forward_batch_generation", forward)
    patch_attribute(dflash.DFlash2DraftModel, "compute_candidates", candidates)
    patch_attribute(FlashInferAttnBackend, "_create_prefill_wrappers", prefill_wrappers)
    patch_attribute(DecodeCudaGraphRunner, "capture_one_shape", capture)
    patch_attribute(DecodeCudaGraphRunner, "_make_graph_key", key)
    patch_attribute(DecodeCudaGraphRunner, "execute", replay)
