"""Reuse fixed FlashInfer FA2 replay inputs for uniform native DFlash."""

import logging
import sys

from .ax_dflash_activity import activity
from .compat import installation, patch_attribute, rebuild_function


FEATURE = "AX_DFLASH_PLAN_REUSE"
logger = logging.getLogger(__name__)
_installed = False


def _graph_inputs(wrapper, spec, rows, total_prefix, device):
    """Return immutable query offsets and a view of the captured page-ID buffer."""
    if (spec.custom_mask is not None or spec.ragged_verify_layout is not None
            or spec.draft_token_num != 8 or rows == 0):
        return None
    backing = wrapper._paged_kv_indices_buf
    needed = total_prefix + rows * spec.draft_token_num
    if not 0 < needed <= backing.numel() or backing.device != device:
        return None

    # The graph wrapper is already bound to one capture bucket and device.
    cached = getattr(wrapper, "_ax_dflash_qo_offsets", None)
    if cached is None or cached[0] != rows:
        import torch

        host = torch.arange(0, (rows + 1) * spec.draft_token_num,
                            spec.draft_token_num, dtype=torch.int32, device="cpu")
        # Publish an immutable device copy once, including when capture used a different stream.
        cached = (rows, host, host.to(device))
        wrapper._ax_dflash_qo_offsets = cached
        logger.info("DFlash plan reuse active for batch size %d", rows)
    return cached[1], cached[2], backing[:needed]


def install():
    if _installed:
        return

    from sglang.srt.layers.attention import flashinfer_backend as fi
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    generate = rebuild_function(DFlashVerifyInput, "generate_attn_arg_prefill", (
        ("kind: IdSpaceKind = IdSpaceKind.FULL,", "kind: IdSpaceKind = IdSpaceKind.FULL,\n"
         "    qo_indptr_buf: Optional[torch.Tensor] = None,", 1),
        ("qo_indptr = torch.arange(", "qo_indptr = qo_indptr_buf if qo_indptr_buf is not None else torch.arange(", 1),
    ), feature=FEATURE)
    updater = rebuild_function(fi.FlashInferIndicesUpdaterPrefill, "call_begin_forward", (
        ("bs = len(seq_lens)", "bs = len(seq_lens)\n    graph_inputs = None", 1),
        ("if spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY:\n",
         """if spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY:
            if (
                seq_lens_cpu is not None
                and not use_ragged
                and window_left < 0
                and not use_sliding_window_kv_pool
                and self.attn_backend.dispatch_reason is None
                and getattr(wrapper_paged.begin_forward, "func", None) is fast_prefill_plan
            ):
                graph_inputs = _ax_dflash_graph_inputs(
                    wrapper_paged, spec_info, bs, paged_kernel_lens_sum, req_pool_indices.device
                )
            dflash_buffers = (
                {"qo_indptr_buf": graph_inputs[1], "kv_indices_buf": graph_inputs[2]}
                if graph_inputs is not None else {}
            )
""", 1),
        ("kv_start_idx=kv_start_idx,\n                )\n            )\n        else:",
         "kv_start_idx=kv_start_idx,\n                    **dflash_buffers,\n                )\n            )\n        else:", 1),
        ("qo_indptr_host = torch.arange(", "qo_indptr_host = graph_inputs[0] if graph_inputs is not None else torch.arange(", 1),
        ("if eagle_verify:\n            paged_plan_kwargs[\"uniform_verify_width\"] = num_tokens_per_req",
         'if graph_inputs is not None:\n            paged_plan_kwargs["ax_packed_page_ids"] = True\n'
         '        if eagle_verify:\n            paged_plan_kwargs["uniform_verify_width"] = num_tokens_per_req', 1),
    ), feature=FEATURE)
    fast_plan = rebuild_function(fi, "fast_prefill_plan", (
        ("uniform_verify_width: Optional[int] = None,", "uniform_verify_width: Optional[int] = None,\n"
         "    ax_packed_page_ids: bool = False,", 1),
        ("""self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
        paged_kv_indices,
        non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
    )""",
         """if ax_packed_page_ids:
        if _ax_dflash_activity is not None:
            _ax_dflash_activity.paths.add("plan_reuse")
    else:
        self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
            paged_kv_indices,
            non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
        )""", 1),
    ), feature=FEATURE)
    worker = rebuild_function(DFlashWorkerV2, "forward_batch_generation", (
        ("""verify_host_seq_lens = seq_lens_cpu_backup + block_size
        batch.seq_lens_cpu = verify_host_seq_lens
        batch.seq_lens_sum = int(verify_host_seq_lens.sum())""",
         """if not self.use_compact_draft_cache and seq_lens_cpu.dtype == seq_lens_cpu_backup.dtype:
            batch.seq_lens_cpu = seq_lens_cpu
            batch.seq_lens_sum = draft_seq_lens_sum
        else:
            verify_host_seq_lens = seq_lens_cpu_backup + block_size
            batch.seq_lens_cpu = verify_host_seq_lens
            batch.seq_lens_sum = int(verify_host_seq_lens.sum())""", 1),
    ), feature=FEATURE)

    with installation():
        patch_attribute(fi, "_ax_dflash_activity", activity)
        patch_attribute(fi, "_ax_dflash_graph_inputs", _graph_inputs)
        patch_attribute(fi, "fast_prefill_plan", fast_plan)
        patch_attribute(fi.FlashInferIndicesUpdaterPrefill, "call_begin_forward", updater)
        patch_attribute(DFlashVerifyInput, "generate_attn_arg_prefill", generate)
        patch_attribute(DFlashWorkerV2, "forward_batch_generation", worker)
        patch_attribute(sys.modules[__name__], "_installed", True)
