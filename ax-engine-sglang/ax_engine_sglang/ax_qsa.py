# SPDX-License-Identifier: Apache-2.0
# The graph-metadata kernels below are adapted from SGLang 0.5.23
# (sglang/srt/layers/attention/qsa/graph_metadata.py, Apache-2.0); see NOTICE.
"""A wider QSA pending ring, so speculative verify windows longer than one compress group are correct.

Qwen4-Exp sparse attention (QSA, Qwen3.8-Flash-Next) compresses index keys in groups of `ratio` (4)
tokens. Keys of the open group wait in a per-request ring addressed as
`req_pool_idx * ratio + position % ratio`. A forward first writes every token's key into the ring and
then compresses the groups its rows complete, reading the members back from the ring. With a
speculative verify window of W tokens starting at position p, the members of a group completed inside
the window are overwritten by later window tokens whenever (p % ratio) + W > ratio: e.g. MTP with 2
tokens at p % 4 == 3 compresses group [p-3, p] with p+1's key in place of p-3's. SGLang 0.5.23 refuses
W > ratio outright, but W <= ratio is affected as well whenever the window crosses a group boundary.
The corruption reaches only the compressed index keys, i.e. which blocks the sparse attention selects,
so it shows only once a context is longer than the selection budget (2048 tokens).

With AX_QSA_RING=R (a multiple of ratio, e.g. 16) the ring holds R slots per request, addressed as
`req_pool_idx * R + position % R`. A window and the open group's earlier members then span at most
(ratio - 1) + W positions, so windows up to W = R - ratio + 1 tokens never overwrite a pending member,
and the backend's W <= ratio check is relaxed to that bound. AX_QSA_RING unset or 0 leaves SGLang as is.
"""
from ax_engine_sglang.compat import patch_attribute
from ax_engine_sglang.config import get_config
import os
import sys

RING = 0


def _ring(ratio):
    return RING if RING else ratio


def build_pending_ring_slots(*, token_to_batch_idx, req_pool_indices, sequence_lengths,
                             logical_positions, compress_ratio, is_extend):
    import torch
    ring = _ring(compress_ratio)
    rows = token_to_batch_idx.long()[: logical_positions.numel()]
    requests = req_pool_indices.long()[rows]
    positions = logical_positions.long()
    slots = requests * ring + positions % ring
    if is_extend:
        # tokens before the pending tail dump into rows [0, ring): request slot 0 is never allocated
        lengths = sequence_lengths.long()[rows]
        pending = positions >= (lengths // compress_ratio) * compress_ratio
        slots = torch.where(pending, slots, positions % ring)
    return slots


def build_group_ring_slots(*, req_pool_indices, group_end_positions, sequence_ids, compress_ratio):
    import torch
    ring = _ring(compress_ratio)
    requests = req_pool_indices.long()[sequence_ids]
    offsets = torch.arange(compress_ratio - 1, -1, -1, device=group_end_positions.device, dtype=torch.long)
    positions = (group_end_positions[:, None] - offsets[None, :]).clamp_min(0)
    return requests[:, None] * ring + positions % ring


def _graph_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def row_metadata_kernel(
        row_seq_lens_ptr, row_req_pool_ptr,
        compressed_lens_ptr, write_locs_ptr, page_table_ptr, logical_positions_ptr, state_slots_ptr,
        ring_locs_ptr,
        req_to_token_ptr, req_to_token_row_stride, max_pages,
        RATIO: tl.constexpr, FULL_PAGE: tl.constexpr, PAGE_BLOCK: tl.constexpr,
        seq_lens_ptr, req_pool_ptr, row_prefix_lens_ptr, bs, num_padding, extend_len, seq_offset,
        MODE: tl.constexpr, RING: tl.constexpr,
    ):
        row = tl.program_id(0)
        if MODE == 0 or MODE == 1:
            req_row = row if MODE == 0 else row // extend_len
            real = req_row < bs - num_padding
            base = tl.load(seq_lens_ptr + req_row, real, other=0).to(tl.int32)
            req = tl.load(req_pool_ptr + req_row, real, other=0).to(tl.int64)
            if MODE == 0:
                seq_len = tl.where(real, base + seq_offset, 1)
                prefix = tl.maximum(seq_len - 1, 0)
            else:
                seq_len = tl.where(real, base + row % extend_len + 1, 1)
                prefix = tl.where(real, base, 0)
            if tl.program_id(1) == 0:
                tl.store(row_seq_lens_ptr + row, seq_len)
                tl.store(row_req_pool_ptr + row, req.to(tl.int32))
                tl.store(row_prefix_lens_ptr + row, prefix)
        else:
            seq_len = tl.load(row_seq_lens_ptr + row).to(tl.int32)
            req = tl.load(row_req_pool_ptr + row).to(tl.int64)
        token_row = req * req_to_token_row_stride
        if tl.program_id(1) == 0:
            current = tl.maximum(seq_len - 1, 0)
            last_loc = tl.load(req_to_token_ptr + token_row + current).to(tl.int32)
            compressed = seq_len // RATIO
            tl.store(compressed_lens_ptr + row, compressed)
            boundary = (seq_len > 0) & (seq_len % RATIO == 0)
            write_loc = tl.where(boundary, last_loc // RATIO, 0)
            tl.store(write_locs_ptr + row, write_loc)
            tl.store(logical_positions_ptr + row, current)
            # ax-engine-sglang: the ring has RING slots per request (SGLang: RATIO)
            tl.store(state_slots_ptr + row, req * RING + (current % RING).to(tl.int64))
            ring_base = row.to(tl.int64) * RATIO
            for k in tl.static_range(RATIO):
                member = tl.maximum(current - (RATIO - 1 - k), 0)
                slot = req * RING + (member % RING).to(tl.int64)
                tl.store(ring_locs_ptr + ring_base + k, slot.to(tl.int32))
        table_row = page_table_ptr + row.to(tl.int64) * max_pages
        offs = tl.arange(0, PAGE_BLOCK)
        row_width_pages = req_to_token_row_stride // FULL_PAGE
        idx = tl.program_id(1) * PAGE_BLOCK + offs
        valid = idx < tl.minimum(max_pages, row_width_pages)
        loc = tl.load(req_to_token_ptr + token_row + idx * FULL_PAGE, mask=valid, other=0)
        tl.store(table_row + idx, tl.maximum(loc // FULL_PAGE, 0), mask=valid)

    @triton.jit
    def draft_metadata_kernel(
        buffers, seq_lens, req_pool, req_to_token, row_stride, bs, num_padding,
        MAX_PAGES: tl.constexpr, RATIO: tl.constexpr, FULL_PAGE: tl.constexpr, RING: tl.constexpr,
    ):
        for step in tl.static_range(len(buffers)):
            if tl.program_id(2) == step:
                buf = buffers[step]
                row_metadata_kernel(
                    buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
                    req_to_token, row_stride, MAX_PAGES, RATIO, FULL_PAGE, 128,
                    seq_lens, req_pool, buf[8], bs, num_padding,
                    extend_len=0, seq_offset=step + 1, MODE=0, RING=RING,
                )

    return row_metadata_kernel, draft_metadata_kernel


def _patch_graph_metadata(gm):
    import triton
    row_kernel, draft_kernel = _graph_kernels()

    def launch_graph_metadata(*, mode, bs, num_rows, seq_lens, req_pool_indices, extend_lens,
                              extend_len, num_padding, metadata, req_to_token, pool, seq_offset=0):
        indexer = metadata.indexer_metadata
        max_pages = indexer.graph_compressed_page_table.shape[1]
        row_seq_lens = metadata.sequence_lengths
        row_req_pool = metadata.row_req_pool_indices
        row_prefix_lens = indexer.graph_prefix_lengths
        if mode == 2:
            gm._qsa_graph_layout_kernel[(bs + 1,)](
                seq_lens, req_pool_indices, extend_lens if extend_lens is not None else row_seq_lens,
                row_seq_lens, row_prefix_lens, row_req_pool, bs, num_rows, num_padding, num_warps=1)
        row_kernel[(num_rows, triton.cdiv(max_pages, 128))](
            row_seq_lens, row_req_pool,
            indexer.graph_compressed_lengths, indexer.graph_write_locs, indexer.graph_compressed_page_table,
            indexer.decode_logical_positions, indexer.pending_ring_slots, indexer.graph_ring_group_locs,
            req_to_token, req_to_token.stride(0), max_pages,
            RATIO=indexer.compress_ratio, FULL_PAGE=pool.qsa_compressed_page_size * indexer.compress_ratio,
            PAGE_BLOCK=128, seq_lens_ptr=seq_lens, req_pool_ptr=req_pool_indices,
            row_prefix_lens_ptr=row_prefix_lens, bs=bs, num_padding=num_padding, extend_len=extend_len,
            seq_offset=seq_offset, MODE=mode, RING=_ring(indexer.compress_ratio), num_warps=1)

    def prepare_draft_graph_metadata(metadata, req_to_token, pool):
        buffers = tuple((m.sequence_lengths, m.row_req_pool_indices,
                         m.indexer_metadata.graph_compressed_lengths, m.indexer_metadata.graph_write_locs,
                         m.indexer_metadata.graph_compressed_page_table,
                         m.indexer_metadata.decode_logical_positions, m.indexer_metadata.pending_ring_slots,
                         m.indexer_metadata.graph_ring_group_locs, m.indexer_metadata.graph_prefix_lengths)
                        for m in metadata)
        ratio = pool.qsa_compress_ratio
        return (buffers, req_to_token, req_to_token.stride(0),
                metadata[0].indexer_metadata.graph_compressed_page_table.shape[1],
                ratio, pool.qsa_compressed_page_size * ratio, _ring(ratio))

    def launch_draft_graph_metadata(args, seq_lens, req_pool_indices, bs, num_padding):
        buffers, req_to_token, row_stride, max_pages, ratio, full_page, ring = args
        if bs == 0:
            return
        draft_kernel[(bs, triton.cdiv(max_pages, 128), len(buffers))](
            buffers, seq_lens, req_pool_indices, req_to_token, row_stride, bs,
            max(0, min(int(num_padding or 0), bs)), max_pages, ratio, full_page, ring, num_warps=1)

    patch_attribute(gm, 'launch_graph_metadata', launch_graph_metadata)
    patch_attribute(gm, 'prepare_draft_graph_metadata', prepare_draft_graph_metadata)
    patch_attribute(gm, 'launch_draft_graph_metadata', launch_draft_graph_metadata)


def max_window(ratio):
    ring = _ring(ratio)
    return ratio if ring == ratio else ring - ratio + 1   # ring == ratio: SGLang's own cap


CHECK = {}


def _record_check_positions(indexer, metadata, logical_positions, rope_positions, state_slots):
    """Diagnostic shadow of the slots passed to the real key/position store.

    Logical sequence offsets and multimodal RoPE coordinates are different. Keep
    both so image warmup cannot look like an overwritten text ring member.
    """
    import torch
    if metadata.is_cuda_graph:
        return
    pool = metadata.token_to_kv_pool
    if not hasattr(pool, "_ax_qsa_check_logical"):
        pool._ax_qsa_check_logical = torch.full_like(pool.qsa_rope_position_buffer[:, 0], -1)
        pool._ax_qsa_check_rope = torch.empty_like(pool.qsa_rope_position_buffer)
    if state_slots is None:
        state_slots = indexer._pending_ring_slots(
            metadata, logical_positions, metadata.compress_member_rows is not None)
    count = logical_positions.numel()
    slots = state_slots[:count].long()
    rope = (rope_positions[:count, None].expand(-1, 3) if rope_positions.ndim == 1
            else rope_positions[:, :count].transpose(0, 1))
    pool._ax_qsa_check_logical[slots] = logical_positions.to(dtype=pool._ax_qsa_check_logical.dtype)
    pool._ax_qsa_check_rope[slots] = rope.to(dtype=pool._ax_qsa_check_rope.dtype)


def _check_groups(indexer, metadata):
    """AX_QSA_CHECK=1: after a paged (decode / verify) forward has stored its keys and compressed its
    completed groups, every group's ring members must still hold its logical positions and the RoPE
    coordinates passed to the real store. Counts per
    pool (the target and the MTP draft model have their own). Write-plan padding (write slot 0) and the
    inert request slot 0 are skipped. Eager forwards only (it syncs)."""
    import torch
    if metadata.is_cuda_graph or metadata.compress_member_rows is not None:
        return
    ends = metadata.compress_group_positions
    if metadata.write_locs is None or ends is None or ends.numel() == 0:
        return
    ends = ends.long()
    seq = metadata.compress_sequence_ids.long()
    locs = metadata.compress_group_ring_locs
    if locs is None:
        locs = indexer._group_ring_slots(metadata, ends, seq)
    real = (metadata.write_locs.long() != 0) & (metadata.req_pool_indices.long()[seq] != 0)
    if not bool(real.any()):
        return
    pool = metadata.token_to_kv_pool
    pos = pool._ax_qsa_check_logical[locs.long()][real]                           # [groups, ratio]
    want = ends[real][:, None] - torch.arange(pos.shape[1] - 1, -1, -1, device=pos.device)[None, :]
    actual_rope = pool.qsa_rope_position_buffer[locs.long()][real]
    expected_rope = pool._ax_qsa_check_rope[locs.long()][real]
    rope_bad = (actual_rope != expected_rope).any(dim=2).any(dim=1)
    bad = (pos != want).any(dim=1) | rope_bad
    c = CHECK.setdefault(id(pool), {"name": f"pool{len(CHECK)}", "groups": 0, "bad": 0, "logged": 0})
    n, nbad = int(real.sum()), int(bad.sum())
    c["groups"] += n
    c["bad"] += nbad
    if nbad and c["logged"] < 3:
        c["logged"] += 1
        i = int(bad.nonzero()[0, 0])
        print(f"ax-engine QSA_CHECK {c['name']} layer {indexer.layer_id}: group ending at "
              f"{int(want[i, -1])} holds logical positions {pos[i].tolist()}; "
              f"RoPE store mismatch={bool(rope_bad[i])}", file=sys.stderr, flush=True)
    if c["groups"] // 2000 != (c["groups"] - n) // 2000:
        print(f"ax-engine QSA_CHECK {c['name']} groups={c['groups']} corrupt={c['bad']}", file=sys.stderr, flush=True)


def install():
    config = get_config()
    ring = config.qsa_ring
    if ring <= 0 and os.environ.get("AX_QSA_CHECK") != "1" and not config.qsa_stable_topk:
        return
    try:
        import sglang.srt.mem_cache.qsa_kv_pool as pool_mod
        import sglang.srt.layers.attention.qsa.metadata as md
        import sglang.srt.layers.attention.qsa.qsa_indexer as qi
        import sglang.srt.layers.attention.qsa.graph_metadata as gm
        import sglang.srt.layers.attention.qwen_sparse_attn_backend as be
    except ImportError:
        return   # an SGLang without QSA: nothing to widen
    if getattr(pool_mod, "_ax_qsa_ring", None):
        return
    if config.qsa_stable_topk:
        from ax_engine_sglang.qsa_topk import stable_topk
        import sglang.srt.layers.attention.qsa.kernel as kernel
        patch_attribute(kernel, 'qsa_fast_topk', stable_topk)
        patch_attribute(qi, 'qsa_fast_topk', stable_topk)
        print("ax-engine-sglang: stable QSA Top-K tensor fallback enabled", file=sys.stderr)
    if os.environ.get("AX_QSA_CHECK") == "1":
        update = qi.QSAIndexer.update_key_state_and_compress

        def update_checked(self, token_k, logical_positions, rope_positions, metadata,
                           state_slots=None, state_stored=False):
            update(self, token_k, logical_positions, rope_positions, metadata,
                   state_slots=state_slots, state_stored=state_stored)
            _record_check_positions(self, metadata, logical_positions, rope_positions, state_slots)
            _check_groups(self, metadata)

        patch_attribute(qi.QSAIndexer, 'update_key_state_and_compress', update_checked)
        print("ax-engine: AX_QSA_CHECK=1 counts QSA groups compressed from overwritten ring members "
              "(diagnostic: syncs every forward; run with --disable-cuda-graph)", file=sys.stderr)
    if ring <= 0:
        patch_attribute(pool_mod, '_ax_qsa_ring', -1)
        return
    patch_attribute(sys.modules[__name__], 'RING', ring)

    init = pool_mod.QSATokenToKVPool.__init__

    def pool_init(self, *args, **kwargs):
        ratio = kwargs["qsa_compress_ratio"]
        if ring % ratio:
            raise RuntimeError(f"ax-engine: AX_QSA_RING={ring} must be a multiple of the QSA compress ratio {ratio}")
        # the pool sizes its ring as num_request_slots * ratio; ask for ring / ratio times as many
        kwargs["num_request_slots"] = kwargs["num_request_slots"] * (ring // ratio)
        init(self, *args, **kwargs)

    patch_attribute(pool_mod.QSATokenToKVPool, '__init__', pool_init)
    for m in (md, qi, be):
        patch_attribute(m, 'build_pending_ring_slots', build_pending_ring_slots)
        patch_attribute(m, 'build_group_ring_slots', build_group_ring_slots)
    _patch_graph_metadata(gm)

    def require_chain_speculation(self, forward_mode, spec_info):
        if forward_mode is None or not forward_mode.is_target_verify():
            return
        if int(getattr(spec_info, "topk", 1) or 1) != 1:
            raise NotImplementedError("Qwen QSA target verification supports only speculative_eagle_topk=1")
        draft_tokens = int(getattr(spec_info, "draft_token_num", 0) or 0)
        if draft_tokens > max_window(self.compress_ratio):
            raise NotImplementedError(
                f"ax-engine: QSA verify window {draft_tokens} exceeds AX_QSA_RING={ring} - ratio + 1 = "
                f"{max_window(self.compress_ratio)}; raise AX_QSA_RING")

    patch_attribute(be.QwenSparseAttnBackend, '_require_chain_speculation', require_chain_speculation)
    patch_attribute(pool_mod, '_ax_qsa_ring', ring)
    print(f"ax-engine: QSA pending ring {ring} slots per request (verify windows up to {ring - 3} tokens "
          f"at ratio 4)", file=sys.stderr)
