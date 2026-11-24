# SPDX-License-Identifier: Apache-2.0
"""Write accepted BF16 draft KV directly from the stacked projection."""

from functools import wraps
import logging
import sys

import torch
import triton
import triton.language as tl

from .ax_dflash_activity import activity
from .compat import installation, patch_attribute, rebuild_function


FEATURE = "AX_DFLASH_KV_DIRECT"
logger = logging.getLogger(__name__)
_installed = False


# The RMSNorm and NeoX RoPE arithmetic follows SGLang's
# kernels/ops/speculative/fused_kv_materialize.py (Apache-2.0, SGLang Team).
@triton.jit
def _write_prefix_kv(
    projected, norm_weights, eps, rope, positions, slots, lengths, keys, values,
    proj_row_stride: tl.constexpr, proj_layer_stride: tl.constexpr,
    norm_stride: tl.constexpr, rope_stride: tl.constexpr,
    width: tl.constexpr, head_dim: tl.constexpr, kv_size: tl.constexpr,
    rotary_dim: tl.constexpr, block_head: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    layer = tl.program_id(2)
    if row % width >= tl.load(lengths + row // width):
        return

    slot = tl.load(slots + row)
    position = tl.load(positions + row)
    key_dst = tl.load(keys + layer).to(tl.pointer_type(tl.bfloat16))
    value_dst = tl.load(values + layer).to(tl.pointer_type(tl.bfloat16))
    src = projected + row * proj_row_stride + layer * proj_layer_stride + head * head_dim
    dest_offset = slot * kv_size + head * head_dim

    offset = tl.arange(0, block_head)
    in_head = offset < head_dim
    half_rotary: tl.constexpr = rotary_dim // 2
    in_rotary_half = offset < half_rotary
    k = tl.load(src + offset, mask=in_head, other=0.0).to(tl.float32)
    v = tl.load(src + kv_size + offset, mask=in_head, other=0.0)
    norm = norm_weights + layer * norm_stride
    inv_rms = tl.rsqrt(tl.sum(k * k) / head_dim + tl.load(eps + layer).to(tl.float32))
    normed = k * inv_rms * tl.load(norm + offset, mask=in_head, other=1.0).to(tl.float32)
    second = tl.load(src + half_rotary + offset, mask=in_rotary_half, other=0.0).to(tl.float32)
    second_weight = tl.load(norm + half_rotary + offset, mask=in_rotary_half, other=1.0).to(tl.float32)
    second = second * inv_rms * second_weight
    first = tl.where(in_rotary_half, normed, 0.0)
    cos = tl.load(rope + position * rope_stride + offset, mask=in_rotary_half, other=1.0).to(tl.float32)
    sin = tl.load(rope + position * rope_stride + half_rotary + offset,
                  mask=in_rotary_half, other=0.0).to(tl.float32)

    tl.store(value_dst + dest_offset + offset, v, mask=in_head)
    # Preserve the native kernel's FMA order even when the destination layout differs.
    tl.store(key_dst + dest_offset + offset, tl.fma(first, cos, -(second * sin)), mask=in_rotary_half)
    tl.store(key_dst + dest_offset + half_rotary + offset, tl.fma(first, sin, second * cos), mask=in_rotary_half)
    tl.store(key_dst + dest_offset + offset, normed, mask=(offset >= rotary_dim) & in_head)


def _compatible_pool(worker, helper, pool):
    expected = (helper.num_kv_heads, helper.head_dim)
    if (pool.kv_cache_layout != "nhd" or pool.dtype != torch.bfloat16 or pool.store_dtype != pool.dtype
            or pool.layer_num != helper.n_layers or not 0 < helper.rotary_dim <= helper.head_dim
            or helper.rotary_dim % 2 or helper.flat_kv_weight_t.dtype != pool.dtype):
        return False

    # The pointer tables are ordered by layer_id - pool.start_layer.
    for i, layer in enumerate(worker.draft_model.layers):
        if layer.self_attn.attn.layer_id != pool.start_layer + i:
            return False
    return all(buffer.shape[1:] == expected and buffer.is_contiguous()
               for buffer in (*pool.k_buffer, *pool.v_buffer))


def _wrap_append(original, pool_type):
    @wraps(original)
    def append(self, ctx_hidden, ctx_positions, ctx_cache_loc, ctx_cache_loc_2d=None, commit_lens=None):
        helper = self._fused_kv_helper
        slots, lengths = ctx_cache_loc_2d, commit_lens
        pool = self.draft_model_runner.token_to_kv_pool
        if (helper is None or slots is None or lengths is None or type(pool) is not pool_type
                or self.block_size != 8 or ctx_hidden.device.type != "cuda"
                or slots.ndim != 2 or slots.shape[1] != self.block_size or slots.shape[0] == 0
                or ctx_hidden.shape[0] != slots.numel() or ctx_positions.shape != (slots.numel(),)
                or lengths.shape != (slots.shape[0],)
                or ctx_positions.dtype != torch.int64 or slots.dtype != torch.int64 or lengths.dtype != torch.int32
                or not all(t.is_contiguous() and t.device == ctx_hidden.device for t in (ctx_positions, slots, lengths))):
            return original(self, ctx_hidden, ctx_positions, ctx_cache_loc, slots, lengths)

        state = getattr(self, "_ax_direct_kv_binding", None)
        if (state is None or state[0] is not pool or state[1] is not helper
                or state[2] is not pool.k_data_ptrs or state[3] is not pool.v_data_ptrs):
            state = (pool, helper, pool.k_data_ptrs, pool.v_data_ptrs, _compatible_pool(self, helper, pool))
            self._ax_direct_kv_binding = state
        if not state[-1]:
            return original(self, ctx_hidden, ctx_positions, ctx_cache_loc, slots, lengths)

        def write(projected, rope, positions):
            _write_prefix_kv[(slots.numel(), helper.num_kv_heads, helper.n_layers)](
                projected, helper.k_norm_weights, helper.eps_values, rope, positions, slots, lengths,
                pool.k_data_ptrs, pool.v_data_ptrs, projected.stride(0), projected.stride(1),
                helper.k_norm_weights.stride(0), rope.stride(0), self.block_size, helper.head_dim,
                helper.kv_size, helper.rotary_dim, triton.next_power_of_2(helper.head_dim),
            )

        helper.materialize(ctx_hidden, ctx_positions, write_layer_kv=None, write_prefix_kv=write)
        if activity is not None:
            activity.paths.add("kv_direct")
        sizes = getattr(self, "_ax_direct_kv_batch_sizes", None)
        if sizes is None:
            sizes = self._ax_direct_kv_batch_sizes = set()
        if slots.shape[0] not in sizes:
            logger.info("DFlash direct KV writer active for batch size %d", slots.shape[0])
            sizes.add(slots.shape[0])

    return append


def install():
    if _installed:
        return
    from sglang.kernels.ops.speculative.fused_kv_materialize import FusedKVMaterializeHelper
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    # Reuse the native projection, dtype handling and RoPE cache reservation.
    materialize = rebuild_function(FusedKVMaterializeHelper, "materialize", (
        ("write_layer_kv: Callable[[int, torch.Tensor, torch.Tensor], None],",
         "write_layer_kv: Callable[[int, torch.Tensor, torch.Tensor], None],\n    *,\n    write_prefix_kv=None,", 1),
        ("proj_out = proj_out_2d.view(total_ctx, self.n_layers, self.layer_out_dim)",
         "proj_out = proj_out_2d.view(total_ctx, self.n_layers, self.layer_out_dim)\n"
         "    if write_prefix_kv is not None:\n"
         "        write_prefix_kv(proj_out, cos_sin_cache, positions)\n"
         "        return", 1),
    ), feature=FEATURE)

    name = "_append_target_hidden_fused"
    with installation():
        patch_attribute(FusedKVMaterializeHelper, "materialize", materialize)
        patch_attribute(DFlashWorkerV2, name, _wrap_append(getattr(DFlashWorkerV2, name), MHATokenToKVPool))
        patch_attribute(sys.modules[__name__], "_installed", True)
