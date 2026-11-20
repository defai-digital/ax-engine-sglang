# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
"""Our-only, single-request tree plan using known host lengths, no GPU readback."""
from ax_engine_sglang.compat import patch_attribute
import inspect
import torch
from ax_engine_sglang.runtime import current_runtime

def install():
    import flashinfer
    from sglang.srt.layers.attention.flashinfer_backend import fast_prefill_plan
    cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
    original = cls.plan
    names = list(inspect.signature(original).parameters)[1:]
    fast_names = set(inspect.signature(fast_prefill_plan).parameters) - {'self'}

    def plan(self, *args, **kwargs):
        values = dict(zip(names, args))
        values.update(kwargs)
        mask = values.get('custom_mask')
        host_kv_len = current_runtime().host_kv_len
        if host_kv_len is None or mask is None or (not self.is_cuda_graph_enabled) or (getattr(self, '_backend', None) != 'fa2') or (getattr(self, '_cached_module', None) is None):
            return original(self, *args, **kwargs)
        qo = values['qo_indptr']
        n = mask.numel() // host_kv_len
        if qo.numel() != 2 or values['page_size'] != 1 or n not in (16, 32, 64) or n * host_kv_len != mask.numel():
            # Not the single-request tree shape this plan was written for (another FC1_TREE_N, or a
            # masked plan that is not this round's tree verify): SGLang's own plan is always correct.
            return original(self, *args, **kwargs)
        host_q = torch.tensor([0, n], dtype=torch.int32)
        host_k = torch.tensor([0, host_kv_len], dtype=torch.int32)
        vals = {k: v for k, v in values.items() if k in fast_names}
        vals.update(qo_indptr_host=host_q, kv_indptr_host=host_k, kv_lens_host=host_k[1:], max_q_len=n, max_kv_len=host_kv_len)
        if 'uniform_verify_width' in fast_names:
            # SGLang's newer fast planner packs the tree mask from host-known row lengths.
            vals['uniform_verify_width'] = n
        else:
            packed = flashinfer.packbits(mask.flatten(), bitorder='little')
            self._custom_mask_buf[:packed.numel()].copy_(packed, non_blocking=True)
            mask_end = torch.tensor([0, packed.numel()], dtype=self._mask_indptr_buf.dtype, device=qo.device)
            self._mask_indptr_buf.copy_(mask_end, non_blocking=True)
        return fast_prefill_plan(self, **vals)
    patch_attribute(cls, 'plan', plan)
    patch_attribute(cls, 'begin_forward', plan)
