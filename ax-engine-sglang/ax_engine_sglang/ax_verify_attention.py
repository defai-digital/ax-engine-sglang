"""Run FlashInfer target verification as one paged decode per proposed history."""
from functools import wraps

from .compat import patch_attribute


def branch_pages(indices, kv_bounds, width, mask):
    """Expand each request's paged KV into the visible history of each candidate."""
    import torch

    lengths = []
    histories = []
    offset = 0
    for start, end in zip(kv_bounds, kv_bounds[1:]):
        pages = indices[start:end]
        size = end - start
        if mask is None:
            steps = torch.arange(width, device=pages.device) + size - width + 1
            visible = torch.arange(size, device=pages.device)[None, :] < steps[:, None]
        else:
            block = width * size
            visible = mask[offset:offset + block].reshape(width, size).bool()
            steps = visible.sum(dim=1)
            offset += block
        histories.append(pages.expand(width, -1)[visible])
        lengths.append(steps)

    lengths = torch.cat(lengths).to(torch.int32)
    indptr = torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    return indptr, torch.cat(histories)


def install():
    import torch
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend, KVWriteLoc

    cls = FlashInferAttnBackend
    original_init = cls.__init__

    @wraps(original_init)
    def init(self, model_runner, *args, **kwargs):
        original_init(self, model_runner, *args, **kwargs)
        self._ax_target_decode = not model_runner.is_draft_worker
        if self._ax_target_decode:
            if (self.page_size != 1 or self.num_wrappers != 1 or not self.decode_use_tensor_cores
                    or self.prefill_uses_dequant_workspace or self.decode_uses_dequant_workspace):
                raise ValueError("NVFP4 verify requires single-wrapper, page-size-one FlashInfer FP8 attention")
            self.use_paged = True
            self.disable_cuda_graph_kv_split = True
            # Eager AR explicitly passes disable_split_kv=False through the updater.
            # Apply the same reduction rule to eager AR, graph AR and grouped verify.
            update = self.indices_updater_decode.update

            @wraps(update)
            def no_split(*args, **kwargs):
                kwargs["disable_split_kv"] = True
                return update(*args, **kwargs)

            self.indices_updater_decode.update = no_split
            self._ax_verify_graphs = {}
            self._ax_graph_pages = None
            self._ax_decode_wrapper = None

    def setup(self, batch, *, graph, capture=False):
        if not self._ax_target_decode or not batch.forward_mode.is_target_verify():
            return
        spec = batch.spec_info
        if spec.ragged_verify_layout is not None:
            raise ValueError("NVFP4 verify requires a uniform draft window")
        indices, bounds, _, mask = spec.generate_attn_arg_prefill(
            req_pool_indices=batch.req_pool_indices, paged_kernel_lens=batch.seq_lens,
            paged_kernel_lens_sum=batch.seq_lens_sum, translator=self.kv_index_translator,
            plan=batch.kv_loc_plan,
        )
        # Capture builds a valid dummy causal layout. Replay supplies the live tree mask.
        indptr, pages = branch_pages(indices, bounds.tolist(), spec.draft_token_num, None if capture else mask)
        rows = indptr.numel() - 1
        wrapper = self.decode_wrappers[0]
        if graph:
            if rows not in self._ax_verify_graphs:
                if not capture:
                    raise RuntimeError(f"NVFP4 verify has no captured decode wrapper for {rows} rows")

                def buffer(size):
                    return torch.empty(size, dtype=torch.int32, device=indices.device)

                if self._ax_graph_pages is None:
                    max_rows = self.cuda_graph_kv_indices[0].numel() // self.max_context_len
                    self._ax_graph_pages = buffer(max_rows * (self.max_context_len + spec.draft_token_num))
                self._ax_verify_graphs[rows] = BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer, "NHD", backend=self.decode_backend,
                    use_cuda_graph=True, use_tensor_cores=self.decode_use_tensor_cores,
                    paged_kv_indptr_buffer=buffer(rows + 1),
                    paged_kv_indices_buffer=self._ax_graph_pages,
                    paged_kv_last_page_len_buffer=buffer(rows),
                )
            wrapper = self._ax_verify_graphs[rows]
        updater = self.indices_updater_prefill
        wrapper.plan(indptr, pages, torch.ones_like(indptr[:-1]), updater.num_qo_heads, updater.num_kv_heads,
                     updater.head_dim, self.page_size, q_data_type=updater.q_data_type,
                     kv_data_type=updater.data_type, disable_split_kv=True)
        self._ax_decode_wrapper = wrapper

    original_eager = cls.init_forward_metadata
    original_graph = cls.init_forward_metadata_out_graph
    original_extend = cls.forward_extend

    @wraps(original_eager)
    def eager(self, forward_batch):
        original_eager(self, forward_batch)
        setup(self, forward_batch, graph=False)

    @wraps(original_graph)
    def graph_plan(self, forward_batch, in_capture=False):
        original_graph(self, forward_batch, in_capture)
        setup(self, forward_batch, graph=True, capture=in_capture)

    @wraps(original_extend)
    def extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if not self._ax_target_decode or not forward_batch.forward_mode.is_target_verify():
            return original_extend(self, q, k, v, layer, forward_batch, save_kv_cache)
        if k is not None and save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc.for_layer(forward_batch, layer, swa_loc=self.forward_metadata.swa_out_cache_loc),
                k, v, *self._kv_write_scales(layer),
            )
        output = self._ax_decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            self.token_to_kv_pool.get_kv_buffer(layer.layer_id), sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap, k_scale=layer.k_scale_float, v_scale=layer.v_scale_float,
        )
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    patch_attribute(cls, "__init__", init)
    patch_attribute(cls, "init_forward_metadata", eager)
    patch_attribute(cls, "init_forward_metadata_out_graph", graph_plan)
    patch_attribute(cls, "forward_extend", extend)
