# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
"""DForkTree branch verification and state commit."""
from contextlib import nullcontext
from ax_engine_sglang.runtime import current_runtime
import torch
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_worker_common import run_eagle_verify
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

def verify_branch(w, batch, tokens, prefix, positions, plan, cache2d, on_publish):
    bs = tokens.shape[0]
    n = int(__import__('os').environ.get('FC1_TREE_N', '64'))
    assert bs == 1, 'single-request latency experiment'
    dev = tokens.device
    import os
    import ax_lattice   # ax-engine-sglang: DFlash2 lattice trees, AX_LATTICE=1
    use_lattice = ax_lattice.enabled()
    logits = None if use_lattice else w._draft_sampler.fc1_logits.view(bs, w.block_size - 1, -1)[0]
    ours = os.environ.get('FC1_OURS') == '1'
    from fc1_builder import Builder
    if not hasattr(w, 'fc1_builder'):
        w.fc1_builder = (ax_lattice.LatticeBuilder(w._draft_sampler, compile_build=os.environ.get('FC1_COMPILE_BUILD') == '1')
                         if use_lattice else Builder(logits, compile_build=os.environ.get('FC1_COMPILE_BUILD') == '1'))
    import ax_chain   # ax-engine-sglang: token before the anchor, kept per request across tree and chain rounds
    pre1 = None
    if not use_lattice:
        pre1 = ax_chain.pre1_for(batch.reqs[0])
        if pre1 is None:
            pre1 = torch.tensor([batch.reqs[0].origin_input_ids[-1]], device=dev)
    tree, depth, parent, anc, nxt, sib = w.fc1_builder(logits, tokens[:, 0], pre1)
    if os.environ.get('FC1_FUSED') == '1':
        import fc1_fused
        assert fc1_fused.context() is not None
        fc1_fused.context().anc.copy_(anc)
        fc1_fused.context().cidx.copy_(w.fc1_builder.cidx)
    from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
    cache = assign_extend_cache_locs_func(req_pool_indices=batch.req_pool_indices, req_to_token=w.model_runner.req_to_token_pool.req_to_token, start_offset=prefix, end_offset=prefix + n, batch_size=bs, draft_token_num=n, device=dev)
    cache2d = cache.view(bs, n)
    import ax_fast   # ax-engine-sglang overhead fixes, AX_FAST=1
    _ax_mode = ax_fast.mode(batch)
    _ax_plan_cpu = ax_fast.plan_lens_cpu(batch, prefix, _ax_mode)
    plan = w.target_worker.model_runner.kv_index_translator.plan(req_pool_indices=batch.req_pool_indices, seq_lens=prefix, seq_lens_cpu=_ax_plan_cpu, write_virtual=cache, read_extent=n)
    positions = (prefix[:, None] + torch.arange(n, device=dev)[None]).flatten()
    seq_lens_cpu_was_none = batch.seq_lens_cpu is None   # ax-engine-sglang: host mirror may be absent
    if _ax_mode == 'fast':
        import fc1_plan
        current_runtime().host_kv_len = None
        mask = ax_fast.tree_mask(w, prefix, anc, n, _ax_plan_cpu)
    else:
        if seq_lens_cpu_was_none:
            batch.seq_lens_cpu = _ax_plan_cpu if _ax_mode == 'dedupe' else prefix.detach().to('cpu', torch.int64)
        plen = int(batch.seq_lens_cpu[0])
        import fc1_plan
        current_runtime().host_kv_len = plen + n
        mask = torch.cat((torch.ones((n, plen), device=dev, dtype=torch.bool), anc.bool()), dim=1).flatten()
    vi = EagleVerifyInput(draft_token=tree.flatten(), custom_mask=mask, positions=(prefix[:, None] + depth[None]).flatten(), retrieve_index=torch.arange(n, device=dev, dtype=torch.long)[None], retrieve_next_token=nxt[None].long(), retrieve_next_sibling=sib[None].long(), retrieve_cum_len=None, spec_steps=min(w.block_size - 1, n - 1), topk=2, draft_token_num=n, capture_hidden_mode=CaptureHiddenMode.FULL, seq_lens_sum=batch.seq_lens_sum, seq_lens_cpu=batch.seq_lens_cpu, prepared_out_cache_loc=cache2d.flatten(), kv_loc_plan=plan)
    vi.retrieve_parent_token = parent[None].long()
    batch.spec_info = vi
    target = w.target_worker
    try:
        result = run_eagle_verify(batch, target_worker=target, req_to_token_pool=target.model_runner.req_to_token_pool, token_to_kv_pool_allocator=target.model_runner.token_to_kv_pool_allocator, plan_stream=None, plan_stream_ctx=nullcontext(), topk=2, num_draft_tokens=n, device=str(dev), metadata_ready_pre_pad=False, finalize_tree_path=True)
    finally:
        current_runtime().host_kv_len = None   # only this verify's plan may use the host length
    if not use_lattice:
        last = result.accept_index.gather(1, (result.accept_lens.long() - 1)[:, None])
        pre1 = tree.gather(1, last.long()).flatten()
    ax_chain.record_tree(batch, pre1, tree, result.accept_index, result.accept_lens)
    bonus = result.next_draft_input.bonus_tokens
    hidden = result.logits_output.hidden_states
    assert hidden is not None
    loc = w._draft_write_ids(plan)
    width = min(n, w.block_size)
    hidden = hidden.view(bs, n, -1)[:, :width].contiguous()
    loc = loc.view(bs, n)[:, :width].contiguous()
    positions = positions.view(bs, n)[:, :width].contiguous().flatten()
    w._append_target_hidden_to_draft_kv_by_loc(target_hidden=hidden.reshape(bs * width, -1), cache_loc=loc.flatten(), cache_loc_2d=loc, positions=positions, commit_lens=result.accept_lens)
    result.logits_output.hidden_states = None
    result.next_draft_input = w._make_next_draft_input_decode(bonus_tokens=bonus, new_seq_lens=result.new_seq_lens)
    if seq_lens_cpu_was_none:
        batch.seq_lens_cpu = None
    if on_publish is not None:
        on_publish(result.new_seq_lens)
    return result

