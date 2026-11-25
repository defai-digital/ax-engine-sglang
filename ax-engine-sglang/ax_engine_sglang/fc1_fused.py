# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
import os
from ax_engine_sglang.runtime import current_runtime
from ax_engine_sglang.compat import patch_attribute
from ax_engine_sglang.config import get_config

NN = get_config().tree_nodes
BS = int(os.environ.get('FC1_DRAFT_BLOCK', '24'))
GL = int(os.environ.get('AX_GDN_LAYERS', '24'))   # ax-engine-sglang: GDN layers of the target
GH = int(os.environ.get('AX_GDN_HEADS', '32'))    # ax-engine-sglang: GDN value heads
AX_DBG = os.environ.get('AX_GDN_DEBUG') == '1'     # ax-engine-sglang: diagnostics, see patch_dforktree.py
# ax-engine-sglang AX_BTREE: batched tree rounds of up to KMAX requests run ax_btree_kernels unless
# AX_BTREE_NATIVE=1 leaves them to SGLang's native GDN tree verify
KMAX = get_config().btree if get_config().fused and not get_config().btree_native else 0
'Our-only GDN adapter. Native baselines never import this module.'
import torch
import triton
import triton.language as tl
import tree_kernels as TK

class Context:

    def __init__(self):
        self.anc = torch.eye(NN, device='cuda', dtype=torch.float32)
        self.keys = torch.empty(GL, GH, NN, 128, device='cuda')
        self.values = torch.empty_like(self.keys)
        self.gains = torch.empty(GL, GH, NN, device='cuda')
        if AX_DBG:   # ax-engine-sglang: flag buffers of the diagnostics, allocated with the context
            self.ax_bad = torch.zeros(GL, 4, device='cuda')
            self.ax_nodes = torch.zeros(GL, NN, device='cuda')
            self.ax_amax = torch.zeros(GL, 3, device='cuda')
            self.ax_att = torch.zeros(256, 2, device='cuda')
            self.ax_dbg = 0
            self.ax_kv_done = False
        self.layers = {}
        self.inter_indices = None
        if KMAX > 1:   # batched tree rounds: aux arrays (layer, request, head, node, ...), trees per request
            self.bkeys = torch.empty(GL, KMAX, GH, NN, 128, device='cuda')
            self.bvalues = torch.empty_like(self.bkeys)
            self.bgains = torch.empty(GL, KMAX, GH, NN, device='cuda')
            self.banc = torch.eye(NN, device='cuda', dtype=torch.float32).repeat(KMAX, 1, 1)
            self.bcidx = torch.zeros(KMAX, NN, 4, device='cuda', dtype=torch.int32)
            # SGLang's positional scratch-row table (request row i -> row i, gdn_backend
            # verify_intermediate_state_indices; one tensor for every batch size, so a graph padded to a
            # larger captured size reads the same rows); a commit takes its first bs entries
            self.binter = None
def context():
    return current_runtime().fused_context


def ensure_context():
    if context() is None:
        current_runtime().fused_context = Context()
    return context()

NATIVE_REFERENCE = False


def _tree_round():
    # ax-engine-sglang: with several requests a chain round can have the tree's shape (W == NN); the
    # round kind, not the shape, decides (ax_chain.py)
    import ax_chain
    return ax_chain.is_tree()


def _batched_round():
    """A batched tree round (AX_BTREE) served by ax_btree_kernels (counted as a btree hook call)."""
    state = current_runtime()
    if KMAX > 1 and state.kind == 'btree':
        state.stats['hook_btree'] += 1
        return True
    return False

COLLECT_LAYER = False
LAYER_STATS = []

@triton.jit
def snapshots(S, K, V, G, A, Path, Ends, Slot, InterSlot, Out, NN: tl.constexpr, BS: tl.constexpr, SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr, SS4: tl.constexpr, OS0: tl.constexpr, OS1: tl.constexpr, OS2: tl.constexpr, OS3: tl.constexpr, OS4: tl.constexpr, OS5: tl.constexpr, GH: tl.constexpr):
    lh = tl.program_id(0)
    l = (lh // GH).to(tl.int64)   # ax-engine-sglang: l * OS0 overflows int32 for the 27B
    h = lh % GH
    ik = tl.program_id(1) * 32 + tl.arange(0, 32)
    q = tl.program_id(2)
    iv = q % 4 * 32 + tl.arange(0, 32)
    which = q // 4
    u = tl.load(Ends + which)
    slot = tl.load(Slot).to(tl.int64)
    dst = tl.load(InterSlot).to(tl.int64)
    t = tl.arange(0, 32)
    ids = tl.load(Path + t, mask=t < BS, other=-1)
    valid = (u >= 0) & (ids >= 0)
    on = tl.load(A + tl.maximum(u, 0) * NN + tl.maximum(ids, 0), mask=valid, other=0) > 0
    gn = tl.load(G + lh * NN + tl.maximum(u, 0))
    gt = tl.load(G + lh * NN + tl.maximum(ids, 0), mask=on, other=0.0)
    weights = tl.exp(tl.where(on, gn - gt, float('-inf')))
    kk = tl.load(K + (lh * NN + tl.maximum(ids[None, :], 0)) * 128 + ik[:, None], mask=on[None, :], other=0.0)
    vv = tl.load(V + (lh * NN + tl.maximum(ids[:, None], 0)) * 128 + iv[None, :], mask=on[:, None], other=0.0)
    old = tl.load(S + l * SS0 + slot * SS1 + h * SS2 + iv[None, :] * SS3 + ik[:, None] * SS4)
    new = old * tl.exp(gn) + tl.dot(kk * weights[None, :], vv, input_precision='tf32x3')
    tl.store(Out + l * OS0 + dst * OS1 + tl.maximum(u, 0) * OS2 + h * OS3 + iv[None, :] * OS4 + ik[:, None] * OS5, new, mask=u >= 0)

def install():

    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.srt.speculative import eagle_worker_common as common
    from sglang.srt.speculative.spec_utils import _verify_commit_step_indices
    original = TritonGDNKernel.target_verify
    commit = common.commit_mamba_states_after_verify

    def fused(self, A_log, dt_bias, q, k, v, a, b, **kw):

        if kw['cache_steps'] == NN and _batched_round():
            return fused_batched(A_log, dt_bias, q, v, a, b, kw)
        if kw['cache_steps'] != NN or not _tree_round():   # ax-engine-sglang: tree rounds only
            return original(self, A_log, dt_bias, q, k, v, a, b, **kw)
        ensure_context()
        key = A_log.data_ptr()
        if key not in context().layers:
            context().layers[key] = (len(context().layers), -A_log.float().exp())
        li, neg = context().layers[key]
        assert li < GL and q.shape[0] == 1 and (q.shape[1] == NN)
        context().inter_indices = kw['intermediate_state_indices']
        pool = kw['ssm_states']
        state = pool[0].transpose(-1, -2)
        y = context().current_mixed
        out = TK.tree_gdn_split(y, a.reshape(NN, -1), b.reshape(NN, -1), A_log, dt_bias, state, context().anc, q.shape[2], v.shape[2], q.shape[3], v.shape[3], aux=(context().keys[li], context().values[li], context().gains[li]), neg_exp_A=neg, state_indices=kw['cache_indices'], state_slot_stride=pool.stride(0))
        if AX_DBG:   # graph-safe: flags stay on the GPU
            context().ax_bad[li, 0] = (~torch.isfinite(out)).any().float()
            context().ax_bad[li, 1] = (~torch.isfinite(y)).any().float()
            context().ax_bad[li, 2] = (~torch.isfinite(pool.index_select(0, kw['cache_indices'][:1].long().clamp(min=0)))).any().float()
            context().ax_bad[li, 3] = (~torch.isfinite(a)).any().float() + 2 * (~torch.isfinite(b)).any().float()
            context().ax_nodes[li] = (~torch.isfinite(out.reshape(NN, -1))).any(-1).float()
            context().ax_amax[li, 0] = y.float().abs().amax()
            context().ax_amax[li, 1] = out.float().abs().nan_to_num(0.0, 0.0, 0.0).amax()
            context().ax_amax[li, 2] = (y.float() * y.float()).reshape(NN, -1)[:, :q.shape[2] * q.shape[3] * 2].sum(-1).min()
        return out.unsqueeze(0)

    def fused_batched(A_log, dt_bias, q, v, a, b, kw):
        import ax_btree_kernels as TKB
        ctx = ensure_context()
        key = A_log.data_ptr()
        if key not in ctx.layers:
            ctx.layers[key] = (len(ctx.layers), -A_log.float().exp())
        li, neg = ctx.layers[key]
        bs = q.shape[1] // NN
        assert li < GL and q.shape[0] == 1 and q.shape[1] == bs * NN and 1 <= bs <= KMAX
        ctx.binter = kw['intermediate_state_indices']
        pool = kw['ssm_states']
        aux = (ctx.bkeys[li, :bs].view(bs * GH, NN, 128), ctx.bvalues[li, :bs].view(bs * GH, NN, 128),
               ctx.bgains[li, :bs].view(bs * GH, NN))
        out = TKB.tree_gdn(ctx.current_mixed, a.reshape(bs * NN, -1), b.reshape(bs * NN, -1), A_log, dt_bias,
                           pool[0].transpose(-1, -2), ctx.banc[:bs], bs, q.shape[2], v.shape[2], q.shape[3], v.shape[3],
                           aux=aux, neg_exp_A=neg, state_indices=kw['cache_indices'], state_slot_stride=pool.stride(0))
        return out.unsqueeze(0)

    def fused_commit(target_worker, batch, accept_lens, accept_index, draft_token_num, prepared_step_indices=None):
        if context() is not None and draft_token_num == NN and accept_index.numel() and _batched_round():
            import ax_btree_kernels as TKB
            last, track = _verify_commit_step_indices(batch=batch, accept_index=accept_index, accept_lens=accept_lens, draft_token_num=draft_token_num)
            if track is None and __import__('os').environ.get('AX_FAST') == '1':
                ends = last.contiguous()
            else:
                ends = torch.cat((last, track if track is not None else torch.full_like(last, -1))).contiguous()
            bs = accept_index.shape[0]
            pool = target_worker.model_runner.req_to_token_pool
            state = pool.get_speculative_mamba2_params_all_layers()
            slots = pool.get_mamba_indices(batch.req_pool_indices)
            import ax_state_cap
            if ax_state_cap.POS:
                # The intermediate SSM cache has fewer positions than a tree has nodes (ax_state_cap.py),
                # as in the one-request branch below: each request's accepted state goes straight into its
                # slot's persistent state (an out view with node stride 0), then only the conv windows,
                # which keep one position per node, are rolled back from SGLang's positional scratch rows.
                if track is not None:
                    raise RuntimeError("ax-engine: AX_SSM_POSITIONS needs the radix cache off (no mamba track)")
                t = state.temporal
                TKB.snapshots(t, context().bkeys, context().bvalues, context().bgains, context().banc, accept_index,
                              last.contiguous(), slots, slots, t.unsqueeze(2).expand(t.shape[0], t.shape[1], NN,
                                                                                    *t.shape[2:]), GH, NN)
                from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_conv_window_scatter_with_mask
                for conv, window in zip(state.conv, state.intermediate_conv_window):
                    if conv.numel() > 0:
                        fused_conv_window_scatter_with_mask(conv, window, slots, last, None)
                return None
            TKB.snapshots(state.temporal, context().bkeys, context().bvalues, context().bgains, context().banc,
                          accept_index, ends, slots, context().binter[:bs], state.intermediate_ssm, GH, NN)
            return commit(target_worker, batch, accept_lens, accept_index, draft_token_num, prepared_step_indices)
        if context() is not None and draft_token_num == NN and accept_index.numel() and _tree_round():
            last, track = _verify_commit_step_indices(batch=batch, accept_index=accept_index, accept_lens=accept_lens, draft_token_num=draft_token_num)
            import ax_state_cap
            if ax_state_cap.POS:
                # ax-engine-sglang: the intermediate SSM cache has fewer positions than the tree has nodes
                # (ax_state_cap.py): write the accepted state straight into the slot's persistent
                # state (same values the commit would copy from the intermediate), then roll back only
                # the conv windows, which keep one position per node
                if track is not None:
                    raise RuntimeError("ax-engine: AX_SSM_POSITIONS needs the radix cache off (no mamba track)")
                pool = target_worker.model_runner.req_to_token_pool
                state = pool.get_speculative_mamba2_params_all_layers()
                slots = pool.get_mamba_indices(batch.req_pool_indices)
                t = state.temporal
                snapshots[GL * GH, 4, 4](t, context().keys, context().values, context().gains, context().anc, accept_index, last.contiguous(), slots, slots, t, NN, accept_index.shape[1], *t.stride(), t.stride(0), t.stride(1), 0, t.stride(2), t.stride(3), t.stride(4), GH=GH, num_warps=4)
                from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_conv_window_scatter_with_mask
                for conv, window in zip(state.conv, state.intermediate_conv_window):
                    if conv.numel() > 0:   # as SGLang's scatter_mamba_states_after_mtp_verify
                        fused_conv_window_scatter_with_mask(conv, window, slots, last, None)
                return None
            ax_q = 8   # ax-engine-sglang: with AX_FAST=1 and no track endpoint, launch only the live half
            if track is None and __import__('os').environ.get('AX_FAST') == '1':
                ends = last.contiguous()
                ax_q = 4
            else:
                if track is None:
                    track = torch.full_like(last, -1)
                ends = torch.cat((last, track)).contiguous()
            pool = target_worker.model_runner.req_to_token_pool
            state = pool.get_speculative_mamba2_params_all_layers()
            slots = pool.get_mamba_indices(batch.req_pool_indices)
            snapshots[GL * GH, 4, ax_q](state.temporal, context().keys, context().values, context().gains, context().anc, accept_index, ends, slots, context().inter_indices, state.intermediate_ssm, NN, accept_index.shape[1], *state.temporal.stride(), *state.intermediate_ssm.stride(), GH=GH, num_warps=4)
        ax_r = commit(target_worker, batch, accept_lens, accept_index, draft_token_num, prepared_step_indices)
        if AX_DBG and context() is not None and draft_token_num == NN:
            context().ax_dbg += 1
            st = target_worker.model_runner.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            sl = target_worker.model_runner.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)
            tbad = (~torch.isfinite(st.temporal[:, sl])).flatten(1).any(-1)
            e = context().ax_bad.cpu()
            if e.any() or tbad.any() or context().ax_att.any() or context().ax_dbg <= 2:
                bad = lambda col: [i for i in range(e.shape[0]) if e[i, col] > 0][:4]
                l0 = bad(0)[0] if bad(0) else 0
                am = context().ax_amax.cpu()
                print(f'[ax_gdn] round {context().ax_dbg} accept {accept_lens.tolist()} out_bad {bad(0)} y_bad {bad(1)} state_in_bad {bad(2)} ab_bad {bad(3)} '
                      f'temporal_bad {tbad.nonzero().flatten().tolist()[:4]} layer {l0} bad_nodes {context().ax_nodes[l0].nonzero().flatten().tolist()[:12]} '
                      f'y_amax {[round(float(x), 2) for x in am[max(l0 - 2, 0):l0 + 1, 0]]} out_amax {[round(float(x), 2) for x in am[max(l0 - 2, 0):l0 + 1, 1]]} '
                      f'min_qk_sq {[float(x) for x in am[max(l0 - 2, 0):l0 + 1, 2]]}', flush=True)
                at = context().ax_att.cpu()
                if at.any() and not context().ax_kv_done:
                    context().ax_kv_done = True
                    mr = target_worker.model_runner
                    lay = int(at.amax(-1).nonzero()[0])   # first layer with any flag (input or output)
                    kb, vb = mr.token_to_kv_pool.get_key_buffer(lay), mr.token_to_kv_pool.get_value_buffer(lay)
                    rp = int(batch.req_pool_indices[0]); L = int(batch.seq_lens[0])
                    locs = mr.req_to_token_pool.req_to_token[rp, :L].long()   # committed prefix only
                    kbad = (~torch.isfinite(kb[locs].float())).flatten(1).any(-1) | (~torch.isfinite(vb[locs].float())).flatten(1).any(-1)
                    pos = kbad.nonzero().flatten().tolist()
                    print(f'[ax_kv] layer {lay} seq_len {L} (after commit) req_pool {rp} nan_positions {pos[:20]} n_nan {len(pos)} '
                          f'locs_at_nan {locs[kbad][:8].tolist()} locs_tail {locs[max(L - 4, 0):].tolist()} kv_buf {tuple(kb.shape)}', flush=True)
                print(f'[ax_att] round {context().ax_dbg} qkv_bad(layer:1q2k4v) {[(i, int(at[i, 0])) for i in range(256) if at[i, 0] > 0][:6]} out_bad {[i for i in range(256) if at[i, 1] > 0][:6]}', flush=True)
        return ax_r
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    routing = GDNAttnBackend._target_verify_supports_strided_qkv

    def direct_routing(self, **kwargs):
        if not NATIVE_REFERENCE and kwargs['draft_token_num'] == NN and (_tree_round() or _batched_round()):
            return True
        return routing(self, **kwargs)
    patch_attribute(GDNAttnBackend, '_target_verify_supports_strided_qkv', direct_routing)
    if AX_DBG:   # diagnostics: finite flags of every full-attention layer in tree verify
        from sglang.srt.layers.radix_attention import RadixAttention
        ax_fwd = RadixAttention.forward
        def ax_dbg_fwd(self, q, k, v, forward_batch, *args, **kwargs):
            o = ax_fwd(self, q, k, v, forward_batch, *args, **kwargs)
            if (q.shape[0] == NN and forward_batch.forward_mode.is_target_verify() and k is not None and v is not None
                    and context() is not None):
                context().ax_att[self.layer_id, 0] = (~torch.isfinite(q)).any().float() + 2 * (~torch.isfinite(k)).any().float() + 4 * (~torch.isfinite(v)).any().float()
                context().ax_att[self.layer_id, 1] = (~torch.isfinite(o)).any().float()
            return o
        patch_attribute(RadixAttention, 'forward', ax_dbg_fwd)
    patch_attribute(TritonGDNKernel, 'target_verify', fused)
    patch_attribute(common, 'commit_mamba_states_after_verify', fused_commit)

