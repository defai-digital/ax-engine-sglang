"""Batched DFlash2 lattice trees: one draft tree per request for rounds of 2..K requests (AX_BTREE).

The tree round (`fc1_branch64.verify_branch`) serves one request. This module holds the pieces a
batched tree round needs that do not depend on SGLang:

- `beam_tree_batched`: `tree_ops.beam_tree` with a leading request dimension and the lattice
  scoring (no path function), selecting the same nodes for every request;
- `tree_arrays`: parent -> ancestor matrix, depth, first child and next sibling per request, the
  arrays `ax_lattice.LatticeBuilder` derives for one request;
- `tree_mask_reference`: the verify mask SGLang's tree attention reads at bs > 1, the
  concatenation over requests of the row-major `N x (prefix_i + N)` blocks
  (prefix columns visible, tree columns from the ancestor matrix).

Selection depends only on each request's own lattice, so a request's tree is the one the batch-1
builder would draft for it.
"""
from __future__ import annotations

import torch

NEG = float("-inf")


def beam_tree_batched(c0, logc, nodes, lmax=99, chain_first=False, positions=None):
    """c0 (B, M) anchor-row log-probs, logc (B, S-1, M, M) transition log-probs.

    Returns, per request, the N-1 selected nodes sorted by depth as in `tree_ops.beam_tree`:
    ss (B, N-1) depth, cand (B, N-1) candidate index at that depth, parent (B, N-1) index of the
    parent among the selected nodes (-1 = the anchor), flag (B, N-1) on the greedy spine."""
    B, M = c0.shape
    S = logc.shape[1] + 1 if positions is None else positions
    W = max(nodes - 1, M)
    dev = c0.device
    big = 10000.0 if chain_first else 0.0
    ar_m = torch.arange(M, device=dev)
    top0 = ar_m[None, :] == c0.argmax(-1, keepdim=True)
    pad = W - M
    sc = torch.cat((c0.float(), torch.full((B, pad), NEG, device=dev)), 1)
    fl = torch.cat((top0, torch.zeros((B, pad), dtype=torch.bool, device=dev)), 1)
    lv = torch.cat(((~top0).long(), torch.full((B, pad), lmax + 1, dtype=torch.long, device=dev)), 1)
    cand = torch.cat((ar_m.expand(B, M), torch.zeros((B, pad), dtype=torch.long, device=dev)), 1)
    par = torch.full((B, W), -1, dtype=torch.long, device=dev)
    key = torch.where((lv <= lmax) & (sc > NEG), sc + big * fl.float(), torch.full_like(sc, NEG))
    keys, pars, cands, fls = [key], [par], [cand], [fl]
    for s in range(1, S):
        lc = logc[:, s - 1].float().gather(1, cand[:, :, None].expand(B, W, M))      # (B, W, M)
        is_top = ar_m[None, None, :] == lc.argmax(-1, keepdim=True)
        sc_c = sc[:, :, None] + lc
        fl_c = fl[:, :, None] & is_top
        lv_c = lv[:, :, None] + (~is_top).long()
        ok = (lv_c <= lmax) & (key[:, :, None] > NEG)
        key_c = torch.where(ok, sc_c + big * fl_c.float(), torch.full_like(sc_c, NEG))
        key, idx = key_c.reshape(B, -1).topk(W, dim=-1)
        sc = sc_c.reshape(B, -1).gather(1, idx)
        fl = fl_c.reshape(B, -1).gather(1, idx)
        lv = lv_c.reshape(B, -1).gather(1, idx)
        par, cand = idx // M, idx % M
        keys.append(key)
        pars.append(par)
        cands.append(cand)
        fls.append(fl)
    K = torch.stack(keys, 1).reshape(B, -1)            # (B, S * W)
    Pb = torch.stack(pars, 1).reshape(B, -1)
    Cb = torch.stack(cands, 1).reshape(B, -1)
    Fb = torch.stack(fls, 1).reshape(B, -1)
    _, gi = K.topk(nodes - 1, dim=-1)
    ss, bi = gi // W, gi % W
    order = torch.argsort(ss * W + torch.arange(nodes - 1, device=dev)[None, :], dim=-1)
    ss, bi = ss.gather(1, order), bi.gather(1, order)
    flat = ss * W + bi
    inv = torch.zeros((B, S * W), dtype=torch.long, device=dev)
    inv.scatter_(1, flat, torch.arange(1, nodes, device=dev).expand(B, -1).contiguous())
    pb = Pb.gather(1, flat)
    up = inv.gather(1, (ss - 1).clamp(min=0) * W + pb.clamp(min=0))
    parent = torch.where(ss == 0, torch.zeros_like(ss), up) - 1
    return ss, Cb.gather(1, flat), parent, Fb.gather(1, flat)


def tree_arrays(parent_full, conv_width=4):
    """parent_full (B, N) with -1 at the anchor (node 0) and parents before children.

    Returns anc (B, N, N) float ancestor-or-self matrix, depth (B, N), nxt (B, N) first child,
    sib (B, N) next sibling (-1 when none), as `ax_lattice.LatticeBuilder.run` builds them, and cidx
    (B, N, conv_width), `tree_ops.tree_struct`'s convolution inputs per node (oldest first: column
    conv_width - 1 + a for the node's ancestor a, a smaller column for the request's conv state)."""
    B, N = parent_full.shape
    dev = parent_full.device
    u = torch.arange(N, device=dev)
    P = torch.zeros((B, N, N), device=dev)
    P[:, 1:, :].scatter_(2, parent_full[:, 1:, None].clamp(min=0), 1.0)
    eye = torch.eye(N, device=dev).expand(B, N, N)
    anc, pk = eye + P, P @ P
    for _ in range(4):                    # ancestors up to depth 2**5 - 1
        anc = anc @ (eye + pk)
        pk = pk @ pk
    anc = (anc > 0.5).float()
    depth = anc.sum(-1).long() - 1
    cols, power = [], eye
    for r in range(conv_width):
        if r > 0:
            power = power @ P
        cols.append(torch.where(r <= depth, conv_width - 1 + power.argmax(-1), conv_width - 1 - (r - depth)))
    cidx = torch.stack(cols[::-1], -1)
    children = (parent_full[:, None, :] == u[None, :, None]) & (u[None, None, :] > 0)
    nxt = torch.where(children, u[None, None, :], N).min(-1).values
    sibling = (parent_full[:, :, None] == parent_full[:, None, :]) & (u[None, None, :] > u[None, :, None])
    sib = torch.where(sibling, u[None, None, :], N).min(-1).values
    return anc, depth, torch.where(nxt < N, nxt, -1), torch.where(sib < N, sib, -1), cidx


def build_trees(cand_ids, c0, logc, anchors, nodes, chain_first=False, positions=None, lmax=99):
    """Draft trees for B requests: tokens (B, N) with the anchor at node 0, parent (B, N) with -1 at
    the anchor, and the arrays of `tree_arrays` (anc, depth, nxt, sib, cidx). cand_ids (B, S, M) are the
    lattice candidates."""
    ss, cidx, parent, _ = beam_tree_batched(c0, logc, nodes, lmax, chain_first, positions)
    B = cand_ids.shape[0]
    M = cand_ids.shape[2]
    picked = cand_ids.reshape(B, -1).gather(1, ss * M + cidx)
    tokens = torch.cat((anchors.reshape(B, 1).to(picked.dtype), picked), 1)
    parent_full = torch.cat((torch.full((B, 1), -1, dtype=torch.long, device=parent.device), parent + 1), 1)
    return (tokens, parent_full, *tree_arrays(parent_full))


def tree_mask_reference(prefix, anc):
    """Concatenated row-major N x (prefix_i + N) masks; prefix (B,) host or device ints."""
    blocks = []
    for p, a in zip(prefix.tolist(), anc):
        n = a.shape[0]
        blocks.append(torch.cat((torch.ones((n, int(p)), dtype=torch.bool, device=a.device), a.bool()), 1).flatten())
    return torch.cat(blocks)


def _mask_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(Out, Anc, Prefix, Start, N: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        p = tl.load(Prefix + b).to(tl.int64)
        width = p + N
        inb = offs < N * width
        row = offs // width
        col = offs - row * width
        in_tree = col >= p
        a = tl.load(Anc + b * N * N + row * N + (col - p), mask=inb & in_tree, other=0.0)
        val = (~in_tree) | (a > 0.5)
        tl.store(Out + tl.load(Start + b) + offs, val.to(tl.uint8), mask=inb)

    return kernel


_KERNEL = None


def tree_mask(buffer, prefix, anc, bound_cpu):
    """GPU tree_mask_reference without a host read of the prefix lengths.

    prefix (B,) device lengths; bound_cpu (B,) host upper bounds of them, which size the launch
    and the buffer (uint8/bool, at least sum N * (bound_i + N) entries). Request b's block starts
    at N * sum_{j<b} (prefix_j + N), the offset SGLang's verify mask packer reads it from."""
    global _KERNEL
    import triton
    if _KERNEL is None:
        _KERNEL = _mask_kernel()
    B, N, _ = anc.shape
    lengths = (prefix.to(torch.int64) + N) * N
    start = torch.cumsum(lengths, 0) - lengths
    longest = N * (int(bound_cpu.max()) + N)
    total = int((N * (bound_cpu.to(torch.int64) + N)).sum())
    if buffer.numel() < total:
        raise RuntimeError(f"ax_btree: mask buffer of {buffer.numel()} entries, {total} needed")
    if total >= 2 ** 31:
        raise RuntimeError(f"ax_btree: mask of {total} entries exceeds the supported size")
    block = 1024
    _KERNEL[(B, triton.cdiv(longest, block))](buffer.view(torch.uint8), anc.contiguous().float(), prefix, start,
                                              N=N, BLOCK=block)
    return buffer[:total]


def lattice_logc_batched(lat, tail, temp=1.0, penalty=0.0, use_tail=True):
    """`ax_lattice.lattice_logc` per request: lat (B, S, K, K), tail (B, S) -> c0 (B, K), logc (B, S-1, K, K)."""
    from ax_engine_sglang.ax_lattice import tree_scores
    lp = tree_scores(lat, tail, temp, penalty, use_tail)
    return lp[:, 0, 0], lp[:, 1:]


class BatchedLatticeBuilder:
    """`ax_lattice.LatticeBuilder` for `bs` requests: one CUDA graph over the first `bs` rows of the
    selector sampler's lattice buffers (written by this round's draft graph)."""

    def __init__(self, sampler, bs, nodes, chain_first=False, graph=True):
        self.sampler, self.bs, self.nodes, self.chain_first = sampler, bs, nodes, chain_first
        from ax_engine_sglang.ax_lattice import tree_shape
        self.shape = tree_shape()
        self.cand = torch.zeros_like(sampler.fc1_cand[:bs])
        self.lat = torch.zeros_like(sampler.fc1_lat[:bs])
        self.tail = torch.zeros_like(sampler.fc1_tail[:bs])
        self.root = torch.zeros(bs, device=self.cand.device, dtype=torch.long)
        self.lat.normal_()                                            # any finite values for warmup
        self.run()
        self.graph = None
        if graph:
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.run()

    def run(self):
        temp, penalty, use_tail, lmax = self.shape
        c0, logc = lattice_logc_batched(self.lat, self.tail, temp, penalty, use_tail)
        (self.tokens, self.parent, self.anc, self.depth, self.nxt, self.sib,
         self.cidx) = build_trees(self.cand, c0, logc, self.root, self.nodes, self.chain_first, lmax=lmax)

    def __call__(self, root):
        s, bs = self.sampler, self.bs
        self.cand.copy_(s.fc1_cand[:bs])
        self.lat.copy_(s.fc1_lat[:bs])
        self.tail.copy_(s.fc1_tail[:bs])
        self.root.copy_(root.reshape(bs))
        if self.graph is None:
            self.run()
        else:
            self.graph.replay()
        return self.tokens, self.depth, self.parent, self.anc, self.nxt, self.sib


def mask_buffer(owner, entries, device):
    """Two alternating mask buffers per worker, so round k's mask is never rewritten by round k + 1
    while a consumer of round k may still run (as ax_fast.tree_mask)."""
    bufs = owner.__dict__.setdefault("_ax_btree_masks", [None, None])
    owner._ax_btree_slot = getattr(owner, "_ax_btree_slot", 0) ^ 1
    buf = bufs[owner._ax_btree_slot]
    if buf is None or buf.numel() < entries:
        buf = bufs[owner._ax_btree_slot] = torch.empty(max(2 * entries, 1 << 20), dtype=torch.bool, device=device)
    return buf
