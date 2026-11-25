"""Fused GDN tree kernels for batched tree rounds (AX_BTREE with FC1_FUSED=1).

The one-request kernels (`fc1_conv.conv`, `tree_kernels._tree_prepare` / `_tree_finish`,
`fc1_fused.snapshots`) with a request dimension: request b's nodes are rows b * N .. b * N + N - 1
of the verify batch, its tree is ancestor matrix b and its state slots are entry b of SGLang's
per-request index tensors. Program arithmetic per (request, head) is that of the one-request
kernels, so a request's outputs match a batch-1 launch on the same inputs.

A padded CUDA-graph row (slot index -1) reads slot 0 and stores nothing.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _conv(CS, X, W, Bias, CI, Slot, ISlot, IW, Y, C: tl.constexpr, CS0, CS1, CS2, X0, X1, X2, I0, I1, I2, I3,
          N: tl.constexpr, BIAS: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // N
    n = pid % N
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < C
    slot_raw = tl.load(Slot + b).to(tl.int64)
    dst_raw = tl.load(ISlot + b).to(tl.int64)
    slot = tl.maximum(slot_raw, 0)
    dst = tl.maximum(dst_raw, 0)
    acc = tl.full((BC,), 0, tl.float32)
    for j in tl.static_range(4):
        col = tl.load(CI + pid * 4 + j)
        old = col < 3
        x0 = tl.load(CS + slot * CS0 + c * CS1 + col * CS2, mask=m & old, other=0).to(tl.float32)
        x1 = tl.load(X + b * X0 + c * X1 + (col - 3) * X2, mask=m & ~old, other=0).to(tl.float32)
        value = tl.where(old, x0, x1)
        weight = tl.load(W + c * 4 + j, mask=m, other=0).to(tl.float32)
        acc += value * weight
        if j > 0:
            tl.store(IW + dst * I0 + n * I1 + c * I2 + (j - 1) * I3, value, mask=m & (dst_raw >= 0))
    if BIAS:
        acc += tl.load(Bias + c, mask=m, other=0).to(tl.float32)
    tl.store(Y + pid * C + c, acc / (1 + tl.exp(-acc)), mask=m)


def tree_conv(x, conv_state, weight, bias, cidx, conv_state_indices, intermediate_state_indices, window):
    """x (B, C, N) -> (B, N, C) SiLU tree convolution; cidx (B, N, 4) int32."""
    B, C, N = x.shape
    out = torch.empty((B, N, C), device=x.device, dtype=x.dtype)
    _conv[B * N, triton.cdiv(C, 256)](
        conv_state, x, weight, bias if bias is not None else x, cidx, conv_state_indices, intermediate_state_indices,
        window, out, C, *conv_state.stride(), *x.stride(), *window.stride(), N=N, BIAS=bias is not None, BC=256,
        num_warps=4)
    return out


@triton.jit
def _prepare(y_ptr, aa_ptr, bb_ptr, alog_ptr, dtb_ptr, anc_ptr, qn_ptr, kn_ptr, gn_ptr, beta_ptr, x_ptr, neg_ptr,
             N, s_y_n, s_a_n, s_b_n, KD, scale, HV: tl.constexpr, R: tl.constexpr, DK: tl.constexpr,
             BN: tl.constexpr, CACHED: tl.constexpr):
    ph = tl.program_id(0)
    b = ph // HV
    h = ph % HV
    hk = h // R
    on = tl.arange(0, BN)
    mn = on < N
    row = b * N + on
    od = tl.arange(0, DK)
    q = tl.load(y_ptr + row[:, None] * s_y_n + (hk * DK + od)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
    k = tl.load(y_ptr + row[:, None] * s_y_n + (KD + hk * DK + od)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, 1) + 1e-06)[:, None] * scale
    k = k * tl.rsqrt(tl.sum(k * k, 1) + 1e-06)[:, None]
    a = tl.load(aa_ptr + row * s_a_n + h, mask=mn, other=0.0).to(tl.float32) + tl.load(dtb_ptr + h).to(tl.float32)
    sp = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(tl.minimum(a, 20.0))))
    dec = tl.load(neg_ptr + h).to(tl.float32) if CACHED else -tl.exp(tl.load(alog_ptr + h).to(tl.float32))
    gl = tl.where(mn, dec * sp, 0.0)
    bl = tl.load(bb_ptr + row * s_b_n + h, mask=mn, other=0.0).to(tl.float32)
    beta = tl.where(mn, 1.0 / (1.0 + tl.exp(-bl)), 0.0)
    anc = tl.load(anc_ptr + b * N * N + on[:, None] * N + on[None, :], mask=mn[:, None] & mn[None, :], other=0.0)
    G = tl.sum(anc * gl[None, :], 1)
    isanc = anc > 0.5
    D = tl.exp(tl.where(isanc, G[:, None] - G[None, :], float('-inf')))
    offd = on[:, None] != on[None, :]
    A = tl.where(isanc & offd, tl.dot(k * beta[:, None], tl.trans(k)) * D, 0.0)
    tl.store(qn_ptr + (ph * N + on[:, None]) * DK + od[None, :], q, mask=mn[:, None])
    tl.store(kn_ptr + (ph * N + on[:, None]) * DK + od[None, :], k, mask=mn[:, None])
    tl.store(gn_ptr + ph * N + on, G, mask=mn)
    tl.store(beta_ptr + ph * N + on, beta, mask=mn)
    tl.store(x_ptr + (ph * N + on[:, None]) * N + on[None, :], A, mask=mn[:, None] & mn[None, :])


@triton.jit
def _finish(y_ptr, s0_ptr, anc_ptr, qn_ptr, kn_ptr, gn_ptr, beta_ptr, x_ptr, vn_ptr, o_ptr, slot_ptr, N, s_slot,
            s_y_n, s_s_h, s_s_k, s_s_v, s_o_n, s_o_h, KD, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
            BN: tl.constexpr, BV: tl.constexpr, SAVE: tl.constexpr):
    ph = tl.program_id(0)
    b = ph // HV
    h = ph % HV
    on = tl.arange(0, BN)
    mn = on < N
    row = b * N + on
    od = tl.arange(0, DK)
    ov = tl.program_id(1) * BV + tl.arange(0, BV)
    q = tl.load(qn_ptr + (ph * N + on[:, None]) * DK + od[None, :], mask=mn[:, None], other=0.0)
    k = tl.load(kn_ptr + (ph * N + on[:, None]) * DK + od[None, :], mask=mn[:, None], other=0.0)
    G = tl.load(gn_ptr + ph * N + on, mask=mn, other=0.0)
    beta = tl.load(beta_ptr + ph * N + on, mask=mn, other=0.0)
    X = tl.load(x_ptr + (ph * N + on[:, None]) * N + on[None, :], mask=mn[:, None] & mn[None, :], other=0.0)
    anc = tl.load(anc_ptr + b * N * N + on[:, None] * N + on[None, :], mask=mn[:, None] & mn[None, :], other=0.0)
    D = tl.exp(tl.where(anc > 0.5, G[:, None] - G[None, :], float('-inf')))
    eG = tl.exp(G)
    v = tl.load(y_ptr + row[:, None] * s_y_n + (2 * KD + h * DV + ov)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
    U = tl.dot(X, v * beta[:, None])
    Wk = tl.dot(X, k * beta[:, None] * eG[:, None])
    slot = tl.maximum(tl.load(slot_ptr + b), 0).to(tl.int64)
    S0 = tl.load(s0_ptr + slot * s_slot + h * s_s_h + od[:, None] * s_s_k + ov[None, :] * s_s_v)
    vn = U - tl.dot(Wk, S0)
    if SAVE:
        tl.store(vn_ptr + (ph * N + on[:, None]) * DV + ov[None, :], vn, mask=mn[:, None])
    Pq = tl.dot(q, tl.trans(k)) * D
    o = tl.dot(q * eG[:, None], S0) + tl.dot(Pq, vn)
    tl.store(o_ptr + row[:, None] * s_o_n + h * s_o_h + ov[None, :], o, mask=mn[:, None])


def tree_gdn(y, aa0, bb0, A_log, dt_bias, S0, anc, batch, Hk, Hv, Dk, Dv, aux, neg_exp_A, state_indices,
             state_slot_stride, scale=None, BV=64, num_warps=8):
    """`tree_kernels.tree_gdn_split` for `batch` requests: y (B * N, C) convolved qkv rows, anc (B, N, N),
    aux = (keys, values, gains) of shape (B * Hv, N, Dk), (B * Hv, N, Dv), (B * Hv, N)."""
    from fc1_block_inverse import invert16, invert32, invert64
    N = y.shape[0] // batch
    BN = max(16, 1 << (int(N) - 1).bit_length())
    scale = Dk ** (-0.5) if scale is None else scale
    assert BN <= 64 and Dv % BV == 0 and N * batch == y.shape[0]
    q = torch.empty(batch * Hv, N, Dk, device=y.device, dtype=torch.float32)
    k, values, G = aux
    beta = torch.empty(batch * Hv, N, device=y.device, dtype=torch.float32)
    X = torch.empty(batch * Hv, N, N, device=y.device, dtype=torch.float32)
    o = torch.empty(batch * N, Hv, Dv, device=y.device, dtype=y.dtype)
    anc = anc if anc.is_contiguous() and anc.dtype == torch.float32 else anc.float().contiguous()
    _prepare[batch * Hv,](y, aa0, bb0, A_log, dt_bias, anc, q, k, G, beta, X,
                          neg_exp_A if neg_exp_A is not None else A_log, N, y.stride(0), aa0.stride(0), bb0.stride(0),
                          Hk * Dk, float(scale), HV=Hv, R=Hv // Hk, DK=Dk, BN=BN, CACHED=neg_exp_A is not None,
                          num_warps=num_warps, num_stages=1)
    {16: invert16, 32: invert32, 64: invert64}[N][batch * Hv,](X, N, num_warps=4, num_stages=1)
    _finish[batch * Hv, Dv // BV](y, S0, anc, q, k, G, beta, X, values, o, state_indices, N, state_slot_stride,
                                  y.stride(0), S0.stride(0), S0.stride(1), S0.stride(2), o.stride(0), o.stride(1),
                                  Hk * Dk, HV=Hv, DK=Dk, DV=Dv, BN=BN, BV=BV, SAVE=True, num_warps=num_warps,
                                  num_stages=1)
    return o


@triton.jit
def _snapshots(S, K, V, G, A, Path, Ends, Slot, InterSlot, Out, B, KM, PS, NN: tl.constexpr, BS: tl.constexpr,
               SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr, SS4: tl.constexpr,
               OS0: tl.constexpr, OS1: tl.constexpr, OS2: tl.constexpr, OS3: tl.constexpr, OS4: tl.constexpr,
               OS5: tl.constexpr, GH: tl.constexpr, Q: tl.constexpr):
    lh = tl.program_id(0)
    l = (lh // GH).to(tl.int64)
    h = lh % GH
    ik = tl.program_id(1) * 32 + tl.arange(0, 32)
    bq = tl.program_id(2)
    b = bq // Q
    q = bq % Q
    iv = q % 4 * 32 + tl.arange(0, 32)
    which = q // 4
    u = tl.load(Ends + which * B + b)
    slot_raw = tl.load(Slot + b).to(tl.int64)
    dst_raw = tl.load(InterSlot + b).to(tl.int64)
    slot = tl.maximum(slot_raw, 0)
    dst = tl.maximum(dst_raw, 0)
    t = tl.arange(0, 32)
    flat = tl.load(Path + b * PS + t, mask=t < BS, other=-1)
    ids = tl.where(flat >= 0, flat - b * NN, -1)                # SGLang's flat node index -> this request's node
    valid = (u >= 0) & (ids >= 0)
    on = tl.load(A + b * NN * NN + tl.maximum(u, 0) * NN + tl.maximum(ids, 0), mask=valid, other=0) > 0
    hb = (l * KM + b) * GH + h                       # aux arrays: (layer, request slot of KM, head)
    gn = tl.load(G + hb * NN + tl.maximum(u, 0))
    gt = tl.load(G + hb * NN + tl.maximum(ids, 0), mask=on, other=0.0)
    weights = tl.exp(tl.where(on, gn - gt, float('-inf')))
    kk = tl.load(K + (hb * NN + tl.maximum(ids[None, :], 0)) * 128 + ik[:, None], mask=on[None, :], other=0.0)
    vv = tl.load(V + (hb * NN + tl.maximum(ids[:, None], 0)) * 128 + iv[None, :], mask=on[:, None], other=0.0)
    old = tl.load(S + l * SS0 + slot * SS1 + h * SS2 + iv[None, :] * SS3 + ik[:, None] * SS4)
    new = old * tl.exp(gn) + tl.dot(kk * weights[None, :], vv, input_precision='tf32x3')
    tl.store(Out + l * OS0 + dst * OS1 + tl.maximum(u, 0) * OS2 + h * OS3 + iv[None, :] * OS4 + ik[:, None] * OS5, new,
             mask=(u >= 0) & (slot_raw >= 0) & (dst_raw >= 0))


def snapshots(temporal, keys, values, gains, anc, accept_index, ends, slots, inter_slots, out, heads, nodes):
    """`fc1_fused.snapshots` per request: the state at each request's last accepted node (ends[0:B]) and,
    when ends holds 2B entries, at its track node (ends[B:2B]), written into `out` at that node.
    keys/values/gains are laid out (layers, capacity, heads, nodes, ...) with B <= capacity."""
    B = accept_index.shape[0]
    layers, capacity = keys.shape[0], keys.shape[1]
    assert B <= capacity and keys.is_contiguous() and values.is_contiguous() and gains.is_contiguous()
    q = 4 * (ends.numel() // B)
    _snapshots[layers * heads, 4, q * B](
        temporal, keys, values, gains, anc, accept_index, ends.contiguous(), slots, inter_slots, out, B, capacity,
        accept_index.stride(0), nodes, accept_index.shape[1], *temporal.stride(), *out.stride(), GH=heads, Q=q,
        num_warps=4)
