"""GPU check of the DForkTree fused tree Gated-DeltaNet kernel against a plain recurrence.

    python benchmarks/sglang/test_tree_gdn.py --plugin /root/dft-site/dforktree

For random trees of N nodes, `tree_kernels.tree_gdn_split` (the plugin's fused verifier) must give,
at every node, the output of the gated delta rule run sequentially along that node's ancestor
path from the committed state S0:
    g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b), q, k L2-normalised, q *= Dk^-0.5
    S <- exp(g) S;  S <- S + k (beta (v - S^T k))^T;  o = S^T q
Shapes: 9B (16 key heads -> 32 value heads) and 27B (16 -> 48); value head h reads key head h // R.
Then the state-commit kernel (fc1_fused.snapshots) at the full 27B buffer size, see snapshot_check.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def reference(y, a, b, A_log, dt_bias, S0, parent, Hk, Hv, Dk, Dv, states=None):
    N = y.shape[0]
    R = Hv // Hk
    KD = Hk * Dk
    q = y[:, :KD].float().view(N, Hk, Dk)
    k = y[:, KD:2 * KD].float().view(N, Hk, Dk)
    v = y[:, 2 * KD:].float().view(N, Hv, Dv)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * Dk ** -0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())       # (N, Hv)
    beta = torch.sigmoid(b.float())
    out = torch.empty(N, Hv, Dv, device=y.device)
    for n in range(N):
        path = []
        u = n
        while u >= 0:
            path.append(u)
            u = int(parent[u])
        S = S0.float().clone()                                               # (Hv, Dk, Dv)
        for t in reversed(path):
            kk = k[t].repeat_interleave(R, 0)                                # (Hv, Dk)
            S = S * g[t].exp()[:, None, None]
            pred = torch.einsum("hk,hkv->hv", kk, S)
            upd = beta[t][:, None] * (v[t] - pred)
            S = S + kk[:, :, None] * upd[:, None, :]
        qq = q[n].repeat_interleave(R, 0)
        out[n] = torch.einsum("hk,hkv->hv", qq, S)
        if states is not None:
            states[n] = S
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--skip-snapshot", action="store_true", help="skip the 9.7 GB commit-kernel check")
    args = ap.parse_args()
    sys.path.insert(0, args.plugin)
    import tree_kernels as TK

    torch.manual_seed(0)
    dev = "cuda"
    ok = True
    for name, Hk, Hv in (("9B", 16, 32), ("27B", 16, 48)):
        Dk = Dv = 128
        for N, kind in ((32, "chain"), (32, "tree"), (16, "tree"), (64, "tree")):
            parent = [-1] + ([i - 1 for i in range(1, N)] if kind == "chain"
                             else [int(torch.randint(0, i, (1,))) for i in range(1, N)])
            anc = torch.zeros(N, N, device=dev)
            for n in range(N):
                u = n
                while u >= 0:
                    anc[n, u] = 1.0
                    u = parent[u]
            y = torch.randn(N, 2 * Hk * Dk + Hv * Dv, device=dev, dtype=torch.bfloat16)
            a = torch.randn(N, Hv, device=dev, dtype=torch.bfloat16)
            b = torch.randn(N, Hv, device=dev, dtype=torch.bfloat16)
            A_log = torch.randn(Hv, device=dev) * 0.5
            dt_bias = torch.randn(Hv, device=dev) * 0.5
            # SGLang keeps the state as (slot, H, V, K); the plugin passes pool[0].transpose(-1, -2)
            pool = torch.randn(2, Hv, Dv, Dk, device=dev) * 0.1
            S0 = pool[0].transpose(-1, -2)
            o = TK.tree_gdn_split(y, a, b, A_log, dt_bias, S0, anc, Hk, Hv, Dk, Dv).float()
            ref = reference(y, a, b, A_log, dt_bias, S0, torch.tensor(parent), Hk, Hv, Dk, Dv)
            err = ((o - ref).abs().max() / ref.abs().max()).item()
            good = err < 2e-2
            ok &= good
            print(f"{'PASS' if good else 'FAIL'} {name} Hk={Hk} Hv={Hv} N={N} {kind}: max rel err {err:.2e}", flush=True)
    if not args.skip_snapshot:
        ok &= snapshot_check(args.plugin)
    sys.exit(0 if ok else 1)


def snapshot_check(plugin):
    """The state-commit kernel `fc1_fused.snapshots` at 27B sizes, for tree N = 32 and 64.

    contract  full-size buffers (48 GDN layers x 48 heads; intermediate SSM (48, 2, N, 48, 128, 128)
              fp32 = 9.7 GB at N = 32, 19.3 GB at N = 64) with synthetic factors: the state written
              must equal old * exp(G_u) + sum over accepted ancestors t of exp(G_u - G_t) k_t v_t^T,
              checked at layers on both sides of the int32 boundary. With the release's int32 layer
              offset the kernel writes out of bounds for layers >= 43 (N = 32) or >= 22 (N = 64) and
              the process usually dies with a CUDA illegal-access error instead of a FAIL line.
    seam      one layer, real factors: the fused verifier (`tree_gdn_split` with aux) stores keys,
              delta-corrected values and path gains; `snapshots` turns them into the committed
              state, which must equal the sequential recurrence's state at the accepted node and at
              a second (track) node. This is the only check of what verify hands to commit.
    Not covered: slot / intermediate-slot indices large enough to wrap int32 (hundreds of GB)."""
    import importlib
    import os
    if plugin not in sys.path:
        sys.path.insert(0, plugin)
    import tree_kernels as TK
    good = True
    for NN in (32, 64):
        saved = {k: os.environ.get(k) for k in ("FC1_TREE_N", "AX_GDN_LAYERS", "AX_GDN_HEADS")}
        os.environ.update(FC1_TREE_N=str(NN), AX_GDN_LAYERS="48", AX_GDN_HEADS="48")
        try:
            sys.modules.pop("fc1_fused", None)
            FU = importlib.import_module("fc1_fused")          # reads the sizes at import
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        good &= _contract(FU, NN)
        good &= _seam(FU, TK, NN)
    return good


def _tree(NN, parent, dev):
    anc = torch.zeros(NN, NN, device=dev)
    for n in range(NN):
        u = n
        while u >= 0:
            anc[n, u] = 1.0
            u = parent[u]
    return anc


def _contract(FU, NN):
    GL, GH, D, dev = 48, 48, 128, "cuda"
    g = torch.Generator(device=dev).manual_seed(1)
    keys = torch.randn(GL, GH, NN, D, device=dev, generator=g) * 0.1
    values = torch.randn(GL, GH, NN, D, device=dev, generator=g) * 0.1
    gains = -torch.rand(GL, GH, NN, device=dev, generator=g).cumsum(-1) * 0.05
    parent = [-1, 0, 1, 1, 3, 4] + [0] * (NN - 6)
    anc = _tree(NN, parent, dev)
    path = torch.full((1, 16), -1, device=dev, dtype=torch.long)
    path[0, :5] = torch.tensor([0, 1, 3, 4, 5])
    slot = torch.tensor([1], device=dev, dtype=torch.long)
    inter = torch.tensor([0], device=dev, dtype=torch.long)
    temporal = torch.randn(GL, 2, GH, D, D, device=dev, generator=g) * 0.1
    wrap = (2 ** 31) // (2 * NN * GH * D * D) + 1               # first layer whose element offset passes 2^31
    layers = sorted({0, wrap - 1, wrap, GL - 1})
    good = True
    # grid 4 = AX_FAST with no track endpoint (one end); grid 8 = the release path, ends = (last, track)
    for grid, end_nodes in ((4, [5]), (8, [5, 3])):
        ends = torch.tensor(end_nodes, device=dev, dtype=torch.long)
        out = torch.zeros(GL, 2, NN, GH, D, D, device=dev)
        FU.snapshots[GL * GH, 4, grid](temporal, keys, values, gains, anc, path, ends, slot, inter, out, NN, path.shape[1],
                                       *temporal.stride(), *out.stride(), GH=GH, num_warps=4)
        torch.cuda.synchronize()
        worst = 0.0
        for u in end_nodes:
            ids = [t for t in path[0].tolist() if t >= 0 and anc[u, t] > 0]
            for l in layers:
                for h in (0, 17, GH - 1):
                    ref = temporal[l, 1, h].T * gains[l, h, u].exp()             # (K, V)
                    for t in ids:
                        ref = ref + (gains[l, h, u] - gains[l, h, t]).exp() * torch.outer(keys[l, h, t], values[l, h, t])
                    got = out[l, 0, u, h].T
                    worst = max(worst, ((got - ref).abs().max() / ref.abs().max()).item())
        good &= worst < 1e-3
        print(f"{'PASS' if worst < 1e-3 else 'FAIL'} snapshot contract 27B N={NN} grid {grid} ends {end_nodes} layers {layers} "
              f"heads 0/17/47: max rel err {worst:.2e}", flush=True)
        del out
    del temporal, keys, values, gains
    torch.cuda.empty_cache()
    return good


def _seam(FU, TK, NN):
    Hk, GH, D, dev = 16, 48, 128, "cuda"
    torch.manual_seed(2)
    parent = [-1] + [int(torch.randint(max(i - 3, 0), i, (1,))) for i in range(1, NN)]   # deep random tree
    anc = _tree(NN, parent, dev)
    depth = anc.sum(-1).long().tolist()
    u = max((n for n in range(NN) if depth[n] <= 16), key=lambda n: depth[n])
    chain = sorted((t for t in range(NN) if anc[u, t] > 0), key=lambda t: depth[t])
    track = chain[len(chain) // 2]
    y = torch.randn(NN, 2 * Hk * D + GH * D, device=dev, dtype=torch.bfloat16)
    a = torch.randn(NN, GH, device=dev, dtype=torch.bfloat16)
    b = torch.randn(NN, GH, device=dev, dtype=torch.bfloat16)
    A_log = torch.randn(GH, device=dev) * 0.5
    dt_bias = torch.randn(GH, device=dev) * 0.5
    temporal = (torch.randn(1, 2, GH, D, D, device=dev) * 0.1).contiguous()      # (layer, slot, H, V, K)
    S0 = temporal[0, 1].transpose(-1, -2)
    keys = torch.empty(1, GH, NN, D, device=dev)
    values = torch.empty_like(keys)
    gains = torch.empty(1, GH, NN, device=dev)
    TK.tree_gdn_split(y, a, b, A_log, dt_bias, S0, anc, Hk, GH, D, D, aux=(keys[0], values[0], gains[0]))
    states = torch.empty(NN, GH, D, D, device=dev)                               # (node, H, K, V)
    reference(y, a, b, A_log, dt_bias, S0, torch.tensor(parent), Hk, GH, D, D, states=states)
    path = torch.full((1, 16), -1, device=dev, dtype=torch.long)
    path[0, :len(chain)] = torch.tensor(chain)
    ends = torch.tensor([u, track], device=dev, dtype=torch.long)
    slot = torch.tensor([1], device=dev, dtype=torch.long)
    inter = torch.tensor([0], device=dev, dtype=torch.long)
    out = torch.zeros(1, 2, NN, GH, D, D, device=dev)
    FU.snapshots[GH, 4, 8](temporal, keys, values, gains, anc, path, ends, slot, inter, out, NN, path.shape[1],
                           *temporal.stride(), *out.stride(), GH=GH, num_warps=4)
    torch.cuda.synchronize()
    worst = 0.0
    for e in (u, track):
        got = out[0, 0, e].transpose(-1, -2)                                     # (H, K, V)
        worst = max(worst, ((got - states[e]).abs().max() / states[e].abs().max()).item())
    good = worst < 2e-2
    print(f"{'PASS' if good else 'FAIL'} verify->commit seam 27B N={NN} accepted depth {len(chain)} end {u} track {track}: "
          f"max rel err {worst:.2e}", flush=True)
    return good


if __name__ == "__main__":
    main()
