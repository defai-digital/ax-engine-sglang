"""CPU checks of benchmarks/sglang/dft_lattice.py against the plugin's own tree search (no model needed).

    python benchmarks/sglang/test_dft_lattice.py --plugin /root/dft-site/dforktree

1 reduce   a lattice whose rows all equal the unary scores gives the same c0 / logc as the release
           Builder's unary path (no n-gram), hence the same tree from tree_ops.beam_tree
2 rows     with a lattice whose row for parent candidate p strongly prefers child (p + 1) mod K,
           every selected node is that child of its parent (the parent's row is the one used)
3 norm     every row of exp(logc) plus the outside mass sums to 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True, help="dforktree package directory (for tree_ops.beam_tree)")
    args = ap.parse_args()
    sys.path.insert(0, args.plugin)
    sys.path.insert(0, str(HERE))
    import tree_ops as TO
    import dft_lattice as DL

    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)

    g = torch.Generator().manual_seed(0)
    K, G, N = 16, 15, 32
    good = True
    for trial in range(20):
        logits = torch.randn(G, 1000, generator=g) * 3
        lp, ids = torch.log_softmax(logits.float(), -1).topk(K, -1)
        rest = torch.log((1 - lp.exp().sum(-1)).clamp(1e-06, 1.0))
        # release Builder, unary path
        c0_ref = lp[0] - torch.logsumexp(torch.cat((lp[0], rest[:1])), 0)
        base = lp[1:, None, :].expand(G - 1, K, K)
        logc_ref = base - torch.logsumexp(torch.cat((base, rest[1:, None, None].expand(G - 1, K, 1)), dim=-1), -1, keepdim=True)
        # lattice with identical rows: scores = unary log-probs, tail = outside log mass
        lat = lp[:, None, :].expand(G, K, K).contiguous()
        c0, logc = DL.lattice_logc(lat, rest)
        good &= torch.allclose(c0, c0_ref, atol=1e-5) and torch.allclose(logc, logc_ref, atol=1e-5)
        a = TO.beam_tree(c0_ref, logc_ref, N, 99, False, None, S=G)
        b = TO.beam_tree(c0, logc, N, 99, False, None, S=G)
        good &= all(torch.equal(x, y) for x, y in zip(a[:3], b[:3]))
    check("reduce", good, "(identical rows = release unary Builder, same beam_tree output)")

    lat = torch.randn(G, K, K, generator=g) * 2
    tail = torch.randn(G, generator=g)
    c0, logc = DL.lattice_logc(lat, tail)
    full = torch.cat((logc.exp().sum(-1), ), 0) + torch.exp(tail[1:, None] - torch.logsumexp(
        torch.cat((lat[1:], tail[1:, None, None].expand(G - 1, K, 1)), -1), -1))
    check("norm", torch.allclose(full, torch.ones_like(full), atol=1e-5)
          and abs(float(c0.exp().sum() + torch.exp(tail[0] - torch.logsumexp(torch.cat((lat[0, 0], tail[:1])), 0))) - 1) < 1e-5)

    # each parent row strongly prefers child (parent's candidate + 1) mod K: every selected non-root
    # node must then be that child of its parent, which holds only if the row of the parent is used
    lat = torch.full((G, K, K), -20.0)
    for p_ in range(K):
        lat[:, p_, (p_ + 1) % K] = 20.0
    c0, logc = DL.lattice_logc(lat, torch.full((G,), -20.0))
    n_chain = G + 1                                   # budget = the strong chain exactly (root + G nodes)
    ss, cand, parent, _ = TO.beam_tree(c0, logc, n_chain, 99, False, None, S=G)
    good = len(ss) == n_chain - 1
    for j in range(len(ss)):
        if parent[j] >= 0:
            good &= int(cand[j]) == (int(cand[int(parent[j])]) + 1) % K and int(ss[j]) == int(ss[int(parent[j])]) + 1
    check("rows", good, f"({len(ss)} nodes follow their parent's lattice row)")
    print(f"{sum(ok)}/{len(ok)} checks passed")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
