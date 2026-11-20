"""Is SGLang's tree_speculative_sampling_target_only lossless? A direct Monte-Carlo check of the kernel.

  python benchmarks/sglang/target_only_kernel_check.py [--trials 400000] [--vocab 512]

Builds one small draft tree with fixed target distributions on every node, runs the kernel on many
independent copies (one batch row each, fresh coins), and compares the distribution of the first
committed token (the accepted depth-1 node, or the bonus token when nothing is accepted) with the
root's target distribution p. A lossless verify makes them equal for any tree whose siblings carry
distinct tokens. Trees checked: a chain, a root with three distinct children, the same with two
children carrying the same token (not a valid tree; shows the effect of duplicates), and a tree
whose depth-1 children list a token with zero target mass.
"""
import argparse

import torch


def build(tree, n):
    """tree: list of (parent, token) for nodes 1..n-1; node 0 is the root."""
    parent = [-1] + [p for p, _ in tree]
    tokens = [0] + [t for _, t in tree]
    nxt, sib = [-1] * n, [-1] * n
    for i in range(1, n):
        p = parent[i]
        if nxt[p] == -1:
            nxt[p] = i
        else:
            j = nxt[p]
            while sib[j] != -1:
                j = sib[j]
            sib[j] = i
    return tokens, nxt, sib


def run(tokens, nxt, sib, probs, trials, depth):
    from sglang.kernels.ops.speculative.sampling import tree_speculative_sampling_target_only
    dev, n, d = "cuda", len(tokens), probs.shape[-1]
    B = trials
    cand = torch.tensor(tokens, device=dev, dtype=torch.int64)[None].expand(B, n).contiguous()
    ridx = (torch.arange(B, device=dev)[:, None] * n + torch.arange(n, device=dev)[None]).contiguous()
    rnt = torch.tensor(nxt, device=dev, dtype=torch.int64)[None].expand(B, n).contiguous()
    rns = torch.tensor(sib, device=dev, dtype=torch.int64)[None].expand(B, n).contiguous()
    tp = probs[None].expand(B, n, d).contiguous()
    dp = torch.zeros_like(tp)
    predicts = torch.full((B * n,), -1, device=dev, dtype=torch.int32)
    acc_idx = torch.full((B, depth + 1), -1, device=dev, dtype=torch.int32)
    acc_num = torch.zeros(B, device=dev, dtype=torch.int32)
    tree_speculative_sampling_target_only(
        predicts=predicts, accept_index=acc_idx, accept_token_num=acc_num, candidates=cand,
        retrive_index=ridx, retrive_next_token=rnt, retrive_next_sibling=rns,
        uniform_samples=torch.rand(B, n, device=dev), uniform_samples_for_final_sampling=torch.rand(B, device=dev),
        target_probs=tp, draft_probs=dp, threshold_single=1.0, threshold_acc=1.0, deterministic=True)
    # first committed token: the token written for the root's retrieve slot
    first = predicts.view(B, n)[:, 0].long()
    # the root slot holds the accepted depth-1 child or, when every child is rejected, the bonus token
    assert bool((first >= 0).all()), "root slot not written: the first committed token is not where this check reads it"
    assert bool((first[acc_num == 0] >= 0).all())
    return torch.bincount(first, minlength=d).double() / B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400000)
    ap.add_argument("--vocab", type=int, default=512)
    a = ap.parse_args()
    torch.manual_seed(0)
    d = a.vocab
    # root distribution: a few heavy tokens and a long tail
    logits = torch.randn(d, device="cuda") * 1.0
    logits[:6] += torch.tensor([3.0, 2.6, 2.2, 1.8, 1.5, 1.2], device="cuda")
    logits[7] = -float("inf")
    p_root = torch.softmax(logits, -1)
    trees = {
        "chain": [(0, 1), (1, 3), (2, 5)],
        "three children": [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5)],
        "duplicate children (invalid)": [(0, 1), (0, 1), (0, 3), (1, 4)],
        "zero-mass child": [(0, 7), (0, 1), (2, 4)],
    }
    worst = 0.0
    for name, tree in trees.items():
        n = len(tree) + 1
        tokens, nxt, sib = build(tree, n)
        probs = torch.stack([p_root] + [torch.softmax(torch.randn(d, device="cuda") * 2, -1) for _ in range(n - 1)])
        depth = 1
        for i in range(1, n):
            k, dd = i, 0
            while k != 0:
                k = ([-1] + [p for p, _ in tree])[k]
                dd += 1
            depth = max(depth, dd)
        emp = run(tokens, nxt, sib, probs, a.trials, depth + 1)
        se = (p_root.double() * (1 - p_root.double()) / a.trials).sqrt().clamp_min(1e-12)
        z = ((emp - p_root.double()) / se).abs()
        top = torch.argsort(p_root, descending=True)[:6]
        print(f"{name}: max |z| = {z.max():.1f}")
        for t in top.tolist():
            print(f"   token {t:4d}  p {p_root[t]:.4f}  kernel {emp[t]:.4f}  z {z[t]:+.1f}")
        if "invalid" not in name:
            worst = max(worst, float(z.max()))
    print("KERNEL_LOSSLESS" if worst < 5 else f"KERNEL_BIASED worst |z| {worst:.1f}")


if __name__ == "__main__":
    main()
