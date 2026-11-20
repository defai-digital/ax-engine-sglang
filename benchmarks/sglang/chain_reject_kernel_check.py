"""Is SGLang's chain_speculative_sampling_triton (DFlash2 sampled verify) lossless? Monte-Carlo check.

  python benchmarks/sglang/chain_reject_kernel_check.py [--trials 200000] [--vocab 512] [--k 16]

DFlash2's sampled verify (`_selector_sampling_accept` -> dspark `accept_sampling`) samples a draft
chain from a sparse proposal q (K candidates per position, `selector_walk_triton`) and runs
classic rejection sampling against the target p. This draws the draft tokens truly from q, runs the
kernel on many independent rows with fresh coins, and compares the distribution of the first
committed token with p at the root. A lossless kernel matches p for any q. --target-topk N zeroes the
target outside its top N (as the model card's top-k 20 does), so some draft candidates carry p = 0;
the "q near one-hot" case puts almost all draft mass on one candidate, as the DFlash2 selector often does.
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200000)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--target-topk", type=int, default=0, help="zero the target outside its top N (0: off)")
    a = ap.parse_args()
    from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton
    from sglang.srt.speculative.dflash_utils import _get_or_create_chain_verify_buffers
    torch.manual_seed(0)
    dev, d, K, g, B = "cuda", a.vocab, a.k, a.gamma, a.trials
    block = g + 1
    logits = torch.randn(block, d, device=dev) * 2
    if a.target_topk:
        kth = torch.topk(logits, a.target_topk, -1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    p_rows = torch.softmax(logits, -1)                                          # target, one per verify row
    base = torch.randn(block, d, device=dev) * 2                                # draft's own view, not p
    worst = 0.0
    for name, q_temp in (("q close to p", 1.0), ("q sharper than p", 0.3), ("q flatter than p", 3.0),
                         ("q near one-hot", 0.02)):
        # sparse proposal: K candidates per draft position, top-K of the draft's own scores (a perturbed
        # copy of the target logits, finite everywhere, so candidates may carry p = 0), softmax with temperature
        src = torch.where(torch.isfinite(logits), logits, base) + 0.5 * torch.randn(block, d, device=dev)
        cand = torch.stack([torch.topk(src[s], K).indices for s in range(g)])
        q_small = torch.stack([torch.softmax(src[s][cand[s]] / q_temp, -1) for s in range(g)])  # (g, K)
        q_dense = torch.zeros(g, d, device=dev)
        q_dense.scatter_(1, cand, q_small)
        # draft tokens truly drawn from q, independently per row
        pick = torch.stack([torch.multinomial(q_small[s], B, replacement=True) for s in range(g)], 1)   # (B, g)
        drafts = cand[torch.arange(g, device=dev)[None], pick]                                           # (B, g)
        candidates = torch.cat((torch.zeros(B, 1, device=dev, dtype=torch.int64), drafts), 1).contiguous()
        retrieve_index, rnt, rns, predicts, accept_index, accept_num = _get_or_create_chain_verify_buffers(
            bs=B, draft_token_num=block, device=dev)
        chain_speculative_sampling_triton(
            predicts=predicts, accept_index=accept_index, accept_token_num=accept_num, candidates=candidates,
            retrive_index=retrieve_index, retrive_next_token=rnt, retrive_next_sibling=rns,
            uniform_samples=torch.rand(B, g, device=dev), uniform_samples_for_final_sampling=torch.rand(B, device=dev),
            target_probs=p_rows[None].expand(B, block, d).contiguous(), draft_probs=q_dense[None].expand(B, g, d).contiguous(),
            threshold_single=1.0, threshold_acc=1.0, deterministic=True)
        # first committed token: drafts[:,0] if accept_num >= 1 else the bonus sampled at row 0
        acc = accept_num.long()
        rows = torch.arange(B, device=dev)
        bonus = predicts[accept_index[rows, acc].long()].long()
        first = torch.where(acc >= 1, drafts[:, 0], bonus)
        emp = torch.bincount(first, minlength=d).double() / B
        p0 = p_rows[0].double()
        z = ((emp - p0) / (p0 * (1 - p0) / B).sqrt().clamp_min(1e-12)).abs()
        z = torch.where(p0 > 0, z, (emp > 0).double() * 1e9)
        print(f"{name}: max |z| = {z.max():.1f}  accept-len mean {acc.double().mean():.2f}")
        for t in torch.argsort(p0, descending=True)[:5].tolist():
            print(f"   token {t:4d}  p {p0[t]:.4f}  kernel {emp[t]:.4f}  in q {bool((cand[0] == t).any())}")
        worst = max(worst, float(z.max()))
    print("CHAIN_KERNEL_LOSSLESS" if worst < 5 else f"CHAIN_KERNEL_BIASED worst |z| {worst:.1f}")


if __name__ == "__main__":
    main()
