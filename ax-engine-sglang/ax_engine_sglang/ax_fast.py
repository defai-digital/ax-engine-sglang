"""Overhead fixes for the DForkTree tree verifier, enabled with AX_FAST=1 (Track L, L0(d) rescue).

Copied into the plugin directory as `ax_fast.py` by benchmarks/sglang/patch_dforktree.py and called from
the patched fc1_branch64.verify_branch. Planned with codex (gpt-6.1-sol) after the round-time
profile; see results/track_l_l0/REPORT.md. Every change is meant to leave outputs unchanged:

1. No host readback of the prefix length when the engine runs without the host-side length mirror
   (triton attention: needs_cpu_seq_lens=False). The release, as we patched it to run, copied the
   prefix length to the host twice per round, a full GPU sync that also stops the scheduler from
   preparing the next round while the GPU works. The KV plan now gets the same planning bound the
   stock DFlash worker uses there (draft_input.nxt_kv_lens_cpu), and the tree mask is written on the
   GPU by one kernel that reads the prefix length from device memory:
       mask[i * (P + n) + j] = (j < P) or anc[i, j - P]
   The triton backend reads the mask through a GPU-side indptr, so a buffer longer than n * (P + n)
   is safe; its length comes from the host-side bound (>= P).
2. (in fc1_fused) the state-snapshot kernel launches half its programs when no track endpoint
   exists; the other half only computed values that were never stored.

AX_FORCE_SYNC=1 (ablation only) puts one full GPU synchronisation back into every round of the
fast path, the cost the host readback had, to test that the sync and nothing else explains the
speed difference.

AX_CHECK=1 additionally reads the exact prefix length back (sync) and compares the GPU mask with
the release's mask every round; the first short bound or mismatching mask raises (for validation
runs only), and the totals are printed at exit.

Buffers: two mask buffers are used alternately, so the mask of round k is never rewritten by round
k+1 even if a consumer of round k still runs on another stream; a request for more than 2**31 mask
entries (only possible with a nonsensical bound) raises instead of overflowing the index math.
"""
import atexit
import os
import sys

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("AX_FAST") == "1"
CHECK = os.environ.get("AX_CHECK") == "1"
FORCE_SYNC = os.environ.get("AX_FORCE_SYNC") == "1"
STATS = {"rounds": 0, "fast": 0, "mask_mismatch": 0, "bound_short": 0}


@triton.jit
def _tree_mask_kernel(Out, Anc, Prefix, L, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < L
    p = tl.load(Prefix).to(tl.int64)
    width = p + N
    row = offs // width
    col = offs - row * width
    in_tree = col >= p
    # The launch covers L = n * (bound + n) entries but rows are laid out with the real width p + N,
    # so the entries past n * (p + N) (present when bound > p) have row >= N and are never loaded
    # from Anc (the load mask), and col < width always, so col - p < N: no read outside Anc. Those
    # padding entries get the defined value (col < p); test_dft_fast.py checks the whole buffer.
    a = tl.load(Anc + row * N + (col - p), mask=inb & in_tree & (row < N), other=0.0)
    val = (~in_tree) | (a > 0.5)
    tl.store(Out + offs, val.to(tl.uint8), mask=inb)


def mode(batch):
    """'base' = release behaviour, 'fast' = no host readback, 'dedupe' = one readback per round."""
    if not ENABLED or batch.seq_lens_cpu is not None:
        return "base"
    di = batch.spec_info
    return "fast" if getattr(di, "nxt_kv_lens_cpu", None) is not None else "dedupe"


def plan_lens_cpu(batch, prefix, how):
    if how == "fast":
        return batch.spec_info.nxt_kv_lens_cpu
    if batch.seq_lens_cpu is not None:
        return batch.seq_lens_cpu
    return prefix.detach().to("cpu", torch.int64)


@atexit.register
def _report():
    if STATS["rounds"]:
        print(f"[ax_fast] final {STATS} check={CHECK}", file=sys.stderr, flush=True)


def tree_mask(w, prefix, anc, n, bound_cpu):
    bound = int(bound_cpu[0])
    length = n * (bound + n)
    if length >= 2 ** 31:
        raise RuntimeError(f"ax_fast: mask of {length} entries (bound {bound}) exceeds the supported size")
    bufs = getattr(w, "_ax_masks", None)
    if bufs is None:
        bufs = w._ax_masks = [None, None]
        w._ax_slot = 0
    w._ax_slot ^= 1
    buf = bufs[w._ax_slot]
    if buf is None or buf.numel() < length:
        buf = bufs[w._ax_slot] = torch.empty(max(2 * length, 1 << 20), dtype=torch.bool, device=anc.device)
    anc = anc.contiguous().float()
    block = 1024
    _tree_mask_kernel[(triton.cdiv(length, block),)](buf.view(torch.uint8), anc, prefix, length, N=n, BLOCK=block)
    STATS["rounds"] += 1
    STATS["fast"] += 1
    if FORCE_SYNC:
        torch.cuda.synchronize()
    if CHECK:
        plen = int(prefix[0].item())
        if bound < plen:
            STATS["bound_short"] += 1
            raise RuntimeError(f"ax_fast: planning bound {bound} below the prefix length {plen}")
        ref = torch.cat((torch.ones((n, plen), device=anc.device, dtype=torch.bool), anc.bool()), dim=1).flatten()
        if not torch.equal(buf[:ref.numel()], ref):
            STATS["mask_mismatch"] += 1
            raise RuntimeError(f"ax_fast: device mask differs from the release mask (prefix {plen})")
        if STATS["rounds"] % 200 == 0:
            print(f"[ax_fast] {STATS}", file=sys.stderr, flush=True)
    return buf[:length]
