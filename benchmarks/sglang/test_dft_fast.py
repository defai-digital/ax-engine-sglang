"""GPU check of the device-side tree mask in benchmarks/sglang/dft_fast.py (needs CUDA and triton).

    python benchmarks/sglang/test_dft_fast.py

For random trees and prefix lengths (including 0, 1, page edges and long contexts) and planning
bounds at or above the prefix, the first n * (P + n) mask entries must equal the release's
torch.cat((ones(n, P), anc), 1) exactly; consecutive calls must use different buffers; an absurd
bound must raise. Prints one line per check, exits non-zero on any failure.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dft_fast                                                        # noqa: E402


class W:
    pass


def random_anc(n, g):
    parent = [-1] + [int(torch.randint(0, i, (1,), generator=g)) for i in range(1, n)]
    anc = torch.zeros(n, n)
    for i in range(n):
        j = i
        while j >= 0:
            anc[i, j] = 1.0
            j = parent[j]
    return anc


def main():
    g = torch.Generator().manual_seed(0)
    ok = []
    w = W()
    for n in (16, 32, 64):
        for P in (0, 1, 7, 63, 64, 65, 1000, 4095, 4096, 20000):
            anc = random_anc(n, g).cuda()
            prefix = torch.tensor([P], device="cuda", dtype=torch.int64)
            for extra in (0, 24, 333):
                bound = torch.tensor([P + extra])
                got = dft_fast.tree_mask(w, prefix, anc, n, bound)
                ref = torch.cat((torch.ones((n, P), device="cuda", dtype=torch.bool), anc.bool()), dim=1).flatten()
                good = got.numel() == n * (P + extra + n) and torch.equal(got[:ref.numel()], ref)
                # the padding past the real rows (bound > prefix) is defined too: entry at offset o
                # is (o mod (P + n)) < P, and the kernel reads nothing from the ancestor table there
                o = torch.arange(ref.numel(), got.numel(), device="cuda")
                good = good and torch.equal(got[ref.numel():], (o % (P + n)) < P)
                ok.append(good)
                if not good:
                    print(f"FAIL mask n={n} P={P} bound=+{extra}")
    print(f"{'PASS' if all(ok) else 'FAIL'} mask: {sum(ok)}/{len(ok)} cases equal the release mask")
    a = dft_fast.tree_mask(w, torch.tensor([5], device="cuda"), random_anc(32, g).cuda(), 32, torch.tensor([5]))
    b = dft_fast.tree_mask(w, torch.tensor([5], device="cuda"), random_anc(32, g).cuda(), 32, torch.tensor([5]))
    alt = a.data_ptr() != b.data_ptr()
    ok.append(alt)
    print(f"{'PASS' if alt else 'FAIL'} buffers: consecutive rounds write different buffers")
    try:
        dft_fast.tree_mask(w, torch.tensor([5], device="cuda"), random_anc(64, g).cuda(), 64, torch.tensor([2 ** 26]))
        raised = False
    except RuntimeError:
        raised = True
    ok.append(raised)
    print(f"{'PASS' if raised else 'FAIL'} guard: a bound giving >= 2**31 entries raises")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
