"""GPU check (not part of the CPU suite): ax_btree_kernels against the one-request fused kernels.

    PYTHONPATH=ax-engine-sglang:ax-engine-sglang/ax_engine_sglang \
        AX_LATTICE=1 FC1_TREE64=1 FC1_BRANCH_PILOT=1 FC1_FUSED=1 FC1_TREE_N=16 FC1_DRAFT_BLOCK=16 AX_BTREE=4 \
        python ax-engine-sglang/tests/gpu_btree_kernels.py

Every request of a batched launch must equal a batch-1 launch of the existing kernel on the same
inputs (conv output and windows, GDN output and saved keys/values/gains, committed snapshots), and a
padded row (slot -1) must store nothing.
"""
import torch

import ax_btree
import ax_btree_kernels as TKB
import fc1_conv
import fc1_fused
import tree_kernels as TK

torch.manual_seed(0)
dev = "cuda"
B, N, Hk, Hv, Dk, Dv = 3, 16, 16, 48, 128, 128
C = 2 * Hk * Dk + Hv * Dv
SLOTS, LAYERS = 9, 2
failures = []


def check(name, got, want):
    if not torch.equal(got, want):
        diff = (got.float() - want.float()).abs().max().item()
        failures.append(f"{name}: max abs diff {diff}")
    print(f"{name}: {'ok' if torch.equal(got, want) else 'DIFFERENT'}")


def random_parents(batch, nodes):
    rows = []
    for _ in range(batch):
        parent = [-1] + [int(torch.randint(0, i, (1,))) for i in range(1, nodes)]
        rows.append(parent)
    return torch.tensor(rows, device=dev)


anc, depth, nxt, sib, cidx = ax_btree.tree_arrays(random_parents(B, N))
slots = torch.tensor([5, 2, 7], device=dev, dtype=torch.int32)
inter = torch.tensor([1, 0, 3], device=dev, dtype=torch.int32)

# convolution
conv_state = torch.randn(SLOTS, C, 3, device=dev, dtype=torch.bfloat16)
x = torch.randn(B, C, N, device=dev, dtype=torch.bfloat16)
weight = torch.randn(C, 4, device=dev, dtype=torch.bfloat16)
bias = torch.randn(C, device=dev, dtype=torch.bfloat16)
window_b = torch.zeros(SLOTS, N, C, 3, device=dev, dtype=torch.bfloat16)
window_1 = torch.zeros_like(window_b)
out_b = TKB.tree_conv(x, conv_state, weight, bias, cidx.int().contiguous(), slots, inter, window_b)
for b in range(B):
    out_1 = torch.empty(1, N, C, device=dev, dtype=x.dtype)
    xb = x[b:b + 1]
    fc1_conv.conv[N, (C + 255) // 256](conv_state, xb, weight, bias, cidx[b].int().contiguous(), slots[b:b + 1],
                                       inter[b:b + 1], window_1, out_1, C, *conv_state.stride(), xb.stride(1),
                                       xb.stride(2), *window_1.stride(), BIAS=True, BC=256, num_warps=4)
    check(f"conv out request {b}", out_b[b], out_1[0])
check("conv windows", window_b, window_1)

# GDN tree verify
y = out_b.reshape(B * N, C)
aa = torch.randn(B * N, Hv, device=dev, dtype=torch.bfloat16)
bb = torch.randn(B * N, Hv, device=dev, dtype=torch.bfloat16)
A_log = torch.randn(Hv, device=dev)
dt_bias = torch.randn(Hv, device=dev)
neg = -A_log.float().exp()
pool = torch.randn(SLOTS, Hv, Dv, Dk, device=dev) * 0.1
state = pool[0].transpose(-1, -2)
kb = torch.empty(B * Hv, N, Dk, device=dev)
vb = torch.empty(B * Hv, N, Dv, device=dev)
gb = torch.empty(B * Hv, N, device=dev)
o_b = TKB.tree_gdn(y, aa, bb, A_log, dt_bias, state, anc, B, Hk, Hv, Dk, Dv, aux=(kb, vb, gb), neg_exp_A=neg,
                   state_indices=slots, state_slot_stride=pool.stride(0))
for b in range(B):
    k1 = torch.empty(Hv, N, Dk, device=dev)
    v1 = torch.empty(Hv, N, Dv, device=dev)
    g1 = torch.empty(Hv, N, device=dev)
    rows = slice(b * N, (b + 1) * N)
    o_1 = TK.tree_gdn_split(y[rows], aa[rows], bb[rows], A_log, dt_bias, state, anc[b], Hk, Hv, Dk, Dv,
                            aux=(k1, v1, g1), neg_exp_A=neg, state_indices=slots[b:b + 1],
                            state_slot_stride=pool.stride(0))
    check(f"gdn out request {b}", o_b[rows], o_1)
    check(f"gdn keys request {b}", kb[b * Hv:(b + 1) * Hv], k1)
    check(f"gdn values request {b}", vb[b * Hv:(b + 1) * Hv], v1)
    check(f"gdn gains request {b}", gb[b * Hv:(b + 1) * Hv], g1)

# commit snapshots (capacity 4 > B: the batched aux arrays keep unused request rows)
KM = 4
keys = torch.randn(LAYERS, KM, Hv, N, 128, device=dev)
values = torch.randn_like(keys)
gains = torch.randn(LAYERS, KM, Hv, N, device=dev) * 0.1
anc_cap = torch.eye(N, device=dev).repeat(KM, 1, 1)
anc_cap[:B] = anc
temporal = torch.randn(LAYERS, SLOTS, Hv, 128, 128, device=dev) * 0.1
steps = 8
accept = torch.full((B, steps), -1, device=dev, dtype=torch.long)
ends = torch.empty(B, device=dev, dtype=torch.long)
for b in range(B):   # a root path: node 0, then children down to some depth
    path, node = [0], 0
    while len(path) < steps:
        kids = [c for c in range(N) if c > 0 and int(anc[b, c, node]) == 1 and int(depth[b, c]) == len(path)]
        if not kids:
            break
        node = kids[0]
        path.append(node)
    path = path[:1 + b * 2] if b < B - 1 else path
    accept[b, :len(path)] = torch.tensor(path, device=dev) + b * N
    ends[b] = path[-1]
out_b = torch.zeros(LAYERS, SLOTS, N, Hv, 128, 128, device=dev)
out_1 = torch.zeros_like(out_b)
padded = torch.tensor([5, -1, 7], device=dev, dtype=torch.int32)
TKB.snapshots(temporal, keys, values, gains, anc_cap, accept, ends, padded, inter, out_b, Hv, N)
for b in (0, 2):
    local = torch.where(accept[b:b + 1] >= 0, accept[b:b + 1] - b * N, -1).contiguous()
    fc1_fused.snapshots[LAYERS * Hv, 4, 4](temporal, keys[:, b].contiguous(), values[:, b].contiguous(),
                                           gains[:, b].contiguous(), anc[b].contiguous(), local, ends[b:b + 1],
                                           padded[b:b + 1], inter[b:b + 1], out_1, N, steps, *temporal.stride(),
                                           *out_1.stride(), GH=Hv, num_warps=4)
check("snapshots (request 1 padded)", out_b, out_1)

# AX_SSM_POSITIONS: the accepted state written in place into the slot's persistent state (node stride 0)
capped = temporal.clone()
TKB.snapshots(capped, keys, values, gains, anc_cap, accept, ends, padded, padded,
              capped.unsqueeze(2).expand(LAYERS, SLOTS, N, Hv, 128, 128), Hv, N)
expected = temporal.clone()
for b in (0, 2):
    expected[:, int(padded[b])] = out_b[:, int(inter[b]), int(ends[b])]
check("snapshots in place (AX_SSM_POSITIONS, request 1 padded)", capped, expected)

print("FAILED: " + "; ".join(failures) if failures else "all batched kernels match the one-request kernels")
raise SystemExit(1 if failures else 0)
