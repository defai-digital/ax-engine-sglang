"""Compare GDN verification with independent unpacked decode, using CUDA tensors."""

from functools import partial

import torch

from ax_engine_sglang.ax_gdn_recurrent import one_row, replay, verify
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_mamba_state_scatter_with_mask
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel


def random_tree(width, generator):
    """Parents with every node below depth 16; node 0 is the root."""
    parents, depth = [-1], [0]
    for node in range(1, width):
        choices = [parent for parent in range(node) if depth[parent] < 15]
        parent = choices[int(torch.randint(len(choices), (1,), generator=generator))]
        parents.append(parent)
        depth.append(depth[parent] + 1)
    return parents


@torch.inference_mode()
def check(batch, width, dtype, tree, generator, capped=False):
    device = "cuda"
    key_heads, value_heads, dim = 16, 48, 128
    kernel = TritonGDNKernel()
    randn = partial(torch.randn, device=device, generator=torch.Generator(device).manual_seed(29 + width))
    q, k = [randn((1, batch * width, key_heads, dim), dtype=torch.bfloat16) for _ in range(2)]
    v = randn((1, batch * width, value_heads, dim), dtype=torch.bfloat16)
    a, b = [randn((batch * width, value_heads), dtype=torch.bfloat16) for _ in range(2)]
    A_log, dt_bias = [randn((value_heads,)) for _ in range(2)]
    slot_order = list(range(batch, 0, -1))  # live slots, not in row order
    slots = torch.tensor(slot_order, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    single = torch.arange(2, dtype=torch.int32, device=device)
    rows_index = torch.arange(batch, dtype=torch.int32, device=device)
    offsets = torch.arange(0, (batch + 1) * width, width, dtype=torch.int32, device=device)
    initial = (randn((batch + 1, value_heads, dim, dim)) * 0.1).to(dtype)
    original = initial.clone()
    ancestry = [random_tree(width, generator) if tree else list(range(-1, width - 1)) for _ in range(batch)]
    parents = torch.tensor(ancestry, dtype=torch.int32, device=device) if tree else None

    # The reference: every candidate alone, from its parent's stored state.
    expected = torch.empty_like(v)
    expected_snapshots = torch.empty((batch, width, value_heads, dim, dim), device=device, dtype=dtype)
    for row in range(batch):
        for node, parent in enumerate(ancestry[row]):
            state = (initial[slot_order[row]] if parent < 0 else expected_snapshots[row, parent])
            state = state.unsqueeze(0).clone()
            pos = row * width + node
            expected[:, pos:pos + 1] = kernel.decode(
                q[:, pos:pos + 1], k[:, pos:pos + 1], v[:, pos:pos + 1],
                a[pos:pos + 1], b[pos:pos + 1], A_log=A_log, dt_bias=dt_bias,
                ssm_states=state, cache_indices=zero, query_start_loc=single,
            )
            expected_snapshots[row, node].copy_(state[0])

    routes = {
        "verify": lambda **kw: verify(kernel.target_verify, A_log, dt_bias, q, k, v, a, b, **kw),
        "replay": lambda **kw: replay(kernel.decode, fused_mamba_state_scatter_with_mask,
                                      A_log, dt_bias, q, k, v, a, b, **kw),
    }
    if capped:  # AX_SSM_POSITIONS=8 with eight slots: nine rows of eight positions for one request
        del routes["replay"]

    for name, route in routes.items():
        snapshots = torch.zeros((9, 8, value_heads, dim, dim), device=device, dtype=dtype) if capped \
            else torch.zeros_like(expected_snapshots)
        kwargs = dict(ssm_states=initial, cache_indices=slots, query_start_loc=offsets,
                      intermediate_states_buffer=snapshots, intermediate_state_indices=rows_index,
                      cache_steps=width, retrieve_parent_token=parents)
        route(**kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = route(**kwargs)
        snapshots.zero_()
        graph.replay()
        label = f"{name}, batch {batch}, width {width}, {dtype}, {'tree' if tree else 'chain'}"
        assert torch.equal(actual, expected), f"outputs differ: {label}"
        stored = one_row(snapshots)[:, :width] if capped else snapshots
        assert torch.equal(stored, expected_snapshots), f"snapshots differ: {label}"
        if capped:  # the commit: SGLang's scatter reads the accepted candidate through the same view
            committed = torch.zeros((1, batch + 1, value_heads, dim, dim), device=device, dtype=dtype)
            accepted = width - 3
            fused_mamba_state_scatter_with_mask(
                committed, one_row(snapshots.unsqueeze(0), layers=1), slots,
                torch.tensor([accepted], dtype=torch.int32, device=device))
            assert torch.equal(committed[0, slot_order[0]], expected_snapshots[0, accepted]), f"commit differs: {label}"
        assert torch.equal(initial, original), f"persistent state changed: {label}"


def main():
    import sys

    if "--parity-tile" in sys.argv:  # the value tile AX_NVFP4_VERIFY_PARITY=1 gives decode and verify
        from ax_engine_sglang.ax_gdn_recurrent import install_value_tile
        from ax_engine_sglang.compat import installation

        with installation():
            install_value_tile()
    generator = torch.Generator().manual_seed(29)
    cases = 0
    for dtype in (torch.float32, torch.bfloat16):
        for batch, width in ((2, 4), (1, 8), (2, 8), (1, 16), (8, 8), (1, 32), (1, 64)):
            for tree in (False, True):
                check(batch, width, dtype, tree, generator)
                cases += 1
        for width in (16, 64):
            check(1, width, dtype, True, generator, capped=True)
            cases += 1
    print(f"PASS: one-launch GDN verification and the per-candidate replay matched unpacked decode "
          f"bit for bit on {cases} chain and tree cases")


if __name__ == "__main__":
    main()
