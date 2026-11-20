"""Compare GDN verification with independent unpacked decode, using CUDA tensors."""

import torch

from ax_engine_sglang.ax_gdn_recurrent import replay
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_mamba_state_scatter_with_mask
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel


@torch.inference_mode()
def main():
    torch.manual_seed(29)
    device = "cuda"
    batch, width, key_heads, value_heads, dim = 2, 4, 16, 48, 128
    kernel = TritonGDNKernel()
    q, k = [torch.randn((1, batch * width, key_heads, dim), device=device, dtype=torch.bfloat16)
            for _ in range(2)]
    v = torch.randn((1, batch * width, value_heads, dim), device=device, dtype=torch.bfloat16)
    a, b = [torch.randn((batch * width, value_heads), device=device, dtype=torch.bfloat16)
            for _ in range(2)]
    A_log, dt_bias = [torch.randn(value_heads, device=device) for _ in range(2)]
    slot_order = [2, 0]
    slots = torch.tensor(slot_order, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    single = torch.arange(2, dtype=torch.int32, device=device)
    offsets = torch.arange(0, (batch + 1) * width, width, dtype=torch.int32, device=device)
    initial = torch.randn((3, value_heads, dim, dim), device=device) * 0.1
    original = initial.clone()
    for rows in (None, [[-1, 0, 0, 1], [-1, 0, 1, 0]]):
        ancestry = rows or [[-1, 0, 1, 2]] * batch
        parents = torch.tensor(rows, dtype=torch.int32, device=device) if rows is not None else None
        snapshots = torch.zeros((batch, width, value_heads, dim, dim), device=device)
        expected = torch.empty_like(v)
        expected_snapshots = torch.empty_like(snapshots)
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
        kwargs = dict(ssm_states=initial, cache_indices=slots, query_start_loc=offsets,
                      intermediate_states_buffer=snapshots, intermediate_state_indices=single,
                      cache_steps=width, retrieve_parent_token=parents)
        replay(kernel.decode, fused_mamba_state_scatter_with_mask, A_log, dt_bias, q, k, v, a, b, **kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = replay(kernel.decode, fused_mamba_state_scatter_with_mask,
                            A_log, dt_bias, q, k, v, a, b, **kwargs)
        snapshots.zero_()
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(snapshots, expected_snapshots, rtol=0, atol=0)
        torch.testing.assert_close(initial, original, rtol=0, atol=0)
    print("PASS: CUDA graph verification matched unpacked GDN decode on chain and tree inputs")


if __name__ == "__main__":
    main()
