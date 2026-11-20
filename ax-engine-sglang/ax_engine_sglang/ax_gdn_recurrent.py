"""Verify GDN branches with the same recurrent kernel as unpacked decode."""

from .compat import patch_attribute


def replay(decode, scatter, A_log, dt_bias, q, k, v, a, b, *,
           ssm_states, cache_indices, query_start_loc, intermediate_states_buffer,
           intermediate_state_indices, cache_steps, retrieve_parent_token, **kwargs):
    import torch

    batch_size = query_start_loc.numel() - 1
    if (batch_size <= 0 or cache_steps <= 0 or q.shape[:2] != (1, batch_size * cache_steps)
            or intermediate_states_buffer.shape[1] < cache_steps):
        raise ValueError("AX GDN recurrent verify requires a uniform verification window and full snapshots")
    if retrieve_parent_token is not None and retrieve_parent_token.shape != (batch_size, cache_steps):
        raise ValueError("AX GDN recurrent verify requires one local parent index per candidate")

    live_slots = cache_indices[:batch_size]
    snapshot_slots = intermediate_state_indices[:batch_size].to(torch.int32)
    has_tokens = query_start_loc[1:] > query_start_loc[:-1]
    valid = (live_slots >= 0) & (snapshot_slots >= 0) & has_tokens
    row_offsets = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
    rows = row_offsets[:-1]
    decode_slots = rows.masked_fill(~valid, -1)
    write_steps = torch.zeros_like(rows).masked_fill(~valid, -1)

    initial = ssm_states.index_select(0, live_slots.clamp_min(0).long())
    state = initial.clone() if retrieve_parent_token is not None else initial
    result = q.new_empty(v.shape)
    for step in range(cache_steps):
        if step and retrieve_parent_token is not None:
            state.copy_(initial)
            parents = retrieve_parent_token[:, step].masked_fill(~valid, -1)
            scatter(state.unsqueeze(0), intermediate_states_buffer.unsqueeze(0),
                    rows, parents, snapshot_slots)

        decoded = decode(
            q[:, step::cache_steps], k[:, step::cache_steps], v[:, step::cache_steps],
            a[..., step::cache_steps, :], b[..., step::cache_steps, :],
            A_log=A_log, dt_bias=dt_bias, ssm_states=state,
            cache_indices=decode_slots, query_start_loc=row_offsets,
        )
        result[:, step::cache_steps].copy_(decoded)
        # Use the native masked scatter to keep padding out of live scratch rows.
        # The scheduler still owns copying the accepted snapshot to persistent state.
        scatter(intermediate_states_buffer[:, step].unsqueeze(0),
                state.unsqueeze(0).unsqueeze(2), snapshot_slots, write_steps, rows)
    return result


def install():
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_mamba_state_scatter_with_mask
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    def target_verify(self, A_log, dt_bias, q, k, v, a, b, **kwargs):
        return replay(self.decode, fused_mamba_state_scatter_with_mask,
                      A_log, dt_bias, q, k, v, a, b, **kwargs)

    patch_attribute(TritonGDNKernel, "target_verify", target_verify)
