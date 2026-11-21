"""Verify GDN branches with the same recurrent arithmetic as unpacked decode.

SGLang's Triton verify kernel walks every candidate in one launch. Given a parent
table it reloads each candidate's state from the snapshot buffer, as stored, and then
matches one unpacked decode per candidate bit for bit. Without the table (a chain) it
carries the state between steps unrounded, which differs when the state is BF16. So a
chain gets a linear parent table.

`replay` is the per-candidate reference: one decode call for each step.

With AX_SSM_POSITIONS the snapshot buffer holds fewer positions per request than a
tree has candidates. A tree round verifies one request, and snapshots live only until
that round commits, so the round uses the whole buffer as one long row (`one_row`).
"""

from .compat import patch_attribute, patch_item


# Value columns per program in SGLang's recurrent GDN kernel under the parity profile.
# On SM120 SGLang uses 32 (192 one-warp programs for 48 heads); 8 gives four times
# the programs. Tiles of 8, 16 and 32 gave the same bits in FP32 and BF16 state, so
# AR outputs do not change; 4 changed bits.
PARITY_VALUE_TILE = 8


def install_value_tile():
    """Give GDN decode and verification the same narrower value tile on SM12x.

    Both call SGLang's `_select_recurrent_launch_config`, so AR decode and every
    verification window keep one tile shape. Other GPUs, KDA and other head sizes
    keep SGLang's choice.
    """
    from sglang.kernels.ops.attention.linear import fused_sigmoid_gating_recurrent as kernels

    native = kernels._select_recurrent_launch_config
    sm12 = []

    def select(n, h, hv, k, v, is_kda, target_verify=False, *, cache_steps=0):
        if not sm12:
            import torch

            sm12.append(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12)
        if sm12[0] and not is_kda and k == 128 and v == 128:
            return PARITY_VALUE_TILE, 1
        return native(n, h, hv, k, v, is_kda, target_verify, cache_steps=cache_steps)

    patch_item(vars(kernels), "_select_recurrent_launch_config", select)


def one_row(snapshots, layers=0):
    """View [(layers,) rows, positions, ...] snapshots as one row holding every position."""
    return snapshots.reshape(*snapshots.shape[:layers], 1, -1, *snapshots.shape[layers + 2:])


def verify(native, A_log, dt_bias, q, k, v, a, b, *, query_start_loc, cache_steps,
           intermediate_states_buffer, intermediate_state_indices, retrieve_parent_token=None, **kwargs):
    import torch

    batch_size = query_start_loc.numel() - 1
    if batch_size <= 0 or cache_steps <= 0 or q.shape[:2] != (1, batch_size * cache_steps):
        raise ValueError("AX GDN recurrent verify requires a uniform verification window")
    if intermediate_states_buffer.shape[1] < cache_steps:
        rows, positions = intermediate_states_buffer.shape[:2]
        if batch_size != 1 or rows * positions < cache_steps:
            raise ValueError(f"AX GDN recurrent verify needs {cache_steps} snapshot positions for each of "
                             f"{batch_size} requests; the buffer has {rows} rows of {positions}")
        intermediate_states_buffer = one_row(intermediate_states_buffer)
        intermediate_state_indices = torch.zeros(1, dtype=torch.int32, device=q.device)
    if retrieve_parent_token is None:
        retrieve_parent_token = (torch.arange(cache_steps, dtype=torch.int32, device=q.device) - 1
                                 ).expand(batch_size, cache_steps).contiguous()
    elif retrieve_parent_token.shape != (batch_size, cache_steps):
        raise ValueError("AX GDN recurrent verify requires one local parent index per candidate")
    return native(A_log, dt_bias, q, k, v, a, b, query_start_loc=query_start_loc, cache_steps=cache_steps,
                  intermediate_states_buffer=intermediate_states_buffer,
                  intermediate_state_indices=intermediate_state_indices,
                  retrieve_parent_token=retrieve_parent_token, **kwargs)


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
    from functools import partial

    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    native = TritonGDNKernel.target_verify

    def target_verify(self, A_log, dt_bias, q, k, v, a, b, **kwargs):
        return verify(partial(native, self), A_log, dt_bias, q, k, v, a, b, **kwargs)

    patch_attribute(TritonGDNKernel, "target_verify", target_verify)

    # The commit after a one-request round reads the accepted snapshot through the same one-row view.
    import dataclasses

    from sglang.srt.layers.attention import hybrid_linear_attn_backend as backend

    scatter = backend.scatter_mamba_states_after_mtp_verify

    def scatter_accepted(mamba_caches, state_indices, last_correct_step_indices, *args, **kwargs):
        snapshots = mamba_caches.intermediate_ssm
        if last_correct_step_indices.shape[0] == 1 and snapshots is not None and kwargs.get("src_indices_raw") is None:
            mamba_caches = dataclasses.replace(mamba_caches, intermediate_ssm=one_row(snapshots, layers=1))
        return scatter(mamba_caches, state_indices, last_correct_step_indices, *args, **kwargs)

    patch_item(vars(backend), "scatter_mamba_states_after_mtp_verify", scatter_accepted)
