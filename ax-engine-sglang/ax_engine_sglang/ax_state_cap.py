"""AX_SSM_POSITIONS=<p>: size the GDN intermediate SSM state cache for p draft positions.

SGLang keeps, for speculative verify on hybrid linear-attention models, one intermediate SSM state
per slot and per draft position ([layers, slots + 1, speculative_num_draft_tokens, H, V, K]). With
the 16-node tree that is 16 positions per slot: about 146 GB for Qwen3.8-27B at 64 slots, which does
not fit next to the KV cache on one B200. Only chain rounds need the cache: a tree round commits its
accepted state straight into the slot's persistent state (fc1_fused, when this cap is on), and the
chain runs at width <= p. With p = 8 the cache is the size of stock chain 8.

The conv windows (small) keep the full draft count, the tree's conv rollback still uses them.

The two SGLang functions that size the cache, MambaPool.__init__ (allocation) and
KVCacheConfigurator._handle_max_mamba_cache (memory budget), are rebuilt from their installed source
with exact-match edits; start-up fails with the function's name if an edit does not match.
"""
from __future__ import annotations

from functools import partial
import sys
from ax_engine_sglang.config import get_config
from ax_engine_sglang.compat import installation, patch_attribute, patch_item, rebuild_function

_rebuild = partial(rebuild_function, feature="AX_SSM_POSITIONS")

POS = get_config().ssm_positions
_installed = False


# Conv share of SGLang's per-slot mamba state (stage_per_req prices SSM + conv together). Qwen3.8-27B:
# conv 0.18 GB vs SSM 9.14 GB at 64 slots, about 2%. The budget uses 5% (an upper bound) for the
# positions whose conv windows stay allocated above the cap.
CONV_SHARE = get_config().ssm_conv_share


def positions(n):
    """Draft positions to reserve in the intermediate SSM cache for a draft count n."""
    if n is None or not POS:
        return n
    return min(int(n), POS)


def budget_positions(n):
    """Position count the memory budget charges per slot: the capped SSM + conv positions, plus the
    conv share of the positions above the cap (their conv windows are still allocated)."""
    if n is None or not POS or int(n) <= POS:
        return n
    return POS + (int(n) - POS) * CONV_SHARE


_dummy_runs = [0]   # depth of BaseRunner._dummy_run (FlashInfer autotune / PP DeepGEMM warmup forwards)


def mark_dummy_run(native):
    """Count SGLang's warmup dummy forwards, the only callers allowed past guard_verify's check."""
    def _dummy_run(self, *args, **kwargs):
        _dummy_runs[0] += 1
        try:
            return native(self, *args, **kwargs)
        finally:
            _dummy_runs[0] -= 1
    return _dummy_run


def guard_verify(native):
    """Keep SGLang's GDN verify kernel inside the capped intermediate SSM cache.

    The kernel writes one snapshot per draft token at row * pitch + step, with the pitch taken from
    the buffer (POS here). Served rounds never exceed it (check_widths; tree rounds take the fused
    hooks), but the FlashInfer autotune dummy forward (BaseRunner._dummy_run, before any graph
    capture sets the round kind) verifies max_running_requests rows at the tree's full draft count:
    rows overlap and the last GDN layer writes past the end of the cache (illegal address). The dummy
    only needs the GEMM shapes, so its snapshot rows are set to -1, which the kernel skips (no
    snapshot reads or writes). Anywhere else the overflow is a bug and fails loudly."""
    def target_verify(self, *args, intermediate_states_buffer=None, intermediate_state_indices=None,
                      cache_steps=0, **kwargs):
        if (intermediate_states_buffer is not None and intermediate_state_indices is not None
                and cache_steps > intermediate_states_buffer.shape[1]):
            if not _dummy_runs[0]:
                raise RuntimeError(
                    f"ax-engine: GDN verify of {cache_steps} draft tokens per request exceeds the "
                    f"{intermediate_states_buffer.shape[1]} positions of the capped SSM cache (AX_SSM_POSITIONS)")
            intermediate_state_indices = intermediate_state_indices.new_full(intermediate_state_indices.shape, -1)
        return native(self, *args, intermediate_states_buffer=intermediate_states_buffer,
                      intermediate_state_indices=intermediate_state_indices, cache_steps=cache_steps, **kwargs)
    return target_verify


def install():
    if _installed or not POS:
        return
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.mem_cache.memory_pool import MambaPool

    # allocation: only the intermediate SSM cache (the line before temporal_state_shape[0])
    init = _rebuild(MambaPool, "__init__", [(
        "speculative_num_draft_tokens,\n                        temporal_state_shape[0],",
        "_ax_ssm_positions(speculative_num_draft_tokens),\n                        temporal_state_shape[0],",
        1)])
    # budget: the intermediate reservation in every branch (explicit cache size, radix off, ratio fit)
    budget = _rebuild(KVCacheConfigurator, "_handle_max_mamba_cache", [
        ("* get_spec().speculative_num_draft_tokens", "* _ax_budget_positions(get_spec().speculative_num_draft_tokens)", 2),
        ("D = get_spec().speculative_num_draft_tokens", "D = _ax_budget_positions(get_spec().speculative_num_draft_tokens)", 1),
    ])
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.srt.model_executor.runner.base_runner import BaseRunner
    # Compile both replacements first. A failed anchor must leave allocation and budget paired.
    with installation():
        for fn in (init, budget):
            namespace = sys.modules[fn.__module__].__dict__
            patch_item(namespace, "_ax_ssm_positions", positions)
            patch_item(namespace, "_ax_budget_positions", budget_positions)
        patch_attribute(MambaPool, "__init__", init)
        patch_attribute(KVCacheConfigurator, "_handle_max_mamba_cache", budget)
        patch_attribute(TritonGDNKernel, "target_verify", guard_verify(TritonGDNKernel.target_verify))
        patch_attribute(BaseRunner, "_dummy_run", mark_dummy_run(BaseRunner._dummy_run))
        patch_attribute(sys.modules[__name__], "_installed", True)
    print(f"[ax_state_cap] intermediate SSM cache capped at {POS} draft positions per slot", flush=True)


def check_widths(widths, nn):
    """Start-up guard: every chain width must fit the capped cache (a chain round at the tree's width
    runs SGLang's own verify, which writes one position per draft token); tree rounds bypass it."""
    if not POS:
        return
    bad = [w for w in widths if w and int(w) > POS]
    if bad:
        raise RuntimeError(f"ax-engine: AX_SSM_POSITIONS={POS} but chain widths {bad} need more positions")
