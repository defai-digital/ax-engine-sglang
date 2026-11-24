"""Compare staged graph replay or direct KV writes with the SGLang eager writer."""

from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_dflash_kv_graph import _wrap_append


@torch.inference_mode()
def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--direct", action="store_true", help="test the direct BF16 cache writer")
    direct = parser.parse_args().direct
    if not torch.cuda.is_available():
        raise SystemExit("This check requires CUDA")

    from sglang.kernels.ops.speculative.fused_kv_materialize import FusedKVMaterializeHelper
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    torch.manual_seed(42)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    head_dim = 128
    layer_count = 2
    angles = torch.arange(256, device=device)[:, None] * torch.linspace(0.01, 0.9, head_dim // 2, device=device)
    rope = NS(cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=1).to(dtype), rotary_dim=head_dim)
    layers = [
        NS(self_attn=NS(
            num_kv_heads=1, head_dim=head_dim, q_size=head_dim, kv_size=head_dim,
            rotary_emb=rope, qkv_proj=torch.nn.Linear(head_dim, 3 * head_dim, device=device, dtype=dtype),
            k_norm=NS(weight=torch.ones(head_dim, device=device, dtype=dtype), variance_epsilon=1e-6),
            attn=NS(layer_id=layer_id, k_scale=None, v_scale=None),
        ))
        for layer_id in range(layer_count)
    ]
    fc = torch.nn.Linear(4 * head_dim, head_dim, bias=False, device=device, dtype=dtype)
    norm = torch.nn.RMSNorm(head_dim, device=device, dtype=dtype)
    draft = NS(layers=layers, fc=fc, hidden_norm=norm, project_target_hidden=lambda hidden: norm(fc(hidden)))

    def make_worker():
        pool = MHATokenToKVPool(
            size=128, page_size=1, dtype=dtype, head_num=1, head_dim=head_dim,
            layer_num=layer_count, device=device, enable_memory_saver=False, enable_alt_stream=False,
        )
        for cache in (*pool.k_buffer, *pool.v_buffer):
            cache.fill_(-2)
        helper = FusedKVMaterializeHelper(
            layers, rope, num_kv_heads=1, head_dim=head_dim, device=device, max_position_hint=256,
        )
        worker = NS(draft_model=draft, draft_model_runner=NS(token_to_kv_pool=pool),
                    model_runner=NS(device=device), draft_parallel_layout=NS(tp_group=None),
                    block_size=8, lilicorr=None, _use_fused_kv_materialize=True, _fused_kv_helper=helper)
        worker._append_target_hidden_fused = MethodType(DFlashWorkerV2._append_target_hidden_fused, worker)
        return worker, (*pool.k_buffer, *pool.v_buffer)

    eager, eager_cache = make_worker()
    captured, captured_cache = make_worker()
    append = DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("An eligible CUDA fixture input used the eager path")

    if direct:
        from ax_engine_sglang.ax_dflash_kv_direct import install
        install()
        captured._append_target_hidden_fused = MethodType(DFlashWorkerV2._append_target_hidden_fused, captured)
        candidate_append = append
    else:
        candidate_append = _wrap_append(unexpected_fallback, torch, MHATokenToKVPool, lambda _: nullcontext())
    for batch_size in (1, 8, 1, 8):
        # Reuse physical slots under new owners, even when the batch shrinks.
        locations = torch.arange(1, batch_size * 8 + 1, device=device).view(batch_size, 8)
        for committed in (0, 5, 8):
            lengths = (torch.arange(batch_size, device=device, dtype=torch.int32) + committed).clamp(max=8)
            if committed == 0:
                lengths.zero_()
            slots = locations.roll(committed).masked_fill(torch.arange(8, device=device) >= lengths[:, None], -1)
            positions = torch.randint(0, 256, (batch_size * 8,), device=device)
            hidden = torch.randn(batch_size * 8, fc.in_features, device=device, dtype=dtype)
            inputs = dict(target_hidden=hidden, positions=positions, cache_loc=slots.flatten(),
                          cache_loc_2d=slots, commit_lens=lengths)
            append(eager, **inputs)
            candidate_append(captured, **inputs)
            for got, expected in zip(captured_cache, eager_cache, strict=True):
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    if direct:
        assert captured._ax_direct_kv_batch_sizes == {1, 8}
        print("Direct DFlash KV writes match the SGLang eager cache across both batch sizes")
    else:
        print("DFlash KV graph matches the SGLang eager cache across both batch sizes")


if __name__ == "__main__":
    main()
