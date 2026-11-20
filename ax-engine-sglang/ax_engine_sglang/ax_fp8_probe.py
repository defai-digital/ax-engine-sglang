# SPDX-License-Identifier: Apache-2.0
"""FP8 KV write probe for two attention backends on one KV pool (AX_FP8_PROBE=1).

docs/decisions/2026-11-16-dual-attention-backend.md, section 5 item 2: with an FP8 KV cache the
triton backend writes K/V through the pool's set_kv_buffer (triton_backend.py:1735-1743) and
trtllm_mha through fused_fp8_qkv_kv_cache directly into the pool buffers
(trtllm_mha_backend.py:1086-1098). Run once at start-up, before any request: for a few attention
layers, the same random K/V are written by each path into slot 1 and the stored bytes are compared.
Not slot 0: that is the pool's padding slot, which SGLang's store kernel skips (padded tokens write
there), while trtllm's fused kernel writes it; no request holds slot 1 before the first request. Any difference is printed and
stops the server, since the tree rounds (triton) read KV the chain rounds (trtllm_mha) wrote and
the other way round. Each write lands on poisoned bytes (0xFF), so a writer that writes nothing
cannot pass, and the slot is zeroed afterwards. Requested on a non-FP8 pool or with no attention layer
found, the probe fails instead of passing. Read paths are covered end to end by the teacher-forced
fresh-prefill check (benchmarks/sglang/sglang_teacher_check.py), not here.
"""
from __future__ import annotations

import torch


def _layers(model, n):
    out = []
    for m in model.modules():
        if type(m).__name__ == "RadixAttention":
            out.append(m)
    step = max(1, len(out) // n)
    return out[::step][:n]


def run(mr, n_layers=4, trials=8):
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    pool = mr.token_to_kv_pool
    full = getattr(pool, "full_kv_pool", pool)
    if getattr(full, "dtype", None) != torch.float8_e4m3fn:
        raise RuntimeError(f"ax-engine: AX_FP8_PROBE=1 but the KV cache is {getattr(full, 'dtype', None)}, not FP8")
    from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import fused_fp8_qkv_kv_cache
    ones = torch.ones(1, dtype=torch.float32, device=mr.device)
    SLOT = 1
    loc = torch.full((1,), SLOT, dtype=torch.int64, device=mr.device)
    bad, checked = [], 0
    g = torch.Generator(device=mr.device).manual_seed(0)
    layers = _layers(mr.model, n_layers)
    if not layers:
        raise RuntimeError("ax-engine: AX_FP8_PROBE found no RadixAttention layer")
    for layer in layers:
        kc, vc = pool.get_kv_buffer(layer.layer_id)
        H, D = layer.tp_k_head_num, layer.head_dim
        Dv = getattr(layer, "v_head_dim", None) or D
        for t in range(trials):
            scale = 0.5 * (t + 1)
            k = (torch.randn(1, H, D, generator=g, device=mr.device) * scale).to(torch.bfloat16)
            v = (torch.randn(1, layer.tp_v_head_num, Dv, generator=g, device=mr.device) * scale).to(torch.bfloat16)
            kc[SLOT].view(torch.uint8).fill_(0xFF)
            vc[SLOT].view(torch.uint8).fill_(0xFF)
            # triton's path (triton_backend.py:1721-1743)
            if layer.k_scale is None:
                pool.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k.clone(), v.clone())
            else:
                pool.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k.clone(), v.clone(), layer.k_scale, layer.v_scale)
            torch.cuda.synchronize()   # the pool may write on its own stream
            a_k = kc[SLOT].view(torch.uint8).clone()
            a_v = vc[SLOT].view(torch.uint8).clone()
            kc[SLOT].view(torch.uint8).fill_(0xFF)
            vc[SLOT].view(torch.uint8).fill_(0xFF)
            # trtllm_mha's path (trtllm_mha_backend.py:1086-1098)
            fused_fp8_qkv_kv_cache(q=None, k=k, v=v, k_cache=kc, v_cache=vc, cache_loc=loc,
                                   k_scale=layer.k_scale if layer.k_scale is not None else ones,
                                   v_scale=layer.v_scale if layer.v_scale is not None else ones)
            torch.cuda.synchronize()
            b_k = kc[SLOT].view(torch.uint8).clone()
            b_v = vc[SLOT].view(torch.uint8).clone()
            checked += 1
            dk = int((a_k != b_k).sum())
            dv = int((a_v != b_v).sum())
            if dk or dv or bool((a_k == 0xFF).all()) or bool((b_k == 0xFF).all()):
                bad.append((layer.layer_id, t, dk, dv, a_k.numel()))
                if len(bad) == 1:
                    ks = layer.k_scale
                    ref = (k.float() / (float(ks) if ks is not None else 1.0)).to(torch.float8_e4m3fn).view(torch.uint8).flatten()
                    print(f"[ax_fp8_probe] detail: kc {tuple(kc.shape)} {kc.dtype} stride {kc.stride()}, "
                          f"pool {type(full).__name__} use_hnd={getattr(full, 'use_hnd', None)} page={getattr(full, 'page_size', None)}, "
                          f"k_scale={None if ks is None else float(ks)}, k {tuple(k.shape)}", flush=True)
                    print(f"[ax_fp8_probe] detail: triton {a_k.flatten()[:12].tolist()} trtllm {b_k.flatten()[:12].tolist()} "
                          f"manual {ref[:12].tolist()}", flush=True)
        kc[SLOT].view(torch.uint8).zero_()
        vc[SLOT].view(torch.uint8).zero_()
    torch.cuda.synchronize()
    if bad:
        for row in bad[:8]:
            print(f"[ax_fp8_probe] layer {row[0]} trial {row[1]}: {row[2]} K and {row[3]} V bytes of {row[4]} differ", flush=True)
        raise RuntimeError(f"ax-engine: FP8 KV writes differ between triton and trtllm_mha ({len(bad)} of {checked} writes)")
    print(f"[ax_fp8_probe] {checked} FP8 K/V writes byte-identical between the triton and trtllm_mha paths", flush=True)
