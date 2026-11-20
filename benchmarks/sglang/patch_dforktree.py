"""Fill the gap in the DForkTree v0.1.0 release so its SGLang plugin can run (Track L, L0(d)).

    python benchmarks/sglang/patch_dforktree.py /root/dft-site/dforktree --release /root/DForkTree/dforktree

1. `tree_ops.ngram_path_fn` in the release (commit eeed8b6) computes the PMI table and then returns a
name `fn` that is never defined, so the first request fails with NameError. This script inserts the
missing closure, reconstructed from the paper (arXiv 2611.12356) and the README defaults, not from
the authors' code: per beam entry, the drafter's log-probabilities of the next position plus
lambda * PMI(child | last two tokens of the branch), lambda = lam_a while the branch is still the
top-1 path and lam_p after a fork, renormalised together with the mass outside the top-M list.
Numbers measured with it are "plugin + our reconstruction of the scoring closure", and may differ
from the authors' build.

2. `fc1_branch64.verify_branch` reads `batch.seq_lens_cpu`, which SGLang 0.5.22 leaves empty when the
scheduler runs without the host-side length mirror; the stock DFlash worker falls back to another
source there. The patch reads the prefix length from the GPU tensor instead (one small device-to-host
copy per round, which can only make the tree lanes slower, never faster) and restores the field.

3. Optional overhead fixes (benchmarks/sglang/dft_fast.py, copied in as ax_fast.py), active only with
AX_FAST=1, so the same install measures the release behaviour (unset) and the fixed one.

4. Target sizes other than the 9B in the fused Gated-DeltaNet verifier (AX_GDN_LAYERS,
AX_GDN_HEADS; defaults are the release's 24 x 32, Qwen3.8-27B is 48 x 48), including an int64
layer offset in the state-commit kernel: with the release's int32 `l * OS0`, the 27B's 9.7 GB
intermediate SSM buffer wraps for layers >= 43 and the kernel wrote fp32 states into the KV cache
(NaN keys in a full-attention layer a few requests in, then token-0 output; benchmarks/sglang/test_tree_gdn.py
reproduces it). Diagnostics with AX_GDN_DEBUG=1 only (graph-safe finite flags per GDN and
attention layer, printed per round; never set in measured runs).

5. DFlash2 lattice trees (benchmarks/sglang/dft_lattice.py as ax_lattice.py, active only with AX_LATTICE=1).

Every run first restores the four patched files from the pristine release (checked against the
SHA-256 of the v0.1.0 files, commit eeed8b6), applies all patches in order and then checks that every
patched snippet is present, so a partly patched or foreign file can never pass as "patched".
All four files are patched and checked in memory before any is written, so a failure leaves the
site unchanged. --release is required.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

OLD = "    T = T.float()\n    return fn\n"
NEW = '''    T = T.float()

    def fn(s, cand, pc, ppc, al):
        # reconstruction (ax-engine-sglang): rows = beam entries, columns = top-M children of position s
        if order == 2:
            pmi = T[s - 1, cand]
        elif order == 3:
            pmi = T[s - 1, pc, cand]
        else:
            pmi = T[s - 1, ppc, pc, cand]
        lam = torch.where(al, lam_a, lam_p).to(pmi.dtype)[:, None]
        score = lp[s].float()[None, :] + lam * pmi
        norm = torch.logsumexp(torch.cat((score, rest[s].float().expand(score.shape[0], 1)), dim=-1), -1, keepdim=True)
        return score - norm
    return fn
'''


OLD2 = "    plen = int(batch.seq_lens_cpu[0])\n"
NEW2 = """    seq_lens_cpu_was_none = batch.seq_lens_cpu is None   # ax-engine-sglang: host mirror may be absent
    if seq_lens_cpu_was_none:
        batch.seq_lens_cpu = prefix.detach().to('cpu', torch.int64)
    plen = int(batch.seq_lens_cpu[0])
"""
OLD2B = "    plan = w.target_worker.model_runner.kv_index_translator.plan(req_pool_indices=batch.req_pool_indices, seq_lens=prefix, seq_lens_cpu=batch.seq_lens_cpu,"
NEW2B = "    plan = w.target_worker.model_runner.kv_index_translator.plan(req_pool_indices=batch.req_pool_indices, seq_lens=prefix, seq_lens_cpu=batch.seq_lens_cpu if batch.seq_lens_cpu is not None else prefix.detach().to('cpu', torch.int64),"
OLD2C = "    if on_publish is not None:\n        on_publish(result.new_seq_lens)\n    return result\n"
NEW2C = "    if seq_lens_cpu_was_none:\n        batch.seq_lens_cpu = None\n" + OLD2C


RELEASE_SHA256 = {
    "tree_ops.py": "5ba3417fcfe251b45859ebc691d7e74f618bb67ebed41d218c1dfbe44c9647c5",
    "fc1_branch64.py": "19f065d4adb453e26a56c688a5428adef1dededf6b288cc55f5f3a53117d6410",
    "fc1_fused.py": "c34f6709a2309232b45acdd485cd3b5d9c8b0969ee75436ed7b3de56e58c2f08",
    "fc1_plugin.py": "c92683cddd1e03efc64e5c6f7c75f58407a704ae84377f928072c180ab1c090d",
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def apply(src, name, pairs):
    """Apply (old, new) replacements to pristine release text; every old must occur exactly once."""
    for old, new in pairs:
        if src.count(old) != 1:
            sys.exit(f"unexpected {name}: a release text this patch targets was not found exactly once")
        src = src.replace(old, new)
    return src


ANCHOR3 = "    plan = w.target_worker.model_runner.kv_index_translator.plan(req_pool_indices=batch.req_pool_indices, seq_lens=prefix, seq_lens_cpu="
NEW3A = ("    import ax_fast   # ax-engine-sglang overhead fixes, AX_FAST=1\n"
         "    _ax_mode = ax_fast.mode(batch)\n"
         "    _ax_plan_cpu = ax_fast.plan_lens_cpu(batch, prefix, _ax_mode)\n")
OLD3B = "seq_lens_cpu=batch.seq_lens_cpu if batch.seq_lens_cpu is not None else prefix.detach().to('cpu', torch.int64), write_virtual=cache"
NEW3B = "seq_lens_cpu=_ax_plan_cpu, write_virtual=cache"
OLD3C = """    if seq_lens_cpu_was_none:
        batch.seq_lens_cpu = prefix.detach().to('cpu', torch.int64)
    plen = int(batch.seq_lens_cpu[0])
    import fc1_plan
    fc1_plan.HOST_KV_LEN = plen + n
    mask = torch.cat((torch.ones((n, plen), device=dev, dtype=torch.bool), anc.bool()), dim=1).flatten()
"""
NEW3C = """    if _ax_mode == 'fast':
        import fc1_plan
        fc1_plan.HOST_KV_LEN = None
        mask = ax_fast.tree_mask(w, prefix, anc, n, _ax_plan_cpu)
    else:
        if seq_lens_cpu_was_none:
            batch.seq_lens_cpu = _ax_plan_cpu if _ax_mode == 'dedupe' else prefix.detach().to('cpu', torch.int64)
        plen = int(batch.seq_lens_cpu[0])
        import fc1_plan
        fc1_plan.HOST_KV_LEN = plen + n
        mask = torch.cat((torch.ones((n, plen), device=dev, dtype=torch.bool), anc.bool()), dim=1).flatten()
"""
OLD4 = """            if track is None:
                track = torch.full_like(last, -1)
            ends = torch.cat((last, track)).contiguous()
"""
NEW4 = """            ax_q = 8   # ax-engine-sglang: with AX_FAST=1 and no track endpoint, launch only the live half
            if track is None and __import__('os').environ.get('AX_FAST') == '1':
                ends = last.contiguous()
                ax_q = 4
            else:
                if track is None:
                    track = torch.full_like(last, -1)
                ends = torch.cat((last, track)).contiguous()
"""
OLD4B = "            snapshots[24 * 32, 4, 8]("
NEW4B = "            snapshots[24 * 32, 4, ax_q]("

# 4. Other target sizes (Qwen3.8-27B: 48 Gated-DeltaNet layers x 48 value heads; the release fixes
# the 9B's 24 x 32). AX_GDN_LAYERS / AX_GDN_HEADS, defaults = the release values.
OLD5A = "BS = int(os.environ.get('FC1_DRAFT_BLOCK', '24'))\n"
NEW5A = OLD5A + "GL = int(os.environ.get('AX_GDN_LAYERS', '24'))   # ax-engine-sglang: GDN layers of the target\nGH = int(os.environ.get('AX_GDN_HEADS', '32'))    # ax-engine-sglang: GDN value heads\nAX_DBG = os.environ.get('AX_GDN_DEBUG') == '1'     # ax-engine-sglang: diagnostics, see patch_dforktree.py\n"
OLD5B = "        self.keys = torch.empty(24, 32, NN, 128, device='cuda')"
NEW5B = "        self.keys = torch.empty(GL, GH, NN, 128, device='cuda')"
OLD5C = "        self.gains = torch.empty(24, 32, NN, device='cuda')"
NEW5C = ("        self.gains = torch.empty(GL, GH, NN, device='cuda')\n"
         "        if AX_DBG:   # ax-engine-sglang: flag buffers of the diagnostics, allocated with the context\n"
         "            self.ax_bad = torch.zeros(GL, 4, device='cuda')\n"
         "            self.ax_nodes = torch.zeros(GL, NN, device='cuda')\n"
         "            self.ax_amax = torch.zeros(GL, 3, device='cuda')\n"
         "            self.ax_att = torch.zeros(256, 2, device='cuda')\n"
         "            self.ax_dbg = 0\n"
         "            self.ax_kv_done = False")
# every index that multiplies a buffer stride is int64. For slot / dst this is defensive only: on
# the 27B dst * OS1 wraps from intermediate slot 86, a buffer of about 415 GB, so it cannot occur
# on one GPU today; the cast is free and removes the class. Not patched, and limited to small
# pools: fc1_conv / tree_kernels index per-layer buffers with int32 slot offsets (conv window
# slot * I0 wraps near slot 2,180, SSM slot * stride near 2,730 on the 27B); the tree lanes run
# one request, so slot is 0 or 1. Revisit before batching (c = 8 / 32).
OLD5K = "    slot = tl.load(Slot)\n    dst = tl.load(InterSlot)\n"
NEW5K = "    slot = tl.load(Slot).to(tl.int64)\n    dst = tl.load(InterSlot).to(tl.int64)\n"
OLD5D = "OS5: tl.constexpr):\n    lh = tl.program_id(0)\n    l = lh // 32\n    h = lh % 32\n"
# the layer offset l * OS0 is int32 in the release: for the 27B intermediate SSM buffer
# (48, 2, 32, 48, 128, 128) fp32, OS0 = 50,331,648 and l >= 43 wraps negative, so the snapshot
# kernel wrote fp32 states before the buffer (into the KV cache: NaN keys, then token-0 output).
# The 9B (24 x 32 heads) stays below 2^31. int64 layer index fixes it for any size.
NEW5D = ("OS5: tl.constexpr, GH: tl.constexpr):\n    lh = tl.program_id(0)\n"
         "    l = (lh // GH).to(tl.int64)   # ax-engine-sglang: l * OS0 overflows int32 for the 27B\n    h = lh % GH\n")
OLD5E = "        assert li < 24 and q.shape[0] == 1"
NEW5E = "        assert li < GL and q.shape[0] == 1"
OLD5F = "            snapshots[24 * 32, 4, ax_q]("
NEW5F = "            snapshots[GL * GH, 4, ax_q]("
OLD5G = "*state.intermediate_ssm.stride(), num_warps=4)"
NEW5G = "*state.intermediate_ssm.stride(), GH=GH, num_warps=4)"
# diagnostics only (AX_GDN_DEBUG=1, a boolean): per GDN layer, finite flags of the fused verifier's
# inputs, outputs and committed state, and per full-attention layer of the tree verify, finite
# flags of q/k/v and the output; all kept on the GPU inside the CUDA graph and printed from the
# commit on rounds 1-2 and on every round with a flag set, plus one dump of the NaN positions in
# the KV cache. It syncs every round: never set in speed or correctness runs (run_tree_plugin.sh
# pins it to 0 unless GDN_DEBUG=1 and then prints a NOTE; the lane metadata records it and
# token_identity.is_diagnostic excludes such a pass from speed statistics).
OLD5H = "        return out.unsqueeze(0)\n\n    def fused_commit"
NEW5H = ("        if AX_DBG:   # graph-safe: flags stay on the GPU\n"
         "            CTX.ax_bad[li, 0] = (~torch.isfinite(out)).any().float()\n"
         "            CTX.ax_bad[li, 1] = (~torch.isfinite(y)).any().float()\n"
         "            CTX.ax_bad[li, 2] = (~torch.isfinite(pool.index_select(0, kw['cache_indices'][:1].long().clamp(min=0)))).any().float()\n"
         "            CTX.ax_bad[li, 3] = (~torch.isfinite(a)).any().float() + 2 * (~torch.isfinite(b)).any().float()\n"
         "            CTX.ax_nodes[li] = (~torch.isfinite(out.reshape(NN, -1))).any(-1).float()\n"
         "            CTX.ax_amax[li, 0] = y.float().abs().amax()\n"
         "            CTX.ax_amax[li, 1] = out.float().abs().nan_to_num(0.0, 0.0, 0.0).amax()\n"
         "            CTX.ax_amax[li, 2] = (y.float() * y.float()).reshape(NN, -1)[:, :q.shape[2] * q.shape[3] * 2].sum(-1).min()\n"
         "        return out.unsqueeze(0)\n\n    def fused_commit")
OLD5J = "    TritonGDNKernel.target_verify = fused\n"
NEW5J = ("    if AX_DBG:   # diagnostics: finite flags of every full-attention layer in tree verify\n"
         "        from sglang.srt.layers.radix_attention import RadixAttention\n"
         "        ax_fwd = RadixAttention.forward\n"
         "        def ax_dbg_fwd(self, q, k, v, forward_batch, *args, **kwargs):\n"
         "            o = ax_fwd(self, q, k, v, forward_batch, *args, **kwargs)\n"
         "            if (q.shape[0] == NN and forward_batch.forward_mode.is_target_verify() and k is not None and v is not None\n"
         "                    and CTX is not None):\n"
         "                CTX.ax_att[self.layer_id, 0] = (~torch.isfinite(q)).any().float() + 2 * (~torch.isfinite(k)).any().float() + 4 * (~torch.isfinite(v)).any().float()\n"
         "                CTX.ax_att[self.layer_id, 1] = (~torch.isfinite(o)).any().float()\n"
         "            return o\n"
         "        RadixAttention.forward = ax_dbg_fwd\n"
         + OLD5J)
OLD5I = "        return commit(target_worker, batch, accept_lens, accept_index, draft_token_num, prepared_step_indices)\n"
NEW5I = ("        ax_r = commit(target_worker, batch, accept_lens, accept_index, draft_token_num, prepared_step_indices)\n"
         "        if AX_DBG and CTX is not None and draft_token_num == NN:\n"
         "            CTX.ax_dbg += 1\n"
         "            st = target_worker.model_runner.req_to_token_pool.get_speculative_mamba2_params_all_layers()\n"
         "            sl = target_worker.model_runner.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)\n"
         "            tbad = (~torch.isfinite(st.temporal[:, sl])).flatten(1).any(-1)\n"
         "            e = CTX.ax_bad.cpu()\n"
         "            if e.any() or tbad.any() or CTX.ax_att.any() or CTX.ax_dbg <= 2:\n"
         "                bad = lambda col: [i for i in range(e.shape[0]) if e[i, col] > 0][:4]\n"
         "                l0 = bad(0)[0] if bad(0) else 0\n"
         "                am = CTX.ax_amax.cpu()\n"
         "                print(f'[ax_gdn] round {CTX.ax_dbg} accept {accept_lens.tolist()} out_bad {bad(0)} y_bad {bad(1)} state_in_bad {bad(2)} ab_bad {bad(3)} '\n"
         "                      f'temporal_bad {tbad.nonzero().flatten().tolist()[:4]} layer {l0} bad_nodes {CTX.ax_nodes[l0].nonzero().flatten().tolist()[:12]} '\n"
         "                      f'y_amax {[round(float(x), 2) for x in am[max(l0 - 2, 0):l0 + 1, 0]]} out_amax {[round(float(x), 2) for x in am[max(l0 - 2, 0):l0 + 1, 1]]} '\n"
         "                      f'min_qk_sq {[float(x) for x in am[max(l0 - 2, 0):l0 + 1, 2]]}', flush=True)\n"
         "                at = CTX.ax_att.cpu()\n"
         "                if at.any() and not CTX.ax_kv_done:\n"
         "                    CTX.ax_kv_done = True\n"
         "                    mr = target_worker.model_runner\n"
         "                    lay = int(at.amax(-1).nonzero()[0])   # first layer with any flag (input or output)\n"
         "                    kb, vb = mr.token_to_kv_pool.get_key_buffer(lay), mr.token_to_kv_pool.get_value_buffer(lay)\n"
         "                    rp = int(batch.req_pool_indices[0]); L = int(batch.seq_lens[0])\n"
         "                    locs = mr.req_to_token_pool.req_to_token[rp, :L].long()   # committed prefix only\n"
         "                    kbad = (~torch.isfinite(kb[locs].float())).flatten(1).any(-1) | (~torch.isfinite(vb[locs].float())).flatten(1).any(-1)\n"
         "                    pos = kbad.nonzero().flatten().tolist()\n"
         "                    print(f'[ax_kv] layer {lay} seq_len {L} (after commit) req_pool {rp} nan_positions {pos[:20]} n_nan {len(pos)} '\n"
         "                          f'locs_at_nan {locs[kbad][:8].tolist()} locs_tail {locs[max(L - 4, 0):].tolist()} kv_buf {tuple(kb.shape)}', flush=True)\n"
         "                print(f'[ax_att] round {CTX.ax_dbg} qkv_bad(layer:1q2k4v) {[(i, int(at[i, 0])) for i in range(256) if at[i, 0] > 0][:6]} out_bad {[i for i in range(256) if at[i, 1] > 0][:6]}', flush=True)\n"
         "        return ax_r\n")

# 5. DFlash2 lattice trees (benchmarks/sglang/dft_lattice.py as ax_lattice.py, active with AX_LATTICE=1): the
# tree is scored with DFlash2's learned transition lattice; the selector sampler has no fc1_logits.
OLD6A = "    logits = w._draft_sampler.fc1_logits.view(bs, w.block_size - 1, -1)[0]\n"
NEW6A = ("    import ax_lattice   # ax-engine-sglang: DFlash2 lattice trees, AX_LATTICE=1\n"
         "    logits = None if ax_lattice.enabled() else w._draft_sampler.fc1_logits.view(bs, w.block_size - 1, -1)[0]\n")
OLD6B = "        w.fc1_builder = Builder(logits, compile_build=os.environ.get('FC1_COMPILE_BUILD') == '1')"
NEW6B = ("        w.fc1_builder = (ax_lattice.LatticeBuilder(w._draft_sampler, compile_build=os.environ.get('FC1_COMPILE_BUILD') == '1')\n"
         "                         if ax_lattice.enabled() else Builder(logits, compile_build=os.environ.get('FC1_COMPILE_BUILD') == '1'))")
OLD6C = "    from sglang.srt.speculative.dflash_info import DFlashVerifyInput\n"
NEW6C = "    import ax_lattice   # ax-engine-sglang: wraps the DFlash2 selector sampler when AX_LATTICE=1\n    ax_lattice.install()\n" + OLD6C


PATCHES = {
    "tree_ops.py": [(OLD, NEW)],
    "fc1_branch64.py": [(OLD2B, NEW2B), (OLD2, NEW2), (OLD2C, NEW2C),
                        (ANCHOR3, NEW3A + ANCHOR3), (OLD3B, NEW3B), (OLD3C, NEW3C), (OLD6A, NEW6A), (OLD6B, NEW6B)],
    "fc1_fused.py": [(OLD4, NEW4), (OLD4B, NEW4B), (OLD5A, NEW5A), (OLD5B, NEW5B), (OLD5C, NEW5C), (OLD5D, NEW5D),
                     (OLD5E, NEW5E), (OLD5F, NEW5F), (OLD5G, NEW5G), (OLD5H, NEW5H), (OLD5I, NEW5I), (OLD5J, NEW5J), (OLD5K, NEW5K)],
    "fc1_plugin.py": [(OLD6C, NEW6C)],
}


# text that must be in the final file (later patches rewrite parts of earlier ones)
FINAL = {
    "tree_ops.py": ["def fn(s, cand, pc, ppc, al)", "    return fn\n"],
    "fc1_branch64.py": ["seq_lens_cpu_was_none = batch.seq_lens_cpu is None", "import ax_fast",
                        "seq_lens_cpu=_ax_plan_cpu, write_virtual=cache", "mask = ax_fast.tree_mask(",
                        "    if seq_lens_cpu_was_none:\n        batch.seq_lens_cpu = None\n    if on_publish"],
    "fc1_fused.py": ["ax_q = 8", "            snapshots[GL * GH, 4, ax_q](", "GH=GH, num_warps=4)", "l = (lh // GH).to(tl.int64)",
                     "dst = tl.load(InterSlot).to(tl.int64)"],
    "fc1_plugin.py": ["ax_lattice.install()"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", help="the plugin package directory that SGLang imports (overlay)")
    ap.add_argument("--release", required=True, help="pristine dforktree package of the v0.1.0 release")
    args = ap.parse_args()
    root = Path(args.site)
    outs = {}
    for name, pairs in PATCHES.items():               # everything in memory first
        src_path = Path(args.release) / name
        src = src_path.read_text()
        if sha(src) != RELEASE_SHA256[name]:
            sys.exit(f"{src_path} does not match the v0.1.0 release (sha256 {sha(src)[:12]}); nothing written")
        out = apply(src, name, pairs)
        missing = [m[:60] for m in FINAL[name] if m not in out]
        if missing:
            sys.exit(f"{name}: patched snippets missing after apply: {missing}; nothing written")
        outs[name] = out
    for name, out in outs.items():
        tmp = root / (name + ".axtmp")
        tmp.write_text(out)
        tmp.replace(root / name)
        print(f"patched {root / name} (sha256 {sha(out)[:12]})")
    shutil.copyfile(Path(__file__).resolve().parents[2] / "ax-engine-sglang/ax_engine_sglang/ax_fast.py", root / "ax_fast.py")
    shutil.copyfile(Path(__file__).resolve().parents[2] / "ax-engine-sglang/ax_engine_sglang/ax_lattice.py", root / "ax_lattice.py")
    print("copied ax_fast.py, ax_lattice.py")


if __name__ == "__main__":
    main()
