# Diagnostics used to locate the stock DFlash2 sampled-decoding bias (docs/decisions/2026-11-16-dual-attention-backend.md
# section 16). Load with PYTHONPATH=benchmarks/sglang/selector_diag on a stock DFlash2 server; never in measured lanes.
#   AX_DIAG_NO_SELECTOR_SAMPLING=1  selector sampling off (greedy drafts, plain sampled verify)
#   AX_DIAG_SELECTOR_CHECK=1        per-round consistency of draft tokens, candidate ids, q and the walk's uniforms
#   AX_DIAG_BRANCH=1                count which accept branch each verify round takes
#   AX_DIAG_REF_ACCEPT=1            replace the selector accept with a pure-torch rejection sampler
#   AX_DIAG_LOG_P=FILE              with AX_DIAG_REF_ACCEPT: log row-0 target probabilities, slot-0 q and u0 per round
#   AX_DIAG_EAGER_UNIFORMS=1        draw the walk's uniforms eagerly before each draft replay
#   AX_DIAG_FI_OLD_OFFSET=1         flashinfer get_seed_and_offset returns the old offset (the fix, see ax_rng.py)
import os
if any(os.environ.get(k) == "1" for k in ("AX_DIAG_NO_SELECTOR_SAMPLING", "AX_DIAG_SELECTOR_CHECK", "AX_DIAG_REF_ACCEPT", "AX_DIAG_BRANCH", "AX_DIAG_EAGER_UNIFORMS")):
    import importlib.abc, importlib.util, sys

    def _patch():
        import torch
        from sglang.srt.speculative import dflash_worker_v2 as m
        cls = m.DFlashWorkerV2
        if os.environ.get("AX_DIAG_NO_SELECTOR_SAMPLING") == "1":
            init0 = cls.__init__
            def __init__(self, *a, **k):
                init0(self, *a, **k)
                self._selector_sampling_enabled = False
                print("[ax-diag] selector sampling disabled", flush=True)
            cls.__init__ = __init__
        if os.environ.get("AX_DIAG_SELECTOR_CHECK") == "1":
            acc0 = cls._selector_sampling_accept
            st = {"rounds": 0, "pos": 0, "not_in_cand": 0, "q_zero": 0, "dup_rows": 0, "q_sum_bad": 0,
                  "u_repeat": 0, "q_onehot_pos": 0, "first_pos_q_top": 0.0, "greedy_mask_true": 0}
            last_u = [None]
            def _selector_sampling_accept(self, *, candidates, next_token_logits, candidate_ids, q_rows, sampling_info, draft_input):
                bs, block = candidates.shape
                g = block - 1
                d = candidates[:, 1:].long()                                    # (bs, g)
                hit = candidate_ids[:, :g] == d[:, :, None]                     # (bs, g, K)
                st["rounds"] += 1
                st["pos"] += bs * g
                st["not_in_cand"] += int((~hit.any(-1)).sum())
                qd = (q_rows[:, :g].float() * hit).sum(-1)
                st["q_zero"] += int((qd <= 0).sum())
                srt = candidate_ids[:, :g].sort(-1).values
                st["dup_rows"] += int((srt[..., 1:] == srt[..., :-1]).any(-1).sum())
                st["q_sum_bad"] += int(((q_rows[:, :g].float().sum(-1) - 1).abs() > 1e-3).sum())
                st["q_onehot_pos"] += int((q_rows[:, :g].float().max(-1).values > 0.999).sum())
                ds = getattr(self, "_draft_sampler", None)
                if ds is not None and hasattr(ds, "uniforms"):
                    u = ds.uniforms[:bs].clone()
                    if last_u[0] is not None and last_u[0].shape == u.shape and torch.equal(u, last_u[0]):
                        st["u_repeat"] += 1
                    last_u[0] = u
                    st["greedy_mask_true"] += int(ds.greedy_mask[:bs].sum())
                    # the accept step draws torch.rand((bs, gamma)) next: peek at it without consuming
                    rs = torch.cuda.get_rng_state()
                    nxt = torch.rand((bs, g), dtype=torch.float32, device=candidates.device)
                    torch.cuda.set_rng_state(rs)
                    st["u_equal_accept"] = st.get("u_equal_accept", 0) + int(torch.equal(nxt, ds.uniforms[:bs]))
                    st["u_close_rows"] = st.get("u_close_rows", 0) + int(((nxt - ds.uniforms[:bs]).abs() < 1e-7).all(-1).sum())
                    if st["rounds"] <= 2:
                        print("[ax-diag] walk u", ds.uniforms[0, :4].tolist(), "accept u", nxt[0, :4].tolist(), flush=True)
                if ds is not None and hasattr(ds, "out"):
                    walk = ds.out[: bs * g].view(bs, g)
                    st["cand_eq_walk_rows"] = st.get("cand_eq_walk_rows", 0) + int((d == walk).all(-1).sum())
                # is the drafted token the argmax of q at its position (i.e. greedy-like drafts)?
                qa = candidate_ids[:, :g].gather(-1, q_rows[:, :g].float().argmax(-1, keepdim=True))[..., 0]
                not1 = q_rows[:, :g].float().max(-1).values < 0.9
                st["pos_q_not_peaked"] = st.get("pos_q_not_peaked", 0) + int(not1.sum())
                st["draft_is_qargmax_when_not_peaked"] = st.get("draft_is_qargmax_when_not_peaked", 0) + int(((d == qa) & not1).sum())
                st["expected_qargmax_when_not_peaked"] = st.get("expected_qargmax_when_not_peaked", 0.0) + float((q_rows[:, :g].float().max(-1).values * not1).sum())
                r = st["rounds"]
                if r in (1, 10, 100) or r % 1000 == 0:
                    print("[ax-diag] selector", st, flush=True)
                return acc0(self, candidates=candidates, next_token_logits=next_token_logits, candidate_ids=candidate_ids,
                            q_rows=q_rows, sampling_info=sampling_info, draft_input=draft_input)
            cls._selector_sampling_accept = _selector_sampling_accept
            print("[ax-diag] selector check installed", flush=True)
        if os.environ.get("AX_DIAG_EAGER_UNIFORMS") == "1":
            S = m._SelectorDraftSampler
            stage0 = S.stage_sampling_params
            def stage_sampling_params(self, *, bs, sampling_info):
                stage0(self, bs=bs, sampling_info=sampling_info)
                # eager draw on the default generator, outside the captured graph
                self.uniforms[:bs].copy_(torch.rand(self.uniforms[:bs].shape, device=self.uniforms.device))
            def __call__(self, hidden_states, input_ids):
                bs = hidden_states.shape[0] // self.block_size
                block_ids = input_ids.view(bs, self.block_size)
                hs = hidden_states.view(bs, self.block_size, -1)[:, 1:, :]
                candidate_ids, scores = m._selector_lattice(self.draft_model, hs, block_ids[:, 0])
                tokens, q_rows = self.selector.sample_path(candidate_ids=candidate_ids, scores=scores,
                    uniforms=self.uniforms[:bs], temperatures=self.temperatures[:bs], greedy_mask=self.greedy_mask[:bs])
                self.out[: tokens.numel()].copy_(tokens.reshape(-1))
                self.candidate_out[:bs].copy_(candidate_ids)
                self.q_out[:bs].copy_(q_rows)
            S.stage_sampling_params = stage_sampling_params
            S.__call__ = __call__
            print("[ax-diag] eager selector uniforms installed", flush=True)
        if os.environ.get("AX_DIAG_BRANCH") == "1":
            ab0 = cls._accept_block
            bc = {}
            def _accept_block(self, *a, **k):
                si = k.get("sampling_info")
                greedy = si is None or si.is_all_greedy
                key = ("greedy" if greedy else "sampled",
                       "selector" if self._selector_sample is not None else "no_selector_sample",
                       "target_predict" if k.get("target_predict") is not None else "-")
                bc[key] = bc.get(key, 0) + 1
                n = sum(bc.values())
                if n in (1, 10, 100, 300) or n % 1000 == 0:
                    print("[ax-diag] accept branches", n, bc, flush=True)
                return ab0(self, *a, **k)
            cls._accept_block = _accept_block
            print("[ax-diag] branch counter installed", flush=True)
        if os.environ.get("AX_DIAG_REF_ACCEPT") == "1":
            from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
            n_calls = [0]
            def _ref_accept(self, *, candidates, next_token_logits, candidate_ids, q_rows, sampling_info, draft_input):
                bs, block = candidates.shape
                g = block - 1
                si = sampling_info
                if not si.need_top_k_sampling and not si.need_top_p_sampling:
                    t = torch.repeat_interleave(si.temperatures.view(-1, 1), block, dim=0)
                    p = torch.softmax(next_token_logits.float() / t, -1).view(bs, block, -1)
                else:
                    p = build_speculative_verify_target_probs(next_token_logits=next_token_logits, sampling_info=si,
                        draft_token_num=block, bs=bs, max_top_k=draft_input.max_top_k,
                        uniform_top_k_value=draft_input.uniform_top_k_value).float()
                V = p.shape[-1]
                q = torch.zeros(bs, g, V, device=p.device)
                q.scatter_(-1, candidate_ids[:, :g], q_rows[:, :g].float())
                acc = torch.zeros(bs, dtype=torch.int32, device=p.device)
                bonus = torch.zeros(bs, dtype=torch.int64, device=p.device)
                for b in range(bs):
                    i = 0
                    while i < g:
                        tok = int(candidates[b, i + 1])
                        pi, qi = float(p[b, i, tok]), float(q[b, i, tok])
                        if qi > 0 and float(torch.rand(())) * qi < pi:
                            i += 1
                            continue
                        r = (p[b, i] - q[b, i]).clamp_min(0)
                        r = r / r.sum() if float(r.sum()) > 0 else p[b, i]
                        bonus[b] = torch.multinomial(r, 1)[0]
                        break
                    else:
                        bonus[b] = torch.multinomial(p[b, g], 1)[0]
                    acc[b] = i
                if os.environ.get("AX_DIAG_LOG_P"):
                    import json as _j
                    with open(os.environ["AX_DIAG_LOG_P"], "a") as fh:
                        for b2 in range(bs):
                            top = torch.topk(p[b2, 0], 6)
                            fh.write(_j.dumps({"anchor": int(candidates[b2, 0]), "d": candidates[b2, 1:3].tolist(),
                                               "top": [[int(i), round(float(v), 5)] for v, i in zip(top.values, top.indices)],
                                               "acc": int(acc[b2]),
                                               "q0": [[int(c), round(float(v), 5)] for c, v in zip(candidate_ids[b2, 0].tolist(), q_rows[b2, 0].float().tolist()) if v > 1e-4],
                                               "u0": float(self._draft_sampler.uniforms[b2, 0]) if getattr(self, "_draft_sampler", None) is not None and hasattr(self._draft_sampler, "uniforms") else None,
                                               "temp": float(self._draft_sampler.temperatures[b2]) if getattr(self, "_draft_sampler", None) is not None and hasattr(self._draft_sampler, "temperatures") else None,
                                               "greedy": bool(self._draft_sampler.greedy_mask[b2]) if getattr(self, "_draft_sampler", None) is not None and hasattr(self._draft_sampler, "greedy_mask") else None}) + "\n")
                n_calls[0] += 1
                if n_calls[0] in (1, 100, 1000):
                    print("[ax-diag] ref accept", n_calls[0], acc[:4].tolist(), flush=True)
                return acc, bonus
            cls._selector_sampling_accept = _ref_accept
            print("[ax-diag] reference accept installed", flush=True)

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "sglang.srt.speculative.dflash_worker_v2":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            exec0 = spec.loader.exec_module
            def exec_module(module):
                exec0(module)
                _patch()
            spec.loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _Hook())


# AX_DIAG_FI_OLD_OFFSET=1: flashinfer sampling reserves [old, new) on the generator but passes the new
# offset to its kernel, so the next generator consumer redraws the same philox numbers; pass the old one.
if os.environ.get("AX_DIAG_FI_OLD_OFFSET") == "1":
    import importlib.abc as _abc, importlib.util as _util, sys as _sys

    class _FiHook(_abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "flashinfer.sampling":
                return None
            _sys.meta_path.remove(self)
            spec = _util.find_spec(name)
            exec0 = spec.loader.exec_module
            def exec_module(module):
                exec0(module)
                import torch
                def get_seed_and_offset(increment, generator=None, device=None):
                    if generator is None:
                        generator = module.get_default_generators(device)
                    state = generator.get_state()
                    seed, offset = state.view(torch.int64)
                    old = int(offset)
                    generator.set_state(torch.tensor([int(seed), old + (increment + 3) // 4 * 4], dtype=torch.int64).view(torch.uint8))
                    return int(seed), old
                module.get_seed_and_offset = get_seed_and_offset
                print("[ax-diag] flashinfer get_seed_and_offset returns the old offset", flush=True)
            spec.loader.exec_module = exec_module
            return spec
    _sys.meta_path.insert(0, _FiHook())
