"""DFlash2 lattice trees for the DForkTree plugin (Track L, 27B). Installed as `ax_lattice.py`.

DFlash2 (`DFlash2DraftModel`) re-ranks the target head's top-16 candidates per position with a
learned low-rank transition score(predecessor -> candidate) and serves the greedy walk over that
16x16 lattice. The plugin's tree search (`tree_ops.beam_tree`) already scores a child through
`logc[s-1][parent's candidate index]`, i.e. it takes exactly a per-depth (parent, child) table, so a
DFlash2 tree needs no new search: this module

1. wraps the plugin's selector draft sampler (`fc1_dflash_worker._SelectorDraftSampler`, run
   inside the draft CUDA graph) so it also keeps the candidate ids, the transition lattice and the
   target-head mass outside the top-16 (one extra full-vocabulary log-sum-exp from the same
   candidate pass, `candidate_topk(..., with_partition=True)`), and
2. provides `LatticeBuilder`, the plugin `Builder` with the lattice in place of the unary logits:
   depth 0 scores the anchor's row, depth s scores row `parent` of position s, each row normalised
   as a log-softmax over its 16 scores plus the outside-top-16 mass (the Eq. 4 form of DForkTree,
   with DFlash2's learned transition in place of the n-gram PMI).

Active only with AX_LATTICE=1 (patched into `fc1_branch64.verify_branch` and `fc1_plugin.register`
by `benchmarks/sglang/patch_dforktree.py`); without it the plugin is unchanged. Losslessness does not depend
on the scores: they only choose which nodes the target verifies.
"""
from __future__ import annotations
try:
    from ax_engine_sglang.compat import patch_attribute
except ModuleNotFoundError as error:
    if error.name != "ax_engine_sglang":
        raise
    patch_attribute = setattr  # Legacy DForkTree overlays copy this file without the package.

import base64
import os

import torch

_installed = False
_d = os.environ.get("AX_LATTICE_DEBUG", "0")
_DEBUG = int(_d) if _d.isdigit() else 0                  # print this many rounds (diagnostics only, syncs)
# AX_LATTICE_DUPCHECK=1 (diagnostics only, syncs every round): count trees in which two children of one
# node carry the same token, and candidate rows with a repeated id. Repeated sibling tokens make the
# sampled tree verify (tree_speculative_sampling_target_only) count that token's probability twice.
_DUPCHECK = os.environ.get("AX_LATTICE_DUPCHECK") == "1"
_dup = {"rounds": 0, "trees_dup_sibling": 0, "dup_pairs": 0, "cand_rows": 0, "cand_rows_dup": 0}


def enabled():
    return os.environ.get("AX_LATTICE") == "1"


def install():
    """Wrap the selector sampler so a decode step also leaves the lattice for the tree builder."""
    global _installed
    if _installed or not enabled():
        return
    from ax_engine_sglang.config import get_config
    if not get_config().lattice_read:
        # The full-width chain graph then captures the release sampler, as the narrower chain widths'
        # samplers do: the same greedy path without the full-vocabulary partition and the lattice
        # (0.31 ms per round on the 27B, B200 trace). AX_TREE is fixed for the server's lifetime and
        # this runs before any draft graph capture, so no captured graph can later serve a tree.
        return
    import fc1_dflash_worker as W
    from sglang.srt.models.dflash import candidate_topk

    cls = W._SelectorDraftSampler
    original = cls.__call__

    def call(self, hidden_states, input_ids):
        if getattr(self, "ax_chain_only", False):
            # a narrower drafting width's sampler (ax_chain._build_draft_runners) serves chain rounds
            # only: same tokens without the lattice and the full-vocabulary partition, which only the
            # tree builder reads (about 0.3 ms per round on the 27B). The main sampler keeps them, so
            # chain rounds at the full block width still pay for them.
            return original(self, hidden_states, input_ids)
        bs = hidden_states.shape[0] // self.block_size
        block_ids = input_ids.view(bs, self.block_size)
        hs = hidden_states.view(bs, self.block_size, -1)[:, 1:, :]
        dm = self.draft_model
        k = self.selector.top_k
        # Mirrors the release __call__ (`_selector_lattice` -> `compute_candidates` -> `build_lattice`
        # -> `sample_path`, same outputs) with the partition added. The outside-top-k mass below is on
        # the raw logit scale, so the unary transform must be the identity (true for the public
        # DFlash2 checkpoints); refuse anything else instead of mis-scaling the rows.
        cfg = dm.draft_config
        if cfg.output_multiplier != 1.0 or cfg.final_logit_softcapping is not None:
            raise RuntimeError("AX_LATTICE needs an identity unary transform (output_multiplier 1.0, no softcap)")
        ids, vals, lse = candidate_topk(hs.reshape(-1, hs.shape[-1]), dm.lm_head, k, with_partition=True)
        unary = dm._transform_unary_logits(vals)
        cand = ids.view(bs, -1, k)
        lat = self.selector.build_lattice(candidate_ids=cand, unary_logits=unary.view(bs, -1, k),
                                          hidden_states=hs, anchor_token_ids=block_ids[:, 0])
        # mass of the target head outside the top-k, on the unary logit scale (exact when the unary
        # transform is the identity, which the public DFlash2 checkpoints use)
        lse_top = torch.logsumexp(vals.float(), -1)
        tail = lse.float() + torch.log1p(-torch.exp(lse_top - lse.float()).clamp(max=1.0 - 1e-6))
        # One sampler serves the draft graph of every captured batch size, so the lattice goes into
        # persistent max-batch buffers like candidate_out: rebinding the attributes would leave them
        # on the tensors of whichever graph was captured last, stale or too short for the others.
        _store_lattice(self, cand, lat.float(), tail.view(bs, -1))
        tokens, q_rows = self.selector.sample_path(candidate_ids=cand, scores=lat, uniforms=self.uniforms[:bs].uniform_(),
                                                   temperatures=self.temperatures[:bs], greedy_mask=self.greedy_mask[:bs])
        self.out[:tokens.numel()].copy_(tokens.reshape(-1))
        self.candidate_out[:bs].copy_(cand)
        self.q_out[:bs].copy_(q_rows)

    call.__wrapped__ = original
    patch_attribute(cls, '__call__', call)
    patch_attribute(__import__("sys").modules[__name__], "_installed", True)


def _store_lattice(sampler, cand, lat, tail):
    """Copy this call's rows into the sampler's max-batch lattice buffers (allocated on first use)."""
    bs = cand.shape[0]
    buffers = getattr(sampler, "_ax_lattice_buffers", None)
    if buffers is None or buffers[0].shape[1:] != cand.shape[1:] or buffers[0].shape[0] < bs:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # A buffer made inside one batch size's capture would be invisible to the graphs captured
            # before it. SGLang warms each size up eagerly (largest first) before capturing it.
            raise RuntimeError("ax_lattice: lattice buffers must be allocated before draft CUDA graph capture")
        rows = max(bs, sampler.candidate_out.shape[0])
        buffers = tuple(torch.zeros((rows, *t.shape[1:]), dtype=t.dtype, device=t.device) for t in (cand, lat, tail))
        sampler._ax_lattice_buffers = buffers
        sampler.fc1_cand, sampler.fc1_lat, sampler.fc1_tail = buffers
    for buffer, value in zip(buffers, (cand, lat, tail)):
        buffer[:bs].copy_(value)


def trace_rows(sampler, count):
    """AX_LATTICE_TRACE: this round's lattice per request, for benchmarks/sglang/simulate_lattice_tree.py.

    Rows of the selector sampler's last call: `cand` (positions x top_k token ids, int32), `lat`
    (positions x top_k x top_k scores, float32: row 0 of position 0 scores the anchor's candidates,
    row p of position s the candidates after candidate p of position s - 1) and `tail` (positions,
    float32: the target-head mass outside the top-k, on the same scale), each little-endian and
    base64-encoded. None when this sampler keeps no lattice (a narrower chain width's sampler) or
    holds fewer rows than the batch. The caller checks that the draft graph ran this round: a graph
    replay at the batch's (padded) size writes the first rows of these buffers without running this
    Python code, while an eager draft leaves the previous round's lattice in place. Reads back to the host: diagnostics only."""
    cand = getattr(sampler, "fc1_cand", None)
    lat = getattr(sampler, "fc1_lat", None)
    tail = getattr(sampler, "fc1_tail", None)
    if cand is None or lat is None or tail is None or cand.shape[0] < count:
        return None

    def encode(tensor):
        array = tensor.contiguous().numpy()
        return base64.b64encode(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()).decode()

    cand = cand[:count].to(torch.int32).cpu()
    lat = lat[:count].float().cpu()
    tail = tail[:count].float().cpu()
    positions, top_k = int(cand.shape[1]), int(cand.shape[2])
    return [{"positions": positions, "top_k": top_k, "cand": encode(cand[i]), "lat": encode(lat[i]),
             "tail": encode(tail[i])} for i in range(count)]


def tree_scores(lat, tail, temp=1.0, penalty=0.0, use_tail=True):
    """Child log-scores of the lattice tree search, (..., gamma, K, K) lattice + (..., gamma) tail.

    Each row is a log-softmax over its K candidates at temperature `temp`, with the mass outside the
    top-k (`tail`, scaled the same way) as one more entry unless `use_tail` is off; a child that is
    not its row's argmax loses `penalty` (a per-branch cost that favours the greedy spine). The
    defaults are the plain lattice log-probabilities. `benchmarks/sglang/simulate_lattice_tree.py`
    implements the same rule; its tests check that both select the same tree."""
    if temp != 1.0:
        lat, tail = lat / temp, tail / temp
    if use_tail:
        lat_full = torch.cat((lat, tail[..., None, None].expand(*lat.shape[:-1], 1)), dim=-1)
    else:
        lat_full = lat
    lp = lat - torch.logsumexp(lat_full, -1, keepdim=True)
    if penalty:
        top = lp.argmax(-1, keepdim=True)
        branch = torch.arange(lp.shape[-1], device=lp.device) != top
        lp = lp - penalty * branch.to(lp.dtype)
    return lp


def tree_shape():
    """(temp, penalty, use_tail, lmax) of the lattice tree search from the plugin configuration."""
    from ax_engine_sglang.config import get_config
    config = get_config()
    return config.tree_temp, config.tree_branch_penalty, config.tree_tail, config.tree_lmax


def lattice_logc(lat, tail, temp=1.0, penalty=0.0, use_tail=True):
    """(gamma, K, K) lattice + (gamma,) tail -> c0 (K,) and logc (gamma-1, K, K) tree scores."""
    lp = tree_scores(lat, tail, temp, penalty, use_tail)
    return lp[0, 0], lp[1:]


class LatticeBuilder:
    """The plugin's Builder (fc1_builder.Builder), fed by the DFlash2 lattice instead of unary logits."""

    def __init__(self, sampler, compile_build=False):
        import tree_ops as TO
        self.TO = TO
        self.sampler = sampler
        from ax_engine_sglang.config import get_config

        self.n = get_config().tree_nodes
        self.chain_first = os.environ.get("FC1_CHAIN_FIRST", "0") == "1"
        self.shape = tree_shape()
        cand, lat, tail = sampler.fc1_cand[0], sampler.fc1_lat[0], sampler.fc1_tail[0]
        self.slots = cand.shape[0]
        self.cand = torch.zeros_like(cand)
        self.lat = torch.zeros_like(lat)
        self.tail = torch.zeros_like(tail)
        self.root = torch.zeros(1, device=cand.device, dtype=torch.long)
        if compile_build:
            self.run = torch.compile(self.run, fullgraph=True, mode="default")
        self._dbg, self._lat_ptr = 0, sampler.fc1_lat.data_ptr()
        self.lat.normal_()                                            # any finite values for warmup
        self.run()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.run()

    def run(self):
        TO = self.TO
        temp, penalty, use_tail, lmax = self.shape
        c0, logc = lattice_logc(self.lat, self.tail, temp, penalty, use_tail)
        ss, cand, parent, _ = TO.beam_tree(c0, logc, self.n, lmax, self.chain_first, None, S=self.slots)
        self.parent = torch.cat((torch.full_like(self.root, -1), parent + 1))
        self.anc, self.depth, self.cidx = TO.tree_struct(self.parent, 4)
        self.tokens = torch.cat((self.root, self.cand[ss, cand]))
        u = torch.arange(self.n, device=self.cand.device)
        children = (self.parent[None, :] == u[:, None]) & (u[None, :] > 0)
        self.nxt = torch.where(children, u[None, :], self.n).min(-1).values
        sibling = (self.parent[:, None] == self.parent[None, :]) & (u[None, :] > u[:, None])
        self.sib = torch.where(sibling, u[None, :], self.n).min(-1).values
        self.nxt = torch.where(self.nxt < self.n, self.nxt, -1)
        self.sib = torch.where(self.sib < self.n, self.sib, -1)

    def __call__(self, logits, root, pre1):
        # The lattice already encodes the transitions. The n-gram argument is
        # kept in the shared builder interface for the legacy overlay.
        s = self.sampler
        self.cand.copy_(s.fc1_cand[0])
        self.lat.copy_(s.fc1_lat[0])
        self.tail.copy_(s.fc1_tail[0])
        self.root.copy_(root)
        self.graph.replay()
        if _DUPCHECK:
            _dupcheck(self.parent, self.tokens, s.fc1_cand[0])
        if _DEBUG and self._dbg < _DEBUG:
            self._dbg += 1
            g = s.fc1_cand.shape[1]
            print(f"[ax_lattice] round {self._dbg} root {int(root.reshape(-1)[0])} chain {s.out[:min(g, 8)].tolist()}"
                  f" cand0 {s.candidate_out[0, :min(g, 8), 0].tolist()} fc1_cand0 {s.fc1_cand[0, :min(g, 8), 0].tolist()}"
                  f" tree {self.tokens[:8].tolist()} parent {self.parent[:8].tolist()}"
                  f" lat_ptr_same {s.fc1_lat.data_ptr() == self._lat_ptr}", flush=True)
        return (self.tokens[None], self.depth, self.parent, self.anc, self.nxt, self.sib)


def _dupcheck(parent, tokens, cand):
    same = (parent[:, None] == parent[None, :]) & (tokens[:, None] == tokens[None, :])
    same &= torch.ones_like(same).triu(1) & (parent[:, None] >= 0)
    pairs = int(same.sum())
    srt = cand.sort(-1).values
    rows_dup = int((srt[:, 1:] == srt[:, :-1]).any(-1).sum())
    _dup["rounds"] += 1
    _dup["trees_dup_sibling"] += pairs > 0
    _dup["dup_pairs"] += pairs
    _dup["cand_rows"] += cand.shape[0]
    _dup["cand_rows_dup"] += rows_dup
    if _dup["rounds"] in (1, 10, 100) or _dup["rounds"] % 1000 == 0:
        print(f"[ax_lattice] dupcheck {_dup}", flush=True)
