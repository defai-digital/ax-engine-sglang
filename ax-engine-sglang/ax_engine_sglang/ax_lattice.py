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
        self.fc1_cand = cand
        self.fc1_lat = lat.float()
        self.fc1_tail = tail.view(bs, -1)
        tokens, q_rows = self.selector.sample_path(candidate_ids=cand, scores=lat, uniforms=self.uniforms[:bs].uniform_(),
                                                   temperatures=self.temperatures[:bs], greedy_mask=self.greedy_mask[:bs])
        self.out[:tokens.numel()].copy_(tokens.reshape(-1))
        self.candidate_out[:bs].copy_(cand)
        self.q_out[:bs].copy_(q_rows)

    call.__wrapped__ = original
    patch_attribute(cls, '__call__', call)
    patch_attribute(__import__("sys").modules[__name__], "_installed", True)


def lattice_logc(lat, tail):
    """(gamma, K, K) lattice + (gamma,) tail -> c0 (K,) and logc (gamma-1, K, K) log-probabilities."""
    g, K, _ = lat.shape
    full = torch.cat((lat, tail[:, None, None].expand(g, K, 1)), dim=-1)
    lp = lat - torch.logsumexp(full, -1, keepdim=True)
    return lp[0, 0], lp[1:]


class LatticeBuilder:
    """The plugin's Builder (fc1_builder.Builder), fed by the DFlash2 lattice instead of unary logits."""

    def __init__(self, sampler, compile_build=False):
        import tree_ops as TO
        self.TO = TO
        self.sampler = sampler
        self.n = int(os.environ.get("FC1_TREE_N", "32"))
        self.chain_first = os.environ.get("FC1_CHAIN_FIRST", "0") == "1"
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
        c0, logc = lattice_logc(self.lat, self.tail)
        ss, cand, parent, _ = TO.beam_tree(c0, logc, self.n, 99, self.chain_first, None, S=self.slots)
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
