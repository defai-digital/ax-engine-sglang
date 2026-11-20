"""Validated startup settings; importing this module never imports a GPU runtime."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from typing import Mapping


@dataclass(frozen=True)
class PluginConfig:
    tree: bool = True
    tree_sampling: bool = False
    tree64: bool = False
    branch_pilot: bool = False
    fused: bool = False
    lattice: bool = False
    tree_nodes: int = 64
    draft_block: int = 24
    gdn_layers: int = 24
    gdn_heads: int = 32
    chain_width: int | None = None
    chain_schedule: tuple[int, ...] = ()
    graph_widths: tuple[int, ...] = ()
    draft_widths: tuple[int, ...] = ()
    chain_attention: str | None = None
    chain_check: bool = False
    round_dump: str | None = None
    ssm_positions: int = 0
    ssm_conv_share: float = 0.05
    qsa_ring: int = 0
    qsa_stable_topk: bool = False
    eagle_joint_sampling: bool = False
    dflash_joint_sampling: bool = False
    allow_untested_sglang: bool = False
    gdn_unpacked_decode: bool = False
    gdn_recurrent_verify: bool = False
    nvfp4_verify_parity: bool = False

    @property
    def tree_requested(self) -> bool:
        return bool(self.tree64 or self.branch_pilot or self.fused or self.lattice
                    or self.chain_width is not None or self.chain_schedule or self.graph_widths
                    or self.draft_widths or self.chain_attention or self.ssm_positions)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PluginConfig:
        env = os.environ if env is None else env

        def flag(name, default=False):
            value = env.get(name, "1" if default else "0")
            if value not in ("0", "1"):
                raise ValueError(f"{name} must be 0 or 1, got {value!r}")
            return value == "1"

        def integer(name, default):
            try:
                return int(env.get(name) or default)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer") from error

        def widths(name):
            try:
                result = tuple(int(x) for x in env.get(name, "").split())
            except ValueError as error:
                raise ValueError(f"{name} must contain integer widths") from error
            if len(set(result)) != len(result):
                raise ValueError(f"{name} contains duplicate widths")
            return result

        schedule = []
        if env.get("AX_CHAIN_SCHEDULE"):
            for part in env["AX_CHAIN_SCHEDULE"].split(","):
                try:
                    width, count = (int(x) for x in part.strip().split("x"))
                except ValueError as error:
                    raise ValueError("AX_CHAIN_SCHEDULE must contain widthxcount entries") from error
                if count <= 0 or len(schedule) + count > 10000:
                    raise ValueError("AX_CHAIN_SCHEDULE needs positive counts, at most 10000 rounds")
                schedule.extend([width] * count)
        try:
            share = float(env.get("AX_SSM_CONV_SHARE") or 0.05)
        except ValueError as error:
            raise ValueError("AX_SSM_CONV_SHARE must be a number") from error
        config = cls(
            tree=flag("AX_TREE", True), tree_sampling=flag("AX_TREE_SAMPLING"),
            tree64=flag("FC1_TREE64"), branch_pilot=flag("FC1_BRANCH_PILOT"),
            fused=flag("FC1_FUSED"), lattice=flag("AX_LATTICE"),
            tree_nodes=integer("FC1_TREE_N", 64), draft_block=integer("FC1_DRAFT_BLOCK", 24),
            gdn_layers=integer("AX_GDN_LAYERS", 24), gdn_heads=integer("AX_GDN_HEADS", 32),
            chain_width=integer("AX_CHAIN_WIDTH", 0) if env.get("AX_CHAIN_WIDTH") else None,
            chain_schedule=tuple(schedule), graph_widths=widths("AX_CHAIN_GRAPH_WIDTHS"),
            draft_widths=widths("AX_CHAIN_DRAFT_WIDTHS"),
            chain_attention=env.get("AX_CHAIN_ATTN") or None,
            chain_check=flag("AX_CHAIN_CHECK"), round_dump=env.get("AX_ROUND_DUMP") or None,
            ssm_positions=integer("AX_SSM_POSITIONS", 0), ssm_conv_share=share,
            qsa_ring=integer("AX_QSA_RING", 0),
            qsa_stable_topk=flag("AX_QSA_STABLE_TOPK"),
            eagle_joint_sampling=flag("AX_EAGLE_JOINT_SAMPLING"),
            dflash_joint_sampling=flag("AX_DFLASH_JOINT_SAMPLING"),
            allow_untested_sglang=flag("AX_ALLOW_UNTESTED_SGLANG"),
            gdn_unpacked_decode=flag("AX_GDN_UNPACKED_DECODE"),
            gdn_recurrent_verify=flag("AX_GDN_RECURRENT_VERIFY"),
            nvfp4_verify_parity=flag("AX_NVFP4_VERIFY_PARITY"),
        )
        config.validate()
        return config

    def validate(self):
        if self.nvfp4_verify_parity and (not self.gdn_unpacked_decode
                or (self.tree_requested and not self.gdn_recurrent_verify)):
            raise ValueError("AX_NVFP4_VERIFY_PARITY requires unpacked GDN decode and recurrent DFlash2 verify")
        if self.gdn_recurrent_verify and self.fused:
            raise ValueError("AX_GDN_RECURRENT_VERIFY requires FC1_FUSED=0 for native convolution and state commit")
        for name, value, minimum in (
            ("FC1_TREE_N", self.tree_nodes, 2), ("FC1_DRAFT_BLOCK", self.draft_block, 2),
            ("AX_GDN_LAYERS", self.gdn_layers, 1), ("AX_GDN_HEADS", self.gdn_heads, 1),
            ("AX_SSM_POSITIONS", self.ssm_positions, 0),
        ):
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.tree64 and self.draft_block > 32:
            # tree_ops.tree_struct closes ancestors up to distance 31; deeper paths would lose them.
            raise ValueError("FC1_DRAFT_BLOCK must be at most 32 with FC1_TREE64=1")
        if self.qsa_ring and (self.qsa_ring < 4 or self.qsa_ring % 4):
            raise ValueError("AX_QSA_RING must be 0 or a positive multiple of 4")
        if not math.isfinite(self.ssm_conv_share) or not 0 <= self.ssm_conv_share <= 1:
            raise ValueError("AX_SSM_CONV_SHARE must be finite and between 0 and 1")
        if self.chain_attention and not self.graph_widths:
            raise ValueError("AX_CHAIN_ATTN needs AX_CHAIN_GRAPH_WIDTHS")
        chain = self.chain_schedule or ((self.chain_width,) if self.chain_width is not None
                                        else (self.draft_block,))
        for width in (*chain, *self.graph_widths, *self.draft_widths):
            if not 2 <= width <= self.draft_block:
                raise ValueError(f"chain width {width} outside 2..{self.draft_block}")
        if self.graph_widths and not set(chain) <= set(self.graph_widths):
            raise ValueError("every configured chain width needs an AX_CHAIN_GRAPH_WIDTHS runner")
        if any(w not in self.graph_widths or w >= self.draft_block for w in self.draft_widths):
            raise ValueError("AX_CHAIN_DRAFT_WIDTHS needs graph widths below FC1_DRAFT_BLOCK")
        if self.ssm_positions:
            if not self.fused or not self.tree or self.chain_width is None:
                raise ValueError("AX_SSM_POSITIONS needs FC1_FUSED=1, AX_TREE=1 and AX_CHAIN_WIDTH")
            if max((*chain, *self.graph_widths)) > self.ssm_positions:
                raise ValueError("chain graph widths exceed AX_SSM_POSITIONS")


@lru_cache(maxsize=1)
def get_config() -> PluginConfig:
    return PluginConfig.from_env()
