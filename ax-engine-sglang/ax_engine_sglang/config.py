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
    gdn_chain_replay: bool = False
    nvfp4_verify_parity: bool = False
    nvfp4_parity_rows: int = 0
    quality_profile: str | None = None
    policy_mode: str = "static"
    adaptive_min_samples: int = 64
    adaptive_margin: float = 0.08
    adaptive_cooldown: int = 50
    adaptive_explore: int = 32

    @property
    def tree_requested(self) -> bool:
        return bool(self.tree64 or self.branch_pilot or self.fused or self.lattice
                    or self.chain_width is not None or self.chain_schedule or self.graph_widths
                    or self.draft_widths or self.chain_attention or self.ssm_positions)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, validate: bool = True) -> PluginConfig:
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
        def number(name, default):
            try:
                return float(env.get(name) or default)
            except ValueError as error:
                raise ValueError(f"{name} must be a number") from error

        share = number("AX_SSM_CONV_SHARE", 0.05)
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
            gdn_chain_replay=flag("AX_GDN_CHAIN_REPLAY"),
            nvfp4_verify_parity=flag("AX_NVFP4_VERIFY_PARITY"),
            nvfp4_parity_rows=integer("AX_NVFP4_PARITY_ROWS", 0),
            quality_profile=env.get("AX_QUALITY_PROFILE") or None,
            policy_mode=env.get("AX_POLICY_MODE") or "static",
            adaptive_min_samples=integer("AX_ADAPTIVE_MIN_SAMPLES", 64),
            adaptive_margin=number("AX_ADAPTIVE_MARGIN", 0.08),
            adaptive_cooldown=integer("AX_ADAPTIVE_COOLDOWN", 50),
            adaptive_explore=integer("AX_ADAPTIVE_EXPLORE", 32),
        )
        if validate:
            if env.get("AX_NVFP4_PARITY_ROWS") not in (None, "") and not config.nvfp4_verify_parity:
                raise ValueError("AX_NVFP4_PARITY_ROWS needs AX_NVFP4_VERIFY_PARITY=1")
            config.validate()
        return config

    def validate(self):
        if self.quality_profile not in (None, "verified", "fast"):
            raise ValueError(f"AX_QUALITY_PROFILE must be verified or fast, got {self.quality_profile!r}")
        if self.quality_profile == "verified" and not self.nvfp4_verify_parity:
            raise ValueError("AX_QUALITY_PROFILE=verified runs the parity path: it needs the launch defaults or "
                             "AX_NVFP4_VERIFY_PARITY=1, and AX_NVFP4_VERIFY_PARITY=0 contradicts it")
        if self.quality_profile == "fast" and self.nvfp4_verify_parity:
            raise ValueError("AX_QUALITY_PROFILE=fast contradicts AX_NVFP4_VERIFY_PARITY=1; use verified or unset one")
        if self.policy_mode not in ("static", "adaptive"):
            raise ValueError(f"AX_POLICY_MODE must be static or adaptive, got {self.policy_mode!r}")
        if self.policy_mode == "adaptive":
            if self.gdn_chain_replay:
                raise ValueError("AX_POLICY_MODE=adaptive does not apply to AX_GDN_CHAIN_REPLAY (fixed width eight)")
            if len(self.graph_widths) < 2:
                raise ValueError("AX_POLICY_MODE=adaptive chooses among AX_CHAIN_GRAPH_WIDTHS; list at least two widths")
            if self.chain_schedule:
                raise ValueError("AX_POLICY_MODE=adaptive and AX_CHAIN_SCHEDULE both choose the chain width; unset one")
            if self.nvfp4_verify_parity:
                raise ValueError("AX_POLICY_MODE=adaptive is not measured with AX_NVFP4_VERIFY_PARITY "
                                 "(parity results cover a fixed chain width)")
            if self.adaptive_min_samples < 1 or self.adaptive_cooldown < 0 or self.adaptive_explore < 2:
                raise ValueError("AX_ADAPTIVE_MIN_SAMPLES must be at least 1, AX_ADAPTIVE_COOLDOWN at least 0 "
                                 "and AX_ADAPTIVE_EXPLORE at least 2")
            if not math.isfinite(self.adaptive_margin) or not 0 <= self.adaptive_margin <= 1:
                raise ValueError("AX_ADAPTIVE_MARGIN must be finite and between 0 and 1")
        if self.gdn_chain_replay and (self.tree_requested or self.gdn_recurrent_verify
                                      or self.nvfp4_verify_parity or self.gdn_unpacked_decode):
            raise ValueError("AX_GDN_CHAIN_REPLAY uses the stock width-eight DFLASH worker without AX tree or parity options")
        if self.fused and self.tree and self.tree_nodes not in (16, 32, 64):
            raise ValueError("FC1_FUSED=1 requires FC1_TREE_N to be 16, 32 or 64")
        if self.nvfp4_verify_parity and (not self.gdn_unpacked_decode
                or (self.tree_requested and not self.gdn_recurrent_verify)):
            raise ValueError("AX_NVFP4_VERIFY_PARITY requires unpacked GDN decode and recurrent DFlash2 verify")
        if self.nvfp4_verify_parity and self.chain_attention:
            # ax_chain builds that backend by name, outside the FlashInfer verification patch.
            raise ValueError("AX_NVFP4_VERIFY_PARITY does not support AX_CHAIN_ATTN")
        if not 0 <= self.nvfp4_parity_rows <= 64:
            raise ValueError("AX_NVFP4_PARITY_ROWS must be between 0 and 64")
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
        if self.policy_mode == "adaptive" and self.chain_width is None:
            chain = self.graph_widths   # the policy chooses among the graph widths, widest first
        for width in (*chain, *self.graph_widths, *self.draft_widths):
            if not 2 <= width <= self.draft_block:
                raise ValueError(f"chain width {width} outside 2..{self.draft_block}")
        if self.graph_widths and not set(chain) <= set(self.graph_widths):
            raise ValueError("every configured chain width needs an AX_CHAIN_GRAPH_WIDTHS runner")
        if any(w not in self.graph_widths or w >= self.draft_block for w in self.draft_widths):
            raise ValueError("AX_CHAIN_DRAFT_WIDTHS needs graph widths below FC1_DRAFT_BLOCK")
        if self.ssm_positions:
            if not (self.fused or self.gdn_recurrent_verify) or not self.tree or self.chain_width is None:
                raise ValueError("AX_SSM_POSITIONS needs FC1_FUSED=1 or AX_GDN_RECURRENT_VERIFY=1, "
                                 "AX_TREE=1 and AX_CHAIN_WIDTH")
            if max((*chain, *self.graph_widths)) > self.ssm_positions:
                raise ValueError("chain graph widths exceed AX_SSM_POSITIONS")


def get_config() -> PluginConfig:
    return _loaded()[0]


def startup_config(environ=None, hooks_available=None):
    """The configuration to validate when the plugin loads, and whether it is provisional.

    In the launcher the tree defaults (FC1_FUSED=1 and the rest of TREE_ENV) are filled in later,
    while SGLang resolves the server arguments. A tree setting that depends on them, such as
    AX_SSM_POSITIONS, is checked against those defaults instead of being refused. The resolver
    clears the cache once it has written the defaults, so the next get_config() is final."""
    if environ is None:
        return _loaded()
    from .defaults import env_defaults, parity_requested, resolution_hooks_available

    try:
        config = PluginConfig.from_env(environ)
        if config.quality_profile == "verified":
            if hooks_available is None:
                hooks_available = resolution_hooks_available()
            if not hooks_available or environ.get("AX_AUTO_DEFAULTS", "1") == "0":
                raise LookupError("AX_QUALITY_PROFILE=verified needs SGLang 0.5.23 launch defaults "
                                  "(resolution hooks, AX_AUTO_DEFAULTS on) to check the qualification")
        return config, False
    except LookupError as error:
        raise ValueError(str(error)) from None
    except ValueError:
        if hooks_available is None:
            hooks_available = resolution_hooks_available()
        if not hooks_available or environ.get("AX_AUTO_DEFAULTS", "1") == "0":
            raise
        tree = PluginConfig.from_env(environ, validate=False).tree_requested
        if not tree and not parity_requested(environ):
            raise
        merged = {**env_defaults(environ, tree),
                  **{name: value for name, value in environ.items() if value != ""}}
        return PluginConfig.from_env(merged), True


@lru_cache(maxsize=1)
def _loaded() -> tuple[PluginConfig, bool]:
    return startup_config(os.environ)


get_config.cache_clear = _loaded.cache_clear
