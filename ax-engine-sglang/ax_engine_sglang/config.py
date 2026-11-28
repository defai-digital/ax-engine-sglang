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
    chain_width_table: tuple[tuple[int, int], ...] = ()   # (first batch size, width), ascending
    graph_widths: tuple[int, ...] = ()
    draft_widths: tuple[int, ...] = ()
    chain_attention: str | None = None
    chain_check: bool = False
    round_dump: str | None = None
    lattice_trace: str | None = None
    btree: int = 0
    btree_native: bool = False
    tree_temp: float = 1.0
    tree_branch_penalty: float = 0.0
    tree_tail: bool = True
    tree_lmax: int = 99
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
    ax_draft_model: str | None = None
    dflash_kv_graph: bool = False
    dflash_kv_direct: bool = False
    dflash_plan_reuse: bool = False
    dflash_draft_vocab: bool = False
    nvfp4_b12x_down128: bool = False
    nvfp4_humming: bool = False
    norm_fp4: bool = False
    gdn_verify_tiles: bool = False
    gdn_verify_views: bool = False
    nvfp4_verify_parity: bool = False
    nvfp4_parity_rows: int = 0
    quality_profile: str | None = None
    policy_mode: str = "static"
    adaptive_min_samples: int = 32
    adaptive_margin: float = 0.08
    adaptive_cooldown: int = 50
    adaptive_explore: int = 32

    @property
    def tree_requested(self) -> bool:
        return bool(self.tree64 or self.branch_pilot or self.fused or self.lattice
                    or self.chain_width is not None or self.chain_schedule or self.chain_width_table
                    or self.graph_widths
                    or self.draft_widths or self.chain_attention or self.ssm_positions)

    @property
    def lattice_read(self) -> bool:
        """Some round reads the DFlash2 lattice: tree rounds (AX_BTREE needs AX_TREE=1 too) or
        AX_LATTICE_TRACE, which records chain rounds' lattices. AX_TREE=0 alone serves chain rounds
        that draft the same greedy path without it."""
        return self.lattice and (self.tree or bool(self.lattice_trace))

    @property
    def configured_chain_widths(self) -> tuple[int, ...]:
        if self.chain_width_table:
            return tuple(width for _, width in self.chain_width_table)
        if self.policy_mode == "adaptive" and self.chain_width is None:
            return self.graph_widths
        return self.chain_schedule or (self.chain_width if self.chain_width is not None else self.draft_block,)

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
        table = []
        for part in (env.get("AX_CHAIN_WIDTH_BY_BATCH") or "").split(","):
            if not part.strip():
                continue
            try:
                first, width = (int(x) for x in part.strip().split(":"))
            except ValueError as error:
                raise ValueError("AX_CHAIN_WIDTH_BY_BATCH must contain requests:width entries, "
                                 "such as 1:16,4:8") from error
            table.append((first, width))
        if (env.get("AX_CHAIN_WIDTH_BY_BATCH") or "").strip() and not table:
            raise ValueError("AX_CHAIN_WIDTH_BY_BATCH must contain requests:width entries, such as 1:16,4:8")

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
            chain_schedule=tuple(schedule), chain_width_table=tuple(table),
            graph_widths=widths("AX_CHAIN_GRAPH_WIDTHS"),
            draft_widths=widths("AX_CHAIN_DRAFT_WIDTHS"),
            chain_attention=env.get("AX_CHAIN_ATTN") or None,
            chain_check=flag("AX_CHAIN_CHECK"), round_dump=env.get("AX_ROUND_DUMP") or None,
            lattice_trace=env.get("AX_LATTICE_TRACE") or None,
            btree=integer("AX_BTREE", 0), btree_native=flag("AX_BTREE_NATIVE"),
            tree_temp=number("AX_TREE_TEMP", 1.0), tree_branch_penalty=number("AX_TREE_BRANCH_PENALTY", 0.0),
            tree_tail=flag("AX_TREE_TAIL", True), tree_lmax=integer("AX_TREE_LMAX", 99),
            ssm_positions=integer("AX_SSM_POSITIONS", 0), ssm_conv_share=share,
            qsa_ring=integer("AX_QSA_RING", 0),
            qsa_stable_topk=flag("AX_QSA_STABLE_TOPK"),
            eagle_joint_sampling=flag("AX_EAGLE_JOINT_SAMPLING"),
            dflash_joint_sampling=flag("AX_DFLASH_JOINT_SAMPLING"),
            allow_untested_sglang=flag("AX_ALLOW_UNTESTED_SGLANG"),
            gdn_unpacked_decode=flag("AX_GDN_UNPACKED_DECODE"),
            gdn_recurrent_verify=flag("AX_GDN_RECURRENT_VERIFY"),
            gdn_chain_replay=flag("AX_GDN_CHAIN_REPLAY"),
            ax_draft_model=env.get("AX_DRAFT_MODEL") or None,
            dflash_kv_graph=flag("AX_DFLASH_KV_GRAPH"),
            dflash_kv_direct=flag("AX_DFLASH_KV_DIRECT"),
            dflash_plan_reuse=flag("AX_DFLASH_PLAN_REUSE"),
            dflash_draft_vocab=flag("AX_DFLASH_DRAFT_VOCAB"),
            nvfp4_b12x_down128=flag("AX_NVFP4_B12X_DOWN128"),
            nvfp4_humming=flag("AX_NVFP4_HUMMING"),
            norm_fp4=flag("AX_NORM_FP4"),
            gdn_verify_tiles=flag("AX_GDN_VERIFY_TILES"),
            gdn_verify_views=flag("AX_GDN_VERIFY_VIEWS"),
            nvfp4_verify_parity=flag("AX_NVFP4_VERIFY_PARITY"),
            nvfp4_parity_rows=integer("AX_NVFP4_PARITY_ROWS", 0),
            quality_profile=env.get("AX_QUALITY_PROFILE") or None,
            policy_mode=env.get("AX_POLICY_MODE") or "static",
            adaptive_min_samples=integer("AX_ADAPTIVE_MIN_SAMPLES", 32),
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
        if self.chain_width_table:
            firsts = [first for first, _ in self.chain_width_table]
            if firsts[0] != 1 or any(b <= a for a, b in zip(firsts, firsts[1:])):
                raise ValueError("AX_CHAIN_WIDTH_BY_BATCH entries are requests:width, starting at 1 request "
                                 "and ascending, such as 1:16,4:8")
            if self.chain_width is not None or self.chain_schedule or self.policy_mode == "adaptive":
                raise ValueError("AX_CHAIN_WIDTH_BY_BATCH chooses the chain width; unset AX_CHAIN_WIDTH, "
                                 "AX_CHAIN_SCHEDULE and AX_POLICY_MODE=adaptive")
            if self.nvfp4_verify_parity:
                raise ValueError("AX_CHAIN_WIDTH_BY_BATCH is not measured with AX_NVFP4_VERIFY_PARITY "
                                 "(parity results cover a fixed chain width)")
            if not self.graph_widths:
                # Measured with one captured graph per width; the tree's own graph serves one width.
                raise ValueError("AX_CHAIN_WIDTH_BY_BATCH needs its widths in AX_CHAIN_GRAPH_WIDTHS")
        if self.lattice_trace and not self.lattice:
            raise ValueError("AX_LATTICE_TRACE records the DFlash2 lattice: it needs AX_LATTICE=1")
        if self.btree:
            if not 2 <= self.btree <= 8:
                raise ValueError("AX_BTREE is the largest batch served by one tree per request: 0 (off) or 2..8")
            if not (self.tree and self.lattice and self.tree64 and self.branch_pilot):
                raise ValueError("AX_BTREE extends the DFlash2 lattice tree: it needs AX_TREE=1, AX_LATTICE=1, "
                                 "FC1_TREE64=1 and FC1_BRANCH_PILOT=1")
            if self.nvfp4_verify_parity or self.gdn_recurrent_verify or self.tree_sampling:
                raise ValueError("AX_BTREE serves greedy rounds with the tree GDN kernels: unset "
                                 "AX_NVFP4_VERIFY_PARITY, AX_GDN_RECURRENT_VERIFY and AX_TREE_SAMPLING")
            if self.ssm_positions and (self.btree_native or not self.fused):
                # SGLang's native GDN tree verify keeps a state per node; the fused kernels commit the
                # accepted state directly, as the one-request tree does.
                raise ValueError("AX_SSM_POSITIONS with AX_BTREE needs the fused kernels (FC1_FUSED=1, "
                                 "not AX_BTREE_NATIVE)")
            if self.tree_nodes > self.draft_block:
                raise ValueError("AX_BTREE needs FC1_TREE_N at most FC1_DRAFT_BLOCK (the draft KV keeps one row per depth)")
            if self.fused and not self.btree_native and self.tree_nodes not in (16, 32, 64):
                raise ValueError("AX_BTREE with FC1_FUSED=1 runs the fused tree kernels: FC1_TREE_N 16, 32 or 64")
        shaped = (self.tree_temp != 1.0 or self.tree_branch_penalty != 0.0 or not self.tree_tail
                  or self.tree_lmax != 99)
        if shaped and not self.lattice:
            raise ValueError("AX_TREE_TEMP, AX_TREE_BRANCH_PENALTY, AX_TREE_TAIL and AX_TREE_LMAX shape the "
                             "DFlash2 lattice tree: they need AX_LATTICE=1")
        if not (math.isfinite(self.tree_temp) and self.tree_temp > 0):
            raise ValueError("AX_TREE_TEMP must be a positive number")
        if not (math.isfinite(self.tree_branch_penalty) and self.tree_branch_penalty >= 0):
            raise ValueError("AX_TREE_BRANCH_PENALTY must be a number at least 0")
        if self.tree_lmax < 1:
            raise ValueError("AX_TREE_LMAX (branches off the greedy child along a path) must be at least 1")
        if self.btree_native and not self.btree:
            raise ValueError("AX_BTREE_NATIVE selects the GDN kernels of AX_BTREE rounds: set AX_BTREE")
        if self.dflash_kv_graph and not self.gdn_chain_replay:
            raise ValueError("AX_DFLASH_KV_GRAPH requires AX_GDN_CHAIN_REPLAY=1")
        if self.dflash_kv_direct and not self.gdn_chain_replay:
            raise ValueError("AX_DFLASH_KV_DIRECT requires AX_GDN_CHAIN_REPLAY=1")
        if self.dflash_kv_direct and self.dflash_kv_graph:
            raise ValueError("AX_DFLASH_KV_DIRECT and AX_DFLASH_KV_GRAPH are separate experiments")
        if self.dflash_plan_reuse and not self.gdn_chain_replay:
            raise ValueError("AX_DFLASH_PLAN_REUSE requires AX_GDN_CHAIN_REPLAY=1")
        if self.dflash_draft_vocab and not self.gdn_chain_replay:
            raise ValueError("AX_DFLASH_DRAFT_VOCAB requires AX_GDN_CHAIN_REPLAY=1")
        if self.nvfp4_b12x_down128 and not self.gdn_chain_replay:
            raise ValueError("AX_NVFP4_B12X_DOWN128 requires AX_GDN_CHAIN_REPLAY=1")
        if self.nvfp4_humming and (self.nvfp4_verify_parity or self.nvfp4_b12x_down128):
            raise ValueError("AX_NVFP4_HUMMING replaces SGLang's SM90 Marlin kernels: it is not measured with "
                             "AX_NVFP4_VERIFY_PARITY and does not combine with AX_NVFP4_B12X_DOWN128 (SM12x)")
        if self.norm_fp4 and self.nvfp4_verify_parity:
            raise ValueError("AX_NORM_FP4 changes the NVFP4 activation codes the parity profile pins")
        if self.ax_draft_model and (not self.gdn_chain_replay or self.dflash_kv_graph
                                    or self.dflash_kv_direct or self.dflash_plan_reuse
                                    or self.dflash_draft_vocab or self.nvfp4_b12x_down128):
            raise ValueError("AX_DRAFT_MODEL requires standalone AX_GDN_CHAIN_REPLAY=1")
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
        chain = self.configured_chain_widths
        for width in (*chain, *self.graph_widths, *self.draft_widths):
            if not 2 <= width <= self.draft_block:
                raise ValueError(f"chain width {width} outside 2..{self.draft_block}")
        if self.graph_widths and not set(chain) <= set(self.graph_widths):
            raise ValueError("every configured chain width needs an AX_CHAIN_GRAPH_WIDTHS runner")
        if any(w not in self.graph_widths or w >= self.draft_block for w in self.draft_widths):
            raise ValueError("AX_CHAIN_DRAFT_WIDTHS needs graph widths below FC1_DRAFT_BLOCK")
        if self.ssm_positions:
            if (not (self.fused or self.gdn_recurrent_verify) or not self.tree
                    or (self.chain_width is None and self.policy_mode != "adaptive"
                        and not self.chain_width_table)):
                raise ValueError("AX_SSM_POSITIONS needs FC1_FUSED=1 or AX_GDN_RECURRENT_VERIFY=1, "
                                 "AX_TREE=1 and AX_CHAIN_WIDTH or AX_CHAIN_WIDTH_BY_BATCH")
            widest = max((*chain, *self.graph_widths))
            if widest > self.ssm_positions:
                raise ValueError("chain graph widths exceed AX_SSM_POSITIONS")
            if self.tree64 and widest > self.tree_nodes:
                raise ValueError(f"chain width {widest} exceeds the {self.tree_nodes} SSM positions "
                                 f"allocated per slot by FC1_TREE_N={self.tree_nodes}")


def server_args_error(config: PluginConfig, server_args) -> str | None:
    """Rules that need SGLang's server arguments (TreeSpec.handle_server_args and the doctor)."""
    if config.ssm_positions and not getattr(server_args, "disable_radix_cache", False):
        return "AX_SSM_POSITIONS requires --disable-radix-cache"
    # The worker drafts FC1_DRAFT_BLOCK tokens with FC1_TREE64=1, else --speculative-num-draft-tokens.
    block = config.draft_block if config.tree64 else getattr(server_args, "speculative_num_draft_tokens", None)
    if config.ssm_positions and not config.tree64 and block is not None:
        # With FC1_TREE64=0 the allocator starts from SGLang's resolved block, even if the cap is larger.
        widest = max((*config.configured_chain_widths, *config.graph_widths))
        if widest > int(block):
            return f"chain width {widest} exceeds the {block} SSM positions allocated per slot by DFlash"
    if config.chain_width_table:
        widest = max(width for _, width in config.chain_width_table)
        if block is not None and widest > int(block):
            return f"AX_CHAIN_WIDTH_BY_BATCH: chain width {widest} exceeds the draft block {block}"
        if (getattr(server_args, "dp_size", 1) or 1) != 1:
            # DP ranks hold different numbers of requests, so they would verify different widths
            # in one step and disagree on the sizes they gather.
            return "AX_CHAIN_WIDTH_BY_BATCH supports DP1 (each DP rank would choose its own width)"
        return None
    if config.policy_mode != "adaptive":
        return None
    if getattr(server_args, "speculative_adaptive", False):
        return "AX_POLICY_MODE=adaptive and --speculative-adaptive both change the draft length; use one"
    if any((getattr(server_args, name, 1) or 1) != 1 for name in ("tp_size", "pp_size", "dp_size")):
        # Each rank would time its own rounds and could choose another width for the same round.
        return "AX_POLICY_MODE=adaptive supports one GPU (TP1, PP1 and DP1)"
    if block is not None and max(config.graph_widths) > int(block):
        return (f"AX_POLICY_MODE=adaptive: chain width {max(config.graph_widths)} in AX_CHAIN_GRAPH_WIDTHS "
                f"exceeds the draft block {block}")
    return None


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
