"""Launch defaults derived from the target, the drafter and the GPU.

SGLang resolves server arguments in the launcher before it starts the scheduler processes, so the
decisions made here reach every worker: environment variables through process inheritance and
server arguments through SGLang's own resolution record (`declare_resolution`).

- A setting given explicitly (environment variable or server argument) always wins.
- An explicit setting that contradicts the checkpoint stops the launch with the reason.
- `AX_AUTO_DEFAULTS=0` turns all of this off (the plugin then behaves as before 2026-11-19).

`plan()` is pure (no SGLang, no GPU) so the rules can be tested offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Callable, Mapping

TREE_ALGORITHM = "FC1_DFLASH_TREE"
CUTLASS = "flashinfer_cutlass"
SOURCE = "ax-engine-sglang defaults"

# The tree configuration measured on Qwen3.5-9B and Qwen3.8-27B (B200, RTX PRO 6000, Jetson Thor).
TREE_ENV = {
    "FC1_TREE64": "1", "FC1_BRANCH_PILOT": "1", "FC1_OURS": "1", "FC1_FUSED": "1",
    "FC1_COMPILE_BUILD": "1", "AX_FAST": "1", "FC1_TREE_N": "32",
}
# AX_NVFP4_VERIFY_PARITY=1: the settings of the measured RTX PRO 6000 parity lanes. The tree
# values replace their TREE_ENV counterparts; the fused verifier has its own GDN arithmetic.
PARITY_ENV = {"AX_GDN_UNPACKED_DECODE": "1"}
PARITY_TREE_ENV = {"FC1_FUSED": "0", "FC1_TREE_N": "16", "AX_GDN_RECURRENT_VERIFY": "1",
                   "AX_DFLASH_JOINT_SAMPLING": "1"}
# ServerArgs field: (value, command-line flag).
PARITY_ARGS = {"attention_backend": ("flashinfer", "--attention-backend"),
               "fp4_gemm_runner_backend": (CUTLASS, "--fp4-gemm-backend")}
PARITY_TREE_ARGS = {"linear_attn_verify_backend": ("triton", "--linear-attn-verify-backend")}
# FlashInfer autotune picks GEMM tactics by row count: with it on, 8 of 10 fixed-length outputs
# at concurrency eight equalled the same requests at concurrency one (10 of 10 with prefill
# graphs alone). The radix cache was not checked with partial prefix hits.
PARITY_DISABLED = ("disable_radix_cache", "disable_flashinfer_autotune")
QSA_RING = "16"


class LaunchConfigError(ValueError):
    """An explicit setting contradicts the checkpoint or the plugin's supported routes."""


@dataclass
class Plan:
    env: dict[str, tuple[str, str]] = field(default_factory=dict)    # name -> (value, reason)
    args: dict[str, tuple[object, str]] = field(default_factory=dict)  # ServerArgs field -> (value, reason)
    notes: list[str] = field(default_factory=list)

    def default_env(self, environ: Mapping[str, str], name: str, value, reason: str):
        if name not in environ or environ[name] == "":
            self.env[name] = (str(value), reason)


def parity_requested(environ: Mapping[str, str]) -> bool:
    return environ.get("AX_NVFP4_VERIFY_PARITY") == "1"


def env_defaults(environ: Mapping[str, str], tree: bool) -> dict[str, str]:
    """Environment defaults of a launch; parity values take precedence over the tree's."""
    values = dict(TREE_ENV) if tree else {}
    if parity_requested(environ):
        values.update(PARITY_ENV)
        if tree:
            values.update(PARITY_TREE_ENV)
    return values


def parity_algorithm_error(algorithm: str | None) -> str | None:
    """Only AR and the plugin's tree algorithm run the matched verification path."""
    name = (algorithm or "").upper()
    if name and name != TREE_ALGORITHM:
        return (f"ax-engine-sglang: AX_NVFP4_VERIFY_PARITY=1 supports AR and --speculative-algorithm "
                f"{TREE_ALGORITHM} (AX_TREE=0 for chain rounds); {name} verifies with SGLang's batched "
                f"Gated-DeltaNet kernel and would not reproduce the AR reference")
    return None


def text_config(config: Mapping) -> Mapping:
    return config.get("text_config") or config


def gdn_shape(config: Mapping) -> tuple[int, int] | None:
    """Gated-DeltaNet layers and value heads of a Qwen3.5 / Qwen3.8 hybrid target."""
    text = text_config(config)
    layers = sum(kind == "linear_attention" for kind in text.get("layer_types") or ())
    heads = text.get("linear_num_value_heads")
    return (layers, int(heads)) if layers and heads else None


def is_qwen4_exp(config: Mapping) -> bool:
    text = text_config(config)
    return (any("Qwen4Exp" in name for name in config.get("architectures") or ())
            or str(text.get("model_type", "")).startswith("qwen4_exp"))


def drafter_kind(config: Mapping | None) -> str | None:
    names = (config or {}).get("architectures") or ()
    if any(name.startswith("DFlash2") for name in names):
        return "dflash2"
    if any(name.startswith("DFlash") for name in names):
        return "dflash"
    return None


def has_nvfp4_moe(config: Mapping, quant: Mapping | None) -> bool:
    text = text_config(config)
    experts = text.get("num_experts") or text.get("n_routed_experts") or 0
    # config.json carries quant_method / quant_algo; ModelOpt's hf_quant_config.json has no quant_method
    # and nests the algorithm under "quantization".
    embedded = config.get("quantization_config") or text.get("quantization_config") or {}
    method = str(embedded.get("quant_method") or ("modelopt" if quant else "")).lower()
    algo = str(embedded.get("quant_algo") or ((quant or {}).get("quantization") or {}).get("quant_algo") or "")
    return bool(experts) and method == "modelopt" and algo.upper() in ("NVFP4", "MIXED_PRECISION")


def _int_env(environ, name):
    value = environ.get(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError as error:
        raise LaunchConfigError(f"{name} must be an integer, got {value!r}") from error


def _plan_gdn_chain_replay(result, args, environ, target, drafter, capability):
    feature = "AX_GDN_CHAIN_REPLAY=1"
    if (getattr(args, "speculative_algorithm", None) or "").upper() != "DFLASH":
        raise LaunchConfigError(f"{feature} requires --speculative-algorithm DFLASH")
    if (getattr(args, "speculative_num_draft_tokens", None) != 8
            or getattr(args, "speculative_eagle_topk", None) not in (None, 1)
            or drafter_kind(drafter) != "dflash2"):
        raise LaunchConfigError(f"{feature} requires a DFlash2 drafter with --speculative-num-draft-tokens 8")
    if (gdn_shape(target or {}) != (48, 48)
            or "Qwen3_5ForConditionalGeneration" not in (target or {}).get("architectures", ())):
        raise LaunchConfigError(f"{feature} requires the dense Qwen3.8-27B GDN target")
    if capability is None or any(getattr(args, name, 1) != 1 for name in ("tp_size", "pp_size", "dp_size")):
        raise LaunchConfigError(f"{feature} requires a single CUDA GPU with TP1, PP1 and DP1")
    if getattr(args, "disaggregation_mode", "null") not in (None, "null"):
        raise LaunchConfigError(f"{feature} requires a non-disaggregated server")
    if not getattr(args, "disable_radix_cache", False):
        raise LaunchConfigError(f"{feature} requires --disable-radix-cache")
    if (environ.get("SGLANG_RAGGED_VERIFY_MODE", "static") != "static"
            or getattr(args, "enable_page_major_kv_layout", False)
            or getattr(args, "speculative_adaptive", False)):
        raise LaunchConfigError(f"{feature} requires static verify windows and the default KV layout")
    for name, value in (("mamba_ssm_dtype", "float32"), ("linear_attn_verify_backend", "triton")):
        given = getattr(args, name, None)
        if given not in (None, value):
            flag = "--" + name.replace("_", "-")
            raise LaunchConfigError(f"{feature} requires {flag} {value}, got {given}")
        result.args[name] = (value, feature)
    result.args["enable_linear_replayssm_spec"] = (True, feature)


def plan(args, environ: Mapping[str, str], target: Mapping | None, drafter: Mapping | None,
         target_quant: Mapping | None = None, capability: tuple[int, int] | None = None) -> Plan:
    """Decide defaults for one launch. `args` is the raw (unresolved) ServerArgs."""
    result = Plan()
    if environ.get("AX_GDN_CHAIN_REPLAY") == "1":
        _plan_gdn_chain_replay(result, args, environ, target, drafter, capability)
        return result
    if environ.get("AX_AUTO_DEFAULTS", "1") == "0":
        return result
    algorithm = (getattr(args, "speculative_algorithm", None) or "").upper()
    if not target and algorithm and algorithm != TREE_ALGORITHM:
        result.notes.append("could not read the target's config.json: the NVFP4 MoE backend and "
                            "AX_QSA_RING defaults were not derived; set them explicitly if needed")
    target = target or {}

    # NVFP4 MoE on SM110 (Jetson Thor) / SM120: SGLang 0.5.23 `auto` picks FlashInfer TRT-LLM (SM100
    # only) for the runner while preparing weights for another layout ('g1_scale_c' missing).
    if capability and capability[0] in (11, 12) and has_nvfp4_moe(target, target_quant):
        why = f"NVFP4 MoE on SM{capability[0]}{capability[1]}: auto selects TRT-LLM, which needs SM100"
        if getattr(args, "moe_runner_backend", "auto") in (None, "auto"):
            result.args["moe_runner_backend"] = (CUTLASS, why)
        if algorithm and getattr(args, "speculative_moe_runner_backend", None) in (None, "auto"):
            result.args["speculative_moe_runner_backend"] = (CUTLASS, why)

    if algorithm and algorithm != TREE_ALGORITHM and is_qwen4_exp(target):
        # Stock SGLang 0.5.23 compresses QSA index keys from overwritten ring members whenever a
        # verify window crosses a 4-token group (MTP 1-3 included) and refuses windows above 4.
        result.default_env(environ, "AX_QSA_RING", QSA_RING,
                           "Qwen4-Exp QSA with speculative decoding: fixes the 4-slot pending ring")

    parity = parity_requested(environ)
    if parity:
        error = parity_algorithm_error(algorithm)
        if error:
            raise LaunchConfigError(error)
        wanted = {**PARITY_ARGS, **(PARITY_TREE_ARGS if algorithm == TREE_ALGORITHM else {})}
        for name, (value, flag) in wanted.items():
            given = getattr(args, name, None)
            if given in (None, "auto"):
                result.args[name] = (value, "AX_NVFP4_VERIFY_PARITY=1: the measured parity backend")
            elif given != value:
                raise LaunchConfigError(
                    f"ax-engine-sglang: AX_NVFP4_VERIFY_PARITY=1 needs {flag} {value}, got {given}; "
                    f"the parity patches cover that backend only")
        for name in PARITY_DISABLED:
            if not getattr(args, name, False):
                result.args[name] = (True, "AX_NVFP4_VERIFY_PARITY=1: outputs not matched with it on")
        for name, value in PARITY_ENV.items():
            result.default_env(environ, name, value, "AX_NVFP4_VERIFY_PARITY=1")

    if algorithm != TREE_ALGORITHM:
        return result

    if is_qwen4_exp(target):
        raise LaunchConfigError(
            "ax-engine-sglang: the tree (FC1_DFLASH_TREE) does not support Qwen4-Exp "
            "(Qwen3.8-Flash-Next); use --speculative-algorithm EAGLE (MTP) with AX_QSA_RING=16")

    shape = gdn_shape(target)
    if shape:
        for name, value in zip(("AX_GDN_LAYERS", "AX_GDN_HEADS"), shape):
            given = _int_env(environ, name)
            if given is not None and given != value:
                raise LaunchConfigError(
                    f"ax-engine-sglang: {name}={given} but the target has {value} "
                    f"(Gated-DeltaNet layers / value heads {shape[0]} / {shape[1]}); unset it")
            result.default_env(environ, name, value, "Gated-DeltaNet shape read from the target config")
    elif not all(_int_env(environ, name) is not None for name in ("AX_GDN_LAYERS", "AX_GDN_HEADS")):
        # The fused verifier sizes its GDN buffers from these; the 24 / 32 fallback (Qwen3.5-9B)
        # on a larger target writes out of bounds instead of failing.
        raise LaunchConfigError(
            "ax-engine-sglang: could not read the target's Gated-DeltaNet shape from its config.json; "
            "set AX_GDN_LAYERS and AX_GDN_HEADS (e.g. 48 / 48 for Qwen3.8-27B, 24 / 32 for Qwen3.5-9B)")

    kind = drafter_kind(drafter)
    block = getattr(args, "speculative_num_draft_tokens", None)
    block_reason = "--speculative-num-draft-tokens"
    if block is None and drafter:
        block = ((drafter.get("dflash_config") or {}).get("block_size"))
        block_reason = "drafter dflash_config.block_size"
    drafter_block = ((drafter or {}).get("dflash_config") or {}).get("block_size")
    if block_reason.startswith("--") and drafter_block is not None and int(drafter_block) != int(block):
        result.notes.append(f"the drafter config lists block_size {drafter_block}; drafting "
                            f"{block} tokens per round as --speculative-num-draft-tokens asks "
                            f"(FC1_DRAFT_BLOCK follows the flag)")
    if block is not None:
        given = _int_env(environ, "FC1_DRAFT_BLOCK")
        if given is not None and given != int(block):
            raise LaunchConfigError(
                f"ax-engine-sglang: FC1_DRAFT_BLOCK={given} but the draft block is {block} "
                f"({block_reason}); they must match")
        result.default_env(environ, "FC1_DRAFT_BLOCK", int(block), block_reason)
    elif _int_env(environ, "FC1_DRAFT_BLOCK") is None:
        # Without it the worker drafts 24 tokens (the DForkTree default), whatever the drafter was trained for.
        raise LaunchConfigError(
            "ax-engine-sglang: no draft block length: pass --speculative-num-draft-tokens "
            "(16 for the measured Qwen3.5-9B and Qwen3.8-27B lanes) or set FC1_DRAFT_BLOCK")

    lattice = environ.get("AX_LATTICE")
    if lattice == "1" and kind == "dflash":
        raise LaunchConfigError("ax-engine-sglang: AX_LATTICE=1 needs a DFlash2 drafter; "
                                "this drafter is DFlash (v1)")
    if lattice in (None, "") and kind == "dflash2":
        result.default_env(environ, "AX_LATTICE", "1", "DFlash2 drafter: build trees from its lattice")
        lattice = "1"
    if lattice in (None, "") and kind is None:
        raise LaunchConfigError(
            "ax-engine-sglang: could not tell the drafter kind from its config.json; set AX_LATTICE=1 "
            "for a DFlash2 drafter, or AX_LATTICE=0 with FC1_NGRAM_PATH for a DFlash (v1) drafter")
    if lattice != "1":
        path = environ.get("FC1_NGRAM_PATH") or "data/ngram_counts.npz"
        if not Path(path).is_file():
            raise LaunchConfigError(
                f"ax-engine-sglang: trees from a DFlash (v1) drafter are scored with an n-gram "
                f"table, but {path} does not exist. Download ngram_counts.npz from the DForkTree "
                f"v0.1.0 release and set FC1_NGRAM_PATH, or use a DFlash2 drafter (AX_LATTICE=1)")

    for name, value in env_defaults(environ, tree=True).items():
        reason = ("AX_NVFP4_VERIFY_PARITY=1: measured parity tree" if parity and name in PARITY_TREE_ENV
                  else "measured tree configuration")
        if name not in PARITY_ENV:
            result.default_env(environ, name, value, reason)

    if getattr(args, "max_running_requests", None) is None:
        reason = ("tree rounds serve one request; the tree's GDN state cache is sized per slot. "
                  "Set --max-running-requests (and AX_CHAIN_* graphs) for multi-user chain rounds")
        result.args["max_running_requests"] = (1, reason)
        if getattr(args, "cuda_graph_max_bs_decode", None) is None:
            result.args["cuda_graph_max_bs_decode"] = (1, "one running request")
    return result


def read_json(path: str | None, filename: str, revision: str | None = None) -> dict | None:
    """config.json of a local directory or a Hugging Face repository (cache or download)."""
    if not path:
        return None
    try:
        local = Path(path) / filename
        if Path(path).is_dir():
            return json.loads(local.read_text()) if local.is_file() else None
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(path, filename, revision=revision)).read_text())
    except Exception:   # a missing file or no network: fall back to explicit settings
        return None


def device_capability(server_args) -> tuple[int, int] | None:
    """Probe the first SGLang worker's CUDA ordinal in the visible device list."""
    if getattr(server_args, "device", None) not in (None, "cuda"):
        return None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_id = getattr(server_args, "base_gpu_id", None)
            if gpu_id is None:
                gpu_id = 0
            return tuple(torch.cuda.get_device_capability(gpu_id))
    except Exception:
        pass
    return None


def describe(result: Plan, environ: Mapping[str, str]) -> str:
    lines = ["ax-engine-sglang: launch settings (auto = derived here; env = set by you)"]
    for name, (value, reason) in sorted(result.env.items()):
        lines.append(f"  {name}={value}  [auto: {reason}]")
    for name, (value, reason) in sorted(result.args.items()):
        flag = {**PARITY_ARGS, **PARITY_TREE_ARGS}.get(name, (None, "--" + name.replace("_", "-")))[1]
        lines.append(f"  {flag} {value}  [auto: {reason}]")
    for name in sorted(k for k in environ if k.startswith(("AX_", "FC1_")) and k not in result.env):
        lines.append(f"  {name}={environ[name]}  [env]")
    lines.extend(f"  note: {note}" for note in result.notes)
    return "\n".join(lines)


def resolve(server_args, *, environ=None, load: Callable = read_json, capability=None,
            declare: Callable | None = None, on_env_change: Callable | None = None) -> Plan:
    """Apply `plan()` to a ServerArgs record once (both resolution hooks call this)."""
    environ = os.environ if environ is None else environ
    if getattr(server_args, "_ax_defaults_resolved", False):
        return Plan()
    server_args._ax_defaults_resolved = True   # private marker, like SGLang's own _resolved_overrides
    if environ.get("AX_AUTO_DEFAULTS", "1") == "0" and environ.get("AX_GDN_CHAIN_REPLAY") != "1":
        return Plan()
    revision = getattr(server_args, "revision", None)
    target = load(getattr(server_args, "model_path", None), "config.json", revision)
    quant = load(getattr(server_args, "model_path", None), "hf_quant_config.json", revision)
    drafter = load(getattr(server_args, "speculative_draft_model_path", None), "config.json",
                   getattr(server_args, "speculative_draft_model_revision", None))
    result = plan(server_args, environ, target, drafter, quant,
                  device_capability(server_args) if capability is None else capability)
    for name, (value, _) in result.env.items():
        environ[name] = value
    if result.args and declare is not None:
        declare(server_args, SOURCE, **{name: value for name, (value, _) in result.args.items()})
    if result.env and on_env_change is not None:
        on_env_change()
    if result.env or result.args or result.notes or algorithm_uses_plugin(server_args):
        print(describe(result, environ), file=sys.stderr, flush=True)
    return result


def algorithm_uses_plugin(server_args) -> bool:
    return (getattr(server_args, "speculative_algorithm", None) or "").upper() == TREE_ALGORITHM


def resolution_hooks_available() -> bool:
    """SGLang 0.5.23 resolves server arguments through hooks; 0.5.22 has none."""
    try:
        from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook  # noqa: F401
    except ImportError:
        return False
    return True


def install():
    """Run `resolve()` inside SGLang's resolution pipeline (SGLang 0.5.23 resolution hooks)."""
    try:
        from sglang.srt.arg_groups.overrides import declare_resolution
        from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook
    except ImportError:   # SGLang 0.5.22: no resolution hooks; explicit settings only
        return False
    from .config import get_config

    def hook(server_args, previous):
        resolve(server_args, declare=declare_resolution, on_env_change=get_config.cache_clear)
        # The plugin may have loaded with a provisional configuration (config.startup_config);
        # read the resolved environment from here on, even when resolution wrote nothing.
        get_config.cache_clear()
        previous(server_args)

    # MoE backends resolve first (pipeline order), the speculative step reads the tree settings.
    register_resolution_hook("handle_moe_kernel_config")(hook)
    register_resolution_hook("handle_speculative_decoding")(hook)
    return True
