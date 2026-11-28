"""`ax-engine-sglang doctor`: check a launch without loading weights or starting SGLang.

It applies the launcher's own rules (`defaults.plan`, `PluginConfig`) to the target's and drafter's
config.json and reports the derived settings, the exact-output qualification and every refusal.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from types import SimpleNamespace
from typing import Callable, Mapping

from . import __version__
from .compat import TESTED_SGLANG
from .config import server_args_error, startup_config
from .defaults import (LaunchConfigError, Plan, _plan_gdn_chain_replay, device_capability, drafter_kind,
                       gdn_shape, is_nvfp4, is_qwen4_exp, plan, read_json, route_of)
from .qualification import lookup, target_key

SUPPORTED, EXPERIMENTAL, UNSUPPORTED = "supported", "experimental", "unsupported"


# The server arguments the launch rules read: flag, ServerArgs field, type (None: a switch), SGLang's default.
SERVER_FLAGS = (
    ("--model-path", "model_path", str, None),
    ("--revision", "revision", str, None),
    ("--speculative-algorithm", "speculative_algorithm", str, None),
    ("--speculative-draft-model-path", "speculative_draft_model_path", str, None),
    ("--speculative-draft-model-revision", "speculative_draft_model_revision", str, None),
    ("--speculative-num-draft-tokens", "speculative_num_draft_tokens", int, None),
    ("--speculative-dflash-block-size", "speculative_dflash_block_size", int, None),
    ("--speculative-eagle-topk", "speculative_eagle_topk", int, None),
    ("--speculative-adaptive", "speculative_adaptive", None, False),
    ("--max-running-requests", "max_running_requests", int, None),
    ("--cuda-graph-max-bs-decode", "cuda_graph_max_bs_decode", int, None),
    ("--disable-radix-cache", "disable_radix_cache", None, False),
    ("--enable-mixed-chunk", "enable_mixed_chunk", None, False),
    ("--disable-flashinfer-autotune", "disable_flashinfer_autotune", None, False),
    ("--disable-prefill-cuda-graph", "disable_prefill_cuda_graph", None, False),
    ("--chunked-prefill-size", "chunked_prefill_size", int, None),
    ("--attention-backend", "attention_backend", str, None),
    ("--fp4-gemm-backend", "fp4_gemm_runner_backend", str, None),
    ("--linear-attn-verify-backend", "linear_attn_verify_backend", str, None),
    ("--mamba-ssm-dtype", "mamba_ssm_dtype", str, None),
    ("--moe-runner-backend", "moe_runner_backend", str, "auto"),
    ("--speculative-moe-runner-backend", "speculative_moe_runner_backend", str, None),
    ("--tp-size", "tp_size", int, 1),
    ("--pp-size", "pp_size", int, 1),
    ("--dp-size", "dp_size", int, 1),
    ("--disaggregation-mode", "disaggregation_mode", str, "null"),
    ("--enable-page-major-kv-layout", "enable_page_major_kv_layout", None, False),
    ("--device", "device", str, None),
    ("--base-gpu-id", "base_gpu_id", int, None),
)


def _capability(text):
    try:
        major, minor = text.split(".")
        return int(major), int(minor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected MAJOR.MINOR such as 12.0, got {text!r}") from None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="ax-engine-sglang doctor", description=__doc__.split("\n\n")[0],
        epilog="Give the model first, as with serve, or with --model-path.")
    for flag, field, kind, default in SERVER_FLAGS:
        if kind is None:
            result.add_argument(flag, dest=field, action="store_true")
        else:
            result.add_argument(flag, dest=field, type=kind, default=default)
    result.add_argument("--sm", type=_capability,
                        help="CUDA compute capability to check, such as 12.0 (default: this machine's GPU)")
    result.add_argument("--json", action="store_true", help="print the report as JSON")
    return result


def parse(argv):
    """The model comes first, as with `serve`; a value after an unknown flag is not taken for it."""
    argv = list(argv)
    model = argv.pop(0) if argv and not argv[0].startswith("-") else None
    options, unchecked = parser().parse_known_args(argv)
    options.model_path = options.model_path or model
    options.unchecked = unchecked
    return options


def _sglang_version():
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return None


def diagnose(options, environ: Mapping[str, str], *, load: Callable = read_json,
             capability: tuple[int, int] | None = None, sglang: str | None = None) -> dict:
    environ = dict(environ)
    model = options.model_path
    args = SimpleNamespace(**{field: getattr(options, field) for _, field, _, _ in SERVER_FLAGS})
    target = load(model, "config.json", options.revision)
    quant = load(model, "hf_quant_config.json", options.revision)
    drafter = load(options.speculative_draft_model_path, "config.json", options.speculative_draft_model_revision)
    algorithm = (options.speculative_algorithm or "").upper()
    errors, warnings = [], []

    if not model:
        errors.append("no target model: pass the model directory or repository of the launch")
    elif target is None:
        warnings.append(f"could not read config.json of {model}")
    # SGLang 0.5.22 has no resolution hooks: the launcher derives nothing and settings must be explicit.
    hooks = sglang is None or sglang.split("+")[0] != "0.5.22"
    if sglang is None:
        warnings.append("SGLang is not installed in this environment; assuming 0.5.23")
    elif sglang.split("+")[0] not in TESTED_SGLANG:
        message = f"SGLang {sglang} is untested (tested: {', '.join(TESTED_SGLANG)})"
        if environ.get("AX_ALLOW_UNTESTED_SGLANG") == "1":
            warnings.append(message)
        else:
            errors.append(message + "; the plugin refuses to load without AX_ALLOW_UNTESTED_SGLANG=1")

    # The launcher validates the environment when the plugin loads, before SGLang resolves the
    # server arguments, and again with the derived defaults.
    try:
        startup_config(dict(environ), hooks_available=hooks)
    except ValueError as error:
        errors.append(f"{error} (checked when the plugin loads, before the launch defaults: "
                      f"set the variables it names explicitly)")
    settings = Plan()
    if hooks and not errors:
        try:
            settings = plan(args, environ, target, drafter, quant, capability)
        except LaunchConfigError as error:
            errors.append(str(error))
    resolved = {**environ, **{name: value for name, (value, _) in settings.env.items()}}
    config = None
    try:
        config, _ = startup_config(resolved, hooks_available=hooks)
    except ValueError as error:
        if not errors:   # a refused plan leaves its defaults unset; report the first cause only
            errors.append(str(error))
    if config and algorithm == "FC1_DFLASH_TREE" and not errors:   # TreeSpec.handle_server_args
        resolved_args = SimpleNamespace(**{**vars(args), **{name: value for name, (value, _) in settings.args.items()}})
        if resolved_args.speculative_num_draft_tokens is None:
            draft_config = drafter or {}
            checkpoint_block = (draft_config.get("dflash_config") or {}).get("block_size", draft_config.get("block_size"))
            resolved_args.speculative_num_draft_tokens = resolved_args.speculative_dflash_block_size
            if resolved_args.speculative_num_draft_tokens is None and type(checkpoint_block) is int and checkpoint_block > 0:
                resolved_args.speculative_num_draft_tokens = checkpoint_block
        error = server_args_error(config, resolved_args)
        if error:
            errors.append(error)

    replay = {"requested": environ.get("AX_GDN_CHAIN_REPLAY") == "1", "eligible": True, "reason": None}
    try:
        _plan_gdn_chain_replay(Plan(), args, environ, target, drafter, capability)
    except LaunchConfigError as error:
        replay.update(eligible=False, reason=str(error))

    route = route_of(algorithm, resolved) if algorithm in ("", "FC1_DFLASH_TREE") else algorithm.lower()
    experimental = [name for name, on in (
        ("AX_CHAIN_WIDTH_BY_BATCH", bool(config and config.chain_width_table)),
        ("AX_GDN_CHAIN_REPLAY=1", replay["requested"]),
        ("AX_DFLASH_KV_GRAPH=1", bool(config and config.dflash_kv_graph)),
        ("AX_DFLASH_KV_DIRECT=1", bool(config and config.dflash_kv_direct)),
        ("AX_DFLASH_PLAN_REUSE=1", bool(config and config.dflash_plan_reuse)),
        ("AX_POLICY_MODE=adaptive", bool(config and config.policy_mode == "adaptive")),
    ) if on]
    if errors:
        status = UNSUPPORTED
    elif experimental or warnings or capability is None:
        status = EXPERIMENTAL
    else:
        status = SUPPORTED
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "experimental_features": experimental,
        "unchecked_flags": list(getattr(options, "unchecked", None) or ()),
        "versions": {"ax_engine_sglang": __version__, "sglang": sglang, "tested_sglang": list(TESTED_SGLANG)},
        "gpu": {"compute_capability": list(capability) if capability else None},
        "target": {
            "path": model, "readable": target is not None, "qualified_as": target_key(target, quant),
            "gdn_shape": list(gdn_shape(target or {}) or ()) or None,
            "nvfp4": is_nvfp4(target or {}, quant), "qwen4_exp": is_qwen4_exp(target or {}),
        },
        "drafter": {"path": options.speculative_draft_model_path, "kind": drafter_kind(drafter)},
        "route": route,
        "quality": {
            "profile": environ.get("AX_QUALITY_PROFILE") or None,
            "claims_exact_output": bool(config and config.quality_profile == "verified" and not errors),
            "exact_output": lookup(capability, target_key(target, quant), route).as_dict(),
        },
        "gdn_chain_replay": replay,
        "policy_mode": config.policy_mode if config else environ.get("AX_POLICY_MODE") or "static",
        "derived": {
            "env": {name: {"value": value, "reason": reason} for name, (value, reason) in sorted(settings.env.items())},
            "args": {name: {"value": value, "reason": reason} for name, (value, reason) in sorted(settings.args.items())},
            "notes": list(settings.notes),
        },
    }


def render(report: dict) -> str:
    capability = report["gpu"]["compute_capability"]
    target, quality, replay = report["target"], report["quality"]["exact_output"], report["gdn_chain_replay"]
    lines = [
        f"status: {report['status']}",
        f"ax-engine-sglang {report['versions']['ax_engine_sglang']}, SGLang {report['versions']['sglang'] or 'not installed'}",
        f"GPU: {'SM%d%d' % tuple(capability) if capability else 'none detected'}",
        f"target: {target['path'] or 'not given'} ({target['qualified_as'] or 'no qualification rows'})",
        f"drafter: {report['drafter']['kind'] or 'none or unknown'}",
        f"route: {report['route']}; policy mode: {report['policy_mode']}",
        ("exact output: claimed by this launch (verified profile). " if report["quality"]["claims_exact_output"]
         else "exact output: not claimed by this launch; AX_QUALITY_PROFILE=verified would find: ")
        + f"{quality['status']}. {quality['reason']}",
    ]
    if replay["eligible"]:
        lines.append("GDN chain replay: eligible" + ("" if replay["requested"] else " (set AX_GDN_CHAIN_REPLAY=1)"))
    if report["quality"]["profile"] != "verified":   # the verified profile lists them in its notes
        lines.extend(f"  requires: {condition}" for condition in quality["conditions"])
    for kind in ("env", "args"):
        for name, item in report["derived"][kind].items():
            lines.append(f"  auto {name}={item['value']}  [{item['reason']}]")
    lines.extend(f"  note: {note}" for note in report["derived"]["notes"])
    lines.extend(f"experimental: {name}" for name in report["experimental_features"])
    if report["unchecked_flags"]:
        lines.append("not checked (flags outside the launch rules): " + " ".join(report["unchecked_flags"]))
    lines.extend(f"warning: {warning}" for warning in report["warnings"])
    lines.extend(f"error: {error}" for error in report["errors"])
    return "\n".join(lines)


def main(argv) -> int:
    options = parse(argv)
    capability = options.sm or device_capability(options)
    report = diagnose(options, os.environ, capability=capability, sglang=_sglang_version())
    print(json.dumps(report, indent=2) if options.json else render(report))
    return 1 if report["status"] == UNSUPPORTED else 0
