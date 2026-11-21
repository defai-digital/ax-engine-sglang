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
from .config import startup_config
from .defaults import (LaunchConfigError, Plan, _plan_gdn_chain_replay, device_capability, drafter_kind,
                       gdn_shape, is_nvfp4, is_qwen4_exp, plan, read_json, route_of)
from .qualification import lookup, target_key

SUPPORTED, EXPERIMENTAL, UNSUPPORTED = "supported", "experimental", "unsupported"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="ax-engine-sglang doctor", description=__doc__.split("\n\n")[0])
    result.add_argument("model", nargs="?", help="target model directory or Hugging Face repository")
    result.add_argument("--model-path")
    result.add_argument("--revision")
    result.add_argument("--speculative-algorithm")
    result.add_argument("--speculative-draft-model-path")
    result.add_argument("--speculative-draft-model-revision")
    result.add_argument("--speculative-num-draft-tokens", type=int)
    result.add_argument("--max-running-requests", type=int)
    result.add_argument("--disable-radix-cache", action="store_true")
    result.add_argument("--speculative-adaptive", action="store_true")
    result.add_argument("--attention-backend")
    result.add_argument("--fp4-gemm-backend", dest="fp4_gemm_runner_backend")
    result.add_argument("--linear-attn-verify-backend")
    result.add_argument("--mamba-ssm-dtype")
    result.add_argument("--disable-flashinfer-autotune", action="store_true")
    result.add_argument("--sm", help="CUDA compute capability to check, such as 12.0 (default: this machine's GPU)")
    result.add_argument("--json", action="store_true", help="print the report as JSON")
    return result


def _capability(text):
    try:
        major, minor = text.split(".")
        return int(major), int(minor)
    except ValueError:
        raise SystemExit(f"--sm expects MAJOR.MINOR such as 12.0, got {text!r}") from None


def _sglang_version():
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return None


def diagnose(options, environ: Mapping[str, str], *, load: Callable = read_json,
             capability: tuple[int, int] | None = None, sglang: str | None = None) -> dict:
    environ = dict(environ)
    model = options.model_path or options.model
    args = SimpleNamespace(
        model_path=model, revision=options.revision,
        speculative_algorithm=options.speculative_algorithm,
        speculative_draft_model_path=options.speculative_draft_model_path,
        speculative_num_draft_tokens=options.speculative_num_draft_tokens,
        max_running_requests=options.max_running_requests, cuda_graph_max_bs_decode=None,
        moe_runner_backend="auto", speculative_moe_runner_backend=None,
        disable_radix_cache=options.disable_radix_cache,
        speculative_adaptive=options.speculative_adaptive,
        attention_backend=options.attention_backend,
        fp4_gemm_runner_backend=options.fp4_gemm_runner_backend,
        linear_attn_verify_backend=options.linear_attn_verify_backend,
        mamba_ssm_dtype=options.mamba_ssm_dtype,
        disable_flashinfer_autotune=options.disable_flashinfer_autotune)
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
    if config and algorithm == "FC1_DFLASH_TREE":   # TreeSpec.handle_server_args
        radix_off = options.disable_radix_cache or settings.args.get("disable_radix_cache", (False,))[0]
        if config.ssm_positions and not radix_off:
            errors.append("AX_SSM_POSITIONS requires --disable-radix-cache")
        if config.policy_mode == "adaptive" and options.speculative_adaptive:
            errors.append("AX_POLICY_MODE=adaptive and --speculative-adaptive both change the draft length; use one")

    replay = {"requested": environ.get("AX_GDN_CHAIN_REPLAY") == "1", "eligible": True, "reason": None}
    try:
        _plan_gdn_chain_replay(Plan(), args, environ, target, drafter, capability)
    except LaunchConfigError as error:
        replay.update(eligible=False, reason=str(error))

    route = route_of(algorithm, resolved) if algorithm in ("", "FC1_DFLASH_TREE") else algorithm.lower()
    experimental = [name for name, on in (
        ("AX_GDN_CHAIN_REPLAY=1", replay["requested"]),
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
        f"exact output ({report['quality']['profile'] or 'no profile'}): {quality['status']}. {quality['reason']}",
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
    options, options.unchecked = parser().parse_known_args(argv)
    capability = _capability(options.sm) if options.sm else device_capability(SimpleNamespace(device="cuda"))
    report = diagnose(options, os.environ, capability=capability, sglang=_sglang_version())
    print(json.dumps(report, indent=2) if options.json else render(report))
    return 1 if report["status"] == UNSUPPORTED else 0
