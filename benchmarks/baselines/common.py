"""Shared helpers for Phase 0 benchmarks (OpenAI-compatible vLLM server)."""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

DEFAULT_BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
SERVED_MODEL = os.environ.get("SERVED_MODEL", "qwen")

DIAGNOSTIC_FLAGS = ("AX_CHECK", "AX_FORCE_SYNC", "AX_GDN_DEBUG", "AX_CHAIN_CHECK",
                    "AX_LATTICE_DUPCHECK", "AX_NXT_DEBUG", "AX_QSA_CHECK")
PROFILE_FLAGS = ("NSYS_TRACE", "TRT_ITER_STATS")
DIAGNOSTIC_ENV_KEYS = (*DIAGNOSTIC_FLAGS, *PROFILE_FLAGS, "AX_LATTICE_DEBUG", "AX_ROUND_DUMP",
                       "AX_LATTICE_TRACE", "TORCH_PROFILE_STEPS", "SGLANG_PLUGINS")


def is_diagnostic(env):
    """Detect server launch options that instrument or synchronize measured rounds."""
    return bool(env and (any(env.get(key) == "1" for key in (*DIAGNOSTIC_FLAGS, *PROFILE_FLAGS))
                         or (env.get("AX_LATTICE_DEBUG") or "0") != "0"
                         or env.get("AX_ROUND_DUMP") or env.get("AX_LATTICE_TRACE")
                         or env.get("TORCH_PROFILE_STEPS")
                         or "axprof" in (env.get("SGLANG_PLUGINS") or "")))


def launch_diagnostic(launch):
    lane = (launch or {}).get("lane")
    return is_diagnostic(lane) if lane else None


# Sampling presets from the Qwen3.6 model card. "greedy" is used for quality and parity.
SAMPLING = {
    "greedy": dict(temperature=0.0, top_p=1.0, extra={}),
    "instruct": dict(temperature=0.7, top_p=0.8, extra={"top_k": 20}, presence_penalty=1.5),
    "thinking": dict(temperature=1.0, top_p=0.95, extra={"top_k": 20}, presence_penalty=1.5),
    # Qwen3.8 model card sampling; the public DFlash2 / DSpark numbers use it with thinking on.
    "qwen38": dict(temperature=1.0, top_p=0.95, extra={"top_k": 20}),
}


def request_kwargs(preset: str, thinking: bool, max_tokens: int, seed: int | None = 0) -> dict:
    p = SAMPLING[preset]
    extra = dict(p["extra"])
    extra["chat_template_kwargs"] = {"enable_thinking": thinking}
    kw = dict(
        model=SERVED_MODEL,
        max_tokens=max_tokens,
        temperature=p["temperature"],
        top_p=p["top_p"],
        extra_body=extra,
    )
    if "presence_penalty" in p:
        kw["presence_penalty"] = p["presence_penalty"]
    if seed is not None:
        kw["seed"] = seed
    return kw


SPEC_METRIC_PREFIXES = ("vllm:spec_decode", "sglang:spec_verify_calls_total",
                        "sglang:generation_tokens_total")


def scrape_spec_metrics(base_url: str = DEFAULT_BASE_URL, retries: int = 3) -> dict[str, float] | None:
    """Sum the speculative-decoding counters from the Prometheus endpoint (labels collapsed).

    vLLM exposes vllm:spec_decode*; SGLang exposes verify calls and generated tokens.
    Returns None if the endpoint cannot be read, so a failed scrape is never mistaken for zero."""
    text = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as r:
                text = r.read().decode()
            break
        except Exception:
            time.sleep(1.0 + attempt)
    if text is None:
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(SPEC_METRIC_PREFIXES):
            continue
        m = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name = m.group(1)
        if name.endswith(("_created", "_bucket")):
            continue
        out[name] = out.get(name, 0.0) + float(m.group(3))
    return out


def spec_summary(before: dict | None, after: dict | None) -> dict:
    """Acceptance statistics for one benchmark level from counter snapshots taken around it.

    The acceptance length is pooled over the level: tokens committed per verify forward of a
    request, including the verifier's own token."""
    if before is None or after is None:
        return {"error": "metrics scrape failed"}
    d = {k: after[k] - before.get(k, 0.0) for k in after}
    if any(v < 0 for v in d.values()):
        return {"error": "counter went backwards (server restart?)", "raw_delta": d}

    def pick(sub):
        for k, v in d.items():
            if sub in k and k.endswith("_total"):
                return v
        for k, v in d.items():
            if sub in k:
                return v
        return None

    res = {"raw_delta": d}
    verify_calls = d.get("sglang:spec_verify_calls_total")
    if verify_calls:
        gen = d.get("sglang:generation_tokens_total")
        if gen:
            res["mean_acceptance_length"] = gen / verify_calls
        return res
    drafts = pick("num_drafts")
    draft_tokens = pick("num_draft_tokens")
    accepted = pick("num_accepted_tokens")
    if drafts:
        res["mean_acceptance_length"] = 1.0 + (accepted or 0.0) / drafts
    if draft_tokens:
        res["draft_acceptance_rate"] = (accepted or 0.0) / draft_tokens
    return res


def check_loopback_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("server port must be in 1..65535")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def wait_for_server(base_url: str = DEFAULT_BASE_URL, timeout_s: int = 1800, *,
                    pid: int | None = None, model: str | None = None) -> None:
    def verify_process():
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                raise RuntimeError(f"server process {pid} exited during startup") from None

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        verify_process()
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                healthy = r.status == 200
            if healthy and model is not None:
                with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
                    models = json.load(r)
                if not any(item.get("id") == model for item in models.get("data", [])):
                    raise RuntimeError(f"server at {base_url} does not serve this run's model {model!r}")
        except OSError:
            pass
        else:
            if healthy:
                verify_process()
                return
        time.sleep(5)
    raise TimeoutError(f"server at {base_url} not healthy after {timeout_s}s")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        pending = Path(directory) / path.name
        pending.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        pending.replace(path)
