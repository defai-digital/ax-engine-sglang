"""Benchmark plugin for native DFlash2 target-verify FP8 projections."""

from collections import Counter
from contextvars import ContextVar
from functools import wraps
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import warnings


PROJECTIONS = {(16_384, 5_120): "gdn_qkvz", (14_336, 5_120): "attention_qkv"}
BENCH_PATH = "fp8_verify_tactics"
ROWS = (48, 64, 96, 128)
NATIVE = "cublas"


def checkpoint_id(target):
    target = Path(target)
    return {name: sha256((target / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")}


def runtime_id(torch):
    return {"sglang": version("sglang"), "flashinfer": version("flashinfer-python"),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability())}


def require_runtime(torch):
    runtime = runtime_id(torch)
    if (runtime["sglang"] != "0.5.23" or runtime["flashinfer"] != "0.7.2"
            or runtime["capability"] != [12, 0] or "RTX PRO 6000" not in runtime["gpu"]):
        raise ValueError(f"FP8 verification benchmarks require RTX PRO 6000, SGLang 0.5.23 "
                         f"and FlashInfer 0.7.2. Found {runtime}")
    return runtime


def tactic_from_json(value):
    if type(value) is int and value >= -1:
        return value
    if (isinstance(value, (list, tuple)) and len(value) == 2 and type(value[0]) is int
            and isinstance(value[1], (list, tuple))
            and all(isinstance(pair, (list, tuple)) and len(pair) == 2
                    and all(type(x) is int for x in pair) for pair in value[1])):
        return (value[0], tuple(map(tuple, value[1])))
    raise ValueError(f"Invalid FP8 tactic {value!r}")


def runner_factories(gemm):
    return {NATIVE: lambda: gemm.get_gemm_module().cublas_fp8_gemm_runner(),
            "cudnn": gemm._cudnn_gemm_fp8_runner,
            "cutlass": lambda: gemm.get_gemm_sm120_module_cutlass_fp8().cutlass_fp8_gemm_runner()}


def read_profile(path, identity, rows):
    profile = json.loads(Path(path).read_text())
    if (profile.get("identity"), profile.get("rows")) != (identity, rows):
        raise ValueError("FP8 profile must match the checkpoint, runtime and verification rows")
    entries = profile["projections"]
    if not entries or set(entries) - set(PROJECTIONS.values()):
        raise ValueError("FP8 profile has no supported projections")
    for entry in entries.values():
        if (entry["backend"] not in (NATIVE, "cudnn", "cutlass")
                or type(entry["samples"]) is not int or entry["samples"] < 2
                or not entry["workspace_bytes"]
                or any(type(n) is not int or n <= 0 for n in entry["workspace_bytes"])):
            raise ValueError("FP8 profiles require a supported backend and at least two exact layer samples")
        entry["tactic"] = tactic_from_json(entry["tactic"])
        entry["native"] = [tactic_from_json(x) for x in entry["native"]]
        if not entry["native"]:
            raise ValueError("FP8 profiles must identify a native tactic")
    return entries


def projection_for(inputs, torch, rows):
    a, b, scale_a, scale_b, out, _ = inputs
    if (a.ndim != 3 or b.ndim != 3 or tuple(a.shape[:2]) != (1, rows)
            or b.shape[0] != 1 or a.shape[-1] != b.shape[-2] or not a.is_cuda
            or a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype
            or out.dtype != torch.bfloat16 or tuple(out.shape) != (1, rows, b.shape[-1])
            or not a.is_contiguous()
            or not b.transpose(-1, -2).is_contiguous() or not out.is_contiguous()
            or any(scale.dtype != torch.float32 or scale.numel() != 1 for scale in (scale_a, scale_b))
            or any(t.device != a.device for t in (b, scale_a, scale_b, out))):
        return None
    return PROJECTIONS.get((b.shape[-1], b.shape[-2]))


class ProjectionCapture:
    def __init__(self, directory, limit, torch):
        self.directory = directory
        self.limit = limit
        self.torch = torch
        self.seen = {name: set() for name in PROJECTIONS.values()}

    def save(self, name, inputs, tactic):
        # Saving precedes reuse of the verify graph's output and workspace.
        seen = self.seen[name]
        pointer = inputs[1].data_ptr()
        if pointer in seen or len(seen) >= self.limit:
            return
        values = {"native": tactic, "workspace_bytes": inputs[5].numel(),
                  "tensors": [tensor.detach().cpu() for tensor in inputs[:5]]}
        self.torch.save(values, self.directory / f"{name}-{len(seen)}.pt")
        seen.add(pointer)


def install(tuner_cls, config, fp8_utils, model_cls, graph_cls, torch, rows,
            capture, entries, factories, activity):
    verify = ContextVar("ax_fp8_verify", default=False)
    capture_hits = ContextVar("ax_fp8_capture_hits", default=None)
    operation = ContextVar("ax_fp8_operation", default=None)
    choose_one = tuner_cls.choose_one
    forward = model_cls.forward
    bmm = fp8_utils._raw_flashinfer_bmm_fp8
    capture_shape = graph_cls.capture_one_shape
    execute = graph_cls.execute
    candidates = {}
    prepared = set()

    @wraps(forward)
    def target_forward(model, input_ids, positions, forward_batch, *args, **kwargs):
        token = verify.set(forward_batch.forward_mode.is_target_verify())
        try:
            return forward(model, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            verify.reset(token)

    @wraps(bmm)
    def scoped_bmm(*args, **kwargs):
        if not verify.get():
            return bmm(*args, **kwargs)
        sample = []
        token = operation.set(sample)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="cuDNN fp8 GEMM tactic failed.*")
                result = bmm(*args, **kwargs)
            if capture and sample:
                capture.save(*sample)
            return result
        finally:
            operation.reset(token)

    @wraps(choose_one)
    def choose(tuner, operation_name, runners, tuning_config, inputs, **kwargs):
        runner, native_tactic = choose_one(tuner, operation_name, runners, tuning_config, inputs, **kwargs)
        sample = operation.get()
        if sample is None or operation_name != "fp8_gemm" or tuning_config is not config:
            return runner, native_tactic
        name = projection_for(inputs, torch, rows)
        if name is None:
            return runner, native_tactic
        capturing = torch.cuda.is_current_stream_capturing()
        graph_counts = capture_hits.get() if capturing else None
        if graph_counts is not None:
            graph_counts[0][name] += 1
        if tuner.is_tuning_mode or type(runner).__name__ != "CublasFp8GemmRunner":
            return runner, native_tactic
        if capture and not capturing:
            sample.extend((name, inputs, native_tactic))
        entry = entries.get(name)
        workspace_bytes = inputs[5].numel()
        if (entry is None or native_tactic not in entry["native"]
                or workspace_bytes not in entry["workspace_bytes"]):
            return runner, native_tactic
        backend, tactic = entry["backend"], entry["tactic"]
        key = (name, inputs[0].device, workspace_bytes)
        if key not in prepared:
            if capturing:
                return runner, native_tactic
            if backend not in candidates:
                candidates[backend] = factories[backend]()
            candidate = candidates[backend]
            if tactic != -1 and tactic not in candidate.get_valid_tactics(inputs, None):
                return runner, native_tactic
            prepared.add(key)
        if graph_counts is not None:
            graph_counts[1][name] += 1
        return candidates[backend], tactic

    @wraps(capture_shape)
    def record_graph(graph, size, forward, stream_idx=None, variant_label=None, attention_variant=None):
        if (not entries or graph.ragged_verify_mode or graph.model_runner.is_draft_worker
                or size * graph.captured_req_width != rows):
            return capture_shape(graph, size, forward, stream_idx, variant_label, attention_variant)
        seen, selected = Counter(), Counter()
        token = capture_hits.set((seen, selected))
        try:
            result = capture_shape(graph, size, forward, stream_idx, variant_label, attention_variant)
        finally:
            capture_hits.reset(token)
        key = graph._make_graph_key(size, stream_idx, variant_label, attention_variant)
        covered = getattr(graph, "_ax_fp8_graphs", set())
        graph._ax_fp8_graphs = covered
        covered.discard(key)
        if all(seen[name] and seen[name] == selected[name] for name in entries):
            covered.add(key)
        return result

    @wraps(execute)
    def record_replay(graph, forward_batch, *args, **kwargs):
        output = execute(graph, forward_batch, *args, **kwargs)
        if (activity is not None and forward_batch.forward_mode.is_target_verify()
                and forward_batch.batch_size * graph.captured_req_width == rows
                and graph._replay_graph_key in getattr(graph, "_ax_fp8_graphs", ())):
            activity.paths.add(BENCH_PATH)
        return output

    tuner_cls.choose_one = choose
    model_cls.forward = target_forward
    fp8_utils._raw_flashinfer_bmm_fp8 = scoped_bmm
    graph_cls.capture_one_shape = record_graph
    graph_cls.execute = record_replay


def register():
    from ax_engine_sglang.ax_dflash_activity import activity
    from flashinfer.gemm import gemm_base as gemm
    from flashinfer.autotuner import AutoTuner
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    import torch

    identity = {"checkpoint": checkpoint_id(os.environ["AX_FP8_TARGET"]), "runtime": require_runtime(torch)}
    rows = int(os.environ["AX_FP8_ROWS"])
    if rows not in ROWS or activity is None:
        raise ValueError("FP8 verification requires a supported row count and DFlash activity tracking")
    mode = os.environ["AX_FP8_MODE"]
    capture = None
    entries = {}
    if mode == "capture":
        directory = Path(os.environ["AX_FP8_CAPTURE"])
        (directory / "manifest.json").write_text(json.dumps({"identity": identity, "rows": rows}) + "\n")
        capture = ProjectionCapture(directory, int(os.environ["AX_FP8_SAMPLES"]), torch)
    elif mode == "candidate":
        entries = read_profile(os.environ["AX_FP8_PROFILE"], identity, rows)
    else:
        raise ValueError(f"Unknown FP8 verification mode {mode}")
    install(AutoTuner, gemm._FP8_GEMM_SM100_TUNING_CONFIG, fp8_utils, Qwen3_5ForConditionalGeneration,
            DecodeCudaGraphRunner, torch, rows, capture, entries, runner_factories(gemm), activity)
