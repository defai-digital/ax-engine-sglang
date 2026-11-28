"""Tree size and route sweep for Qwen3.8-27B on one Hopper GPU (SM90), with the 128-token retest workload of
results/rtx_pro_6000_retest/run.py: the first 8 (C1) and 24 (C8) prompts per workload, two sweeps.

    cd <work dir> && python <repo>/benchmarks/sglang/hopper_tree_sweep.py [model[:lane] ...]
    e.g. ... hopper_tree_sweep.py 27b-fp8 27b-nvfp4:tree16 27b-fp8:prof-tree32

The work directory holds prompts.json (the B300 prompt file), models/<model> and models/27b-dflash2, and a
stage.log with a "ready <model>" line per staged model; results go to <work dir>/results/<model>/<lane>.

Lanes come from benchmarks/sglang/bench_27b_nvfp4.py (stock8, tree16 / tree32 / tree64, chain8,
replay8). A `prof-<lane>` lane launches <lane>, warms it up, and records a torch-profiler trace of C1 code
requests (compare two with benchmarks/sglang/trace_kernels.py); it reports no throughput.
AX_SWEEP_FP4_BACKEND picks the NVFP4 GEMM backend (default marlin; on Blackwell e.g. flashinfer_cutlass or
flashinfer_cutedsl). NVFP4 has no SM90 tensor-core path, so on Hopper NVFP4 targets use SGLang's Marlin W4A16 fallback, which needs
SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1 (SGLang 0.5.23 otherwise hands Marlin an (fp4, scale) tuple)."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request

from transformers import AutoTokenizer

ROOT = Path.cwd()
OUT = ROOT / "results"
BASE = "http://127.0.0.1:30517"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_27b_nvfp4 import LANES, launch_config, wait_for_server, stop_server, write_json
from argparse import Namespace

SWEEP = ("stock8", "tree32", "tree16", "chain8", "replay8")
MODELS = {"27b-fp8": SWEEP, "27b-nvfp4": SWEEP}
PROFILE_REQUESTS = 2
FAMILIES = ("chat", "code", "math")
LENGTH = 128
# Smaller GPUs: AX_SWEEP_SLOTS / AX_SWEEP_KV_TOKENS / AX_SWEEP_MEM set the server shape, AX_SWEEP_CONCURRENCY
# the two client counts ("1,2" on a 32 GB RTX 5090), AX_SWEEP_FLAGS (JSON) extra lane environment.
# Prefill CUDA graphs stay on; AX_BENCH_EAGER_PREFILL=1 adds --disable-prefill-cuda-graph, as the reports
# before results/rtx_pro_6000_prefill_graph ran.
SLOTS = int(os.environ.get("AX_SWEEP_SLOTS", "8"))
LOW, HIGH = (int(c) for c in os.environ.get("AX_SWEEP_CONCURRENCY", "1,8").split(","))
PROMPTS = {str(LOW): 8, str(HIGH): 24}


def request(input_ids, length, speculative):
    body = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": length, "ignore_eos": True},
        "stream": True,
    }
    req = urllib.request.Request(BASE + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    ttft = None
    result = None
    output = []
    with urllib.request.urlopen(req, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            result = json.loads(payload)
            meta = result.get("meta_info", {})
            count = meta.get("completion_tokens", 0)
            if count and ttft is None:
                ttft = time.perf_counter() - started
            ids = result.get("output_ids", [])
            if len(ids) == count:
                output = ids
            else:
                output.extend(ids)
    elapsed = time.perf_counter() - started
    if not result:
        raise RuntimeError("no stream response")
    meta = result.get("meta_info", {})
    finish = meta.get("finish_reason", {})
    verify = meta.get("spec_verify_ct")
    if (meta.get("completion_tokens"), finish.get("type")) != (length, "length"):
        raise RuntimeError(f"incomplete stream: {meta.get('completion_tokens')}, {finish}")
    if len(output) != length or ttft is None or speculative and not verify:
        raise RuntimeError(f"incomplete stream metadata: ids={len(output)}, ttft={ttft}, verify={verify}")
    return {"seconds": elapsed, "ttft": ttft, "verify": verify,
            "sha256": sha256(json.dumps(output).encode()).hexdigest()}


def run_pass(ids, concurrency, length, speculative):
    with ThreadPoolExecutor(concurrency) as pool:
        started = time.perf_counter()
        rows = list(pool.map(lambda p: request(p, length, speculative), ids))
        elapsed = time.perf_counter() - started
    return {"concurrency": concurrency, "wall_s": elapsed,
            "output_tokens": length * len(rows), "rows": rows}


def launch(model, lane, target):
    args = Namespace(target=target, draft=ROOT / "models" / "27b-dflash2", port=30517, slots=SLOTS,
                     context_length=4096, kv_tokens=int(os.environ.get("AX_SWEEP_KV_TOKENS", "16384")),
                     mem_fraction_static=float(os.environ.get("AX_SWEEP_MEM", "0.90")), verify_parity=False,
                     verified_profile=False)
    command, flags, algorithm = launch_config(args, lane.removeprefix("prof-"))
    if algorithm and os.environ.get("AX_SWEEP_FLAGS"):
        extra = json.loads(os.environ["AX_SWEEP_FLAGS"])
        if flags.get("AX_GDN_CHAIN_REPLAY") == "1":   # replay refuses tree settings such as AX_SSM_POSITIONS
            extra.pop("AX_SSM_POSITIONS", None)
        flags = dict(flags, **extra)
    # AX_SWEEP_SERVER_ARGS (JSON object) overrides server options: {"--chunked-prefill-size": "8192"} sets a
    # value, {"--disable-flashinfer-autotune": null} drops a flag, {"--enable-x": ""} adds a bare flag.
    for option, value in json.loads(os.environ.get("AX_SWEEP_SERVER_ARGS", "{}")).items():
        if option in command:
            index = command.index(option)
            has_value = index + 1 < len(command) and not command[index + 1].startswith("--")
            del command[index:index + (2 if has_value else 1)]
        if value is not None:
            command += [option] + ([value] if value != "" else [])
    index = command.index("--fp4-gemm-backend")
    del command[index:index + 2]
    if "nvfp4" in model:
        backend = os.environ.get("AX_SWEEP_FP4_BACKEND", "marlin")
        command += ["--fp4-gemm-backend", backend]
        if backend == "marlin":
            # Marlin W4A16 cannot take the fused SiLU+FP4-quant (fp4, scale) tuple.
            flags = dict(flags, SGLANG_DISABLE_SILU_FP4_QUANT_FUSION="1")
    return command, flags, algorithm


def post(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300).read()


def profile(out, ids, speculative):
    """Torch-profiler trace of PROFILE_REQUESTS C1 code requests, stopped explicitly afterwards (a
    step count would outlast short speculative requests and never flush the trace)."""
    trace = out / "torch_profile"
    trace.mkdir(exist_ok=True)
    post("/start_profile", {"output_dir": str(trace), "activities": ["CPU", "GPU"]})
    rows = [run_pass(ids["code"][i:i + 1], 1, LENGTH, speculative) for i in range(PROFILE_REQUESTS)]
    post("/stop_profile")
    deadline = time.monotonic() + 300
    while not list(trace.rglob("*.json.gz")) + list(trace.rglob("*.trace.json")):
        if time.monotonic() > deadline:
            raise RuntimeError("no torch-profiler trace written")
        time.sleep(2)
    write_json(out / "profile.json", {"requests": rows,
                                      "note": "profiled run; not a throughput result"})


def wait_for_staging(model):
    deadline = time.monotonic() + 2700
    while time.monotonic() < deadline:
        if f"ready {model}\n" in (ROOT / "stage.log").read_text():
            return
        time.sleep(3)
    raise TimeoutError(f"staging not complete: {model}")


def model_ids(model, target):
    token = AutoTokenizer.from_pretrained(target, local_files_only=True)
    raw = (ROOT / "prompts.json").read_bytes()
    prompts = json.loads(raw)
    ids = {family: [token.apply_chat_template([{"role": "user", "content": prompt}],
                   tokenize=True, return_dict=False, add_generation_prompt=True,
                   enable_thinking=False) for prompt in prompts[family][:24]] for family in FAMILIES}
    write_json(OUT / model / "inputs.json", {
        "prompt_source_sha256": sha256(raw).hexdigest(), "input_ids": ids,
        "token_range": [min(map(len, sum(ids.values(), []))), max(map(len, sum(ids.values(), [])))]})
    return ids


def run_lane(model, lane, target, ids):
    out = OUT / model / lane
    out.mkdir(parents=True, exist_ok=True)
    command, flags, algorithm = launch(model, lane, target)
    write_json(out / "launch.json", {"command": command, "flags": flags})
    env = {key: value for key, value in os.environ.items() if not key.startswith(("FC1_", "AX_", "SGLANG_"))}
    env.update(flags, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            print(f"{model} {lane} starting", flush=True)
            info = wait_for_server(process, BASE, target, algorithm)
            fields = ("max_running_requests", "max_total_num_tokens", "speculative_algorithm",
                      "enable_linear_replayssm_spec", "disable_radix_cache", "attention_backend",
                      "moe_runner_backend", "kv_cache_dtype", "chunked_prefill_size")
            write_json(out / "server-info.json", {key: info.get(key) for key in fields})
            capped = re.findall(r"max_running_requests is capped to (\d+)",
                                (out / "server.log").read_text(errors="replace"))
            if capped and int(capped[-1]) < HIGH:
                raise RuntimeError(f"effective server request capacity {capped[-1]} < {HIGH}")
            if bool(info.get("enable_linear_replayssm_spec")) != (lane.removeprefix("prof-") == "replay8"):
                raise RuntimeError(f"unexpected replay state on {model} {lane}")
            run_pass(ids["chat"][:1], 1, 32, algorithm is not None)
            run_pass(ids["chat"][:8], HIGH, 32, algorithm is not None)
            if lane.startswith("prof-"):
                profile(out, ids, algorithm is not None)
                (out / "done").touch()
                return
            passes = []
            for sweep, concurrencies in enumerate(((HIGH, LOW), (LOW, HIGH)), 1):
                for concurrency in concurrencies:
                    for family in FAMILIES:
                        result = run_pass(ids[family][:PROMPTS[str(concurrency)]], concurrency,
                                          LENGTH, algorithm is not None)
                        result.update(sweep=sweep, family=family)
                        passes.append(result)
                        write_json(out / "passes.json", passes)
                        print(f"{model} {lane} pass{sweep} C{concurrency} {family}: "
                              f"{result['output_tokens'] / result['wall_s']:.2f} tok/s", flush=True)
            (out / "done").touch()
        finally:
            stop_server(process)


def main():
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "environment.json", {
        "packages": {name: version(name) for name in ("sglang", "torch", "flashinfer-python",
                                                      "transformers", "triton", "ax-engine-sglang")},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                        "--format=csv,noheader"], text=True).strip(),
        "tokens": LENGTH, "requests_by_concurrency": PROMPTS, "seed": 42,
        "workloads": list(FAMILIES)})
    selected = []
    for item in sys.argv[1:] or list(MODELS):
        model, _, lane = item.partition(":")
        if model not in MODELS or lane and lane.removeprefix("prof-") not in LANES:
            raise SystemExit(f"unknown model or lane: {item}")
        selected.append((model, (lane,) if lane else MODELS[model]))
    failed = []
    for model, lanes in selected:
        wait_for_staging(model)
        target = (ROOT / "models" / model).resolve()
        (OUT / model).mkdir(parents=True, exist_ok=True)
        ids = model_ids(model, target)
        for lane in lanes:
            out = OUT / model / lane
            if (out / "done").exists():
                continue
            try:
                run_lane(model, lane, target, ids)
            except Exception:
                failed.append(f"{model}/{lane}")
                print(f"FAILED {model} {lane}", flush=True)
                traceback.print_exc()
            time.sleep(3)
    print("Benchmark finished. Failed lanes:", failed, flush=True)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
