"""Compare stock AR, stock DFlash2 and AX Engine on frozen Qwen3.8-27B inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request


DEFAULT_LANES = ("ar", "stock8", "tree32")
LANES = (*DEFAULT_LANES, "replay8", "tree16")
PROMPT_SHA256 = "e2b5edfeb7a18b4b01c2a67501a6659fe06eeb3ff2f98e95ef3cd957e17a7fe0"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def launch_config(args, lane):
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", str(args.target.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port), "--attention-backend", "flashinfer",
        "--fp4-gemm-backend", "flashinfer_cutlass", "--max-running-requests", str(args.slots),
        "--cuda-graph-max-bs-decode", str(args.slots), "--context-length", "4096",
        "--max-total-tokens", str(args.kv_tokens), "--mem-fraction-static", str(args.mem_fraction_static),
        "--chunked-prefill-size", "512", "--skip-server-warmup",
        "--disable-radix-cache", "--disable-flashinfer-autotune", "--disable-prefill-cuda-graph",
        "--random-seed", "42",
    ]
    flags = {"SGLANG_PLUGINS": "none", "SGLANG_FLASHINFER_WORKSPACE_SIZE": str(64 * 1024 * 1024)}
    if lane == "ar":
        return command, flags, None
    tree = lane in ("tree16", "tree32")
    algorithm = "FC1_DFLASH_TREE" if tree else "DFLASH"
    command += ["--speculative-algorithm", algorithm, "--speculative-draft-model-path",
                str(args.draft.resolve()), "--speculative-num-draft-tokens", "16" if tree else "8"]
    if not tree:
        command += ["--mamba-ssm-dtype", "float32", "--linear-attn-verify-backend", "triton"]
    if lane == "replay8":
        flags.update(SGLANG_PLUGINS="ax-engine", AX_GDN_CHAIN_REPLAY="1", AX_FI_OFFSET_FIX="0")
    if tree:
        flags.update(
            SGLANG_PLUGINS="ax-engine", AX_NVFP4_VERIFY_PARITY="0", AX_FAST="1", AX_LATTICE="1",
            AX_GDN_LAYERS="48", AX_GDN_HEADS="48", FC1_TREE64="1", FC1_TREE_N=lane[4:], FC1_DRAFT_BLOCK="16",
            FC1_BRANCH_PILOT="1", FC1_OURS="1", FC1_FUSED="1", FC1_COMPILE_BUILD="1", AX_TREE="1",
            AX_CHAIN_WIDTH="8", AX_CHAIN_GRAPH_WIDTHS="8", AX_CHAIN_DRAFT_WIDTHS="8", AX_SSM_POSITIONS="8",
        )
    return command, flags, algorithm


def wait_for_server(process, base_url, model_path, algorithm):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/get_server_info", timeout=2) as response:
                info = json.load(response)
        except OSError:
            time.sleep(1)
            continue
        if (info.get("model_path"), info.get("speculative_algorithm")) != (str(model_path.resolve()), algorithm):
            raise RuntimeError(f"Unexpected server at {base_url}")
        return info
    raise TimeoutError("SGLang startup timed out")


def generate(base_url, input_ids, length, speculative):
    body = {"input_ids": input_ids, "sampling_params": {
        "temperature": 0, "max_new_tokens": length, "ignore_eos": True,
    }}
    request = urllib.request.Request(f"{base_url}/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.load(response)
    wall_s = time.perf_counter() - started
    output_ids = result["output_ids"]
    meta = result["meta_info"]
    if len(output_ids) != length or meta["finish_reason"]["type"] != "length":
        raise ValueError(f"Expected {length} output tokens, got {len(output_ids)} and {meta['finish_reason']}")
    verify_count = meta.get("spec_verify_ct")
    if speculative and not verify_count:
        raise ValueError("Speculative route reported no verification rounds")
    return {"wall_s": wall_s, "output_ids": output_ids, "spec_verify_ct": verify_count}


def run_pass(base_url, prompts, concurrency, length, speculative):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        start = time.perf_counter()
        rows = list(pool.map(lambda ids: generate(base_url, ids, length, speculative), prompts))
        wall_s = time.perf_counter() - start
    return {"concurrency": concurrency, "wall_s": wall_s, "output_tokens": length * len(rows),
            "median_request_s": statistics.median(row["wall_s"] for row in rows), "rows": rows}


def stop_server(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_lane(args, lane, prompts):
    out = args.out / lane
    out.mkdir(parents=True, exist_ok=True)
    (out / "passes.json").unlink(missing_ok=True)
    command, flags, algorithm = launch_config(args, lane)
    write_json(out / "launch.json", {"command": command, "flags": flags, "prompt_sha256": PROMPT_SHA256})
    env = {key: value for key, value in os.environ.items() if not key.startswith(("FC1_", "AX_", "SGLANG_"))}
    env.update(flags, HF_HUB_OFFLINE="1")
    base_url = f"http://127.0.0.1:{args.port}"
    speculative = algorithm is not None
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            info = wait_for_server(process, base_url, args.target, algorithm)
            if bool(info.get("enable_linear_replayssm_spec")) != (lane == "replay8"):
                raise RuntimeError(f"Unexpected ReplaySSM setting for {lane}")
            run_pass(base_url, [max(prompts, key=len)], 1, 32, speculative)
            run_pass(base_url, prompts[:args.slots], args.slots, 32, speculative)
            passes = []
            for concurrency, repeat in ((1, 1), (args.slots, 4), (args.slots, 4), (1, 1)):
                result = run_pass(base_url, prompts * repeat, concurrency, 128, speculative)
                passes.append(result)
                print(f"{lane} C{concurrency}: {result['output_tokens'] / result['wall_s']:.2f} output tok/s",
                      flush=True)
            write_json(out / "passes.json", passes)
        finally:
            stop_server(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lanes", nargs="+", choices=LANES, default=DEFAULT_LANES)
    parser.add_argument("--slots", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--kv-tokens", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.94)
    parser.add_argument("--port", type=int, default=30517)
    args = parser.parse_args()
    raw_prompts = args.input_ids.read_bytes()
    if hashlib.sha256(raw_prompts).hexdigest() != PROMPT_SHA256:
        parser.error("--input-ids must be the published parity-expanded-ids.json asset")
    prompts = json.loads(raw_prompts)
    for lane in args.lanes:
        run_lane(args, lane, prompts)


if __name__ == "__main__":
    main()
