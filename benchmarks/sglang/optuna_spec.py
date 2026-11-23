#!/usr/bin/env python3
"""Tune the adaptive speculative-length policy of a running server with Optuna.

Works with both engines through a live policy file:
  SGLang: ax-engine-sglang with AX_ADAPTIVE_LIVE=<file> and --speculative-adaptive
  vLLM:   ax-engine-vllm with AX_VLLM_ADAPTIVE_K=<file> and a num_speculative_tokens_per_batch_size
          schedule that lists every K to explore
Each trial writes a policy (SGLang adaptive-config format plus "version"), waits until the server
acknowledges it in <file>.ack, runs bench_speed.py on a tuning prompt set and scores the geometric
mean output tok/s over prompt sets and concurrency levels. The tuning prompts come from other rows
of the same datasets as the evaluation prompts (HumanEval, GSM8K, UltraChat), so the 48-prompt
evaluation stays unseen.

  python benchmarks/sglang/optuna_spec.py prompts --out tune_prompts.json
  python benchmarks/sglang/optuna_spec.py tune --live /w/live.json --steps 1 2 3 4 5 6 7 \
      --base-url http://127.0.0.1:30000 --prompt-file tune_prompts.json --out /w/optuna --trials 24
  python benchmarks/sglang/optuna_spec.py evaluate --live /w/live.json --configs /w/optuna/best.json \
      --base-url http://127.0.0.1:30000 --out /w/optuna/eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "baselines"))

DEFAULTS = {"ema_alpha": 0.2, "update_interval": 5, "warmup_batches": 10,
            "up_hysteresis": 0.0, "down_hysteresis": -0.25}
# Policy slots: "1" serves one request in flight, "2" two or more.
SLOTS = ("1", "2")


def build_prompts(n: int, offsets: dict[str, int]) -> dict[str, list[str]]:
    from datasets import load_dataset
    from prompts import CHAT_PROMPTS

    he = load_dataset("openai/openai_humaneval", split="test")
    code = ["Complete the following Python function. Reply with the full function in a ```python code block.\n\n"
            + ex["prompt"] for ex in he.select(range(offsets["code"], offsets["code"] + n))]
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    math_ = ["Solve the problem step by step, then give the final answer on the last line as '#### <number>'.\n\n"
             + ex["question"] for ex in gsm.select(range(offsets["math"], offsets["math"] + n))]
    # The evaluation takes CHAT_PROMPTS and then the first qualifying UltraChat prompts in order;
    # skip far past them.
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    seen, qualifying, chat = set(CHAT_PROMPTS), 0, []
    for ex in uc:
        p = ex["prompt"].strip()
        if 20 <= len(p) <= 600 and p not in seen:
            seen.add(p)
            qualifying += 1
            if qualifying > offsets["chat"]:
                chat.append(p)
                if len(chat) == n:
                    break
    return {"code": code, "math": math_, "chat": chat}


def slot_steps(lo: int, hi: int, stride: int, allowed: list[int]) -> list[int]:
    lo, hi = min(lo, hi), max(lo, hi)
    steps = sorted({*range(lo, hi + 1, stride), hi} & set(allowed))
    # With gaps in the captured steps the range can miss them all; the server would reject any
    # uncaptured step, so fall back to the captured step nearest to hi.
    return steps or [min(allowed, key=lambda s: (abs(s - hi), s))]


def policy_from_params(params: dict, allowed: list[int], version) -> dict:
    cfg = {"version": version, **{k: params.get(k, v) for k, v in DEFAULTS.items()}}
    for slot in SLOTS:
        cfg[slot] = {"candidate_steps": slot_steps(params[f"lo{slot}"], params[f"hi{slot}"],
                                                   params[f"stride{slot}"], allowed)}
    return cfg


def suggest(trial, allowed: list[int]) -> dict:
    params = {
        "ema_alpha": trial.suggest_float("ema_alpha", 0.05, 0.8, log=True),
        "update_interval": trial.suggest_int("update_interval", 1, 8),
        "warmup_batches": trial.suggest_int("warmup_batches", 0, 16),
        "up_hysteresis": trial.suggest_float("up_hysteresis", -0.5, 1.0),
        "down_hysteresis": trial.suggest_float("down_hysteresis", -1.0, 0.5),
    }
    for slot in SLOTS:
        params[f"lo{slot}"] = trial.suggest_int(f"lo{slot}", min(allowed), max(allowed))
        params[f"hi{slot}"] = trial.suggest_int(f"hi{slot}", min(allowed), max(allowed))
        params[f"stride{slot}"] = trial.suggest_int(f"stride{slot}", 1, 2)
    return params


def fixed(k: int) -> dict:
    return {**DEFAULTS, **{f"{p}{s}": v for s in SLOTS for p, v in (("lo", k), ("hi", k), ("stride", 1))}}


def baselines(allowed: list[int]) -> dict[str, dict]:
    """Starting points: the engine default adaptive policy and fixed lengths."""
    lo = 1 if min(allowed) == 1 else 3
    hi = 7 if 7 in allowed else max(allowed)
    default = {**DEFAULTS, **{f"{p}{s}": v for s in SLOTS for p, v in (("lo", lo), ("hi", hi), ("stride", 2))}}
    # Fixed lengths only where the boot captured them; anything else would be rejected.
    return {"default": default, **{f"fixed{k}": fixed(k) for k in (3, 4) if k in allowed}}


def chat_once(base_url: str, model: str, timeout: float = 120) -> None:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Count from 1 to 40."}],
                       "max_tokens": 64, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def apply_policy(live: Path, cfg: dict, base_url: str, model: str, timeout: float = 300) -> dict:
    """Write `cfg`, then send short requests until the server acknowledges its version."""
    ack_path = Path(f"{live}.ack")
    ack_path.unlink(missing_ok=True)   # an ack left by the previous policy must not answer for this one
    tmp = live.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg))
    os.replace(tmp, live)
    deadline = time.time() + timeout
    while time.time() < deadline:
        chat_once(base_url, model)
        if ack_path.exists():
            try:
                ack = json.loads(ack_path.read_text())
            except json.JSONDecodeError:
                continue
            if ack.get("version") != cfg["version"]:
                continue
            if not ack.get("applied"):
                raise RuntimeError(f"policy rejected: {ack.get('error')}")
            return ack
    raise TimeoutError(f"server did not acknowledge policy version {cfg['version']}")


def run_bench(mode: str, args, prompt_file: Path | None, n: int, out: Path) -> list[dict]:
    cmd = [sys.executable, str(ROOT / "benchmarks/baselines/bench_speed.py"), "--mode", mode, "--out", str(out),
           "--base-url", args.base_url, "--preset", "greedy", "--max-tokens", str(args.max_tokens),
           "--n-requests", str(n), "--concurrency", *map(str, args.concurrency)]
    if prompt_file is not None:
        cmd += ["--prompt-file", str(prompt_file)]
    # bench_speed reads the served model name from SERVED_MODEL.
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "SERVED_MODEL": args.model})
    return json.loads((out / mode / "speed_greedy.json").read_text())["results"]


def score(results: list[dict]) -> float:
    rates = [r["output_tok_s"] for r in results]
    return math.exp(sum(math.log(x) for x in rates) / len(rates))


def tune(args) -> None:
    import optuna

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "trials.jsonl").open("a")
    study = optuna.create_study(study_name=args.study, storage=f"sqlite:///{out / 'study.db'}",
                                direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup))
    if not study.trials:
        for params in baselines(args.steps).values():
            study.enqueue_trial(params)

    def objective(trial):
        params = suggest(trial, args.steps)
        cfg = policy_from_params(params, args.steps, trial.number)
        ack = apply_policy(Path(args.live), cfg, args.base_url, args.model)
        start = time.time()
        results = run_bench(f"t{trial.number:03d}", args, Path(args.prompt_file), args.n_requests, out / "trials")
        value = score(results)
        row = {"trial": trial.number, "value": value, "seconds": round(time.time() - start),
               "policy": cfg, "previous_step_counts": ack.get("step_counts"),
               "results": [{k: r.get(k) for k in ("prompt_set", "concurrency", "output_tok_s")}
                           | {"accept_len": r.get("spec", {}).get("mean_acceptance_length")} for r in results]}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"TRIAL {trial.number} score={value:.2f} policy={json.dumps(cfg)}", flush=True)
        return value

    done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL,
                                                    optuna.trial.TrialState.PRUNED))
    remaining = args.trials - len(done)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, catch=(RuntimeError, TimeoutError, subprocess.CalledProcessError))
    defaults = baselines(args.steps)
    (out / "default.json").write_text(json.dumps(policy_from_params(defaults["default"], args.steps, "default"),
                                                 indent=1))
    try:
        best = study.best_trial
    except ValueError:   # every trial failed: nothing to report as best
        sys.exit("NO_COMPLETE_TRIAL: every trial failed; see the server log and trials.jsonl")
    cfg = policy_from_params(best.params, args.steps, "best")
    (out / "best.json").write_text(json.dumps(cfg, indent=1))
    print(f"BEST trial {best.number} score={best.value:.2f} policy={json.dumps(cfg)}", flush=True)


def evaluate(args) -> None:
    """Apply each named policy and run the evaluation prompts (bench_speed defaults)."""
    out = Path(args.out)
    for path in args.configs:
        cfg = json.loads(Path(path).read_text())
        name = Path(path).stem
        cfg["version"] = f"eval-{name}-{time.time():.0f}"
        apply_policy(Path(args.live), cfg, args.base_url, args.model)
        for r in run_bench(f"{args.prefix}{name}", args, None, args.n_requests, out):
            print(f"EVAL {name} {r['prompt_set']} c={r['concurrency']} out={r['output_tok_s']:.1f} "
                  f"accept_len={r.get('spec', {}).get('mean_acceptance_length')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20, help="per set; bench_speed warms up on the 4 after --n-requests")
    p.add_argument("--code-offset", type=int, default=100)
    p.add_argument("--math-offset", type=int, default=500)
    p.add_argument("--chat-offset", type=int, default=200)
    for name in ("tune", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--live", required=True)
        p.add_argument("--base-url", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--max-tokens", type=int, default=256)
        p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
        p.add_argument("--model", default=os.environ.get("SERVED_MODEL", "qwen"),
                       help="served model name (vLLM's default is the checkpoint path)")
    tune_p = sub.choices["tune"]
    tune_p.add_argument("--steps", type=int, nargs="+", required=True,
                        help="speculative lengths the boot captured for every batch size (boot with one "
                             "candidate list shared by all slots; the server rejects steps a slot lacks)")
    tune_p.add_argument("--prompt-file", required=True)
    tune_p.add_argument("--n-requests", type=int, default=16)
    tune_p.add_argument("--trials", type=int, default=24)
    tune_p.add_argument("--startup", type=int, default=8, help="random trials before TPE")
    tune_p.add_argument("--seed", type=int, default=0)
    tune_p.add_argument("--study", default="adaptive")
    eval_p = sub.choices["evaluate"]
    eval_p.add_argument("--configs", nargs="+", required=True)
    eval_p.add_argument("--n-requests", type=int, default=48)
    eval_p.add_argument("--prefix", default="")
    args = ap.parse_args()
    # run_bench works from the repository root, so relative paths are resolved here first.
    for name in ("out", "live", "prompt_file"):
        if getattr(args, name, None):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    args.configs = [str(Path(c).resolve()) for c in getattr(args, "configs", None) or []]
    if args.cmd == "prompts":
        sets = build_prompts(args.n, {"code": args.code_offset, "math": args.math_offset, "chat": args.chat_offset})
        Path(args.out).write_text(json.dumps(sets, indent=1))
        print({k: len(v) for k, v in sets.items()})
    elif args.cmd == "tune":
        tune(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
