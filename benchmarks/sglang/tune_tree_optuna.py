#!/usr/bin/env python3
"""Tune the shape of the DFlash2 lattice tree offline with Optuna, on `AX_LATTICE_TRACE` recordings.

The tree verifies a fixed number of rows (`FC1_TREE_N`), so its round time does not depend on which
nodes it picks: at a fixed N, more accepted tokens per round is a proportional speedup. This tool
searches the plugin's tree-shape settings for the most accepted tokens per round, scoring each trial
with `simulate_lattice_tree.py` (exact for greedy requests at the traced anchors; see its biases):

  temperature  AX_TREE_TEMP            lattice rows (and the tail) divided by T before the softmax
  penalty      AX_TREE_BRANCH_PENALTY  log-score cost of every branch off a row's argmax
  tail         AX_TREE_TAIL            count the mass outside the top-k in the softmax
  lmax         AX_TREE_LMAX            at most this many branches off the argmax along a path
  spine        FC1_CHAIN_FIRST         reserve the greedy chain first

Requests are split in two halves by a hash of their id: the study maximises the geometric mean over
prompt families (from the trace file name, e.g. `c1-code.jsonl` -> code) of the tokens per round on
the tuning half, relative to the default tree; the summary reports the other half, which the search
never saw. The first trial is the default shape. A recommended shape still needs a GPU A/B on its
own prompts (`tree_width_study.py lanes` with the printed environment).

    python benchmarks/sglang/tune_tree_optuna.py trace-tree16/*.jsonl trace-chain16/*.jsonl \\
        --nodes 16 --trials 120 --out tune16.json

Needs `optuna` (`uv pip install optuna`).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_lattice_tree as sim   # noqa: E402

DEFAULT = {"temperature": 1.0, "penalty": 0.0, "tail": True, "lmax": 99, "spine": False}
LMAX_CHOICES = (1, 2, 3, 4, 5, 6, 8, 99)


def family_of(path):
    """`c1-code.jsonl` -> code; a name without a dash is its own family."""
    return Path(path).stem.rsplit("-", 1)[-1]


def load_rounds(paths, families=None):
    """Scorable rounds (lattice present, continuation not cut) as (family, half, lattice, truth)."""
    rounds, skipped = [], defaultdict(int)
    for (path, rid), rows in sim.load_requests(paths).items():
        family = family_of(path)
        if families and family not in families:
            continue
        half = zlib.crc32(f"{Path(path).name}:{rid}".encode()) & 1
        for row in rows:
            lattice = sim.decode_lattice(row)
            if lattice is None:
                skipped["without_lattice"] += 1
                continue
            truth = row["continuation"][:lattice.positions]
            if len(truth) < lattice.positions:
                skipped["cut"] += 1
                continue
            rounds.append((family, half, lattice, tuple(truth)))
    return rounds, dict(skipped)


def policy_name(params, nodes):
    parts = [f"tree:{nodes}"]
    if not params["tail"]:
        parts.append("notail")
    if params["spine"]:
        parts.append("spine")
    if params["temperature"] != 1.0:
        parts.append(f"t={params['temperature']:.4g}")
    if params["penalty"]:
        parts.append(f"b={params['penalty']:.4g}")
    if params["lmax"] != 99:
        parts.append(f"l={params['lmax']}")
    return ":".join(parts)


def environment(params):
    env = {}
    if params["temperature"] != 1.0:
        env["AX_TREE_TEMP"] = f"{params['temperature']:.4g}"
    if params["penalty"]:
        env["AX_TREE_BRANCH_PENALTY"] = f"{params['penalty']:.4g}"
    if not params["tail"]:
        env["AX_TREE_TAIL"] = "0"
    if params["lmax"] != 99:
        env["AX_TREE_LMAX"] = str(params["lmax"])
    if params["spine"]:
        env["FC1_CHAIN_FIRST"] = "1"
    return env


def tokens_per_round(rounds, params, nodes, half=None):
    """{family: tokens per round} (accepted drafts plus the target's own token) on one half or both."""
    accepted, count = defaultdict(int), defaultdict(int)
    lmax = None if params["lmax"] == 99 else params["lmax"]
    for family, h, lattice, truth in rounds:
        if half is not None and h != half:
            continue
        tree = sim.build_tree(lattice, nodes, None, params["tail"], params["spine"],
                              params["temperature"], params["penalty"], lmax)
        accepted[family] += sim.tree_accept(lattice, tree, truth)
        count[family] += 1
    return {f: (accepted[f] + count[f]) / count[f] for f in sorted(count)}


def geomean_ratio(values, baseline):
    return math.exp(sum(math.log(values[f] / baseline[f]) for f in baseline) / len(baseline))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--nodes", type=int, default=16, help="tree rows (FC1_TREE_N)")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family", action="append", help="tune on these prompt families only")
    parser.add_argument("--tune-half", type=int, choices=(0, 1), default=0,
                        help="request half the study sees; the other half is the holdout")
    parser.add_argument("--top", type=int, default=5, help="trials to report on the holdout")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.nodes < 2:
        parser.error("--nodes must be at least 2")
    try:
        import optuna
    except ModuleNotFoundError:
        parser.error("needs optuna: uv pip install optuna")

    rounds, skipped = load_rounds(args.trace, set(args.family or ()))
    tune, hold = args.tune_half, 1 - args.tune_half
    by_half = defaultdict(int)
    for family, h, *_ in rounds:
        by_half[(family, h)] += 1
    families = sorted({f for f, *_ in rounds})
    if not families or any(by_half[(f, tune)] == 0 or by_half[(f, hold)] == 0 for f in families):
        parser.error(f"every family needs rounds in both halves: {dict(by_half)}")
    print(f"{len(rounds)} rounds ({skipped}); families {families}; "
          f"tuning half {tune}: {sum(by_half[(f, tune)] for f in families)} rounds", flush=True)

    started = time.time()
    baseline = tokens_per_round(rounds, DEFAULT, args.nodes, tune)
    print(f"default {policy_name(DEFAULT, args.nodes)}: {baseline} ({time.time() - started:.1f} s per trial)",
          flush=True)

    def objective(trial):
        params = {
            "temperature": trial.suggest_float("temperature", 0.5, 3.0, log=True),
            "penalty": trial.suggest_float("penalty", 0.0, 3.0),
            "tail": trial.suggest_categorical("tail", [True, False]),
            "lmax": trial.suggest_categorical("lmax", list(LMAX_CHOICES)),
            "spine": trial.suggest_categorical("spine", [False, True]),
        }
        values = tokens_per_round(rounds, params, args.nodes, tune)
        trial.set_user_attr("tokens_per_round", values)
        return geomean_ratio(values, baseline)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.enqueue_trial(DEFAULT)
    study.optimize(objective, n_trials=args.trials)

    hold_default = tokens_per_round(rounds, DEFAULT, args.nodes, hold)
    seen, report = set(), []
    for trial in sorted(study.trials, key=lambda t: -(t.value or 0)):
        params = {**DEFAULT, **trial.params}
        name = policy_name(params, args.nodes)
        if name in seen:
            continue
        seen.add(name)
        held = tokens_per_round(rounds, params, args.nodes, hold)
        report.append({"policy": name, "params": params, "environment": environment(params),
                       "tune_gain": trial.value, "tune_tokens_per_round": trial.user_attrs["tokens_per_round"],
                       "holdout_gain": geomean_ratio(held, hold_default), "holdout_tokens_per_round": held})
        if len(report) == args.top:
            break

    print(f"\nholdout (half {hold}) default: " + ", ".join(f"{f} {v:.3f}" for f, v in hold_default.items()))
    for item in report:
        per_family = ", ".join(f"{f} {v:.3f}" for f, v in item["holdout_tokens_per_round"].items())
        print(f"{item['policy']:<40} tune {100 * (item['tune_gain'] - 1):+.2f}%  "
              f"holdout {100 * (item['holdout_gain'] - 1):+.2f}%  ({per_family})")
    best = report[0]
    print("\nbest environment: " + (" ".join(f"{k}={v}" for k, v in best["environment"].items()) or "(defaults)"))
    if args.out:
        args.out.write_text(json.dumps({
            "traces": [str(p) for p in args.trace], "nodes": args.nodes, "trials": args.trials, "seed": args.seed,
            "tune_half": tune, "rounds": len(rounds), "skipped": skipped,
            "rounds_by_family_half": {f"{f}/{h}": n for (f, h), n in sorted(by_half.items())},
            "default_tune_tokens_per_round": baseline, "default_holdout_tokens_per_round": hold_default,
            "top": report}, indent=2) + "\n")


if __name__ == "__main__":
    main()
