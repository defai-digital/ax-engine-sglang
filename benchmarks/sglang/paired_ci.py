"""Paired speed statistics and token identity across serving lanes (Track L, gates GL0/GL1 at c=1).

Input: the per-request files written by `benchmarks/sglang/token_identity.py collect` (one request at a
time, so each prompt's wall time is its whole latency, prefill included). One lane = one server
config; a lane may have several repeat files (complete passes, interleaved with the other lanes
on the same GPU).

Estimator, fixed before looking at results: per prompt family, throughput of a lane =
sum(output tokens) / sum(wall time) over its prompts, with each prompt's time and token count
averaged over that lane's repeats. Ratio = throughput(candidate) / throughput(baseline). 95%
interval: 10,000 paired bootstrap resamples of prompts (the same prompt indices for both lanes;
repeats are averaged inside a prompt, not resampled as independent prompts).

    python benchmarks/sglang/paired_ci.py --lane tree=R/r1/tree.jsonl,R/r2/tree.jsonl \
        --lane chain16=... --lane ar=... --baseline chain16 --identity-ref ar

Prints, per family: throughput of every lane, ratio and CI of every lane against the baseline
and against the best of the --best-of lanes (the best lane is re-picked inside every bootstrap
resample, so choosing the fastest chain is not a free advantage), the output-length ratio (a lane
that writes shorter answers is not faster), the median per-prompt speed ratio, the ratio over the
prompts whose outputs are identical to the baseline (pure speed, no length effect), within-lane
reproducibility across repeats, and token identity to the reference lane (first repeat and all
repeats). Inputs are validated strictly: equal prompt sets, prompt hashes and repeat counts.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import json
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/baselines"))

import numpy as np
from common import is_diagnostic


FINISH_OK = ("stop", "length")


def load_meta(name, p, legacy):
    """The completion sidecar token_identity.py writes after the ids file; None for legacy files."""
    m = Path(p).with_suffix(".meta.json")
    if not m.exists():
        if legacy:
            return None
        raise SystemExit(f"{name}: {p} has no completion sidecar {m.name} (incomplete pass? --legacy for old files)")
    meta = json.loads(m.read_text())
    launch_env = (meta.get("server_launch") or {}).get("lane")
    if meta.get("diagnostic") or is_diagnostic(meta.get("env")) or is_diagnostic(launch_env):
        raise SystemExit(f"{name}: {p} is a diagnostic run (AX_CHECK or profiler): not a speed measurement")
    if meta.get("diagnostic") is None and not legacy:
        raise SystemExit(f"{name}: {p} has no server launch record, so it cannot be shown not to be diagnostic")
    if not (meta.get("server") or {}).get("model_path") and not legacy:
        raise SystemExit(f"{name}: {p} does not record the served model")
    if (meta.get("concurrency") or 1) != 1:
        raise SystemExit(f"{name}: {p} was collected with {meta['concurrency']} requests in flight: not a one-request speed measurement")
    if meta.get("complete") is not True and not legacy:
        raise SystemExit(f"{name}: {p} sidecar does not mark a complete pass")
    if meta.get("ids_sha1") and hashlib.sha1(Path(p).read_bytes()).hexdigest() != meta["ids_sha1"]:
        raise SystemExit(f"{name}: {p} does not match the hash in its sidecar (file replaced after collection)")
    return meta


def load_lane(name, paths, allow_missing_hash=False):
    """{(set, i): {"t": mean wall, "n": mean tokens, "runs": [ids of every repeat], "sha": prompt hash}}

    Strict: no duplicate prompt inside a file, every repeat file of a lane covers the same prompts,
    and the prompt hash (token_identity.py >= QA fix) is the same in every repeat."""
    acc = defaultdict(lambda: {"t": [], "n": [], "runs": [], "sha": None})
    first_keys = None
    for p in paths:
        seen = set()
        for line in open(p):
            r = json.loads(line)
            k = (r["set"], r["i"])
            if k in seen:
                raise SystemExit(f"{name}: prompt {k} appears twice in {p}")
            seen.add(k)
            a = acc[k]
            sha = r.get("prompt_sha1")
            if sha is None and not allow_missing_hash:
                raise SystemExit(f"{name}: {p} has no prompt_sha1 (pass --allow-missing-hash for older files)")
            if a["sha"] is not None and sha is not None and sha != a["sha"]:
                raise SystemExit(f"{name}: prompt {k} has a different prompt hash in {p}")
            a["sha"] = a["sha"] or sha
            a["t"].append(r["wall_s"])
            a["n"].append(r["n_out"])
            if r["n_out"] != len(r["output_ids"]):
                raise SystemExit(f"{name}: {p} prompt {k}: n_out {r['n_out']} != len(output_ids)")
            fin = r.get("finish")
            fin = fin.get("type") if isinstance(fin, dict) else fin
            if fin is not None and fin not in FINISH_OK:
                raise SystemExit(f"{name}: {p} prompt {k} finished with {fin!r} (aborted answer)")
            a["runs"].append(r["output_ids"])
        if first_keys is None:
            first_keys = seen
        elif seen != first_keys:
            raise SystemExit(f"{name}: {p} covers different prompts than {paths[0]}")
    return {k: {"t": float(np.mean(v["t"])), "n": float(np.mean(v["n"])), "runs": v["runs"],
                "ids": v["runs"][0], "reps": len(v["t"]), "sha": v["sha"]} for k, v in acc.items()}


def first_diff(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", action="append", required=True, help="name=file1,file2,...")
    ap.add_argument("--baseline", required=True, help="lane every ratio is taken against")
    ap.add_argument("--best-of", nargs="*", default=[],
                    help="lanes forming the 'fastest tuned baseline' (per family, the fastest of them)")
    ap.add_argument("--identity-ref", help="lane whose outputs are the identity reference (AR)")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out")
    ap.add_argument("--allow-missing-hash", action="store_true", help="accept files without prompt_sha1")
    ap.add_argument("--legacy", action="store_true",
                    help="files from before the completion sidecar and prompt hashes (implies --allow-missing-hash)")
    ap.add_argument("--allow-subset", action="store_true", help="intersect lanes that cover different prompts")
    ap.add_argument("--allow-unequal-repeats", action="store_true")
    args = ap.parse_args()
    lanes, metas = {}, []
    for spec in args.lane:
        name, files = spec.split("=", 1)
        lanes[name] = load_lane(name, files.split(","), args.allow_missing_hash or args.legacy)
        for f in files.split(","):
            metas.append((name, f, load_meta(name, f, args.legacy)))
    # every lane must have run the same workload on the same target
    for field in ("max_tokens", "n", "tokenizer", "warmup"):
        vals = {m.get(field) for _, _, m in metas if m is not None}
        if len(vals) > 1:
            raise SystemExit(f"lanes differ in {field}: {vals}")
    # repeats of one lane must be the same server configuration (same algorithm, block, switches)
    def lane_cfg(m):
        # the overlay directory may differ between GPUs; its content hash (AX_SITE_SHA) must not
        lane = dict((m.get("server_launch") or {}).get("lane") or {})
        if lane.get("AX_SITE_SHA"):
            lane.pop("PYTHONPATH", None)
        return lane

    for name in lanes:
        cfgs = {json.dumps([{k: v for k, v in (m.get("server") or {}).items() if k != "port"}, lane_cfg(m)],
                           sort_keys=True)
                for n2, _, m in metas if n2 == name and m is not None}
        if len(cfgs) > 1:
            raise SystemExit(f"lane {name}: its repeat files come from different server configurations")
    models = {(m.get("server") or {}).get("model_path") for _, _, m in metas if m is not None}
    if len(models - {None}) > 1:
        raise SystemExit(f"lanes serve different targets: {models}")
    for ref in [args.baseline, args.identity_ref, *args.best_of]:
        if ref and ref not in lanes:
            raise SystemExit(f"unknown lane {ref}")
    # every lane must cover the same prompts with the same prompt text and the same repeat count
    names = list(lanes)
    key0 = set(lanes[names[0]])
    for n in names[1:]:
        if set(lanes[n]) != key0 and not args.allow_subset:
            raise SystemExit(f"lane {n} covers different prompts than {names[0]} (--allow-subset to intersect)")
    common = sorted(set.intersection(*[set(l) for l in lanes.values()]))
    for k in common:
        shas = {lanes[n][k]["sha"] for n in names} - {None}
        if len(shas) > 1:
            raise SystemExit(f"prompt {k} has different prompt hashes across lanes")
    rep_counts = {n: {lanes[n][k]["reps"] for k in common} for n in names}
    if len(set().union(*rep_counts.values())) != 1 and not args.allow_unequal_repeats:
        raise SystemExit(f"unequal repeat counts {rep_counts} (--allow-unequal-repeats to proceed)")
    rng = np.random.default_rng(args.seed)
    report = {}
    families = sorted({k[0] for k in common})
    for fam in families:
        keys = [k for k in common if k[0] == fam]
        idx = rng.integers(0, len(keys), size=(args.resamples, len(keys)))
        T = {n: np.array([lane[k]["t"] for k in keys]) for n, lane in lanes.items()}
        N = {n: np.array([lane[k]["n"] for k in keys]) for n, lane in lanes.items()}
        thr = {n: N[n].sum() / T[n].sum() for n in lanes}
        boot = {n: N[n][idx].sum(1) / T[n][idx].sum(1) for n in lanes}
        reps = {n: min(lanes[n][k]["reps"] for k in keys) for n in lanes}
        print(f"== {fam}: {len(keys)} prompts")
        rows = {}
        best = max(args.best_of, key=lambda n: thr[n]) if args.best_of else None
        # 'best tuned baseline' is re-chosen inside every resample, so the selection is not free
        best_boot = np.max(np.stack([boot[b] for b in args.best_of]), 0) if args.best_of else None
        for n in lanes:
            row = {"tok_s": thr[n], "repeats": reps[n]}
            # within-lane reproducibility: prompts whose repeats all produced the same tokens
            row["reproducible"] = sum(all(r == lanes[n][k]["runs"][0] for r in lanes[n][k]["runs"]) for k in keys)
            for ref_name, ref in (("vs_" + args.baseline, args.baseline), ("vs_best", best)):
                if ref is None or ref == n or (ref_name == "vs_best" and n in args.best_of):
                    continue
                r = boot[n] / (best_boot if ref_name == "vs_best" else boot[ref])
                ref_thr = thr[ref]
                per_prompt = (N[n] / T[n]) / (N[ref] / T[ref])
                # every repeat of both lanes must give the same answer, or the prompt is not pure speed
                same = [j for j, k in enumerate(keys)
                        if all(x == lanes[n][k]["ids"] for x in lanes[n][k]["runs"] + lanes[ref][k]["runs"])]
                row[ref_name] = {
                    "ratio": thr[n] / ref_thr, "lo": float(np.quantile(r, 0.025)),
                    "hi": float(np.quantile(r, 0.975)), "ref": ref,
                    "length_ratio": float(N[n].sum() / N[ref].sum()),
                    "median_prompt_speed_ratio": float(np.median(per_prompt)),
                    "identical_subset": len(same),
                    "identical_subset_ratio": (float((N[n][same].sum() / T[n][same].sum())
                                                     / (N[ref][same].sum() / T[ref][same].sum())) if same else None),
                }
            if args.identity_ref and n != args.identity_ref:
                ref_l = lanes[args.identity_ref]
                diffs = [first_diff(ref_l[k]["ids"], lanes[n][k]["ids"]) for k in keys]
                bad = [d for d in diffs if d is not None]
                row["identical"] = len(keys) - len(bad)
                # strict: every repeat of this lane equals every repeat of the reference
                row["identical_all_repeats"] = sum(
                    all(a == b for a in lanes[n][k]["runs"] for b in ref_l[k]["runs"]) for k in keys)
                row["first_diff_median"] = float(np.median(bad)) if bad else None
            rows[n] = row
            line = f"  {n:16s} {thr[n]:8.1f} tok/s  reps {reps[n]}  repro {row['reproducible']}/{len(keys)}"
            for ref_name in ("vs_" + args.baseline, "vs_best"):
                if ref_name in row:
                    v = row[ref_name]
                    line += (f"  {ref_name}({v['ref']}) {v['ratio']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
                             f" len {v['length_ratio']:.3f} med {v['median_prompt_speed_ratio']:.3f}")
                    if v["identical_subset_ratio"] is not None:
                        line += f" same-output({v['identical_subset']}) {v['identical_subset_ratio']:.3f}"
            if "identical" in row:
                line += (f"  identical to {args.identity_ref} {row['identical']}/{len(keys)}"
                         f" (all repeats {row['identical_all_repeats']})")
                if row["first_diff_median"] is not None:
                    line += f" first diff median {row['first_diff_median']:.0f}"
            print(line)
        report[fam] = {"prompts": len(keys), "best_baseline": best, "lanes": rows}
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
