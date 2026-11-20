"""Compare GPU kernel time between two SGLang torch-profiler traces (Chrome trace JSON, .json.gz).

  python benchmarks/sglang/trace_kernels.py A_DIR B_DIR [--top 40]

Each directory holds the trace(s) written by POST /start_profile (TORCH_PROFILE_STEPS in
benchmarks/sglang/run_baselines.sh). Kernel durations are summed by kernel name (shapes stripped) and divided by
the number of profiled scheduler steps, so the two sides compare per step even if their step counts
differ. Prints total kernel time per step, kernel launches per step, and the kernels with the largest
per-step difference. Kernel time excludes gaps between kernels (launch latency, host work), which
shows up as the difference between this total and the measured round time.
"""
import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path


def load(d):
    files = sorted(Path(d).rglob("*.json.gz")) + sorted(Path(d).rglob("*.trace.json"))
    if not files:
        raise SystemExit(f"no trace in {d}")
    events = []
    for f in files:
        op = gzip.open if f.suffix == ".gz" else open
        with op(f, "rt") as fh:
            events += json.load(fh).get("traceEvents", [])
    return files, events


def summarize(events):
    kern = defaultdict(lambda: [0, 0.0])
    steps = 0
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, name = e.get("cat", ""), e.get("name", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            key = re.sub(r"<.*", "", name)[:90]
            kern[key][0] += 1
            kern[key][1] += float(e.get("dur", 0))
        elif cat == "user_annotation" and re.match(r"(?i)(profiler)?step", name):
            steps += 1
    return kern, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--steps", type=int, default=0, help="override the step count of both traces")
    args = ap.parse_args()
    out = []
    for d in (args.a, args.b):
        files, ev = load(d)
        k, steps = summarize(ev)
        steps = args.steps or steps or 1
        out.append((d, k, steps))
        tot = sum(v[1] for v in k.values()) / steps
        n = sum(v[0] for v in k.values()) / steps
        print(f"{d}: {len(files)} file(s), {steps} steps, kernel time {tot / 1e3:.3f} ms/step, {n:.0f} kernels/step")
    (_, ka, sa), (_, kb, sb) = out
    names = set(ka) | set(kb)
    rows = []
    for n in names:
        a = ka.get(n, [0, 0.0]); b = kb.get(n, [0, 0.0])
        rows.append((a[1] / sa - b[1] / sb, n, a[0] / sa, a[1] / sa, b[0] / sb, b[1] / sb))
    rows.sort(key=lambda r: -abs(r[0]))
    print(f"\n{'diff us/step':>12} {'A n/step':>9} {'A us/step':>10} {'B n/step':>9} {'B us/step':>10}  kernel")
    for d, n, an, au, bn, bu in rows[:args.top]:
        print(f"{d:12.1f} {an:9.2f} {au:10.1f} {bn:9.2f} {bu:10.1f}  {n}")


if __name__ == "__main__":
    main()
