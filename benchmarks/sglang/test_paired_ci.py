"""CPU check of benchmarks/sglang/paired_ci.py on synthetic lanes: known ratios, strict input validation.

    python benchmarks/sglang/test_paired_ci.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(path, rows, meta=True, diagnostic=False, max_tokens=100):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if meta:
        Path(path).with_suffix(".meta.json").write_text(json.dumps({
            "complete": True, "diagnostic": diagnostic, "rows": len(rows), "max_tokens": max_tokens,
            "n": 40, "tokenizer": "t", "warmup": 4, "server": {"model_path": "m"},
            "ids_sha1": hashlib.sha1(Path(path).read_bytes()).hexdigest()}))


def lane(speed, n_prompts=40, tokens=100, alter=(), sha=True):
    rows = []
    for fam in ("code", "math"):
        for i in range(n_prompts):
            ids = list(range(tokens))
            if (fam, i) in alter:
                ids[5] = -1
            r = {"set": fam, "i": i, "output_ids": ids, "n_out": tokens, "finish": "length",
                 "wall_s": tokens / speed * (1 + 0.01 * (i % 3))}
            if sha:
                r["prompt_sha1"] = f"{fam}{i}"
            rows.append(r)
    return rows


def run(args, ok=True):
    p = subprocess.run([sys.executable, str(HERE / "paired_ci.py"), *args], capture_output=True, text=True)
    if ok and p.returncode != 0:
        raise AssertionError(p.stderr)
    if not ok and p.returncode == 0:
        raise AssertionError("expected failure:\n" + p.stdout)
    return p


def main():
    d = Path(tempfile.mkdtemp())
    write(d / "ar.jsonl", lane(100))
    write(d / "c8.jsonl", lane(300))
    write(d / "c16.jsonl", lane(400, alter={("code", 1)}))
    write(d / "tree.jsonl", lane(500, alter={("math", 2)}))
    write(d / "tree_b.jsonl", lane(500))
    out = d / "r.json"
    run(["--lane", f"tree={d/'tree.jsonl'},{d/'tree_b.jsonl'}", "--lane", f"c8={d/'c8.jsonl'},{d/'c8.jsonl'}",
         "--lane", f"c16={d/'c16.jsonl'},{d/'c16.jsonl'}", "--lane", f"ar={d/'ar.jsonl'},{d/'ar.jsonl'}",
         "--baseline", "ar", "--best-of", "c8", "c16", "--identity-ref", "ar", "--resamples", "500", "--json-out", str(out)])
    rep = json.load(open(out))
    t = rep["math"]["lanes"]["tree"]
    assert abs(t["vs_best"]["ratio"] - 1.25) < 1e-6, t
    assert t["vs_best"]["lo"] <= 1.25 <= t["vs_best"]["hi"] + 1e-9
    assert abs(t["vs_ar"]["ratio"] - 5.0) < 1e-6
    assert t["identical"] == 39 and t["identical_all_repeats"] == 39 and t["reproducible"] == 39, t
    assert rep["code"]["lanes"]["tree"]["identical"] == 40
    assert rep["code"]["best_baseline"] == "c16"
    assert abs(t["vs_best"]["length_ratio"] - 1.0) < 1e-9
    # strict validation
    write(d / "short.jsonl", lane(300, n_prompts=39))
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'short.jsonl'}", "--baseline", "a"], ok=False)
    write(d / "nosha.jsonl", lane(300, sha=False))
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'nosha.jsonl'}", "--baseline", "a"], ok=False)
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'nosha.jsonl'}", "--baseline", "a", "--allow-missing-hash"])
    run(["--lane", f"a={d/'ar.jsonl'},{d/'ar.jsonl'}", "--lane", f"b={d/'c8.jsonl'}", "--baseline", "a"], ok=False)
    bad = lane(300)
    bad[3]["prompt_sha1"] = "other"
    write(d / "badsha.jsonl", bad)
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'badsha.jsonl'}", "--baseline", "a"], ok=False)
    write(d / "dup.jsonl", lane(300) + lane(300)[:1])
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'dup.jsonl'}", "--baseline", "a"], ok=False)
    # completion sidecar: missing, diagnostic, replaced file, different workload
    write(d / "nometa.jsonl", lane(300), meta=False)
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'nometa.jsonl'}", "--baseline", "a"], ok=False)
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'nometa.jsonl'}", "--baseline", "a", "--legacy"])
    write(d / "diag.jsonl", lane(300), diagnostic=True)
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'diag.jsonl'}", "--baseline", "a"], ok=False)
    write(d / "repl.jsonl", lane(300))
    with open(d / "repl.jsonl", "a") as fh:
        fh.write("")
    Path(d / "repl.jsonl").write_text(Path(d / "repl.jsonl").read_text().replace('"wall_s": 0.3', '"wall_s": 0.1', 1))
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'repl.jsonl'}", "--baseline", "a"], ok=False)
    write(d / "long.jsonl", lane(300), max_tokens=200)
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'long.jsonl'}", "--baseline", "a"], ok=False)
    ab = lane(300)
    ab[4]["finish"] = {"type": "abort"}
    write(d / "abort.jsonl", ab)
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'abort.jsonl'}", "--baseline", "a"], ok=False)
    # same-output subset needs every repeat to agree: repeat 2 of 'x' differs on every math prompt
    x2 = lane(500, alter={("math", i) for i in range(40)})
    write(d / "x2.jsonl", x2)
    run(["--lane", f"x={d/'tree_b.jsonl'},{d/'x2.jsonl'}", "--lane", f"ar={d/'ar.jsonl'},{d/'ar.jsonl'}",
         "--baseline", "ar", "--identity-ref", "ar", "--resamples", "200", "--json-out", str(out)])
    rx = json.load(open(out))["math"]["lanes"]["x"]
    assert rx["vs_ar"]["identical_subset"] == 0 and rx["identical_all_repeats"] == 0 and rx["identical"] == 40, rx
    # provenance: a legacy sidecar whose env shows AX_CHECK is rejected even with --legacy; a sidecar
    # without a server launch record (diagnostic unknown) or without a model is rejected
    write(d / "legdiag.jsonl", lane(300))
    Path(d / "legdiag.meta.json").write_text(json.dumps({"env": {"AX_CHECK": "1"}, "max_tokens": 100, "n": 40,
                                                          "tokenizer": "t"}))
    run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/'legdiag.jsonl'}", "--baseline", "a", "--legacy"], ok=False)
    for name, patch in (("unk", {"diagnostic": None}), ("nomodel", {"server": {}})):
        write(d / f"{name}.jsonl", lane(300))
        m = json.loads(Path(d / f"{name}.meta.json").read_text())
        m.update(patch)
        Path(d / f"{name}.meta.json").write_text(json.dumps(m))
        run(["--lane", f"a={d/'ar.jsonl'}", "--lane", f"b={d/(name + '.jsonl')}", "--baseline", "a"], ok=False)
    # two different configurations passed as repeats of one lane
    write(d / "cfg2.jsonl", lane(500))
    m = json.loads(Path(d / "cfg2.meta.json").read_text())
    m["server"]["speculative_num_draft_tokens"] = 64
    Path(d / "cfg2.meta.json").write_text(json.dumps(m))
    run(["--lane", f"a={d/'ar.jsonl'},{d/'ar.jsonl'}", "--lane", f"b={d/'tree_b.jsonl'},{d/'cfg2.jsonl'}",
         "--baseline", "a"], ok=False)
    print("paired_ci tests passed")


if __name__ == "__main__":
    main()
