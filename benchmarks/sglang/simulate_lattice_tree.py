#!/usr/bin/env python3
"""Price DFlash2 draft trees offline from an `AX_LATTICE_TRACE` recording.

The server, launched with `AX_LATTICE=1 AX_LATTICE_TRACE=<file>`, writes one JSON line per request and
decode round: the anchor, the round's verified tokens (`tokens`, anchor first), how many drafts were
accepted (`accept`) and, when the draft graph ran, the drafter's lattice (`cand`, `lat`, `tail`, see
`ax_engine_sglang/ax_lattice.py:trace_rows`). For greedy requests the concatenation of
`tokens[:accept + 1]` over a request's rounds is its output after the first token, and the target's
continuation of any anchor is that output. Each recorded round is scored at its own anchor
("paired" mode): a policy commits its accepted drafts plus the target's own token.

Policies (`--policy`, repeatable):

* `chain:W`   the greedy walk over the lattice, cut to W verify rows (anchor + W - 1 drafts). With
              W = positions + 1 it is the drafted chain of a full-width chain round; the tool checks
              that against the recorded drafts.
* `tree:N[:C][:notail][:spine]`  best-first tree of N rows (anchor included): a node's children are
              the next position's candidates scored by the log-softmax of its lattice row, with the
              mass outside the top-k as one more entry unless `notail`; a child's path score is its
              parent's plus its own; at most C children per node (default: all); `spine` first
              reserves the greedy chain (`FC1_CHAIN_FIRST=1`). Without a cap and with the tail this is
              the set `tree_ops.beam_tree` selects for `ax_lattice` (up to ties); with `4:notail` it
              is the rule of ax-engine's Metal tree builder. Keyed options shape the search as the
              plugin's AX_TREE_* settings do: `t=T` temperature, `b=B` per-branch penalty, `l=L` at
              most L branches off the argmax along a path (e.g. `tree:16:t=1.2:b=0.5:l=4`).
* `oracle[:N]` leading positions whose true token is among the position's candidates, at most N - 1
              (an upper bound for any tree of N rows drawn from this lattice; unbounded without N).

Before trusting other shapes, the tool replays the logged route: for every full-width chain round the
greedy walk must equal the drafted chain and give the recorded accept length, and for every tree
round `tree:<nodes>` must give the recorded accept length (`checks` in the summary).

Biases: rounds are the traced route's anchors, not each policy's own; a round whose continuation is
shorter than the lattice (end of output) is excluded unless `--include-cut`; NVFP4 near-ties depend on
the batch, so the continuation is that of the traced run. Sampled requests break the premise (their
continuation is not the target's argmax) and must not be traced.

    python benchmarks/sglang/simulate_lattice_tree.py trace.jsonl \\
        --policy chain:16 --policy chain:8 --policy tree:16 --policy tree:8 --policy oracle \\
        --cost chain:16=24.0 --cost tree:16=25.1 --out summary.json

Deterministic, stdlib only.
"""
from __future__ import annotations

import argparse
import base64
import heapq
import json
import math
import struct
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path


class TraceError(ValueError):
    """A trace that cannot be decoded or does not match its own rounds."""


@dataclass(frozen=True)
class Lattice:
    positions: int
    top_k: int
    cand: tuple          # [positions][top_k] token ids
    lat: tuple           # [positions][top_k][top_k] scores
    tail: tuple          # [positions] log of the mass outside the top-k

    def row(self, position, parent):
        """Scores of the candidates at `position` after candidate `parent` of the previous position
        (row 0 at position 0: after the anchor)."""
        return self.lat[position][parent if position else 0]


def _decode(text, fmt, count):
    raw = base64.b64decode(text)
    size = struct.calcsize("<" + fmt)
    if len(raw) != size * count:
        raise TraceError(f"expected {count} values of {fmt!r}, got {len(raw)} bytes")
    return struct.unpack(f"<{count}{fmt}", raw)


def decode_lattice(row):
    if "cand" not in row:
        return None
    p, k = int(row["positions"]), int(row["top_k"])
    cand = _decode(row["cand"], "i", p * k)
    lat = _decode(row["lat"], "f", p * k * k)
    tail = _decode(row["tail"], "f", p)
    return Lattice(p, k, tuple(tuple(cand[s * k:(s + 1) * k]) for s in range(p)),
                   tuple(tuple(tuple(lat[(s * k + i) * k:(s * k + i + 1) * k]) for i in range(k)) for s in range(p)),
                   tuple(tail))


def log_softmax(scores, tail=None):
    values = list(scores) + ([tail] if tail is not None else [])
    top = max(values)
    if top == -math.inf:        # an all-masked row: every child is impossible
        return [-math.inf for _ in scores]
    norm = top + math.log(sum(math.exp(v - top) for v in values))
    return [v - norm for v in scores]


def argmax(values):
    best = 0
    for i, v in enumerate(values):
        if v > values[best]:
            best = i
    return best


def greedy_walk(lattice):
    """Candidate indices of the greedy path over the lattice (the selector's greedy draft)."""
    path, parent = [], 0
    for s in range(lattice.positions):
        parent = argmax(lattice.row(s, parent))
        path.append(parent)
    return path


def child_scores(lattice, position, parent, tail=True, temp=1.0, penalty=0.0):
    """`ax_lattice.tree_scores` of one lattice row: log-softmax at temperature `temp` (with the tail
    mass unless `tail` is off), minus `penalty` for every child that is not the row's argmax."""
    row = [v / temp for v in lattice.row(position, parent)]
    logp = log_softmax(row, lattice.tail[position] / temp if tail else None)
    if penalty:
        top = argmax(logp)
        logp = [v if j == top else v - penalty for j, v in enumerate(logp)]
    return logp


def build_tree(lattice, rows, children=None, tail=True, spine=False, temp=1.0, penalty=0.0, lmax=None):
    """Best-first tree of `rows` rows (anchor included). Returns nodes as (parent node, position,
    candidate index); node 0 is the anchor (parent -1, position -1). `lmax` bounds the branches off
    a row's argmax along any path (`tree_ops.beam_tree`'s lmax; unbounded by default)."""
    nodes = [(-1, -1, 0)]
    scores = [0.0]
    branches = [0]
    frontier = []
    counter = 0

    def child_logp(node):
        _, position, cand = nodes[node]
        nxt = position + 1
        if nxt >= lattice.positions:
            return nxt, []
        logp = child_scores(lattice, nxt, cand, tail, temp, penalty)
        order = sorted(range(lattice.top_k), key=lambda j: (-logp[j], j))
        top = order[0]
        options = [(j, logp[j], branches[node] + (j != top)) for j in order]
        if lmax is not None:
            options = [o for o in options if o[2] <= lmax]
        return nxt, options if children is None else options[:children]

    def add(parent, position, cand, score, branch=0):
        nodes.append((parent, position, cand))
        scores.append(score)
        branches.append(branch)

    if spine:
        # FC1_CHAIN_FIRST: the greedy chain is selected first, as far as the rows allow.
        for position, cand in enumerate(greedy_walk(lattice)[:rows - 1]):
            parent = len(nodes) - 1
            option = {j: (logp, branch) for j, logp, branch in child_logp(parent)[1]}.get(cand, (-math.inf, 0))
            add(parent, position, cand, scores[parent] + option[0], option[1])
    reserved = {(nodes[i][0], nodes[i][2]) for i in range(1, len(nodes))}

    def expand(node):
        nonlocal counter
        nxt, options = child_logp(node)
        for j, logp, branch in options:
            if (node, j) in reserved:
                continue
            # ties: earlier insertion first, as a stable top-k would keep them
            heapq.heappush(frontier, (-(scores[node] + logp), counter, node, nxt, j, branch))
            counter += 1

    for node in range(len(nodes)):
        expand(node)
    while len(nodes) < rows and frontier:
        negative, _, parent, position, cand, branch = heapq.heappop(frontier)
        add(parent, position, cand, -negative, branch)
        expand(len(nodes) - 1)
    return nodes


def tree_accept(lattice, nodes, truth):
    """Drafts accepted when the target's continuation is `truth`."""
    children = defaultdict(list)
    for index, (parent, position, cand) in enumerate(nodes):
        if parent >= 0:
            children[parent].append(index)
    node, accepted = 0, 0
    while accepted < len(truth):
        match = next((c for c in children[node]
                      if lattice.cand[nodes[c][1]][nodes[c][2]] == truth[accepted]), None)
        if match is None:
            break
        node, accepted = match, accepted + 1
    return accepted


def chain_accept(lattice, width, truth):
    accepted = 0
    for s, cand in enumerate(greedy_walk(lattice)[:width - 1]):
        if accepted >= len(truth) or lattice.cand[s][cand] != truth[accepted]:
            break
        accepted += 1
    return accepted


def oracle_accept(lattice, truth, rows=None):
    limit = min(len(truth), lattice.positions, rows - 1 if rows else lattice.positions)
    accepted = 0
    while accepted < limit and truth[accepted] in lattice.cand[accepted]:
        accepted += 1
    return accepted


def parse_policy(text):
    parts = text.split(":")
    try:
        if parts[0] == "oracle" and len(parts) <= 2:
            rows = int(parts[1]) if len(parts) == 2 else None
            if rows is not None and rows < 2:
                raise ValueError
            return {"name": text, "kind": "oracle", "rows": rows}
        if parts[0] == "chain" and len(parts) == 2 and int(parts[1]) >= 2:
            return {"name": text, "kind": "chain", "rows": int(parts[1])}
        if parts[0] == "tree" and len(parts) >= 2 and int(parts[1]) >= 2:
            extra = parts[2:]
            flags = [x for x in extra if x in ("notail", "spine")]
            keyed = dict(x.split("=", 1) for x in extra if "=" in x)
            caps = [int(x) for x in extra if x not in ("notail", "spine") and "=" not in x]
            if (len(caps) > 1 or any(c < 1 for c in caps) or len(set(flags)) != len(flags)
                    or len(keyed) != sum("=" in x for x in extra) or set(keyed) - {"t", "b", "l"}):
                raise ValueError
            temp, penalty = float(keyed.get("t", 1.0)), float(keyed.get("b", 0.0))
            lmax = int(keyed["l"]) if "l" in keyed else None
            if not (temp > 0 and math.isfinite(temp)) or not (penalty >= 0 and math.isfinite(penalty)) \
                    or (lmax is not None and lmax < 1):
                raise ValueError
            return {"name": text, "kind": "tree", "rows": int(parts[1]), "children": caps[0] if caps else None,
                    "tail": "notail" not in flags, "spine": "spine" in flags, "temp": temp, "penalty": penalty,
                    "lmax": lmax}
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"policy {text!r}: use chain:W, tree:N[:C][:notail][:spine][:t=T][:b=B][:l=L] "
                                     "or oracle[:N]")


def score(policy, lattice, truth):
    if policy["kind"] == "chain":
        return chain_accept(lattice, policy["rows"], truth)
    if policy["kind"] == "oracle":
        return oracle_accept(lattice, truth, policy["rows"])
    nodes = build_tree(lattice, policy["rows"], policy["children"], policy["tail"], policy["spine"],
                       policy.get("temp", 1.0), policy.get("penalty", 0.0), policy.get("lmax"))
    return tree_accept(lattice, nodes, truth)


def load_requests(paths):
    """Rows grouped by request in file order, with each round's offset into the request's output."""
    requests = OrderedDict()
    for path in paths:
        with open(path) as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TraceError(f"{path}:{number}: {error}") from None
                key = (str(path), row["rid"])
                requests.setdefault(key, []).append(row)
    for rows in requests.values():
        committed = []
        for row in rows:
            tokens, accept = row["tokens"], int(row["accept"])
            if accept < 0 or accept + 1 > len(tokens) or tokens[0] != row["anchor"]:
                raise TraceError(f"request {row['rid']} round {row['round']}: anchor or accept length out of range")
            row["offset"] = len(committed)
            committed.extend(tokens[:accept + 1])
        for row in rows:
            row["continuation"] = committed[row["offset"] + 1:]
    return requests


def simulate(requests, policies, include_cut=False):
    totals = {p["name"]: {"rounds": 0, "accepted": 0} for p in policies}
    by_bs = defaultdict(lambda: {p["name"]: {"rounds": 0, "accepted": 0} for p in policies})
    checks = {"chain_rounds_checked": 0, "chain_draft_mismatch": 0, "recorded_accept_mismatch": 0,
              "tree_rounds_checked": 0, "tree_accept_mismatch": 0,
              "rounds": 0, "without_lattice": 0, "cut": 0}
    for rows in requests.values():
        for row in rows:
            checks["rounds"] += 1
            lattice = decode_lattice(row)
            if lattice is None:
                checks["without_lattice"] += 1
                continue
            truth = row["continuation"][:lattice.positions]
            cut = len(truth) < lattice.positions
            width = row.get("width")
            drafts_match = True
            if row["kind"] == "chain" and width == lattice.positions + 1:
                # The drafted chain needs no continuation: checked on every round, cut or not.
                checks["chain_rounds_checked"] += 1
                walk = [lattice.cand[s][c] for s, c in enumerate(greedy_walk(lattice))]
                drafts_match = walk == list(row["tokens"][1:width])
                checks["chain_draft_mismatch"] += not drafts_match
            if cut and not include_cut:
                # The request's output ends inside this round (its last accepted tokens may be dropped
                # at max_new_tokens), so its accept length cannot be checked or scored.
                checks["cut"] += 1
                continue
            if row["kind"] == "chain" and width == lattice.positions + 1:
                if drafts_match and chain_accept(lattice, width, truth) != int(row["accept"]):
                    checks["recorded_accept_mismatch"] += 1
            elif row["kind"] == "tree" and width:
                checks["tree_rounds_checked"] += 1
                if tree_accept(lattice, build_tree(lattice, int(width)), truth) != int(row["accept"]):
                    checks["tree_accept_mismatch"] += 1
            for policy in policies:
                accepted = score(policy, lattice, truth)
                for bucket in (totals[policy["name"]], by_bs[int(row["bs"])][policy["name"]]):
                    bucket["rounds"] += 1
                    bucket["accepted"] += accepted
    def finish(table):
        return {name: {**v, "tokens_per_round": (v["accepted"] + v["rounds"]) / v["rounds"] if v["rounds"] else None}
                for name, v in table.items()}
    return {"policies": finish(totals), "by_batch_size": {bs: finish(t) for bs, t in sorted(by_bs.items())},
            "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--policy", action="append", type=parse_policy, required=True)
    parser.add_argument("--cost", action="append", default=[], metavar="POLICY=MS",
                        help="measured round time of a policy, to project output tokens per second")
    parser.add_argument("--include-cut", action="store_true",
                        help="also score rounds whose recorded continuation is shorter than the lattice")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    costs = {}
    for item in args.cost:
        name, _, value = item.partition("=")
        try:
            costs[name] = float(value)
        except ValueError:
            parser.error(f"--cost {item!r}: use POLICY=MS")
        if costs[name] <= 0:
            parser.error(f"--cost {item!r}: the round time must be positive")
    if len({p["name"] for p in args.policy}) != len(args.policy):
        parser.error("each --policy once")
    unknown = set(costs) - {p["name"] for p in args.policy}
    if unknown:
        parser.error(f"--cost names policies that are not simulated: {sorted(unknown)}")
    try:
        summary = simulate(load_requests(args.trace), args.policy, args.include_cut)
    except TraceError as error:
        parser.exit(2, f"simulate_lattice_tree: {error}\n")
    reference = args.policy[0]["name"]
    base = summary["policies"][reference]["tokens_per_round"]
    for name, row in summary["policies"].items():
        row["vs_" + reference] = row["tokens_per_round"] / base if base and row["tokens_per_round"] else None
        if name in costs and row["tokens_per_round"]:
            row["round_ms"] = costs[name]
            row["projected_tok_s_per_request"] = row["tokens_per_round"] / costs[name] * 1000.0
    checks = summary["checks"]
    print(f"rounds {checks['rounds']}, without lattice {checks['without_lattice']}, cut {checks['cut']}, "
          f"chain drafts checked {checks['chain_rounds_checked']} (mismatch {checks['chain_draft_mismatch']}, "
          f"accept mismatch {checks['recorded_accept_mismatch']}), tree rounds checked "
          f"{checks['tree_rounds_checked']} (accept mismatch {checks['tree_accept_mismatch']})")
    print(f"{'policy':<20}{'rounds':>8}{'tokens/round':>14}{'vs ' + reference:>16}{'tok/s':>10}")
    for name, row in summary["policies"].items():
        ratio = row["vs_" + reference]
        projected = row.get("projected_tok_s_per_request")
        print(f"{name:<20}{row['rounds']:>8}{row['tokens_per_round'] or 0:>14.3f}"
              f"{ratio or 0:>16.3f}{projected if projected is not None else float('nan'):>10.1f}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
