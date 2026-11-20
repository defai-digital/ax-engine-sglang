"""Host-side speculative commit oracle and benchmark evidence records.

GPU backends own verification and rollback; this oracle consumes already chosen
target tokens. It does not replace SGLang's greedy or sampled verifier.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import Sequence


@dataclass(frozen=True)
class GreedyCommit:
    accepted_drafts: int
    emitted: list[int]
    retained_draft_rows: int


def plan_greedy_commit(drafts: Sequence[int], predicted: Sequence[int], budget: int,
                       eos: Sequence[int]) -> GreedyCommit:
    if len(predicted) != len(drafts) + 1:
        raise ValueError("prediction_count")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    accepted = 0
    for draft, target in zip(drafts, predicted):
        if draft != target:
            break
        accepted += 1
    verified = [*drafts[:accepted], predicted[accepted]]
    emitted = []
    for token in verified[:budget]:
        emitted.append(token)
        if token in eos:
            break
    return GreedyCommit(accepted, emitted, min(accepted, len(emitted)))


def benchmark_record(*, rows, wall_seconds, server, model, concurrency, max_tokens,
                     diagnostic, accelerator=None):
    """Measured workload wall time, not the sum of concurrent request latencies.

    Unknown pack revisions, precision and hardware stay null. A run without a
    diagnostic launch record stays unclassified instead of becoming a speed claim.
    """
    if not rows or wall_seconds <= 0:
        raise ValueError("benchmark needs completed rows and positive workload wall time")
    prompts = [(row["set"], row["i"], row["prompt_sha1"]) for row in rows]
    return {
        "schema_version": "ax.engine.benchmark.v1",
        "engine": "sglang",
        "engine_version": server.get("version"),
        "model": {"id": server.get("model_path") or model, "revision": server.get("revision")},
        "drafter": {"id": server.get("speculative_draft_model_path"), "revision": None},
        "precision": server.get("quantization") or server.get("dtype"),
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None},
        "workload": {
            "prompt_sha256": hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest(),
            "requests": len(rows), "concurrency": concurrency, "max_new_tokens": max_tokens,
        },
        "hardware": {"accelerator": accelerator, "memory_bytes": None},
        "measurement": {
            "kind": "unclassified" if diagnostic is None else "diagnostic" if diagnostic else "measured",
            "output_tokens": sum(row["n_out"] for row in rows), "wall_seconds": wall_seconds,
        },
        "speculation": {"drafted_tokens": None, "accepted_draft_tokens": None, "verify_calls": None},
    }
