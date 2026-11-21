"""Which (GPU, target, route) combinations have exact-output evidence. Data only: no SGLang, no GPU.

`AX_QUALITY_PROFILE=verified` starts only on a row that admits it. A row records what was
measured, so a launch outside the measured conditions is visible in the launcher's output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Passed the release gate: at least 1,000 greedy sequences, concurrency 1, 2 and 8, 128 and 1,024 tokens.
EXACT_VERIFIED = "exact_verified"
# Every measured sequence equalled the reference, on fewer sequences than the release gate asks.
EXACT_MEASURED = "exact_measured"
EXPERIMENTAL = "experimental"
UNSUPPORTED = "unsupported"

QWEN38_27B_NVFP4 = "qwen3.8-27b-nvfp4"
ROUTES = ("ar", "tree", "chain")


@dataclass(frozen=True)
class Qualification:
    status: str
    reason: str
    evidence: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()

    @property
    def admits_verified(self) -> bool:
        return self.status in (EXACT_VERIFIED, EXACT_MEASURED)

    def as_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "evidence": list(self.evidence),
                "conditions": list(self.conditions)}


_EVIDENCE = ("docs/reports/2026-11-21-verified-profile-gate.md",
             "results/rtx_5090_27b_nvfp4/REPORT.md#fp8-parity-retest",
             "docs/reports/2026-11-20-rtx-nvfp4-parity-forced-128.md")
_CONDITIONS = ("each prompt fits one prefill chunk (set to 4,096 tokens; a 512-token chunk changed a "
               "2,266-token prompt's first token)",
               "greedy requests; 10 distinct prompts of 23 to 2,266 tokens were measured")

# RTX PRO 6000, 2026-11-21, launched with the profile alone: every sequence equal to AR at concurrency 1.
_SM120_27B = Qualification(
    EXACT_MEASURED,
    "RTX PRO 6000: 1,000 of 1,000 sequences of 128 tokens and 120 of 120 of 1,024 tokens equal to "
    "serialized AR, at concurrency 1, 2 and 8. RTX 5090: 100 of 100 at concurrency 1 and 2, 128 tokens. "
    "Below the release gate: 10 distinct prompts, and 120 sequences of 1,024 tokens",
    _EVIDENCE, _CONDITIONS)

# The tree route (one running request) left the AR sequence after 329 and 492 tokens on 2 of 10 prompts.
_SM120_27B_TREE = Qualification(
    EXPERIMENTAL,
    "tree rounds equalled AR for 128 tokens (1,000 of 1,000 sequences) but not for 1,024: 2 of 10 "
    "prompts differed from token 329 and 492, the same way in both passes (RTX PRO 6000). "
    "Use the chain route: AX_TREE=0 with --speculative-num-draft-tokens 8",
    _EVIDENCE, _CONDITIONS)

# (CUDA compute capability, target, route) -> measured qualification.
ROWS: Mapping[tuple[tuple[int, int], str, str], Qualification] = {
    ((12, 0), QWEN38_27B_NVFP4, route): _SM120_27B_TREE if route == "tree" else _SM120_27B for route in ROUTES
}


def target_key(target: Mapping | None, quant: Mapping | None = None) -> str | None:
    """Name of a target with qualification rows, from its config.json (and hf_quant_config.json)."""
    from .defaults import gdn_shape, is_nvfp4

    target = target or {}
    if ("Qwen3_5ForConditionalGeneration" in (target.get("architectures") or ())
            and gdn_shape(target) == (48, 48) and is_nvfp4(target, quant)):
        return QWEN38_27B_NVFP4
    return None


def lookup(capability: tuple[int, int] | None, target: str | None, route: str) -> Qualification:
    if capability is None:
        return Qualification(UNSUPPORTED, "no CUDA GPU was detected")
    if target is None:
        return Qualification(UNSUPPORTED, "the target is not a measured checkpoint "
                                          "(measured: Qwen3.8-27B NVFP4), or its config.json could not be read")
    row = ROWS.get((tuple(capability), target, route))
    if row is None:
        return Qualification(UNSUPPORTED, f"no exact-output measurement for {target} on "
                                          f"SM{capability[0]}{capability[1]} ({route})")
    return row
