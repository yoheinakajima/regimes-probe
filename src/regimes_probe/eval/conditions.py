"""Same-conditions enforcement for a valid comparison.

The Level 1 headline ("does frozen policy memory help?") is only meaningful if
the two compared conditions are identical in EVERY respect except memory access.
This module captures a :class:`ConditionSpec` per run and asserts that the only
difference between a baseline and a treatment is the intended one (memory
access). See ``docs/METHODOLOGY_RISKS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConditionSpec:
    """The fingerprint of a run condition, used for same-conditions checks."""

    answer_model: str
    answer_prompt_version: str
    enabled_tools: tuple[str, ...]
    tool_budget: int
    split_id: str               # stable hash of the OPTIMIZE/CONFIRM split
    grader: str
    provider_config_id: str     # hash of provider configs (excl. memory)
    query_policy: str
    verify_policy: str
    stop_policy: str
    memory_access: str          # "none" | "frozen_snapshot" | "random" | "online"
    #: LLM frontier proposer settings (mode + model + prompt) — must be IDENTICAL
    #: across compared conditions (the only intended difference is memory access).
    llm_frontier_settings: str = ""

    #: Fields compared by :func:`same_conditions`. ``memory_access`` is compared
    #: too, but it is the *intended* difference for the Level 1 headline.
    COMPARED = (
        "answer_model", "answer_prompt_version", "enabled_tools", "tool_budget",
        "split_id", "grader", "provider_config_id", "query_policy",
        "verify_policy", "stop_policy", "llm_frontier_settings", "memory_access",
    )

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.COMPARED}
        d["enabled_tools"] = list(self.enabled_tools)
        return d


@dataclass
class SameConditionsResult:
    ok: bool
    diffs: list[str]
    unexpected_diffs: list[str]
    intended_diffs: list[str]
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "diffs": self.diffs,
            "unexpected_diffs": self.unexpected_diffs,
            "intended_diffs": self.intended_diffs,
            "detail": self.detail,
        }


def same_conditions(
    baseline: ConditionSpec,
    treatment: ConditionSpec,
    *,
    intended_diffs: tuple[str, ...] = ("memory_access",),
) -> SameConditionsResult:
    """Assert baseline and treatment differ only in the intended field(s).

    For the Level 1 headline, ``intended_diffs == ("memory_access",)``. For a
    Level 2 ablation that deliberately varies, e.g., the query policy, pass
    ``intended_diffs=("memory_access", "query_policy")``.
    """
    diffs: list[str] = []
    detail: dict[str, Any] = {}
    for field_name in ConditionSpec.COMPARED:
        b, t = getattr(baseline, field_name), getattr(treatment, field_name)
        if b != t:
            diffs.append(field_name)
            detail[field_name] = {"baseline": b, "treatment": t}
    unexpected = [d for d in diffs if d not in intended_diffs]
    return SameConditionsResult(
        ok=(len(unexpected) == 0),
        diffs=diffs,
        unexpected_diffs=unexpected,
        intended_diffs=list(intended_diffs),
        detail=detail,
    )
