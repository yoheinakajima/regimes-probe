"""OPTIMIZE and CONFIRM gates for promoting a policy update.

Mirrors the discipline of ``yoheinakajima/regimes``:
  * OPTIMIZE is in-sample and is NOT the headline — it only gates whether an
    update is worth confirming.
  * CONFIRM is held out — an update is promoted only if it improves (or does not
    regress beyond a threshold) on CONFIRM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GateResult:
    passed: bool
    before: float
    after: float
    delta: float
    threshold: float
    metric: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "metric": self.metric,
            "before": round(self.before, 6),
            "after": round(self.after, 6),
            "delta": round(self.delta, 6),
            "threshold": self.threshold,
            "passed": self.passed,
        }


def optimize_gate(before: float, after: float, *, min_delta: float = 0.01,
                  metric: str = "correct_per_tool_call") -> GateResult:
    """In-sample gate: require a minimum improvement to bother confirming."""
    delta = after - before
    return GateResult(delta >= min_delta, before, after, delta, min_delta, metric, "optimize")


def confirm_gate(before: float, after: float, *, max_regression: float = 0.0,
                 metric: str = "correct_per_tool_call") -> GateResult:
    """Held-out gate: promote if the update does not regress beyond threshold.

    ``max_regression`` is the largest tolerated drop (e.g. 0.0 = no regression;
    0.005 = tolerate a tiny drop). Passing means after >= before - max_regression.
    """
    delta = after - before
    return GateResult(delta >= -max_regression, before, after, delta, max_regression, metric, "confirm")


@dataclass
class PromotionDecision:
    accepted: bool
    optimize_gate: GateResult
    confirm_gate: GateResult
    update: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "optimize_gate": self.optimize_gate.to_dict(),
            "confirm_gate": self.confirm_gate.to_dict(),
            "update": self.update,
        }
