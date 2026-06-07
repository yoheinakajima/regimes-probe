"""Level 2: stopping / continuation policy.

Decides, at each step, whether to stop or which continuation action to take.
Stop policy arms: ``stop_now``, ``search_more``, ``fetch_page``, ``corroborate``,
``contradiction_check``. Three modes the benchmark distinguishes:

  * ``always_full``     — never stop early; spend the whole budget.
  * ``first_candidate`` — stop the instant a candidate answer appears.
  * ``learned``         — a contextual bandit over the arms, with the
                          verification state folded into the context.

False-stop and over-search errors are derived downstream (see
``docs/VERIFICATION_AND_STOPPING.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.policy.contextual_bandit import ContextualBandit
from regimes_probe.policy.verification_policy import VerificationState

STOP_ARMS: tuple[str, ...] = (
    "stop_now",
    "search_more",
    "fetch_page",
    "corroborate",
    "contradiction_check",
)

#: Which arms terminate the loop.
TERMINAL_ARMS = frozenset({"stop_now"})


@dataclass
class StopConfig:
    """Mutable thresholds (regimes loop may tune these)."""

    stop_threshold: float = 0.6     # verification score at/above which we may stop
    min_calls_before_stop: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_threshold": self.stop_threshold,
            "min_calls_before_stop": self.min_calls_before_stop,
        }


@dataclass
class StopDecision:
    arm: str
    stop: bool
    explanation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "stop": self.stop, "explanation": self.explanation}


class StoppingPolicy:
    def __init__(self, mode: str = "learned", config: Optional[StopConfig] = None) -> None:
        assert mode in ("always_full", "first_candidate", "learned")
        self.mode = mode
        self.config = config or StopConfig()

    def stop_context_key(self, cluster_key: str, vstate: VerificationState) -> str:
        """Stopping context = signature cluster + verification-state code."""
        return f"{cluster_key}|{vstate.context_code()}"

    def decide(
        self,
        cluster_key: str,
        vstate: VerificationState,
        *,
        calls_used: int,
        budget: int,
        fetch_available: bool,
        bandit: Optional[ContextualBandit] = None,
        neighbor: Optional[dict[str, tuple[float, float]]] = None,
        explore: bool = False,
        salt: str = "",
    ) -> StopDecision:
        budget_exhausted = calls_used >= budget
        if budget_exhausted:
            return StopDecision("stop_now", True, {"reason": "budget_exhausted"})

        if self.mode == "always_full":
            arm = "fetch_page" if (fetch_available and vstate.candidate_found
                                   and not vstate.support_found) else "search_more"
            return StopDecision(arm, False, {"reason": "always_full"})

        if self.mode == "first_candidate":
            if vstate.candidate_found and calls_used >= self.config.min_calls_before_stop:
                return StopDecision("stop_now", True, {"reason": "first_candidate"})
            return StopDecision("search_more", False, {"reason": "no_candidate_yet"})

        # learned
        ctx = self.stop_context_key(cluster_key, vstate)
        ranked = None
        if bandit is not None:
            ranked = bandit.choose_tool_plan(
                ctx, list(STOP_ARMS), k=len(STOP_ARMS), explore=explore,
                neighbor=neighbor, salt=salt,
            )
            arm = ranked[0].arm if ranked else "search_more"
        else:
            arm = "stop_now" if vstate.score >= self.config.stop_threshold else "search_more"

        # Guardrails: don't stop below threshold or before min calls; don't
        # fetch if no fetch tool. These keep the learned policy sane in v0.
        if arm == "stop_now" and (
            calls_used < self.config.min_calls_before_stop
            or vstate.score < self.config.stop_threshold
        ):
            arm = "search_more"
        if arm == "fetch_page" and not fetch_available:
            arm = "search_more"

        stop = arm in TERMINAL_ARMS
        expl: dict[str, Any] = {"context": ctx, "vscore": round(vstate.score, 3),
                                "threshold": self.config.stop_threshold}
        if ranked is not None:
            expl["ranked"] = [s.to_dict() for s in ranked]
        return StopDecision(arm, stop, expl)
