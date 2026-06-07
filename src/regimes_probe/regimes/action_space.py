"""Bounded policy-mutation space for the regimes improvement loop.

v0 mutates ONLY safe numeric policy parameters (and a couple of medium-risk
thresholds). It never mutates prompts or code (see
``docs/REGIMES_IMPROVEMENT_LOOP.md``). Every mutation is bounded.

The :class:`PolicyConfigBundle` is the full set of tunable knobs; a
:class:`Mutation` describes one bounded change to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.router import RouterConfig
from regimes_probe.policy.stopping_policy import StopConfig
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.eval.reward import RewardWeights


@dataclass
class PolicyConfigBundle:
    bandit: BanditParams = field(default_factory=BanditParams)
    router: RouterConfig = field(default_factory=RouterConfig)
    stop: StopConfig = field(default_factory=StopConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    nearest_k: int = 8

    def copy(self) -> "PolicyConfigBundle":
        return PolicyConfigBundle(
            bandit=self.bandit.copy(),
            router=replace(self.router),
            stop=replace(self.stop),
            verification=replace(self.verification),
            reward=RewardWeights.from_dict(self.reward.to_dict()),
            nearest_k=self.nearest_k,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bandit": self.bandit.to_dict(),
            "router": self.router.to_dict(),
            "stop": self.stop.to_dict(),
            "verification": self.verification.to_dict(),
            "reward": self.reward.to_dict(),
            "nearest_k": self.nearest_k,
        }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Mutation:
    """One bounded change to a :class:`PolicyConfigBundle`."""

    target: str                       # e.g. "stop.stop_threshold"
    delta: float
    bounds: tuple[float, float]
    rationale: str = ""
    tier: str = "safe"                # safe | medium | high

    def apply(self, bundle: PolicyConfigBundle) -> tuple[PolicyConfigBundle, dict[str, Any]]:
        out = bundle.copy()
        obj_name, attr = self.target.split(".", 1)
        obj = getattr(out, obj_name)
        old = getattr(obj, attr)
        new = _clamp(float(old) + self.delta, *self.bounds)
        if attr == "nearest_k" or obj_name == "nearest_k":
            new = int(round(new))
        setattr(obj, attr, new)
        record = {"target": self.target, "old": old, "new": new,
                  "delta": self.delta, "tier": self.tier, "rationale": self.rationale}
        return out, record


# nearest_k lives on the bundle itself, not a sub-config — handle specially.
@dataclass
class NearestKMutation(Mutation):
    def apply(self, bundle: PolicyConfigBundle) -> tuple[PolicyConfigBundle, dict[str, Any]]:
        out = bundle.copy()
        old = out.nearest_k
        new = int(round(_clamp(old + self.delta, *self.bounds)))
        out.nearest_k = new
        return out, {"target": "nearest_k", "old": old, "new": new,
                     "delta": self.delta, "tier": self.tier, "rationale": self.rationale}


#: Catalogue of safe numeric mutations the regimes loop may draw from.
def safe_mutations() -> dict[str, Callable[[float, str], Mutation]]:
    return {
        "reward.freshness": lambda d, why: Mutation("reward.freshness", d, (0.0, 1.5), why),
        "reward.extra_call_penalty": lambda d, why: Mutation("reward.extra_call_penalty", d, (0.0, 1.0), why),
        "router.cost_weight": lambda d, why: Mutation("router.cost_weight", d, (0.0, 3.0), why),
        "bandit.exploration_coeff": lambda d, why: Mutation("bandit.exploration_coeff", d, (0.0, 2.0), why),
        "bandit.recency_decay": lambda d, why: Mutation("bandit.recency_decay", d, (0.8, 1.0), why),
        "bandit.confidence_penalty": lambda d, why: Mutation("bandit.confidence_penalty", d, (0.0, 1.0), why),
        "stop.stop_threshold": lambda d, why: Mutation("stop.stop_threshold", d, (0.1, 0.95), why),
        "verification.authority_threshold": lambda d, why: Mutation("verification.authority_threshold", d, (0.3, 0.95), why),
        "nearest_k": lambda d, why: NearestKMutation("nearest_k", d, (1.0, 32.0), why),
    }
