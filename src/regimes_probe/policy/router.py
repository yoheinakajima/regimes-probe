"""Level 1: tool routing policy.

The router ranks the available search/fetch tools for a question and returns a
tool sequence under the budget cap, with a numeric explanation. It NEVER calls
an LLM: it uses nearest-neighbour trace retrieval, contextual-bandit scores,
and cost/latency penalties, with exploration allowed during OPTIMIZE.

See ``docs/ROUTING_POLICY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from regimes_probe.policy.contextual_bandit import ContextualBandit, ScoredArm
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.signatures import QuerySignature


@dataclass
class RoutingPlan:
    ranked: list[dict[str, Any]]
    sequence: list[str]
    explanation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ranked": self.ranked, "sequence": self.sequence,
                "explanation": self.explanation}


@dataclass
class RouterConfig:
    """Mutable router knobs (regimes loop may tune these)."""

    cost_weight: float = 0.5     # how strongly per-call $ cost penalizes a tool
    blend_neighbor: float = 0.7  # mirrors bandit neighbour blend for transparency

    def to_dict(self) -> dict[str, Any]:
        return {"cost_weight": self.cost_weight, "blend_neighbor": self.blend_neighbor}


class Router:
    """Rank tools and build a budgeted tool sequence. No LLM, deterministic."""

    def __init__(self, config: Optional[RouterConfig] = None) -> None:
        self.config = config or RouterConfig()

    def route(
        self,
        signature: QuerySignature,
        memory: PolicyMemory,
        available_tools: list[str],
        *,
        budget: int,
        tool_costs: Optional[dict[str, Decimal]] = None,
        explore: bool = False,
        salt: str = "",
    ) -> RoutingPlan:
        bandit: ContextualBandit = memory.bandits["tool"]
        neighbor = memory.estimate("tool", signature.embedding)
        ranked: list[ScoredArm] = bandit.choose_tool_plan(
            signature.cluster_key,
            available_tools,
            k=len(available_tools),
            explore=explore,
            neighbor=neighbor,
            salt=salt,
        )

        tool_costs = tool_costs or {}
        adjusted: list[tuple[float, ScoredArm, float]] = []
        for s in ranked:
            cost = float(tool_costs.get(s.arm, Decimal("0")))
            penalty = self.config.cost_weight * cost
            adjusted.append((s.score - penalty, s, penalty))
        # Deterministic: adjusted score desc, then arm asc.
        adjusted.sort(key=lambda t: (-t[0], t[1].arm))

        sequence = [s.arm for _, s, _ in adjusted[: max(1, budget)]]
        explanation = {
            "cluster_key": signature.cluster_key,
            "strategy": bandit.params.strategy,
            "explore": explore,
            "neighbor_arms": {a: [round(m, 4), round(w, 3)] for a, (m, w) in neighbor.items()},
            "ranked": [
                {**s.to_dict(), "cost_penalty": round(pen, 6), "adjusted": round(adj, 6)}
                for adj, s, pen in adjusted
            ],
        }
        return RoutingPlan(
            ranked=[s.to_dict() for _, s, _ in adjusted],
            sequence=sequence,
            explanation=explanation,
        )
