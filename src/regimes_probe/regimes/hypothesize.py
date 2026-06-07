"""Propose bounded policy updates from dominant failure regimes.

Maps each regime to the safe numeric mutation(s) most likely to repair it. The
proposal is a small, ordered list of :class:`Mutation` objects (a
``policy_update``) — never a prompt or code change in v0.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from regimes_probe.regimes.action_space import Mutation, PolicyConfigBundle, safe_mutations

# regime -> list of (mutation_key, delta, rationale)
_REGIME_TO_MUTATIONS: dict[str, list[tuple[str, float, str]]] = {
    "over_search": [("stop.stop_threshold", -0.1, "stop sooner once evidence suffices"),
                    ("reward.extra_call_penalty", +0.1, "penalize redundant calls")],
    "stop_too_late": [("stop.stop_threshold", -0.1, "lower sufficiency bar to stop")],
    "stop_too_early": [("stop.stop_threshold", +0.1, "require more evidence before stopping")],
    "under_search": [("stop.stop_threshold", +0.1, "keep searching longer")],
    "stale_evidence": [("reward.freshness", +0.2, "reward fresh sources more"),
                       ("verification.authority_threshold", -0.05, "accept fresh non-top-authority")],
    "route_miss": [("bandit.exploration_coeff", +0.2, "explore tool routing more"),
                   ("router.cost_weight", -0.2, "stop over-penalizing useful pricier tools")],
    "query_miss": [("bandit.exploration_coeff", +0.2, "explore query templates more")],
    "evidence_sparse": [("nearest_k", +4.0, "borrow priors from more neighbours")],
    "verification_miss": [("verification.authority_threshold", +0.05, "demand stronger support")],
    "contradiction_unresolved": [("verification.authority_threshold", +0.05, "prefer authoritative sources")],
    "answer_extraction_miss": [("nearest_k", +2.0, "more corroboration priors")],
    "support_answer_mismatch": [("verification.authority_threshold", +0.05, "tighten support check")],
}


@dataclass
class PolicyUpdateProposal:
    dominant: list[tuple[str, int]]
    mutations: list[Mutation]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_regimes": self.dominant,
            "mutations": [{"target": m.target, "delta": m.delta, "tier": m.tier,
                           "rationale": m.rationale} for m in self.mutations],
            "rationale": self.rationale,
        }


def propose_update(regimes: Counter, *, top_k: int = 2, max_mutations: int = 3) -> PolicyUpdateProposal:
    """Propose a bounded update targeting the top failure regimes."""
    catalog = safe_mutations()
    dominant = regimes.most_common(top_k)
    mutations: list[Mutation] = []
    seen_targets: set[str] = set()
    for regime, _count in dominant:
        for key, delta, why in _REGIME_TO_MUTATIONS.get(regime, []):
            if key in seen_targets or key not in catalog:
                continue
            mutations.append(catalog[key](delta, f"{regime}: {why}"))
            seen_targets.add(key)
            if len(mutations) >= max_mutations:
                break
        if len(mutations) >= max_mutations:
            break
    rationale = "; ".join(f"{r}×{c}" for r, c in dominant) or "no dominant regime"
    return PolicyUpdateProposal(dominant=dominant, mutations=mutations, rationale=rationale)


def apply_update(bundle: PolicyConfigBundle, proposal: PolicyUpdateProposal) -> tuple[PolicyConfigBundle, list[dict[str, Any]]]:
    """Apply all proposed mutations in order, returning the new bundle + records."""
    cur = bundle.copy()
    records: list[dict[str, Any]] = []
    for m in proposal.mutations:
        cur, rec = m.apply(cur)
        records.append(rec)
    return cur, records
