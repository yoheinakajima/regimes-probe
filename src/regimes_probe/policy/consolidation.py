"""Consolidate raw traces into per-cluster :class:`PolicyFragment` objects.

Consolidation is part of the cold path: after experience, it summarizes the
trace store into compact, answer-free policy fragments keyed by signature
cluster. Fragments are what the hot path reads as priors (and what a future
natural-language "lesson" generator would describe).
"""

from __future__ import annotations

from typing import Any

from regimes_probe.policy.memory import FAMILIES, PolicyMemory, TraceRecord
from regimes_probe.policy.policy_fragment import PolicyFragment


def _aggregate(traces: list[TraceRecord], family: str) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, float]] = {}
    for t in traces:
        for arm, reward in t.arm_rewards.get(family, {}).items():
            slot = acc.setdefault(arm, {"sum": 0.0, "n": 0.0})
            slot["sum"] += float(reward)
            slot["n"] += 1.0
    return {
        arm: {"mean": (s["sum"] / s["n"]) if s["n"] else 0.0, "n": s["n"]}
        for arm, s in acc.items()
    }


def consolidate(memory: PolicyMemory) -> dict[str, PolicyFragment]:
    """Build/refresh ``memory.fragments`` from ``memory.traces``. Returns them."""
    by_cluster: dict[str, list[TraceRecord]] = {}
    for t in memory.traces:
        by_cluster.setdefault(t.cluster_key, []).append(t)

    fragments: dict[str, PolicyFragment] = {}
    for cluster_key, traces in by_cluster.items():
        frag = PolicyFragment(
            cluster_key=cluster_key,
            tool_rewards=_aggregate(traces, "tool"),
            query_rewards=_aggregate(traces, "query"),
            verify_rewards=_aggregate(traces, "verify"),
            stop_rewards=_aggregate(traces, "stop"),
            support_count=len(traces),
            correct_count=sum(1 for t in traces if t.correct),
        )
        fragments[cluster_key] = frag
    memory.fragments = fragments
    return fragments
