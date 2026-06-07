"""Policy fragments: consolidated, answer-free procedural summaries.

A :class:`PolicyFragment` is what consolidation distills from many traces in a
single signature cluster: which tool, which query template, which stop/verify
choice tends to pay off here. It stores **reward statistics keyed by arm** and
nothing about any specific answer (see ``docs/POLICY_MEMORY.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Field names that must never appear in policy memory (leakage guard).
FORBIDDEN_KEYS = ("answer", "final_answer", "gold", "gold_answer", "label", "solution")


def assert_no_answer_leakage(payload: dict[str, Any], where: str = "policy memory") -> None:
    """Raise if a payload smuggles answer text into policy memory.

    Recurses through dicts/lists checking for forbidden keys. This is called on
    every object written into policy memory and the frozen snapshot.
    """

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in FORBIDDEN_KEYS:
                    raise ValueError(
                        f"answer leakage into {where}: forbidden key '{k}' at {path}"
                    )
                walk(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    walk(payload, where)


@dataclass
class PolicyFragment:
    """Per-cluster consolidated policy. No answer text — reward stats only."""

    cluster_key: str
    tool_rewards: dict[str, dict[str, float]] = field(default_factory=dict)
    query_rewards: dict[str, dict[str, float]] = field(default_factory=dict)
    verify_rewards: dict[str, dict[str, float]] = field(default_factory=dict)
    stop_rewards: dict[str, dict[str, float]] = field(default_factory=dict)
    support_count: int = 0          # traces backing this fragment
    correct_count: int = 0          # how many were graded correct (a reward, not an answer)

    def _best(self, table: dict[str, dict[str, float]]) -> str | None:
        if not table:
            return None
        return max(table.items(), key=lambda kv: (kv[1].get("mean", 0.0), kv[0]))[0]

    @property
    def best_tool(self) -> str | None:
        return self._best(self.tool_rewards)

    @property
    def best_query_arm(self) -> str | None:
        return self._best(self.query_rewards)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "cluster_key": self.cluster_key,
            "tool_rewards": self.tool_rewards,
            "query_rewards": self.query_rewards,
            "verify_rewards": self.verify_rewards,
            "stop_rewards": self.stop_rewards,
            "support_count": self.support_count,
            "correct_count": self.correct_count,
            "best_tool": self.best_tool,
            "best_query_arm": self.best_query_arm,
        }
        assert_no_answer_leakage(d, "policy_fragment")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyFragment":
        return cls(
            cluster_key=d["cluster_key"],
            tool_rewards=d.get("tool_rewards", {}),
            query_rewards=d.get("query_rewards", {}),
            verify_rewards=d.get("verify_rewards", {}),
            stop_rewards=d.get("stop_rewards", {}),
            support_count=int(d.get("support_count", 0)),
            correct_count=int(d.get("correct_count", 0)),
        )
