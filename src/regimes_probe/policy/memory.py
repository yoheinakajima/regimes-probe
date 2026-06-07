"""Policy memory: traces, rewards, priors and policy fragments — no answers.

Policy memory is the *procedural* memory of the agent. It stores, per question
signature: an embedding, the normalized-text hash, the arms that were tried and
the reward they earned, and consolidated policy fragments. It NEVER stores the
final answer text or supports full Q/A retrieval (see
``docs/LEAKAGE_CONTROLS.md``).

Two roles:
  * **OPTIMIZE / experience phase** — a live, mutable memory accumulating
    traces and updating bandits (exploration allowed).
  * **CONFIRM phase** — a *frozen snapshot* used read-only with deterministic
    exploitation; attempts to mutate it raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.policy.contextual_bandit import BanditParams, ContextualBandit
from regimes_probe.policy.embeddings import cosine
from regimes_probe.policy.policy_fragment import (
    PolicyFragment,
    assert_no_answer_leakage,
)
from regimes_probe.policy.signatures import QuerySignature

#: The four arm families policy memory tracks.
FAMILIES: tuple[str, ...] = ("tool", "query", "verify", "stop")


@dataclass
class TraceRecord:
    """One attempt's answer-free procedural trace.

    ``correct`` and the per-family arm rewards are *reward signals*, not the
    answer. The question text itself is reduced to an embedding and a hash.
    """

    attempt_id: str
    cluster_key: str
    norm_hash: str
    embedding: list[float]
    arm_rewards: dict[str, dict[str, float]]   # family -> {arm: reward}
    correct: bool
    tool_calls: int

    def to_dict(self) -> dict[str, Any]:
        d = {
            "attempt_id": self.attempt_id,
            "cluster_key": self.cluster_key,
            "norm_hash": self.norm_hash,
            "embedding": self.embedding,
            "arm_rewards": self.arm_rewards,
            "correct": self.correct,
            "tool_calls": self.tool_calls,
        }
        assert_no_answer_leakage(d, "trace_record")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceRecord":
        return cls(
            attempt_id=d["attempt_id"],
            cluster_key=d["cluster_key"],
            norm_hash=d["norm_hash"],
            embedding=list(d["embedding"]),
            arm_rewards=d.get("arm_rewards", {}),
            correct=bool(d.get("correct", False)),
            tool_calls=int(d.get("tool_calls", 0)),
        )


class FrozenMemoryError(RuntimeError):
    """Raised when a frozen (CONFIRM) snapshot is asked to mutate."""


@dataclass
class PolicyMemorySnapshot:
    """A serializable, frozen image of policy memory used during CONFIRM."""

    bandits: dict[str, dict[str, Any]]
    traces: list[dict[str, Any]]
    fragments: dict[str, dict[str, Any]]
    params: dict[str, Any]
    nearest_k: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "bandits": self.bandits,
            "traces": self.traces,
            "fragments": self.fragments,
            "params": self.params,
            "nearest_k": self.nearest_k,
            "meta": self.meta,
        }
        assert_no_answer_leakage(d, "policy_memory_snapshot")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyMemorySnapshot":
        return cls(
            bandits=d.get("bandits", {}),
            traces=d.get("traces", []),
            fragments=d.get("fragments", {}),
            params=d.get("params", {}),
            nearest_k=int(d.get("nearest_k", 8)),
            meta=d.get("meta", {}),
        )


class PolicyMemory:
    """Mutable policy memory: bandits per family + a trace store + fragments."""

    def __init__(
        self,
        params: Optional[BanditParams] = None,
        *,
        nearest_k: int = 8,
        frozen: bool = False,
    ) -> None:
        self.params = params or BanditParams()
        self.nearest_k = nearest_k
        self.frozen = frozen
        self.bandits: dict[str, ContextualBandit] = {
            fam: ContextualBandit(fam, self.params.copy()) for fam in FAMILIES
        }
        self.traces: list[TraceRecord] = []
        self.fragments: dict[str, PolicyFragment] = {}

    # ---------------- mutation (OPTIMIZE only) ----------------

    def observe(
        self,
        signature: QuerySignature,
        attempt_id: str,
        arm_rewards: dict[str, dict[str, float]],
        *,
        correct: bool,
        tool_calls: int,
    ) -> None:
        """Record one attempt: update every family bandit and store the trace."""
        if self.frozen:
            raise FrozenMemoryError("cannot observe into a frozen snapshot (CONFIRM)")
        for fam, arms in arm_rewards.items():
            bandit = self.bandits.get(fam)
            if bandit is None:
                continue
            for arm, reward in arms.items():
                bandit.update_reward(signature.cluster_key, arm, float(reward))
        self.traces.append(
            TraceRecord(
                attempt_id=attempt_id,
                cluster_key=signature.cluster_key,
                norm_hash=signature.norm_hash,
                embedding=list(signature.embedding),
                arm_rewards=arm_rewards,
                correct=bool(correct),
                tool_calls=int(tool_calls),
            )
        )

    # ---------------- nearest-neighbour estimates ----------------

    def neighbors(self, embedding: list[float], *, k: Optional[int] = None) -> list[tuple[float, TraceRecord]]:
        """Top-``k`` traces by cosine similarity to ``embedding``."""
        k = k or self.nearest_k
        scored = [(cosine(embedding, t.embedding), t) for t in self.traces]
        scored.sort(key=lambda st: (-st[0], st[1].attempt_id))
        return scored[:k]

    def estimate(
        self, family: str, embedding: list[float], *, k: Optional[int] = None
    ) -> dict[str, tuple[float, float]]:
        """Similarity-weighted reward estimate per arm from nearest traces.

        Returns ``{arm: (weighted_mean, total_weight)}`` for blending into the
        bandit score (the nearest-neighbour weighted reward term).
        """
        acc: dict[str, list[float]] = {}
        wsum: dict[str, float] = {}
        for sim, trace in self.neighbors(embedding, k=k):
            if sim <= 0:
                continue
            for arm, reward in trace.arm_rewards.get(family, {}).items():
                acc.setdefault(arm, [0.0, 0.0])
                acc[arm][0] += sim * reward
                acc[arm][1] += sim
                wsum[arm] = wsum.get(arm, 0.0) + sim
        out: dict[str, tuple[float, float]] = {}
        for arm, (num, den) in acc.items():
            if den > 0:
                out[arm] = (num / den, wsum[arm])
        return out

    # ---------------- snapshot / freeze ----------------

    def set_params(self, params: BanditParams) -> None:
        """Apply mutated params to every family bandit (regimes loop)."""
        self.params = params
        for fam, bandit in self.bandits.items():
            self.bandits[fam] = bandit.with_params(params.copy())

    def snapshot(self, meta: Optional[dict[str, Any]] = None) -> PolicyMemorySnapshot:
        """Freeze the current state into a serializable, answer-free snapshot."""
        return PolicyMemorySnapshot(
            bandits={fam: b.to_dict() for fam, b in self.bandits.items()},
            traces=[t.to_dict() for t in self.traces],
            fragments={ck: f.to_dict() for ck, f in self.fragments.items()},
            params=self.params.to_dict(),
            nearest_k=self.nearest_k,
            meta=meta or {},
        )

    @classmethod
    def from_snapshot(cls, snap: PolicyMemorySnapshot, *, frozen: bool = True) -> "PolicyMemory":
        """Rebuild memory from a snapshot. Frozen by default (CONFIRM use)."""
        params = BanditParams.from_dict(snap.params)
        mem = cls(params, nearest_k=snap.nearest_k, frozen=frozen)
        mem.bandits = {fam: ContextualBandit.from_dict(d) for fam, d in snap.bandits.items()}
        for fam in FAMILIES:
            mem.bandits.setdefault(fam, ContextualBandit(fam, params.copy()))
        mem.traces = [TraceRecord.from_dict(t) for t in snap.traces]
        mem.fragments = {ck: PolicyFragment.from_dict(f) for ck, f in snap.fragments.items()}
        return mem
