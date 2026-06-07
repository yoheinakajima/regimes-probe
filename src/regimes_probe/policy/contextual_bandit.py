"""Contextual bandit over epistemic policy arms.

Arms are: search/fetch tools (Level 1), and query templates, verification
choices, and stop/continue choices (Level 2). The context is the
:class:`~regimes_probe.policy.signatures.QuerySignature` cluster key plus, at
scoring time, nearest-neighbour reward estimates supplied by policy memory.

Reward decomposition (see ``docs/CONTEXTUAL_BANDIT.md`` and
``docs/GRADING_AND_REWARD.md`` — identical terms)::

    reward = correctness_reward + evidence_quality_reward + source_authority_reward
           + freshness_reward + first_tool_hit_reward
           - cost_penalty - latency_penalty - stale_penalty
           - contradiction_penalty - extra_call_penalty

Strategies: epsilon-greedy, UCB (default, fully deterministic), and optional
Gaussian Thompson sampling. All "randomness" is seeded and salted so every
decision is reproducible under replay. During OPTIMIZE exploration is allowed;
during CONFIRM the policy is exploited deterministically against a frozen
snapshot.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


def _seed_int(*parts: Any) -> int:
    """Deterministic 63-bit seed from string parts (replayable RNG)."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


@dataclass
class ArmStats:
    """Recency-decayed reward statistics for one arm in one context.

    ``n`` is the raw pull count; ``w`` is the decayed effective count used for
    confidence terms; ``mean``/``var`` are recency-weighted.
    """

    n: int = 0
    w: float = 0.0
    mean: float = 0.0
    var: float = 0.0

    def update(self, reward: float, decay: float) -> None:
        # Recency decay: old mass shrinks by ``decay`` before adding the new
        # observation, so w saturates near 1/(1-decay) and recent rewards
        # dominate. decay == 1.0 reduces to a plain running average.
        self.n += 1
        self.w = self.w * decay + 1.0
        prev_mean = self.mean
        self.mean = prev_mean + (reward - prev_mean) / self.w
        # Recency-weighted variance (EWMA of squared deviation).
        self.var = (1.0 - 1.0 / self.w) * (self.var + (reward - prev_mean) * (reward - self.mean))

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "w": self.w, "mean": self.mean, "var": self.var}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArmStats":
        return cls(n=int(d["n"]), w=float(d["w"]), mean=float(d["mean"]), var=float(d["var"]))


@dataclass
class BanditParams:
    """Tunable parameters — the surface the regimes loop is allowed to mutate."""

    strategy: str = "ucb"               # "ucb" | "epsilon_greedy" | "thompson"
    exploration_coeff: float = 0.6      # UCB c / epsilon probability
    recency_decay: float = 0.98         # 0<decay<=1; <1 weights recent rewards
    confidence_penalty: float = 0.1     # pessimism on high-variance/low-count arms
    blend_local: float = 1.0            # weight on this-cluster stats
    blend_global: float = 0.4           # weight on global prior fallback
    blend_neighbor: float = 0.7         # weight on nearest-neighbour estimates
    seed: int = 1729

    def copy(self) -> "BanditParams":
        return BanditParams(**self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "exploration_coeff": self.exploration_coeff,
            "recency_decay": self.recency_decay,
            "confidence_penalty": self.confidence_penalty,
            "blend_local": self.blend_local,
            "blend_global": self.blend_global,
            "blend_neighbor": self.blend_neighbor,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BanditParams":
        return cls(**{k: d[k] for k in cls().to_dict() if k in d})


@dataclass
class ScoredArm:
    arm: str
    score: float
    mean: float
    w: float
    explore_bonus: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "score": round(self.score, 6),
            "mean": round(self.mean, 6),
            "w": round(self.w, 4),
            "explore_bonus": round(self.explore_bonus, 6),
        }


class ContextualBandit:
    """A bandit for one *family* of arms (tools, or query templates, …).

    Stats are keyed ``(cluster_key, arm)`` with a ``(arm)`` global prior. The
    bandit is pure in-memory state; persistence is via :meth:`to_dict`.
    """

    def __init__(self, family: str, params: Optional[BanditParams] = None) -> None:
        self.family = family
        self.params = params or BanditParams()
        self._ctx: dict[str, dict[str, ArmStats]] = {}
        self._global: dict[str, ArmStats] = {}

    # ---------------- updates ----------------

    def update_reward(self, cluster_key: str, arm: str, reward: float) -> None:
        """Apply one observed reward to context-local and global stats."""
        decay = self.params.recency_decay
        self._ctx.setdefault(cluster_key, {}).setdefault(arm, ArmStats()).update(reward, decay)
        self._global.setdefault(arm, ArmStats()).update(reward, decay)

    # ---------------- blended estimate ----------------

    def _blended(
        self,
        cluster_key: str,
        arm: str,
        neighbor: Optional[dict[str, tuple[float, float]]],
    ) -> tuple[float, float, float]:
        """Return (mean, w, var) blending local, global and neighbour stats."""
        p = self.params
        local = self._ctx.get(cluster_key, {}).get(arm, ArmStats())
        glob = self._global.get(arm, ArmStats())
        n_mean, n_w = (neighbor or {}).get(arm, (0.0, 0.0))
        ws = [
            (local.mean, local.w * p.blend_local, local.var),
            (glob.mean, glob.w * p.blend_global, glob.var),
            (n_mean, n_w * p.blend_neighbor, 0.0),
        ]
        total_w = sum(w for _, w, _ in ws)
        if total_w <= 0:
            return 0.0, 0.0, 1.0
        mean = sum(m * w for m, w, _ in ws) / total_w
        var = sum(v * w for _, w, v in ws) / total_w
        return mean, total_w, var

    # ---------------- scoring ----------------

    def score_tool(
        self,
        cluster_key: str,
        arm: str,
        *,
        total_w: float,
        explore: bool,
        neighbor: Optional[dict[str, tuple[float, float]]] = None,
        salt: str = "",
    ) -> ScoredArm:
        """Score one arm. ``total_w`` is summed effective count across arms."""
        p = self.params
        mean, w, var = self._blended(cluster_key, arm, neighbor)
        # Pessimism: shrink toward mean by confidence penalty on uncertain arms.
        conf = p.confidence_penalty * math.sqrt((var + 1e-9) / (w + 1.0))
        base = mean - conf
        bonus = 0.0
        if explore:
            if p.strategy == "ucb":
                bonus = p.exploration_coeff * math.sqrt(math.log(total_w + 2.0) / (w + 1.0))
            elif p.strategy == "thompson":
                rng = random.Random(_seed_int(p.seed, cluster_key, arm, salt, round(w, 3)))
                sigma = math.sqrt((var + 1e-6) / (w + 1.0))
                bonus = rng.gauss(0.0, 1.0) * sigma * max(p.exploration_coeff, 1e-6)
            # epsilon-greedy bonus handled in choose_tool_plan (whole-arm flip)
        return ScoredArm(arm=arm, score=base + bonus, mean=mean, w=w, explore_bonus=bonus)

    def choose_tool_plan(
        self,
        cluster_key: str,
        arms: Iterable[str],
        *,
        k: int,
        explore: bool,
        neighbor: Optional[dict[str, tuple[float, float]]] = None,
        salt: str = "",
    ) -> list[ScoredArm]:
        """Rank ``arms`` and return the top ``k`` (deterministic tie-break).

        For epsilon-greedy, with probability ``exploration_coeff`` (seeded by
        ``salt``) the ranking is shuffled deterministically to force
        exploration; otherwise it is pure exploitation. UCB/Thompson fold
        exploration into the per-arm bonus instead.
        """
        arms = list(arms)
        total_w = 0.0
        for a in arms:
            _, w, _ = self._blended(cluster_key, a, neighbor)
            total_w += w
        scored = [
            self.score_tool(
                cluster_key, a, total_w=total_w, explore=explore, neighbor=neighbor, salt=salt
            )
            for a in arms
        ]
        if explore and self.params.strategy == "epsilon_greedy":
            rng = random.Random(_seed_int(self.params.seed, cluster_key, salt, "eps"))
            if rng.random() < self.params.exploration_coeff:
                rng.shuffle(scored)
                return scored[: max(1, k)]
        # Deterministic sort: score desc, then arm name asc for stable ties.
        scored.sort(key=lambda s: (-s.score, s.arm))
        return scored[: max(1, k)]

    # ---------------- persistence ----------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "params": self.params.to_dict(),
            "ctx": {
                ck: {a: st.to_dict() for a, st in arms.items()}
                for ck, arms in self._ctx.items()
            },
            "global": {a: st.to_dict() for a, st in self._global.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContextualBandit":
        b = cls(d["family"], BanditParams.from_dict(d.get("params", {})))
        for ck, arms in d.get("ctx", {}).items():
            b._ctx[ck] = {a: ArmStats.from_dict(s) for a, s in arms.items()}
        b._global = {a: ArmStats.from_dict(s) for a, s in d.get("global", {}).items()}
        return b

    def with_params(self, params: BanditParams) -> "ContextualBandit":
        """Return a copy sharing stats but using new params (regimes mutation)."""
        b = ContextualBandit.from_dict(self.to_dict())
        b.params = params
        return b
