"""Reward computation and credit attribution (cold path).

Reward computation is allowed to see the gold answer — it is the cold path. But
only the resulting *reward scalars* (and answer-free flags) flow into policy
memory; the answer text never does.

Reward decomposition (identical terms to ``docs/CONTEXTUAL_BANDIT.md``)::

    reward = correctness_reward + evidence_quality_reward + source_authority_reward
           + freshness_reward + first_tool_hit_reward
           - cost_penalty - latency_penalty - stale_penalty
           - contradiction_penalty - extra_call_penalty

The scalar attempt reward is attributed back to the chosen arms in each family
(tool / query / stop / verify) so each bandit learns. The hitting tool/query arm
earns the first-tool-hit bonus; arms that produced no support take a miss
penalty; the stop arm is rewarded for stopping well and penalized for false
stops and over-search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Optional

from regimes_probe.agent.search_loop import AttemptTrace
from regimes_probe.eval.grader import normalize_answer

_VERIFY_ARMS = {"corroborate", "contradiction_check", "fetch_page"}
_MISS_PENALTY = 0.3
_STOP_GOOD_BONUS = 0.3


@dataclass
class RewardWeights:
    correctness: float = 1.0
    evidence_quality: float = 0.3
    source_authority: float = 0.2
    freshness: float = 0.2
    first_tool_hit: float = 0.5
    cost_penalty: float = 2.0          # multiplies total $ cost (small per call)
    latency_penalty: float = 0.0
    stale_penalty: float = 0.4
    contradiction_penalty: float = 0.3
    extra_call_penalty: float = 0.2
    tool_failure_penalty: float = 0.5  # extra penalty on a tool/query arm that errored
    contamination_penalty: float = 0.5  # penalty per benchmark-contaminated result

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "evidence_quality": self.evidence_quality,
            "source_authority": self.source_authority,
            "freshness": self.freshness,
            "first_tool_hit": self.first_tool_hit,
            "cost_penalty": self.cost_penalty,
            "latency_penalty": self.latency_penalty,
            "stale_penalty": self.stale_penalty,
            "contradiction_penalty": self.contradiction_penalty,
            "extra_call_penalty": self.extra_call_penalty,
            "tool_failure_penalty": self.tool_failure_penalty,
            "contamination_penalty": self.contamination_penalty,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RewardWeights":
        return cls(**{k: float(d[k]) for k in cls().to_dict() if k in d})

    # ---- Study 4 reward ablation presets ----
    @classmethod
    def correctness_only(cls) -> "RewardWeights":
        return cls(correctness=1.0, evidence_quality=0.0, source_authority=0.0,
                   freshness=0.0, first_tool_hit=0.0, cost_penalty=0.0,
                   latency_penalty=0.0, stale_penalty=0.0, contradiction_penalty=0.0,
                   extra_call_penalty=0.0)

    @classmethod
    def correctness_minus_cost(cls) -> "RewardWeights":
        return cls(correctness=1.0, evidence_quality=0.0, source_authority=0.0,
                   freshness=0.0, first_tool_hit=0.0, cost_penalty=2.0,
                   latency_penalty=0.0, stale_penalty=0.0, contradiction_penalty=0.0,
                   extra_call_penalty=0.25)

    @classmethod
    def correctness_plus_evidence(cls) -> "RewardWeights":
        return cls(correctness=1.0, evidence_quality=0.4, source_authority=0.2,
                   freshness=0.2, first_tool_hit=0.3, cost_penalty=0.0,
                   latency_penalty=0.0, stale_penalty=0.0, contradiction_penalty=0.2,
                   extra_call_penalty=0.0)

    @classmethod
    def full(cls) -> "RewardWeights":
        return cls()


@dataclass
class RewardResult:
    attempt_reward: float
    arm_rewards: dict[str, dict[str, float]]
    flags: dict[str, bool]
    components: dict[str, float]
    hit_index: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_reward": round(self.attempt_reward, 6),
            "arm_rewards": self.arm_rewards,
            "flags": self.flags,
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "hit_index": self.hit_index,
        }


def compute_rewards(
    trace: AttemptTrace,
    *,
    correct: bool,
    gold_norms: list[str],
    weights: RewardWeights,
    freshness_sensitive: bool,
) -> RewardResult:
    vs = trace.vstate
    w = weights

    def call_hit(call) -> bool:
        return any(
            o.supports and o.asserts and normalize_answer(o.asserts) in gold_norms
            for o in call.observations
        )

    hits = [call_hit(c) for c in trace.calls]
    hit_index = next((i for i, h in enumerate(hits) if h), None)
    calls_used = len(trace.calls)
    first_tool_hit_flag = hit_index == 0
    needed = (hit_index + 1) if hit_index is not None else calls_used
    extra_calls = max(0, calls_used - needed)

    # Benchmark-contaminated results (snippets mirroring the question / eval hosts)
    # earn a penalty so the query/tool form that surfaces them learns to avoid it.
    n_contaminated = sum(getattr(c, "contaminated_results", 0) for c in trace.calls)

    stale_error = bool(freshness_sensitive and vs.support_found and not vs.freshness_ok)
    stopped = bool(trace.calls and trace.calls[-1].stop_arm == "stop_now")
    false_stop = bool(stopped and not correct and calls_used < trace.budget)
    over_search = bool(correct and extra_calls > 0)
    under_search = bool(not correct and stopped and calls_used < trace.budget and hit_index is None)

    components = {
        "correctness": w.correctness * (1.0 if correct else 0.0),
        "evidence_quality": w.evidence_quality * vs.score,
        "source_authority": w.source_authority * (1.0 if vs.authority_ok else 0.0),
        "freshness": w.freshness * (1.0 if vs.freshness_ok else 0.0),
        "first_tool_hit": w.first_tool_hit * (1.0 if first_tool_hit_flag else 0.0),
        "cost_penalty": -w.cost_penalty * trace.total_cost,
        "latency_penalty": -w.latency_penalty * trace.total_latency,
        "stale_penalty": -w.stale_penalty * (1.0 if stale_error else 0.0),
        "contradiction_penalty": -w.contradiction_penalty * (1.0 if vs.contradiction else 0.0),
        "extra_call_penalty": -w.extra_call_penalty * extra_calls,
        "contamination_penalty": -w.contamination_penalty * n_contaminated,
    }
    R = sum(components.values())

    acc: dict[str, dict[str, list[float]]] = {"tool": {}, "query": {}, "stop": {}, "verify": {}}

    def add(fam: str, arm: str, val: float) -> None:
        acc[fam].setdefault(arm, []).append(val)

    n_failed = sum(1 for c in trace.calls if getattr(c, "failed", False))
    for i, c in enumerate(trace.calls):
        tool_bonus = w.first_tool_hit if hits[i] else 0.0
        miss_pen = 0.0 if c.supported else _MISS_PENALTY
        # A failed tool call (provider/API error) earns no evidence gain and an
        # explicit penalty, so the bandit learns the tool failed in this context.
        fail_pen = w.tool_failure_penalty if getattr(c, "failed", False) else 0.0
        add("tool", c.tool, R + tool_bonus - miss_pen - w.cost_penalty * c.cost - fail_pen)
        add("query", c.query_arm, R + tool_bonus - miss_pen - fail_pen)

        stop_bonus = 0.0
        if c.stop_arm == "stop_now":
            if correct and not over_search:
                stop_bonus += _STOP_GOOD_BONUS
            if false_stop:
                stop_bonus -= _STOP_GOOD_BONUS
        else:
            if over_search and hit_index is not None and i >= hit_index:
                stop_bonus -= w.extra_call_penalty
        add("stop", c.stop_arm, R + stop_bonus)
        if c.stop_arm in _VERIFY_ARMS:
            add("verify", c.stop_arm, R + stop_bonus)

    arm_rewards = {
        fam: {arm: mean(vals) for arm, vals in tbl.items() if vals}
        for fam, tbl in acc.items()
    }

    flags = {
        "first_tool_hit": first_tool_hit_flag,
        "false_stop": false_stop,
        "over_search": over_search,
        "under_search": under_search,
        "stale_error": stale_error,
        "stopped": stopped,
        "contradiction": vs.contradiction,
        "found_hit": hit_index is not None,
        "had_tool_failure": n_failed > 0,
        "contaminated": n_contaminated > 0,
    }
    return RewardResult(
        attempt_reward=R,
        arm_rewards=arm_rewards,
        flags=flags,
        components=components,
        hit_index=hit_index,
    )
