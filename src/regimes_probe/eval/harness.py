"""Run-condition orchestration.

Ties the ActiveGraph pack (event recording), the agent (hot path), the grader
and reward (cold path), and the metrics into runnable *conditions*:

  * :func:`run_condition`   — run every item once at one budget, recording
    events; optionally update policy memory (experience) or run frozen (CONFIRM).
  * :func:`experience_phase`— repeated OPTIMIZE passes with exploration that
    accumulate trace experience into policy memory.

Each attempt yields an :class:`AttemptOutcome` for the metrics layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.activegraph_pack.behaviors import (
    EventLog,
    record_attempt,
    record_grade_and_reward,
    record_policy_update,
    record_run_start,
)
from regimes_probe.agent.planner import EpistemicAgent
from regimes_probe.datasets.base import Item
from regimes_probe.eval.metrics import AttemptOutcome
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.tools.base import SearchProvider


@dataclass
class ConditionResult:
    condition: str
    budget: int
    outcomes: list[AttemptOutcome]
    log: EventLog
    debug: list = field(default_factory=list)   # list[DebugRecord]


def _outcome(trace, grade, reward, *, condition: str, budget: int) -> AttemptOutcome:
    return AttemptOutcome(
        item_id=trace.item_id,
        condition=condition,
        budget=budget,
        correct=grade.correct,
        abstained=grade.abstained,
        tool_calls=trace.tool_calls,
        cost=trace.total_cost,
        latency=trace.total_latency,
        evidence_score=trace.vstate.score,
        first_tool_hit=reward.flags["first_tool_hit"],
        false_stop=reward.flags["false_stop"],
        over_search=reward.flags["over_search"],
        under_search=reward.flags["under_search"],
        stale_error=reward.flags["stale_error"],
        found_hit=reward.flags["found_hit"],
        cluster_key=trace.signature.cluster_key,
        attempt_reward=reward.attempt_reward,
        authority_ok=trace.vstate.authority_ok,
        contradiction=trace.vstate.contradiction,
        support_found=trace.vstate.support_found,
        failed_tool_calls=sum(1 for c in trace.calls if getattr(c, "failed", False)),
        failed_tools=[c.tool for c in trace.calls if getattr(c, "failed", False)],
        contaminated_results=sum(getattr(c, "contaminated_results", 0) for c in trace.calls),
        total_results=sum(1 for c in trace.calls for o in c.observations
                          if not getattr(o, "failed", False)),
    )


def run_condition(
    items: list[Item],
    agent: EpistemicAgent,
    providers: dict[str, SearchProvider],
    memory: PolicyMemory,
    *,
    condition: str,
    budget: int,
    weights: Optional[RewardWeights] = None,
    explore: bool = False,
    update_memory: bool = False,
    dataset_version: str = "unknown",
    log: Optional[EventLog] = None,
    run_id: str = "regimes-probe-run",
    pass_tag: str = "p0",
) -> ConditionResult:
    """Run one condition over all items at a fixed budget."""
    weights = weights or RewardWeights.full()
    log = log or EventLog(run_id=run_id)
    record_run_start(log, dataset_version=dataset_version, condition=condition,
                     config={"budget": budget, "explore": explore,
                             "update_memory": update_memory})
    from regimes_probe.eval.debug import build_debug_record
    outcomes: list[AttemptOutcome] = []
    debug: list = []
    for item in items:
        attempt_id = f"{condition}-b{budget}-{pass_tag}-{item.id}"
        rec = record_attempt(log, "benchmark_run#1", agent, item, memory, providers,
                             budget=budget, explore=explore, attempt_id=attempt_id)
        sig = rec["signature"]
        freshness_sensitive = bool(sig.features.get("freshness_sensitive"))
        grade, reward = record_grade_and_reward(log, rec, item, weights=weights,
                                                freshness_sensitive=freshness_sensitive)
        if update_memory:
            record_policy_update(log, memory, sig, attempt_id, reward, rec["trace"],
                                 correct=grade.correct)
        outcome = _outcome(rec["trace"], grade, reward, condition=condition, budget=budget)
        outcomes.append(outcome)
        debug.append(build_debug_record(item=item, trace=rec["trace"], grade=grade,
                                        reward=reward, condition=condition, budget=budget,
                                        outcome=outcome))
    return ConditionResult(condition=condition, budget=budget, outcomes=outcomes,
                           log=log, debug=debug)


def experience_phase(
    items: list[Item],
    agent: EpistemicAgent,
    providers: dict[str, SearchProvider],
    memory: PolicyMemory,
    *,
    budget: int,
    weights: Optional[RewardWeights] = None,
    passes: int = 3,
    dataset_version: str = "unknown",
    log: Optional[EventLog] = None,
) -> EventLog:
    """Accumulate OPTIMIZE experience into policy memory (exploration on)."""
    weights = weights or RewardWeights.full()
    log = log or EventLog(run_id="experience")
    for p in range(passes):
        run_condition(items, agent, providers, memory, condition="experience",
                      budget=budget, weights=weights, explore=True, update_memory=True,
                      dataset_version=dataset_version, log=log, pass_tag=f"p{p}")
    return log
