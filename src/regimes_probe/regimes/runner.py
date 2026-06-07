"""The regimes improvement loop.

Sits above the contextual bandit: run a baseline on OPTIMIZE, detect dominant
failure regimes, propose a bounded policy update, gate it in-sample (OPTIMIZE),
then promote only if it holds up on the held-out CONFIRM split. Every promotion
decision is recorded as events + objects (auditable). See
``docs/REGIMES_IMPROVEMENT_LOOP.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.activegraph_pack.behaviors import EventLog
from regimes_probe.activegraph_pack.events import Events
from regimes_probe.activegraph_pack.objects import Objects
from regimes_probe.activegraph_pack.relations import Relations
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.base import Item
from regimes_probe.eval.harness import experience_phase, run_condition
from regimes_probe.eval.metrics import AttemptOutcome, compute_metrics
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.regimes.action_space import PolicyConfigBundle
from regimes_probe.regimes.detectors import dominant_regimes
from regimes_probe.regimes.gates import PromotionDecision, confirm_gate, optimize_gate
from regimes_probe.regimes.hypothesize import apply_update, propose_update
from regimes_probe.tools.base import SearchProvider


def build_agent(bundle: PolicyConfigBundle, tools: list[str], *, as_of: str = "2026-06-01") -> EpistemicAgent:
    cfg = AgentConfig(
        available_tools=tools,
        query_mode="learned",
        stop_mode="learned",
        as_of=as_of,
        router=bundle.router,
        stop=bundle.stop,
        verification=bundle.verification,
    )
    return EpistemicAgent(cfg)


@dataclass
class BundleEval:
    optimize_metrics: dict[str, Any]
    confirm_metrics: dict[str, Any]
    optimize_outcomes: list[AttemptOutcome]
    confirm_outcomes: list[AttemptOutcome]
    snapshot: dict[str, Any]


def evaluate_bundle(
    bundle: PolicyConfigBundle,
    opt_items: list[Item],
    con_items: list[Item],
    providers: dict[str, SearchProvider],
    tools: list[str],
    *,
    passes: int = 4,
    experience_budget: int = 5,
    gate_budget: int = 3,
    dataset_version: str = "synthetic",
) -> BundleEval:
    """Experience on OPTIMIZE, then exploit (frozen) on OPTIMIZE and CONFIRM."""
    agent = build_agent(bundle, tools)
    mem = PolicyMemory(bundle.bandit.copy(), nearest_k=bundle.nearest_k)
    experience_phase(opt_items, agent, providers, mem, budget=experience_budget,
                     passes=passes, weights=bundle.reward, dataset_version=dataset_version)
    snap = mem.snapshot()

    def _eval(items: list[Item], condition: str) -> list[AttemptOutcome]:
        frozen = PolicyMemory.from_snapshot(snap, frozen=True)
        return run_condition(items, agent, providers, frozen, condition=condition,
                             budget=gate_budget, weights=bundle.reward, explore=False,
                             dataset_version=dataset_version).outcomes

    opt_out = _eval(opt_items, "regimes_optimize")
    con_out = _eval(con_items, "regimes_confirm")
    return BundleEval(
        optimize_metrics=compute_metrics(opt_out),
        confirm_metrics=compute_metrics(con_out),
        optimize_outcomes=opt_out,
        confirm_outcomes=con_out,
        snapshot=snap.to_dict(),
    )


@dataclass
class RegimesLoopResult:
    promotion: PromotionDecision
    baseline: BundleEval
    candidate: BundleEval
    proposal: dict[str, Any]
    mutation_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotion": self.promotion.to_dict(),
            "proposal": self.proposal,
            "mutation_records": self.mutation_records,
            "baseline_optimize": self.baseline.optimize_metrics,
            "baseline_confirm": self.baseline.confirm_metrics,
            "candidate_optimize": self.candidate.optimize_metrics,
            "candidate_confirm": self.candidate.confirm_metrics,
        }


def run_regimes_loop(
    items: list[Item],
    providers: dict[str, SearchProvider],
    tools: list[str],
    *,
    base_bundle: Optional[PolicyConfigBundle] = None,
    passes: int = 4,
    gate_budget: int = 3,
    confirm_fraction: float = 0.4,
    min_delta: float = 0.01,
    max_regression: float = 0.0,
    log: Optional[EventLog] = None,
    metric: str = "correct_per_tool_call",
) -> RegimesLoopResult:
    log = log or EventLog(run_id="regimes-loop")
    base = base_bundle or PolicyConfigBundle()
    split = build_split(items, confirm_fraction=confirm_fraction)
    opt, con = partition(items, split)

    baseline = evaluate_bundle(base, opt, con, providers, tools,
                               passes=passes, gate_budget=gate_budget)

    regimes = dominant_regimes(baseline.optimize_outcomes)
    for regime, count in regimes.most_common():
        ev = log.emit(Events.REGIME_DETECTED, {"regime": regime, "count": count})
        log.add_object(Objects.REGIME_LABEL, {"regime": regime, "count": count}, caused_by=ev.id)

    proposal = propose_update(regimes)
    pev = log.emit(Events.POLICY_UPDATE_PROPOSED, proposal.to_dict())
    update_obj = log.add_object(Objects.POLICY_UPDATE, proposal.to_dict(), caused_by=pev.id)

    candidate_bundle, records = apply_update(base, proposal)
    candidate = evaluate_bundle(candidate_bundle, opt, con, providers, tools,
                                passes=passes, gate_budget=gate_budget)

    og = optimize_gate(baseline.optimize_metrics.get(metric, 0.0),
                       candidate.optimize_metrics.get(metric, 0.0),
                       min_delta=min_delta, metric=metric)
    cg = confirm_gate(baseline.confirm_metrics.get(metric, 0.0),
                      candidate.confirm_metrics.get(metric, 0.0),
                      max_regression=max_regression, metric=metric)
    accepted = bool(og.passed and cg.passed)
    promotion = PromotionDecision(accepted=accepted, optimize_gate=og, confirm_gate=cg,
                                  update={"mutations": records})

    event_type = Events.PROMOTION_ACCEPTED if accepted else Events.PROMOTION_REJECTED
    prom_ev = log.emit(event_type, promotion.to_dict())
    prom_obj = log.add_object(Objects.PROMOTION_DECISION, promotion.to_dict(), caused_by=prom_ev.id)
    log.add_relation(prom_obj, update_obj, Relations.PROMOTION_FOR_UPDATE, caused_by=prom_ev.id)
    log.emit(Events.EVALUATION_COMPLETED,
             {"accepted": accepted, "metric": metric,
              "baseline_confirm": baseline.confirm_metrics.get(metric, 0.0),
              "candidate_confirm": candidate.confirm_metrics.get(metric, 0.0)})

    return RegimesLoopResult(promotion=promotion, baseline=baseline, candidate=candidate,
                             proposal=proposal.to_dict(), mutation_records=records)
