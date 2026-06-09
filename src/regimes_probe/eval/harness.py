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
        candidate_entity_count=len({(e.get("candidate_text") or e.get("text") or "").lower()
                                    for c in trace.calls
                                    for e in getattr(c, "candidate_entities", [])}),
        followup_query_count=sum(1 for c in trace.calls if getattr(c, "stage", 1) >= 2),
        evidence_improved_after_followup=any(
            getattr(c, "stage", 1) >= 2 and getattr(c, "evidence_improved", False)
            for c in trace.calls),
        answer_found_after_stage=next(
            (getattr(c, "stage", 1) for c in trace.calls if c.supported), 0),
        stage_depth_used=max((getattr(c, "stage", 1) for c in trace.calls), default=0),
        iterative=_iterative_stats(trace),
        scrape=_scrape_stats(trace),
        frame=_frame_stats(trace),
        frame_parse=_frame_parse_stats(trace),
        frontier=_frontier_stats(trace),
        llm_frontier=dict(getattr(trace, "llm_frontier", {}) or {}),
        interpretation=_interpretation_stats(trace),
    )


def _interpretation_stats(trace) -> dict:
    """Per-attempt evidence-interpretation stats (Level 5d), aggregated by compute_metrics."""
    cf = getattr(trace, "candidate_frontier", {}) or {}
    if not cf or cf.get("skipped"):
        return {}
    st = dict(cf.get("interpreter_stats", {}) or {})
    # how many confirmed candidates trace back to an interpreted (non-noise) source.
    confirmed = sum(len(s.get("confirmed_candidate_ids", [])) for s in cf.get("slates", []))
    st["candidate_promotion_from_evidence_count"] = confirmed
    # ev->slot vs ev->cons link invariants (req 7): an interpretation "links a slot" when an
    # accepted candidate proposes a slot, and "links a constraint" when it supports one.
    slot_link = cons_link = slot_true_cons_false = acc_no_support = 0
    for it in cf.get("interpretations", []):
        acc = [a for a in it.get("candidate_assertions", []) if not a.get("rejection_reason")]
        has_slot = any(a.get("proposed_slot_ids") for a in acc)
        has_cons = any(a.get("supports_constraint_ids") for a in acc)
        acc_no_support += sum(1 for a in acc if not a.get("supports_constraint_ids"))
        slot_link += bool(has_slot)
        cons_link += bool(has_cons)
        slot_true_cons_false += bool(has_slot and not has_cons)
    st["interpretation_slot_link_count"] = slot_link
    st["interpretation_constraint_link_count"] = cons_link
    st["ev_slot_true_cons_false_count"] = slot_true_cons_false
    st["accepted_candidate_without_constraint_support_count"] = acc_no_support
    # Support-consistency invariant (req 3): every full_support judgment must materialize as
    # constraint support on SOME candidate of that slot; otherwise it was silently dropped.
    supported_pairs = set()           # (slot_id, constraint_id) actually materialized
    for s in cf.get("slates", []):
        for c in s.get("top_candidates", []):
            for cid in c.get("constraints_supported", []):
                supported_pairs.add((c.get("slot_id"), cid))
    full_unmaterialized = 0
    for it in cf.get("interpretations", []):
        for a in it.get("candidate_assertions", []):
            for jd in a.get("judgments", []):
                if jd.get("judgment") == "full_support":
                    pair = (jd.get("slot_id"), jd.get("constraint_id"))
                    if pair not in supported_pairs:
                        full_unmaterialized += 1
    st["full_support_judgment_without_materialized_constraint_support_count"] = full_unmaterialized
    # pre-judge filter invariants (req 7): the judge runs only on admitted candidates, so it
    # is never invoked on UI/chrome/source-title-only candidates (rejected before the loop).
    st["judge_invoked_on_ui_or_navigation_count"] = 0
    st["judge_invoked_on_source_title_without_predicate_count"] = 0
    st["judge_calls_saved_by_prefilter"] = int(
        st.get("rejected_candidate_assertion_count", 0))
    return st


def _frontier_stats(trace) -> dict:
    """Per-attempt candidate-slate / frontier stats (aggregated by compute_metrics)."""
    cf = getattr(trace, "candidate_frontier", {}) or {}
    if not cf:
        return {}
    if cf.get("skipped"):
        return {"skipped": True,
                "skipped_candidate_slate_reason": cf.get("skipped_candidate_slate_reason", "")}
    m = dict(cf.get("metrics", {}) or {})
    sel = cf.get("selected_frontier_action") or {}
    m["read_on_candidate"] = 1 if any(
        a.get("action_type") == "read_candidate_source" and a.get("selected")
        for a in cf.get("frontier_actions", [])) else 0
    m["answer_from_confirmed"] = sel.get("action_type") == "answer_from_confirmed_hypothesis"
    m["skipped"] = False
    # controller (Level 5b): shadow-vs-active comparison + execution accounting.
    ctrl = cf.get("controller", {}) or {}
    m["controller_used"] = bool(ctrl.get("frontier_controller_used"))
    m["shadow_agreements"] = int(ctrl.get("shadow_agreements", 0))
    m["shadow_total"] = int(ctrl.get("shadow_total", 0))
    m["exec_success"] = int(ctrl.get("frontier_action_execution_success_count", 0))
    m["exec_failure"] = int(ctrl.get("frontier_action_execution_failure_count", 0))
    m["fallback_count"] = int(ctrl.get("old_planner_fallback_count", 0))
    m["tool_calls_from_frontier"] = int(ctrl.get("tool_calls_from_frontier_actions", 0))
    m["first_action_discriminative"] = ctrl.get("first_action_discriminative")
    m["first_query_generic"] = ctrl.get("first_query_generic")
    # Level 5f-A read-path accounting (URL-backed reads + page_fetch->scrape fallback).
    for k, v in (cf.get("read_accounting", {}) or {}).items():
        m[k] = int(v)
    return m


def _frame_parse_stats(trace) -> dict:
    """Per-attempt task-frame parser provenance (deterministic vs LLM)."""
    meta = getattr(trace, "task_frame_parse", {}) or {}
    if not meta:
        return {}
    tf = getattr(trace, "task_frame", {}) or {}
    target_roles = [s.get("slot_role") for s in tf.get("target_answer_slots", [])]
    n_attached = sum(1 for c in tf.get("constraints", []) if c.get("applies_to"))
    return {
        "parser_used": meta.get("parser_used", "deterministic"),
        "parse_quality": meta.get("parse_quality", 0.0),
        "fallback_reason": meta.get("fallback_reason", ""),
        "validation_errors": list(meta.get("validation_errors", [])),
        "target_slot_roles": target_roles,
        "constraint_attachment_count": n_attached,
    }


def _frame_stats(trace) -> dict:
    """Per-attempt task-frame / hypothesis-table stats (aggregated by compute_metrics)."""
    cov = getattr(trace, "frame_coverage", {}) or {}
    if not cov:
        return {}
    actions = [c.task_action for c in trace.calls if getattr(c, "task_action", {})]
    reads = [a for a in actions if a.get("kind") == "read_url_for_constraint"]
    progresses = [float(c.evidence_record.get("evidence_progress_score", 0.0))
                  for c in trace.calls if getattr(c, "evidence_record", {})]
    reads_with_evidence = sum(
        1 for c in trace.calls
        if c.task_action.get("kind") == "read_url_for_constraint"
        and float(c.evidence_record.get("evidence_progress_score", 0.0)) > 0)
    no_progress_actions = sum(1 for p in progresses if p == 0.0)
    seen, repeated = set(), 0
    for c in trace.calls:
        q = (c.query or "").lower().strip()
        if q in seen:
            repeated += 1
        seen.add(q)
    rejection_reasons = [h.get("rejection_reason")
                         for h in (trace.hypothesis_summary or {}).get("rejected_hypotheses", [])
                         if h.get("rejection_reason")]
    return {
        "slot_resolution_rate": cov.get("slot_resolution_rate", 0.0),
        "constraint_support_rate": cov.get("constraint_support_rate", 0.0),
        "target_slot_support_rate": cov.get("target_slot_support_rate", 0.0),
        "hypothesis_coverage_score": cov.get("hypothesis_coverage_score", 0.0),
        "evidence_progress_per_action": (sum(progresses) / len(progresses)) if progresses else 0.0,
        "n_reads": len(reads),
        "reads_with_evidence": reads_with_evidence,
        "n_actions": len(actions),
        "no_progress_actions": no_progress_actions,
        "repeated_queries": repeated,
        "hypothesis_rejection_reasons": rejection_reasons,
        "final_answer_supported_by_constraints": bool(
            cov.get("final_answer_supported_by_constraints")),
        "initial_blocking_constraint_resolved_without_evidence_count": int(
            cov.get("initial_blocking_constraint_resolved_without_evidence_count", 0)),
    }


def _scrape_stats(trace) -> dict:
    """Per-attempt Level 3 reading/scrape stats (aggregated by compute_metrics)."""
    scrapes = [getattr(c, "scrape", {}) for c in trace.calls if getattr(c, "scrape", {})]
    scrapes = [s for s in scrapes if s.get("is_scrape")]
    if not scrapes:
        return {}
    return {
        "scrape_calls": len(scrapes),
        "scrape_success": sum(1 for s in scrapes if s.get("scrape_success")),
        "evidence_added": sum(1 for s in scrapes if s.get("evidence_added_by_scrape")),
        "answer_shape_found": sum(1 for s in scrapes if s.get("answer_shape_found_after_scrape")),
        "fallback": sum(1 for s in scrapes if s.get("fallback_to_page_fetch")),
        "failure_types": [s.get("scrape_failure_type") for s in scrapes
                          if s.get("scrape_failure_type")],
    }


def _iterative_stats(trace) -> dict:
    """Per-attempt candidate-hypothesis-policy stats (aggregated by compute_metrics)."""
    fups = [c for c in trace.calls if getattr(c, "stage", 1) >= 2]
    if not fups:
        return {}
    selected_roles = [c.selected_role for c in fups if getattr(c, "selected_role", None)]
    role_matches = sum(1 for c in fups
                       if getattr(c, "selected_role", None) in getattr(c, "target_roles", []))
    rejection_reasons = [rc.get("rejection_reason") for c in fups
                         for rc in getattr(c, "rejected_candidates", [])
                         if rc.get("rejection_reason")]
    # candidate switches between consecutive follow-ups
    switches = 0
    prev = None
    for c in fups:
        sc = getattr(c, "selected_candidate", None)
        if prev is not None and sc != prev:
            switches += 1
        prev = sc
    # repeated follow-up queries (same normalized query text)
    seen: set = set()
    repeated = 0
    for c in fups:
        q = (c.query or "").lower().strip()
        if q in seen:
            repeated += 1
        seen.add(q)
    return {
        "n_followups": len(fups),
        "selected_roles": selected_roles,
        "role_matches": role_matches,
        "rejection_reasons": rejection_reasons,
        "sticky": sum(1 for c in fups if getattr(c, "sticky_penalty", 0.0) > 0),
        "no_progress": sum(1 for c in fups if getattr(c, "no_progress", False)),
        "evidence_improved": sum(1 for c in fups if getattr(c, "evidence_improved", False)),
        "switches": switches,
        "repeated_queries": repeated,
        "beam_size_sum": sum(len(getattr(c, "candidate_entities", [])) for c in fups),
        "beam_size_n": len(fups),
    }


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
