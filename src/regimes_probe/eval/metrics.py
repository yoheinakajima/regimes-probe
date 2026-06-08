"""Metrics over a set of graded attempts.

Primary metric: ``correct_per_tool_call``. Plus accuracy, efficiency, and the
epistemic-error rates defined in ``docs/EVALUATION_PROTOCOL.md``.

An :class:`AttemptOutcome` is the per-attempt record the metrics aggregate. It
is produced by the eval harness from a trace + grade + reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Optional


@dataclass
class AttemptOutcome:
    item_id: str
    condition: str
    budget: int
    correct: bool
    abstained: bool
    tool_calls: int
    cost: float
    latency: float
    evidence_score: float
    first_tool_hit: bool
    false_stop: bool
    over_search: bool
    under_search: bool
    stale_error: bool
    found_hit: bool
    cluster_key: str
    attempt_reward: float
    authority_ok: bool = True
    contradiction: bool = False
    support_found: bool = True
    failed_tool_calls: int = 0
    failed_tools: list = field(default_factory=list)
    contaminated_results: int = 0
    total_results: int = 0
    # iterative clue resolution
    candidate_entity_count: int = 0
    followup_query_count: int = 0
    evidence_improved_after_followup: bool = False
    answer_found_after_stage: int = 0
    stage_depth_used: int = 1
    # candidate-hypothesis policy (per-attempt aggregables)
    iterative: dict = field(default_factory=dict)
    # Level 3 evidence reading (page_fetch vs firecrawl_scrape)
    scrape: dict = field(default_factory=dict)
    # Level 4 task-frame / hypothesis-table stats
    frame: dict = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "condition": self.condition,
            "budget": self.budget,
            "correct": int(self.correct),
            "abstained": int(self.abstained),
            "tool_calls": self.tool_calls,
            "cost": round(self.cost, 6),
            "latency": round(self.latency, 4),
            "evidence_score": round(self.evidence_score, 4),
            "first_tool_hit": int(self.first_tool_hit),
            "false_stop": int(self.false_stop),
            "over_search": int(self.over_search),
            "under_search": int(self.under_search),
            "stale_error": int(self.stale_error),
            "found_hit": int(self.found_hit),
            "cluster_key": self.cluster_key,
            "attempt_reward": round(self.attempt_reward, 6),
            "authority_ok": int(self.authority_ok),
            "contradiction": int(self.contradiction),
            "support_found": int(self.support_found),
            "failed_tool_calls": self.failed_tool_calls,
            "contaminated_results": self.contaminated_results,
            "followup_query_count": self.followup_query_count,
            "stage_depth_used": self.stage_depth_used,
            "regime": "",  # filled by the regime detectors if run
        }


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _iterative_metrics(stats: list[dict]) -> dict[str, Any]:
    """Aggregate candidate-hypothesis-policy stats across a cell's attempts."""
    from collections import Counter
    role_counts: Counter = Counter()
    rejection_counts: Counter = Counter()
    followups = role_matches = sticky = no_progress = improved = 0
    switches = repeated = beam_sum = beam_n = 0
    for s in stats:
        if not s:
            continue
        followups += s.get("n_followups", 0)
        role_matches += s.get("role_matches", 0)
        sticky += s.get("sticky", 0)
        no_progress += s.get("no_progress", 0)
        improved += s.get("evidence_improved", 0)
        switches += s.get("switches", 0)
        repeated += s.get("repeated_queries", 0)
        beam_sum += s.get("beam_size_sum", 0)
        beam_n += s.get("beam_size_n", 0)
        role_counts.update(s.get("selected_roles", []))
        rejection_counts.update(s.get("rejection_reasons", []))
    return {
        "selected_candidate_role_counts": dict(role_counts),
        "candidate_role_match_rate": _safe_div(role_matches, followups),
        "candidate_rejection_counts": dict(rejection_counts),
        "sticky_candidate_count": sticky,
        "no_progress_followup_count": no_progress,
        "evidence_improved_after_candidate_rate": _safe_div(improved, followups),
        "mean_beam_size": _safe_div(beam_sum, beam_n),
        "candidate_switch_count": switches,
        "repeated_candidate_query_count": repeated,
    }


def _scrape_metrics(outcomes: list["AttemptOutcome"]) -> dict[str, Any]:
    """Aggregate Level 3 reading/scrape stats across a cell's attempts."""
    from collections import Counter
    failure_counts: Counter = Counter()
    scrape_calls = success = evidence_added = shape_found = fallback = 0
    scraped_attempts = scraped_correct = 0
    for o in outcomes:
        s = o.scrape or {}
        if not s:
            continue
        scrape_calls += s.get("scrape_calls", 0)
        success += s.get("scrape_success", 0)
        evidence_added += s.get("evidence_added", 0)
        shape_found += s.get("answer_shape_found", 0)
        fallback += s.get("fallback", 0)
        failure_counts.update(s.get("failure_types", []))
        if s.get("scrape_calls", 0) > 0:
            scraped_attempts += 1
            if o.correct:
                scraped_correct += 1
    return {
        "scrape_call_count": scrape_calls,
        "scrape_success_count": success,
        "scrape_success_rate": _safe_div(success, scrape_calls),
        "evidence_added_by_scrape_rate": _safe_div(evidence_added, scrape_calls),
        "answer_shape_found_after_scrape_count": shape_found,
        "scrape_failure_counts": dict(failure_counts),
        "scrape_fallback_count": fallback,
        "scrape_to_answer_rate": _safe_div(scraped_correct, scraped_attempts),
    }


def _frame_metrics(stats: list[dict]) -> dict[str, Any]:
    """Aggregate Level 4 task-frame / hypothesis-table stats across a cell."""
    from collections import Counter
    used = [s for s in stats if s]
    if not used:
        return {}
    rejection_counts: Counter = Counter()
    for s in used:
        rejection_counts.update(s.get("hypothesis_rejection_reasons", []))

    def avg(key: str) -> float:
        vals = [s.get(key, 0.0) for s in used]
        return sum(vals) / len(vals) if vals else 0.0

    n_reads = sum(s.get("n_reads", 0) for s in used)
    reads_with_evidence = sum(s.get("reads_with_evidence", 0) for s in used)
    n_actions = sum(s.get("n_actions", 0) for s in used)
    no_progress_actions = sum(s.get("no_progress_actions", 0) for s in used)
    repeated_queries = sum(s.get("repeated_queries", 0) for s in used)
    return {
        "slot_resolution_rate": avg("slot_resolution_rate"),
        "constraint_support_rate": avg("constraint_support_rate"),
        "target_slot_support_rate": avg("target_slot_support_rate"),
        "hypothesis_coverage_score": avg("hypothesis_coverage_score"),
        "evidence_progress_per_action": avg("evidence_progress_per_action"),
        "read_value_precision": _safe_div(reads_with_evidence, n_reads),
        "no_progress_action_rate": _safe_div(no_progress_actions, n_actions),
        "repeated_equivalent_query_rate": _safe_div(repeated_queries, n_actions),
        "hypothesis_rejection_counts": dict(rejection_counts),
        "final_answer_supported_by_constraints_rate": _safe_div(
            sum(1 for s in used if s.get("final_answer_supported_by_constraints")), len(used)),
    }


def compute_metrics(outcomes: list[AttemptOutcome]) -> dict[str, Any]:
    """Aggregate metrics for one condition+budget cell."""
    n = len(outcomes)
    if n == 0:
        return {"n": 0}
    correct = sum(o.correct for o in outcomes)
    calls = sum(o.tool_calls for o in outcomes)
    cost = sum(o.cost for o in outcomes)
    latency = sum(o.latency for o in outcomes)
    answered = [o for o in outcomes if not o.abstained]
    abstained = [o for o in outcomes if o.abstained]
    # Abstention precision/recall: an abstention is "correct" when the agent
    # would otherwise have been wrong; precision = abstained that were right to,
    # recall = of all would-be-wrong, how many abstained.
    would_be_wrong = sum(1 for o in outcomes if not o.correct)
    correct_abstentions = sum(1 for o in abstained if not o.found_hit)
    failed_calls = sum(o.failed_tool_calls for o in outcomes)
    contaminated = sum(o.contaminated_results for o in outcomes)
    total_results = sum(o.total_results for o in outcomes)
    failed_by_tool: dict[str, int] = {}
    for o in outcomes:
        for t in o.failed_tools:
            failed_by_tool[t] = failed_by_tool.get(t, 0) + 1
    return {
        "n": n,
        "accuracy": _safe_div(correct, n),
        "failed_tool_calls": failed_calls,
        "failed_tool_call_count": failed_calls,
        "provider_failure_rate": _safe_div(failed_calls, calls),
        "failed_tool_calls_by_tool": failed_by_tool,
        "support_found_rate": _safe_div(sum(1 for o in outcomes if o.support_found), n),
        "evidence_score_mean": _safe_div(sum(o.evidence_score for o in outcomes), n),
        "benchmark_contaminated_result_count": contaminated,
        "contamination_rate": _safe_div(contaminated, total_results),
        # iterative clue resolution
        "mean_candidate_entity_count": _safe_div(sum(o.candidate_entity_count for o in outcomes), n),
        "mean_followup_query_count": _safe_div(sum(o.followup_query_count for o in outcomes), n),
        "evidence_improved_after_followup_rate": _safe_div(
            sum(1 for o in outcomes if o.evidence_improved_after_followup), n),
        "mean_stage_depth_used": _safe_div(sum(o.stage_depth_used for o in outcomes), n),
        "answer_found_after_stage_mean": _safe_div(
            sum(o.answer_found_after_stage for o in outcomes if o.answer_found_after_stage),
            sum(1 for o in outcomes if o.answer_found_after_stage)),
        # candidate-hypothesis policy aggregates
        **_iterative_metrics([o.iterative for o in outcomes]),
        # Level 3 scrape/read aggregates
        **_scrape_metrics(outcomes),
        # Level 4 task-frame / hypothesis aggregates
        **_frame_metrics([o.frame for o in outcomes]),
        "correct_per_tool_call": _safe_div(correct, calls),
        "correct_per_dollar": _safe_div(correct, cost) if cost else 0.0,
        "correct_per_second": _safe_div(correct, latency) if latency else 0.0,
        "total_tool_calls": calls,
        "mean_tool_calls": _safe_div(calls, n),
        "first_tool_hit_rate": _safe_div(sum(o.first_tool_hit for o in outcomes), n),
        "evidence_gain_per_call": _safe_div(sum(o.evidence_score for o in outcomes), calls),
        "primary_or_authoritative_source_rate": _safe_div(
            sum(1 for o in outcomes if o.found_hit), n
        ),
        "stale_source_error_rate": _safe_div(sum(o.stale_error for o in outcomes), n),
        "false_stop_rate": _safe_div(sum(o.false_stop for o in outcomes), n),
        "over_search_rate": _safe_div(sum(o.over_search for o in outcomes), n),
        "under_search_rate": _safe_div(sum(o.under_search for o in outcomes), n),
        "abstention_rate": _safe_div(len(abstained), n),
        "abstention_precision": _safe_div(correct_abstentions, len(abstained)) if abstained else 0.0,
        "abstention_recall": _safe_div(correct_abstentions, would_be_wrong) if would_be_wrong else 0.0,
        "mean_attempt_reward": mean(o.attempt_reward for o in outcomes),
    }
