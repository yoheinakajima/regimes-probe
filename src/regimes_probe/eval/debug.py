"""Per-question DEBUG records — bounded previews to diagnose why a run failed.

These carry the question/gold/prediction previews, the tool sequence + provider
names, per-call result counts and sanitized errors, top evidence (title/url/
snippet previews), the regime, and an inferred *failure seam*. Previews are
length-bounded; full pages are never stored. Error messages come from the
already-sanitized provider ``error_meta`` (no secrets).

This is a debug artifact (``results/{run_id}/debug_questions.jsonl``), separate
from the answer-free policy memory — it is allowed to contain gold/prediction
previews for the operator's eyes (live run dirs are git-ignored).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

_PREVIEW = 300          # default chars for question/answer previews
_GOLD = 120             # chars for the gold-answer preview (audit only)
_SNIPPET = 300          # chars for an evidence snippet preview
_MAX_EVIDENCE = 3       # top evidence rows kept per question


def _prev(s: Optional[str], n: int = _PREVIEW) -> str:
    if not s:
        return ""
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _bounded_frontier(cf: dict[str, Any], *, max_events: int = 80) -> dict[str, Any]:
    """Bound the candidate-frontier debug dict (cap the event list)."""
    if not cf:
        return {}
    out = dict(cf)
    evs = out.get("events") or []
    if len(evs) > max_events:
        out["events"] = evs[:max_events]
        out["events_truncated"] = len(evs)
    return out


#: Failure seams (ordered most-specific first).
SEAMS = ("provider_error", "provider_returned_no_results", "evidence_absent",
         "evidence_not_selected", "answer_extraction", "grader_strictness",
         "exact_answer_missing", "weak_query", "unknown")


def infer_failure_seam(*, correct: bool, abstained: bool, prediction: Optional[str],
                       gold_norms: list[str], pred_norm: str, n_tool_calls: int,
                       n_failed_calls: int, n_results: int, found_hit: bool,
                       support_found: bool) -> str:
    """Best-effort label for where a wrong/empty answer went off the rails."""
    if correct:
        return "ok"
    if n_failed_calls > 0 and n_results == 0:
        return "provider_error"
    if n_failed_calls > 0:
        return "provider_error"
    if n_tool_calls > 0 and n_results == 0:
        return "provider_returned_no_results"
    if found_hit and not correct:
        return "answer_extraction"          # had the answer-bearing evidence, answered wrong
    if abstained or not (prediction and prediction.strip()):
        return "evidence_absent" if n_results == 0 else "exact_answer_missing"
    if not support_found:
        return "evidence_not_selected"      # results returned but none selected as support
    # prediction present, supported, still wrong: close-but-graded-wrong vs weak query
    if pred_norm and any(pred_norm in g or g in pred_norm for g in gold_norms if g):
        return "grader_strictness"
    if n_results > 0:
        return "weak_query"
    return "unknown"


@dataclass
class DebugRecord:
    item_id: str
    condition: str
    budget: int
    question_preview: str
    gold_preview: str
    prediction_preview: str
    correct: bool
    abstained: bool
    tool_sequence: list[str]
    provider_names: list[str]
    calls: list[dict[str, Any]]
    failed_tool_errors: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    regime: str
    regime_names: list[str]
    support_found: bool
    found_hit: bool
    authority_ok: bool
    contradiction: bool
    n_results: int
    failed_tool_calls: int
    failure_seam: str
    contaminated_results: int = 0
    stage_depth_used: int = 1
    followup_query_count: int = 0
    target_roles: list[str] = field(default_factory=list)
    task_frame: dict[str, Any] = field(default_factory=dict)
    hypothesis_summary: dict[str, Any] = field(default_factory=dict)
    frame_coverage: dict[str, Any] = field(default_factory=dict)
    task_frame_parse: dict[str, Any] = field(default_factory=dict)
    epistemic_mode: dict[str, Any] = field(default_factory=dict)
    candidate_frontier: dict[str, Any] = field(default_factory=dict)
    evidence_titles: list[str] = field(default_factory=list)
    evidence_urls: list[str] = field(default_factory=list)
    evidence_snippet_previews: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_debug_record(*, item, trace, grade, reward, condition: str, budget: int,
                       outcome=None, preview: int = _PREVIEW,
                       gold_chars: int = _GOLD,
                       snippet_chars: int = _SNIPPET,
                       max_evidence: int = _MAX_EVIDENCE) -> DebugRecord:
    """Assemble a bounded debug record from one attempt."""
    from regimes_probe.regimes.detectors import label_outcome
    from regimes_probe.eval.grader import normalize_answer

    # Gold tokens for an AUDIT-only "result contains the exact answer" flag. This
    # is a debug artifact (may contain gold); policy memory stays answer-free.
    _gold_audit = [normalize_answer(g) for g in
                   (item.gold_answers() if hasattr(item, "gold_answers") else [])]
    _gold_audit = [g for g in _gold_audit if g]

    def _contains_gold(*parts: str) -> bool:
        hay = normalize_answer(" ".join(p for p in parts if p))
        return any(g and g in hay for g in _gold_audit)

    calls_info: list[dict[str, Any]] = []
    failed_errors: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    n_results = 0
    contaminated_results = 0
    for c in trace.calls:
        n_ok = sum(1 for o in c.observations if not getattr(o, "failed", False))
        n_results += n_ok
        c_cont = sum(1 for o in c.observations if getattr(o, "benchmark_contaminated", False))
        contaminated_results += c_cont
        cand_prev = [{"arm": q.get("arm"), "query_preview": _prev(q.get("query", ""), preview),
                      "expected_search_quality": q.get("expected_search_quality")}
                     for q in getattr(c, "query_candidates", [])[:6]]
        # Candidate hypotheses (bounded): role + raw/adjusted scores + penalties.
        ent_prev = [{"candidate_text": _prev(e.get("candidate_text", e.get("text", "")), 80),
                     "role": e.get("role"), "raw_score": e.get("raw_score"),
                     "adjusted_score": e.get("adjusted_score"),
                     "role_match_score": e.get("role_match_score"),
                     "source_entity_penalty": e.get("source_entity_penalty"),
                     "location_penalty": e.get("location_penalty"),
                     "genericity_penalty": e.get("genericity_penalty"),
                     "sticky_penalty": e.get("sticky_penalty")}
                    for e in getattr(c, "candidate_entities", [])[:6]]
        rej_prev = [{"candidate_text": _prev(e.get("candidate_text", ""), 80),
                     "role": e.get("role"), "rejection_reason": e.get("rejection_reason"),
                     "adjusted_score": e.get("adjusted_score")}
                    for e in getattr(c, "rejected_candidates", [])[:6]]
        calls_info.append({
            "call_index": c.call_index, "tool": c.tool, "query_arm": c.query_arm,
            "query_preview": _prev(c.query, preview),
            "query_text_hash": getattr(c, "query_text_hash", ""),
            "clue_ids": list(getattr(c, "clue_ids", [])),
            "query_quality": round(float(getattr(c, "query_quality", 0.0)), 3),
            "n_query_candidates": getattr(c, "n_query_candidates", 0),
            "n_query_candidates_dropped": getattr(c, "n_query_candidates_dropped", 0),
            "query_candidates": cand_prev,
            # iterative clue resolution (typed candidate hypotheses)
            "stage": getattr(c, "stage", 1),
            "parent_query_id": getattr(c, "parent_query_id", None),
            "target_roles": list(getattr(c, "target_roles", [])),
            "candidate_entities": ent_prev,
            "selected_candidate": getattr(c, "selected_candidate", None),
            "selected_role": getattr(c, "selected_role", None),
            "selection_reason": _prev(getattr(c, "selection_reason", "") or "", preview),
            "rejected_candidates": rej_prev,
            "sticky_penalty": round(float(getattr(c, "sticky_penalty", 0.0)), 3),
            "no_progress": bool(getattr(c, "no_progress", False)),
            "evidence_improved": bool(getattr(c, "evidence_improved", False)),
            # Level 3 reading (page_fetch vs firecrawl_scrape); bounded counts only
            "scrape": dict(getattr(c, "scrape", {})),
            # Level 4 task-frame action + evidence record
            "task_action": dict(getattr(c, "task_action", {})),
            "evidence_record": dict(getattr(c, "evidence_record", {})),
            "n_results": n_ok, "contaminated_results": c_cont,
            "failed": bool(getattr(c, "failed", False)),
            "error_type": getattr(c, "error_type", None),
            "status_code": getattr(c, "status_code", None),
        })
        if getattr(c, "failed", False):
            err_obs = next((o for o in c.observations if getattr(o, "failed", False)), None)
            failed_errors.append({
                "tool": c.tool, "error_type": getattr(c, "error_type", None),
                "status_code": getattr(c, "status_code", None),
                "message_preview": _prev(getattr(err_obs, "error_message", "") if err_obs else "", preview),
            })
        for o in c.observations:
            if getattr(o, "failed", False) or len(evidence) >= max_evidence:
                continue
            evidence.append({
                "title_preview": _prev(getattr(o, "title", ""), preview),
                "url": (o.url or "")[:300],
                "snippet_preview": _prev(o.snippet, snippet_chars),
                "supports": bool(o.supports),
                "source_authority": round(float(o.source_authority), 3),
                "benchmark_contaminated": bool(getattr(o, "benchmark_contaminated", False)),
                "contamination_reason": getattr(o, "contamination_reason", None),
                "contains_gold": _contains_gold(getattr(o, "title", ""), o.url, o.snippet),
            })

    gold_norms = [g for g in (grade.gold_norm or [])]
    pred_norm = grade.prediction_norm or ""
    seam = infer_failure_seam(
        correct=grade.correct, abstained=grade.abstained, prediction=trace.final_answer,
        gold_norms=gold_norms, pred_norm=pred_norm, n_tool_calls=trace.tool_calls,
        n_failed_calls=sum(1 for c in trace.calls if getattr(c, "failed", False)),
        n_results=n_results, found_hit=reward.flags.get("found_hit", False),
        support_found=trace.vstate.support_found)

    gold = "; ".join(item.gold_answers()) if hasattr(item, "gold_answers") else ""
    record = DebugRecord(
        item_id=item.id, condition=condition, budget=budget,
        question_preview=_prev(item.question, preview),
        gold_preview=_prev(gold, gold_chars),
        prediction_preview=_prev(trace.final_answer, preview),
        correct=grade.correct, abstained=grade.abstained,
        tool_sequence=trace.tools_used(), provider_names=sorted(set(trace.tools_used())),
        calls=calls_info, failed_tool_errors=failed_errors, evidence=evidence,
        regime=(label_outcome(outcome) if outcome is not None else ""),
        regime_names=[],
        support_found=trace.vstate.support_found,
        found_hit=reward.flags.get("found_hit", False),
        authority_ok=trace.vstate.authority_ok, contradiction=trace.vstate.contradiction,
        n_results=n_results,
        failed_tool_calls=sum(1 for c in trace.calls if getattr(c, "failed", False)),
        failure_seam=seam,
        contaminated_results=contaminated_results,
        stage_depth_used=max((getattr(c, "stage", 1) for c in trace.calls), default=1),
        followup_query_count=sum(1 for c in trace.calls if getattr(c, "stage", 1) >= 2),
        target_roles=list(next((getattr(c, "target_roles", []) for c in trace.calls
                                if getattr(c, "target_roles", [])), [])),
        task_frame=dict(getattr(trace, "task_frame", {}) or {}),
        hypothesis_summary=dict(getattr(trace, "hypothesis_summary", {}) or {}),
        frame_coverage=dict(getattr(trace, "frame_coverage", {}) or {}),
        task_frame_parse=dict(getattr(trace, "task_frame_parse", {}) or {}),
        epistemic_mode=dict(getattr(trace, "epistemic_mode", {}) or {}),
        candidate_frontier=_bounded_frontier(getattr(trace, "candidate_frontier", {}) or {}),
        evidence_titles=[e["title_preview"] for e in evidence],
        evidence_urls=[e["url"] for e in evidence],
        evidence_snippet_previews=[e["snippet_preview"] for e in evidence])
    # Structured canonical regime names (seam + answer-free detectors).
    from regimes_probe.eval.failure_regime import regime_names as _regime_names
    record.regime_names = _regime_names(record.to_dict())
    return record


def write_debug_jsonl(path, records: list[DebugRecord]) -> None:
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
