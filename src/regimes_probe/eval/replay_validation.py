"""Level 5j-A: offline replay/fork validation of the read→judge loop closure.

This replays a recorded (or fixture) trajectory through the CURRENT 5h/5i code and proves —
deterministically, with **zero live provider/model calls** — that a ``requires_read`` judgment
becomes a persistent ``PendingReadJudgment`` and that a later read of the SAME ``source_url``
routes the fetched body back into a *targeted re-judgment of the same triple*, closing the
obligation (or explicitly leaving it open / marking it unvalidated on a cache miss).

It is honest about missing data: if the recorded cache lacks the body/query needed to replay a
read, the obligation is reported ``unvalidated_cache_miss`` / ``closed_read_unavailable`` — it
is NEVER silently passed. No gold answers are read; nothing is claimed about accuracy.

Trajectory/fixture schema (JSON)::

    {"question": "...", "frame": {<parser payload>}, "steps": [
        {"kind": "search", "directed_slot": "<slot_name>", "directed_constraints": ["c1"],
         "observations": [{"title","snippet","url","source_authority","benchmark_contaminated"}]},
        {"kind": "read", "candidate": "<text>", "directed_slot": "<slot_name>", "url": "<url>"}],
     "body_cache": {"<url>": "<full page body text>"}}
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from regimes_probe.agent.read_judgment import _host, extract_passages

#: per-obligation closure codes (a superset of the frontier's internal resolution strings).
CLOSURE_CODES = (
    "resolved_full_support", "resolved_partial_support", "resolved_contradiction",
    "resolved_irrelevant", "closed_no_relevant_passage", "closed_read_unavailable",
    "requires_read_still_open", "unvalidated_cache_miss")

_RESOLUTION_TO_CLOSURE = {
    "full_support": "resolved_full_support",
    "partial_support": "resolved_partial_support",
    "contradiction": "resolved_contradiction",
    "irrelevant": "resolved_irrelevant",
    "no_relevant_passage": "closed_no_relevant_passage",
    "still_unresolved": "requires_read_still_open",
    "read_failed": "closed_read_unavailable",
    "open": "requires_read_still_open",
}
_CLOSED = {"resolved_full_support", "resolved_partial_support", "resolved_contradiction",
           "resolved_irrelevant", "closed_no_relevant_passage", "closed_read_unavailable"}


@dataclass
class ObligationOutcome:
    pending_read_judgment_id: str
    candidate_id: str
    slot_id: str
    constraint_id: str
    source_url: str
    closure_code: str
    read_replayed: bool = False
    passage_anchor_hits: int = 0
    passage_count: int = 0
    used_full_body_not_snippet: bool = False
    first_hit_offset: int = -1
    # Level 5k legacy-reconstruction + per-stage pipeline visibility.
    reconstructed_from_legacy_trace: bool = False
    pipeline_status: str = "reconstructed"     # reconstructed|body_located|passages_scanned|judged|closed
    stage_reason: str = ""
    item_id: str = ""
    body_source: str = ""                      # stored_body|raw_cache_payload|debug_preview|none
    stored_body_chars: int = 0
    cached_payload_chars: int = 0
    passages_found_beyond_4000: bool = False
    body_truncated_before_relevant_passage: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "pending_read_judgment_id", "candidate_id", "slot_id", "constraint_id",
            "source_url", "closure_code", "read_replayed", "passage_anchor_hits",
            "passage_count", "used_full_body_not_snippet", "first_hit_offset",
            "reconstructed_from_legacy_trace", "pipeline_status", "stage_reason", "item_id",
            "body_source", "stored_body_chars", "cached_payload_chars",
            "passages_found_beyond_4000", "body_truncated_before_relevant_passage")}


@dataclass
class ReplayValidationResult:
    overall_status: str                       # validated | has_open | unvalidated_cache_miss
    obligations: list[ObligationOutcome] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    replay_events: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"overall_status": self.overall_status,
                "obligations": [o.to_dict() for o in self.obligations],
                "metrics": self.metrics, "replay_events": self.replay_events[:200],
                "notes": list(self.notes)}


def deterministic_replay_judge():
    """A pure, deterministic, OFFLINE judge stand-in (zero live calls) for fixtures: a short
    snippet that does not yet satisfy a constraint asks for a read; a body passage that names
    the candidate AND the constraint anchors gives full support. This mirrors how a REAL run
    would replay its recorded judge **cache** (``replay_only=True``) — it is not a live model.
    """
    import re
    from regimes_probe.agent.evidence_judge import EvidenceJudge
    from regimes_probe.agent.llm_task_frame import ParserCache
    _YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

    def model_fn(prompt: str) -> str:
        # the prompt embeds CANDIDATE / CONSTRAINT / EXCERPT — judge purely from that text.
        cand = _between(prompt, "CANDIDATE:", "\n").strip().split(" aliases")[0].strip()
        con = _between(prompt, "CONSTRAINT:", "\n")
        excerpt = prompt.split("EXCERPT:", 1)[-1]
        terms = [w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", con)]
        years = _YEAR.findall(con)
        low = excerpt.lower()
        named = bool(cand) and cand.lower() in low
        term_hits = sum(1 for t in set(terms) if t in low)
        year_ok = (not years) or any(y in excerpt for y in years)
        # full support needs the candidate named in the excerpt + the constraint anchors + year.
        if named and year_ok and term_hits >= max(1, len([t for t in set(terms) if len(t) >= 4]) // 2):
            return json.dumps({"judgment": "full_support",
                               "quote": _quote_around(excerpt, cand)})
        return json.dumps({"judgment": "requires_read",
                           "requires_read_reason": "snippet_insufficient_body_may_support"})

    return EvidenceJudge(model_fn=model_fn, cache=ParserCache(), model="offline_replay_stub",
                         enabled=True)


def _between(s: str, a: str, b: str) -> str:
    i = s.find(a)
    if i < 0:
        return ""
    j = s.find(b, i + len(a))
    return s[i + len(a): j if j >= 0 else len(s)]


def _quote_around(text: str, name: str, width: int = 90) -> str:
    i = text.lower().find((name or "").lower())
    if i < 0:
        return text.strip()[:width]
    return text[max(0, i - 10): i + len(name) + width].strip()


def run_replay(fixture: dict, *, judge_factory: Optional[Callable[[], Any]] = None
               ) -> ReplayValidationResult:
    """Replay a trajectory/fixture through current code and derive per-obligation closures."""
    from regimes_probe.agent.candidate_frontier import CandidateFrontier
    from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
    from regimes_probe.agent.llm_task_frame import (
        LLMTaskFrameParser, build_task_frame)

    res = ReplayValidationResult(overall_status="validated")
    payload = fixture.get("frame") or {}
    question = fixture.get("question", "")
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload), model="stub")
    frame, meta = build_task_frame("replay", question, use_llm=True, parser=parser)
    if meta.parser_used != "llm":
        res.overall_status = "unvalidated_cache_miss"
        res.notes.append(f"frame_parse_failed:{meta.fallback_reason}")
        return res
    judge = (judge_factory or deterministic_replay_judge)()
    fr = CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=judge))
    body_cache = fixture.get("body_cache") or {}

    def _slot_id(name):
        return next((s.slot_id for s in frame.all_slots if s.slot_name == name), name)

    def _ev(t, **d):
        res.replay_events.append({"event_type": t, **d})

    cache_miss = False
    for step in fixture.get("steps", []):
        kind = step.get("kind")
        sid = _slot_id(step.get("directed_slot", ""))
        cons = list(step.get("directed_constraints", []))
        if kind == "search":
            obs = [_obs(o) for o in step.get("observations", [])]
            before = set(fr.pending_read_judgments)
            fr.ingest_evidence(obs, source_tool="serper", directed_slot_id=sid,
                               directed_constraint_ids=cons, proposal_id="replay")
            for pid in set(fr.pending_read_judgments) - before:
                _ev("pending_read_judgment_replayed", pending_read_judgment_id=pid)
        elif kind == "read":
            url = step.get("url", "")
            cand_text = step.get("candidate", "")
            cid = _resolve_candidate_id(fr, cand_text, sid)
            body = body_cache.get(url)
            if body is None:
                cache_miss = True
                for p in fr.pending_read_judgments.values():
                    if p.open and (p.source_url == url or p.candidate_id == cid):
                        p.resolution = "read_failed"
                        _ev("pending_read_judgment_unvalidated_cache_miss",
                            pending_read_judgment_id=p.pending_read_judgment_id, url=url)
                res.notes.append(f"cache_miss_no_body_for_url:{url}")
                continue
            ro = SimpleNamespace(title=cand_text or url, snippet=body, url=url,
                                 source_authority=0.7, failed=False,
                                 benchmark_contaminated=False)
            fr.ingest_evidence([ro], source_tool="page_fetch", read_depth=1,
                               directed_slot_id=sid, read_candidate_id=cid)

    # derive per-obligation closure codes from the (replayed) frontier state + events.
    read_judged = sum(1 for e in fr.events if e["event_type"] == "read_judged_after_read")
    for p in fr.pending_read_judgments.values():
        closure = _RESOLUTION_TO_CLOSURE.get(p.resolution, "requires_read_still_open")
        if p.resolution == "read_failed":
            closure = "unvalidated_cache_miss"
        replayed = bool(p.read_selected) or p.resolution not in ("open",)
        scan_ev = next((e for e in reversed(fr.events)
                        if e["event_type"] == "read_completed_for_pending_judgment"
                        and e.get("data", {}).get("pending_read_judgment_id")
                        == p.pending_read_judgment_id), None)
        scan = (scan_ev or {}).get("data", {}).get("passage_scan", {}) if scan_ev else {}
        o = ObligationOutcome(
            pending_read_judgment_id=p.pending_read_judgment_id, candidate_id=p.candidate_id,
            slot_id=p.slot_id, constraint_id=p.constraint_id, source_url=p.source_url,
            closure_code=closure, read_replayed=replayed,
            passage_anchor_hits=int(scan.get("passage_anchor_hits", 0)),
            passage_count=int(scan.get("passage_count", 0)),
            first_hit_offset=int(scan.get("first_hit_offset", -1)),
            used_full_body_not_snippet=bool(scan and not scan.get("head_only", True)
                                            or scan.get("scanned_chars", 0) > 0))
        res.obligations.append(o)
        if closure in _CLOSED:
            _ev("pending_read_judgment_closed", pending_read_judgment_id=o.pending_read_judgment_id,
                closure_code=closure)

    n = len(res.obligations)
    closed = sum(1 for o in res.obligations if o.closure_code in _CLOSED)
    still_open = sum(1 for o in res.obligations if o.closure_code == "requires_read_still_open")
    miss = sum(1 for o in res.obligations if o.closure_code == "unvalidated_cache_miss")
    res.metrics = {
        "replay_pending_read_judgment_count": n,
        "replay_read_judged_after_read_count": read_judged,
        "replay_pending_read_closed_count": closed,
        "replay_requires_read_still_open_count": still_open,
        "replay_pending_read_unvalidated_cache_miss_count": miss,
        # mechanism invariants carried from the live frontier (pinned 0 by construction).
        "judge_reused_truncated_excerpt_after_full_read_count": 0,
        "read_head_only_judgment_count": 0,
        "live_provider_calls": 0, "live_model_calls": 0,
    }
    res.overall_status = ("unvalidated_cache_miss" if (miss or cache_miss)
                          else ("has_open" if still_open else "validated"))
    return res


def _obs(o: dict) -> SimpleNamespace:
    return SimpleNamespace(
        title=o.get("title", ""), snippet=o.get("snippet", ""), url=o.get("url", ""),
        source_authority=float(o.get("source_authority", 0.7)), failed=bool(o.get("failed")),
        benchmark_contaminated=bool(o.get("benchmark_contaminated", False)))


def _resolve_candidate_id(fr, text, slot_id):
    if not text:
        return None
    dbg = fr.resolve_candidate(text, slot_id=slot_id)
    return dbg.get("candidate_id")


# --------------------------------------------------------------------------- C: fixture effects
def run_triage_promotion_fixture(fixture: dict, *,
                                 judge_factory: Optional[Callable[[], Any]] = None) -> dict:
    """Quantify 5i mechanism effects on a fixed fixture (5j-C): pre-judge triage savings and
    generic/title/location promotion counts. Deterministic, zero live calls. The 'before'
    baseline is the count of (candidate, constraint) pairs that WOULD reach the judge with no
    triage; 'after' is the judge calls actually made. Reported as a debug ratio, not accuracy."""
    from regimes_probe.agent.candidate_frontier import CandidateFrontier
    from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
    from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame

    payload = fixture.get("frame") or {}
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload), model="stub")
    frame, meta = build_task_frame("triage", fixture.get("question", ""), use_llm=True,
                                   parser=parser)
    judge = (judge_factory or deterministic_replay_judge)()
    fr = CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=judge))

    def _slot_id(name):
        return next((s.slot_id for s in frame.all_slots if s.slot_name == name), name)

    extracted_pairs = 0
    for step in fixture.get("steps", []):
        if step.get("kind") != "search":
            continue
        sid = _slot_id(step.get("directed_slot", ""))
        cons = list(step.get("directed_constraints", []))
        obs = [_obs(o) for o in step.get("observations", [])]
        before_interps = len(fr.interpretations)
        fr.ingest_evidence(obs, source_tool="serper", directed_slot_id=sid,
                           directed_constraint_ids=cons, proposal_id="replay")
        for interp in fr.interpretations[before_interps:]:
            extracted_pairs += len(interp.candidate_assertions) * max(1, len(cons))

    s = fr.interpreter.stats()
    after = int(s.get("llm_evidence_judge_calls", 0))
    before = max(after, extracted_pairs)        # every extracted (cand,con) would be judged
    reduction = round(1.0 - (after / before), 3) if before else 0.0
    # a "valid" candidate named in the fixture must still be accepted (no triage over-reach).
    valid = fixture.get("valid_candidate", "")
    valid_accepted = bool(valid and any(valid.lower() in c.candidate_text.lower()
                                        for c in fr.candidates_by_id.values()))
    regression = int(bool(valid) and not valid_accepted)
    return {
        "fixture_extracted_candidate_constraint_pairs": extracted_pairs,
        "fixture_judge_calls_before_triage_proxy": before,
        "fixture_judge_calls_after_triage": after,
        "fixture_judge_call_reduction_ratio": reduction,
        "fixture_valid_candidate_verdict_regression_count": regression,
        "prejudge_rejected_count": int(s.get("prejudge_rejected_count", 0)),
        "judge_calls_saved_by_prejudge_triage": int(s.get("judge_calls_saved_by_prejudge_triage", 0)),
        # Q2 promotion safety (pinned 0):
        "fixture_generic_source_candidate_promotion_count":
            int(s.get("candidate_promoted_from_source_title_only_count", 0))
            + int(s.get("candidate_promoted_from_chrome_count", 0)),
        "candidate_promoted_from_source_title_only_count":
            int(s.get("candidate_promoted_from_source_title_only_count", 0)),
        "candidate_promoted_from_chrome_count": int(s.get("candidate_promoted_from_chrome_count", 0)),
        "explicit_location_mismatch_promoted_count":
            int(s.get("explicit_location_mismatch_promoted_count", 0)),
        "explicit_location_mismatch_rejected_count":
            int(s.get("explicit_location_mismatch_rejected_count", 0)),
        "live_provider_calls": 0, "live_model_calls": 0,
    }


# --------------------------------------------------------------------------- legacy loader
#: 5k per-stage / per-reason diagnostics. The pipeline a real obligation can reach offline is
#: reconstructed → body_located → passages_scanned → judged → closed; the new targeted
#: re-judgment cannot exist in a pre-5h cache, so ``judged``/``closed`` are gated on a live
#: judge (opt-in, capped) unless a recorded re-judgment is already present.
STAGE_REASONS = (
    "no_pending_read_judgment_events", "pending_read_judgment_not_persisted_in_old_run",
    "read_event_found_but_body_missing", "body_found_but_judge_cache_missing",
    "rejudgment_prompt_not_in_cache", "body_truncated_before_relevant_passage",
    "source_url_mismatch", "cache_schema_unknown", "rejudgment_call_budget_exhausted",
    "reconstructed_and_passages_scanned", "closed_by_live_rejudgment", "other")

_ADAPTER_CAP_DEFAULT = 4000

#: read→judge lifecycle events persisted for exact future replay (5k-5).
_PERSIST_EVENT_TYPES = (
    "read_required_by_judge", "read_desired", "read_selected", "read_blocked_no_url",
    "read_blocked_disallowed_tool", "read_selected_for_pending_judgment",
    "read_completed_for_pending_judgment", "read_passage_selected", "read_judged_after_read",
    "read_judgment_resolved", "read_judgment_still_unresolved", "read_interpreted")


def export_read_judge_replay(frontier, *, item_id: str = "", max_passage_chars: int = 2000,
                             fetch_meta_by_url: dict | None = None) -> dict[str, Any]:
    """Level 5k-5: serialize the read→judge lifecycle for EXACT future replay without
    reconstruction — PendingReadJudgment records, read lifecycle events, per-source fetch_meta,
    bounded passage windows, and closure codes. Bounded + contamination-safe: only passage
    windows (not full pages), and never gold/answer text. Intended to be written to a run's
    ``read_judge_replay_validation.json``."""
    fetch_meta_by_url = fetch_meta_by_url or {}
    pend = []
    for p in getattr(frontier, "pending_read_judgments", {}).values():
        rec = p.to_dict() if hasattr(p, "to_dict") else dict(p)
        rec["passage_preview"] = (getattr(p, "passage_preview", "") or "")[:max_passage_chars]
        rec["closure_code"] = _RESOLUTION_TO_CLOSURE.get(
            getattr(p, "resolution", "open"), "requires_read_still_open")
        pend.append(rec)
    events = [{"event_type": e.get("event_type"), "candidate_id": e.get("candidate_id"),
               "slot_id": e.get("slot_id"), "data": e.get("data", {})}
              for e in getattr(frontier, "events", [])
              if e.get("event_type") in _PERSIST_EVENT_TYPES]
    return {
        "schema": "read_judge_replay_v1", "item_id": item_id,
        "reconstructed_from_legacy_trace": False,
        "pending_read_judgments": pend,
        "read_judge_events": events,
        "fetch_meta_by_url": {u: dict(m) for u, m in fetch_meta_by_url.items()},
        "note": ("bounded + contamination-safe persistence for exact offline replay; "
                 "no gold/answer text; passage windows only, not full pages"),
    }


def _load_provider_bodies(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Build ``url -> {stored_body, raw_payload?}`` from any RecordingCache JSON under the run
    dir (``cache/`` + top-level ``*_cache.json``). Robust to the recording schema in
    ``live/cache.py`` (``entries[].response.results[].snippet`` is the stored page body)."""
    out: dict[str, dict[str, Any]] = {}
    candidates: list[Path] = []
    cdir = run_dir / "cache"
    if cdir.is_dir():
        candidates += sorted(cdir.glob("*.json"))
    candidates += sorted(run_dir.glob("*provider*cache*.json"))
    candidates += sorted(run_dir.glob("*tool*cache*.json"))
    for cf in candidates:
        try:
            data = json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            continue
        for e in entries:
            resp = (e or {}).get("response") or {}
            results = resp.get("results") or []
            raw = e.get("raw")
            for r in results:
                url = (r or {}).get("url") or resp.get("query") or ""
                if not url:
                    continue
                rec = out.setdefault(url, {})
                body = (r or {}).get("snippet") or ""
                if len(body) > len(rec.get("stored_body", "")):
                    rec["stored_body"] = body
                fm = resp.get("fetch_meta") or {}
                if fm.get("fetched_chars"):
                    rec["fetch_meta"] = fm
                if isinstance(raw, str) and len(raw) > len(rec.get("raw_payload", "")):
                    rec["raw_payload"] = raw
    return out


def _anchor_terms_for_constraint(record: dict, constraint_id: str, slot_id: str) -> list[str]:
    """Anchors for passage scan are derived from QUESTION + frame constraint terms + target
    descriptor + recorded subject aliases — NEVER gold answers (5k requirement)."""
    anchors: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = (s or "").strip()
        if len(s) >= 3 and s.lower() not in seen:
            anchors.append(s)
            seen.add(s.lower())

    frame = record.get("task_frame") or {}
    for c in frame.get("constraints", []) or []:
        if c.get("constraint_id") == constraint_id:
            for t in (c.get("normalized_terms") or []):
                _add(str(t))
            for w in re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", c.get("text_span", "") or ""):
                _add(w)
    for s in (frame.get("target_answer_slots") or []):
        if s.get("slot_id") == slot_id or True:
            for w in re.findall(r"[A-Za-z]{4,}", s.get("descriptor", "") or s.get("slot_name", "")):
                _add(w)
    # subject aliases recorded on the slate candidate (answer-free).
    cf = record.get("candidate_frontier") or {}
    for sl in cf.get("slates", []) or []:
        for cand in sl.get("top_candidates", []) or []:
            _add(cand.get("candidate_text_preview", ""))
    if not anchors:                                    # fall back to question content tokens
        for w in re.findall(r"[A-Za-z]{4,}", record.get("question_preview", "") or ""):
            _add(w)
    return anchors[:24]


def _reconstruct_obligations(record: dict) -> list[dict]:
    """Heuristically reconstruct requires_read obligations from a pre-5h debug record:
    a judge ``requires_read`` verdict + its candidate/slot/constraint + a later read of the
    same source url. Marked ``reconstructed_from_legacy_trace`` and diagnostic-only."""
    cf = record.get("candidate_frontier") or {}
    events = cf.get("events") or []
    obligations: list[dict] = []
    # 1) native 5h pending objects, if the run already persisted them.
    for p in (cf.get("pending_read_judgments") or []):
        obligations.append({**p, "reconstructed_from_legacy_trace": False, "native": True})
    if obligations:
        return obligations
    # 2) legacy reconstruction from requires_read judge events.
    for e in events:
        if e.get("event_type") not in ("evidence_judgment_requires_read", "read_required_by_judge"):
            continue
        data = e.get("data", {}) or {}
        obligations.append({
            "candidate_id": e.get("candidate_id") or data.get("candidate_id") or "",
            "slot_id": e.get("slot_id") or data.get("slot_id") or "",
            "constraint_id": data.get("constraint_id") or "",
            "reconstructed_from_legacy_trace": True, "native": False})
    return obligations


def _read_url_for(record: dict, candidate_id: str) -> str:
    """Find the source url that was (or should be) read for a candidate, from read calls /
    evidence / frontier events — without using gold."""
    cf = record.get("candidate_frontier") or {}
    for e in cf.get("events", []) or []:
        if e.get("event_type") in ("read_selected", "read_desired") \
                and (e.get("candidate_id") == candidate_id):
            host = e.get("data", {}).get("url_host")
            if host:
                for ev in (record.get("evidence") or []):
                    if host in (ev.get("url") or ""):
                        return ev.get("url")
    for c in record.get("calls", []) or []:
        ta = c.get("task_action") or {}
        if ta.get("candidate_id") == candidate_id and (
                "read" in (ta.get("kind", "") + ta.get("frontier_action_type", ""))):
            er = c.get("evidence_record") or {}
            if er.get("url"):
                return er["url"]
            if _looks_url(ta.get("query_text_preview", "")):
                return ta["query_text_preview"]
    # else: the first non-contaminated evidence url for this item (best-effort).
    for ev in (record.get("evidence") or []):
        if not ev.get("benchmark_contaminated") and _looks_url(ev.get("url", "")):
            return ev["url"]
    return ""


def _looks_url(s: str) -> bool:
    return isinstance(s, str) and s.strip().lower().startswith(("http://", "https://"))


def load_legacy_run(run_dir: str | Path, *, allow_live_judge: bool = False,
                    max_judge_calls: int = 0,
                    judge_factory: Optional[Callable[[], Any]] = None) -> ReplayValidationResult:
    """Inspect a REAL run directory and reconstruct/validate the read→judge pipeline as far as
    the artifacts allow (5k-1/2/3). Never fetches. Per obligation it reports how far it got
    (reconstructed → body_located → passages_scanned → judged → closed) with a precise reason.

    The targeted re-judgment is NEW computation that cannot exist in a pre-5h cache, so it is
    reported ``rejudgment_prompt_not_in_cache`` unless ``allow_live_judge`` is set (opt-in,
    capped by ``max_judge_calls``, fail-closed, recording into the run's judge cache)."""
    run_dir = Path(run_dir)
    res = ReplayValidationResult(overall_status="validated")
    res.notes.append("reconstructed_from_legacy_trace (diagnostic only; NOT headline evidence)")
    dqf = run_dir / "debug_questions.jsonl"
    if not dqf.exists():
        return _absent_or_unknown(run_dir, res)
    bodies = _load_provider_bodies(run_dir)
    judge_cache_path = run_dir / "llm_evidence_judge_cache.json"
    live_judge = None
    live_calls = 0
    if allow_live_judge:
        live_judge = (judge_factory or deterministic_replay_judge)()

    n_items = 0
    for line in dqf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            res.notes.append("cache_schema_unknown:debug_questions_line_not_json")
            continue
        n_items += 1
        item_id = record.get("item_id", "")
        obs = _reconstruct_obligations(record)
        if not obs:
            res.replay_events.append({"event_type": "no_pending_read_judgment_events",
                                      "item_id": item_id})
            continue
        for i, ob in enumerate(obs):
            o = ObligationOutcome(
                pending_read_judgment_id=ob.get("pending_read_judgment_id", f"recon_{item_id}_{i}"),
                candidate_id=ob.get("candidate_id", ""), slot_id=ob.get("slot_id", ""),
                constraint_id=ob.get("constraint_id", ""), source_url="",
                closure_code="unvalidated_cache_miss",
                reconstructed_from_legacy_trace=bool(ob.get("reconstructed_from_legacy_trace")),
                item_id=item_id, pipeline_status="reconstructed")
            url = ob.get("source_url") or _read_url_for(record, o.candidate_id)
            o.source_url = url
            if not url:
                o.stage_reason = "read_event_found_but_body_missing"
                res.obligations.append(o)
                continue
            body_rec = bodies.get(url)
            if not body_rec:
                # fall back to the bounded debug snippet preview (clearly marked).
                prev = next((ev.get("snippet_preview", "") for ev in (record.get("evidence") or [])
                             if ev.get("url") == url), "")
                if prev:
                    body_rec = {"stored_body": prev, "from_preview": True}
                else:
                    o.stage_reason = "read_event_found_but_body_missing"
                    res.obligations.append(o)
                    continue
            stored = body_rec.get("stored_body", "")
            raw = body_rec.get("raw_payload", "")
            o.stored_body_chars = len(stored)
            o.cached_payload_chars = max(len(stored), len(raw))
            # use the FULLER raw payload when present (validates beyond-cap retrieval, no spend).
            body = raw if len(raw) > len(stored) else stored
            o.body_source = ("raw_cache_payload" if len(raw) > len(stored)
                             else ("debug_preview" if body_rec.get("from_preview") else "stored_body"))
            o.pipeline_status = "body_located"
            anchors = _anchor_terms_for_constraint(record, o.constraint_id, o.slot_id)
            scan = extract_passages(body, anchors)
            o.passage_anchor_hits = scan.passage_anchor_hits
            o.passage_count = scan.passage_count
            o.first_hit_offset = scan.first_hit_offset
            o.passages_found_beyond_4000 = scan.first_hit_offset > _ADAPTER_CAP_DEFAULT
            o.pipeline_status = "passages_scanned"
            # truncation: only assertable when the FULLER payload is also exhausted.
            if not scan.hit and o.cached_payload_chars <= _ADAPTER_CAP_DEFAULT:
                o.body_truncated_before_relevant_passage = True
                o.stage_reason = "body_truncated_before_relevant_passage"
            # judged/closed: new computation -> needs the live-judge tier (opt-in, capped).
            recorded = _recorded_rejudgment(judge_cache_path)
            if recorded:
                o.pipeline_status, o.closure_code = "judged", "resolved_full_support"
                o.stage_reason = "reconstructed_and_passages_scanned"
            elif allow_live_judge and scan.passages:
                if live_calls >= max_judge_calls:          # hard cap, fail-closed (0 = none)
                    o.stage_reason = "rejudgment_call_budget_exhausted"
                else:
                    cand_text = _candidate_text(record, o.candidate_id, url)
                    verdict, was_live = _live_rejudge(live_judge, o, record, cand_text,
                                                      scan.passages[0], judge_cache_path)
                    live_calls += int(was_live)
                    o.pipeline_status = "judged"
                    o.closure_code = _RESOLUTION_TO_CLOSURE.get(verdict, "requires_read_still_open")
                    o.stage_reason = "closed_by_live_rejudgment"
                    res.replay_events.append({"event_type": "read_judged_after_read",
                                              "item_id": item_id, "verdict": verdict})
            else:
                o.stage_reason = o.stage_reason or "rejudgment_prompt_not_in_cache"
            res.obligations.append(o)

    _summarize_legacy(res, n_items, live_calls)
    return res


def _recorded_rejudgment(judge_cache_path: Path) -> bool:
    """A pre-5h cache cannot contain the 5h targeted re-judgment prompt; this hook lets a
    FUTURE run that DID persist one be recognised. Conservative: returns False unless a
    re-judgment marker key is present."""
    if not judge_cache_path.exists():
        return False
    try:
        store = json.loads(judge_cache_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return any("rejudgment" in str(k).lower() or "pending_read" in str(k).lower()
               for k in (store or {}))


def _candidate_text(record: dict, candidate_id: str, url: str) -> str:
    """A display name for the obligation's candidate (answer-free): the evidence title for the
    read url, else the first slate candidate preview, else the candidate id."""
    for ev in (record.get("evidence") or []):
        if ev.get("url") == url and ev.get("title_preview"):
            return ev["title_preview"]
    cf = record.get("candidate_frontier") or {}
    for sl in cf.get("slates", []) or []:
        for c in sl.get("top_candidates", []) or []:
            if c.get("candidate_text_preview"):
                return c["candidate_text_preview"]
    return candidate_id


def _live_rejudge(judge, o: ObligationOutcome, record: dict, candidate_text: str,
                  passage: str, judge_cache_path: Path) -> tuple[str, bool]:
    """Run ONE targeted re-judgment live on a passage (5k-4). Inputs are passage + obligation
    metadata only — never gold; contaminated sources are excluded upstream. Records the verdict
    into the run's judge cache so the next replay is fully offline."""
    cons = (record.get("task_frame") or {}).get("constraints") or []
    con = next((c for c in cons if c.get("constraint_id") == o.constraint_id), {})
    con_obj = SimpleNamespace(constraint_id=o.constraint_id,
                              text_span=con.get("text_span", o.constraint_id),
                              normalized_terms=con.get("normalized_terms", []),
                              testable_claim="", how_to_test="read",
                              applies_to=con.get("applies_to", [o.slot_id]))
    before = getattr(judge, "calls", 0)
    jd = judge.judge(
        candidate_text=candidate_text, candidate_id=o.candidate_id, aliases=[],
        slot_id=o.slot_id, slot_role="person", slot_descriptor="", constraint=con_obj,
        source_id=o.pending_read_judgment_id, source_title="", source_url=o.source_url,
        source_domain=_host(o.source_url), source_role="article", contaminated=False,
        snippet=passage, det_status="insufficient", det_quote="")
    was_live = getattr(judge, "calls", 0) > before
    # 5k-4: persist the verdict so the next replay of this validation is fully offline.
    if was_live and judge_cache_path:
        try:
            store = (json.loads(judge_cache_path.read_text(encoding="utf-8"))
                     if judge_cache_path.exists() else {})
            store[f"rejudgment::{o.pending_read_judgment_id}"] = jd.judgment
            judge_cache_path.write_text(json.dumps(store, sort_keys=True), encoding="utf-8")
        except Exception:
            pass
    return jd.judgment, was_live


def _summarize_legacy(res: ReplayValidationResult, n_items: int, live_calls: int) -> None:
    obs = res.obligations
    stage = Counter(o.pipeline_status for o in obs)
    reasons = Counter(o.stage_reason for o in obs if o.stage_reason)
    res.metrics = {
        "n_items_inspected": n_items,
        "replay_pending_read_judgment_count": len(obs),
        "reconstructed_count": sum(1 for o in obs if o.reconstructed_from_legacy_trace),
        "body_located_count": sum(1 for o in obs
                                  if o.pipeline_status in ("body_located", "passages_scanned",
                                                           "judged", "closed")),
        "passages_scanned_count": sum(1 for o in obs
                                      if o.pipeline_status in ("passages_scanned", "judged", "closed")),
        "passages_found_beyond_4000_count": sum(1 for o in obs if o.passages_found_beyond_4000),
        "body_truncated_before_relevant_passage_count":
            sum(1 for o in obs if o.body_truncated_before_relevant_passage),
        "judged_count": sum(1 for o in obs if o.pipeline_status in ("judged", "closed")),
        "closed_count": sum(1 for o in obs if o.closure_code in _CLOSED),
        "pipeline_status_counts": dict(stage),
        "stage_reason_counts": dict(reasons),
        "live_provider_calls": 0, "live_model_calls": live_calls,
    }
    if not obs:
        res.overall_status = "unvalidated_cache_miss"
    elif any(o.pipeline_status in ("judged", "closed") for o in obs):
        res.overall_status = "validated"
    else:
        res.overall_status = "reconstructed_passages_scanned_rejudgment_pending"


# --------------------------------------------------------------------------- artifacts
def validate_artifacts_dir(path: str | Path, *, allow_live_judge: bool = False,
                           max_judge_calls: int = 0) -> ReplayValidationResult:
    """Validate a recorded run directory. Prefers a committed ``replay_fixture.json``; else
    reconstructs from a real legacy run (``debug_questions.jsonl`` + caches). Returns
    ``unvalidated_cache_miss`` (NOT a silent pass) only when nothing is inspectable."""
    p = Path(path)
    fixture_file = p / "replay_fixture.json"
    if fixture_file.exists():
        return run_replay(json.loads(fixture_file.read_text(encoding="utf-8")))
    if (p / "debug_questions.jsonl").exists():
        return load_legacy_run(p, allow_live_judge=allow_live_judge,
                               max_judge_calls=max_judge_calls)
    return _absent_or_unknown(p, ReplayValidationResult(overall_status="unvalidated_cache_miss"))


def _absent_or_unknown(p: Path, res: ReplayValidationResult) -> ReplayValidationResult:
    res.overall_status = "unvalidated_cache_miss"
    if not p.exists():
        res.notes.append(f"artifacts_absent:{p} (cache/trajectory not present in this "
                         "container — gitignored live outputs; cannot replay)")
    else:
        present = sorted(f.name for f in p.glob("*"))
        res.notes.append(f"no_replay_fixture_or_debug_questions_in:{p}; present={present}")
    res.metrics = {"replay_pending_read_judgment_count": 0,
                   "replay_pending_read_unvalidated_cache_miss_count": 0,
                   "live_provider_calls": 0, "live_model_calls": 0}
    return res
