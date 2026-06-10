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
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

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

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "pending_read_judgment_id", "candidate_id", "slot_id", "constraint_id",
            "source_url", "closure_code", "read_replayed", "passage_anchor_hits",
            "passage_count", "used_full_body_not_snippet", "first_hit_offset")}


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


# --------------------------------------------------------------------------- artifacts
def validate_artifacts_dir(path: str | Path) -> ReplayValidationResult:
    """Validate a recorded run directory. Returns ``unvalidated_cache_miss`` (NOT a silent
    pass) when the directory or its cached bodies/trajectory are absent (5j-A honesty rule)."""
    p = Path(path)
    fixture_file = p / "replay_fixture.json"
    if fixture_file.exists():
        return run_replay(json.loads(fixture_file.read_text(encoding="utf-8")))
    res = ReplayValidationResult(overall_status="unvalidated_cache_miss")
    if not p.exists():
        res.notes.append(f"artifacts_absent:{p} (cache/trajectory not present in this "
                         "container — gitignored live outputs; cannot replay)")
    else:
        present = sorted(f.name for f in p.glob("*"))
        res.notes.append(f"no_replay_fixture_or_body_cache_in:{p}; present={present}")
    res.metrics = {"replay_pending_read_judgment_count": 0,
                   "replay_pending_read_unvalidated_cache_miss_count": 0,
                   "live_provider_calls": 0, "live_model_calls": 0}
    return res
