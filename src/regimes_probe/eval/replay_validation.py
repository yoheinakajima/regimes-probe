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
    body_source: str = ""                      # cache_stored_text|cache_raw_payload|debug_snippet_only|not_found
    stored_body_chars: int = 0
    cached_payload_chars: int = 0
    passages_found_beyond_4000: bool = False
    body_truncated_before_relevant_passage: bool = False
    # Level 5l structured-interpretations reconstruction + read/url matching provenance.
    candidate_text: str = ""
    source_role: str = ""
    source_domain: str = ""
    requires_read_reason: str = ""
    judgment_id: str = ""
    prompt_hash: str = ""
    reconstruction_method: str = ""            # structured_interpretations|legacy_events|native_5h
    reconstruction_missing_fields: list = field(default_factory=list)
    matched_read: bool = False
    read_tool: str = ""
    read_success: Optional[bool] = None
    read_chars: int = 0
    url_match_method: str = "none"             # exact|prefix|host_only|none
    store_raw_was_enabled: Optional[bool] = None
    raw_unavailable: bool = False
    # Level 5m strict body semantics: read-url vs body-url provenance, debug-snippet split,
    # and the source actually fed to a live re-judgment (never debug_snippet_only by default).
    matched_read_url: str = ""
    matched_body_url: str = ""
    debug_snippet_chars: int = 0
    live_rejudgment_source: str = ""
    # Level 5n provider-class + passage-relevance strictness.
    body_provider: str = ""
    body_is_actual_read_body: bool = False
    body_is_search_snippet: bool = False
    #: read_call | read_provider_cache | replay_export | search_provider_cache | debug_record | none
    body_match_source: str = "none"
    anchor_category_counts: dict = field(default_factory=dict)
    #: predicate_relevant | subject_only | no_relevant_anchor | ""
    passage_relevance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "pending_read_judgment_id", "candidate_id", "slot_id", "constraint_id",
            "source_url", "closure_code", "read_replayed", "passage_anchor_hits",
            "passage_count", "used_full_body_not_snippet", "first_hit_offset",
            "reconstructed_from_legacy_trace", "pipeline_status", "stage_reason", "item_id",
            "body_source", "stored_body_chars", "cached_payload_chars",
            "passages_found_beyond_4000", "body_truncated_before_relevant_passage",
            "candidate_text", "source_role", "source_domain", "requires_read_reason",
            "judgment_id", "prompt_hash", "reconstruction_method",
            "reconstruction_missing_fields", "matched_read", "read_tool", "read_success",
            "read_chars", "url_match_method", "store_raw_was_enabled", "raw_unavailable",
            "matched_read_url", "matched_body_url", "debug_snippet_chars",
            "live_rejudgment_source", "body_provider", "body_is_actual_read_body",
            "body_is_search_snippet", "body_match_source", "anchor_category_counts",
            "passage_relevance")}


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
#: a call-embedded "body" shorter than this is a bounded preview, not a real page body (5m-2).
_MIN_REAL_BODY_CHARS = 600

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
        # 5m-7: persist the FULL, untruncated source_url (to_dict keeps only the host) and the
        # actual passage window used for the targeted re-judgment (bounded, contamination-safe).
        rec["source_url"] = getattr(p, "source_url", "") or rec.get("source_url", "")
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


#: response fields a tool-specific schema adapter recognises as a page BODY (5m item 1/2).
_BODY_FIELDS = ("snippet", "markdown", "text", "content", "body", "html")
#: read-class tools whose cache entries carry page bodies.
#: legacy alias retained for reference; provider CLASSES are decided by ``_provider_class``.
_READ_TOOLS = ("firecrawl_scrape", "page_fetch")


def _extract_entry_bodies(entry: dict) -> list[tuple[str, str]]:
    """Tool-schema adapter: extract ``(url, body)`` pairs from one RecordingCache entry.
    Recognises the standard SearchResponse shape (``response.results[].snippet``) plus
    common body field aliases (markdown/text/content/body) at the result or response level."""
    resp = (entry or {}).get("response") or {}
    pairs: list[tuple[str, str]] = []
    results = resp.get("results") or []
    for r in results:
        if not isinstance(r, dict):
            continue
        url = r.get("url") or resp.get("query") or ""
        body = ""
        for f in _BODY_FIELDS:
            v = r.get(f)
            if isinstance(v, str) and len(v) > len(body):
                body = v
        if url and body:
            pairs.append((url, body))
    # response-level body fields (some scrape schemas put the document on the response).
    url0 = resp.get("query") or ""
    if url0 and _looks_url(url0):
        for f in _BODY_FIELDS:
            v = resp.get(f)
            if isinstance(v, str) and v:
                pairs.append((url0, v))
    return pairs


def _discover_cache_files(run_dir: Path) -> list[Path]:
    """All candidate cache files: the run_manifest's recorded ``cache.path``, EVERYTHING under
    ``cache/`` (recursive, any extension — never silently ignored), and top-level
    ``*cache*.json`` files (5m item 1)."""
    files: list[Path] = []
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        try:
            mp = ((json.loads(manifest.read_text(encoding="utf-8")) or {})
                  .get("cache") or {}).get("path")
            if mp:
                for cand in (Path(mp), run_dir / Path(mp).name, run_dir / mp):
                    if cand.exists() and cand.is_file():
                        files.append(cand)
                        break
        except Exception:
            pass
    cdir = run_dir / "cache"
    if cdir.is_dir():
        files += sorted(p for p in cdir.rglob("*") if p.is_file())
    files += sorted(run_dir.glob("*cache*.json"))
    seen: set[Path] = set()
    out = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(f)
    return out


def _load_replay_export_bodies(run_dir: Path) -> dict[str, str]:
    """Persisted passage windows from a ``read_judge_replay_v1`` export (5n-2: an actual-body
    source for FUTURE runs that persisted them; a pre-5h run has none)."""
    out: dict[str, str] = {}
    f = run_dir / "read_judge_replay_validation.json"
    if not f.exists():
        return out
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(data, dict) or data.get("schema") != "read_judge_replay_v1":
        return out
    for p in (data.get("pending_read_judgments") or []):
        url = p.get("source_url") or ""
        body = p.get("passage_preview") or ""
        if url and len(body) > len(out.get(url, "")):
            out[url] = body
    return out


def _provider_class(provider: str) -> str:
    """Classify a cache provider (5n-1): only READ-BODY providers can yield actual page
    bodies; search providers yield snippets; model caches are never body sources. Generic
    name-based typing so future fetch/scrape tools classify correctly."""
    p = (provider or "").lower()
    if p.startswith("openai_responses") or "answerer" in p:
        return "model"
    if p in ("firecrawl_scrape", "page_fetch") or "scrape" in p or "fetch" in p:
        return "read_body"
    if "search" in p:
        return "search_snippet"
    return "unknown"


def _load_provider_bodies(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build ``url -> body record`` from the run's recording cache(s), CLASSIFIED by provider
    class (5n-1): read-body provider payloads are kept under ``read_body``/``raw_payload``;
    search-provider snippets under ``search_snippet`` (diagnostic only — never an actual
    body). Plus a CACHE REPORT split by provider class, shared with ``inspect_run_schema``.
    Never silently ignores a cache file."""
    out: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"files": [], "n_files": 0, "n_entries": 0,
                              "entries_by_provider": {}, "bodies_by_provider": {},
                              "read_body_entries_by_provider": {},
                              "search_snippet_entries_by_provider": {},
                              "model_entries_by_provider": {},
                              "raw_payload_entries": 0, "store_raw_headers": [],
                              "unrecognized_schema_files": []}
    by_prov: Counter = Counter()
    bodies_prov: Counter = Counter()
    read_prov: Counter = Counter()
    search_prov: Counter = Counter()
    model_prov: Counter = Counter()
    for cf in _discover_cache_files(run_dir):
        finfo = {"file": str(cf.relative_to(run_dir)) if str(cf).startswith(str(run_dir))
                 else str(cf), "entries": 0, "any_raw": False, "store_raw_header": None}
        try:
            data = json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            finfo["error"] = "not_json"
            report["unrecognized_schema_files"].append(finfo["file"])
            report["files"].append(finfo)
            continue
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            # a flat ParserCache (LLM caches) is recognised but carries no page bodies.
            finfo["schema"] = ("flat_kv_llm_cache" if isinstance(data, dict)
                               else "unknown")
            if finfo["schema"] == "unknown":
                report["unrecognized_schema_files"].append(finfo["file"])
            report["files"].append(finfo)
            continue
        store_raw = bool(data.get("store_raw"))     # cache file header (live/cache.py)
        finfo["schema"] = "recording_cache"
        finfo["store_raw_header"] = store_raw
        finfo["entries"] = len(entries)
        report["n_entries"] += len(entries)
        report["store_raw_headers"].append({finfo["file"]: store_raw})
        for e in entries:
            prov = (e or {}).get("provider") or (e or {}).get("name") or "?"
            pclass = _provider_class(prov)
            by_prov[prov] += 1
            (read_prov if pclass == "read_body"
             else search_prov if pclass == "search_snippet"
             else model_prov)[prov] += 1
            raw = (e or {}).get("raw")
            rawtext = raw if isinstance(raw, str) else (json.dumps(raw) if raw else "")
            if rawtext:
                finfo["any_raw"] = True
                report["raw_payload_entries"] += 1
            if pclass == "model":
                continue                            # model caches are never body sources
            pairs = _extract_entry_bodies(e)
            if pairs:
                bodies_prov[prov] += 1
            for url, body in pairs:
                rec = out.setdefault(url, {})
                if pclass == "read_body":
                    rec["store_raw"] = store_raw or rec.get("store_raw", False)
                    rec["read_provider"] = prov
                    if len(body) > len(rec.get("read_body", "")):
                        rec["read_body"] = body
                    fm = ((e or {}).get("response") or {}).get("fetch_meta") or {}
                    if fm.get("fetched_chars"):
                        rec["fetch_meta"] = fm
                    if rawtext and len(rawtext) > len(rec.get("raw_payload", "")):
                        rec["raw_payload"] = rawtext
                else:                               # search_snippet / unknown -> snippet only
                    rec["search_provider"] = prov
                    if len(body) > len(rec.get("search_snippet", "")):
                        rec["search_snippet"] = body
        report["files"].append(finfo)
    report["n_files"] = len(report["files"])
    report["entries_by_provider"] = dict(by_prov)
    report["bodies_by_provider"] = dict(bodies_prov)
    report["read_body_entries_by_provider"] = dict(read_prov)
    report["search_snippet_entries_by_provider"] = dict(search_prov)
    report["model_entries_by_provider"] = dict(model_prov)
    report["urls_with_actual_read_bodies"] = sum(1 for r in out.values() if r.get("read_body"))
    report["urls_with_search_snippets_only"] = sum(
        1 for r in out.values() if r.get("search_snippet") and not r.get("read_body"))
    report["unrecognized_schema_count"] = len(report["unrecognized_schema_files"])
    return out, report


def _anchor_terms_for_constraint(record: dict, constraint_id: str, slot_id: str,
                                 *, candidate_text: str = "") -> list[str]:
    """Anchors for passage scan are derived from QUESTION + frame constraint terms + target
    descriptor + assertion candidate_text/aliases — NEVER gold answers (5k/5l requirement)."""
    anchors: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = (s or "").strip()
        if len(s) >= 3 and s.lower() not in seen:
            anchors.append(s)
            seen.add(s.lower())

    # the assertion's candidate text + its content tokens (answer-free; from the trace).
    _add(candidate_text)
    for w in re.findall(r"[A-Za-z]{4,}", candidate_text or ""):
        _add(w)
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


#: anchor categories that establish PREDICATE relevance (5n-4). Subject categories
#: (candidate_alias / source_title) alone are NOT sufficient to judge a passage.
_PREDICATE_CATEGORIES = ("constraint_label", "constraint_facet", "target_descriptor",
                         "relation_predicate", "numeric_or_year", "quoted_phrase")
_SUBJECT_CATEGORIES = ("candidate_alias", "source_title")
#: SPECIFIC relation/predicate verbs only (generic glue like "from"/"is"/"year" would make
#: every page predicate-relevant, defeating strictness). Generic verbs, not domain phrases.
_PREDICATE_VERBS = frozenset({
    "born", "birth", "founded", "opened", "established", "worked", "served", "studied",
    "graduated", "died", "death", "located", "designed", "wrote", "directed", "published",
    "appointed", "elected", "moved", "joined", "retired", "married"})
_GENERIC_ANCHOR_WORDS = frozenset({
    "the", "this", "that", "with", "from", "have", "been", "were", "their", "about",
    "which", "what", "when", "where", "year", "years", "name", "also"})


def _categorized_anchors(record: dict, constraint_id: str, slot_id: str,
                         *, candidate_text: str = "",
                         source_title: str = "") -> dict[str, list[str]]:
    """Anchors per CATEGORY (5n-4), derived ONLY from question text, task-frame constraints,
    target descriptors, and trace candidate aliases/titles — never gold answers."""
    cats: dict[str, list[str]] = {c: [] for c in _SUBJECT_CATEGORIES + _PREDICATE_CATEGORIES}

    def _add(cat: str, s: str) -> None:
        s = (s or "").strip()
        if len(s) >= 3 and s.lower() not in _GENERIC_ANCHOR_WORDS \
                and s.lower() not in (x.lower() for x in cats[cat]):
            cats[cat].append(s)

    _add("candidate_alias", candidate_text)
    for w in re.findall(r"[A-Za-z]{4,}", candidate_text or ""):
        _add("candidate_alias", w)
    _add("source_title", source_title)
    frame = record.get("task_frame") or {}
    question = record.get("question_preview", "") or ""
    cand_l = (candidate_text or "").lower()
    for c in frame.get("constraints", []) or []:
        if c.get("constraint_id") != constraint_id:
            continue
        for t in (c.get("normalized_terms") or []):
            # a constraint term that merely names the SUBJECT is not a predicate anchor.
            if str(t).lower() not in cand_l:
                _add("constraint_label", str(t))
        for w in re.findall(r"[A-Za-z]{4,}", c.get("text_span", "") or ""):
            if w.lower() not in cand_l:
                _add("constraint_facet", w)
        for y in re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", c.get("text_span", "") or ""):
            _add("numeric_or_year", y)
        for t in (c.get("normalized_terms") or []):
            if re.fullmatch(r"(1[5-9]\d{2}|20\d{2})", str(t)):
                _add("numeric_or_year", str(t))
    for s in (frame.get("target_answer_slots") or []):
        for w in re.findall(r"[A-Za-z]{4,}", s.get("descriptor", "") or s.get("slot_name", "")):
            if w.lower() not in cand_l:
                _add("target_descriptor", w)
    for v in sorted(_PREDICATE_VERBS):
        _add("relation_predicate", v)
    for q in re.findall(r'"([^"]{6,80})"', question):
        _add("quoted_phrase", q)
    return cats


def _scan_with_relevance(body: str, cats: dict[str, list[str]],
                         *, config=None) -> tuple[Any, dict[str, int], str]:
    """Scan a body with categorized anchors (5n-4). Returns ``(scan, category_counts,
    passage_relevance)`` where relevance is ``predicate_relevant`` (≥1 non-subject category
    hit) / ``subject_only`` / ``no_relevant_anchor``. Deterministic, zero model calls."""
    from regimes_probe.agent.read_judgment import DEFAULT_READ_CONFIG
    all_anchors = [a for terms in cats.values() for a in terms]
    scan = extract_passages(body, all_anchors[:64], config=config or DEFAULT_READ_CONFIG)
    matched = {m.lower() for m in scan.matched_anchors}
    counts = {cat: sum(1 for t in terms if t.lower() in matched)
              for cat, terms in cats.items()}
    predicate_hits = sum(counts.get(c, 0) for c in _PREDICATE_CATEGORIES)
    subject_hits = sum(counts.get(c, 0) for c in _SUBJECT_CATEGORIES)
    relevance = ("predicate_relevant" if predicate_hits > 0
                 else ("subject_only" if subject_hits > 0 else "no_relevant_anchor"))
    return scan, counts, relevance


def _constraint_slots(record: dict, constraint_id: str) -> list[str]:
    """Slots a constraint applies to, from the persisted task frame (best-effort)."""
    for c in ((record.get("task_frame") or {}).get("constraints") or []):
        if c.get("constraint_id") == constraint_id:
            return [str(s) for s in (c.get("applies_to") or [])]
    return []


def _reconstruct_obligations(record: dict) -> tuple[list[dict], dict]:
    """Reconstruct ``requires_read`` obligations from a debug record (5l fix). Returns
    ``(obligations, coverage)``. Three paths, in order of fidelity:

    1. native 5h ``pending_read_judgments`` (if the run already persisted them);
    2. **structured interpretations** — ``candidate_frontier.interpretations[].
       candidate_assertions[].judgments[]`` with ``judgment == "requires_read"`` (the path a
       pre-5h 5g run actually records; the 5h-era *events* do not exist there);
    3. legacy 5h *events* (``evidence_judgment_requires_read``).

    Obligations are never silently dropped for missing slot/candidate/constraint ids — they are
    kept with ``reconstruction_missing_fields`` and a ``legacy_missing_candidate_slot_or_constraint``
    reason. ``coverage`` reports persistence-cap truncation so a low count is attributable."""
    cf = record.get("candidate_frontier") or {}
    interps = cf.get("interpretations") or []
    events = cf.get("events") or []
    coverage = {
        "events_truncated": int(cf.get("events_truncated", 0)),
        "n_interpretations": len(interps),
        # CandidateFrontier.to_debug caps interpretations at [:20]; an exact 20 is suspect.
        "interpretations_possibly_bounded": len(interps) >= 20,
        "reconstruction_coverage_bounded": bool(cf.get("events_truncated")) or len(interps) >= 20,
    }
    obligations: list[dict] = []
    # 1) native 5h pending objects.
    for p in (cf.get("pending_read_judgments") or []):
        obligations.append({**p, "reconstructed_from_legacy_trace": False, "native": True,
                            "reconstruction_method": "native_5h"})
    if obligations:
        return obligations, coverage

    # 2) STRUCTURED INTERPRETATIONS (the real 5g path).
    seen: set[tuple] = set()
    for interp in interps:
        src_url = interp.get("source_url", "") or ""
        src_role = interp.get("source_role", "") or ""
        src_dom = interp.get("source_domain", "") or ""
        for a in (interp.get("candidate_assertions") or []):
            canon = a.get("canonical_candidate_ids") or {}     # slot_id -> candidate_id
            proposed = a.get("proposed_slot_ids") or []
            cand_text = a.get("candidate_text", "") or ""
            judged_cons: set[str] = set()
            for j in (a.get("judgments") or []):
                if j.get("judgment") != "requires_read":
                    continue
                cid = j.get("constraint_id", "") or ""
                judged_cons.add(cid)
                slot = j.get("slot_id") or _best_slot(cid, canon, proposed, record)
                cand = canon.get(slot) or (next(iter(canon.values()), "") if canon else "")
                missing = []
                if not cand:
                    missing.append("candidate_id")
                if not slot:
                    missing.append("slot_id")
                if not cid:
                    missing.append("constraint_id")
                key = (cand, slot, cid, src_url)
                if key in seen:
                    continue
                seen.add(key)
                obligations.append({
                    "candidate_id": cand, "slot_id": slot, "constraint_id": cid,
                    "candidate_text": cand_text, "source_url": src_url,
                    "source_role": src_role, "source_domain": src_dom,
                    "requires_read_reason": j.get("requires_read_reason") or "",
                    "judgment_id": j.get("judgment_id") or "", "prompt_hash": j.get("prompt_hash") or "",
                    "reconstructed_from_legacy_trace": True, "native": False,
                    "reconstruction_method": "structured_interpretations",
                    "reconstruction_missing_fields": missing})
            # cross-check: requires_read_constraint_ids without a judgment detail (5l item 1).
            for cid in (a.get("requires_read_constraint_ids") or []):
                if cid in judged_cons:
                    continue
                slot = _best_slot(cid, canon, proposed, record)
                cand = canon.get(slot) or (next(iter(canon.values()), "") if canon else "")
                key = (cand, slot, cid, src_url)
                if key in seen:
                    continue
                seen.add(key)
                obligations.append({
                    "candidate_id": cand, "slot_id": slot, "constraint_id": cid,
                    "candidate_text": cand_text, "source_url": src_url,
                    "source_role": src_role, "source_domain": src_dom,
                    "reconstructed_from_legacy_trace": True, "native": False,
                    "reconstruction_method": "structured_interpretations",
                    "reconstruction_missing_fields": ["judgment_detail"]})
    if obligations:
        return obligations, coverage

    # 3) legacy 5h events (forward-compatible).
    for e in events:
        if e.get("event_type") not in ("evidence_judgment_requires_read", "read_required_by_judge"):
            continue
        data = e.get("data", {}) or {}
        obligations.append({
            "candidate_id": e.get("candidate_id") or data.get("candidate_id") or "",
            "slot_id": e.get("slot_id") or data.get("slot_id") or "",
            "constraint_id": data.get("constraint_id") or "",
            "reconstructed_from_legacy_trace": True, "native": False,
            "reconstruction_method": "legacy_events"})
    return obligations, coverage


def _best_slot(constraint_id: str, canon: dict, proposed: list, record: dict) -> str:
    """Best-match slot for a constraint: a slot the constraint applies to that the assertion
    actually proposed/canonicalized; else the assertion's first proposed/canonical slot."""
    applies = set(_constraint_slots(record, constraint_id))
    for s in (list(canon.keys()) + list(proposed)):
        if s in applies:
            return s
    return (next(iter(canon.keys()), "") or (proposed[0] if proposed else ""))


def _norm_url(u: str) -> tuple[str, str]:
    """Normalize a (possibly truncated) URL to ``(host, path)`` for matching: lowercase, strip
    scheme + trailing slash. Persisted URLs are truncated (source_url[:160] / url[:300]) so
    matching must treat the stored value as a PREFIX of the full cache URL (5l item 3)."""
    s = (u or "").strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("#", 1)[0].rstrip("/")
    host, _, path = s.partition("/")
    return host, path


def _match_url(stored: str, candidates: list[str]) -> tuple[str, str]:
    """Match a stored (truncated) url against full cache urls. Returns ``(matched_url, method)``
    where method is exact|prefix|host_only|none. Host must match exactly; path by longest
    prefix (the stored path is a prefix of the full path when truncated)."""
    sh, sp = _norm_url(stored)
    if not sh:
        return "", "none"
    best, best_method, best_len = "", "none", -1
    for c in candidates:
        ch, cp = _norm_url(c)
        if ch != sh:
            continue
        if cp == sp:
            return c, "exact"
        if sp and cp.startswith(sp):
            if len(sp) > best_len:
                best, best_method, best_len = c, "prefix", len(sp)
        elif not sp:
            if best_method == "none":
                best, best_method = c, "host_only"
    if best:
        return best, best_method
    # host-only fallback: any cache url on the same host.
    for c in candidates:
        if _norm_url(c)[0] == sh:
            return c, "host_only"
    return "", "none"


def _call_embedded_body(call: dict) -> str:
    """Any page body persisted ON the read call itself (evidence_record body fields). Debug
    records usually keep only bounded previews, but a future run may embed the body."""
    er = call.get("evidence_record") or {}
    best = ""
    for f in ("body", "text", "markdown", "content", "snippet"):
        v = er.get(f)
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best


def _match_read_call(record: dict, candidate_id: str, source_url: str) -> dict:
    """Find the read CALL for an obligation (5l item 2 / 5m item 3). INVARIANT:
    ``matched_read=True`` requires a URL relation (exact|prefix|host_only) between the
    obligation's source_url and the call's read url — a candidate-id coincidence with no URL
    relation is NOT a match (``matched_read_with_no_url_match_method_count`` pinned 0)."""
    no_match = {"matched_read": False, "read_tool": "", "read_success": None, "read_chars": 0,
                "read_url": "", "read_url_match_method": "none", "call_body": ""}
    best = None
    for c in record.get("calls", []) or []:
        tool = c.get("tool", "") or ""
        ta = c.get("task_action") or {}
        # READ-class calls only (5n-1): firecrawl_search/exa/serper are SEARCH tools and a
        # match against them must never set matched_read.
        is_read = (_provider_class(tool) == "read_body"
                   or "read" in (ta.get("kind", "") + ta.get("frontier_action_type", "")))
        if not is_read:
            continue
        er = c.get("evidence_record") or {}
        url = er.get("url") or (ta.get("query_text_preview", "")
                                if _looks_url(ta.get("query_text_preview", "")) else "")
        if not url:
            continue
        # both urls may be truncated — try matching in BOTH directions, keep the best method.
        _, m1 = _match_url(source_url, [url])
        _, m2 = _match_url(url, [source_url])
        method = min((m for m in (m1, m2) if m != "none"),
                     key=lambda m: ("exact", "prefix", "host_only").index(m), default="none")
        if method == "none":
            continue                              # URL relation REQUIRED for matched_read
        scrape = c.get("scrape") or {}
        cand = {"matched_read": True, "read_tool": tool,
                "read_success": (not c.get("failed", False)) if "failed" in c else None,
                "read_chars": int(scrape.get("scrape_chars", 0) or 0),
                "read_url": url, "read_url_match_method": method,
                "call_body": _call_embedded_body(c)}
        rank = ("exact", "prefix", "host_only").index(method) \
            - (1 if ta.get("candidate_id") == candidate_id else 0)
        if best is None or rank < best[0]:
            best = (rank, cand)
    return best[1] if best else no_match


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
    bodies, cache_report = _load_provider_bodies(run_dir)
    export_bodies = _load_replay_export_bodies(run_dir)
    judge_cache_path = run_dir / "llm_evidence_judge_cache.json"
    live_judge = None
    live_calls = 0
    rejudge_version_mismatches = 0
    if allow_live_judge:
        live_judge = (judge_factory or deterministic_replay_judge)()

    n_items = 0
    coverage_bounded = False
    cache_urls = list(bodies.keys())
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
        obs, coverage = _reconstruct_obligations(record)
        if coverage.get("reconstruction_coverage_bounded"):
            coverage_bounded = True
            res.replay_events.append({"event_type": "reconstruction_coverage_bounded",
                                      "item_id": item_id, **coverage})
        if not obs:
            res.replay_events.append({"event_type": "no_pending_read_judgment_events",
                                      "item_id": item_id, **coverage})
            continue
        for i, ob in enumerate(obs):
            o = ObligationOutcome(
                pending_read_judgment_id=ob.get("pending_read_judgment_id", f"recon_{item_id}_{i}"),
                candidate_id=ob.get("candidate_id", ""), slot_id=ob.get("slot_id", ""),
                constraint_id=ob.get("constraint_id", ""), source_url=ob.get("source_url", ""),
                closure_code="unvalidated_cache_miss",
                reconstructed_from_legacy_trace=bool(ob.get("reconstructed_from_legacy_trace")),
                item_id=item_id, pipeline_status="reconstructed",
                candidate_text=ob.get("candidate_text", ""),
                source_role=ob.get("source_role", ""), source_domain=ob.get("source_domain", ""),
                requires_read_reason=ob.get("requires_read_reason", ""),
                judgment_id=ob.get("judgment_id", ""), prompt_hash=ob.get("prompt_hash", ""),
                reconstruction_method=ob.get("reconstruction_method", ""),
                reconstruction_missing_fields=list(ob.get("reconstruction_missing_fields", [])))
            # never DROP an obligation for missing ids — keep it with a precise reason.
            if not o.constraint_id or (not o.candidate_id and not o.candidate_text):
                o.stage_reason = "legacy_missing_candidate_slot_or_constraint"
                res.obligations.append(o)
                continue
            # 5l-2 / 5m-3: match the read CALL. matched_read REQUIRES a URL relation
            # (exact|prefix|host_only) — never a candidate-id coincidence alone.
            rc = _match_read_call(record, o.candidate_id, o.source_url)
            o.matched_read, o.read_tool = rc["matched_read"], rc["read_tool"]
            o.read_success, o.read_chars = rc["read_success"], rc["read_chars"]
            o.matched_read_url = rc.get("read_url", "")
            if o.matched_read:
                o.url_match_method = rc["read_url_match_method"]
                o.pipeline_status = "read_matched"
            stored_url = o.source_url or rc.get("read_url", "") or _read_url_for(record, o.candidate_id)
            o.source_url = stored_url
            if not stored_url:
                o.stage_reason = "read_event_found_but_body_missing"
                res.obligations.append(o)
                continue
            # 5n-2 body locator priority: (A) call-embedded READ body, (A2) replay-export
            # persisted body, (B) READ-provider cache, (C) SEARCH-provider snippet
            # (DIAGNOSTIC ONLY), (D) debug snippet (DIAGNOSTIC ONLY). Only A/A2/B are
            # actual read bodies; only those can be scanned/judged.
            body, raw = "", ""
            body_rec: dict = {}
            if rc.get("call_body") and len(rc["call_body"]) >= _MIN_REAL_BODY_CHARS:
                body = rc["call_body"]
                o.body_source = "call_embedded_read_body"
                o.body_match_source = "read_call"
                o.body_provider = rc["read_tool"]
                o.body_is_actual_read_body = True
                o.matched_body_url = o.matched_read_url
            else:
                exp_url, exp_method = _match_url(stored_url, list(export_bodies.keys()))
                if exp_url and len(export_bodies.get(exp_url, "")) >= 80:
                    body = export_bodies[exp_url]
                    o.body_source = "replay_export_body"
                    o.body_match_source = "replay_export"
                    o.body_provider = "read_judge_replay_v1"
                    o.body_is_actual_read_body = True
                    o.matched_body_url = exp_url
                    if not o.matched_read:
                        o.url_match_method = exp_method
                else:
                    matched_url, method = _match_url(stored_url, cache_urls)
                    if not o.matched_read:
                        o.url_match_method = method
                    if matched_url:
                        o.matched_body_url = matched_url
                        body_rec = bodies.get(matched_url) or {}
                        if body_rec.get("read_body"):
                            stored = body_rec["read_body"]
                            raw = body_rec.get("raw_payload", "")
                            body = raw if len(raw) > len(stored) else stored
                            o.body_source = "cache_read_body"
                            o.body_match_source = "read_provider_cache"
                            o.body_provider = body_rec.get("read_provider", "")
                            o.body_is_actual_read_body = True
                            # a read-provider cache entry with a URL match IS a matched read.
                            if not o.matched_read:
                                o.matched_read = True
                                o.read_tool = o.read_tool or o.body_provider
                                o.pipeline_status = "read_matched"
                        elif body_rec.get("search_snippet"):
                            # (C) SEARCH SNIPPET: diagnostic only — never an actual body,
                            # never advances, never judged (5n-1/2).
                            o.body_source = "cache_search_snippet_only"
                            o.body_match_source = "search_provider_cache"
                            o.body_provider = body_rec.get("search_provider", "")
                            o.body_is_search_snippet = True
                            o.debug_snippet_chars = len(body_rec["search_snippet"])
                            o.pipeline_status = ("read_matched_search_snippet_only"
                                                 if o.matched_read else "reconstructed")
                            o.stage_reason = ("read_body_not_persisted_legacy_run"
                                              if o.matched_read
                                              else "search_snippet_only_no_read_body")
                            res.obligations.append(o)
                            continue
            o.stored_body_chars = (len(body_rec.get("read_body", ""))
                                   if body_rec else len(body))
            o.cached_payload_chars = len(raw)
            o.store_raw_was_enabled = body_rec.get("store_raw") if body_rec else None
            o.raw_unavailable = not bool(raw)
            fm = body_rec.get("fetch_meta") or {}
            if not body:
                # (D) debug snippet: last-resort DIAGNOSTIC preview — does NOT count as a
                # located body, does NOT advance to passages_scanned, NOT live-judged.
                prev = ""
                for ev in (record.get("evidence") or []):
                    if _match_url(ev.get("url", ""), [stored_url])[1] != "none" \
                            or _norm_url(ev.get("url", ""))[0] == _norm_url(stored_url)[0]:
                        prev = ev.get("snippet_preview", "") or prev
                if prev:
                    o.body_source = "debug_snippet_only"
                    o.body_match_source = "debug_record"
                    o.debug_snippet_chars = len(prev)
                    scan = extract_passages(prev, _anchor_terms_for_constraint(
                        record, o.constraint_id, o.slot_id, candidate_text=o.candidate_text))
                    o.passage_anchor_hits = scan.passage_anchor_hits
                    o.pipeline_status = "debug_snippet_scanned"
                    o.stage_reason = ("read_body_not_persisted_legacy_run" if o.matched_read
                                      else "debug_snippet_only_no_body")
                else:
                    o.body_source = "not_found"
                    o.stage_reason = ("source_url_mismatch" if o.url_match_method == "none"
                                      and not o.matched_read
                                      else "read_event_found_but_body_missing")
                res.obligations.append(o)
                continue
            # an ACTUAL read body (call-embedded / replay-export / read-provider cache).
            o.pipeline_status = "actual_body_located"
            cats = _categorized_anchors(record, o.constraint_id, o.slot_id,
                                        candidate_text=o.candidate_text,
                                        source_title=_candidate_text(record, o.candidate_id,
                                                                     o.source_url))
            scan, cat_counts, relevance = _scan_with_relevance(body, cats)
            o.anchor_category_counts = cat_counts
            o.passage_relevance = relevance
            o.passage_anchor_hits = scan.passage_anchor_hits
            o.passage_count = scan.passage_count
            o.first_hit_offset = scan.first_hit_offset
            o.passages_found_beyond_4000 = scan.first_hit_offset > _ADAPTER_CAP_DEFAULT
            o.used_full_body_not_snippet = len(raw) > 0
            near_cap = (abs(o.stored_body_chars - _ADAPTER_CAP_DEFAULT) <= 16
                        or bool(fm.get("body_truncated_for_storage")))
            if relevance != "predicate_relevant":
                # 5n-4: subject/title hits alone are NOT judgeable passages. If the actual
                # body is capped with no raw, the predicate may lie beyond the cap.
                if o.raw_unavailable and near_cap:
                    o.body_truncated_before_relevant_passage = True
                    o.stage_reason = "body_truncated_before_relevant_passage(raw_unavailable)"
                elif relevance == "subject_only":
                    o.stage_reason = "subject_only_passage_no_predicate_anchor"
                else:
                    o.stage_reason = "no_relevant_anchor_in_actual_body"
                res.obligations.append(o)
                continue
            o.pipeline_status = "passages_scanned"
            # judged/closed: STRICT versioned rejudgment cache, else the live tier (opt-in).
            recorded, mismatch = _recorded_rejudgment(judge_cache_path, o, body)
            if mismatch:
                rejudge_version_mismatches += 1
                res.replay_events.append({"event_type":
                                          "recorded_rejudgment_cache_version_mismatch",
                                          "item_id": item_id,
                                          "pending_read_judgment_id": o.pending_read_judgment_id})
            if recorded:
                closure = _RESOLUTION_TO_CLOSURE.get(recorded, "requires_read_still_open")
                o.closure_code = closure
                o.live_rejudgment_source = "recorded_rejudgment_cache"
                o.pipeline_status = "closed" if closure in _CLOSED else "judged_unclosed"
                o.stage_reason = "closed_by_recorded_strict_rejudgment"
            elif allow_live_judge and scan.passages:
                if live_calls >= max_judge_calls:          # hard cap, fail-closed (0 = none)
                    o.pipeline_status = "rejudgment_pending"
                    o.stage_reason = "rejudgment_call_budget_exhausted"
                else:
                    cand_text = o.candidate_text or _candidate_text(
                        record, o.candidate_id, o.source_url)
                    verdict, was_live = _live_rejudge(live_judge, o, record, cand_text,
                                                      scan.passages[0], judge_cache_path,
                                                      body_for_hash=body)
                    live_calls += int(was_live)
                    closure = _RESOLUTION_TO_CLOSURE.get(verdict, "requires_read_still_open")
                    o.closure_code = closure
                    o.live_rejudgment_source = o.body_source
                    o.pipeline_status = "closed" if closure in _CLOSED else "judged_unclosed"
                    o.stage_reason = "closed_by_live_rejudgment"
                    res.replay_events.append({"event_type": "read_judged_after_read",
                                              "item_id": item_id, "verdict": verdict,
                                              "rejudgment_source": o.body_source})
            else:
                o.pipeline_status = "rejudgment_pending"
                o.stage_reason = o.stage_reason or "rejudgment_prompt_not_in_cache"
            res.obligations.append(o)

    _summarize_legacy(res, n_items, live_calls, coverage_bounded=coverage_bounded,
                      allow_live_judge=allow_live_judge, cache_report=cache_report,
                      rejudge_version_mismatches=rejudge_version_mismatches)
    return res


#: strict rejudgment cache namespace (5n-5). Entries from older validator versions (the
#: pre-strict ``rejudgment::*`` keys, or strict keys with mismatched provenance/hash) are
#: IGNORED and reported as version mismatches — they may have been judged on snippets.
STRICT_REJUDGE_VERSION = "read_judge_replay_strict_body_v2"


def _body_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _recorded_rejudgment(judge_cache_path: Path, o: "ObligationOutcome",
                         body: str = "") -> tuple[Optional[str], bool]:
    """Per-obligation strict-cache lookup (5n-5). Returns ``(verdict, version_mismatch)``:
    a verdict only when a STRICT entry exists with matching version + body provenance/hash;
    ``version_mismatch=True`` when a legacy (pre-strict) or provenance-mismatched entry was
    found and ignored."""
    if not judge_cache_path.exists():
        return None, False
    try:
        store = json.loads(judge_cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None, False
    oid = o.pending_read_judgment_id
    strict = (store or {}).get(f"{STRICT_REJUDGE_VERSION}::{oid}")
    if isinstance(strict, dict):
        ok = (strict.get("version") == STRICT_REJUDGE_VERSION
              and strict.get("body_source") == o.body_source
              and strict.get("body_provider") == o.body_provider
              and strict.get("body_hash") == _body_hash(body)
              and strict.get("constraint_id") == o.constraint_id
              and strict.get("slot_id") == o.slot_id)
        if ok:
            return str(strict.get("verdict") or "") or None, False
        return None, True                            # strict entry, wrong provenance/hash
    if (store or {}).get(f"rejudgment::{oid}"):
        return None, True                            # legacy pre-strict entry: ignored
    return None, False


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
                  passage: str, judge_cache_path: Path,
                  body_for_hash: str = "") -> tuple[str, bool]:
    """Run ONE targeted re-judgment live on a passage (5k-4). Inputs are passage + obligation
    metadata only — never gold; contaminated sources are excluded upstream. Records the verdict
    into the run's judge cache (strict versioned namespace) so the next replay is offline."""
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
    # 5n-5: persist under the STRICT versioned namespace with full body provenance, so a
    # future replay accepts it only when the same actual body backs it.
    if was_live and judge_cache_path:
        try:
            store = (json.loads(judge_cache_path.read_text(encoding="utf-8"))
                     if judge_cache_path.exists() else {})
            store[f"{STRICT_REJUDGE_VERSION}::{o.pending_read_judgment_id}"] = {
                "version": STRICT_REJUDGE_VERSION, "verdict": jd.judgment,
                "body_source": o.body_source, "body_provider": o.body_provider,
                "body_hash": _body_hash(body_for_hash), "passage_hash": _body_hash(passage),
                "source_url": _norm_url(o.source_url)[0] + "/" + _norm_url(o.source_url)[1],
                "candidate_id": o.candidate_id, "candidate_text": candidate_text,
                "slot_id": o.slot_id, "constraint_id": o.constraint_id,
                "contaminated": False}
            judge_cache_path.write_text(json.dumps(store, sort_keys=True), encoding="utf-8")
        except Exception:
            pass
    return jd.judgment, was_live


#: body sources counted as ACTUAL bodies (debug_snippet_only is diagnostic-only — 5m-2/5).
#: body sources counted as ACTUAL read bodies (5n-1): search/debug snippets never qualify.
_REAL_BODY_SOURCES = ("cache_read_body", "call_embedded_read_body", "replay_export_body")
#: per-obligation pipeline stages that imply an actual body was located.
_BODY_STAGES = ("actual_body_located", "passages_scanned", "rejudgment_pending",
                "judged_unclosed", "closed")


def _summarize_legacy(res: ReplayValidationResult, n_items: int, live_calls: int, *,
                      coverage_bounded: bool = False, allow_live_judge: bool = False,
                      cache_report: Optional[dict] = None,
                      rejudge_version_mismatches: int = 0) -> None:
    obs = res.obligations
    stage = Counter(o.pipeline_status for o in obs)
    reasons = Counter(o.stage_reason for o in obs if o.stage_reason)
    methods = Counter(o.reconstruction_method for o in obs if o.reconstruction_method)
    # STRICT semantics (5n): only ACTUAL read bodies count; search/debug snippets are
    # diagnostic and never advance the pipeline or feed the judge.
    real_body = [o for o in obs if o.body_is_actual_read_body]
    n_scanned = sum(1 for o in real_body
                    if o.pipeline_status in ("passages_scanned", "rejudgment_pending",
                                             "judged_unclosed", "closed"))
    judged = sum(1 for o in obs if o.pipeline_status in ("judged_unclosed", "closed"))
    closed = sum(1 for o in obs if o.pipeline_status == "closed"
                 and o.closure_code in _CLOSED and o.body_is_actual_read_body)
    res.metrics = {
        "n_items_inspected": n_items,
        "replay_pending_read_judgment_count": len(obs),
        "reconstructed_count": sum(1 for o in obs if o.reconstructed_from_legacy_trace),
        "reconstruction_method_counts": dict(methods),
        "read_matched_count": sum(1 for o in obs if o.matched_read),
        "body_located_count": len(real_body),
        "passages_scanned_count": n_scanned,
        "debug_snippet_scanned_count": sum(
            1 for o in obs if o.pipeline_status == "debug_snippet_scanned"),
        "search_snippet_only_count": sum(1 for o in obs if o.body_is_search_snippet),
        "passages_found_beyond_4000_count": sum(
            1 for o in real_body if o.passages_found_beyond_4000),
        "body_truncated_before_relevant_passage_count":
            sum(1 for o in real_body if o.body_truncated_before_relevant_passage),
        "judged_count": judged,
        "closed_count": closed,
        # 5n-4 passage-relevance split (subject hits alone are not judgeable).
        "predicate_relevant_passage_count": sum(
            1 for o in real_body if o.passage_relevance == "predicate_relevant"),
        "subject_only_passage_count": sum(
            1 for o in real_body if o.passage_relevance == "subject_only"),
        "alias_only_passage_count": sum(
            1 for o in real_body if o.passage_relevance == "subject_only"
            and o.anchor_category_counts.get("candidate_alias", 0) > 0
            and o.anchor_category_counts.get("source_title", 0) == 0),
        "predicate_relevant_passages_scanned_count": n_scanned,
        "pipeline_status_counts": dict(stage),
        "stage_reason_counts": dict(reasons),
        "url_match_method_counts": dict(Counter(o.url_match_method for o in obs)),
        "body_source_counts": dict(Counter(o.body_source for o in obs if o.body_source)),
        "body_provider_counts": dict(Counter(o.body_provider for o in obs if o.body_provider)),
        "recorded_rejudgment_cache_version_mismatch_count": rejudge_version_mismatches,
        # INVARIANTS (5n-2/3), pinned 0 by construction:
        "matched_read_with_no_url_match_method_count": sum(
            1 for o in obs if o.matched_read and o.url_match_method == "none"),
        "actual_body_located_without_read_body_provenance_count": sum(
            1 for o in obs if o.body_is_actual_read_body and o.body_match_source not in
            ("read_call", "read_provider_cache", "replay_export")),
        "search_snippet_counted_as_body_count": sum(
            1 for o in obs if o.body_is_search_snippet
            and o.pipeline_status in _BODY_STAGES),
        "reconstruction_coverage_bounded": coverage_bounded,
        "cache_report": dict(cache_report or {}),
        "live_provider_calls": 0, "live_model_calls": live_calls,
    }
    # 5n-6: unambiguous live-judge tier; only actual-body PREDICATE-RELEVANT passages qualify.
    if allow_live_judge:
        skip = None
        if not obs:
            skip = "no_reconstructed_obligations"
        elif n_scanned == 0:
            skip = "no_actual_body_predicate_relevant_passages_to_judge"
        res.metrics["live_judge_tier"] = {"enabled": True, "live_judge_skipped_reason": skip,
                                          "live_model_calls": live_calls}
    else:
        res.metrics["live_judge_tier"] = {"enabled": False, "live_judge_skipped_reason": None,
                                          "live_model_calls": 0}
    # overall status (5n-3): "validated_closed" is reserved for actual replay CLOSURE.
    # judged-but-unclosed (e.g. requires_read_still_open) is NOT validated.
    if not obs:
        res.overall_status = ("no_reconstructed_obligations" if n_items > 0
                              else "unvalidated_cache_miss")
    elif closed > 0:
        res.overall_status = "validated_closed"
    elif judged > 0:
        res.overall_status = "judged_unclosed"
    elif n_scanned > 0 or any(o.pipeline_status == "actual_body_located" for o in obs):
        res.overall_status = "reconstructed_actual_body_rejudgment_pending"
    elif any(o.body_is_search_snippet for o in obs):
        res.overall_status = "reconstructed_search_snippet_only"
    elif any(o.pipeline_status == "debug_snippet_scanned" for o in obs):
        res.overall_status = "reconstructed_debug_only"
    else:
        res.overall_status = "reconstructed_body_missing"


def inspect_run_schema(run_dir: str | Path) -> dict[str, Any]:
    """5l-7: a LIGHT zero-call schema probe — a safety net against future schema drift. Per
    debug record: top-level keys, interpretation/assertion/judgment-by-verdict counts, events
    count + events_truncated, calls-by-tool, and cache files + whether entries carry raw."""
    run_dir = Path(run_dir)
    out: dict[str, Any] = {"path": str(run_dir), "exists": run_dir.exists(), "records": [],
                           "cache_files": [], "live_provider_calls": 0, "live_model_calls": 0}
    dqf = run_dir / "debug_questions.jsonl"
    if dqf.exists():
        for line in dqf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                out["records"].append({"error": "line_not_json"})
                continue
            cf = rec.get("candidate_frontier") or {}
            verdicts: Counter = Counter()
            n_assert = 0
            for interp in (cf.get("interpretations") or []):
                for a in (interp.get("candidate_assertions") or []):
                    n_assert += 1
                    for j in (a.get("judgments") or []):
                        verdicts[j.get("judgment", "?")] += 1
            out["records"].append({
                "item_id": rec.get("item_id", ""),
                "top_level_keys": sorted(rec.keys()),
                "n_interpretations": len(cf.get("interpretations") or []),
                "n_candidate_assertions": n_assert,
                "judgment_verdict_counts": dict(verdicts),
                "n_events": len(cf.get("events") or []),
                "events_truncated": cf.get("events_truncated", 0),
                "calls_by_tool": dict(Counter(c.get("tool", "?") for c in (rec.get("calls") or []))),
            })
    # 5m-1: the provider recording cache under cache/ is inspected RECURSIVELY via the same
    # discovery the body locator uses (manifest cache.path + cache/ rglob + *cache*.json) —
    # never silently ignored.
    bodies, cache_report = _load_provider_bodies(run_dir)
    out["cache_files"] = cache_report.get("files", [])
    out["cache_report"] = {k: v for k, v in cache_report.items() if k != "files"}
    out["cache_report"]["urls_with_bodies"] = len(bodies)
    # 5n-7: provider-class splits are computed in the shared cache report
    # (read_body_entries_by_provider / search_snippet_entries_by_provider /
    # model_entries_by_provider / urls_with_actual_read_bodies / urls_with_search_snippets_only).
    return out


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
