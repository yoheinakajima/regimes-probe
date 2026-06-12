"""Level 5h-A/B: close the read -> judge loop with targeted passage retrieval.

When the evidence judge returns ``requires_read`` for a specific
``(candidate, slot, constraint, source_url)`` triple, that is a *pending obligation*,
not a vague hint to "go read something". This module makes that obligation a
first-class, persistent object (``PendingReadJudgment``) and provides the
deterministic passage retrieval (``extract_passages``) used to re-judge the SAME
triple against the fetched page body — not against the truncated snippet that
created the obligation, and not against only the first N characters of the page.

Everything here is generic (no benchmark ids / gold answers / names / URLs),
deterministic, and answer-free. The judge call (if any) is the existing narrow
``EvidenceJudge``; this module only routes passages into it and records the
resolution. Config knobs keep the default cheap and bounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&-]*")
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
#: generic answer-role / relation cue words searched for inside a read body when a
#: target-answer or relation constraint is pending (NOT benchmark-specific anchors).
_RELATION_CUES = frozenset({
    "born", "birth", "founded", "opened", "established", "from", "until", "since",
    "between", "worked", "served", "studied", "graduated", "died", "death",
    "located", "designed", "wrote", "directed", "published", "named", "year", "years"})
_STOP = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "for", "with", "by",
    "is", "are", "was", "were", "be", "as", "that", "this", "it", "from", "which",
    "who", "what", "where", "when", "how", "their", "his", "her", "its"})
#: 5t-4: relation-cue words too generic to establish PREDICATE relevance on their own (a
#: matched "from"/"year" says nothing about the constraint's predicate). The specific verbs
#: in ``_RELATION_CUES`` (born/founded/studied/...) still count as predicate anchors.
_GENERIC_RELATION_CUES = frozenset({"from", "until", "since", "between", "year", "years"})

#: 5t-4: strict rejudgment namespace SHARED between the live run and replay validation, so
#: a live targeted rejudgment is persisted under exactly the key/format the validator reads
#: (the 5s smoke showed live rejudgments invisible to replay: rejudgment_prompt_not_in_cache).
STRICT_REJUDGE_VERSION = "read_judge_replay_strict_body_v2"
#: body_source marker for entries recorded BY THE LIVE RUN (the validator accepts them only
#: when it independently locates the same actual read body — hash-checked, never snippets).
LIVE_RUN_BODY_SOURCE = "live_run_read_body"


def strict_rejudgment_key(pending_read_judgment_id: str) -> str:
    return f"{STRICT_REJUDGE_VERSION}::{pending_read_judgment_id}"


def body_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def normalize_service_url(url: str) -> str:
    """5u-1: the URL-group normalization shared by the service registry and validation."""
    return (url or "").strip().lower().rstrip("/")


#: 5u-1: terminal per-obligation service statuses. Every pending obligation must end the
#: item in exactly one of these (event-derived, never ambiguous control flow).
#: ``service_not_attempted_invariant_violation`` is reserved for true bugs (an open, clean,
#: unserved obligation while budget remained) and is pinned 0 in normal mechanics.
SERVICE_TERMINAL_STATUSES = (
    "service_attempted_success", "service_attempted_failed",
    "service_blocked_contaminated_or_noise", "service_blocked_disallowed_tool",
    "service_blocked_no_clean_url", "service_blocked_source_not_readable",
    "service_already_satisfied_by_same_url_read", "service_deduped_to_url_group",
    "service_budget_exhausted", "service_suppressed_non_executable",
    "service_not_attempted_invariant_violation")

#: 5u-3: explicit targeted-rejudgment lifecycle statuses. Every ATTEMPTED rejudgment lands
#: in exactly one bucket; ``rejudgment_missing_invariant_violation`` is reserved for a
#: claimed-recorded verdict whose strict cache entry is entirely absent (a true bug).
REJUDGMENT_STATUSES = (
    "rejudgment_not_needed_no_predicate_passage", "rejudgment_attempted_recorded",
    "rejudgment_attempted_model_error", "rejudgment_attempted_cache_write_failed",
    "rejudgment_attempted_invalid_response_recorded_fail_closed",
    "rejudgment_skipped_budget_exhausted", "rejudgment_skipped_body_hash_mismatch",
    "rejudgment_skipped_contaminated_or_noise", "rejudgment_missing_invariant_violation")

#: judge verdicts the route recognizes; anything else is recorded FAIL-CLOSED as
#: requires_read (5u-3: invalid_response_recorded_fail_closed, never silently dropped).
KNOWN_REJUDGE_VERDICTS = frozenset({
    "full_support", "partial_support", "contradiction", "irrelevant",
    "requires_read", "still_unresolved", "insufficient"})


@dataclass(frozen=True)
class ReadJudgmentConfig:
    """Cheap, deterministic defaults — do NOT raise the cap as the only fix (5h-B).

    5k-6 read-cap policy (config, not hardcode): the GLOBAL ``page_fetch`` adapter cap stays at
    its conservative default (4000 chars) so ordinary reads stay cheap; a read that is BACKED by
    a pending read-judgment may use a higher ``read_judgment_max_chars`` cap, because the judge
    input is already bounded by ``read_passage_window_chars`` (so a bigger fetch does not blow up
    judge cost — it only widens where a passage can be found). A single bounded RE-READ at
    ``reread_max_chars`` is allowed (live runs only) when a truncated read cannot close a pending
    obligation; replay never fetches."""
    read_max_chars_total: int = 24000
    read_passage_window_chars: int = 400
    read_max_passages_per_pending_judgment: int = 4
    #: global adapter default (kept conservative/cheap).
    page_fetch_default_max_chars: int = 4000
    #: higher cap for a read-judgment-backed read (5m-7: 20000, since judge input stays
    #: bounded by passage windows — a bigger retrieval body widens where a passage can be
    #: found without increasing the judge prompt size; storage cost is the only growth).
    read_judgment_max_chars: int = 20000
    #: single bounded re-read cap when a truncated read cannot close a pending obligation.
    reread_max_chars: int = 24000
    #: never re-read unboundedly; a re-read fires at most once per obligation.
    max_rereads_per_pending_judgment: int = 1
    #: 5m-7: store bounded RAW payloads for read-class tools in live runs (config-gated;
    #: contamination-safe via the recording cache's sanitizer), so future replay validation
    #: never has to fall back to debug snippets.
    store_raw_read_tools: bool = False
    read_cache_raw_chars: int = 40000


DEFAULT_READ_CONFIG = ReadJudgmentConfig()


def _content_tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")
            if w.lower() not in _STOP and len(w) >= 2]


def _normalize_year_tokens(tokens: list[str]) -> set[str]:
    """Collapse bare 4-digit years to a single ``<year>`` class so "1955" matches "1955"
    regardless of surrounding punctuation, while keeping the literal too."""
    out: set[str] = set()
    for t in tokens:
        out.add(t)
        if _YEAR.fullmatch(t):
            out.add("<year>")
    return out


@dataclass
class PassageScan:
    """Result of scanning a read body for a pending judgment's anchors (5h-B)."""
    passages: list[str] = field(default_factory=list)
    matched_anchors: list[str] = field(default_factory=list)
    missing_anchors: list[str] = field(default_factory=list)
    hit: bool = False                 # at least one direct anchor hit
    head_only: bool = False           # the document was shorter than one window (head == all)
    input_chars: int = 0              # 5j-B: chars of body handed in (pre-cap)
    scanned_chars: int = 0            # 5j-B: chars actually scanned (after read_max_chars_total)
    first_hit_offset: int = -1        # 5j-B: char offset of the first anchor hit (-1 = none)

    @property
    def passage_anchor_hits(self) -> int:
        return len(self.matched_anchors)

    @property
    def passage_count(self) -> int:
        return len(self.passages)

    def to_dict(self) -> dict[str, Any]:
        return {"n_passages": len(self.passages),
                "matched_anchors": list(self.matched_anchors)[:12],
                "missing_anchors": list(self.missing_anchors)[:12],
                "hit": self.hit, "head_only": self.head_only,
                "input_chars": self.input_chars, "scanned_chars": self.scanned_chars,
                "first_hit_offset": self.first_hit_offset,
                "passage_anchor_hits": self.passage_anchor_hits,
                "passage_count": self.passage_count}


def extract_passages(text: str, anchors, *, config: ReadJudgmentConfig = DEFAULT_READ_CONFIG,
                     extra_terms=()) -> PassageScan:
    """Find compact passages around anchor hits in a read body (5h-B).

    ``anchors`` are concrete strings to search for (constraint terms, target-slot
    descriptor words, subject aliases, relation cues, years). A passage is a window of
    ``read_passage_window_chars`` characters centred on a hit. When there are no direct
    hits we fall back to the highest lexical-overlap windows (deterministic), so the judge
    still sees the most relevant part of the page rather than the head. The whole body is
    bounded by ``read_max_chars_total`` first, so cost stays cheap and replayable."""
    raw = " ".join((text or "").split())
    body = raw[: config.read_max_chars_total]
    win = config.read_passage_window_chars
    head_only = len(body) <= win
    anchor_list = [str(a).strip() for a in anchors if str(a).strip()]
    extra = {str(t).lower() for t in extra_terms if str(t).strip()}
    scan = PassageScan(head_only=head_only, input_chars=len(raw), scanned_chars=len(body))
    if not body:
        scan.missing_anchors = list(dict.fromkeys(anchor_list))
        return scan
    low = body.lower()
    spans: list[tuple[int, int]] = []
    matched: list[str] = []
    for a in anchor_list:
        al = a.lower()
        idx = low.find(al)
        if idx >= 0:
            if a not in matched:
                matched.append(a)
            if scan.first_hit_offset < 0 or idx < scan.first_hit_offset:
                scan.first_hit_offset = idx
            half = max(0, (win - len(al)) // 2)
            spans.append((max(0, idx - half), min(len(body), idx + len(al) + half)))
    scan.matched_anchors = matched
    scan.missing_anchors = [a for a in anchor_list if a not in matched]
    if spans:
        scan.hit = True
        spans.sort()
        merged: list[list[int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        scan.passages = [body[s:e].strip()
                         for s, e in merged[: config.read_max_passages_per_pending_judgment]]
        return scan
    # no direct anchor hit: rank fixed-size windows by anchor/extra/relation token overlap.
    want = ({t for a in anchor_list for t in _content_tokens(a)} | extra | _RELATION_CUES)
    if not want:
        scan.passages = [body[:win]] if body else []
        return scan
    best: list[tuple[int, int]] = []      # (score, start)
    step = max(1, win // 2)
    for start in range(0, max(1, len(body) - win + 1), step):
        chunk = low[start:start + win]
        score = sum(1 for t in want if t in chunk)
        if score > 0:
            best.append((score, start))
    best.sort(key=lambda x: (-x[0], x[1]))
    scan.passages = [body[s:s + win].strip()
                     for _, s in best[: config.read_max_passages_per_pending_judgment]]
    if not scan.passages and body:
        scan.passages = [body[:win]]
    return scan


@dataclass
class PendingReadJudgment:
    """A persistent obligation created when the judge says ``requires_read`` (5h-A)."""
    pending_read_judgment_id: str
    candidate_id: str
    slot_id: str
    constraint_id: str
    source_url: str = ""
    source_subject: str = ""
    missing_anchors: list[str] = field(default_factory=list)
    target_terms: list[str] = field(default_factory=list)
    created_step: int = 0
    resolved_step: Optional[int] = None
    #: full_support | partial_support | contradiction | irrelevant |
    #: still_unresolved | read_failed | no_relevant_passage
    resolution: str = "open"
    read_selected: bool = False
    passage_preview: str = ""
    # 5q: source-safety + execution linkage (persisted for replay validation).
    source_role: str = ""
    suppressed_reason: str = ""
    selected_read_url: str = ""
    read_tool: str = ""
    # 5t: pending-service + targeted-rejudgment state (persisted for replay validation).
    #: end-of-item service reason — one of the pending_service_* vocabulary; never ambiguous.
    service_status: str = ""
    #: closure-state vocabulary after a targeted rejudgment (resolved_full_support /
    #: resolved_contradiction / requires_read_still_open / irrelevant_after_read /
    #: partial_support_after_read / no_relevant_passage /
    #: body_truncated_before_relevant_passage(raw_unavailable)).
    closure_state: str = ""
    #: lightweight LIVE passage relevance (predicate_relevant|subject_only|no_relevant_anchor)
    passage_relevance: str = ""
    #: an actual read body for this obligation's URL was acquired during the run.
    body_available: bool = False
    #: the targeted rejudgment ran / was persisted under the strict replay namespace.
    rejudgment_attempted: bool = False
    rejudgment_recorded: bool = False
    #: a bounded predicate re-read is scheduled for this obligation (consumed by the route).
    reread_pending: bool = False
    # 5u-1/3: explicit service + rejudgment lifecycle (persisted; never ambiguous).
    #: detail behind service_status (block reason / outcome detail / dedupe marker).
    service_stage_reason: str = ""
    #: URL-group id when several obligations share one normalized URL.
    service_group_id: str = ""
    #: concrete block reason when blocked/suppressed (reading-policy reason or suppression).
    service_block_reason: str = ""
    #: read tool + outcome of the service attempt for this obligation's URL group.
    service_read_success: Optional[bool] = None
    service_budget_remaining_when_decided: Optional[int] = None
    #: one of REJUDGMENT_STATUSES once a body was routed for this obligation.
    rejudgment_status: str = ""

    @property
    def open(self) -> bool:
        return self.resolution in ("open",)

    def to_dict(self) -> dict[str, Any]:
        return {"pending_read_judgment_id": self.pending_read_judgment_id,
                "candidate_id": self.candidate_id, "slot_id": self.slot_id,
                "constraint_id": self.constraint_id,
                # 5p-1: persist the FULL source_url (bounded) — host-only persistence made
                # native obligations unmatchable against read bodies in replay validation.
                "source_url": (self.source_url or "")[:300],
                "source_url_host": _host(self.source_url),
                "source_subject": self.source_subject[:80],
                "missing_anchors": list(self.missing_anchors)[:8],
                "target_terms": list(self.target_terms)[:8],
                "created_step": self.created_step, "resolved_step": self.resolved_step,
                "resolution": self.resolution, "read_selected": self.read_selected,
                "passage_preview": self.passage_preview[:160],
                "source_role": self.source_role,
                "suppressed_reason": self.suppressed_reason,
                "selected_read_url": (self.selected_read_url or "")[:300],
                "read_tool": self.read_tool,
                "service_status": self.service_status,
                "closure_state": self.closure_state,
                "passage_relevance": self.passage_relevance,
                "body_available": self.body_available,
                "rejudgment_attempted": self.rejudgment_attempted,
                "rejudgment_recorded": self.rejudgment_recorded,
                "service_stage_reason": self.service_stage_reason,
                "service_url": (self.source_url or "")[:300],
                "normalized_service_url": normalize_service_url(self.source_url)[:300],
                "service_group_id": self.service_group_id,
                "service_block_reason": self.service_block_reason,
                "service_attempted_read_tool": self.read_tool,
                "service_read_success": self.service_read_success,
                "service_read_body_linked": self.body_available,
                "service_budget_remaining_when_decided":
                    self.service_budget_remaining_when_decided,
                "rejudgment_status": self.rejudgment_status}


def _host(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url or "").hostname or "").lower()
    except Exception:
        return ""


def build_anchor_terms(constraint, *, slot=None, aliases=(), include_relation_cues: bool = True):
    """Assemble the concrete anchor strings to scan a read body for, for one pending triple.

    Generic: constraint terms + the constraint's literal years + target-slot descriptor
    content words + subject aliases + relation cue words. No benchmark-specific phrases."""
    anchors: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = (s or "").strip()
        if s and s.lower() not in seen:
            anchors.append(s)
            seen.add(s.lower())

    for t in getattr(constraint, "normalized_terms", []) or []:
        if len(str(t)) >= 3:
            _add(str(t))
    for y in _YEAR.findall(getattr(constraint, "text_span", "") or ""):
        _add(y)
    if slot is not None:
        for w in _content_tokens(getattr(slot, "descriptor_text", "")
                                 or getattr(slot, "slot_name", "")):
            if len(w) >= 4:
                _add(w)
    for al in aliases:
        _add(str(al))
    if include_relation_cues:
        for cue in sorted(_RELATION_CUES):
            _add(cue)
    return anchors
