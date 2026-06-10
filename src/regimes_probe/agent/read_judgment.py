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
                "read_tool": self.read_tool}


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
