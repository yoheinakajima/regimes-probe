"""ActiveGraph-native evidence interpretation layer (Level 5d).

Retrieval is not understanding. The parser and frontier-proposal layers now produce
good task frames and good queries, but raw results still fill candidate slates with page
chrome and generic terms ("Datasets", "Hugging Face", "Translate", "Login", "Merriam",
"Username Generator", "FOUNDER Definition", …), and constraint support stays near zero
even when the right entity appears. The fix is to stop populating slates from raw n-grams
and instead **interpret each result into structured assertions**:

- a **source role** (is this a primary source / professional profile / scholarly paper /
  database record … or a definition page / UI-navigation chrome / contaminated page?),
- **candidate assertions** (this text is a plausible entity of a slot's role, with a
  quote, proposed slot(s), and which constraints it supports/contradicts), and
- **constraint assertions** (this constraint is supported/contradicted/irrelevant, with a
  quote) — so constraint support changes *only* via an explicit assertion, never from
  arbitrary term overlap on a noise page.

The classification is **generic** — source *role* + page *intent*, not a BrowseComp domain
stoplist: dictionary pages define terms (they do not identify task entities), UI/navigation
text is not evidence, contaminated pages cannot support answers, social/forum pages are
weak. A small set of well-known platform *hosts* is used only to TYPE a source's role
(``linkedin.com/in`` → professional_profile), never as the rejection mechanism.

Deterministic by default; an optional cached/replayable LLM interpreter hook
(``--enable-llm-evidence-interpreter``) may only re-classify the source role / accept-reject
the bounded snippet — it never answers globally, and replay/dry-run make zero model calls.
Everything is event-sourced + projected so slates, assertions, and constraint support
replay identically and are learnable from traces. Nothing here reaches policy memory.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from regimes_probe.agent.candidate_frontier import (
    _contradicts, _hash, _noise_kind, _prev, _role_compatible, _supports)
from regimes_probe.agent.clue_resolution import (
    ROLES, _entities_in_field, _is_generic_entity, _norm, classify_entity_role)
from regimes_probe.agent.hypothesis_table import _context_role, _host

INTERPRETER_VERSION = "v1"

SOURCE_ROLES = (
    "primary_source", "professional_profile", "official_page", "article",
    "scholarly_paper", "database_record", "directory_listing", "social_page",
    "forum_page", "generic_definition_page", "benchmark_contaminated",
    "ui_or_navigation_noise", "unknown")

#: Source roles that cannot generate task-specific candidates / constraint support.
HARD_NOISE_ROLES = frozenset(
    {"generic_definition_page", "ui_or_navigation_noise", "benchmark_contaminated"})
#: Weak sources: allowed, but candidate confidence is downweighted.
WEAK_ROLES = frozenset({"social_page", "forum_page"})
#: Sources trustworthy enough that ONE distinctive constraint anchor (not two-term
#: overlap) plus a role-compatible candidate is enough to attach constraint support.
CORROBORATING_ROLES = frozenset({"professional_profile", "official_page", "scholarly_paper",
                                 "database_record", "primary_source", "directory_listing"})

_MONTHS = frozenset({"january", "february", "march", "april", "may", "june", "july",
                     "august", "september", "october", "november", "december"})
_WEEKDAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday",
                       "saturday", "sunday"})
#: generic category/type words that, ALONE, are not a task entity (need a local predicate).
_GENERIC_TYPE_WORDS = frozenset({
    "tv", "show", "shows", "series", "film", "films", "movie", "movies", "publication",
    "publications", "article", "articles", "report", "reports", "page", "pages", "website",
    "site", "overview", "summary", "list", "lists", "category", "categories", "results",
    "answers", "answer", "visit", "select", "available", "religious", "religion", "various",
    "general", "related", "more", "other", "news", "story", "stories", "topic", "topics"})
_TEMPORAL_PREDICATES = ("opened", "open", "founded", "established", "launched", "aired",
                        "premiered", "released", "published", "born", "built", "created",
                        "incorporated", "debuted", "started", "began", "formed")
#: venue/organization SUFFIX words that type an entity by its own name (a local predicate):
#: "Pecos Trail Inn"/"Pecos Trail Cafe" are organizations, not persons.
_ORG_SUFFIX = frozenset({
    "inn", "cafe", "café", "hotel", "motel", "lodge", "resort", "restaurant", "diner",
    "bistro", "tavern", "bar", "pub", "grill", "cantina", "museum", "gallery", "theatre",
    "theater", "church", "cathedral", "temple", "monastery", "library", "observatory",
    "company", "corporation", "inc", "corp", "ltd", "llc", "foundation", "institute",
    "university", "college", "school", "hospital", "society", "association", "consortium",
    "agency", "bureau", "department", "ministry", "commission", "club", "academy"})

# --- generic source-TYPE host lexicons (role typing only, NOT a rejection stoplist) ---
_DEFINITION_HOSTS = frozenset({
    "merriam-webster.com", "dictionary.com", "wiktionary.org", "thesaurus.com",
    "vocabulary.com", "yourdictionary.com", "collinsdictionary.com", "dictionary.cambridge.org",
    "definitions.net", "wordnik.com"})
_PROFILE_HOSTS = frozenset({"linkedin.com"})
_SOCIAL_HOSTS = frozenset({"twitter.com", "x.com", "facebook.com", "instagram.com",
                           "tiktok.com", "threads.net", "bsky.app", "mastodon.social"})
_FORUM_HOSTS = frozenset({"reddit.com", "quora.com", "stackexchange.com",
                          "stackoverflow.com", "news.ycombinator.com"})
_SCHOLARLY_HOSTS = frozenset({"arxiv.org", "pubmed.ncbi.nlm.nih.gov", "doi.org",
                              "semanticscholar.org", "jstor.org", "researchgate.net",
                              "biorxiv.org", "ssrn.com", "nature.com", "sciencedirect.com"})
_DB_HOSTS = frozenset({"wikidata.org", "imdb.com", "musicbrainz.org", "viaf.org",
                       "worldcat.org", "geonames.org", "openlibrary.org"})
_DIRECTORY_HOSTS = frozenset({"yelp.com", "tripadvisor.com", "yellowpages.com",
                              "crunchbase.com", "zoominfo.com"})

#: page-intent cues
_DEFINITION_CUES = ("definition", "meaning of", "dictionary", "thesaurus", "synonyms of",
                    "pronunciation", "what does", "spelling of", "crossword clue")
#: single-word / chrome titles that are navigation, not content.
_NAV_TITLES = frozenset({
    "login", "log in", "sign in", "signin", "sign up", "signup", "register", "menu",
    "home", "homepage", "search", "settings", "translate", "download", "downloads",
    "datasets", "dataset", "models", "spaces", "docs", "documentation", "pricing",
    "about", "about us", "contact", "contact us", "subscribe", "newsletter", "cart",
    "checkout", "explore", "trending", "categories", "topics", "tags", "sitemap",
    "preferences", "language", "results", "browse", "directory", "index"})
_WORD = re.compile(r"[A-Za-z0-9]+")


def _base_host(host: str) -> str:
    h = (host or "").lower()
    if h.startswith("www."):
        h = h[4:]
    parts = h.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else h


def _is_ui_navigation(title: str, snippet: str) -> bool:
    t = " ".join((title or "").split()).strip().lower().rstrip(".")
    if t in _NAV_TITLES:
        return True
    # a very short chrome title with no real content snippet.
    if t and len(t.split()) <= 2 and t in _NAV_TITLES:
        return True
    return False


def classify_source_role(url: str, title: str, snippet: str,
                         contaminated: bool) -> tuple[str, list[str]]:
    """Classify a result/page by GENERIC source role + page intent (no domain stoplist).

    Returns ``(source_role, noise_reasons)``; ``noise_reasons`` is non-empty only for
    roles that cannot generate task candidates."""
    if contaminated:
        return "benchmark_contaminated", ["benchmark_contaminated_source"]
    host = _host(url)
    base = _base_host(host)
    tl = f"{title} {snippet}".lower()
    if _is_ui_navigation(title, snippet):
        return "ui_or_navigation_noise", ["ui_navigation_noise"]
    if base in _DEFINITION_HOSTS or any(c in tl for c in _DEFINITION_CUES):
        return "generic_definition_page", ["generic_definition_noise"]
    if base in _PROFILE_HOSTS and "/in/" in (url or "").lower():
        return "professional_profile", []
    if base in _SCHOLARLY_HOSTS:
        return "scholarly_paper", []
    if base in _DB_HOSTS:
        return "database_record", []
    if base in _DIRECTORY_HOSTS:
        return "directory_listing", []
    if base in _SOCIAL_HOSTS:
        return "social_page", []
    if base in _FORUM_HOSTS:
        return "forum_page", []
    if host.endswith(".gov") or host.endswith(".mil"):
        return "official_page", []
    if host.endswith(".edu"):
        return "official_page", []
    return "article", []


# --------------------------------------------------------------------------- records
@dataclass
class CandidateAssertion:
    candidate_text: str
    normalized_text_hash: str
    inferred_role: str
    assertion_id: str = ""
    proposed_slot_ids: list[str] = field(default_factory=list)
    supports_constraint_ids: list[str] = field(default_factory=list)
    contradicts_constraint_ids: list[str] = field(default_factory=list)
    evidence_quote_or_span: str = ""
    source_role: str = "unknown"
    confidence: float = 0.0
    rejection_reason: str = ""               # "" => accepted
    from_ctx: bool = False                   # role inferred from surrounding context
    #: filled after the canonical SlotCandidate is materialized (req 1): the id the
    #: frontier/hypothesis registry knows this candidate by, per accepted slot.
    canonical_candidate_ids: dict[str, str] = field(default_factory=dict)
    #: per-slot (supported, contradicted) constraint ids, computed by the support
    #: recognizers and APPLIED verbatim by the frontier (not recomputed from overlap).
    slot_support: dict[str, list[list[str]]] = field(default_factory=dict)
    #: Level 5f LLM evidence judge: per-slot partial-support + requires-read constraint ids,
    #: aliases the judge extracted, and the bounded judgment records (trace/projection).
    slot_partial: dict[str, list[str]] = field(default_factory=dict)
    slot_requires_read: dict[str, list[str]] = field(default_factory=dict)
    candidate_aliases: list[str] = field(default_factory=list)
    judgments: list[dict] = field(default_factory=list)
    #: Level 5i-K2: kept but flagged when the explicit-location facet is ambiguous for it.
    needs_location_support: bool = False

    @property
    def accepted(self) -> bool:
        return not self.rejection_reason

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_text": _prev(self.candidate_text, 80),
                "assertion_id": self.assertion_id,
                "normalized_text_hash": self.normalized_text_hash,
                "inferred_role": self.inferred_role,
                "proposed_slot_ids": list(self.proposed_slot_ids),
                "supports_constraint_ids": list(self.supports_constraint_ids),
                "contradicts_constraint_ids": list(self.contradicts_constraint_ids),
                "evidence_quote_or_span": _prev(self.evidence_quote_or_span, 160),
                "source_role": self.source_role, "confidence": round(self.confidence, 3),
                "rejection_reason": self.rejection_reason,
                "canonical_candidate_ids": dict(self.canonical_candidate_ids),
                "partial_support_constraint_ids": sorted(
                    {c for v in self.slot_partial.values() for c in v}),
                "requires_read_constraint_ids": sorted(
                    {c for v in self.slot_requires_read.values() for c in v}),
                "candidate_aliases": list(self.candidate_aliases)[:8],
                "judgments": list(self.judgments)[:12]}


@dataclass
class ConstraintAssertion:
    constraint_id: str
    status: str                               # supports | contradicts | irrelevant | insufficient
    evidence_quote_or_span: str = ""
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"constraint_id": self.constraint_id, "status": self.status,
                "evidence_quote_or_span": _prev(self.evidence_quote_or_span, 160),
                "confidence": round(self.confidence, 3), "reason": self.reason}


@dataclass
class EvidenceInterpretation:
    interpretation_id: str
    source_evidence_id: str
    source_tool: str
    source_url: str = ""
    source_domain: str = ""
    source_title_preview: str = ""
    source_snippet_preview: str = ""
    source_role: str = "unknown"
    target_slot_ids: list[str] = field(default_factory=list)
    tested_constraint_ids: list[str] = field(default_factory=list)
    candidate_assertions: list[CandidateAssertion] = field(default_factory=list)
    constraint_assertions: list[ConstraintAssertion] = field(default_factory=list)
    noise_reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    interpreter_version: str = INTERPRETER_VERSION
    interpreter_mode: str = "deterministic"
    source_subject: Any = None                # Level 5i-I: the source's extracted subject

    @property
    def is_noise_source(self) -> bool:
        return self.source_role in HARD_NOISE_ROLES

    def accepted_candidates(self) -> list[CandidateAssertion]:
        return [a for a in self.candidate_assertions if a.accepted]

    def to_dict(self) -> dict[str, Any]:
        return {"interpretation_id": self.interpretation_id,
                "source_evidence_id": self.source_evidence_id,
                "source_tool": self.source_tool, "source_url": self.source_url[:160],
                "source_domain": self.source_domain,
                "source_title_preview": _prev(self.source_title_preview, 100),
                "source_snippet_preview": _prev(self.source_snippet_preview, 160),
                "source_role": self.source_role,
                "target_slot_ids": list(self.target_slot_ids),
                "tested_constraint_ids": list(self.tested_constraint_ids),
                "candidate_assertions": [a.to_dict() for a in self.candidate_assertions],
                "constraint_assertions": [a.to_dict() for a in self.constraint_assertions],
                "noise_reasons": list(self.noise_reasons),
                "confidence": round(self.confidence, 3),
                "interpreter_version": self.interpreter_version,
                "interpreter_mode": self.interpreter_mode,
                "source_subject": (self.source_subject.to_dict()
                                   if self.source_subject is not None else None)}


#: a generic location gazetteer (US states + DC) used ONLY to detect explicit-location
#: facets and clear location MISMATCHES (5i-K2). Generic infrastructure, not benchmark data.
_US_STATES = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "ohio",
    "oklahoma", "oregon", "pennsylvania", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "wisconsin", "wyoming"})
_US_STATES_MULTIWORD = ("new mexico", "new york", "new hampshire", "new jersey",
                        "north carolina", "north dakota", "south carolina", "south dakota",
                        "rhode island", "west virginia", "district of columbia")
_LOCATION_BRANCH_CUES = ("branch", "location in", "also in", "outpost", "second location",
                         "opened a", "franchise")
#: a location is only read from free text when it follows a location preposition, so a state
#: NAME used as a person/brand name ("Georgia O'Keeffe") is not a false location match (5i-K2).
_LOC_PREP = re.compile(
    r"\b(?:in|at|near|from|located in|based in|of)\s+([a-z][a-z'.\- ]{2,40})")


def _gazetteer_match(text: str) -> set[str]:
    """Direct gazetteer match on CURATED text (frame terms): punctuation-robust via \\b."""
    low = (text or "").lower()
    found = {s for s in _US_STATES_MULTIWORD if re.search(rf"\b{re.escape(s)}\b", low)}
    return found | (set(re.findall(r"[a-z]+", low)) & _US_STATES)


def _locations_in_context(text: str) -> set[str]:
    """Locations named in FREE text, gated by a location preposition (conservative)."""
    low = (text or "").lower()
    found: set[str] = set()
    for m in _LOC_PREP.finditer(low):
        tail = m.group(1)
        for s in _US_STATES_MULTIWORD:
            if tail.startswith(s):
                found.add(s)
        for w in re.findall(r"[a-z]+", tail)[:2]:
            if w in _US_STATES:
                found.add(w)
                break
    return found


def explicit_location_terms(frame) -> set[str]:
    """Explicit location facet of a task frame (5i-K2): location tokens appearing in
    known-context terms or in any constraint's text/terms. Conservative — gazetteer-bounded."""
    locs: set[str] = set()
    for t in getattr(frame, "known_context_terms", []) or []:
        locs |= _gazetteer_match(t)
    for c in getattr(frame, "constraints", []) or []:
        locs |= _gazetteer_match(getattr(c, "text_span", "") or "")
        for term in getattr(c, "normalized_terms", []) or []:
            locs |= _gazetteer_match(str(term))
    return locs


def _quote_for(terms: list[str], title: str, snippet: str) -> str:
    """A short evidence span from title/snippet containing a constraint term (audit trail)."""
    for field_text in (title, snippet):
        low = (field_text or "").lower()
        for t in terms:
            if len(t) >= 4 and t in low:
                i = low.find(t)
                start = max(0, i - 30)
                return _prev((field_text or "")[start:i + len(t) + 40], 160)
    return _prev(snippet or title, 120)


_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "")}


def _distinctive_anchors(con) -> set[str]:
    """A constraint's *distinctive* anchor terms — proper-cased / rare / year tokens — that
    a result must contain to count as evidence (not generic glue words like 'the'/'report')."""
    from regimes_probe.policy.query_decomposition import _is_rare
    anchors: set[str] = set()
    span = getattr(con, "text_span", "") or ""
    proper = {w.lower() for w in re.findall(r"[A-Z][A-Za-z'&]{3,}", span)}
    for t in getattr(con, "normalized_terms", []):
        tl = str(t).lower()
        if len(tl) < 4:
            continue
        if tl in proper or _is_rare(t) or str(t)[:1].isupper():
            anchors.add(tl)
    anchors |= set(_YEAR.findall(span))
    return anchors


def recognize_constraint_support(con, cand_text, cand_role, source_role, title, snippet,
                                 *, contaminated: bool):
    """Generic deterministic recognizer: does THIS evidence support/contradict the
    constraint for this candidate? Support requires real anchors + an acceptable source —
    never arbitrary term overlap on a noise/contaminated page. Returns ``(status, quote)``."""
    text_l = f"{title} {snippet}".lower()
    if contaminated or source_role in HARD_NOISE_ROLES:
        return "irrelevant", ""
    if _contradicts(con, text_l, (cand_text or "").lower()):
        return "contradicts", _quote_for([str(t).lower() for t in con.normalized_terms],
                                          title, snippet)
    terms = [str(t) for t in getattr(con, "normalized_terms", [])]
    quote = _quote_for([t.lower() for t in terms], title, snippet)
    # (a) strong: the existing >=2-anchor-term overlap rule (preserves prior behavior).
    if _supports(con, text_l):
        return "supports", quote
    anchors = _distinctive_anchors(con)
    present = anchors & _tokens(text_l)
    # (b) temporal pattern: a constraint year present + an opening/founding/etc. predicate.
    cyears = set(_YEAR.findall(getattr(con, "text_span", "") or ""))
    if cyears and (cyears & set(_YEAR.findall(text_l))) and \
            any(p in text_l for p in _TEMPORAL_PREDICATES):
        return "supports", quote
    # (c) corroborated: ONE distinctive anchor from a trustworthy source typing a real
    #     entity (a professional profile / official / scholarly / db record / directory).
    if present and source_role in CORROBORATING_ROLES and cand_role not in ("unknown", ""):
        return "supports", quote
    if present:
        return "insufficient", quote
    return "irrelevant", ""


def _normalize_role(text: str, role_e: str, title: str, snippet: str) -> str:
    """Tighten the inferred role using local predicates so chrome/dates don't bind wrong
    slots: months/weekdays -> date_or_time; an explicit 'in <X>' location cue -> location."""
    tl = text.lower().strip()
    if tl in _MONTHS or tl in _WEEKDAYS:
        return "date_or_time"
    # an entity whose own last word is a venue/org type is an organization, not a person.
    words = [w.lower() for w in _WORD.findall(text)]
    if words and words[-1] in _ORG_SUFFIX:
        return "organization"
    if role_e in ("person", "unknown"):
        ctx = f"{title} {snippet}"
        if re.search(rf"\bin {re.escape(text)}\b", ctx) or re.search(
                rf"{re.escape(text)},\s+[A-Z]", ctx):
            # "located in <X>" / "<X>, State" reads as a place, not a person.
            return "location"
    return role_e


def _hygiene_reason(text: str, role_e: str) -> str:
    """Reject obvious non-entities BEFORE slot assignment (generic, not domain-specific)."""
    toks = [w.lower() for w in _WORD.findall(text or "")]
    content = [t for t in toks if len(t) >= 2]
    if not content:
        return "insufficient_context"
    # a phrase made up ENTIRELY of generic type/category words is not a task entity.
    if all(t in _GENERIC_TYPE_WORDS for t in content):
        return "generic_type_without_predicate"
    # a lone month/weekday is not a person/org candidate.
    if len(content) == 1 and (content[0] in _MONTHS or content[0] in _WEEKDAYS):
        return "date_or_weekday_not_entity"
    return ""


def _is_blocking(con) -> bool:
    return bool(getattr(con, "blocks_answer_if_unresolved", False)
                or getattr(con, "required", False)
                or getattr(con, "priority", "") == "high"
                or "can_block_answer" in getattr(con, "affordances", []))


def _object_anchored(con, subject_slot, frame, frontier, text_l: str) -> bool:
    """For a relational constraint (binds >=1 OTHER slot), is the dependent OBJECT anchored
    in this evidence? Generic: a candidate or known-context term for another applies_to slot
    appears in the text. No object candidate yet -> not anchored (full support must wait)."""
    other_slots = [sid for sid in getattr(con, "applies_to", [])
                   if sid != getattr(subject_slot, "slot_id", None)]
    if not other_slots:
        return True
    for sid in other_slots:
        slate = getattr(frontier, "slates", {}).get(sid)
        if slate is None:
            continue
        for cand in slate.candidates.values():
            if cand.candidate_text and cand.candidate_text.lower() in text_l:
                return True
    return False


def _slot_compatible(role_e: str, slot_role: str) -> bool:
    """Stricter than ``_role_compatible``: unknown never fans out, and a date/time entity
    only binds a date/time (or untyped) slot."""
    from regimes_probe.agent.candidate_frontier import _role_compatible
    if role_e in ("", "unknown"):
        return False
    if role_e == "date_or_time":
        return slot_role in ("date_or_time", "unknown")
    return _role_compatible(role_e, slot_role)


# --------------------------------------------------------------------------- interpreter
class EvidenceInterpreter:
    """Deterministic (default) evidence interpreter; optional cached/replayable LLM hook.

    The deterministic path classifies the source role, rejects candidates from noise
    sources / chrome text, and emits candidate + constraint assertions tied to the
    selected/role-compatible slots. The LLM hook (when armed) may only re-classify the
    source role from bounded snippets; it never answers and replay makes no model call.
    """

    def __init__(self, model_fn: Optional[Callable[[str], str]] = None, *,
                 cache=None, model: str = "deterministic", replay_only: bool = False,
                 enabled_llm: bool = False, judge=None) -> None:
        self.model_fn = model_fn
        self.cache = cache
        self.model = model
        self.replay_only = replay_only
        self.enabled_llm = enabled_llm
        #: Level 5f: optional LLM evidence judge for candidate/slot/constraint support fit.
        self.judge = judge
        self._ic = 0
        self.interpretation_count = 0
        self.model_calls = 0
        self.cache_hits = 0
        self.source_role_counts: Counter = Counter()
        self.candidate_assertion_count = 0
        self.accepted_candidate_count = 0
        self.rejected_candidate_count = 0
        self.rejection_counts: Counter = Counter()
        self.constraint_support_count = 0
        self.constraint_contradiction_count = 0
        #: Level 5i-I source-subject extraction + promotion hygiene.
        self.source_subject_extracted_count = 0
        self.source_subject_promoted_count = 0
        self.source_subject_rejected_count = 0
        self.source_subject_chrome_rejected_count = 0
        self.source_subject_title_only_rejected_count = 0
        self.source_subject_predicate_grounded_count = 0
        self.candidate_promoted_from_source_title_only_count = 0   # pinned 0
        self.candidate_promoted_from_chrome_count = 0              # pinned 0
        #: Level 5i-K1 pre-judge triage accounting.
        self.prejudge_rejected_count = 0
        self.prejudge_rejected_by_reason: Counter = Counter()
        self.judge_calls_saved_by_prejudge_triage = 0
        #: Level 5i-K2 explicit-location filtering.
        self.explicit_location_filter_applied_count = 0
        self.explicit_location_mismatch_rejected_count = 0
        self.explicit_location_ambiguous_kept_count = 0
        self.explicit_location_supported_count = 0

    def stats(self) -> dict[str, Any]:
        return {
            "evidence_interpreter_model": self.model,
            "evidence_interpretation_count": self.interpretation_count,
            "evidence_interpreter_model_calls": self.model_calls,
            "evidence_interpreter_cache_hits": self.cache_hits,
            "source_role_counts": dict(self.source_role_counts),
            "candidate_assertion_count": self.candidate_assertion_count,
            "accepted_candidate_assertion_count": self.accepted_candidate_count,
            "rejected_candidate_assertion_count": self.rejected_candidate_count,
            "candidate_assertion_rejection_counts": dict(self.rejection_counts),
            "constraint_assertion_support_count": self.constraint_support_count,
            "constraint_assertion_contradiction_count": self.constraint_contradiction_count,
            "evidence_judge_enabled": bool(self.judge is not None
                                           and getattr(self.judge, "enabled", False)),
            # Level 5i-I source-subject extraction + promotion hygiene.
            "source_subject_extracted_count": self.source_subject_extracted_count,
            "source_subject_promoted_count": self.source_subject_promoted_count,
            "source_subject_rejected_count": self.source_subject_rejected_count,
            "source_subject_chrome_rejected_count": self.source_subject_chrome_rejected_count,
            "source_subject_title_only_rejected_count": self.source_subject_title_only_rejected_count,
            "source_subject_predicate_grounded_count": self.source_subject_predicate_grounded_count,
            "candidate_promoted_from_source_title_only_count":
                self.candidate_promoted_from_source_title_only_count,        # pinned 0
            "candidate_promoted_from_chrome_count": self.candidate_promoted_from_chrome_count,  # 0
            # Level 5i-K1 pre-judge triage.
            "prejudge_rejected_count": self.prejudge_rejected_count,
            "prejudge_rejected_by_reason": dict(self.prejudge_rejected_by_reason),
            "judge_calls_saved_by_prejudge_triage": self.judge_calls_saved_by_prejudge_triage,
            "judge_invoked_on_obvious_chrome_count": 0,                      # pinned 0
            "judge_invoked_on_source_title_only_count": 0,                   # pinned 0 (concrete)
            "judge_invoked_on_generic_definition_candidate_count": 0,        # pinned 0 (concrete)
            # Level 5i-K2 explicit-location filtering.
            "explicit_location_filter_applied_count": self.explicit_location_filter_applied_count,
            "explicit_location_mismatch_rejected_count": self.explicit_location_mismatch_rejected_count,
            "explicit_location_ambiguous_kept_count": self.explicit_location_ambiguous_kept_count,
            "explicit_location_supported_count": self.explicit_location_supported_count,
            "explicit_location_mismatch_promoted_count": 0,                 # pinned 0
            **(self.judge.stats() if self.judge is not None else {}),
        }

    def _next_id(self) -> str:
        self._ic += 1
        return f"interp{self._ic}"

    def interpret(self, obs, *, frame, frontier, source_evidence_id: str, source_tool: str,
                  selected_slot_id: Optional[str] = None,
                  selected_constraint_ids: Optional[list[str]] = None,
                  known_norms: Optional[set[str]] = None) -> EvidenceInterpretation:
        title = getattr(obs, "title", "") or ""
        snippet = getattr(obs, "snippet", "") or ""
        url = getattr(obs, "url", "") or ""
        authority = float(getattr(obs, "source_authority", 0.0))
        contaminated = bool(getattr(obs, "benchmark_contaminated", False))
        known_norms = known_norms or set()
        sel_cons = list(selected_constraint_ids or [])

        role, noise_reasons = classify_source_role(url, title, snippet, contaminated)
        if self.enabled_llm:                       # optional cached/replayable re-classify
            role = self._llm_source_role(role, frame, title, snippet, url) or role
        interp = EvidenceInterpretation(
            interpretation_id=self._next_id(), source_evidence_id=source_evidence_id,
            source_tool=source_tool, source_url=url, source_domain=_host(url),
            source_title_preview=title, source_snippet_preview=snippet, source_role=role,
            target_slot_ids=([selected_slot_id] if selected_slot_id else []),
            tested_constraint_ids=sel_cons,
            interpreter_mode=("llm" if self.enabled_llm else "deterministic"),
            confidence=0.4 if role in WEAK_ROLES else (0.0 if role in HARD_NOISE_ROLES else 0.8))

        text_l = f"{title} {snippet}".lower()
        dslot = frame.slot(selected_slot_id) if selected_slot_id else None
        # Level 5i-I: extract the source's SUBJECT (what real-world entity the page is about)
        # before promoting candidates, so a source title/chrome/topic can't become a candidate
        # unless body predicate text grounds it.
        from regimes_probe.agent.source_subject import extract_source_subject
        con_terms = [t for cid in sel_cons
                     for c in frame.constraints if c.constraint_id == cid
                     for t in getattr(c, "normalized_terms", [])]
        ss = extract_source_subject(title=title, snippet=snippet, url=url, source_role=role,
                                    source_id=f"{interp.interpretation_id}_ss",
                                    constraint_terms=con_terms)
        interp.source_subject = ss
        self.source_subject_extracted_count += 1
        if ss.is_predicate_grounded:
            self.source_subject_predicate_grounded_count += 1
        if ss.is_chrome_or_source_title_only:
            if ss.subject_role in ("source_chrome", "generic_topic"):
                self.source_subject_chrome_rejected_count += 1
            else:
                self.source_subject_title_only_rejected_count += 1
        self._explicit_locations = explicit_location_terms(frame)   # K2 facet (per interpret)
        # entities -> candidate assertions (accepted or rejected, always recorded).
        for e in _entities_in_field(title) + _entities_in_field(snippet):
            if len(e) < 3 or _is_generic_entity(e):
                continue
            norm = _norm(e)
            if any(a.normalized_text_hash == _hash(norm) for a in interp.candidate_assertions):
                continue
            is_known = norm in known_norms
            if is_known:
                role_e, from_ctx = classify_entity_role(e), False
            else:
                role_e, from_ctx = _context_role(e, title, snippet)
            role_e = _normalize_role(e, role_e, title, snippet)
            assertion = self._assert_candidate(
                e, norm, role_e, frame, frontier, dslot, sel_cons, role, noise_reasons,
                text_l, title, snippet, is_known=is_known, from_ctx=from_ctx,
                contaminated=contaminated, source_url=url,
                source_id=source_evidence_id, source_subject=ss)
            assertion.assertion_id = f"{interp.interpretation_id}_a{len(interp.candidate_assertions)}"
            interp.candidate_assertions.append(assertion)
            # 5i-I promotion accounting (+ pinned-0 violation guards).
            if assertion.accepted and ss is not None and _norm(ss.subject_name) == norm:
                from regimes_probe.agent.source_subject import is_promotable_subject
                if is_promotable_subject(ss):
                    self.source_subject_promoted_count += 1
                elif ss.is_chrome_or_source_title_only:    # must never happen (rejected above)
                    if ss.subject_role in ("source_chrome", "generic_topic"):
                        self.candidate_promoted_from_chrome_count += 1
                    else:
                        self.candidate_promoted_from_source_title_only_count += 1
            elif not assertion.accepted and ss is not None and _norm(ss.subject_name) == norm:
                self.source_subject_rejected_count += 1
        # constraint assertions: the SELECTED/tested constraints, reflecting whether an
        # ACCEPTED candidate actually supports/contradicts them (req 2: support flows from
        # interpreted candidate evidence, not standalone overlap).
        supported_by_cand = {cid for a in interp.candidate_assertions if a.accepted
                             for cid in a.supports_constraint_ids}
        contradicted_by_cand = {cid for a in interp.candidate_assertions if a.accepted
                                for cid in a.contradicts_constraint_ids}
        for cid in sel_cons:
            con = next((c for c in frame.constraints if c.constraint_id == cid), None)
            if con is None:
                continue
            if cid in contradicted_by_cand:
                interp.constraint_assertions.append(ConstraintAssertion(
                    cid, "contradicts", reason="candidate_contradicts",
                    evidence_quote_or_span=_quote_for(
                        [str(t).lower() for t in con.normalized_terms], title, snippet)))
            elif cid in supported_by_cand:
                interp.constraint_assertions.append(ConstraintAssertion(
                    cid, "supports", confidence=0.8, reason="supported_by_accepted_candidate",
                    evidence_quote_or_span=_quote_for(
                        [str(t).lower() for t in con.normalized_terms], title, snippet)))
            else:
                interp.constraint_assertions.append(
                    self._assert_constraint(con, role, text_l, title, snippet))
        self._tally(interp)
        return interp

    def _assert_candidate(self, text, norm, role_e, frame, frontier, dslot, sel_cons,
                          source_role, noise_reasons, text_l, title, snippet,
                          *, is_known: bool, from_ctx: bool = False,
                          contaminated: bool = False, source_url: str = "",
                          source_id: str = "", source_subject=None) -> CandidateAssertion:
        a = CandidateAssertion(candidate_text=text, normalized_text_hash=_hash(norm),
                               inferred_role=role_e, source_role=source_role,
                               from_ctx=from_ctx)
        # 1) text-level chrome / platform / definition noise.
        nk = _noise_kind(text)
        if nk:
            a.rejection_reason = nk
            return a
        # 2) hygiene: generic-type-only phrases / lone dates+weekdays are not entities.
        hr = _hygiene_reason(text, role_e)
        if hr:
            a.rejection_reason = hr
            return a
        # 3) source-role noise: definition / ui / contaminated pages identify no entities.
        if source_role in HARD_NOISE_ROLES:
            a.rejection_reason = (noise_reasons[0] if noise_reasons
                                  else "no_slot_compatible_evidence")
            return a
        # 4) unknown role does NOT fan out into every slate (req 4): a weak observation
        #    only becomes a candidate once a local predicate disambiguates its role.
        if role_e in ("", "unknown"):
            a.rejection_reason = "weak_observation_not_candidate"
            return a
        # 4b) Level 5i-I/K1 pre-judge triage: the source's UNGROUNDED title/chrome/generic-topic
        #     subject is not a promotable candidate from the title alone. Reject BEFORE the
        #     judge (so no judge call is spent on a source-title-only / chrome candidate).
        if (not is_known and source_subject is not None
                and source_subject.is_chrome_or_source_title_only
                and _norm(source_subject.subject_name) == norm):
            reason = ("source_title_only_not_predicate_grounded"
                      if source_subject.subject_role not in ("source_chrome", "generic_topic")
                      else "source_chrome_or_generic_topic_subject")
            a.rejection_reason = reason
            self._note_prejudge_reject(reason, sel_cons)
            return a
        # 5) slot compatibility: role-compatible target/intermediate or selected slot
        #    (date/location typing respected; no unknown wildcard).
        slots = [s for s in frame.all_slots
                 if s.slot_id in frontier.slates and _slot_compatible(role_e, s.slot_role)]
        if dslot is not None and dslot.slot_id in frontier.slates and dslot not in slots \
                and _slot_compatible(role_e, dslot.slot_role):
            slots.append(dslot)
        if not slots:
            a.rejection_reason = "role_incompatible"
            return a
        # 5b) Level 5i-K2 explicit-location filter: when the task gives an explicit location
        #     facet, an organization/place candidate whose source clearly names a DIFFERENT
        #     location (with no branch/location evidence) is rejected BEFORE judging; an
        #     ambiguous one is kept but flagged needs_location_support.
        explicit_locs = getattr(self, "_explicit_locations", set())
        if (explicit_locs and not is_known
                and role_e in ("organization", "place", "location")):
            self.explicit_location_filter_applied_count += 1
            loc_status = self._location_compat(explicit_locs, text, title, snippet)
            if loc_status == "mismatch":
                a.rejection_reason = "explicit_location_mismatch"
                self.explicit_location_mismatch_rejected_count += 1
                self._note_prejudge_reject("explicit_location_mismatch", sel_cons)
                return a
            if loc_status == "supported":
                self.explicit_location_supported_count += 1
            elif loc_status == "ambiguous":
                self.explicit_location_ambiguous_kept_count += 1
                a.needs_location_support = True
        # 6) per-slot constraint support via the generic recognizers (NOT raw overlap).
        proposed, supports, contradicts = [], set(), set()
        for slot in slots:
            cons = list(frontier._constraints_for_slot(slot.slot_id))
            if slot.slot_id == (dslot.slot_id if dslot else None):
                cons += [c for c in (frontier._con(cid) for cid in sel_cons)
                         if c is not None and c not in cons]
            sup, con, partial, req_read = [], [], [], []
            judge_on = self.judge is not None and getattr(self.judge, "enabled", False)
            judge_cap = getattr(self.judge, "max_calls_per_candidate", 6) if judge_on else 0
            judged_here = 0
            stop_candidate = False                      # contradiction early-stop (5f-H)
            # Level 5i-K3: batch judge one (source, candidate) over all constraints in a single
            # cached call. Per-constraint verdicts + post-model hard rules are preserved, so
            # support materializes identically to the per-constraint path.
            if judge_on and getattr(self.judge, "batch_enabled", False) and cons:
                det_by, rel_by, obj_by = {}, {}, {}
                for c in cons:
                    st, q = recognize_constraint_support(
                        c, text, role_e, source_role, title, snippet, contaminated=contaminated)
                    det_by[c.constraint_id] = (st, q)
                    rel = len(getattr(c, "applies_to", []) or []) > 1
                    rel_by[c.constraint_id] = rel
                    obj_by[c.constraint_id] = ((not rel)
                                               or _object_anchored(c, slot, frame, frontier, text_l))
                results = self.judge.judge_batch(
                    candidate_text=text, candidate_id=None, aliases=list(a.candidate_aliases),
                    slot_id=slot.slot_id, slot_role=slot.slot_role,
                    slot_descriptor=(getattr(slot, "descriptor_text", "") or slot.slot_name),
                    constraints=cons, source_id=source_id, source_title=title,
                    source_url=source_url, source_domain=_host(source_url),
                    source_role=source_role, contaminated=contaminated, snippet=snippet,
                    det_by_cid=det_by, relational_by_cid=rel_by, object_anchored_by_cid=obj_by)
                for c in cons:
                    jd = results[c.constraint_id]
                    a.judgments.append(jd.to_dict())
                    for al in jd.candidate_aliases:
                        if al and al not in a.candidate_aliases:
                            a.candidate_aliases.append(al)
                    if stop_candidate:
                        if jd.judgment == "contradiction":
                            con.append(c.constraint_id)
                        continue
                    if jd.judgment == "full_support":
                        sup.append(c.constraint_id)
                    elif jd.judgment == "contradiction":
                        con.append(c.constraint_id)
                        if _is_blocking(c):             # block support after a blocking contra
                            stop_candidate = True
                    elif jd.judgment == "partial_support":
                        partial.append(c.constraint_id)
                    elif jd.judgment == "requires_read":
                        req_read.append(c.constraint_id)
                if is_known and not sup:
                    continue
                proposed.append(slot.slot_id)
                a.slot_support[slot.slot_id] = [sorted(sup), sorted(con)]
                if partial:
                    a.slot_partial[slot.slot_id] = sorted(partial)
                if req_read:
                    a.slot_requires_read[slot.slot_id] = sorted(req_read)
                supports.update(sup)
                contradicts.update(con)
                continue
            for c in cons:
                status, q = recognize_constraint_support(
                    c, text, role_e, source_role, title, snippet, contaminated=contaminated)
                # When the judge is enabled, it decides support fit for this triple; the
                # deterministic recognizer is its fallback + input (req: prefer the judge).
                if judge_on and not stop_candidate:
                    if judged_here >= judge_cap:
                        self.judge.max_calls_cap_hit += 1   # budget cap (5f-H): fall back to det
                        if status == "supports":
                            sup.append(c.constraint_id)
                        elif status == "contradicts":
                            con.append(c.constraint_id)
                        continue
                    judged_here += 1
                    relational = len(getattr(c, "applies_to", []) or []) > 1
                    obj_anchored = (not relational) or _object_anchored(c, slot, frame, frontier,
                                                                        text_l)
                    jd = self.judge.judge(
                        candidate_text=text, candidate_id=None, aliases=list(a.candidate_aliases),
                        slot_id=slot.slot_id, slot_role=slot.slot_role,
                        slot_descriptor=(getattr(slot, "descriptor_text", "") or slot.slot_name),
                        constraint=c, source_id=source_id, source_title=title,
                        source_url=source_url, source_domain=_host(source_url),
                        source_role=source_role, contaminated=contaminated, snippet=snippet,
                        det_status=status, det_quote=q, relational=relational,
                        object_anchored=obj_anchored)
                    a.judgments.append(jd.to_dict())
                    for al in jd.candidate_aliases:
                        if al and al not in a.candidate_aliases:
                            a.candidate_aliases.append(al)
                    if jd.judgment == "full_support":
                        sup.append(c.constraint_id)
                    elif jd.judgment == "contradiction":
                        con.append(c.constraint_id)
                        # stop judging this candidate for this result on a blocking contradiction.
                        if _is_blocking(c):
                            stop_candidate = True
                            self.judge.contradiction_early_stop += 1
                            self.judge.calls_saved_by_contradiction_stop += max(
                                0, len(cons) - cons.index(c) - 1)
                    elif jd.judgment == "partial_support":
                        partial.append(c.constraint_id)
                    elif jd.judgment == "requires_read":
                        req_read.append(c.constraint_id)
                elif judge_on and stop_candidate:
                    # deterministic-only after a blocking contradiction (no more judge calls).
                    if status == "contradicts":
                        con.append(c.constraint_id)
                else:
                    if status == "supports":
                        sup.append(c.constraint_id)
                    elif status == "contradicts":
                        con.append(c.constraint_id)
            if is_known and not sup:
                continue                # a known constant binds a slot only via support
            proposed.append(slot.slot_id)
            a.slot_support[slot.slot_id] = [sorted(sup), sorted(con)]
            if partial:
                a.slot_partial[slot.slot_id] = sorted(partial)
            if req_read:
                a.slot_requires_read[slot.slot_id] = sorted(req_read)
            supports.update(sup)
            contradicts.update(con)
        if not proposed:
            a.rejection_reason = ("unsupported_by_selected_constraint" if is_known
                                  else "no_slot_compatible_evidence")
            return a
        a.proposed_slot_ids = proposed
        a.supports_constraint_ids = sorted(supports)
        a.contradicts_constraint_ids = sorted(contradicts)
        a.evidence_quote_or_span = _quote_for(
            [norm] + [t.lower() for cid in supports
                      for t in (next((c.normalized_terms for c in frame.constraints
                                      if c.constraint_id == cid), []))],
            title, snippet)
        a.confidence = (0.4 if source_role in WEAK_ROLES else 0.85) + 0.1 * bool(supports)
        return a

    def _note_prejudge_reject(self, reason: str, sel_cons) -> None:
        """Record a pre-judge triage rejection (5i-K1) and the judge calls it saved."""
        self.prejudge_rejected_count += 1
        self.prejudge_rejected_by_reason[reason] += 1
        if self.judge is not None and getattr(self.judge, "enabled", False):
            saved = max(1, len(sel_cons or []))
            self.judge_calls_saved_by_prejudge_triage += saved

    @staticmethod
    def _location_compat(explicit_locs: set, text: str, title: str, snippet: str) -> str:
        """Compatibility of a candidate's source location against the explicit facet (5i-K2):
        ``supported`` (explicit location present) / ``mismatch`` (only a different location,
        no branch cue) / ``ambiguous`` (no location named) / ``ok``."""
        body = f"{title} {snippet}"
        present = _locations_in_context(body)
        if explicit_locs & present:
            return "supported"
        other = present - explicit_locs
        low = body.lower()
        has_branch = any(cue in low for cue in _LOCATION_BRANCH_CUES)
        if other and not has_branch:
            return "mismatch"
        if not present:
            return "ambiguous"
        return "ok"

    def _assert_constraint(self, con, source_role, text_l, title, snippet) -> ConstraintAssertion:
        cid = con.constraint_id
        if source_role in HARD_NOISE_ROLES:
            return ConstraintAssertion(cid, "irrelevant", reason=f"noise_source:{source_role}")
        terms = [t for t in con.normalized_terms if len(t) >= 4]
        if not terms:
            return ConstraintAssertion(cid, "insufficient", reason="no_anchorable_terms")
        if _supports(con, text_l):
            self_quote = _quote_for([t.lower() for t in terms], title, snippet)
            return ConstraintAssertion(cid, "supports", evidence_quote_or_span=self_quote,
                                       confidence=0.8, reason="constraint_terms_present")
        hits = sum(1 for t in terms if t.lower() in text_l)
        if hits == 0:
            return ConstraintAssertion(cid, "irrelevant", reason="no_constraint_terms")
        return ConstraintAssertion(cid, "insufficient", confidence=0.3,
                                   reason="partial_constraint_terms")

    def _tally(self, interp: EvidenceInterpretation) -> None:
        self.interpretation_count += 1
        self.source_role_counts[interp.source_role] += 1
        for a in interp.candidate_assertions:
            self.candidate_assertion_count += 1
            if a.accepted:
                self.accepted_candidate_count += 1
            else:
                self.rejected_candidate_count += 1
                self.rejection_counts[a.rejection_reason] += 1
        for c in interp.constraint_assertions:
            if c.status == "supports":
                self.constraint_support_count += 1
            elif c.status == "contradicts":
                self.constraint_contradiction_count += 1

    # ---------- optional cached/replayable LLM source-role re-classification ----------
    def _llm_source_role(self, deterministic_role: str, frame, title, snippet, url) -> str:
        if self.cache is None and self.model_fn is None:
            return deterministic_role
        from regimes_probe.agent.llm_task_frame import _sha
        ev_hash = _sha(f"{title}\n{snippet}\n{url}")
        frame_hash = _sha(getattr(frame, "item_id", "") or "")
        key = _sha(f"evidence_interpreter|{self.model}|{frame_hash}|{ev_hash}")
        raw = self.cache.get(key) if self.cache is not None else None
        if raw is not None:
            self.cache_hits += 1
            return self._parse_role(raw) or deterministic_role
        if self.replay_only or self.model_fn is None:
            return deterministic_role          # miss in replay/dry-run: never spend
        prompt = ("Classify ONLY the source role of this single web result; do NOT answer "
                  "the question. Roles: " + ", ".join(SOURCE_ROLES) +
                  f'.\nTITLE: {title[:200]}\nSNIPPET: {snippet[:400]}\nURL: {url[:160]}\n'
                  'Reply JSON: {"source_role": "<role>"}')
        try:
            self.model_calls += 1
            raw = self.model_fn(prompt)
        except Exception:
            return deterministic_role
        if self.cache is not None:
            self.cache.put(key, raw or "")
        return self._parse_role(raw) or deterministic_role

    @staticmethod
    def _parse_role(raw: str) -> str:
        try:
            r = json.loads(raw).get("source_role", "")
            return r if r in SOURCE_ROLES else ""
        except Exception:
            return ""
