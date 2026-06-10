"""Level 5i-I: lightweight, generic source-subject extraction.

For every evidence result/read we extract *what (real-world) entity the source is about* —
its `SourceSubject` — separately from the page's chrome/title. This lets candidate promotion
ask a sharper question than "does this string overlap a constraint?": **is this the
predicate-grounded subject of the source, or just its title / a generic topic label / UI
chrome?** A source title may help *identify* the subject only when the body/snippet predicate
text also grounds it.

Deterministic by default; an optional LLM refinement must be cached/replayable, answer-free,
and may never see gold or emit a final answer (the same contract as the source-role hook).

Examples are diagnostic only (no benchmark-specific logic keyed to them):
- "Kenyan writer Ken X is a novelist who…" → Ken X, person, article_subject, predicate-grounded.
- "Founder Definition & Meaning" → generic_topic / source_chrome, NOT a founder candidate.
- "The Founder (2016) — cast" → a title/work/database page; not a real-world founder.
- "Pecos Trail Inn — About Us … opened in 1955" → organization subject, grounded by the body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.agent.candidate_frontier import _hash, _noise_kind, _prev
from regimes_probe.agent.clue_resolution import (
    _entities_in_field, _is_generic_entity, _norm, classify_entity_role)

SUBJECT_TYPES = ("person", "organization", "location", "title_or_work", "date_or_time",
                 "concept", "unknown")
SUBJECT_ROLES = ("article_subject", "profile_subject", "page_owner", "mentioned_entity",
                 "generic_topic", "source_chrome", "unknown")

#: relation/predicate cue words that, near a subject name, indicate the source asserts a
#: predicate ABOUT that subject (generic — not benchmark phrases).
_PREDICATE_CUES = frozenset({
    "is", "was", "are", "were", "founded", "opened", "established", "born", "studied",
    "worked", "served", "directed", "wrote", "designed", "published", "located", "named",
    "known", "called", "based", "headquartered", "graduated", "died", "created", "led",
    "produced", "developed", "built", "discovered", "invented", "owns", "owned", "runs"})
_DEFINITION_TITLE = re.compile(r"\b(definition|meaning|synonyms?|antonyms?|pronunciation"
                               r"|wiktionary|dictionary|thesaurus|defined)\b", re.I)
#: organization SUFFIX words that type an entity by its own name (a local predicate).
_ORG_SUFFIX = frozenset({
    "inn", "cafe", "café", "hotel", "motel", "lodge", "resort", "restaurant", "diner",
    "bistro", "tavern", "bar", "pub", "grill", "cantina", "museum", "gallery", "theatre",
    "theater", "company", "corporation", "inc", "corp", "ltd", "llc", "foundation",
    "institute", "university", "college", "school", "hospital", "society", "association",
    "club", "academy", "library", "observatory"})
#: a parenthetical year in a title typically marks a creative WORK ("The Founder (2016)").
_WORK_YEAR_TITLE = re.compile(r".+\((1[5-9]\d{2}|20\d{2})\)\s*$")

#: source roles that are STRUCTURALLY about their title subject (profile owner / record).
_STRUCTURAL_SUBJECT_ROLES = frozenset({
    "professional_profile", "official_page", "database_record", "primary_source"})
_ROLE_TO_SUBJECT_TYPE = {
    "person": "person", "organization": "organization", "place": "location",
    "location": "location", "title_or_work": "title_or_work", "work": "title_or_work",
    "publication_or_source": "title_or_work", "date_or_time": "date_or_time",
    "date": "date_or_time", "number": "concept", "concept": "concept"}


@dataclass
class SourceSubject:
    source_subject_id: str
    source_url: str = ""
    source_title: str = ""
    source_role: str = "unknown"
    subject_name: str = ""
    subject_type: str = "unknown"
    subject_role: str = "unknown"
    confidence: float = 0.0
    evidence_quote: str = ""
    extraction_method: str = "deterministic"     # deterministic | llm_optional | none
    is_predicate_grounded: bool = False
    is_chrome_or_source_title_only: bool = False
    source_subject_aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"source_subject_id": self.source_subject_id,
                "source_url": self.source_url[:160], "source_title": _prev(self.source_title, 100),
                "source_role": self.source_role, "subject_name": _prev(self.subject_name, 80),
                "subject_type": self.subject_type, "subject_role": self.subject_role,
                "confidence": round(self.confidence, 3),
                "evidence_quote": _prev(self.evidence_quote, 160),
                "extraction_method": self.extraction_method,
                "is_predicate_grounded": self.is_predicate_grounded,
                "is_chrome_or_source_title_only": self.is_chrome_or_source_title_only,
                "source_subject_aliases": list(self.source_subject_aliases)[:6]}


def _predicate_grounds(name: str, text: str, constraint_terms) -> tuple[bool, str]:
    """Is there predicate text in ``text`` asserting something ABOUT ``name``? Generic: the
    subject name appears and, within a short window, a predicate cue OR a constraint term."""
    low = (text or "").lower()
    nm = (name or "").lower()
    if not nm or nm not in low:
        return False, ""
    i = low.find(nm)
    window = low[max(0, i - 40): i + len(nm) + 90]
    cterms = {str(t).lower() for t in (constraint_terms or []) if len(str(t)) >= 4}
    win_tokens = set(re.findall(r"[a-z']+", window))
    if win_tokens & _PREDICATE_CUES or (cterms and any(t in window for t in cterms)):
        start = max(0, i - 30)
        return True, _prev((text or "")[start:i + len(name) + 70], 160)
    return False, ""


def extract_source_subject(*, title: str, snippet: str, url: str, source_role: str,
                           source_id: str = "ss", constraint_terms=()) -> SourceSubject:
    """Deterministically extract the source's subject (5i-I). The page is usually *about* its
    title's lead entity; we only call that a real subject when body/snippet predicate text
    grounds it, otherwise it is chrome / a source-title-only / a generic topic label."""
    title = title or ""
    snippet = snippet or ""
    ss = SourceSubject(source_subject_id=source_id, source_url=url or "", source_title=title,
                       source_role=source_role or "unknown")
    title_l = title.lower()
    # chrome / definition / source-title-only pages have no real-world subject.
    if source_role in ("ui_or_navigation_noise", "benchmark_contaminated"):
        ss.subject_role, ss.is_chrome_or_source_title_only = "source_chrome", True
        return ss
    if source_role == "generic_definition_page" or _DEFINITION_TITLE.search(title):
        ss.subject_role, ss.is_chrome_or_source_title_only = "generic_topic", True
        return ss
    # candidate subject = the first non-generic proper-noun entity, preferring the TITLE.
    title_ents = [e for e in _entities_in_field(title) if len(e) >= 3 and not _is_generic_entity(e)]
    snip_ents = [e for e in _entities_in_field(snippet) if len(e) >= 3 and not _is_generic_entity(e)]
    from_title = bool(title_ents)
    name = (title_ents or snip_ents or [""])[0]
    if not name or _noise_kind(name):
        ss.subject_role = "source_chrome" if not name else "generic_topic"
        ss.is_chrome_or_source_title_only = True
        return ss
    ss.subject_name = name
    role = classify_entity_role(name)
    ss.subject_type = _ROLE_TO_SUBJECT_TYPE.get(role, "unknown")
    # an org-SUFFIX name ("… Inn"/"… Cafe"/"… Museum") types the subject as an organization
    # by its own name (a local predicate), even if the role classifier was unsure.
    last = (re.findall(r"[A-Za-z]+", name) or [""])[-1].lower()
    if last in _ORG_SUFFIX:
        ss.subject_type = "organization"
    grounded, quote = _predicate_grounds(name, f"{title}. {snippet}", constraint_terms)
    # a profile / official / database / primary source is STRUCTURALLY about its title
    # subject (the page owner / record), so the title entity is grounded by the source role
    # itself even when the snippet carries no explicit predicate.
    if from_title and source_role in _STRUCTURAL_SUBJECT_ROLES:
        grounded = True
        quote = quote or _prev(f"{title}. {snippet}", 160)
    ss.is_predicate_grounded = grounded
    ss.evidence_quote = quote
    # a creative-work page ("Title (YYYY)") is a title_or_work subject, not a real entity.
    if _WORK_YEAR_TITLE.match(title.strip()):
        ss.subject_type = "title_or_work"
        ss.subject_role = "mentioned_entity"
    # subject_role: grounded title entity = the article/profile subject; ungrounded title-only
    # entity = source-title-only (not promotable); a grounded snippet entity is "mentioned".
    name_only_in_title = name.lower() in title_l and name.lower() not in (snippet or "").lower()
    if grounded:
        if source_role == "professional_profile":
            ss.subject_role = "profile_subject"
        elif source_role == "official_page":
            ss.subject_role = "page_owner"
        elif ss.subject_role == "unknown":
            ss.subject_role = "article_subject" if from_title else "mentioned_entity"
        ss.confidence = 0.85 if from_title else 0.6
    else:
        # only a NON-named title subject (a generic-topic / source-title word, not a real
        # proper-noun entity) is "source-title-only chrome"; a real named entity the snippet
        # simply doesn't repeat stays a weak mentioned-entity candidate, not chrome.
        is_named = any(w[:1].isupper() for w in re.findall(r"[A-Za-z][A-Za-z'&.-]*", name))
        if name_only_in_title and not is_named:
            ss.subject_role = "source_chrome" if ss.subject_role == "unknown" else ss.subject_role
            ss.is_chrome_or_source_title_only = True
        elif ss.subject_role == "unknown":
            ss.subject_role = "mentioned_entity"
        ss.confidence = 0.3
    return ss


def subject_supports_candidate(ss: SourceSubject, candidate_text: str, slot_role: str) -> bool:
    """Whether the source subject may PROMOTE this candidate (5i-I): the candidate must BE the
    grounded subject (name/alias match), be role-compatible, and be predicate-grounded — never
    a chrome / source-title-only / generic-topic subject."""
    from regimes_probe.agent.candidate_frontier import _role_compatible
    if ss is None or ss.is_chrome_or_source_title_only or not ss.is_predicate_grounded:
        return False
    if ss.subject_role in ("source_chrome", "generic_topic"):
        return False
    cand = (candidate_text or "").strip().lower()
    names = {ss.subject_name.lower(), *[a.lower() for a in ss.source_subject_aliases]}
    if cand not in names and not any(cand and (cand in n or n in cand) for n in names if n):
        return False
    return _role_compatible(ss.subject_type, slot_role)


def is_promotable_subject(ss: Optional[SourceSubject]) -> bool:
    return bool(ss and ss.is_predicate_grounded and not ss.is_chrome_or_source_title_only
                and ss.subject_role not in ("source_chrome", "generic_topic"))


def _norm_name(name: str) -> str:
    return _hash(_norm(name or ""))
