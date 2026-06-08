"""Iterative clue resolution (deterministic v0): extract candidate entities from
search results and compose follow-up queries.

BrowseComp answers often need *iterative* resolution: search a clue, read the
returned titles/snippets to identify an intermediate entity, then search that
entity together with the next clue. v1/v2 decomposition only issued parallel clue
queries and hoped the answer appeared in a snippet (seam: ``exact_answer_missing``).

This module is gold-free and answer-free: it reads only result titles/snippets/
URLs + the question's own clue terms. It never sees or stores the gold answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from regimes_probe.policy.query_decomposition import (
    _cap, _tokens, _is_rare, _GENERIC_ADJ, _BOILER, _STOPWORDS, _GENERIC_NOUN,
    _COMMON_WORDS, MAX_QUERY_CHARS, MAX_QUERY_TOKENS)

#: Tight entity regex (no run-on across 'of/the'): 1-4 consecutive capitalized words.
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}\b")
#: Domain-label noise that is never a real intermediate entity.
_DOMAIN_NOISE = {"example", "news", "blog", "site", "web", "page", "home", "index",
                 "www", "search", "results", "wiki", "info", "online", "daily"}

#: Hubs / platforms that are never the intermediate ENTITY we want to chase.
GENERIC_ENTITIES = {
    "facebook", "wikipedia", "wikimedia", "wikidata", "youtube", "reddit",
    "twitter", "instagram", "linkedin", "amazon", "google", "pinterest",
    "tiktok", "quora", "imdb", "github", "medium", "tumblr", "yelp", "bing",
    "yahoo", "msn", "fandom", "wikihow", "britannica", "goodreads", "spotify",
    "apple", "microsoft", "netflix", "ebay", "tripadvisor", "wordpress",
    "blogspot", "substack", "archive", "scribd", "slideshare", "academia",
}
_FOLLOWUP_ARMS = ("candidate_entity_followup", "answer_shape_followup")


def _domain_label(url: str) -> str:
    host = (urlparse(url or "").hostname or "").lower()
    parts = [p for p in host.split(".") if p not in ("www", "com", "org", "net",
             "edu", "gov", "io", "co", "uk", "us")]
    return parts[-1] if parts else ""


def _is_generic_entity(text: str) -> bool:
    tl = text.lower().strip()
    return any(g == w or g in tl.split() for g in GENERIC_ENTITIES for w in [tl])


@dataclass
class CandidateEntity:
    text: str
    frequency: int = 0
    source_authority: float = 0.0
    proximity: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "frequency": self.frequency,
                "source_authority": round(self.source_authority, 3),
                "proximity": round(self.proximity, 3), "score": round(self.score, 3)}


def _entities_in_field(text: str) -> list[str]:
    """Multiword (2-4 caps) entities preferred; standalone caps only if NOT part of
    a multiword entity in this field (so 'Edrin Vael' wins over bare 'Edrin')."""
    out: list[str] = []
    covered: set[str] = set()
    multi: list[str] = []
    for m in _ENTITY_RE.findall(text):
        words = m.split()
        if len(words) >= 2 and not all(w.lower() in _GENERIC_ADJ or w.lower() in _BOILER
                                       for w in words):
            multi.append(m.strip())
            covered.update(w.lower() for w in words)
    out.extend(multi)
    for w in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text):
        wl = w.lower()
        if (wl in covered or wl in _GENERIC_ADJ or wl in _BOILER or wl in _STOPWORDS
                or wl in _GENERIC_NOUN):
            continue
        if _is_rare(w) or len(w) >= 6:
            out.append(w)
    return _dedupe_lower(out)


def _dedupe_lower(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for s in seq:
        k = s.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def extract_candidate_entities(observations, clue_terms: list[str], *,
                               top_k: int = 5) -> list[CandidateEntity]:
    """Rank intermediate entities seen across results (deterministic, gold-free).

    ``observations`` are :class:`EvidenceObservation`; ``clue_terms`` are
    lowercased tokens from the question's clue spans (for proximity). An entity is
    counted at most once per observation (title+snippet), so frequency reflects how
    many distinct results mention it.
    """
    clue_set = {t.lower() for t in clue_terms if len(t) >= 3}
    agg: dict[str, CandidateEntity] = {}
    for o in observations:
        if getattr(o, "failed", False):
            continue
        title = getattr(o, "title", "") or ""
        snippet = getattr(o, "snippet", "") or ""
        ents = _dedupe_lower(_entities_in_field(title) + _entities_in_field(snippet))
        dl = _domain_label(getattr(o, "url", ""))
        if (dl and dl not in GENERIC_ENTITIES and dl not in _DOMAIN_NOISE
                and dl not in _COMMON_WORDS and len(dl) >= 4):
            ents.append(dl.capitalize())
        ents = _dedupe_lower(ents)
        near = bool(clue_set & {t.lower() for t in _tokens(f"{title} {snippet}")})
        for e in ents:
            if _is_generic_entity(e) or len(e) < 3:
                continue
            ce = agg.setdefault(e.lower(), CandidateEntity(text=e))
            ce.frequency += 1
            ce.source_authority = max(ce.source_authority, float(getattr(o, "source_authority", 0.0)))
            if near:
                ce.proximity = 1.0
    for ce in agg.values():
        multi = 0.6 if " " in ce.text else 0.0
        ce.score = ce.frequency * 1.0 + ce.source_authority * 1.0 + ce.proximity * 0.5 + multi
    return sorted(agg.values(), key=lambda c: (-c.score, c.text))[:top_k]


@dataclass
class FollowupQuery:
    query: str
    query_arm: str
    selected_candidate: str
    reason: str
    clue_used: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "query_arm": self.query_arm,
                "selected_candidate": self.selected_candidate, "reason": self.reason,
                "clue_used": self.clue_used}


def compose_followup_query(candidate: CandidateEntity, *, clue_spans: list[str],
                           used_clue_spans: set[str], answer_shape: list[str],
                           max_chars: int = MAX_QUERY_CHARS,
                           max_tokens: int = MAX_QUERY_TOKENS) -> Optional[FollowupQuery]:
    """Combine the top candidate entity with the next unused clue span (or an
    answer-shape hint). Returns ``None`` if there is nothing useful to add."""
    ent = candidate.text
    next_clue = next((s for s in clue_spans if s.lower() not in used_clue_spans
                      and ent.lower() not in s.lower()), None)
    if next_clue:
        q = _cap(f'"{ent}" {next_clue}', max_chars=max_chars, max_tokens=max_tokens)
        return FollowupQuery(query=q, query_arm="candidate_entity_followup",
                             selected_candidate=ent, clue_used=next_clue,
                             reason=f"top candidate '{ent}' (score={candidate.score:.2f}) + "
                                    f"next clue span '{next_clue}'")
    if answer_shape:
        q = _cap(f'"{ent}" {answer_shape[0]}', max_chars=max_chars, max_tokens=max_tokens)
        return FollowupQuery(query=q, query_arm="answer_shape_followup",
                             selected_candidate=ent, clue_used=None,
                             reason=f"top candidate '{ent}' + answer-shape '{answer_shape[0]}'")
    # last resort: re-anchor on the entity alone (still better than re-issuing the clue)
    q = _cap(f'"{ent}"', max_chars=max_chars, max_tokens=max_tokens)
    return FollowupQuery(query=q, query_arm="candidate_entity_followup",
                         selected_candidate=ent, clue_used=None,
                         reason=f"top candidate '{ent}' (no remaining clue/shape)")
