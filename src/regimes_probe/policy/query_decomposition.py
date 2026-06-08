"""Level 2 query decomposition: turn one long clue-dense question into several
targeted candidate queries (deterministic v0; no model, no network).

BrowseComp questions pack many clues into one long prompt. Searching the whole
prompt verbatim tends to surface spam / benchmark-mirroring pages rather than the
evidence page that actually carries the answer. This module extracts the clues
(quoted phrases, proper-noun entities, years/date ranges, rare terms, title /
institution phrases, source-type hints) and emits 3-6 SHORTER candidate queries,
one per arm. The learned query policy / bandit then chooses among them.

Everything here is a pure function of the question text, so it is replayable and
needs no cache. An optional LLM decomposition can be layered later behind
``--enable-query-decomposition`` (it must go through the answerer cache and carry
a prompt-version hash); v0 is heuristic and always available.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

#: Canonical decomposition query arms (each a generic transform; no topical info).
DECOMPOSITION_ARMS: tuple[str, ...] = (
    "full_question_compressed",
    "exact_phrase_clue",
    "entity_clue",
    "relation_clue",
    "rare_terms_clue",
    "date_range_clue",
    "negative_noise_removed",
    "quoted_anchor_terms",
    "source_type_query",
)

#: Default caps so we never send a giant prompt to a search provider.
MAX_QUERY_CHARS = 180
MAX_QUERY_TOKENS = 18

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_QUOTED = re.compile(r"[\"“”']([^\"“”']{3,80})[\"“”']")
_PROPER = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+(?:of|the|and|de|von|van|al)\s+|\s+)?"
                     r"(?:[A-Z][a-zA-Z]+)*\b")
_PROPER_SIMPLE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*\b")
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_YEAR_RANGE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\s*(?:-|–|to|and|through)\s*(1[0-9]{3}|20[0-9]{2})\b")

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "what", "which", "who", "whom", "whose", "how", "when",
    "where", "did", "do", "does", "with", "by", "as", "at", "that", "this",
    "from", "into", "about", "between", "during", "after", "before", "than",
    "then", "there", "their", "its", "it", "be", "been", "being", "has", "have",
    "had", "but", "not", "no", "also", "such", "some", "any", "one", "two",
}
#: Meta-instruction / noise phrases that are not part of the actual clue.
_NOISE = {
    "im", "i", "m", "looking", "trying", "figure", "find", "identify", "please",
    "help", "tell", "me", "need", "want", "know", "question", "answer", "guess",
    "name", "give", "provide", "could", "would", "can", "you", "your", "we",
    "out", "exact", "exactly", "specific", "specifically", "recall", "remember",
}
#: Source-type hints → a source term appended to a targeted query.
_SOURCE_HINTS = {
    "paper": "paper", "study": "study", "patent": "patent", "filing": "filing",
    "report": "report", "press release": "press release", "official": "official",
    "obituary": "obituary", "wikipedia": "wikipedia", "article": "article",
    "journal": "journal", "thesis": "thesis", "dissertation": "dissertation",
    "interview": "interview", "court": "court ruling", "ruling": "court ruling",
    "census": "census", "statistic": "statistics", "database": "database",
}
#: Title / occupation / institution suffixes that anchor a good query.
_TITLE_WORDS = ("president", "ceo", "chief", "director", "founder", "minister",
                "professor", "chairman", "chairwoman", "secretary", "governor",
                "mayor", "author", "winner", "champion", "owner", "inventor")
_INSTITUTION_SUFFIX = ("university", "college", "institute", "laboratory", "company",
                       "corporation", "foundation", "association", "committee",
                       "department", "ministry", "academy", "society", "agency")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def _cap(query: str, *, max_chars: int = MAX_QUERY_CHARS,
         max_tokens: int = MAX_QUERY_TOKENS) -> str:
    """Trim a query to the token/char caps without splitting mid-token."""
    toks = query.split()
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    out = " ".join(toks).strip()
    return out[:max_chars].strip()


@dataclass
class Clues:
    quoted: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    years: list[str] = field(default_factory=list)
    date_ranges: list[str] = field(default_factory=list)
    rare_terms: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)
    content_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) for k, v in self.__dict__.items()}


#: Capitalized words that begin sentences/questions but are not real entities.
_COMMON_CAP = {
    "what", "who", "when", "where", "which", "why", "how", "the", "a", "an", "i",
    "in", "on", "at", "is", "are", "was", "were", "did", "does", "do", "this",
    "that", "he", "she", "it", "they", "we", "you", "my", "their", "his", "her",
    "its", "of", "to", "and", "or", "but", "for", "with", "there", "then",
}


def _dedupe(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for s in seq:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out


def _clean_entity(e: str) -> str:
    """Strip leading sentence/question words so 'What is Acme' -> 'Acme'."""
    toks = e.split()
    while toks and toks[0].lower() in _COMMON_CAP:
        toks = toks[1:]
    return " ".join(toks).strip()


def extract_clues(question: str) -> Clues:
    """Deterministically extract clue spans from a question."""
    q = question
    ql = question.lower()
    quoted = _dedupe(_QUOTED.findall(q))
    entities = _dedupe([e for m in _PROPER_SIMPLE.findall(q)
                        if len(e := _clean_entity(m)) >= 3])
    ranges = _dedupe([f"{a}-{b}" for a, b in _YEAR_RANGE.findall(q)])
    years = _dedupe(_YEAR.findall(q))
    # rare/unusual terms: long, low-frequency content tokens (deterministic order).
    content = [t for t in _tokens(q) if t.lower() not in _STOPWORDS and t.lower() not in _NOISE]
    lowered = [t.lower() for t in content if not t[0].isupper()]
    rare = sorted({t for t in lowered if len(t) >= 8}, key=lambda t: (-len(t), t))[:6]
    titles = _dedupe([w for w in _TITLE_WORDS if w in ql])
    institutions = _dedupe([e for e in entities
                            if any(s in e.lower() for s in _INSTITUTION_SUFFIX)])
    source_hints = _dedupe([v for k, v in _SOURCE_HINTS.items() if k in ql])
    content_terms = [t for t in lowered if len(t) >= 4][:12]
    return Clues(quoted=quoted, entities=entities, years=years, date_ranges=ranges,
                 rare_terms=rare, titles=titles, institutions=institutions,
                 source_hints=source_hints, content_terms=content_terms)


@dataclass
class CandidateQuery:
    arm: str
    query: str
    clue_ids: list[str] = field(default_factory=list)

    @property
    def query_text_hash(self) -> str:
        return hashlib.sha256(self.query.lower().encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "query": self.query, "clue_ids": list(self.clue_ids),
                "query_text_hash": self.query_text_hash}


def _compress(question: str) -> str:
    toks = [t for t in _tokens(question)
            if t.lower() not in _STOPWORDS]
    return _cap(" ".join(toks))


def _noise_removed(question: str) -> str:
    toks = [t for t in _tokens(question)
            if t.lower() not in _STOPWORDS and t.lower() not in _NOISE]
    return _cap(" ".join(toks))


def decompose_queries(question: str, *, max_queries: int = 6,
                      max_chars: int = MAX_QUERY_CHARS,
                      max_tokens: int = MAX_QUERY_TOKENS) -> list[CandidateQuery]:
    """Generate 3-6 targeted candidate queries for a clue-dense question.

    Deterministic. Only arms that produce a non-trivial query are returned, and
    the long ``full_question_compressed`` arm is always LAST so it is never the
    cold-start default when targeted clue arms exist.
    """
    c = extract_clues(question)
    ents = c.entities[:4]
    out: list[CandidateQuery] = []

    def add(arm: str, query: str, clue_ids: list[str]) -> None:
        q = _cap(query, max_chars=max_chars, max_tokens=max_tokens)
        if q and len(q.split()) >= 1:
            out.append(CandidateQuery(arm=arm, query=q, clue_ids=clue_ids))

    # Targeted arms in priority order (the cap trims from the END of this list,
    # so the highest-value clue forms survive). Alphabetical tie-break also keeps
    # full_question_compressed out of the cold-start default slot.
    if c.quoted:
        add("exact_phrase_clue", " ".join(f'"{p}"' for p in c.quoted[:3]),
            [f"quoted[{i}]" for i in range(min(3, len(c.quoted)))])

    if ents:
        add("entity_clue", " ".join(ents), [f"entity[{i}]" for i in range(len(ents))])

    if c.date_ranges or c.years:
        dparts = (c.date_ranges[:1] or c.years[:2])
        add("date_range_clue", " ".join(ents[:2] + dparts),
            [f"entity[{i}]" for i in range(min(2, len(ents)))]
            + [f"date[{i}]" for i in range(len(dparts))])

    if c.rare_terms:
        add("rare_terms_clue", " ".join(c.rare_terms + ents[:1]),
            [f"rare[{i}]" for i in range(len(c.rare_terms))])

    if c.quoted:
        anchors = _dedupe(c.quoted[:2] + ents[:2])
        add("quoted_anchor_terms", " ".join(f'"{a}"' for a in anchors),
            [f"quoted[{i}]" for i in range(min(2, len(c.quoted)))]
            + [f"entity[{i}]" for i in range(min(2, len(ents)))])
    elif ents:
        add("quoted_anchor_terms", " ".join(f'"{e}"' for e in ents[:3]),
            [f"entity[{i}]" for i in range(min(3, len(ents)))])

    if ents:
        rel_terms = (c.titles[:2] + [t for t in c.content_terms if t not in
                     {e.lower() for e in ents}][:4])
        if rel_terms:
            add("relation_clue", " ".join(ents[:2] + rel_terms),
                [f"entity[{i}]" for i in range(min(2, len(ents)))]
                + [f"title[{i}]" for i in range(len(c.titles[:2]))])

    if c.source_hints and ents:
        add("source_type_query", " ".join(ents[:2] + c.source_hints[:2]),
            [f"entity[{i}]" for i in range(min(2, len(ents)))]
            + [f"source[{i}]" for i in range(len(c.source_hints[:2]))])

    # Always-available fallbacks (LAST, so never the alphabetical/cold default).
    add("negative_noise_removed", _noise_removed(question), ["noise_removed"])
    add("full_question_compressed", _compress(question), ["full_question"])

    # De-dupe by arm (keep first occurrence).
    seen_arms: set[str] = set()
    deduped: list[CandidateQuery] = []
    for cq in out:
        if cq.arm in seen_arms:
            continue
        seen_arms.add(cq.arm)
        deduped.append(cq)

    # Keep the targeted arms first, then ALWAYS retain the fallbacks last (so the
    # long full_question_compressed arm stays selectable but is never the cold
    # default / first arm). Trim the middle if over the cap.
    fallback_arms = ("negative_noise_removed", "full_question_compressed")
    targeted = [cq for cq in deduped if cq.arm not in fallback_arms]
    fallback = [cq for cq in deduped if cq.arm in fallback_arms]
    keep_targeted = max(1, max_queries - len(fallback))
    return (targeted[:keep_targeted] + fallback)[:max_queries]
