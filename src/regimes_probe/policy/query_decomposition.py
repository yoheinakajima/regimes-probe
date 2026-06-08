"""Level 2 query decomposition (v1): clue-SPAN extraction + quality scoring.

BrowseComp questions pack many clues into one long prompt. v0 searched the whole
prompt (fixed) but then extracted single capitalized tokens, producing weak/broad
queries like ``African One`` / ``Mexican`` / ``Name December 2023``. v1 instead
extracts meaningful **clue spans** (quoted phrases, multiword entities, noun
phrases around distinctive predicates, date+noun phrases, rare terms), composes
queries clause-by-clause, scores each candidate for expected search quality, and
DROPS low-quality candidates (single generic tokens, pure-date, all-boilerplate).

Everything here is a pure function of the question text (deterministic, replayable,
no model, no network). An optional LLM decomposer can be layered later behind the
same ``--enable-query-decomposition`` flag (through the answerer cache, with a
prompt-version hash); v1 is heuristic and always available.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

#: Canonical decomposition query arms (each a generic transform; no topical info).
DECOMPOSITION_ARMS: tuple[str, ...] = (
    "full_question_compressed",
    "exact_phrase_clue",
    "entity_clue",
    "relation_clue",
    "rare_terms_clue",
    "date_range_clue",
    "location_constraint_clue",
    "negative_noise_removed",
    "quoted_anchor_terms",
    "source_type_query",
)

#: Default caps so we never send a giant prompt to a search provider.
MAX_QUERY_CHARS = 180
MAX_QUERY_TOKENS = 18
#: A candidate must clear this expected-search-quality score to survive.
QUALITY_THRESHOLD = 1.5
#: Arm priority for the under-cap selection: keep the concise/distinctive arms
#: first rather than whichever bag-of-words query scores highest.
_ARM_PRIORITY = {
    "exact_phrase_clue": 0, "entity_clue": 1, "date_range_clue": 2,
    "location_constraint_clue": 3, "relation_clue": 4, "rare_terms_clue": 5,
    "source_type_query": 6, "quoted_anchor_terms": 7,
}
#: Min meaningful tokens unless the query is a quoted phrase / named entity.
MIN_MEANINGFUL_TOKENS = 2

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_QUOTED = re.compile(r"[\"“”']([^\"“”']{3,80})[\"“”']")
_CAP_RUN = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+(?:of|the|and|de|von|van|al)\s+[A-Z][a-zA-Z]+"
                      r"|\s+[A-Z][a-zA-Z0-9]+)*\b")
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_YEAR_RANGE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\s*(?:-|–|to|and|through)\s*(1[0-9]{3}|20[0-9]{2})\b")
#: Clause boundaries: punctuation + subordinating/coordinating connectors.
_CLAUSE_SPLIT = re.compile(
    r"[.;?!]|,|\b(?:who|whom|whose|which|that|where|when|while|because|"
    r"although|though|and|but|so that|in which|by which)\b", re.IGNORECASE)

#: Function words that BREAK a span (a phrase never crosses these).
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "what", "which", "who", "whom", "whose", "how", "when",
    "where", "did", "do", "does", "with", "by", "as", "at", "that", "this",
    "from", "into", "about", "between", "during", "before", "after", "than",
    "then", "there", "their", "its", "it", "be", "been", "being", "has", "have",
    "had", "but", "not", "no", "also", "such", "any", "i", "we", "you", "he",
    "she", "they", "my", "our", "your", "his", "her", "whether", "if", "while",
    "later", "whose", "near", "every", "some", "most", "more", "very", "out",
    "up", "down", "over", "under", "per", "via", "within", "without",
}
#: Placeholder / boilerplate tokens — never a clue on their own; trimmed off ends.
_BOILER = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "fourth", "fifth", "last", "next", "early", "late",
    "name", "named", "called", "various", "certain", "particular", "specific",
    "specifically", "known", "famous", "popular", "decade", "era", "period",
    "looking", "trying", "identify", "find", "recall", "remember", "guess",
    "individual", "person", "thing", "something", "someone", "exact", "exactly",
    "answer", "question", "please", "help", "tell", "give", "provide", "need",
    "want", "know", "figure", "later", "former", "around", "approximately",
}
#: Demonyms / generic adjectives: allowed INSIDE a phrase, never a standalone clue
#: and never count as the "noun-like" head of a phrase.
_GENERIC_ADJ = {
    "african", "american", "european", "asian", "western", "eastern", "northern",
    "southern", "british", "english", "french", "german", "indian", "chinese",
    "japanese", "korean", "russian", "italian", "spanish", "mexican", "canadian",
    "australian", "brazilian", "egyptian", "greek", "roman", "dutch", "swedish",
    "national", "international", "regional", "local", "global", "central",
    "modern", "ancient", "classic", "classical", "general", "major", "minor",
    "small", "large", "big", "old", "new", "young", "great", "good", "bad",
}
#: Generic nouns: a real noun (so a phrase headed by one is fine) but not
#: distinctive — penalized as a STANDALONE clue and not counted toward rarity.
_GENERIC_NOUN = {
    "year", "years", "author", "writer", "paper", "papers", "journal", "series",
    "restaurant", "novel", "book", "film", "movie", "show", "song", "album",
    "company", "organization", "group", "team", "member", "people", "place",
    "city", "town", "country", "region", "area", "event", "story", "title",
    "work", "works", "project", "product", "model", "version", "edition",
    "number", "list", "set", "type", "kind", "part", "piece", "item",
}
#: Source-type hints → a source term appended to a targeted query.
_SOURCE_HINTS = {
    "paper": "paper", "study": "study", "patent": "patent", "filing": "filing",
    "press release": "press release", "obituary": "obituary",
    "wikipedia": "wikipedia", "journal": "journal article", "thesis": "thesis",
    "dissertation": "dissertation", "interview": "interview",
    "court": "court ruling", "census": "census", "database": "database",
    "documentary": "documentary", "encyclopedia": "encyclopedia",
}
_TITLE_WORDS = ("president", "ceo", "chief", "director", "founder", "minister",
                "professor", "chairman", "secretary", "governor", "mayor",
                "author", "winner", "champion", "owner", "inventor", "soldier",
                "monk", "priest", "scientist", "engineer", "architect")
_INSTITUTION_SUFFIX = ("university", "college", "institute", "laboratory",
                       "company", "corporation", "foundation", "association",
                       "committee", "department", "ministry", "academy",
                       "society", "agency", "ironworks", "factory", "museum")
_LOCATION_PREP = ("in", "near", "at", "from", "around", "outside", "inside")
#: A small common-word list: 5+ letter words that are NOT "rare/distinctive".
_COMMON_WORDS = {
    "about", "above", "after", "again", "their", "there", "these", "those",
    "which", "while", "would", "could", "should", "first", "later", "where",
    "world", "years", "place", "thing", "people", "person", "school", "story",
    "series", "author", "novel", "movie", "paper", "music", "water", "house",
    "house", "family", "child", "woman", "history", "science", "language",
    "country", "region", "during", "between", "before", "around", "called",
    "named", "known", "famous", "popular", "modern", "classic", "century",
    "national", "general", "number", "company", "member", "group", "event",
    "title", "early", "decade", "period", "former", "winner", "founded",
    "another", "various", "certain", "several", "include", "including", "located",
    "featured", "dedicated", "published", "released", "produced", "created",
}


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def _is_boiler(t: str) -> bool:
    tl = t.lower()
    return tl in _STOPWORDS or tl in _BOILER or tl.isdigit()


def _is_meaningful(t: str) -> bool:
    """Counts toward query content: not a stopword/placeholder."""
    return not _is_boiler(t)


def _is_noun_like(t: str) -> bool:
    """Approximate 'a real noun head' (no POS tagger): not a stopword/placeholder
    and not a demonym/generic adjective; length>=3."""
    tl = t.lower()
    return (len(t) >= 3 and tl not in _STOPWORDS and tl not in _BOILER
            and tl not in _GENERIC_ADJ and not t.isdigit())


def _is_rare(t: str) -> bool:
    tl = t.lower()
    if tl in _GENERIC_NOUN or tl in _GENERIC_ADJ or tl in _COMMON_WORDS or _is_boiler(t):
        return False
    return len(t) >= 8 or (len(t) >= 5 and tl not in _COMMON_WORDS)


def _dedupe_tokens(query: str) -> str:
    """Drop repeated tokens (order-preserving, case-insensitive) for unquoted
    queries so overlapping spans don't yield 'Acme Robotics Robotics Acme'."""
    if '"' in query:
        return query                       # keep exact phrases intact
    out, seen = [], set()
    for t in query.split():
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return " ".join(out)


def _cap(query: str, *, max_chars: int = MAX_QUERY_CHARS,
         max_tokens: int = MAX_QUERY_TOKENS) -> str:
    toks = _dedupe_tokens(query).split()
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    return " ".join(toks).strip()[:max_chars].strip()


def _dedupe(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for s in seq:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out


# --------------------------------------------------------------------------- spans
def _trim_span(toks: list[str]) -> list[str]:
    """Drop leading/trailing boilerplate/placeholder tokens so a span starts and
    ends on content. Demonyms are not boilerplate, so 'Mexican restaurant' keeps
    'Mexican' while 'African One' trims to 'African' (then dropped as too short)."""
    i, j = 0, len(toks)
    while i < j and _is_boiler(toks[i]):
        i += 1
    while j > i and _is_boiler(toks[j - 1]):
        j -= 1
    return toks[i:j]


def _good_span(toks: list[str]) -> bool:
    """A span is a usable clue if it has >=2 meaningful tokens and a noun-like head,
    OR it is a multiword capitalized entity."""
    if len(toks) < 2:
        return False
    meaningful = [t for t in toks if _is_meaningful(t)]
    if len(meaningful) < 2:
        return False
    if not any(_is_noun_like(t) for t in toks):
        return False
    return True


def extract_spans(clause: str) -> list[str]:
    """Maximal runs of non-stopword tokens → trimmed noun-phrase-ish spans."""
    spans: list[str] = []
    run: list[str] = []
    for tok in _tokens(clause):
        if tok.lower() in _STOPWORDS:
            if run:
                spans.append(run)
                run = []
        else:
            run.append(tok)
    if run:
        spans.append(run)
    out: list[str] = []
    for r in spans:
        t = _trim_span(r)
        if _good_span(t):
            out.append(" ".join(t[:6]))
    return _dedupe(out)


def _multiword_entities(text: str) -> list[str]:
    ents = []
    for m in _CAP_RUN.findall(text):
        toks = m.split()
        if len(toks) < 2:
            continue
        if all(w.lower() in _GENERIC_ADJ or w.lower() in _BOILER for w in toks):
            continue
        ents.append(m.strip())
    return _dedupe(ents)


def _single_entities(text: str) -> list[str]:
    """Distinctive single capitalized words (NOT generic demonyms / boilerplate)."""
    out = []
    for m in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text):
        ml = m.lower()
        if ml in _GENERIC_ADJ or ml in _BOILER or ml in _STOPWORDS or ml in _GENERIC_NOUN:
            continue
        if _is_rare(m) or len(m) >= 5:
            out.append(m)
    return _dedupe(out)


def split_clauses(question: str) -> list[str]:
    parts = _CLAUSE_SPLIT.split(question)
    return [p.strip() for p in parts if p and p.strip() and len(p.strip()) > 2]


@dataclass
class Clues:
    quoted: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)          # multiword + distinctive single
    phrase_spans: list[str] = field(default_factory=list)      # noun-phrase-ish spans
    clause_clues: list[str] = field(default_factory=list)      # best span per clause, in order
    years: list[str] = field(default_factory=list)
    date_ranges: list[str] = field(default_factory=list)
    rare_terms: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) for k, v in self.__dict__.items()}


def _span_specificity(span: str) -> float:
    toks = span.split()
    return (sum(1 for t in toks if _is_rare(t)) * 2.0
            + sum(1 for t in toks if t[:1].isupper()) * 1.0
            + sum(1 for t in toks if _is_noun_like(t)) * 0.5
            + (0.5 if len(toks) >= 3 else 0.0))


def extract_clues(question: str) -> Clues:
    """Deterministically extract clue spans (v1: spans, not single tokens)."""
    quoted = _dedupe(_QUOTED.findall(question))
    multi = _multiword_entities(question)
    singles = _single_entities(question)
    entities = _dedupe(multi + singles)

    clauses = split_clauses(question)
    all_spans: list[str] = []
    clause_clues: list[str] = []
    for cl in clauses:
        spans = extract_spans(cl)
        all_spans.extend(spans)
        if spans:
            clause_clues.append(max(spans, key=_span_specificity))
    phrase_spans = sorted(_dedupe(all_spans), key=_span_specificity, reverse=True)

    ranges = _dedupe([f"{a}-{b}" for a, b in _YEAR_RANGE.findall(question)])
    years = _dedupe(_YEAR.findall(question))
    rare_terms = sorted({t for t in _tokens(question) if _is_rare(t) and not t[:1].isupper()},
                        key=lambda t: (-len(t), t))[:8]
    ql = question.lower()
    titles = _dedupe([w for w in _TITLE_WORDS if w in ql])
    institutions = _dedupe([e for e in entities
                            if any(s in e.lower() for s in _INSTITUTION_SUFFIX)]
                           + [s for s in phrase_spans
                              if any(suf in s.lower() for suf in _INSTITUTION_SUFFIX)])
    # locations: a capitalized entity that follows a location preposition.
    locations: list[str] = []
    toks = _tokens(question)
    low = [t.lower() for t in toks]
    for i, t in enumerate(toks):
        if low[i] in _LOCATION_PREP and i + 1 < len(toks) and toks[i + 1][:1].isupper():
            for e in entities:
                if e.split()[0] == toks[i + 1]:
                    locations.append(e)
                    break
    locations = _dedupe(locations)
    source_hints = _dedupe([v for k, v in _SOURCE_HINTS.items() if k in ql])
    return Clues(quoted=quoted, entities=entities, phrase_spans=phrase_spans,
                 clause_clues=clause_clues, years=years, date_ranges=ranges,
                 rare_terms=rare_terms, titles=titles, institutions=institutions,
                 locations=locations, source_hints=source_hints)


# ----------------------------------------------------------------- answer shape
def answer_shape_terms(question: str) -> list[str]:
    """Infer answer-shape phrasing from the question (no gold answer used)."""
    ql = question.lower()
    terms: list[str] = []
    if ("name" in ql and ("year" in ql or "born" in ql)) or "born" in ql:
        terms.append("born")
    if "series" in ql and ("tv" in ql or "television" in ql or "show" in ql or "anime" in ql):
        terms.append("TV series cast")
    if "paper" in ql or "journal" in ql or "study" in ql:
        terms.append("journal")
    if "manga" in ql:
        terms.append("manga")
    if any(w in ql for w in ("what time", "what year", "what date", "when ")):
        terms.append("date")
    return _dedupe(terms)


# ----------------------------------------------------------------- quality score
@dataclass
class QueryQuality:
    meaningful_token_count: int
    rare_token_count: int
    generic_token_penalty: int
    span_length_score: float
    clue_specificity_score: float
    expected_search_quality: float
    is_quoted: bool
    has_multiword_entity: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "meaningful_token_count": self.meaningful_token_count,
            "rare_token_count": self.rare_token_count,
            "generic_token_penalty": self.generic_token_penalty,
            "span_length_score": round(self.span_length_score, 3),
            "clue_specificity_score": round(self.clue_specificity_score, 3),
            "expected_search_quality": round(self.expected_search_quality, 3),
        }


def score_query(query: str) -> QueryQuality:
    raw = query.replace('"', " ")
    toks = _tokens(raw)
    meaningful = [t for t in toks if _is_meaningful(t)]
    rare = [t for t in toks if _is_rare(t)]
    generic = [t for t in toks if t.lower() in _GENERIC_ADJ or t.lower() in _GENERIC_NOUN]
    noun_like = [t for t in toks if _is_noun_like(t)]
    n = len(meaningful)
    span_len = 1.0 if 3 <= n <= 8 else (0.5 if n == 2 else (0.2 if n > 8 else 0.0))
    is_quoted = '"' in query
    has_multi = len(_multiword_entities(query)) > 0
    specificity = (2.0 * len(rare) + 1.0 * len(noun_like) + (1.0 if has_multi else 0.0)
                   + (1.0 if is_quoted else 0.0) - 0.5 * len(generic))
    expected = (1.5 * len(noun_like) + 1.0 * len(rare) + 2.0 * span_len
                + (1.5 if (is_quoted or has_multi) else 0.0)
                - 0.7 * max(0, len(generic) - 1)
                - (2.0 if len(noun_like) == 0 else 0.0))
    return QueryQuality(
        meaningful_token_count=n, rare_token_count=len(rare),
        generic_token_penalty=len(generic), span_length_score=span_len,
        clue_specificity_score=specificity, expected_search_quality=expected,
        is_quoted=is_quoted, has_multiword_entity=has_multi)


def _passes(q: str, qual: QueryQuality) -> Optional[str]:
    """Return a drop-reason string if the query should be dropped, else None."""
    toks = _tokens(q.replace('"', " "))
    noun_like = [t for t in toks if _is_noun_like(t)]
    if not noun_like and not qual.is_quoted:
        return "no_noun_like_token"
    if qual.meaningful_token_count < MIN_MEANINGFUL_TOKENS and not (
            qual.is_quoted or qual.has_multiword_entity):
        return "too_short"
    if qual.expected_search_quality < QUALITY_THRESHOLD:
        return "below_quality_threshold"
    return None


@dataclass
class CandidateQuery:
    arm: str
    query: str
    clue_ids: list[str] = field(default_factory=list)
    quality: float = 0.0

    @property
    def query_text_hash(self) -> str:
        return hashlib.sha256(self.query.lower().encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "query": self.query, "clue_ids": list(self.clue_ids),
                "query_text_hash": self.query_text_hash,
                "expected_search_quality": round(self.quality, 3)}


@dataclass
class DecompositionResult:
    candidates: list[CandidateQuery]                 # kept, ranked
    dropped: list[dict[str, Any]]                    # {arm, query, reason, quality}
    all_scored: list[dict[str, Any]]                 # every candidate + score

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    def to_debug(self) -> dict[str, Any]:
        return {
            "candidate_queries": [c.to_dict() for c in self.candidates],
            "dropped_count": self.dropped_count,
            "dropped": self.dropped[:8],
            "all_candidates": self.all_scored[:16],
        }


def _compress(question: str) -> str:
    return _cap(" ".join(t for t in _tokens(question) if t.lower() not in _STOPWORDS))


def _noise_removed(question: str) -> str:
    return _cap(" ".join(t for t in _tokens(question) if _is_meaningful(t)))


def decompose(question: str, *, max_queries: int = 6, max_chars: int = MAX_QUERY_CHARS,
              max_tokens: int = MAX_QUERY_TOKENS) -> DecompositionResult:
    """Build, score, and filter candidate queries. Returns kept + dropped + scores."""
    c = extract_clues(question)
    shape = answer_shape_terms(question)
    raw: list[CandidateQuery] = []

    def add(arm: str, parts: list[str], clue_ids: list[str]) -> None:
        q = _cap(" ".join(p for p in parts if p), max_chars=max_chars, max_tokens=max_tokens)
        if q:
            raw.append(CandidateQuery(arm=arm, query=q, clue_ids=clue_ids))

    # exact_phrase_clue: quoted phrases, else the most distinctive multi-word span.
    if c.quoted:
        add("exact_phrase_clue", [" ".join(f'"{p}"' for p in c.quoted[:2])],
            [f"quoted[{i}]" for i in range(min(2, len(c.quoted)))])
    elif c.phrase_spans:
        add("exact_phrase_clue", [f'"{c.phrase_spans[0]}"'], ["span[0]"])

    # entity_clue: multiword entity / top phrase spans (never single generic words);
    # an inferred answer-shape term influences phrasing (no gold answer used).
    ent_parts = _dedupe(((c.entities[:2] or []) + c.phrase_spans[:2]))[:3]
    if ent_parts:
        add("entity_clue", ent_parts + shape[:1],
            [f"entity[{i}]" for i in range(min(2, len(c.entities)))]
            + [f"span[{i}]" for i in range(min(2, len(c.phrase_spans)))]
            + (["answer_shape"] if shape else []))

    # relation_clue: clue spans from DIFFERENT clauses (predicate+object), + a title.
    if len(c.clause_clues) >= 2:
        add("relation_clue", _dedupe(c.clause_clues[:2] + c.titles[:1]),
            [f"clause[{i}]" for i in range(min(2, len(c.clause_clues)))])
    elif c.phrase_spans:
        add("relation_clue", _dedupe(c.phrase_spans[:2] + c.titles[:1]),
            [f"span[{i}]" for i in range(min(2, len(c.phrase_spans)))])

    # rare_terms_clue: 4-8 rare terms from across clauses + the top span.
    if c.rare_terms:
        add("rare_terms_clue", c.rare_terms[:6] + c.phrase_spans[:1],
            [f"rare[{i}]" for i in range(len(c.rare_terms[:6]))])

    # date_range_clue: date/range + the nearest distinctive noun phrase (NOT bare date).
    if (c.date_ranges or c.years) and (c.phrase_spans or c.entities):
        dparts = (c.date_ranges[:1] or c.years[:1])
        anchor = (c.phrase_spans[0] if c.phrase_spans else c.entities[0])
        add("date_range_clue", [anchor] + dparts,
            ["span[0]"] + [f"date[{i}]" for i in range(len(dparts))])

    # location_constraint_clue: location + object-type span.
    if c.locations and (c.phrase_spans or c.entities):
        obj = next((s for s in c.phrase_spans if s.lower() not in
                    {loc.lower() for loc in c.locations}), c.phrase_spans[:1] and c.phrase_spans[0])
        add("location_constraint_clue", _dedupe([c.locations[0]] + ([obj] if obj else [])),
            ["location[0]", "span[0]"])

    # source_type_query: source hint + entity/span.
    if c.source_hints and (c.entities or c.phrase_spans):
        add("source_type_query",
            _dedupe((c.entities[:1] or c.phrase_spans[:1]) + c.source_hints[:1]),
            ["entity[0]", "source[0]"])

    # Fallbacks LAST (so the long arm is never the cold default).
    add("negative_noise_removed", [_noise_removed(question)], ["noise_removed"])
    add("full_question_compressed", [_compress(question)], ["full_question"])

    # Score, filter, de-dupe by arm.
    kept: list[CandidateQuery] = []
    dropped: list[dict[str, Any]] = []
    all_scored: list[dict[str, Any]] = []
    seen_arms: set[str] = set()
    fallback_arms = {"negative_noise_removed", "full_question_compressed"}
    for cq in raw:
        qual = score_query(cq.query)
        cq.quality = qual.expected_search_quality
        all_scored.append({**cq.to_dict(), **qual.to_dict()})
        reason = _passes(cq.query, qual)
        # Fallbacks are exempt from the drop filter (they are the safety net), but
        # still ranked last by their score.
        if reason and cq.arm not in fallback_arms:
            dropped.append({"arm": cq.arm, "query": cq.query, "reason": reason,
                            "expected_search_quality": round(cq.quality, 3)})
            continue
        if cq.arm in seen_arms:
            continue
        seen_arms.add(cq.arm)
        kept.append(cq)

    targeted = [c for c in kept if c.arm not in fallback_arms]
    fallback = [c for c in kept if c.arm in fallback_arms]
    # Keep the concise/distinctive arms first (priority), breaking ties by quality.
    targeted.sort(key=lambda c: (_ARM_PRIORITY.get(c.arm, 9), -c.quality))
    keep_targeted = max(1, max_queries - len(fallback))
    final = (targeted[:keep_targeted] + fallback)[:max_queries]
    return DecompositionResult(candidates=final, dropped=dropped, all_scored=all_scored)


def decompose_queries(question: str, *, max_queries: int = 6,
                      max_chars: int = MAX_QUERY_CHARS,
                      max_tokens: int = MAX_QUERY_TOKENS) -> list[CandidateQuery]:
    """Kept candidate queries only (back-compat wrapper around :func:`decompose`)."""
    return decompose(question, max_queries=max_queries, max_chars=max_chars,
                     max_tokens=max_tokens).candidates
