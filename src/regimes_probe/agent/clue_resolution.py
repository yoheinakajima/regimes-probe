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


# ===========================================================================
# Typed candidate-hypothesis policy (generic, not BrowseComp-specific).
#
# The staged search latched onto wrong intermediate candidates (sources like
# "Brittle Paper", broad orgs like "World Health Organization", broad locations
# like "Tennessee", concepts like "Art Deco") and re-exploited them. The fix is
# to treat each candidate as a TYPED hypothesis, carry it forward only if its role
# is compatible with the unresolved target and follow-up evidence improves, and
# avoid sticky exploitation via a small beam.
# ===========================================================================

#: Generic candidate roles (no topical categories).
ROLES = ("person", "organization", "location", "title_or_work",
         "publication_or_source", "event", "concept", "date_or_time", "unknown")

_SOURCE_WORDS = {"paper", "papers", "press", "journal", "magazine", "news", "times",
                 "post", "gazette", "review", "media", "books", "publishing",
                 "publisher", "publishers", "daily", "weekly", "wire", "bulletin",
                 "tribune", "herald", "chronicle", "dispatch", "digest", "podcast"}
_ORG_WORDS = {"organization", "organisation", "association", "department", "bureau",
              "agency", "council", "committee", "office", "service", "program",
              "programme", "network", "foundation", "society", "union", "federation",
              "ministry", "commission", "company", "corporation", "inc", "corp",
              "institute", "university", "college", "school", "hospital", "club",
              "league", "academy", "authority", "board", "sheriffs", "police"}
_ORG_BROAD_PREFIX = {"world", "national", "international", "federal", "global",
                     "royal", "united", "pan", "european", "american", "general"}
_CONCEPTS = {"art", "deco", "style", "movement", "genre", "theme", "concept",
             "antagonist", "protagonist", "character", "era", "period", "design",
             "architecture", "philosophy", "theory", "ideology", "color", "colour",
             "modernism", "realism", "romanticism", "minimalism", "cubism",
             "abstract", "aesthetic", "narrative", "trope", "motif"}
#: Generic broad-location lexicon (continents + US states + common countries).
_BROAD_LOCATIONS = {
    "africa", "asia", "europe", "america", "north america", "south america",
    "antarctica", "australia", "oceania", "atlantic", "pacific", "arctic",
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana",
    "nebraska", "nevada", "ohio", "oklahoma", "oregon", "pennsylvania", "tennessee",
    "texas", "utah", "vermont", "virginia", "washington", "wisconsin", "wyoming",
    "new mexico", "new york", "new jersey", "north carolina", "south carolina",
    "north dakota", "south dakota", "west virginia", "rhode island",
    "england", "scotland", "wales", "ireland", "france", "germany", "spain",
    "italy", "china", "japan", "india", "russia", "canada", "mexico", "brazil",
    "egypt", "nigeria", "kenya", "ghana", "australia", "england", "britain",
}
_ARTICLES = {"the", "a", "an"}
_DATE_RE = re.compile(r"^(1[0-9]{3}|20[0-9]{2})s?$")


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def classify_entity_role(text: str) -> str:
    """Deterministically type a candidate entity into a generic role. No gold."""
    t = text.strip()
    tl = _norm(t)
    words = t.split()
    low = [w.lower() for w in words]
    if _DATE_RE.match(tl):
        return "date_or_time"
    if tl in _BROAD_LOCATIONS:
        return "location"
    if low and low[0] in _ARTICLES and len(words) >= 2:
        return "title_or_work"
    if any(w in _SOURCE_WORDS for w in low):
        return "publication_or_source"
    if any(w in _ORG_WORDS for w in low) or (low and low[0] in _ORG_BROAD_PREFIX
                                             and len(words) >= 2):
        return "organization"
    if any(w in _CONCEPTS for w in low):
        return "concept"
    if len(words) >= 2:
        return "person"          # default for multiword proper names
    return "unknown"             # a single ambiguous capitalized token


def _is_broad_org(text: str) -> bool:
    low = [w.lower() for w in text.split()]
    return bool(low) and low[0] in _ORG_BROAD_PREFIX


#: Question cues → likely target / intermediate roles (generic, gold-free).
_TARGET_CUES = {
    "person": ("who ", "whom", "founder", "author", "writer", "journalist",
               "person", "name and", "full name", "first name", "last name",
               "surname", "middle name", "ceo", "director", "president",
               "inventor", "scientist", "artist", "actor", "actress", "musician"),
    "title_or_work": ("which tv series", "tv series", "which series", "which book",
                      "what book", "which film", "what film", "which movie",
                      "which manga", "what manga", "which album", "which song",
                      "which novel", "what novel", "title of", "name of the show"),
    "publication_or_source": ("which paper", "what paper", "which journal",
                              "what journal", "which report", "which magazine",
                              "which newspaper", "which publication"),
    "organization": ("which restaurant", "what restaurant", "which hotel",
                     "which museum", "which company", "which organization",
                     "which institution", "which university", "which society",
                     "which team", "which club", "name of the restaurant"),
    "location": ("which town", "which city", "which place", "where ", "which country",
                 "which monument", "which building", "which street", "which park"),
    "event": ("which event", "which battle", "which war", "which festival",
              "which competition", "which conference", "which ceremony"),
    "date_or_time": ("what year", "which year", "what date", "what time", "when "),
}
#: Roles that are worth chasing as INTERMEDIATE hops regardless of the final target.
_DEFAULT_INTERMEDIATE = ("person", "organization", "title_or_work", "location", "event")


def infer_target_roles(question: str) -> tuple[list[str], list[str]]:
    """Infer likely target role(s) + intermediate role(s) from the question text.

    Returns ``(target_roles, intermediate_roles)``. Never uses the gold answer.
    """
    ql = " " + question.lower() + " "
    targets: list[str] = []
    for role, cues in _TARGET_CUES.items():
        if any(c in ql for c in cues):
            targets.append(role)
    if not targets:
        targets = ["person", "title_or_work"]          # generic default
    # Intermediate roles: the default chase set, plus the targets themselves.
    intermediate = list(dict.fromkeys(list(targets) + list(_DEFAULT_INTERMEDIATE)))
    # date_or_time is an answer SHAPE, not a search candidate to chase.
    intermediate = [r for r in intermediate if r != "date_or_time"]
    return targets, intermediate


@dataclass
class CandidateHypothesis:
    candidate_text: str
    normalized_text: str
    role: str
    source_fields: list[str] = field(default_factory=list)
    source_tool: Optional[str] = None
    source_result_id: Optional[str] = None
    stage_found: int = 1
    raw_score: float = 0.0
    role_match_score: float = 0.0
    evidence_progress_score: float = 0.0
    genericity_penalty: float = 0.0
    source_entity_penalty: float = 0.0
    location_penalty: float = 0.0
    sticky_penalty: float = 0.0
    adjusted_score: float = 0.0
    selected: bool = False
    rejected: bool = False
    rejection_reason: Optional[str] = None
    # provenance for debug
    frequency: int = 0
    n_domains: int = 0
    contaminated: bool = False

    def recompute(self) -> None:
        self.adjusted_score = (self.raw_score + self.role_match_score
                               + self.evidence_progress_score
                               - self.genericity_penalty - self.source_entity_penalty
                               - self.location_penalty - self.sticky_penalty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_text": self.candidate_text, "role": self.role,
            "source_fields": list(self.source_fields), "stage_found": self.stage_found,
            "raw_score": round(self.raw_score, 3),
            "role_match_score": round(self.role_match_score, 3),
            "evidence_progress_score": round(self.evidence_progress_score, 3),
            "genericity_penalty": round(self.genericity_penalty, 3),
            "source_entity_penalty": round(self.source_entity_penalty, 3),
            "location_penalty": round(self.location_penalty, 3),
            "sticky_penalty": round(self.sticky_penalty, 3),
            "adjusted_score": round(self.adjusted_score, 3),
            "selected": self.selected, "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


def _score_role(h: CandidateHypothesis, target_roles, intermediate_roles, *,
                clue_token_set: set[str]) -> None:
    """Set role_match_score + the role-based penalties (generic rules)."""
    role = h.role
    if role in target_roles:
        h.role_match_score = 2.0
    elif role in intermediate_roles:
        h.role_match_score = 0.5
    else:
        h.role_match_score = -0.5
    # Penalize source/publisher candidates unless the question wants a source.
    if role == "publication_or_source" and "publication_or_source" not in target_roles:
        h.source_entity_penalty = 2.0
    # Penalize broad locations unless the question wants a place.
    if role == "location" and "location" not in target_roles:
        h.location_penalty = 2.0 if _norm(h.candidate_text) in _BROAD_LOCATIONS else 1.0
    # Penalize generic concepts unless the question wants a concept.
    if role == "concept" and "concept" not in target_roles:
        h.genericity_penalty = 2.0
    # Penalize BROAD organizations (World/National/…) unless org is the target.
    if role == "organization" and "organization" not in target_roles and _is_broad_org(h.candidate_text):
        h.genericity_penalty = max(h.genericity_penalty, 1.5)
    # Penalize a candidate that is just an obvious clue term already in the question.
    if {t for t in _tokens(h.candidate_text.lower())} <= clue_token_set and clue_token_set:
        h.genericity_penalty += 0.8
    # Penalize candidates seen only in contaminated / domain-only context.
    if h.contaminated:
        h.genericity_penalty += 1.0
    if h.source_fields == ["domain"]:
        h.source_entity_penalty = max(h.source_entity_penalty, 1.0)
    # Cross-domain corroboration boost.
    if h.n_domains >= 2:
        h.role_match_score += 0.5
    h.recompute()


def build_hypotheses(observations, *, question: str, target_roles: list[str],
                     intermediate_roles: list[str], clue_terms: list[str],
                     stage_found: int, top_k: int = 8) -> list[CandidateHypothesis]:
    """Extract candidates from results, type them, and score them as hypotheses."""
    clue_set = {t.lower() for t in clue_terms if len(t) >= 3}
    q_token_set = {t.lower() for t in _tokens(question)}
    agg: dict[str, CandidateHypothesis] = {}
    domains_by: dict[str, set[str]] = {}
    for o in observations:
        if getattr(o, "failed", False):
            continue
        title = getattr(o, "title", "") or ""
        snippet = getattr(o, "snippet", "") or ""
        url = getattr(o, "url", "") or ""
        authority = float(getattr(o, "source_authority", 0.0))
        contaminated = bool(getattr(o, "benchmark_contaminated", False))
        host = (urlparse(url).hostname or "").lower()
        near = bool(clue_set & {t.lower() for t in _tokens(f"{title} {snippet}")})
        seen_this_obs: dict[str, list[str]] = {}
        for fieldname, text in (("title", title), ("snippet", snippet)):
            for e in _entities_in_field(text):
                if _is_generic_entity(e) or len(e) < 3:
                    continue
                seen_this_obs.setdefault(e, []).append(fieldname)
        dl = _domain_label(url)
        if (dl and dl not in GENERIC_ENTITIES and dl not in _DOMAIN_NOISE
                and dl not in _COMMON_WORDS and len(dl) >= 4):
            seen_this_obs.setdefault(dl.capitalize(), []).append("domain")
        for e, fields in seen_this_obs.items():
            key = _norm(e)
            h = agg.get(key)
            if h is None:
                h = CandidateHypothesis(
                    candidate_text=e, normalized_text=key, role=classify_entity_role(e),
                    source_fields=[], source_tool=getattr(o, "tool", None),
                    source_result_id=(url or None), stage_found=stage_found)
                agg[key] = h
                domains_by[key] = set()
            h.frequency += 1
            for f in fields:
                if f not in h.source_fields:
                    h.source_fields.append(f)
            h.raw_score = max(h.raw_score, 0.0)
            h._auth = max(getattr(h, "_auth", 0.0), authority)  # type: ignore[attr-defined]
            h._near = getattr(h, "_near", False) or near        # type: ignore[attr-defined]
            h.contaminated = h.contaminated or contaminated
            if host:
                domains_by[key].add(host)
    for key, h in agg.items():
        h.n_domains = len(domains_by.get(key, set()))
        multi = 0.6 if " " in h.candidate_text else 0.0
        auth = getattr(h, "_auth", 0.0)
        near = getattr(h, "_near", False)
        h.raw_score = h.frequency * 1.0 + auth * 1.0 + (0.5 if near else 0.0) + multi
        _score_role(h, target_roles, intermediate_roles, clue_token_set=q_token_set)
    return sorted(agg.values(), key=lambda c: (-c.adjusted_score, c.candidate_text))[:top_k]



# --------------------------------------------------------------- anti-sticky beam
@dataclass
class BeamSelection:
    hypothesis: CandidateHypothesis
    reason: str
    rejected: list[dict[str, Any]] = field(default_factory=list)
    sticky_applied: bool = False
    switched: bool = False


class HypothesisBeam:
    """A small beam of candidate hypotheses with anti-sticky / forced-exploration.

    Prevents loops like 'Art Deco' -> 'Art Deco' -> 'Art Deco' by: applying a
    sticky_penalty when the last-selected candidate made no progress, and FORCING
    exploration of the next-best candidate after a candidate fails twice.
    """

    STICKY_PENALTY = 2.0
    MAX_FAILS = 2

    def __init__(self, beam_size: int = 3) -> None:
        self.beam_size = beam_size
        self.pool: dict[str, CandidateHypothesis] = {}
        self.state: dict[str, dict[str, Any]] = {}   # norm -> {selected, no_progress, failed}
        self.last_selected: Optional[str] = None
        self.used_query_hashes: set[str] = set()
        # metrics
        self.switch_count = 0
        self.sticky_count = 0
        self.no_progress_count = 0
        self.repeated_query_count = 0
        self.beam_sizes: list[int] = []

    def observe(self, hypotheses: list[CandidateHypothesis]) -> None:
        """Refresh the pool with the latest scored hypotheses (state is kept)."""
        for h in hypotheses:
            self.pool[h.normalized_text] = h

    def record_progress(self, norm: str, improved: bool) -> None:
        st = self.state.setdefault(norm, {"selected": 0, "no_progress": 0,
                                          "failed": False, "progress": False})
        if improved:
            st["progress"] = True
            st["no_progress"] = 0          # progress clears the sticky streak
        else:
            st["no_progress"] += 1
            self.no_progress_count += 1
            if st["no_progress"] >= self.MAX_FAILS:
                st["failed"] = True

    def select(self) -> Optional[BeamSelection]:
        """Pick the best non-failed candidate, applying sticky penalties. Records
        the rejected candidates and reasons."""
        if not self.pool:
            return None
        ranked = sorted(self.pool.values(),
                        key=lambda h: (-h.adjusted_score, h.candidate_text))
        beam = ranked[: self.beam_size]
        self.beam_sizes.append(len(beam))
        # Apply sticky penalty (in-place on a copy of the score) using state.
        scored: list[tuple[float, CandidateHypothesis, bool]] = []
        for h in beam:
            st = self.state.get(h.normalized_text, {})
            sticky = self.STICKY_PENALTY if (st.get("no_progress", 0) > 0) else 0.0
            h.sticky_penalty = sticky
            # Evidence-progress gate: a candidate whose follow-up improved evidence
            # is preferred for the next hop.
            h.evidence_progress_score = 1.0 if st.get("progress") else 0.0
            h.recompute()
            scored.append((h.adjusted_score, h, st.get("failed", False)))
        # Drop failed candidates (forced exploration); keep as rejected.
        rejected: list[dict[str, Any]] = []
        viable = []
        for adj, h, failed in scored:
            if failed:
                h.rejected = True
                h.rejection_reason = "failed_twice_no_progress"
                rejected.append(h.to_dict())
            else:
                viable.append((adj, h))
        if not viable:                       # everything failed -> allow the top one
            adj, h = max(((a, hh) for a, hh, _ in scored), key=lambda t: t[0])
            viable = [(adj, h)]
        viable.sort(key=lambda t: (-t[0], t[1].candidate_text))
        best_adj, best = viable[0]
        switched = self.last_selected is not None and best.normalized_text != self.last_selected
        sticky_applied = best.sticky_penalty > 0
        if switched:
            self.switch_count += 1
        for adj, h in viable[1:]:
            h.rejected = True
            h.rejection_reason = "lower_adjusted_score"
            rejected.append(h.to_dict())
        best.selected = True
        if any(h.sticky_penalty > 0 for _, h, _ in scored):
            self.sticky_count += 1
        st = self.state.setdefault(best.normalized_text,
                                   {"selected": 0, "no_progress": 0, "failed": False})
        st["selected"] += 1
        self.last_selected = best.normalized_text
        reason = (f"role={best.role} adjusted={best.adjusted_score:.2f} "
                  f"(raw={best.raw_score:.2f}, role_match={best.role_match_score:+.1f}, "
                  f"src_pen={best.source_entity_penalty:.1f}, loc_pen={best.location_penalty:.1f}, "
                  f"generic_pen={best.genericity_penalty:.1f}, sticky={best.sticky_penalty:.1f})"
                  + ("; switched after no-progress" if switched else ""))
        return BeamSelection(hypothesis=best, reason=reason, rejected=rejected[:6],
                             sticky_applied=sticky_applied, switched=switched)

    def register_query(self, query: str) -> bool:
        """Track follow-up query strings; True if this query is a repeat."""
        import hashlib
        h = hashlib.sha256(query.lower().encode()).hexdigest()[:16]
        repeat = h in self.used_query_hashes
        if repeat:
            self.repeated_query_count += 1
        self.used_query_hashes.add(h)
        return repeat

    def mean_beam_size(self) -> float:
        return (sum(self.beam_sizes) / len(self.beam_sizes)) if self.beam_sizes else 0.0


def compose_followup_from_hypothesis(
        hyp: CandidateHypothesis, *, clue_spans: list[str], used_clue_spans: set[str],
        answer_shape: list[str], beam: Optional[HypothesisBeam] = None,
        max_chars: int = MAX_QUERY_CHARS, max_tokens: int = MAX_QUERY_TOKENS) -> FollowupQuery:
    """Compose a follow-up = selected candidate + one unresolved distinctive clue
    span (preferred) or an answer-shape hint. Avoids repeating an equivalent query
    and never pairs the candidate with an already-used / generic-only term."""
    ent = hyp.candidate_text
    # Try each unused clue span; skip ones that duplicate a prior query.
    for s in clue_spans:
        sl = s.lower()
        if sl in used_clue_spans or ent.lower() in sl or sl in ent.lower():
            continue
        q = _cap(f'"{ent}" {s}', max_chars=max_chars, max_tokens=max_tokens)
        if beam is not None and beam.register_query(q):
            continue                          # equivalent query already issued
        return FollowupQuery(query=q, query_arm="candidate_entity_followup",
                             selected_candidate=ent, clue_used=s,
                             reason=f"{ent} [{hyp.role}] + unresolved clue '{s}'")
    if answer_shape:
        q = _cap(f'"{ent}" {answer_shape[0]}', max_chars=max_chars, max_tokens=max_tokens)
        if not (beam is not None and beam.register_query(q)):
            return FollowupQuery(query=q, query_arm="answer_shape_followup",
                                 selected_candidate=ent, clue_used=None,
                                 reason=f"{ent} [{hyp.role}] + answer-shape '{answer_shape[0]}'")
    q = _cap(f'"{ent}"', max_chars=max_chars, max_tokens=max_tokens)
    if beam is not None:
        beam.register_query(q)
    return FollowupQuery(query=q, query_arm="candidate_entity_followup",
                         selected_candidate=ent, clue_used=None,
                         reason=f"{ent} [{hyp.role}] (no remaining distinctive clue/shape)")
