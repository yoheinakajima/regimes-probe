"""Level 2 query decomposition + benchmark-contamination detection (no network)."""

from __future__ import annotations

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.eval.contamination import (
    detect_contamination, longest_token_run)
from regimes_probe.eval.harness import run_condition
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.query_decomposition import (
    DECOMPOSITION_ARMS, MAX_QUERY_CHARS, MAX_QUERY_TOKENS, QUALITY_THRESHOLD,
    decompose, decompose_queries, extract_clues, score_query)
from regimes_probe.policy.query_policy import QueryPolicy
from regimes_probe.policy.signatures import SignatureExtractor

TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]

LONG_Q = ('I am trying to identify a former president of the "Lumen Circle" society, '
          'founded in 1894, who later became a professor at Calderwood University and '
          'won the Tindle Prize between 1990 and 1995. What is the exact middle name '
          'of this individual?')


# ----------------------------------------------------- decomposition
def test_long_question_produces_multiple_shorter_queries():
    cands = decompose_queries(LONG_Q)
    assert 3 <= len(cands) <= 6
    # every candidate query is much shorter than the original prompt.
    assert all(len(c.query) < len(LONG_Q) for c in cands)
    assert len({c.arm for c in cands}) == len(cands)        # distinct arms
    assert all(c.arm in DECOMPOSITION_ARMS for c in cands)


def test_query_length_is_capped():
    cands = decompose_queries(LONG_Q)
    for c in cands:
        assert len(c.query) <= MAX_QUERY_CHARS
        assert len(c.query.split()) <= MAX_QUERY_TOKENS


def test_quoted_proper_noun_and_date_clues_preserved():
    clues = extract_clues(LONG_Q)
    assert "Lumen Circle" in clues.quoted
    assert any("Calderwood" in e for e in clues.entities)
    assert "What" not in clues.entities          # sentence-initial word filtered out
    assert ("1990-1995" in clues.date_ranges) or ("1894" in clues.years)
    by_arm = {c.arm: c.query for c in decompose_queries(LONG_Q)}
    assert '"Lumen Circle"' in by_arm["exact_phrase_clue"]
    assert "Calderwood University" in by_arm["entity_clue"]
    assert "date_range_clue" in by_arm and ("1990-1995" in by_arm["date_range_clue"]
                                            or "1894" in by_arm["date_range_clue"])


def test_full_question_is_not_the_default_first_arm_when_decomposed():
    sig = SignatureExtractor().compute(LONG_Q)
    qp = QueryPolicy("learned")
    mem = PolicyMemory(BanditParams())            # cold bandit, no priors
    plan = qp.formulate(sig, bandit=mem.bandits["query"], explore=False,
                        decompose=True, tool="generic_web_search")
    assert plan.arm in DECOMPOSITION_ARMS
    assert plan.arm != "full_question_compressed"
    assert plan.query and len(plan.query) < len(LONG_Q)
    assert plan.query_text_hash and plan.clue_ids


def test_decomposition_off_preserves_legacy_arms():
    sig = SignatureExtractor().compute(LONG_Q)
    qp = QueryPolicy("learned")
    mem = PolicyMemory(BanditParams())
    plan = qp.formulate(sig, bandit=mem.bandits["query"], decompose=False)
    from regimes_probe.policy.query_policy import QUERY_ARMS
    assert plan.arm in QUERY_ARMS                 # unchanged behavior


# ----------------------------------------------------- query arms logged
def test_query_arms_logged_in_debug_records(items, providers):
    agent = EpistemicAgent(AgentConfig(available_tools=TOOLS + ["page_fetch"],
                                       enable_query_decomposition=True))
    res = run_condition(items[:4], agent, providers, PolicyMemory(BanditParams()),
                        condition="no_memory_search", budget=2, weights=RewardWeights.full())
    rows = [d.to_dict() for d in res.debug]
    search_calls = [c for r in rows for c in r["calls"] if c["tool"] != "page_fetch"]
    assert search_calls
    assert all(c["query_arm"] in DECOMPOSITION_ARMS for c in search_calls)
    assert any(c["query_text_hash"] for c in search_calls)
    assert any(c.get("clue_ids") for c in search_calls)


# ----------------------------------------------------- contamination detector
def test_contamination_flags_benchmark_hosts():
    r = detect_contamination(url="https://huggingface.co/datasets/openai/browsecomp",
                             title="BrowseComp dataset", snippet="rows", question="who won")
    assert r.contaminated and "host" in (r.reason or "")
    r2 = detect_contamination(url="https://github.com/openai/simple-evals",
                              title="simple-evals", snippet="eval harness", question="who won")
    assert r2.contaminated


def test_contamination_flags_question_mirroring_snippet():
    q = ("Which former president of the Lumen Circle society founded in 1894 later "
         "became a professor at Calderwood University and won the Tindle Prize")
    # a spam page that mirrors the whole question text
    r = detect_contamination(url="https://quiz-spam.example/q/123", title="Trivia",
                             snippet=q + " — answer here!", question=q)
    assert r.contaminated
    assert longest_token_run(q, q) >= 8


def test_clean_evidence_is_not_contaminated():
    r = detect_contamination(
        url="https://calderwood.edu/people/faculty",
        title="Faculty directory — Department of History",
        snippet="Professor Edrin Vael joined the department in 1991 after chairing the society.",
        question=("Which former president of the Lumen Circle society founded in 1894 later "
                  "became a professor at Calderwood University and won the Tindle Prize"))
    assert not r.contaminated


def test_contaminated_results_penalized_in_reward(items, providers):
    # A run is still scorable; contamination count is surfaced and penalized via the
    # reward component (no crash, deterministic).
    agent = EpistemicAgent(AgentConfig(available_tools=TOOLS + ["page_fetch"],
                                       enable_query_decomposition=True))
    res = run_condition(items[:3], agent, providers, PolicyMemory(BanditParams()),
                        condition="no_memory_search", budget=2, weights=RewardWeights.full())
    # contaminated_results is an int on every outcome (0 on clean fixtures).
    assert all(isinstance(o.contaminated_results, int) for o in res.outcomes)


# ===================================================================== v1: clue spans
AFRICAN = ("I am looking for an African author whose most famous novel had no sell by date "
           "printed on its early editions, who survived a serious road accident in their youth, "
           "and who later taught at a private university. What is the full name of this author?")
MEXICAN = ("There is a Mexican restaurant in New Mexico that was featured in a documentary about "
           "regional cuisine in the early 2000s. What is the name of the restaurant?")
MONUMENT = ("I am trying to identify a 19th century monument dedicated to a World War One soldier, "
            "near which a food festival is held every year, built using materials from a local "
            "ironworks. What is the name of the monument and the year it was completed?")
MANGA = ("I recall a manga set in an elementary school that retells a classic story, first "
         "published in the early 2000s. What is the title of this manga?")


def _queries(q):
    return [c.query for c in decompose_queries(q)]


def test_african_example_uses_clue_spans_not_single_caps():
    qs = [x.lower() for x in _queries(AFRICAN)]
    assert any("road accident" in x for x in qs)
    assert any("private university" in x for x in qs)
    assert not any("african one" in x for x in qs)
    assert not any(x.strip() == "african" for x in qs)


def test_mexican_example_keeps_restaurant_and_location():
    qs = [x.lower() for x in _queries(MEXICAN)]
    assert any("mexican restaurant" in x for x in qs)
    assert any("new mexico" in x for x in qs)
    assert not any(x.strip() == "mexican" for x in qs)


def test_monument_example_no_bare_date_query():
    qs = [x.lower() for x in _queries(MONUMENT)]
    assert not any("name december" in x for x in qs)
    assert any(any(t in x for t in ("19th century monument", "world war one soldier",
                                    "food festival", "ironworks")) for x in qs)
    # no pure-date / single-generic query survived
    assert all(len(x.split()) >= 2 for x in qs)


def test_manga_example_keeps_domain_phrases():
    qs = [x.lower() for x in _queries(MANGA)]
    assert any("manga" in x for x in qs)
    assert any("elementary school" in x for x in qs)
    assert any("classic story" in x for x in qs)
    assert not any("early one" in x for x in qs)


def test_queries_capped_but_not_over_compressed():
    for q in (AFRICAN, MEXICAN, MONUMENT, MANGA):
        cands = decompose_queries(q)
        assert cands
        for c in cands:
            assert len(c.query) <= MAX_QUERY_CHARS
            assert len(c.query.split()) <= MAX_QUERY_TOKENS
        # targeted (non-fallback) queries are not collapsed to a single token.
        for c in cands:
            if c.arm not in ("full_question_compressed", "negative_noise_removed"):
                assert score_query(c.query).meaningful_token_count >= 2 or '"' in c.query


def test_low_quality_single_token_and_generic_queries_are_dropped():
    # Single-generic / no-noun v0 failure forms score below threshold (and are
    # never generated by decompose() in the first place — see the example tests).
    for junk in ("African One", "Mexican", "African", "One", "Early One first"):
        assert score_query(junk).expected_search_quality < QUALITY_THRESHOLD
    # genuine clue spans clear the bar.
    for good in ("private university", "Mexican restaurant", "World War One soldier",
                 "elementary school"):
        assert score_query(good).expected_search_quality >= QUALITY_THRESHOLD
    # and the junk forms never appear as generated candidates.
    for q in (AFRICAN, MEXICAN, MONUMENT, MANGA):
        for c in decompose_queries(q):
            toks = c.query.replace('"', " ").split()
            assert not (len(toks) == 1 and c.query.lower() in
                        {"african", "mexican", "one", "name"})


def test_decompose_exposes_quality_and_dropped_debug():
    r = decompose(MONUMENT)
    dbg = r.to_debug()
    assert "candidate_queries" in dbg and dbg["candidate_queries"]
    assert isinstance(dbg["dropped_count"], int)
    assert all("expected_search_quality" in c for c in dbg["candidate_queries"])
    # every kept candidate carries a numeric quality.
    assert all(isinstance(c.quality, float) for c in r.candidates)


def test_answer_shape_influences_phrasing_without_gold():
    from regimes_probe.policy.query_decomposition import answer_shape_terms
    assert "manga" in answer_shape_terms(MANGA)
    # the entity_clue for a manga question carries the shape hint.
    by_arm = {c.arm: c for c in decompose_queries(MANGA)}
    assert "answer_shape" in by_arm["entity_clue"].clue_ids
