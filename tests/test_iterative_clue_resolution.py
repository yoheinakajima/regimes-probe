"""Iterative clue resolution: staged candidate-entity search (no network)."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from regimes_probe.agent.clue_resolution import (
    CandidateEntity, compose_followup_query, extract_candidate_entities)
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.base import Item
from regimes_probe.eval.harness import run_condition
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
from regimes_probe.tools.fake import FakePageFetch


def _obs(title, snippet, url, authority=0.6):
    return SimpleNamespace(title=title, snippet=snippet, url=url,
                           source_authority=authority, failed=False)


# --------------------------------------------------- candidate extraction
def test_candidate_entities_extracted_and_scored():
    obs = [
        _obs("Edrin Vael — biography", "Edrin Vael survived a road accident.",
             "https://example.org/edrin-vael", 0.7),
        _obs("Profile of Edrin Vael", "Edrin Vael taught at Calderwood University.",
             "https://news.example/edrin", 0.5),
    ]
    cands = extract_candidate_entities(obs, ["road", "accident", "university"])
    texts = [c.text for c in cands]
    assert "Edrin Vael" in texts
    top = cands[0]
    assert top.text == "Edrin Vael"          # repeated across results -> ranked first
    assert top.frequency >= 2 and top.proximity == 1.0


def test_generic_entities_are_filtered():
    obs = [
        _obs("Wikipedia", "From Wikipedia, the free encyclopedia.", "https://wikipedia.org/x"),
        _obs("Facebook", "Log in to Facebook.", "https://facebook.com/x"),
        _obs("YouTube video", "Watch on YouTube.", "https://youtube.com/watch"),
        _obs("Reddit thread", "posted to Reddit", "https://reddit.com/r/x"),
    ]
    cands = extract_candidate_entities(obs, ["anything"])
    texts = {c.text.lower() for c in cands}
    for g in ("wikipedia", "facebook", "youtube", "reddit"):
        assert g not in texts


def test_compose_followup_combines_entity_with_next_clue():
    ce = CandidateEntity(text="Edrin Vael", frequency=3, source_authority=0.7, score=4.0)
    fq = compose_followup_query(ce, clue_spans=["private university", "road accident"],
                                used_clue_spans=set(), answer_shape=["born"])
    assert fq is not None
    assert '"Edrin Vael"' in fq.query
    assert "private university" in fq.query          # next unused clue span
    assert fq.query_arm == "candidate_entity_followup"
    # when no clue spans remain, fall back to answer-shape.
    fq2 = compose_followup_query(ce, clue_spans=[], used_clue_spans=set(),
                                 answer_shape=["born"])
    assert fq2.query_arm == "answer_shape_followup" and "born" in fq2.query


# --------------------------------------------------- end-to-end staged search
class _EntityProvider(SearchProvider):
    name = "generic_web_search"
    cost_per_call = Decimal("0.002")
    deterministic = True

    def available(self) -> bool:
        return True

    def search(self, q, *, limit=5, **o):
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="Edrin Vael profile", url="https://calderwood.edu/edrin-vael",
                         snippet="Edrin Vael taught at a private university.",
                         source_authority=0.7, rank=0),), cost=self.cost_per_call)


def _iter_agent():
    return EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "page_fetch"],
        enable_query_decomposition=True, enable_iterative_clue_resolution=True))


def _item():
    return Item(id="x", answer="Edrin Vael",
                question=("Who is the African author who survived a serious road accident "
                          "and later taught at a private university?"))


def test_followup_query_contains_extracted_entity():
    providers = {"generic_web_search": _EntityProvider(), "page_fetch": FakePageFetch([])}
    tr = _iter_agent().attempt(_item(), PolicyMemory(BanditParams()), providers,
                               budget=3, explore=False, attempt_id="t")
    stages = [c.stage for c in tr.calls]
    assert max(stages) >= 2                          # staged beyond the first query
    followups = [c for c in tr.calls if c.stage >= 2]
    assert followups
    fc = followups[0]
    assert '"Edrin Vael"' in fc.query                # follow-up anchors on the entity
    assert fc.query_arm in ("candidate_entity_followup", "answer_shape_followup")
    assert fc.selected_candidate == "Edrin Vael"
    assert fc.parent_query_id is not None


def test_budget_includes_every_followup():
    providers = {"generic_web_search": _EntityProvider(), "page_fetch": FakePageFetch([])}
    for budget in (1, 2, 4):
        tr = _iter_agent().attempt(_item(), PolicyMemory(BanditParams()), providers,
                                   budget=budget, explore=False, attempt_id="t")
        assert len(tr.calls) <= budget               # follow-ups count against budget


def test_debug_artifact_shows_stage_chain():
    from regimes_probe.eval.debug import build_debug_record
    from regimes_probe.eval.grader import grade
    from regimes_probe.eval.reward import compute_rewards
    providers = {"generic_web_search": _EntityProvider(), "page_fetch": FakePageFetch([])}
    item = _item()
    tr = _iter_agent().attempt(item, PolicyMemory(BanditParams()), providers,
                               budget=3, explore=False, attempt_id="t")
    g = grade(item, tr.final_answer)
    rr = compute_rewards(tr, correct=g.correct, gold_norms=[], weights=RewardWeights.full(),
                         freshness_sensitive=False)
    rec = build_debug_record(item=item, trace=tr, grade=g, reward=rr,
                             condition="no_memory_search", budget=3).to_dict()
    assert rec["stage_depth_used"] >= 2
    assert rec["followup_query_count"] >= 1
    stages = [c["stage"] for c in rec["calls"]]
    assert stages == sorted(stages) and max(stages) >= 2
    fc = next(c for c in rec["calls"] if c["stage"] >= 2)
    assert fc["selected_candidate"] == "Edrin Vael"
    assert fc["candidate_entities"]                  # entities recorded for the stage


def test_iterative_metrics_present(items, providers):
    agent = EpistemicAgent(AgentConfig(available_tools=["generic_web_search", "news_search",
                                       "official_domain_search", "brave_search", "page_fetch"],
                                       enable_query_decomposition=True,
                                       enable_iterative_clue_resolution=True))
    res = run_condition(items[:3], agent, providers, PolicyMemory(BanditParams()),
                        condition="no_memory_search", budget=3, weights=RewardWeights.full())
    from regimes_probe.eval.metrics import compute_metrics
    m = compute_metrics(res.outcomes)
    for k in ("mean_candidate_entity_count", "mean_followup_query_count",
              "evidence_improved_after_followup_rate", "mean_stage_depth_used"):
        assert k in m
