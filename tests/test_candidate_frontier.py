"""ActiveGraph-native candidate-slate / frontier layer (no model/provider calls)."""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.base import Item
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
from regimes_probe.tools.fake import FakePageFetch


def _slot(sid, name, role, **kw):
    return {"slot_id": sid, "slot_name": name, "slot_role": role, **kw}


def _con(cid, span, terms, applies, **kw):
    return {"constraint_id": cid, "text_span": span, "normalized_terms": terms,
            "applies_to": applies, **kw}


def _obs(title, snippet, url="https://x.example/p", auth=0.7, contam=False):
    return SimpleNamespace(title=title, snippet=snippet, url=url, source_authority=auth,
                           failed=False, benchmark_contaminated=contam)


def _build(payload, q):
    f, meta = build_task_frame("x", q, use_llm=True,
                               parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload),
                                                         model="stub"))
    assert meta.parser_used == "llm", meta.fallback_reason
    return f, {v: k for k, v in meta.id_mapping.items()}   # internal_id -> raw_id


def _sid(rawmap, raw):
    return next(internal for internal, r in rawmap.items() if r == raw)


# restaurant frame: 5 distinct slots
RESTAURANT_Q = ("There is a Mexican restaurant in New Mexico near a hotel opened in 1955 "
                "and a museum founded in 2005, established by a chef born in which year?")
RESTAURANT = {
    "target_answer_slots": [_slot("byear", "birth year", "date_or_time",
                                  is_target_answer_slot=True, is_intermediate_slot=False,
                                  depends_on=["founder"])],
    "latent_slots": [
        _slot("rest", "Mexican restaurant in NM", "organization", depends_on=["hotel", "museum"]),
        _slot("hotel", "hotel opened in 1955", "organization"),
        _slot("museum", "museum founded in 2005", "organization"),
        _slot("founder", "chef founder", "person", depends_on=["rest"])],
    "constraints": [
        _con("c_rest", "Mexican restaurant in New Mexico",
             ["mexican", "restaurant", "casa", "verde"], ["rest"], required=True,
             priority="high", testable_claim="x", how_to_test="read"),
        _con("c_hotel", "hotel opened in 1955", ["hotel", "opened", "1955"], ["hotel"],
             required=True, priority="high", testable_claim="x", how_to_test="read"),
        _con("c_year", "chef born in which year", ["born", "year", "chef"], ["byear", "founder"])],
    "dependency_edges": [["rest", "founder"], ["founder", "byear"]],
    "known_context_terms": ["New Mexico", "1955", "2005"]}

TV_Q = "Which TV series starred an actor born in Tennessee and an actor who won a Gracie Award?"
TV = {
    "target_answer_slots": [_slot("series", "TV series", "title_or_work",
                                  is_target_answer_slot=True, is_intermediate_slot=False,
                                  depends_on=["a1", "a2"])],
    "latent_slots": [_slot("a1", "actor born in Tennessee", "person"),
                     _slot("a2", "actor who won a Gracie Award", "person")],
    "constraints": [_con("c_cast", "TV series starred actors",
                         ["series", "starred", "actors"], ["series", "a1"], required=True,
                         priority="high", testable_claim="x", how_to_test="read")],
    "dependency_edges": [["a1", "series"], ["a2", "series"]],
    "known_context_terms": ["Tennessee", "Gracie Award"]}


# ---------------------------------------------------------------- slate creation
def test_slate_created_per_unresolved_slot():
    f, _ = _build(RESTAURANT, RESTAURANT_Q)
    fr = CandidateFrontier(f)
    assert len(fr.slates) == len(f.all_slots) == 5
    assert any(e["event_type"] == "candidate_slate.created" for e in fr.events)


def test_restaurant_creates_five_role_slates():
    f, _ = _build(RESTAURANT, RESTAURANT_Q)
    fr = CandidateFrontier(f)
    roles = sorted(s.slot_role for s in fr.slates.values())
    assert roles == ["date_or_time", "organization", "organization", "organization", "person"]


def test_tv_series_keeps_separate_series_and_actor_slates():
    f, rm = _build(TV, TV_Q)
    fr = CandidateFrontier(f)
    assert _sid(rm, "series") in fr.slates
    assert _sid(rm, "a1") in fr.slates and _sid(rm, "a2") in fr.slates
    assert fr.slates[_sid(rm, "series")].slot_role == "title_or_work"
    assert fr.slates[_sid(rm, "a1")].slot_role == "person"


# ---------------------------------------------------------------- keep / reject
def test_restaurant_candidate_active_while_founder_unresolved():
    f, rm = _build(RESTAURANT, RESTAURANT_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Casa Verde restaurant",
                             "Casa Verde restaurant is a Mexican restaurant in New Mexico.")],
                       source_tool="generic_web_search")
    rest = fr.slates[_sid(rm, "rest")]
    founder = fr.slates[_sid(rm, "founder")]
    assert any(c.candidate_text == "Casa Verde" and c.status in ("active", "confirmed")
               for c in rest.candidates.values())
    assert not founder.candidates                       # founder still unresolved


def test_candidate_rejected_on_opening_year_contradiction():
    f, rm = _build(RESTAURANT, RESTAURANT_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Grand Hotel",
                             "Grand Hotel is a hotel opened in 1972 in New Mexico.")],
                       source_tool="news_search")
    hotel = fr.slates[_sid(rm, "hotel")]
    gh = next(c for c in hotel.candidates.values() if c.candidate_text == "Grand Hotel")
    assert gh.status == "rejected" and gh.status_reason == "contradicted_constraint"
    assert any(e["event_type"] == "candidate.rejected" for e in fr.events)


def test_known_context_not_promoted_to_candidate():
    f, rm = _build(RESTAURANT, RESTAURANT_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Casa Verde restaurant",
                             "Casa Verde restaurant is a Mexican restaurant in New Mexico.")],
                       source_tool="generic_web_search")
    # "New Mexico" is a known constant (location) — never a candidate in an org slate.
    for slate in fr.slates.values():
        assert all(c.candidate_text != "New Mexico" for c in slate.candidates.values())


def test_candidate_expanded_when_upstream_confirmed_unlocks_dependent():
    f, rm = _build(RESTAURANT, RESTAURANT_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Casa Verde restaurant",
                             "Casa Verde restaurant is a Mexican restaurant in New Mexico.")],
                       source_tool="generic_web_search")
    rest = fr.slates[_sid(rm, "rest")]
    assert any(c.status == "confirmed" for c in rest.candidates.values())
    founder_sid = _sid(rm, "founder")
    assert any(a.action_type == "expand_candidate_to_dependent_slot"
               and a.target_slot_id == founder_sid for a in fr.frontier_actions)


# ---------------------------------------------------------------- frontier scheduler
def test_scheduler_prefers_blocking_constraint_over_read():
    f, rm = _build(RESTAURANT, RESTAURANT_Q)
    fr = CandidateFrontier(f)
    # an active (unconfirmed) restaurant candidate with an unresolved blocking constraint
    fr.ingest_evidence([_obs("Casa Hotel", "Casa Hotel is a hotel.", auth=0.6)],
                       source_tool="generic_web_search")
    sel = fr.select_frontier_action(budget_remaining=3)
    assert sel is not None
    assert sel.action_type in ("verify_candidate_constraint", "generate_candidates_for_slot")
    # the selected action out-scores any read_candidate_source action.
    reads = [a for a in fr.frontier_actions if a.action_type == "read_candidate_source"]
    if reads:
        net = lambda a: a.expected_information_gain - 0.5 * a.estimated_cost
        assert net(sel) >= max(net(r) for r in reads)


def test_read_action_requires_candidate_slot_constraint_triple():
    f, rm = _build(RESTAURANT, RESTAURANT_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Casa Verde restaurant",
                             "Casa Verde restaurant in New Mexico.")],
                       source_tool="generic_web_search")
    # a page mentioning the candidate -> read selected with a triple.
    good = fr.frontier_read_value(_obs("Casa Verde history",
                                       "Casa Verde restaurant was founded by a chef."))
    assert good.selected and good.candidate_id and good.target_slot_id and good.constraint_ids
    # a page mentioning no candidate -> rejected (no triple).
    bad = fr.frontier_read_value(_obs("Unrelated blog", "A recipe for tacos."))
    assert not bad.selected and bad.rejected_reason == "no_candidate_slot_constraint_affected"


def test_duplicate_candidates_merge_and_preserve_provenance():
    f, rm = _build(RESTAURANT, RESTAURANT_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Casa Verde restaurant", "Casa Verde restaurant in New Mexico.",
                             url="https://a.example/x")], source_tool="generic_web_search")
    fr.ingest_evidence([_obs("Casa Verde restaurant", "Casa Verde restaurant in New Mexico.",
                             url="https://b.example/y")], source_tool="news_search")
    rest = fr.slates[_sid(rm, "rest")]
    cv = [c for c in rest.candidates.values() if c.candidate_text == "Casa Verde"]
    assert len(cv) == 1                                 # one candidate, deduped
    assert {"a.example", "b.example"} <= set(cv[0].source_domains)   # both providers kept
    assert any(e["event_type"] == "candidate.merged" for e in fr.events)


# ---------------------------------------------------------------- projection
def _frontier_projection(payload, q, *, evidences):
    from regimes_probe.eval.projection import build_graph_projection
    f, _ = _build(payload, q)
    fr = CandidateFrontier(f, attempt_id="t", item_id="x")
    for e in evidences:
        fr.ingest_evidence([e], source_tool="generic_web_search")
    rec = {"condition": "no_memory_search", "budget": 3, "item_id": "x",
           "task_frame": f.to_dict(), "candidate_frontier": fr.to_debug(),
           "reward": {}, "grade": {}}
    proj = build_graph_projection("rid", runs=[], debug_records=[rec], eligibility_verdict={})
    return proj, fr


def test_candidate_rejection_appears_in_projection():
    proj, _ = _frontier_projection(
        RESTAURANT, RESTAURANT_Q,
        evidences=[_obs("Grand Hotel", "Grand Hotel is a hotel opened in 1972.")])
    types = {o["type"] for o in proj["objects"]}
    rels = {r["type"] for r in proj["relations"]}
    assert "candidate_slate" in types and "slot_candidate" in types
    assert "candidate_rejection" in types
    assert "candidate_rejected_by_evidence" in rels and "slate_for_slot" in rels


def test_candidate_promotion_appears_in_projection():
    proj, _ = _frontier_projection(
        RESTAURANT, RESTAURANT_Q,
        evidences=[_obs("Casa Verde restaurant", "Casa Verde restaurant in New Mexico.")])
    types = {o["type"] for o in proj["objects"]}
    assert "candidate_promotion" in types and "frontier_action" in types


def test_replay_projection_is_deterministic():
    proj1, fr = _frontier_projection(
        RESTAURANT, RESTAURANT_Q,
        evidences=[_obs("Casa Verde restaurant", "Casa Verde restaurant in New Mexico.")])
    # re-project the SAME frontier debug -> identical objects/relations.
    from regimes_probe.eval.projection import build_graph_projection
    rec = {"condition": "no_memory_search", "budget": 3, "item_id": "x",
           "task_frame": fr.frame.to_dict(), "candidate_frontier": fr.to_debug(),
           "reward": {}, "grade": {}}
    proj2 = build_graph_projection("rid", runs=[], debug_records=[rec], eligibility_verdict={})
    assert proj1["objects"] == proj2["objects"]
    assert proj1["relations"] == proj2["relations"]


# ---------------------------------------------------------------- easy skip + e2e
class _FrameProvider(SearchProvider):
    name = "generic_web_search"
    cost_per_call = Decimal("0.002")
    deterministic = True

    def available(self): return True

    def search(self, q, *, limit=5, **o):
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="x", url="https://gov.example/x", snippet="x",
                         source_authority=0.5, rank=0, extra={"item_id": "x"}),),
            cost=self.cost_per_call)


def test_easy_direct_question_skips_candidate_slates():
    item = Item(id="x", answer="Paris", question="What is the capital of France?")
    agent = EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "page_fetch"],
        enable_task_frame=True, auto_epistemic_mode=True))
    tr = agent.attempt(item, PolicyMemory(BanditParams()),
                       {"generic_web_search": _FrameProvider(), "page_fetch": FakePageFetch([])},
                       budget=2, explore=False, attempt_id="t")
    assert tr.epistemic_mode["selected_epistemic_mode"] in ("direct_answer_possible", "simple_lookup")
    assert tr.candidate_frontier.get("skipped") is True
    assert "epistemic_mode=" in tr.candidate_frontier.get("skipped_candidate_slate_reason", "")


def test_policy_memory_stays_answer_free_with_frontier():
    from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
    from regimes_probe.eval.harness import experience_phase
    from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
    from regimes_probe.tools.fake import build_fake_providers, load_corpus
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    items = SyntheticBrowseAdapter().load()[:6]
    providers = build_fake_providers(load_corpus(root / "fixtures" / "fake_search_corpus.json"),
                                     ["generic_web_search", "news_search"])
    agent = EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "news_search", "page_fetch"],
        enable_task_frame=True, force_task_frame=True))
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=3, passes=2)
    assert_no_answer_leakage(mem.snapshot().to_dict(), "snapshot")
