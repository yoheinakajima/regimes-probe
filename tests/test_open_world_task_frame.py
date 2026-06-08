"""Open-world semantic constraints + affordances + escalation (no model/provider calls)."""

from __future__ import annotations

import json
from decimal import Decimal

from regimes_probe.agent.action_planner import evaluate_answer_support
from regimes_probe.agent.affordances import AFFORDANCES, derive_affordances, normalize_facets
from regimes_probe.agent.epistemic_mode import decide_epistemic_mode
from regimes_probe.agent.hypothesis_table import (
    Candidate, EvidenceRecord, Hypothesis, HypothesisTable)
from regimes_probe.agent.llm_task_frame import (
    LLMTaskFrameParser, build_task_frame, parser_warnings, validate_payload)
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.agent.task_frame import Constraint, Slot, TaskFrame, enrich_constraint
from regimes_probe.datasets.base import Item
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
from regimes_probe.tools.fake import FakePageFetch


# --------------------------------------------------------------- payload helpers
def _slot(sid, name, role, *, target=False, depends_on=None):
    return {"slot_id": sid, "slot_name": name, "slot_role": role,
            "is_target_answer_slot": target, "is_intermediate_slot": not target,
            "depends_on": depends_on or [], "expected_evidence_type": role}


def _con(cid, span, label, facets, applies, **kw):
    d = {"constraint_id": cid, "text_span": span, "semantic_label": label,
         "semantic_facets": facets, "applies_to": applies}
    d.update(kw)
    return d


def _payload(targets, latents, constraints, *, known=None, edges=None):
    return {"target_answer_slots": targets, "latent_slots": latents,
            "constraints": constraints, "dependency_edges": edges or [],
            "known_context_terms": known or [], "parse_quality": 0.9}


def _parser(payload):
    return LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload), model="stub")


WHO_Q = "Which university employed the scientist who authored the 2021 WHO air-quality report?"


def _rich_payload():
    return _payload(
        [_slot("s0", "university", "organization", target=True, depends_on=["s1"])],
        [_slot("s1", "scientist", "person")],
        [_con("c0", "employed the scientist", "employment_relation",
              ["employment", "relation"], "s0",  # NOTE: scalar applies_to
              required=True, priority="high", testable_claim="University X employed Y",
              evidence_needed="affiliation page", how_to_test="read faculty page",
              supports_answer_slot_ids=["s0"]),
         _con("c1", "authored the 2021 report", "authorship + educational background",
              ["authorship", "biographical"], ["s1"],
              required=True, priority="high", testable_claim="Y authored the report",
              how_to_test="read the report front matter"),
         _con("c2", "within 50 miles by 2019", "distance_and_temporal_attribute",
              ["distance", "temporal"], ["s1"], testable_claim="x")],
        known=["WHO", "2021"], edges=[["s1", "s0"]])


# --------------------------------------------------- open-world acceptance
def test_unknown_semantic_label_accepted_with_valid_affordances():
    p = _rich_payload()
    assert validate_payload(p, WHO_Q) == []                  # not rejected for labels
    frame, meta = build_task_frame("x", WHO_Q, use_llm=True, parser=_parser(p))
    assert meta.parser_used == "llm" and not meta.fallback_reason
    c0 = next(c for c in frame.constraints if c.constraint_id == "c0")
    assert c0.semantic_label == "employment_relation"
    # affordances are the closed planner vocabulary, derived from the open semantics.
    assert set(c0.affordances) <= set(AFFORDANCES)
    assert "can_search" in c0.affordances and "can_bind_slot" in c0.affordances


def test_employment_relation_label_does_not_fallback():
    p = _rich_payload()
    _, meta = build_task_frame("x", WHO_Q, use_llm=True, parser=_parser(p))
    assert meta.parser_used == "llm", f"fell back: {meta.fallback_reason}"


def test_distance_and_temporal_attribute_label_does_not_fallback():
    p = _rich_payload()
    frame, meta = build_task_frame("x", WHO_Q, use_llm=True, parser=_parser(p))
    assert meta.parser_used == "llm"
    c2 = next(c for c in frame.constraints if c.constraint_id == "c2")
    assert "can_compare" in c2.affordances        # distance/temporal -> compare affordance


def test_unfamiliar_facets_preserved_not_rejected():
    p = _rich_payload()
    frame, _ = build_task_frame("x", WHO_Q, use_llm=True, parser=_parser(p))
    c0 = next(c for c in frame.constraints if c.constraint_id == "c0")
    assert "employment" in c0.semantic_facets      # novel facet preserved verbatim
    warns = parser_warnings(p, WHO_Q)
    assert any(w.startswith("novel_facet:employment") for w in warns)


def test_applies_to_scalar_normalized_to_list():
    p = _rich_payload()                            # c0 has applies_to="s0" (scalar)
    frame, _ = build_task_frame("x", WHO_Q, use_llm=True, parser=_parser(p))
    c0 = next(c for c in frame.constraints if c.constraint_id == "c0")
    assert c0.applies_to == ["s0"]


def test_malformed_slot_reference_falls_back():
    p = _payload([_slot("s0", "university", "organization", target=True)], [],
                 [_con("c0", "x", "relation", ["relation"], ["s9"], testable_claim="t")])
    frame, meta = build_task_frame("x", WHO_Q, use_llm=True, parser=_parser(p))
    assert meta.parser_used == "deterministic"
    assert any("constraint_refs_unknown_slot" in e for e in meta.validation_errors)


# --------------------------------------------------- affordance derivation unit
def test_derive_affordances_matches_examples():
    # "employment_relation" -> search/verify/bind + support (applies to target s0)
    aff = derive_affordances(facets=["employment", "relation"], applies_to=["s0"],
                             target_slot_ids={"s0"}, intermediate_slot_ids=set(),
                             supports_answer_slot_ids=["s0"])
    assert {"can_search", "can_verify", "can_bind_slot", "can_support_answer"} <= set(aff)
    # "source_reference" -> can_read
    aff2 = derive_affordances(facets=["source"], applies_to=["s1"], target_slot_ids={"s0"},
                              intermediate_slot_ids={"s1"})
    assert "can_read" in aff2
    # distance/temporal -> can_compare
    aff3 = derive_affordances(facets=["distance", "temporal"], applies_to=["s1"],
                              target_slot_ids={"s0"}, intermediate_slot_ids={"s1"})
    assert "can_compare" in aff3


# --------------------------------------------------- answer-support = evidence path
def _gate_frame(resolved: bool):
    f = TaskFrame(item_id="x")
    f.target_answer_slots = [Slot("s0", "university", "organization", is_target_answer_slot=True)]
    c = Constraint("c0", "employed the scientist", ["employed", "scientist"],
                   specificity_score=2.0, discriminative_score=2.0,
                   status=("resolved" if resolved else "unresolved"))
    enrich_constraint(c, {"s0"}, set())            # gives it can_block_answer affordance
    f.constraints = [c]
    return f


def test_answer_support_requires_evidence_path_not_slot_assignment():
    # slot bound, blocking constraint UNRESOLVED, no evidence -> not supported.
    f = _gate_frame(resolved=False)
    t = HypothesisTable(f)
    t._by_id["c1"] = Candidate("c1", "MIT", "mit", "organization")
    t.hypotheses = {"h1": Hypothesis("h1", slot_assignments={"s0": "c1"},
                                     evidence_ids=[], support_score=0.0)}
    t.evidence = []
    assert evaluate_answer_support(f, t).supported is False

    # resolve the blocking constraint + add clean evidence on the path + support>=2.
    f2 = _gate_frame(resolved=True)
    t2 = HypothesisTable(f2)
    t2._by_id["c1"] = Candidate("c1", "MIT", "mit", "organization")
    t2.evidence = [EvidenceRecord("e1", "generic_web_search", supports_slot_ids=["s0"],
                                  contamination_score=0.0)]
    t2.hypotheses = {"h1": Hypothesis("h1", slot_assignments={"s0": "c1"},
                                      evidence_ids=["e1"], support_score=2.0,
                                      constraint_status={"c0": "supported"})}
    res = evaluate_answer_support(f2, t2)
    assert res.supported is True, res.missing_support_reasons


# --------------------------------------------------- escalation controller
def test_easy_question_routes_to_direct_or_simple():
    d = decide_epistemic_mode("What is the capital of France?", budget=3)
    assert d.selected_epistemic_mode in ("direct_answer_possible", "simple_lookup")
    d2 = decide_epistemic_mode("Who won the 2026 Eurovision Song Contest?", budget=3)
    assert d2.selected_epistemic_mode in ("direct_answer_possible", "simple_lookup")


def test_multihop_clue_dense_routes_to_task_frame():
    d = decide_epistemic_mode(
        "Which TV series featured an actor who immigrated from the Caribbean and "
        "won an award in 2019?", budget=3)
    assert d.selected_epistemic_mode == "task_frame_required"


def test_force_task_frame_overrides_controller():
    d = decide_epistemic_mode("What is the capital of France?", budget=3,
                              force_task_frame=True)
    assert d.selected_epistemic_mode == "task_frame_required"


# --------------------------------------------------- graph projection
class _FrameProvider(SearchProvider):
    name = "generic_web_search"
    cost_per_call = Decimal("0.002")
    deterministic = True

    def available(self): return True

    def search(self, q, *, limit=5, **o):
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="A university — affiliation", url="https://gov.example/u",
                         snippet="A university employed a scientist.",
                         source_authority=0.7, rank=0, extra={"item_id": "x"}),),
            cost=self.cost_per_call)


def test_graph_projection_includes_semantic_constraints_and_affordance_edges():
    from regimes_probe.eval.debug import build_debug_record
    from regimes_probe.eval.grader import grade
    from regimes_probe.eval.projection import build_graph_projection
    from regimes_probe.eval.reward import RewardWeights, compute_rewards
    item = Item(id="x", answer="Some University", question=WHO_Q)
    providers = {"generic_web_search": _FrameProvider(), "page_fetch": FakePageFetch([])}
    agent = EpistemicAgent(
        AgentConfig(available_tools=["generic_web_search", "page_fetch"],
                    enable_task_frame=True, enable_llm_task_frame_parser=True,
                    verification=VerificationConfig(min_support=2)),
        task_frame_parser=_parser(_rich_payload()))
    tr = agent.attempt(item, PolicyMemory(BanditParams()), providers,
                       budget=3, explore=False, attempt_id="t")
    rec = build_debug_record(item=item, trace=tr, grade=grade(item, tr.final_answer),
                             reward=compute_rewards(tr, correct=False, gold_norms=[],
                                                    weights=RewardWeights.full(),
                                                    freshness_sensitive=False),
                             condition="no_memory_search", budget=3)
    proj = build_graph_projection("rid", runs=[], debug_records=[rec],
                                  eligibility_verdict={})
    types = {o["type"] for o in proj["objects"]}
    rels = {r["type"] for r in proj["relations"]}
    assert "semantic_constraint" in types
    assert "operational_affordance" in types and "constraint_facet" in types
    assert "epistemic_mode_decision" in types
    assert "constraint_has_affordance" in rels and "constraint_has_facet" in rels
    assert "epistemic_mode_for_attempt" in rels


def test_policy_memory_remains_answer_free_with_open_world_frames():
    from regimes_probe.eval.harness import experience_phase
    from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
    from regimes_probe.tools.fake import build_fake_providers, load_corpus
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    items = SyntheticBrowseAdapter().load()[:6]
    providers = build_fake_providers(load_corpus(root / "fixtures" / "fake_search_corpus.json"),
                                     ["generic_web_search", "news_search"])
    agent = EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "news_search", "page_fetch"],
        enable_task_frame=True, auto_epistemic_mode=True))
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=3, passes=2)
    assert_no_answer_leakage(mem.snapshot().to_dict(), "snapshot")
