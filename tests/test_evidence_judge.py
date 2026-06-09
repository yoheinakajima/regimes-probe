"""Level 5f LLM evidence judge: a narrow, cached, replayable candidate/slot/constraint
support-fit function. No live providers/models — every "model" is a local stub.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.evidence_judge import EvidenceJudge, enforce_hard_rules, EvidenceJudgment
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache, build_task_frame


def _slot(sid, name, role, **kw):
    return {"slot_id": sid, "slot_name": name, "slot_role": role, **kw}


def _con(cid, span, terms, applies, disc, **kw):
    return {"constraint_id": cid, "text_span": span, "normalized_terms": terms,
            "applies_to": applies, "discriminative_score": disc, "specificity_score": disc, **kw}


def _frame(payload, q):
    f, meta = build_task_frame("x", q, use_llm=True,
                               parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload),
                                                         model="stub"))
    assert meta.parser_used == "llm", meta.fallback_reason
    return f


def _obs(title, snippet, url, *, auth=0.6, contam=False):
    return SimpleNamespace(title=title, snippet=snippet, url=url, source_authority=auth,
                           failed=False, benchmark_contaminated=contam)


# A person slot with two body-only constraints.
PERSON_Q = "Who is the engineer who studied at Caltech and worked at Bell Labs?"
PERSON = {
    "target_answer_slots": [_slot("D", "engineer", "person", is_target_answer_slot=True,
                                  is_intermediate_slot=False)],
    "latent_slots": [],
    "constraints": [
        _con("c_edu", "studied at Caltech", ["caltech", "studied"], ["D"], 1.0, required=True,
             priority="high", testable_claim="studied at Caltech", evidence_needed="bio",
             how_to_test="read", semantic_label="education"),
        _con("c_emp", "worked at Bell Labs", ["bell", "labs"], ["D"], 1.0, required=True,
             priority="high", testable_claim="employed at Bell Labs", evidence_needed="bio",
             how_to_test="read", semantic_label="employment")],
    "dependency_edges": [], "known_context_terms": []}


def _judge_fn(rules: dict) -> "callable":
    """A stub judge model_fn keyed on which constraint testable_claim is in the prompt."""
    def _fn(prompt: str) -> str:
        for needle, payload in rules.items():
            if needle in prompt:
                return json.dumps(payload)
        return json.dumps({"judgment": "irrelevant", "rationale": "no rule"})
    return _fn


def _frontier_with_judge(frame, model_fn, *, cache=None):
    judge = EvidenceJudge(model_fn=model_fn, cache=cache or ParserCache(), model="stub",
                          enabled=True)
    return CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=judge)), judge


def _sid(f):
    return f.target_answer_slots[0].slot_id


# --------------------------------------------------------------------------- core judgments
def test_full_support_materializes_to_candidate_and_constraint_state():
    f = _frame(PERSON, PERSON_Q)
    fr, _ = _frontier_with_judge(f, _judge_fn({
        "Caltech": {"judgment": "full_support", "quote": "Dana Wu studied at Caltech",
                    "candidate_role_fit": "fits", "source_role_fit": "acceptable",
                    "supported_facets": ["education"], "confidence": 0.9}}))
    sid = _sid(f)
    ev = fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Profile of Dana Wu.",
                                  "https://linkedin.com/in/dana-wu")],
                            source_tool="serper", directed_slot_id=sid,
                            directed_constraint_ids=["c_edu"], proposal_id="p1")
    dana = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    assert "c_edu" in dana.constraints_supported
    assert "c_edu" in ev.supports_constraint_ids
    assert any(e["event_type"] == "evidence_judgment_supports_candidate_constraint"
               for e in fr.events)


def test_partial_support_does_not_resolve_blocking_but_raises_priority():
    f = _frame(PERSON, PERSON_Q)
    fr, _ = _frontier_with_judge(f, _judge_fn({
        "Caltech": {"judgment": "partial_support", "rationale": "mentions Caltech weakly",
                    "confidence": 0.4}}))
    sid = _sid(f)
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Dana Wu, engineer.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    dana = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    assert "c_edu" in dana.constraints_partial          # partial recorded
    assert "c_edu" not in dana.constraints_supported    # but NOT a full support
    assert not fr._confirmable(dana, contaminated=False)   # blocking constraint unresolved
    assert dana.evidence_score > 0                      # partial raises ranking/EIG


def test_contradiction_attaches_and_rejects_candidate():
    f = _frame(PERSON, PERSON_Q)
    fr, _ = _frontier_with_judge(f, _judge_fn({
        "Caltech": {"judgment": "contradiction", "contradicted_facets": ["education"],
                    "rationale": "studied at MIT not Caltech", "confidence": 0.8}}))
    sid = _sid(f)
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Dana Wu studied at MIT.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    dana = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    assert "c_edu" in dana.constraints_contradicted
    assert dana.status == "rejected"
    assert any(e["event_type"] == "evidence_judgment_contradicts_candidate_constraint"
               for e in fr.events)


def test_contaminated_source_cannot_produce_support_even_if_stub_tries():
    # the judge's hard rule downgrades any support on a contaminated source.
    judge = EvidenceJudge(model_fn=lambda _p: json.dumps({"judgment": "full_support",
                                                          "quote": "x"}),
                          cache=ParserCache(), model="stub", enabled=True)
    con = SimpleNamespace(constraint_id="c", text_span="studied at Caltech",
                          testable_claim="studied", normalized_terms=["caltech"], how_to_test="read")
    jd = judge.judge(candidate_text="Dana Wu", candidate_id="cand1", aliases=[], slot_id="D",
                     slot_role="person", slot_descriptor="engineer", constraint=con,
                     source_id="e1", source_title="mirror", source_url="https://hf.co/m",
                     source_domain="hf.co", source_role="benchmark_contaminated",
                     contaminated=True, snippet="Dana Wu studied at Caltech",
                     det_status="irrelevant", det_quote="")
    assert jd.judgment not in ("full_support", "partial_support")
    assert judge.stats()["full_support_from_contaminated_source_count"] == 0


def test_full_support_without_quote_downgraded_to_partial():
    j = EvidenceJudgment(judgment="full_support", quote="")
    j = enforce_hard_rules(j, contaminated=False, source_role="professional_profile",
                           has_quote=False)
    assert j.judgment == "partial_support"


# --------------------------------------------------------------------------- requires_read
def test_requires_read_schedules_read_when_url_available():
    f = _frame(PERSON, PERSON_Q)
    fr, _ = _frontier_with_judge(f, _judge_fn({
        "Caltech": {"judgment": "requires_read", "requires_read_reason": "snippet insufficient",
                    "confidence": 0.5}}))
    sid = _sid(f)
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Dana Wu, engineer.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    dana = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    assert "c_edu" in dana.requires_read_constraint_ids
    fr.generate_frontier_actions()
    sel = fr.select_frontier_action()
    assert sel.action_type == "read_candidate_source"
    assert sel.selected_reason == "evidence_judge_requires_read"
    assert fr.metrics()["requires_read_scheduled"] >= 1


def test_requires_read_records_skipped_when_no_source():
    f = _frame(PERSON, PERSON_Q)
    fr, _ = _frontier_with_judge(f, _judge_fn({
        "Caltech": {"judgment": "requires_read", "requires_read_reason": "no snippet body"}}))
    sid = _sid(f)
    # a result with NO url -> the candidate has no source to read.
    fr.ingest_evidence([_obs("Dana Wu - Engineer", "Dana Wu, engineer.", "")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    fr.generate_frontier_actions()
    assert any(e["event_type"] == "skipped_read_after_requires_read" for e in fr.events)


# --------------------------------------------------------------------------- guardrails
def test_page_chrome_is_irrelevant_and_not_judged():
    f = _frame(PERSON, PERSON_Q)
    fr, judge = _frontier_with_judge(f, _judge_fn({
        "Caltech": {"judgment": "full_support", "quote": "x"}}))  # stub would over-credit
    fr.ingest_evidence([_obs("Login", "Sign in to continue.", "https://x.example/login")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    assert all(len(s.candidates) == 0 for s in fr.slates.values())
    assert judge.stats()["llm_full_support_count"] == 0   # judge never consulted for chrome


def test_candidate_alias_from_judge_resolves_verifier_proposal():
    from regimes_probe.agent.llm_frontier import _proposal_from_dict, validate_proposal
    f = _frame(PERSON, PERSON_Q)
    fr, _ = _frontier_with_judge(f, _judge_fn({
        "Caltech": {"judgment": "full_support", "quote": "Dana Wu studied at Caltech",
                    "candidate_aliases": ["D. Wu"], "confidence": 0.9}}))
    sid = _sid(f)
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Profile.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    dbg = fr.resolve_candidate("D. Wu", slot_id=sid)
    assert dbg["found"]
    p = _proposal_from_dict({"proposal_id": "v1", "action_type": "verify_candidate_constraint",
                             "target_slot_id": sid, "candidate_id": "D. Wu",
                             "constraint_ids": ["c_emp"],
                             "proposed_query": "D. Wu Bell Labs"}, 0)
    ok, _ = validate_proposal(p, f, fr, failed_norms=set(),
                              available_tools=["serper_search"], remaining_budget=3)
    assert ok and p.candidate_id in fr.candidates_by_id


def test_replay_uses_cached_judgment_with_zero_model_calls():
    cache = ParserCache()
    con = SimpleNamespace(constraint_id="c", text_span="studied at Caltech",
                          testable_claim="studied", normalized_terms=["caltech"], how_to_test="read")
    kw = dict(candidate_text="Dana Wu", candidate_id="cand1", aliases=[], slot_id="D",
              slot_role="person", slot_descriptor="engineer", constraint=con, source_id="e1",
              source_title="Dana Wu - LinkedIn", source_url="https://linkedin.com/in/dana-wu",
              source_domain="linkedin.com", source_role="professional_profile",
              contaminated=False, snippet="Dana Wu studied at Caltech.",
              det_status="insufficient", det_quote="Caltech")
    armed = EvidenceJudge(model_fn=lambda _p: json.dumps({"judgment": "full_support",
                                                          "quote": "studied at Caltech"}),
                          cache=cache, model="stub", enabled=True)
    j1 = armed.judge(**kw)
    assert j1.judgment == "full_support" and armed.calls == 1
    replay = EvidenceJudge(model_fn=None, cache=cache, model="stub", replay_only=True, enabled=True)
    j2 = replay.judge(**kw)
    assert j2.judgment == "full_support" and replay.calls == 0
    assert replay.stats()["llm_evidence_judge_cache_hits"] >= 1


def test_one_read_emits_multiple_constraint_supports():
    f = _frame(PERSON, PERSON_Q)
    fr, _ = _frontier_with_judge(f, _judge_fn({
        "studied at Caltech": {"judgment": "full_support", "quote": "studied at Caltech",
                               "confidence": 0.9},
        "employed at Bell Labs": {"judgment": "full_support", "quote": "worked at Bell Labs",
                                  "confidence": 0.9}}))
    sid = _sid(f)
    fr.ingest_evidence([_obs("Dana Wu - Engineer", "Dana Wu studied at Caltech and worked at Bell Labs.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="firecrawl_scrape", read_depth=2, directed_slot_id=sid,
                       directed_constraint_ids=["c_edu", "c_emp"], proposal_id="p1")
    dana = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    assert {"c_edu", "c_emp"} <= set(dana.constraints_supported)


def test_partial_only_does_not_make_candidate_answerable():
    f = _frame(PERSON, PERSON_Q)
    fr, _ = _frontier_with_judge(f, _judge_fn({
        "Caltech": {"judgment": "partial_support"}, "Bell": {"judgment": "partial_support"}}))
    sid = _sid(f)
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Dana Wu engineer.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu", "c_emp"], proposal_id="p1")
    best = fr.best_hypothesis()
    assert best is None or not fr._answerable(best)     # gate stays strict


def test_support_consistency_no_drop_with_judge():
    f = _frame(PERSON, PERSON_Q)
    fr, _ = _frontier_with_judge(f, _judge_fn({
        "Caltech": {"judgment": "full_support", "quote": "Dana Wu studied at Caltech"}}))
    sid = _sid(f)
    ev = fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "profile",
                                  "https://linkedin.com/in/dana-wu")],
                            source_tool="serper", directed_slot_id=sid,
                            directed_constraint_ids=["c_edu"], proposal_id="p1")
    assert ev.progress_components["selected_constraint_support_count"] >= 1
    assert fr.metrics()["support_dropped_count"] == 0


# --------------------------------------------------------------------------- agent-level
def test_easy_question_skips_judge_entirely():
    from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
    from regimes_probe.agent.evidence_judge import EvidenceJudge
    from regimes_probe.datasets.base import Item
    from regimes_probe.policy.contextual_bandit import BanditParams
    from regimes_probe.policy.memory import PolicyMemory
    from regimes_probe.tools.fake import FakePageFetch
    from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
    from decimal import Decimal

    class _P(SearchProvider):
        name = "generic_web_search"; cost_per_call = Decimal("0.002"); deterministic = True
        def available(self): return True
        def search(self, q, *, limit=5, **o):
            return SearchResponse(provider=self.name, query=q, results=(
                SearchResult(title="Paris", url="https://x/p", snippet="capital",
                             source_authority=0.6, rank=0, extra={"item_id": "x"}),),
                cost=self.cost_per_call)
    judge = EvidenceJudge(model_fn=lambda _p: json.dumps({"judgment": "full_support"}),
                          cache=ParserCache(), model="stub", enabled=True)
    agent = EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "page_fetch"], enable_task_frame=True,
        auto_epistemic_mode=True, enable_frontier_controller=True,
        enable_llm_evidence_judge=True), evidence_judge=judge)
    item = Item(id="x", answer="Paris", question="What is the capital of France?")
    agent.attempt(item, PolicyMemory(BanditParams()),
                  {"generic_web_search": _P(), "page_fetch": FakePageFetch([])},
                  budget=2, explore=False, attempt_id="t")
    assert judge.calls == 0          # easy question never engages the judge


def test_policy_memory_answer_free_with_judge():
    from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
    from regimes_probe.agent.evidence_judge import EvidenceJudge
    from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
    from regimes_probe.eval.harness import experience_phase
    from regimes_probe.policy.contextual_bandit import BanditParams
    from regimes_probe.policy.memory import PolicyMemory
    from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
    from regimes_probe.tools.fake import build_fake_providers, load_corpus
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    items = SyntheticBrowseAdapter().load()[:6]
    providers = build_fake_providers(load_corpus(root / "fixtures" / "fake_search_corpus.json"),
                                     ["generic_web_search", "news_search"])
    judge = EvidenceJudge(model_fn=lambda _p: json.dumps({"judgment": "partial_support"}),
                          cache=ParserCache(), model="stub", enabled=True)
    agent = EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "news_search", "page_fetch"],
        enable_task_frame=True, force_task_frame=True, enable_frontier_controller=True,
        enable_llm_evidence_judge=True), evidence_judge=judge)
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=3, passes=2)
    assert_no_answer_leakage(mem.snapshot().to_dict(), "snapshot")
