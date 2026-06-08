"""Typed candidate-hypothesis policy: role typing, role-compatible scoring,
evidence-progress gate, anti-sticky beam (no providers, no network)."""

from __future__ import annotations

from types import SimpleNamespace

from regimes_probe.agent.clue_resolution import (
    CandidateHypothesis, HypothesisBeam, build_hypotheses, classify_entity_role,
    compose_followup_from_hypothesis, infer_target_roles)
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.eval.harness import experience_phase
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.policy_fragment import assert_no_answer_leakage


def _obs(title, snippet, url, authority=0.6, contaminated=False):
    return SimpleNamespace(title=title, snippet=snippet, url=url, source_authority=authority,
                           failed=False, benchmark_contaminated=contaminated, tool="generic_web_search")


def _by_text(hyps, text):
    return next(h for h in hyps if h.candidate_text.lower() == text.lower())


# ----------------------------------------------------- role typing
def test_role_typing_is_generic():
    assert classify_entity_role("Brittle Paper") == "publication_or_source"
    assert classify_entity_role("World Health Organization") == "organization"
    assert classify_entity_role("National Sheriffs") == "organization"
    assert classify_entity_role("Tennessee") == "location"
    assert classify_entity_role("Art Deco") == "concept"
    assert classify_entity_role("Antagonist") == "concept"
    assert classify_entity_role("Edrin Vael") == "person"
    assert classify_entity_role("The Hidden Garden") == "title_or_work"


def test_target_role_inference():
    assert "person" in infer_target_roles("Who is the journalist who...?")[0]
    assert "title_or_work" in infer_target_roles("Which TV series featured...?")[0]
    assert "organization" in infer_target_roles("What is the name of the restaurant?")[0]


# ----------------------------------------------------- role-compatible scoring
def test_source_publisher_penalized_when_target_is_person():
    obs = [_obs("Brittle Paper interview", "Edrin Vael spoke to Brittle Paper about the novel.",
                "https://brittlepaper.example/x", 0.6)]
    hyps = build_hypotheses(obs, question="Who is the author?", target_roles=["person"],
                            intermediate_roles=["person", "organization"],
                            clue_terms=["novel"], stage_found=1)
    src = _by_text(hyps, "Brittle Paper")
    assert src.role == "publication_or_source" and src.source_entity_penalty > 0


def test_broad_location_penalized_when_target_is_title_or_work():
    obs = [_obs("Tennessee", "A story set in Tennessee.", "https://x.example/t", 0.6)]
    hyps = build_hypotheses(obs, question="Which TV series...?", target_roles=["title_or_work"],
                            intermediate_roles=["title_or_work", "location"],
                            clue_terms=["story"], stage_found=1)
    loc = _by_text(hyps, "Tennessee")
    assert loc.role == "location" and loc.location_penalty > 0


def test_generic_concept_penalized_when_target_is_title_or_work():
    obs = [_obs("Art Deco style", "The building uses Art Deco motifs.", "https://x.example/a", 0.6)]
    hyps = build_hypotheses(obs, question="Which building...?", target_roles=["title_or_work"],
                            intermediate_roles=["title_or_work"], clue_terms=["building"],
                            stage_found=1)
    con = _by_text(hyps, "Art Deco")
    assert con.role == "concept" and con.genericity_penalty > 0


def test_person_beats_source_site_when_target_is_person():
    obs = [_obs("Brittle Paper", "Edrin Vael wrote the novel; via Brittle Paper.",
                "https://brittlepaper.example/x", 0.7),
           _obs("Edrin Vael", "Edrin Vael survived a road accident.",
                "https://news.example/v", 0.6)]
    hyps = build_hypotheses(obs, question="Who is the author?", target_roles=["person"],
                            intermediate_roles=["person", "organization"],
                            clue_terms=["road", "accident", "novel"], stage_found=1)
    assert hyps[0].candidate_text == "Edrin Vael"          # person ranks first
    assert _by_text(hyps, "Edrin Vael").adjusted_score > _by_text(hyps, "Brittle Paper").adjusted_score


def test_title_work_beats_location_when_target_is_title_or_work():
    obs = [_obs("Tennessee", "set in Tennessee", "https://x.example/t", 0.7),
           _obs("The Hidden Garden", "The Hidden Garden is a TV series.", "https://x.example/h", 0.6)]
    hyps = build_hypotheses(obs, question="Which TV series...?", target_roles=["title_or_work"],
                            intermediate_roles=["title_or_work", "location"],
                            clue_terms=["series"], stage_found=1)
    assert hyps[0].candidate_text == "The Hidden Garden"
    assert (_by_text(hyps, "The Hidden Garden").adjusted_score
            > _by_text(hyps, "Tennessee").adjusted_score)


# ----------------------------------------------------- anti-sticky beam
def _hyp(text, role, adj):
    h = CandidateHypothesis(candidate_text=text, normalized_text=text.lower(), role=role,
                            raw_score=adj, role_match_score=0.0)
    h.recompute()
    return h


def test_sticky_candidate_downweighted_after_no_progress():
    beam = HypothesisBeam(beam_size=3)
    beam.observe([_hyp("Art Deco", "concept", 3.0), _hyp("Edrin Vael", "person", 2.0)])
    sel1 = beam.select()
    assert sel1.hypothesis.candidate_text == "Art Deco"
    beam.record_progress("art deco", improved=False)         # no progress
    sel2 = beam.select()
    # Art Deco now carries a sticky penalty; the alternative is explored.
    assert sel2.hypothesis.candidate_text == "Edrin Vael"
    assert beam.sticky_count >= 1 and beam.switch_count >= 1


def test_beam_forces_exploration_after_repeated_no_progress():
    beam = HypothesisBeam(beam_size=3)
    beam.observe([_hyp("World Health Organization", "organization", 4.0),
                  _hyp("Edrin Vael", "person", 1.0)])
    beam.select()                                            # picks WHO (top)
    beam.record_progress("world health organization", improved=False)
    beam.select()
    beam.record_progress("world health organization", improved=False)  # 2nd fail -> failed
    sel = beam.select()
    assert sel.hypothesis.candidate_text == "Edrin Vael"     # forced exploration
    assert any(r["rejection_reason"] == "failed_twice_no_progress" for r in sel.rejected)


# ----------------------------------------------------- follow-up composition
def test_followup_combines_candidate_and_unresolved_clue_not_filler():
    h = _hyp("Edrin Vael", "person", 3.0)
    fq = compose_followup_from_hypothesis(
        h, clue_spans=["road accident", "private university"], used_clue_spans=set(),
        answer_shape=["born"], beam=HypothesisBeam())
    assert '"Edrin Vael"' in fq.query and "road accident" in fq.query
    assert fq.query_arm == "candidate_entity_followup"
    assert fq.clue_used == "road accident"


def test_evidence_progress_boosts_candidate():
    beam = HypothesisBeam(beam_size=3)
    beam.observe([_hyp("Edrin Vael", "person", 2.0), _hyp("Brittle Paper", "publication_or_source", 2.2)])
    beam.select()
    beam.record_progress("edrin vael", improved=True)        # progress
    sel = beam.select()
    assert sel.hypothesis.candidate_text == "Edrin Vael"     # progress bonus wins


# ----------------------------------------------------- leakage / no providers
def test_iterative_policy_memory_stays_answer_free(items, providers):
    agent = EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "news_search", "official_domain_search",
                         "brave_search", "page_fetch"],
        enable_query_decomposition=True, enable_iterative_clue_resolution=True))
    mem = PolicyMemory(BanditParams())
    experience_phase(items[:6], agent, providers, mem, budget=3, passes=2)
    snap = mem.snapshot().to_dict()
    assert_no_answer_leakage(snap, "snapshot")               # does not raise
    for forbidden in ("answer", "final_answer", "gold", "solution"):
        assert forbidden not in __import__("json").dumps(snap).lower().split('"')  # no such key
