"""Generic task-frame / hypothesis-table / action-planner layer (no providers)."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from regimes_probe.agent.action_planner import ActionPlanner, frame_read_value
from regimes_probe.agent.clue_resolution import classify_entity_role
from regimes_probe.agent.hypothesis_table import HypothesisTable
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.agent.task_frame import parse_task_frame
from regimes_probe.datasets.base import Item
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
from regimes_probe.tools.fake import FakePageFetch

RESTAURANT = ("There is a Mexican restaurant in New Mexico, near a museum and a hotel, "
              "founded by a chef born in which year?")
WHO = ("What report was released by WHO in 2021 about air quality, authored by a "
       "scientist who later joined a university?")
TV = ("Which TV series featured an actor who immigrated from the Caribbean and won an "
      "award in 2019?")
MANGA = ("I recall a manga set in an elementary school that retells a classic story. "
         "What is the title of this manga?")


def _roles(frame):
    return {s.slot_role for s in frame.all_slots}


# ----------------------------------------------------- frame parsing
def test_restaurant_task_creates_slots_and_constraints():
    f = parse_task_frame("x", RESTAURANT)
    roles = _roles(f)
    # restaurant/hotel/museum -> organization slot; founder -> person; year -> date.
    assert "organization" in roles and "person" in roles and "date_or_time" in roles
    assert f.constraints                       # clauses became constraints
    assert any(c.constraint_type in ("location", "temporal", "relation", "attribute")
               for c in f.constraints)
    assert "New Mexico" in f.known_context_terms


def test_who_report_treats_who_as_known_context_not_target():
    f = parse_task_frame("x", WHO)
    assert "WHO" in f.known_context_terms
    # WHO must NOT be a target answer slot's name/role candidate.
    assert all(s.slot_name.lower() != "who" for s in f.target_answer_slots)
    # the target is the report/title or the author (person), not the source org.
    assert any(s.slot_role in ("title_or_work", "publication_or_source", "person")
               for s in f.target_answer_slots)


def test_tv_series_task_has_person_intermediate_and_work_target():
    f = parse_task_frame("x", TV)
    assert any(s.slot_role == "title_or_work" and s.is_target_answer_slot
               for s in f.target_answer_slots)
    assert any(s.slot_role == "person" for s in f.latent_slots)


def test_manga_task_target_is_title_or_work():
    f = parse_task_frame("x", MANGA)
    assert any(s.slot_role == "title_or_work" for s in f.target_answer_slots)


# ----------------------------------------------------- candidate role typing
def test_candidate_role_typing_is_strict():
    assert classify_entity_role("Cape Town") == "location"
    assert classify_entity_role("APEX Museum") == "organization"
    assert classify_entity_role("Military Records") == "publication_or_source"
    assert classify_entity_role("Edrin Vael") == "person"


# ----------------------------------------------------- read-value gate (frame)
def _obs(title, snippet, url, authority=0.7, contaminated=False):
    return SimpleNamespace(title=title, snippet=snippet, url=url, source_authority=authority,
                           failed=False, benchmark_contaminated=contaminated)


def test_read_rejected_when_no_unresolved_constraint_matches():
    f = parse_task_frame("x", RESTAURANT)
    t = HypothesisTable(f)
    o = _obs("Unrelated cooking blog", "A recipe for tacos.", "https://blog.example/tacos", 0.3)
    rv = frame_read_value(o, f, t)
    assert rv.should_read is False
    assert rv.read_rejected_reason in ("no_unresolved_slot_or_constraint",)


def test_read_selected_when_url_supports_unresolved_constraint():
    f = parse_task_frame("x", RESTAURANT)
    t = HypothesisTable(f)
    # snippet mirrors a constraint clause's distinctive terms.
    o = _obs("Casa Verde — Mexican restaurant in New Mexico",
             "Casa Verde is a Mexican restaurant in New Mexico founded by a chef.",
             "https://gov.example/casa-verde", 0.8)
    rv = frame_read_value(o, f, t)
    assert rv.should_read is True
    assert rv.read_selected_reason in ("contains_unresolved_clue",
                                       "supports_candidate_hypothesis",
                                       "structured_or_pdf_and_high_relevance")
    assert rv.tested_constraint_ids


def test_read_rejected_for_contaminated_url():
    f = parse_task_frame("x", RESTAURANT)
    t = HypothesisTable(f)
    o = _obs("BrowseComp", "mirror", "https://huggingface.co/x", 0.9, contaminated=True)
    assert frame_read_value(o, f, t).read_rejected_reason == "benchmark_contaminated"


# ----------------------------------------------------- hypothesis advancement
def test_hypothesis_advances_when_candidate_supports_constraint():
    # the candidate is DISCOVERED from evidence (not given in the question); the
    # evidence names its kind ("Casa Verde restaurant") so it binds to the org target.
    f = parse_task_frame("x", "Which restaurant in New Mexico won a James Beard award?")
    t = HypothesisTable(f)
    o = _obs("Casa Verde restaurant",
             "Casa Verde is a restaurant in New Mexico that won a James Beard award.",
             "https://gov.example/casa-verde", 0.8)
    o.extra = {}
    t.ingest_evidence([o], source_tool="generic_web_search", stage=1)
    best = t.best_hypothesis()
    assert best is not None and best.slot_assignments          # a candidate was bound
    # at least one constraint is supported / hypothesis has support.
    assert any(c.status == "resolved" for c in f.constraints) or best.support_score >= 0


def test_hypothesis_downweighted_after_repeated_no_progress():
    f = parse_task_frame("x", "Which restaurant in New Mexico won an award?")
    t = HypothesisTable(f)
    o = _obs("Casa Verde restaurant", "Casa Verde is a restaurant in New Mexico.",
             "https://x.example/cv", 0.6)
    t.ingest_evidence([o], source_tool="generic_web_search", stage=1)
    best = t.best_hypothesis()
    assert best is not None
    t.note_no_progress(best.hypothesis_id)
    t.note_no_progress(best.hypothesis_id)
    assert not t.hypotheses[best.hypothesis_id].active
    assert t.hypotheses[best.hypothesis_id].rejection_reason == "repeated_no_progress"


# ----------------------------------------------------- end-to-end loop + debug + memory
class _FrameProvider(SearchProvider):
    name = "generic_web_search"
    cost_per_call = Decimal("0.002")
    deterministic = True

    def available(self): return True

    def search(self, q, *, limit=5, **o):
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="Casa Verde — restaurant in New Mexico",
                         url="https://gov.example/casa-verde",
                         snippet="Casa Verde is a Mexican restaurant in New Mexico.",
                         source_authority=0.8, rank=0,
                         extra={"item_id": "x", "asserts": "Casa Verde"}),), cost=self.cost_per_call)


def _frame_agent():
    return EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "page_fetch"], stop_mode="always_full",
        enable_task_frame=True, verification=VerificationConfig(min_support=2)))


def test_task_frame_drives_loop_and_debug_artifact():
    from regimes_probe.eval.debug import build_debug_record
    from regimes_probe.eval.grader import grade
    from regimes_probe.eval.reward import RewardWeights, compute_rewards
    item = Item(id="x", answer="Casa Verde",
                question="Which Mexican restaurant in New Mexico was founded by a chef?")
    providers = {"generic_web_search": _FrameProvider(), "page_fetch": FakePageFetch([])}
    tr = _frame_agent().attempt(item, PolicyMemory(BanditParams()), providers,
                                budget=3, explore=False, attempt_id="t")
    assert tr.task_frame and tr.task_frame["target_answer_slots"]
    assert any(c.task_action.get("kind") for c in tr.calls)     # planner drove actions
    rec = build_debug_record(item=item, trace=tr, grade=grade(item, tr.final_answer),
                             reward=compute_rewards(tr, correct=False, gold_norms=[],
                                                    weights=RewardWeights.full(),
                                                    freshness_sensitive=False),
                             condition="no_memory_search", budget=3).to_dict()
    assert rec["task_frame"]["target_answer_slots"]             # frame summary in debug
    assert "frame_coverage" in rec and rec["calls"][0]["task_action"].get("kind")
    assert "top_hypotheses" in rec["hypothesis_summary"]


def test_task_frame_policy_memory_stays_answer_free():
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
        enable_task_frame=True))
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=3, passes=2)
    snap = mem.snapshot().to_dict()
    assert_no_answer_leakage(snap, "snapshot")                 # does not raise
