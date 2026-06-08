"""Real-data-SHAPED smoke test (no keys, no network).

Verifies that the *real benchmark adapter path* works on inputs with the same
structure as BrowseComp (obfuscated CSV) and LiveBrowseComp (JSONL of recent-fact
rows), and that the split builder, report generator, leakage checks, budget
curves, replay check, and memory snapshot all accept that shape.

IMPORTANT: the fixtures are obviously-fictional placeholders. Nothing here is a
BrowseComp/LiveBrowseComp result and must never be reported as one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regimes_probe.activegraph_pack import replay_check
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.base import DatasetUnavailable
from regimes_probe.datasets.browsecomp import BrowseCompAdapter, decrypt, encrypt
from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
from regimes_probe.eval.harness import experience_phase, run_condition
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.report import ConditionRun, write_full_report
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.tools.fake import build_fake_providers, load_corpus

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "fixtures" / "real_shaped"
SEARCH_TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]


# ---------------------------------------------------------------- adapter shape
def test_browsecomp_obfuscation_round_trip():
    assert decrypt(encrypt("hello world", "C-1"), "C-1") == "hello world"


def test_browsecomp_adapter_decodes_placeholder_csv():
    bc = BrowseCompAdapter(RS / "browsecomp_sample.csv")
    items = bc.load()
    assert len(items) == 8 and bc.skipped == 0
    assert items[0].answer == "Novograd"
    assert "placeholder" in items[0].question.lower()
    # version carries a content checksum; stable across loads
    assert bc.version().startswith("browsecomp@")
    assert bc.checksum(items) == bc.checksum(items)


def test_browsecomp_adapter_skips_undecodable_rows_gracefully(tmp_path):
    bad = tmp_path / "bad.csv"
    good_q = encrypt("ok question placeholder", "C")
    good_a = encrypt("ok", "C")
    bad.write_text(f'problem,answer,canary\n"{good_q}","{good_a}","C"\n"%%notb64%%","x","C"\n',
                   encoding="utf-8")
    bc = BrowseCompAdapter(bad)
    items = bc.load()
    assert len(items) == 1 and bc.skipped == 1  # bad row skipped, not crashed


def test_browsecomp_missing_file_is_clear_error():
    with pytest.raises(DatasetUnavailable):
        BrowseCompAdapter(ROOT / "fixtures" / "does_not_exist.csv").load()


def test_livebrowsecomp_local_jsonl_shape():
    lb = LiveBrowseCompAdapter(local_jsonl=RS / "livebrowsecomp_sample.jsonl")
    items = lb.load()
    assert len(items) == 8
    assert all(i.released_at for i in items)        # recent-fact dates present
    assert lb.version().startswith("livebrowsecomp@local:")


def test_livebrowsecomp_missing_local_is_clear_error():
    with pytest.raises(DatasetUnavailable):
        LiveBrowseCompAdapter(local_jsonl=ROOT / "fixtures" / "nope.jsonl").load()


# ----------------------------------------------------------- full pipeline shape
def _pipeline_items_and_providers():
    # Use the LiveBrowseComp adapter for the pipeline: it preserves the row ``id``
    # (rsq-NNN), which the fake corpus is keyed on. (BrowseComp reassigns ids, so
    # it is used above only for the decode/shape assertions.)
    items = LiveBrowseCompAdapter(local_jsonl=RS / "livebrowsecomp_sample.jsonl").load()
    corpus = load_corpus(RS / "real_shaped_corpus.json")
    providers = build_fake_providers(corpus, SEARCH_TOOLS)
    agent = EpistemicAgent(AgentConfig(available_tools=SEARCH_TOOLS + ["page_fetch"]))
    return items, providers, agent


def test_real_shaped_pipeline_runs_and_reports(tmp_path):
    items, providers, agent = _pipeline_items_and_providers()
    params = BanditParams()

    split = build_split(items, confirm_fraction=0.5)
    split.assert_disjoint()
    opt, con = partition(items, split)
    assert opt and con

    mem = PolicyMemory(params.copy())
    experience_phase(opt, agent, providers, mem, budget=4, passes=4,
                     dataset_version="real_shaped@test")
    snap = mem.snapshot()

    # leakage: no placeholder gold answer text may appear in the snapshot
    blob = json.dumps(snap.to_dict())
    for it in items:
        for gold in it.gold_answers():
            assert gold not in blob

    runs = []
    by_budget = {}
    for b in (1, 3, 5):
        base = run_condition(con, agent, providers, PolicyMemory(params.copy()),
                             condition="no_memory", budget=b).outcomes
        pol = run_condition(con, agent, providers,
                            PolicyMemory.from_snapshot(snap, frozen=True),
                            condition="policy_memory", budget=b).outcomes
        runs += [ConditionRun("no_memory", b, base), ConditionRun("policy_memory", b, pol)]
        by_budget[b] = (base, pol)
        assert all(o.tool_calls <= b for o in base + pol)  # budget binds

    # Non-degenerate: the learned policy is not worse than no-memory on the
    # placeholder corpus (deterministic; this is a sanity check on the harness,
    # NOT a benchmark result).
    base1, pol1 = by_budget[1]
    assert sum(o.correct for o in pol1) >= sum(o.correct for o in base1)

    # report generator accepts the real-shaped outcomes
    run_dir = write_full_report("real_shaped_smoke", runs=runs, snapshot=snap.to_dict(),
                                meta={"dataset": "real_shaped_placeholder"},
                                results_root=tmp_path)
    for fname in ("report.json", "summary.md", "budget_curve.csv", "per_question.csv",
                  "memory_snapshot.json", "replay_check.md"):
        assert (run_dir / fname).exists(), f"missing {fname}"

    # the report's snapshot is also answer-free
    snap_blob = (run_dir / "memory_snapshot.json").read_text()
    for it in items:
        for gold in it.gold_answers():
            assert gold not in snap_blob


def test_real_shaped_recorded_run_replays(tmp_path):
    items, providers, agent = _pipeline_items_and_providers()
    mem = PolicyMemory(BanditParams())
    res = run_condition(items, agent, providers, mem, condition="policy_memory",
                        budget=3, dataset_version="real_shaped@test")
    report = replay_check(res.log)
    assert report.projection_matches is True
    assert report.n_events > 0
