"""Live executor safety — NONE of these call a provider or require keys."""

from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent, build_closed_book_agent
from regimes_probe.agent.answerer import build_closed_book_knowledge
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.live.cache import RecordingCache, ReplayMiss, sanitize
from regimes_probe.live.providers import (
    CachedProvider, NotArmed, build_live_providers, missing_keys)
from regimes_probe.live.runner import build_plan, run_live_pipeline
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.tools.fake import FakeSearchProvider, build_fake_providers, load_corpus
from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "fixtures" / "real_shaped"
SEARCH_TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]


def _cfg():
    return yaml.safe_load((ROOT / "config" / "default.yaml").read_text())


def _rs_items_and_providers():
    items = LiveBrowseCompAdapter(local_jsonl=RS / "livebrowsecomp_sample.jsonl").load()
    corpus = load_corpus(RS / "real_shaped_corpus.json")
    providers = build_fake_providers(corpus, SEARCH_TOOLS)
    return items, providers


# --------------------------------------------------- un-armed providers can't spend
def test_unarmed_provider_refuses_to_call():
    cache = RecordingCache(mode="off")
    inner = FakeSearchProvider("news_search", [], cost_per_call="0.01")
    prov = CachedProvider(inner, cache, armed=False)   # dry-run => not armed
    with pytest.raises(NotArmed):
        prov.search("anything")
    assert cache.calls == 0


def test_build_live_providers_unarmed_by_default_path():
    cache = RecordingCache(mode="off")
    provs = build_live_providers(["openai_web_search", "page_fetch"], cache=cache, armed=False)
    assert set(provs) == {"openai_web_search", "page_fetch"}
    with pytest.raises(NotArmed):
        provs["openai_web_search"].search("q")   # would be the only network; refused
    assert cache.calls == 0


# --------------------------------------------------- recording cache / replay
def test_cache_auto_stores_then_serves_without_respend():
    corpus = load_corpus(RS / "real_shaped_corpus.json")
    inner = FakeSearchProvider("news_search", corpus)  # deterministic, no network
    cache = RecordingCache(mode="auto")
    prov = CachedProvider(inner, cache, armed=True)
    r1 = prov.search("greyhold steward")
    assert cache.calls == 1 and cache.hits == 0
    r2 = prov.search("greyhold steward")               # identical -> served from cache
    assert cache.calls == 1 and cache.hits == 1        # NO re-spend
    assert r1.to_dict() == r2.to_dict()


def test_cache_replay_miss_never_calls(tmp_path):
    cache = RecordingCache(tmp_path / "c.json", mode="replay")
    inner = FakeSearchProvider("news_search", [])
    prov = CachedProvider(inner, cache, armed=True)
    with pytest.raises(ReplayMiss):
        prov.search("uncached query")
    assert cache.calls == 0                              # replay never spends


def test_cache_sanitizes_secrets(tmp_path):
    cache = RecordingCache(tmp_path / "c.json", mode="auto")
    cache.store("h1", provider="p", name="n",
                request_meta={"query": "x", "api_key": "sk-SECRET", "Authorization": "Bearer T"},
                response_payload={"text": "ok"})
    blob = (tmp_path / "c.json").read_text()
    assert "sk-SECRET" not in blob and "Bearer T" not in blob and "<redacted>" in blob


def test_sanitize_redacts_key_like_fields():
    s = sanitize({"token": "abc", "nested": {"secret": "x", "ok": 1}})
    assert s["token"] == "<redacted>" and s["nested"]["secret"] == "<redacted>"
    assert s["nested"]["ok"] == 1


# --------------------------------------------------- missing keys
def test_missing_keys_reports_names(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    miss = missing_keys(["brave_search", "page_fetch"])
    assert "BRAVE_SEARCH_API_KEY" in miss and "OPENAI_API_KEY" in miss
    # page_fetch needs no key
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert missing_keys(["page_fetch"], answer_model_needs_openai=False) == []


# --------------------------------------------------- execute pipeline (mocked, no network)
def test_run_live_pipeline_with_mocks_passes_plumbing(tmp_path):
    cfg = _cfg()
    items, providers = _rs_items_and_providers()
    agent = EpistemicAgent(AgentConfig(available_tools=SEARCH_TOOLS + ["page_fetch"]))
    cb = build_closed_book_agent(AgentConfig(available_tools=SEARCH_TOOLS + ["page_fetch"]),
                                 build_closed_book_knowledge(items))
    cache = RecordingCache(mode="off")
    out = run_live_pipeline(
        cfg, items, providers=providers, search_agent=agent, cb_agent=cb, cache=cache,
        conditions=list(("closed_book", "no_memory_search", "random_memory", "policy_memory")),
        budgets=[1, 3], optimize=4, confirm=4, split_seed="t", run_id="mocked",
        results_root=str(tmp_path), dataset_label="real_shaped_placeholder",
        dataset_version="v", dataset_path=None, is_real=False, search_tools=SEARCH_TOOLS,
        weights=RewardWeights.full(), params=BanditParams())
    rd = Path(out["run_dir"])
    assert (rd / "report.json").exists() and (rd / "run_manifest.json").exists()
    report = json.loads((rd / "report.json").read_text())
    # same-conditions plumbing holds; synthetic/placeholder => not headline eligible
    assert report["same_conditions"]["ok"] is True
    assert report["headline_eligible"] is False
    assert out["replay"]["projection_matches"] is True
    assert cache.calls == 0                              # mock providers, no live calls


# --------------------------------------------------- the script defaults to no-spend
def test_script_dry_run_makes_no_calls(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_live.py"),
         "--dataset", "real-shaped", "--optimize", "4", "--confirm", "6",
         "--budgets", "1", "--conditions", "closed_book,no_memory_search",
         "--results-root", str(tmp_path)],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "DRY-RUN" in out.stdout and "NO providers were called" in out.stdout
    # a plan was written, but no report/cache (nothing executed)
    run_dirs = list(tmp_path.glob("live-*"))
    assert run_dirs and (run_dirs[0] / "plan.json").exists()
    assert not (run_dirs[0] / "report.json").exists()
