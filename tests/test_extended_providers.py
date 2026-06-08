"""Monid / Firecrawl / Wokelo adapters + safety gating (no network, no keys).

Network is faked by monkeypatching ``urllib.request.urlopen`` — no real provider
call is ever made.
"""

from __future__ import annotations

import json

import pytest

from regimes_probe.eval.manifest import build_manifest
from regimes_probe.live.cache import RecordingCache
from regimes_probe.live.providers import build_live_providers
from regimes_probe.live.settings import resolve_live_settings
from regimes_probe.tools.base import ProviderUnavailable, ToolDisabled
from regimes_probe.tools.firecrawl import (
    FirecrawlInteract, FirecrawlScrape, FirecrawlSearch)
from regimes_probe.tools.monid import MonidDiscover, MonidInspect, MonidRun
from regimes_probe.tools.wokelo import WokeloResearch


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake_http(monkeypatch):
    captured = {"reqs": [], "payload": {}}

    def fake_urlopen(req, timeout=None):
        captured["reqs"].append(req)
        return _FakeResp(captured["payload"])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return captured


# --------------------------------------------------------------- Monid
def test_monid_discover_builds_request_and_normalizes(monkeypatch, fake_http):
    monkeypatch.setenv("MONID_API_KEY", "sk-MONID-SECRET")
    fake_http["payload"] = {"tools": [
        {"name": "Acme Search", "description": "search tool", "pricing": "$0.01/call",
         "input_schema": {"properties": {"query": {}, "limit": {}}}, "url": "https://t/acme"}]}
    resp = MonidDiscover().search("find a search tool", limit=3)
    req = fake_http["reqs"][0]
    assert req.full_url.endswith("/discover")
    assert json.loads(req.data)["query"] == "find a search tool"
    assert req.headers.get("Authorization") == "Bearer sk-MONID-SECRET"   # real call needs it
    r = resp.results[0]
    assert r.title == "Acme Search" and "pricing" in r.snippet and "inputs: limit, query" in r.snippet
    assert r.extra["tool_family"] == "agentic_discovery" and r.extra["provider"] == "monid"


def test_monid_run_disabled_by_default():
    with pytest.raises(ToolDisabled):
        MonidRun().search("some_tool")             # not enabled -> refuses


def test_monid_run_enabled_requires_key(monkeypatch, fake_http):
    monkeypatch.delenv("MONID_API_KEY", raising=False)
    with pytest.raises(ProviderUnavailable):
        MonidRun(enabled=True).search("tool")      # enabled but no key


def test_cache_redacts_monid_secret(monkeypatch, fake_http):
    monkeypatch.setenv("MONID_API_KEY", "sk-MONID-SECRET")
    fake_http["payload"] = {"tools": []}
    cache = RecordingCache(mode="auto")
    prov = build_live_providers(["monid_discover"], cache=cache, armed=True)["monid_discover"]
    prov.search("q")                               # faked transport, no real network
    blob = json.dumps(list(cache.entries.values()))
    assert "sk-MONID-SECRET" not in blob and "Bearer" not in blob   # cache stores no secret


# --------------------------------------------------------------- Firecrawl
def test_firecrawl_search_normalizes(monkeypatch, fake_http):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
    fake_http["payload"] = {"data": [
        {"title": "Result", "url": "https://x/1", "description": "snippet text"}]}
    resp = FirecrawlSearch().search("q", limit=2)
    assert resp.results[0].url == "https://x/1"
    assert resp.results[0].extra["provider"] == "firecrawl"


def test_firecrawl_scrape_returns_markdown(monkeypatch, fake_http):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
    fake_http["payload"] = {"data": {"markdown": "# Title\n\nbody text",
                                     "metadata": {"title": "Title"}}}
    resp = FirecrawlScrape().search("https://x/page", formats=["markdown"])
    r = resp.results[0]
    assert "body text" in r.snippet and r.extra["tool_family"] == "scrape"
    req = fake_http["reqs"][0]
    assert req.full_url.endswith("/scrape") and json.loads(req.data)["url"] == "https://x/page"


def test_firecrawl_interact_disabled_by_default():
    with pytest.raises(ToolDisabled):
        FirecrawlInteract().search("https://x")


# --------------------------------------------------------------- Wokelo (fail closed)
def test_wokelo_fails_closed_without_base_url(monkeypatch):
    monkeypatch.setenv("WOKELO_API_KEY", "wk-key")
    monkeypatch.delenv("WOKELO_BASE_URL", raising=False)
    monkeypatch.delenv("WOKELO_RESEARCH_PATH", raising=False)
    w = WokeloResearch()
    assert w.available() is False
    with pytest.raises(ProviderUnavailable) as exc:
        w.search("research acme")
    assert "WOKELO_BASE_URL" in str(exc.value)


def test_wokelo_available_when_fully_configured(monkeypatch):
    monkeypatch.setenv("WOKELO_API_KEY", "wk-key")
    monkeypatch.setenv("WOKELO_BASE_URL", "https://api.wokelo.test")
    monkeypatch.setenv("WOKELO_RESEARCH_PATH", "/v1/research")
    assert WokeloResearch().available() is True     # configured; not called (no network)


# --------------------------------------------------------------- safety gating
def _all_keys():
    return {"OPENAI_API_KEY": "o", "SERPER_API_KEY": "s", "FIRECRAWL_API_KEY": "f",
            "MONID_API_KEY": "m", "BRAVE_SEARCH_API_KEY": "b"}


def test_cheap_mode_never_includes_stateful_or_browserish():
    s = resolve_live_settings(mode="cheap", env=_all_keys())
    for unsafe in ("monid_run", "firecrawl_interact", "browser_use", "firecrawl_scrape",
                   "monid_discover", "openai_web_search"):
        assert unsafe not in s.tools
    assert s.stateful_or_paid_tools_allowed is False
    assert s.browserish_tools_enabled is False


def test_diverse_includes_discovery_but_not_run_or_interact():
    s = resolve_live_settings(mode="diverse", env=_all_keys())
    assert "monid_discover" in s.tools                  # discovery auto in diverse
    assert "monid_run" not in s.tools and "firecrawl_interact" not in s.tools
    assert "firecrawl_scrape" not in s.tools             # needs --enable-scrape-tools


def test_scrape_only_with_flag():
    s = resolve_live_settings(mode="diverse", enable_scrape_tools=True, env=_all_keys())
    assert "firecrawl_scrape" in s.tools


def test_monid_run_only_with_allow_flag():
    dropped = resolve_live_settings(cli_tools=["monid_run", "serper_search"], env=_all_keys())
    assert "monid_run" not in dropped.tools
    allowed = resolve_live_settings(cli_tools=["monid_run", "serper_search"],
                                    allow_stateful_or_paid_tools=True, env=_all_keys())
    assert "monid_run" in allowed.tools and allowed.stateful_or_paid_tools_allowed


def test_browser_use_never_enabled():
    s = resolve_live_settings(cli_tools=["browser_use", "serper_search"],
                              enable_browserish_tools=True, allow_stateful_or_paid_tools=True,
                              env=_all_keys())
    assert "browser_use" not in s.tools
    assert any("DEFERRED" in n for n in s.notes)


# --------------------------------------------------------------- metadata in manifest
def test_provider_metadata_in_manifest():
    from regimes_probe.eval.split import build_split
    from regimes_probe.datasets.base import Item
    items = [Item(id=f"i{i}", question="q?", answer="a") for i in range(6)]
    split = build_split(items, confirm_fraction=0.5)
    s = resolve_live_settings(mode="diverse", enable_scrape_tools=True, env=_all_keys())
    m = build_manifest(
        run_id="t", cfg={"live": {}, "budgets": [1]}, dataset_label="d", dataset_version="v",
        dataset_checksum="c", dataset_path=None, split=split, search_tools=s.search_tools,
        tools_cfg=None, memory_cfg={"confirm_uses_frozen_snapshot": True},
        eligibility_preflight={}, cost_estimate={}, live_settings=s.to_dict())
    assert "tools_meta" in m and "provider_classes" in m
    assert "agentic_discovery" in m["provider_classes"]
    assert m["tools_meta"]["monid_discover"]["tool_family"] == "agentic_discovery"
    assert m["tools_meta"]["firecrawl_scrape"]["stateful"] is False
    assert m["scrape_tools_enabled"] is True
