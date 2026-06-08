"""Cheap-first provider/model config — OpenAI hosted web_search is opt-in.

No providers are called and no keys are required.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from regimes_probe.live.cache import RecordingCache
from regimes_probe.live.providers import CachedProvider, build_live_providers
from regimes_probe.live.settings import resolve_live_settings
from regimes_probe.tools.openai_web_search import OpenAIWebSearch, openai_web_search

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------- no hardcoded gpt-5.5
def test_openai_web_search_default_is_not_gpt55():
    assert OpenAIWebSearch().model != "gpt-5.5"
    assert openai_web_search().model != "gpt-5.5"
    assert OpenAIWebSearch().model == "gpt-5.4-mini"


def test_default_live_config_is_cheap_first():
    live = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())["live"]
    assert live["answer_model"] != "gpt-5.5"
    assert live["answer_model"] == "gpt-5.4-mini"
    assert live["web_search_model"] != "gpt-5.5"
    assert live.get("search_provider_mode") == "cheap"
    # openai_web_search is NOT in the default tool set (opt-in only)
    assert "openai_web_search" not in live.get("tools_enabled", [])


def test_settings_defaults_not_gpt55():
    s = resolve_live_settings(env={"OPENAI_API_KEY": "x"})
    assert s.answer_model == "gpt-5.4-mini" and s.web_search_model == "gpt-5.4-mini"


# ------------------------------------------------- --web-search-model propagates
def test_web_search_model_propagates_into_adapter():
    cache = RecordingCache(mode="off")
    provs = build_live_providers(["openai_web_search"], cache=cache, armed=False,
                                 web_search_model="gpt-5.5", web_search_context="high")
    inner = provs["openai_web_search"].inner
    assert isinstance(provs["openai_web_search"], CachedProvider)
    assert inner.model == "gpt-5.5" and inner.context_size == "high"
    assert cache.calls == 0


# ------------------------------------------------- --tools can exclude openai
def test_explicit_tools_can_exclude_openai_web_search():
    s = resolve_live_settings(cli_tools=["serper_search"], env={"OPENAI_API_KEY": "x"})
    assert "openai_web_search" not in s.tools
    assert "serper_search" in s.tools and "page_fetch" in s.tools
    assert s.openai_web_search_enabled is False


def test_disable_flag_removes_openai_web_search():
    s = resolve_live_settings(mode="diverse", disable_openai_web_search=True,
                              env={"OPENAI_API_KEY": "x", "BRAVE_SEARCH_API_KEY": "k"})
    assert "openai_web_search" not in s.tools and "brave_search" in s.tools


# ------------------------------------------------- cheap mode never silently uses openai
def test_cheap_mode_does_not_silently_enable_openai():
    # OPENAI key present, but NO external search key -> cheap must NOT add openai
    s = resolve_live_settings(mode="cheap", env={"OPENAI_API_KEY": "x"})
    assert s.openai_web_search_enabled is False
    assert s.tools == ["page_fetch"]
    assert s.missing_search_keys  # explains which env vars would enable a provider
    assert any("NOT falling back to OpenAI" in n for n in s.notes)


def test_cheap_mode_uses_external_provider_if_key_present():
    s = resolve_live_settings(mode="cheap",
                              env={"OPENAI_API_KEY": "x", "SERPER_API_KEY": "k"})
    assert "serper_search" in s.tools and "openai_web_search" not in s.tools


# ------------------------------------------------- diverse = separate arms
def test_diverse_mode_treats_providers_as_separate_arms():
    env = {"OPENAI_API_KEY": "x", "BRAVE_SEARCH_API_KEY": "b",
           "TAVILY_API_KEY": "t", "SERPER_API_KEY": "s"}
    s = resolve_live_settings(mode="diverse", env=env)
    for arm in ("brave_search", "tavily_search", "serper_search", "openai_web_search", "page_fetch"):
        assert arm in s.tools
    # each is a distinct provider object (distinct bandit arm)
    cache = RecordingCache(mode="off")
    provs = build_live_providers(s.tools, cache=cache, armed=False)
    assert len({id(p) for p in provs.values()}) == len(provs)
    assert set(provs) == set(s.tools)
    assert cache.calls == 0


# ------------------------------------------------- openai-hosted baseline + warning
def test_openai_hosted_mode_enables_openai():
    s = resolve_live_settings(mode="openai-hosted", env={"OPENAI_API_KEY": "x"})
    assert s.openai_web_search_enabled and "openai_web_search" in s.tools


def test_gpt55_plus_openai_warns():
    s = resolve_live_settings(mode="openai-hosted", answer_model="gpt-5.5",
                              web_search_model="gpt-5.5", env={"OPENAI_API_KEY": "x"})
    assert any("EXPENSIVE" in w and "gpt-5.5" in w for w in s.warnings)


def test_cheap_mode_gpt55_no_warning_when_openai_absent():
    # gpt-5.5 answerer but no openai_web_search arm -> no "expensive hosted" warning
    s = resolve_live_settings(mode="cheap", answer_model="gpt-5.5",
                              env={"OPENAI_API_KEY": "x", "SERPER_API_KEY": "k"})
    assert s.openai_web_search_enabled is False
    assert not any("EXPENSIVE" in w for w in s.warnings)
