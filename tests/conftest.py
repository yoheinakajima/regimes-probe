"""Shared test fixtures. No network, no keys."""

from __future__ import annotations

import pytest

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
from regimes_probe.tools.fake import build_fake_providers, load_corpus

TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]


@pytest.fixture(scope="session")
def items():
    return SyntheticBrowseAdapter().load()


@pytest.fixture(scope="session")
def corpus():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    return load_corpus(root / "fixtures" / "fake_search_corpus.json")


@pytest.fixture()
def providers(corpus):
    return build_fake_providers(corpus, TOOLS)


@pytest.fixture()
def agent():
    cfg = AgentConfig(available_tools=TOOLS + ["page_fetch"],
                      query_mode="learned", stop_mode="learned")
    return EpistemicAgent(cfg)
