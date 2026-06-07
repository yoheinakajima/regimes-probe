"""Shared setup helpers for the CLI scripts (synthetic, no keys/network)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.router import RouterConfig
from regimes_probe.policy.stopping_policy import StopConfig
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.tools.fake import build_fake_providers, load_corpus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def reward_weights(cfg: dict[str, Any]) -> RewardWeights:
    preset = cfg.get("reward", {}).get("preset", "full")
    if preset != "full" and hasattr(RewardWeights, preset):
        return getattr(RewardWeights, preset)()
    w = cfg.get("reward", {}).get("weights", {})
    return RewardWeights.from_dict(w) if w else RewardWeights.full()


def bandit_params(cfg: dict[str, Any]) -> BanditParams:
    return BanditParams.from_dict(cfg.get("bandit", {}))


def build_synthetic(cfg: dict[str, Any]):
    """Return (items, providers, tools, search_tools)."""
    ds = cfg.get("dataset", {})
    items = SyntheticBrowseAdapter(ROOT / ds.get("synthetic_path", "fixtures/synthetic_browse.json")).load()
    corpus = load_corpus(ROOT / ds.get("fake_corpus_path", "fixtures/fake_search_corpus.json"))
    search_tools = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]
    providers = build_fake_providers(corpus, search_tools)
    tools = search_tools + ["page_fetch"]
    return items, providers, tools, search_tools


def build_agent(cfg: dict[str, Any], tools: list[str]) -> EpistemicAgent:
    pol = cfg.get("policy", {})
    agent_cfg = AgentConfig(
        available_tools=tools,
        query_mode=pol.get("query_mode", "learned"),
        stop_mode=pol.get("stop_mode", "learned"),
        as_of=cfg.get("run", {}).get("as_of", "2026-06-01"),
        router=RouterConfig(**cfg.get("router", {})) if cfg.get("router") else RouterConfig(),
        stop=StopConfig(**cfg.get("stopping", {})) if cfg.get("stopping") else StopConfig(),
        verification=VerificationConfig(**cfg.get("verification", {})) if cfg.get("verification") else VerificationConfig(),
    )
    return EpistemicAgent(agent_cfg)


def base_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--run-id", default=None)
    p.add_argument("--results-root", default=str(ROOT / "results"))
    return p
