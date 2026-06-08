"""Resolve live model + tool settings from mode/CLI/config/env (no network).

This centralizes the *cheap-first* policy: OpenAI hosted ``web_search`` is opt-in,
never the silent default. Three provider modes:

  * ``cheap``        — default. Answerer ``gpt-5.4-mini``; search = ``page_fetch``
    plus any low-cost external search adapter whose key is present. OpenAI
    web_search is NOT added. If no external search key exists, we explain which
    env vars would enable one (rather than falling back to expensive hosted search).
  * ``diverse``      — the main experiment. ``page_fetch`` + every external search
    adapter with a key (Brave/Tavily/Exa/Serper), and OpenAI web_search as ONE arm
    among many (unless disabled). Each provider is a separate bandit arm.
  * ``openai-hosted``— explicit strong/expensive baseline: ``openai_web_search`` +
    ``page_fetch``; may use ``gpt-5.5`` if requested.

``--tools`` overrides a mode's tool set explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.live.providers import _ADAPTER_ENV

#: External search adapters, cheapest-first, eligible for cheap/diverse auto-add.
_EXTERNAL_SEARCH = ("serper_search", "brave_search", "tavily_search", "exa_search")
_EXTERNAL_KEYS = {
    "serper_search": "SERPER_API_KEY", "brave_search": "BRAVE_SEARCH_API_KEY",
    "tavily_search": "TAVILY_API_KEY", "exa_search": "EXA_API_KEY",
}
MODES = ("cheap", "diverse", "openai-hosted")


def _has_key(tool: str, env) -> bool:
    var = _ADAPTER_ENV.get(tool)
    return var is None or bool(env.get(var))


@dataclass
class LiveSettings:
    provider_mode: str
    tools: list[str]
    answer_model: str
    web_search_model: str
    web_search_context_size: str
    openai_web_search_enabled: bool
    missing_search_keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def search_tools(self) -> list[str]:
        return [t for t in self.tools if t != "page_fetch"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_mode": self.provider_mode,
            "tools": self.tools,
            "answer_model": self.answer_model,
            "web_search_model": self.web_search_model,
            "web_search_context_size": self.web_search_context_size,
            "openai_web_search_enabled": self.openai_web_search_enabled,
            "missing_search_keys": self.missing_search_keys,
            "notes": self.notes,
            "warnings": self.warnings,
        }


def resolve_live_settings(
    *,
    mode: str = "cheap",
    cli_tools: Optional[list[str]] = None,
    answer_model: str = "gpt-5.4-mini",
    web_search_model: str = "gpt-5.4-mini",
    web_search_context_size: str = "low",
    disable_openai_web_search: bool = False,
    env: Optional[dict[str, str]] = None,
) -> LiveSettings:
    if mode not in MODES:
        raise ValueError(f"unknown --search-provider-mode {mode!r}; valid: {MODES}")
    env = env if env is not None else dict(os.environ)
    notes: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []

    if cli_tools:                                   # explicit --tools wins
        tools = list(dict.fromkeys(cli_tools))
        notes.append("tool set fixed by --tools (overrides provider mode default)")
    elif mode == "openai-hosted":
        tools = ["openai_web_search"]
    elif mode == "diverse":
        external = [t for t in _EXTERNAL_SEARCH if _has_key(t, env)]
        tools = list(external)
        if _has_key("openai_web_search", env) and not disable_openai_web_search:
            tools.append("openai_web_search")
        if not external:
            missing = [v for v in _EXTERNAL_KEYS.values() if not env.get(v)]
            notes.append(
                "diverse mode found NO external (non-OpenAI) search providers; set some of "
                + " / ".join(sorted(set(missing)))
                + " so the bandit has multiple provider arms. "
                + ("openai_web_search is the only search arm right now."
                   if "openai_web_search" in tools else
                   "Only page_fetch is available right now."))
    else:  # cheap (default)
        tools = [t for t in _EXTERNAL_SEARCH if _has_key(t, env)]
        if not tools:
            missing = [_EXTERNAL_KEYS[t] for t in _EXTERNAL_SEARCH]
            notes.append(
                "cheap mode found NO non-OpenAI search provider key. The agent will "
                "only be able to page_fetch known URLs. Set one of "
                + " / ".join(missing)
                + ", or use --search-provider-mode openai-hosted (expensive). "
                "NOT falling back to OpenAI web_search.")

    if disable_openai_web_search:
        tools = [t for t in tools if t != "openai_web_search"]
        notes.append("openai_web_search disabled via --disable-openai-web-search")
    if "page_fetch" not in tools:                   # free; always available
        tools.append("page_fetch")

    openai_enabled = "openai_web_search" in tools
    if openai_enabled and (answer_model == "gpt-5.5" or web_search_model == "gpt-5.5"):
        warnings.append(
            "EXPENSIVE: gpt-5.5 + openai_web_search selected. This is the strong "
            "hosted baseline, NOT the default regimes-probe experiment. Prefer "
            "cheap/diverse with gpt-5.4-mini for the learning runs.")
    if mode == "openai-hosted" and not openai_enabled:
        warnings.append("openai-hosted mode but openai_web_search is not enabled "
                        "(disabled or overridden by --tools).")

    return LiveSettings(
        provider_mode=mode, tools=tools, answer_model=answer_model,
        web_search_model=web_search_model, web_search_context_size=web_search_context_size,
        openai_web_search_enabled=openai_enabled, missing_search_keys=sorted(set(missing)),
        notes=notes, warnings=warnings)
