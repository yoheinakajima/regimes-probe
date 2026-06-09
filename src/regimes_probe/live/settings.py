"""Resolve live model + tool settings from mode/CLI/config/env (no network).

Centralizes the *cheap-first, safety-gated* policy across many tool families:
ordinary search, page fetch/scrape, agentic tool discovery, and specialized
research. OpenAI hosted ``web_search`` is opt-in; stateful/paid and browser-like
tools are off unless explicitly allowed.

Provider modes:
  * ``cheap``        — default. Answerer ``gpt-5.4-mini``; first-hop search =
    cheap external search adapters with keys (Serper/Brave/Tavily/Exa) +
    ``firecrawl_search`` if its key is present. ``page_fetch`` is included as a
    FOLLOW-UP tool only (operates on a URL from evidence), never a first-hop
    search arm. NO OpenAI web_search, NO scrape, NO discovery, NO stateful/browser
    tools. If no search provider key exists, we explain which env vars to set
    rather than falling back to hosted search.
  * ``diverse``      — the main experiment. Everything in cheap, plus
    ``monid_discover``/``monid_inspect`` (if MONID key) and OpenAI web_search as
    one arm among many (unless disabled). Each provider is a separate bandit arm.
  * ``openai-hosted``— explicit strong/expensive baseline: ``openai_web_search`` +
    ``page_fetch``; may use ``gpt-5.5``.

Safety gates (apply to mode- and ``--tools``-derived sets alike):
  * ``firecrawl_scrape``   needs ``--enable-scrape-tools``.
  * ``monid_discover/inspect`` need ``--enable-agentic-tool-discovery`` OR diverse.
  * ``monid_run`` (stateful/paid) needs ``--allow-stateful-or-paid-tools``; never auto.
  * ``firecrawl_interact`` (browser-like) needs ``--enable-browserish-tools``; never auto.
  * ``browser_use`` is DEFERRED and never enabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.tools.metadata import tool_meta, tools_meta_dict

_EXTERNAL_SEARCH = ("serper_search", "brave_search", "tavily_search", "exa_search")
MODES = ("cheap", "diverse", "openai-hosted")
_GATED = ("monid_run", "firecrawl_interact", "browser_use",
          "firecrawl_scrape", "monid_discover", "monid_inspect")


def _has_key(tool: str, env) -> bool:
    var = tool_meta(tool).requires_api_key
    return var is None or bool(env.get(var))


@dataclass
class LiveSettings:
    provider_mode: str
    tools: list[str]
    answer_model: str
    web_search_model: str
    web_search_context_size: str
    openai_web_search_enabled: bool
    agentic_tool_discovery_enabled: bool = False
    scrape_tools_enabled: bool = False
    browserish_tools_enabled: bool = False
    stateful_or_paid_tools_allowed: bool = False
    query_decomposition_enabled: bool = False
    iterative_clue_resolution_enabled: bool = False
    task_frame_enabled: bool = False
    llm_task_frame_parser_enabled: bool = False
    frontier_controller_enabled: bool = False
    task_frame_parser_model: Optional[str] = None
    missing_search_keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def search_tools(self) -> list[str]:
        return [t for t in self.tools if t != "page_fetch"]

    @property
    def first_hop_tools(self) -> list[str]:
        """Tools eligible as first-hop (search) bandit arms. Excludes follow-up
        tools like page_fetch/scrape, which operate on URLs from evidence."""
        from regimes_probe.tools.metadata import is_first_hop
        return [t for t in self.tools if is_first_hop(t)]

    @property
    def followup_tools(self) -> list[str]:
        """URL-operating follow-up tools (page_fetch / scrape), not first-hop arms."""
        from regimes_probe.tools.metadata import is_followup
        return [t for t in self.tools if is_followup(t)]

    def tools_meta(self) -> dict[str, Any]:
        return tools_meta_dict(self.tools)

    def provider_classes(self) -> list[str]:
        return sorted({tool_meta(t).family for t in self.tools})

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_mode": self.provider_mode,
            "tools": self.tools,
            # First-hop (search) bandit arms vs URL-only follow-up tools. Only the
            # first-hop arms are routed/learned; follow-up tools (page_fetch/scrape)
            # run on a URL from evidence and are NOT bandit arms.
            "first_hop_tools": self.first_hop_tools,
            "followup_tools": self.followup_tools,
            "answer_model": self.answer_model,
            "web_search_model": self.web_search_model,
            "web_search_context_size": self.web_search_context_size,
            "openai_web_search_enabled": self.openai_web_search_enabled,
            "agentic_tool_discovery_enabled": self.agentic_tool_discovery_enabled,
            "scrape_tools_enabled": self.scrape_tools_enabled,
            "browserish_tools_enabled": self.browserish_tools_enabled,
            "stateful_or_paid_tools_allowed": self.stateful_or_paid_tools_allowed,
            "query_decomposition_enabled": self.query_decomposition_enabled,
            "iterative_clue_resolution_enabled": self.iterative_clue_resolution_enabled,
            "task_frame_enabled": self.task_frame_enabled,
            "llm_task_frame_parser_enabled": self.llm_task_frame_parser_enabled,
            "frontier_controller_enabled": self.frontier_controller_enabled,
            "task_frame_parser_model": self.task_frame_parser_model,
            "provider_classes": self.provider_classes(),
            "tools_meta": self.tools_meta(),
            "missing_search_keys": self.missing_search_keys,
            "notes": self.notes,
            "warnings": self.warnings,
        }


def _safety_filter(tools, *, mode, enable_agentic, enable_scrape, enable_browserish,
                   allow_stateful, notes):
    out: list[str] = []
    for t in tools:
        if t == "browser_use":
            notes.append("browser_use is DEFERRED (browser control; prompt-injection/state "
                         "risk) — not enabled in v0.")
        elif t == "monid_run" and not allow_stateful:
            notes.append("monid_run requires --allow-stateful-or-paid-tools (stateful/paid) "
                         "— dropped.")
        elif t == "firecrawl_interact" and not enable_browserish:
            notes.append("firecrawl_interact requires --enable-browserish-tools — dropped.")
        elif t == "firecrawl_scrape" and not enable_scrape:
            notes.append("firecrawl_scrape requires --enable-scrape-tools — dropped.")
        elif t in ("monid_discover", "monid_inspect") and not (enable_agentic or mode == "diverse"):
            notes.append(f"{t} requires --enable-agentic-tool-discovery or diverse mode — dropped.")
        else:
            out.append(t)
    return out


def resolve_live_settings(
    *,
    mode: str = "cheap",
    cli_tools: Optional[list[str]] = None,
    answer_model: str = "gpt-5.4-mini",
    web_search_model: str = "gpt-5.4-mini",
    web_search_context_size: str = "low",
    disable_openai_web_search: bool = False,
    enable_agentic_tool_discovery: bool = False,
    enable_scrape_tools: bool = False,
    enable_browserish_tools: bool = False,
    allow_stateful_or_paid_tools: bool = False,
    enable_query_decomposition: bool = False,
    enable_iterative_clue_resolution: bool = False,
    enable_task_frame: bool = False,
    enable_llm_task_frame_parser: bool = False,
    enable_frontier_controller: bool = False,
    env: Optional[dict[str, str]] = None,
) -> LiveSettings:
    if mode not in MODES:
        raise ValueError(f"unknown --search-provider-mode {mode!r}; valid: {MODES}")
    env = env if env is not None else dict(os.environ)
    notes: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []

    def has(t: str) -> bool:
        return _has_key(t, env)

    if cli_tools:                                   # explicit --tools wins (still gated)
        tools = list(dict.fromkeys(cli_tools))
        notes.append("tool set fixed by --tools (overrides provider-mode default; safety "
                     "gates still apply)")
    elif mode == "openai-hosted":
        tools = ["openai_web_search"]
    elif mode == "diverse":
        tools = [t for t in _EXTERNAL_SEARCH if has(t)]
        if has("firecrawl_search"):
            tools.append("firecrawl_search")
        if has("monid_discover"):                   # diverse auto-includes discovery
            tools += ["monid_discover", "monid_inspect"]
        if enable_scrape_tools and has("firecrawl_scrape"):
            tools.append("firecrawl_scrape")
        if has("openai_web_search") and not disable_openai_web_search:
            tools.append("openai_web_search")
        if not any(t in tools for t in (*_EXTERNAL_SEARCH, "firecrawl_search")):
            notes.append("diverse mode found NO non-OpenAI search providers; set some of "
                         "SERPER_API_KEY / BRAVE_SEARCH_API_KEY / TAVILY_API_KEY / "
                         "EXA_API_KEY / FIRECRAWL_API_KEY for real provider arms.")
    else:  # cheap (default)
        tools = [t for t in _EXTERNAL_SEARCH if has(t)]
        if has("firecrawl_search"):
            tools.append("firecrawl_search")
        if enable_agentic_tool_discovery and has("monid_discover"):
            tools += ["monid_discover", "monid_inspect"]
        if enable_scrape_tools and has("firecrawl_scrape"):
            tools.append("firecrawl_scrape")
        if not tools:
            missing = ["SERPER_API_KEY", "BRAVE_SEARCH_API_KEY", "TAVILY_API_KEY",
                       "EXA_API_KEY", "FIRECRAWL_API_KEY"]
            notes.append(
                "cheap mode found NO non-OpenAI search provider key. The agent can only "
                "page_fetch known URLs. Set one of " + " / ".join(missing)
                + ", or use --search-provider-mode openai-hosted (expensive). NOT falling "
                "back to OpenAI web_search.")

    tools = _safety_filter(tools, mode=mode, enable_agentic=enable_agentic_tool_discovery,
                           enable_scrape=enable_scrape_tools,
                           enable_browserish=enable_browserish_tools,
                           allow_stateful=allow_stateful_or_paid_tools, notes=notes)
    if disable_openai_web_search:
        tools = [t for t in tools if t != "openai_web_search"]
        notes.append("openai_web_search disabled via --disable-openai-web-search")
    tools = list(dict.fromkeys(tools))
    if "page_fetch" not in tools:                   # free; FOLLOW-UP tool only —
        tools.append("page_fetch")                  # never a first-hop search arm
    # (page_fetch is family=fetch: the router never routes it first-hop; it is
    #  used only as a follow-up on a URL returned by a search provider. See
    #  agent/search_loop.py and tools/metadata.py:FOLLOWUP_FAMILIES.)

    openai_enabled = "openai_web_search" in tools
    if openai_enabled and (answer_model == "gpt-5.5" or web_search_model == "gpt-5.5"):
        warnings.append(
            "EXPENSIVE: gpt-5.5 + openai_web_search selected. This is the strong hosted "
            "baseline, NOT the default regimes-probe experiment. Prefer cheap/diverse with "
            "gpt-5.4-mini for the learning runs.")
    if mode == "openai-hosted" and not openai_enabled:
        warnings.append("openai-hosted mode but openai_web_search is not enabled "
                        "(disabled or overridden by --tools).")

    return LiveSettings(
        provider_mode=mode, tools=tools, answer_model=answer_model,
        web_search_model=web_search_model, web_search_context_size=web_search_context_size,
        openai_web_search_enabled=openai_enabled,
        agentic_tool_discovery_enabled=any(t.startswith("monid") for t in tools),
        scrape_tools_enabled="firecrawl_scrape" in tools,
        browserish_tools_enabled=any(tool_meta(t).family == "browserish" for t in tools),
        stateful_or_paid_tools_allowed=allow_stateful_or_paid_tools,
        query_decomposition_enabled=enable_query_decomposition,
        iterative_clue_resolution_enabled=enable_iterative_clue_resolution,
        task_frame_enabled=enable_task_frame,
        llm_task_frame_parser_enabled=enable_llm_task_frame_parser,
        frontier_controller_enabled=enable_frontier_controller and enable_task_frame,
        missing_search_keys=sorted(set(missing)), notes=notes, warnings=warnings)
