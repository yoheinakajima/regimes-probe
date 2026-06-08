"""Tool metadata registry — the single source of truth for each tool arm.

Every search/fetch/scrape/discovery tool the project knows about is described
here with its provider *family*, expected cost class, and safety flags. The
contextual bandit treats each tool as a separate arm; this registry is what the
provider-mode resolver, the cost estimator, the manifest, and the router consult
to decide which arms are safe to enable and how to label them.

Nothing here performs I/O. ``requires_api_key`` records the env-var NAME only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: Canonical tool families (a.k.a. provider class/type).
FAMILIES = ("search", "scrape", "fetch", "specialized_research",
            "agentic_discovery", "browserish")
COST_CLASSES = ("low", "medium", "high", "unknown")

#: Families that may be selected as a FIRST-HOP arm (they take a query and can be
#: the agent's first action). The router only routes over these.
FIRST_HOP_FAMILIES = ("search", "specialized_research", "agentic_discovery")
#: Families that operate on a URL produced by earlier evidence — FOLLOW-UP only
#: (e.g. page_fetch / firecrawl_scrape). Never a first-hop search arm.
FOLLOWUP_FAMILIES = ("fetch", "scrape", "browserish")


@dataclass(frozen=True)
class ToolMeta:
    name: str
    family: str                       # one of FAMILIES
    expected_cost_class: str          # low | medium | high | unknown
    stateful: bool                    # can it take state-changing / paid actions?
    requires_network: bool
    requires_api_key: Optional[str]   # env-var NAME, or None if no key
    safe_default: bool                # may auto-enable in cheap/diverse w/o a flag
    implemented: bool = True          # is a live adapter actually built?
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tool_family": self.family,
            "expected_cost_class": self.expected_cost_class,
            "stateful": self.stateful,
            "requires_network": self.requires_network,
            "requires_api_key": bool(self.requires_api_key),
            "api_key_env": self.requires_api_key or "",
            "safe_default": self.safe_default,
            "implemented": self.implemented,
        }


def _m(name, family, cost, *, stateful=False, key=None, safe_default=False,
       net=True, implemented=True, notes="") -> ToolMeta:
    return ToolMeta(name=name, family=family, expected_cost_class=cost, stateful=stateful,
                    requires_network=net, requires_api_key=key, safe_default=safe_default,
                    implemented=implemented, notes=notes)


TOOL_META: dict[str, ToolMeta] = {
    # --- ordinary search providers (low-cost, safe to auto-enable with a key) ---
    "serper_search": _m("serper_search", "search", "low", key="SERPER_API_KEY", safe_default=True),
    "brave_search": _m("brave_search", "search", "low", key="BRAVE_SEARCH_API_KEY", safe_default=True),
    "tavily_search": _m("tavily_search", "search", "low", key="TAVILY_API_KEY", safe_default=True),
    "exa_search": _m("exa_search", "search", "low", key="EXA_API_KEY", safe_default=True),
    "news_search": _m("news_search", "search", "low", key="NEWS_API_KEY",
                      notes="interface-only; needs an endpoint"),
    "official_domain_search": _m("official_domain_search", "search", "low",
                                 key="OFFICIAL_SEARCH_API_KEY", notes="constrained search"),
    "generic_web_search": _m("generic_web_search", "search", "low", key=None,
                             notes="interface-only; needs an endpoint"),
    # --- page fetch / scrape ---
    "page_fetch": _m("page_fetch", "fetch", "low", key=None, safe_default=True, net=True),
    "firecrawl_search": _m("firecrawl_search", "search", "low", key="FIRECRAWL_API_KEY",
                           safe_default=True, notes="search + optional page content"),
    "firecrawl_scrape": _m("firecrawl_scrape", "scrape", "medium", key="FIRECRAWL_API_KEY",
                           safe_default=False, notes="full-page markdown; richer evidence"),
    "firecrawl_interact": _m("firecrawl_interact", "browserish", "high", stateful=True,
                             key="FIRECRAWL_API_KEY", safe_default=False,
                             notes="browser-like interaction; disabled by default"),
    # --- agentic tool discovery (Monid) ---
    "monid_discover": _m("monid_discover", "agentic_discovery", "low", key="MONID_API_KEY",
                         safe_default=False, notes="discover tools for a task"),
    "monid_inspect": _m("monid_inspect", "agentic_discovery", "low", key="MONID_API_KEY",
                        safe_default=False, notes="inspect tool schema/pricing/docs"),
    "monid_run": _m("monid_run", "agentic_discovery", "unknown", stateful=True,
                    key="MONID_API_KEY", safe_default=False,
                    notes="executes a discovered tool; paid/stateful; never default"),
    # --- specialized company/market research (Wokelo) ---
    "wokelo_research": _m("wokelo_research", "specialized_research", "unknown", key="WOKELO_API_KEY",
                          safe_default=False, notes="needs WOKELO_BASE_URL + endpoint path"),
    "wokelo_company_lookup": _m("wokelo_company_lookup", "specialized_research", "unknown",
                                key="WOKELO_API_KEY", safe_default=False,
                                notes="needs WOKELO_BASE_URL + endpoint path"),
    # --- OpenAI hosted browsing (opt-in expensive baseline) ---
    "openai_web_search": _m("openai_web_search", "search", "high", key="OPENAI_API_KEY",
                            safe_default=False, notes="hosted browsing; strong/expensive baseline"),
    "openai_web_search_low_context": _m("openai_web_search_low_context", "search", "high",
                                        key="OPENAI_API_KEY", safe_default=False),
    # --- future / not implemented ---
    "browser_use": _m("browser_use", "browserish", "high", stateful=True, key=None,
                      safe_default=False, implemented=False,
                      notes="DEFERRED: browser control; prompt-injection/state risk"),
}


def tool_meta(name: str) -> ToolMeta:
    """Metadata for ``name`` (an unknown tool gets a conservative default)."""
    return TOOL_META.get(name) or _m(name, "search", "unknown", safe_default=False)


def tools_meta_dict(names: list[str]) -> dict[str, Any]:
    return {n: tool_meta(n).to_dict() for n in names}


def env_key(name: str) -> Optional[str]:
    return tool_meta(name).requires_api_key


def is_first_hop(name: str) -> bool:
    """True if ``name`` may be a first-hop arm (a query tool, not a URL fetcher)."""
    return tool_meta(name).family in FIRST_HOP_FAMILIES


def is_followup(name: str) -> bool:
    """True if ``name`` is a follow-up tool that operates on a URL from evidence."""
    return tool_meta(name).family in FOLLOWUP_FAMILIES


def first_hop_tools(names: list[str]) -> list[str]:
    """The subset of ``names`` eligible as first-hop (search) arms, order preserved."""
    return [n for n in names if is_first_hop(n)]


def followup_tools(names: list[str]) -> list[str]:
    """The subset of ``names`` that are URL-operating follow-up tools."""
    return [n for n in names if is_followup(n)]
