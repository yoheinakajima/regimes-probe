"""Search/fetch provider adapter interface.

A *provider adapter* is the only place network I/O is allowed. Adapters are
wrapped as ActiveGraph tools in ``activegraph_pack.tools`` so every call is
recorded as a ``tool.requested`` / ``tool.responded`` event pair and is
replayable.

The router NEVER contains provider-specific logic: it sees only the common
``SearchProvider`` interface and the adapter ``name``. This keeps tool families
swappable and lets the contextual bandit learn tool-selection differences.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from typing import Any, Optional


@dataclass(frozen=True)
class SearchResult:
    """One result returned by a provider adapter.

    Fields are deliberately provider-agnostic. ``source_authority`` and
    ``published_at`` are best-effort; adapters that cannot populate them leave
    the defaults and the evidence layer treats them as unknown.
    """

    title: str
    url: str
    snippet: str
    published_at: Optional[str] = None          # ISO date, best-effort
    source_authority: float = 0.5               # 0..1 prior on the domain
    rank: int = 0                               # position in the result list
    extra: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Stable hash of the page-identifying content (url + snippet)."""
        h = hashlib.sha256()
        h.update(self.url.encode("utf-8"))
        h.update(b"\x00")
        h.update(self.snippet.encode("utf-8"))
        return h.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResponse:
    """A provider's response to one query: results plus accounting metadata."""

    provider: str
    query: str
    results: tuple[SearchResult, ...]
    cost: Decimal = Decimal("0")
    latency_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "cost": str(self.cost),
            "latency_s": self.latency_s,
            "error": self.error,
        }


class SearchProvider(ABC):
    """Common interface for every search/fetch tool family.

    Implementations must NOT be imported or constructed inside ActiveGraph
    behavior bodies. They are invoked only through the ActiveGraph tool wrapper.
    """

    #: stable adapter name; also the contextual-bandit arm id for routing.
    name: str = "abstract_provider"
    #: nominal per-call cost in USD, used by the router's cost penalty.
    cost_per_call: Decimal = Decimal("0")
    #: whether two calls with identical args return identical results. Fake and
    #: fixture-backed adapters are deterministic; live adapters are not.
    deterministic: bool = False
    #: whether this adapter fetches a page rather than running a search.
    is_fetch: bool = False

    def available(self) -> bool:
        """Whether the adapter can run now (keys present, deps installed).

        Default: available. Live adapters override to check env vars. Tests
        must never depend on a live adapter being available.
        """
        return True

    @abstractmethod
    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        """Run one search/fetch and return a :class:`SearchResponse`."""
        raise NotImplementedError


class ProviderUnavailable(RuntimeError):
    """Raised when a live adapter is invoked without its credentials/deps."""


class ToolDisabled(RuntimeError):
    """Raised when a tool exists but is disabled by a safety flag (e.g. stateful/
    paid actions, browser-like interaction) and was not explicitly enabled."""
