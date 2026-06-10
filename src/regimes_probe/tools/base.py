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
    error_meta: Optional[dict[str, Any]] = None   # structured failure (sanitized)
    #: 5j-B: read/fetch accounting (e.g. fetched_chars vs stored_body_chars + truncation
    #: flags). Additive + optional, so non-fetch tools and the synthetic demo are unaffected.
    fetch_meta: Optional[dict[str, Any]] = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "provider": self.provider,
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "cost": str(self.cost),
            "latency_s": self.latency_s,
            "error": self.error,
            "error_meta": self.error_meta,
        }
        if self.fetch_meta is not None:
            d["fetch_meta"] = self.fetch_meta
        return d


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


# ---------------------------------------------------------------------------
# Tool-call failure handling.
#
# A provider/API error during an INDIVIDUAL tool call must NOT crash the
# benchmark — it becomes a recorded *failed* SearchResponse (error set) so the
# agent can continue, the bandit can learn the tool failed, and replay can
# reproduce the failure. Config/preflight/replay-control errors are NOT swallowed.
# ---------------------------------------------------------------------------
import os as _os
import re as _re

#: Exception class NAMES that are config/preflight/replay errors — these must
#: propagate (they indicate misconfiguration, not a transient API failure).
_PASSTHROUGH_ERROR_NAMES = frozenset({
    "ProviderUnavailable", "NotArmed", "ToolDisabled", "ReplayMiss",
})

_SECRET_TOKEN_RE = _re.compile(
    r'(?i)(bearer\s+[A-Za-z0-9._\-]+|"?api_?key"?\s*[:=]\s*"?[A-Za-z0-9._\-]+"?'
    r'|sk-[A-Za-z0-9]+|tvly-[A-Za-z0-9]+)')


def _known_secret_values() -> list[str]:
    out = []
    for k, v in _os.environ.items():
        if v and len(v) >= 8 and any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET")):
            out.append(v)
    return out


def sanitize_error_text(text: str, *, max_len: int = 500) -> str:
    """Redact secrets from an error string and truncate. Never leaks keys."""
    if not text:
        return ""
    for val in _known_secret_values():           # redact actual key values
        text = text.replace(val, "<redacted>")
    text = _SECRET_TOKEN_RE.sub("<redacted>", text)  # redact key-shaped tokens
    text = " ".join(text.split())
    return text[:max_len]


def normalize_error(provider: str, tool: str, exc: BaseException) -> dict[str, Any]:
    """Build a sanitized, secret-free structured error record for a failed call."""
    status_code = getattr(exc, "code", None)      # HTTPError.code
    body = ""
    read = getattr(exc, "read", None)             # HTTPError has a readable body
    if callable(read):
        try:
            raw = read()
            body = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        except Exception:
            body = ""
    reason = getattr(exc, "reason", "")
    message = sanitize_error_text(f"{exc} {reason} {body}".strip())
    return {
        "tool": tool,
        "provider": provider,
        "error_type": type(exc).__name__,
        "status_code": status_code,
        "message": message,
        "success": False,
    }


def failed_response(provider: str, query: str, exc: BaseException, *,
                    tool: Optional[str] = None, cost: Decimal = Decimal("0"),
                    latency_s: float = 0.0) -> SearchResponse:
    meta = normalize_error(provider, tool or provider, exc)
    return SearchResponse(provider=provider, query=query, results=(), cost=cost,
                          latency_s=latency_s, error=meta["message"] or meta["error_type"],
                          error_meta=meta)


def is_passthrough_error(exc: BaseException) -> bool:
    """True for config/preflight/replay errors that must NOT be swallowed."""
    if isinstance(exc, ProviderUnavailable):
        return True
    return type(exc).__name__ in _PASSTHROUGH_ERROR_NAMES


def safe_search(provider: "SearchProvider", query: str, *, limit: int = 5,
                **opts: Any) -> SearchResponse:
    """Call ``provider.search`` but turn API/network errors into a failed
    SearchResponse instead of raising. Config/preflight/replay errors still raise.

    Cost is still attributed for a failed call (the provider may have been hit).
    """
    import time as _time
    t0 = _time.monotonic()
    try:
        return provider.search(query, limit=limit, **opts)
    except BaseException as exc:                   # noqa: BLE001 — deliberate boundary
        if is_passthrough_error(exc):
            raise
        return failed_response(
            getattr(provider, "name", "unknown"), query, exc,
            tool=getattr(provider, "name", "unknown"),
            cost=getattr(provider, "cost_per_call", Decimal("0")),
            latency_s=_time.monotonic() - t0)
