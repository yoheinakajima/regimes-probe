"""Generic web-search provider interface + a thin HTTP reference adapter.

``GenericWebSearch`` defines the provider-agnostic shape every independent
search adapter follows. It is not bound to any vendor; ``news_search`` and
``official_domain_search`` are expressed as configurations of the same shape.

A concrete network call requires an explicit ``endpoint`` and an env-var key
name; without them the adapter is unavailable, so tests never hit the network.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from decimal import Decimal
from typing import Any, Callable, Optional

from regimes_probe.tools.base import (
    ProviderUnavailable,
    SearchProvider,
    SearchResponse,
    SearchResult,
)


class GenericWebSearch(SearchProvider):
    """Reference adapter: POST a JSON query to ``endpoint`` and map results.

    ``parse`` maps the decoded JSON response to a list of ``SearchResult``.
    Subclasses (news/official-domain) override defaults rather than the router
    learning anything provider-specific.
    """

    deterministic = False

    def __init__(
        self,
        *,
        name: str = "generic_web_search",
        endpoint: Optional[str] = None,
        api_key_env: Optional[str] = None,
        parse: Optional[Callable[[dict[str, Any], int], list[SearchResult]]] = None,
        cost_per_call: Decimal | str = "0.005",
        default_opts: Optional[dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self._parse = parse or _default_parse
        self.cost_per_call = (
            cost_per_call if isinstance(cost_per_call, Decimal) else Decimal(str(cost_per_call))
        )
        self._default_opts = default_opts or {}

    def available(self) -> bool:
        if self.endpoint is None:
            return False
        if self.api_key_env and not os.environ.get(self.api_key_env):
            return False
        return True

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        if not self.available():
            raise ProviderUnavailable(
                f"{self.name} needs endpoint + {self.api_key_env or 'no key'}"
            )
        body = {"query": query, "limit": limit, **self._default_opts, **opts}
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        req = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode(), headers=headers
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        return SearchResponse(
            provider=self.name,
            query=query,
            results=tuple(self._parse(data, limit)),
            cost=self.cost_per_call,
            latency_s=time.monotonic() - t0,
        )


def _default_parse(data: dict[str, Any], limit: int) -> list[SearchResult]:
    rows = data.get("results") or data.get("data") or []
    out: list[SearchResult] = []
    for i, row in enumerate(rows[:limit]):
        out.append(
            SearchResult(
                title=row.get("title", ""),
                url=row.get("url", row.get("link", "")),
                snippet=row.get("snippet", row.get("description", "")),
                published_at=row.get("published_at") or row.get("date"),
                source_authority=float(row.get("authority", 0.5)),
                rank=i,
            )
        )
    return out


def news_search(endpoint: Optional[str] = None, api_key_env: str = "NEWS_API_KEY") -> GenericWebSearch:
    """Freshness-oriented search interface (sorted by recency)."""
    return GenericWebSearch(
        name="news_search",
        endpoint=endpoint,
        api_key_env=api_key_env,
        default_opts={"sort": "date"},
        cost_per_call="0.004",
    )


def official_domain_search(
    endpoint: Optional[str] = None,
    api_key_env: str = "OFFICIAL_SEARCH_API_KEY",
    allowed_domains: Optional[list[str]] = None,
) -> GenericWebSearch:
    """Constrained search restricted to authoritative/allowed domains."""
    return GenericWebSearch(
        name="official_domain_search",
        endpoint=endpoint,
        api_key_env=api_key_env,
        default_opts={"allowed_domains": allowed_domains or []},
        cost_per_call="0.006",
    )
