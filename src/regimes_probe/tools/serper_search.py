"""Serper.dev (Google SERP) adapter (optional). Requires ``SERPER_API_KEY``."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from decimal import Decimal
from typing import Any

from regimes_probe.tools.base import (
    ProviderUnavailable,
    SearchProvider,
    SearchResponse,
    SearchResult,
)

_ENDPOINT = "https://google.serper.dev/search"


class SerperSearch(SearchProvider):
    name = "serper_search"
    deterministic = False
    cost_per_call = Decimal("0.002")

    def available(self) -> bool:
        return bool(os.environ.get("SERPER_API_KEY"))

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        if not self.available():
            raise ProviderUnavailable("serper_search requires SERPER_API_KEY")
        body: dict[str, Any] = {"q": query, "num": limit}
        if "allowed_domains" in opts and opts["allowed_domains"]:
            body["q"] = query + " " + " OR ".join(
                f"site:{d}" for d in opts["allowed_domains"]
            )
        req = urllib.request.Request(
            _ENDPOINT,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": os.environ["SERPER_API_KEY"],
            },
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        rows = data.get("organic", [])
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                published_at=r.get("date"),
                rank=i,
            )
            for i, r in enumerate(rows[:limit])
        ]
        return SearchResponse(
            provider=self.name,
            query=query,
            results=tuple(results),
            cost=self.cost_per_call,
            latency_s=time.monotonic() - t0,
        )
