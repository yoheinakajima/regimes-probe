"""Wokelo adapters (optional, SCAFFOLD): specialized company/market research.

Wokelo is promising for company/market research, but ``docs.wokelo.ai`` is
JS-rendered and the endpoint shape is not accessible in this environment, so we do
NOT guess it. These adapters are a **guarded generic HTTP scaffold**: they are
*unavailable* unless an explicit base URL + endpoint path are configured (env or
constructor), and otherwise fail closed with a clear message. Once the official
endpoint shape (or an OpenAPI spec) is known, wire it in.

Env / config:
  WOKELO_API_KEY        (required to call)
  WOKELO_BASE_URL       (required; no default — must be provided)
  WOKELO_RESEARCH_PATH  (endpoint path for wokelo_research)
  WOKELO_COMPANY_PATH   (endpoint path for wokelo_company_lookup)
  WOKELO_OPENAPI_PATH   (optional; reserved for loading a spec later)

Docs: https://docs.wokelo.ai/
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from decimal import Decimal
from typing import Any, Optional

from regimes_probe.tools.base import (
    ProviderUnavailable, SearchProvider, SearchResponse, SearchResult)


class _WokeloBase(SearchProvider):
    deterministic = False
    cost_per_call = Decimal("0")   # unknown until endpoint/pricing known
    _path_env = ""

    def __init__(self, *, base_url: Optional[str] = None, path: Optional[str] = None) -> None:
        self.base_url = base_url or os.environ.get("WOKELO_BASE_URL")
        self.path = path or os.environ.get(self._path_env)

    def _why_unavailable(self) -> Optional[str]:
        if not os.environ.get("WOKELO_API_KEY"):
            return "WOKELO_API_KEY is not set"
        if not self.base_url:
            return ("WOKELO_BASE_URL is not configured. docs.wokelo.ai is JS-rendered; "
                    "the endpoint shape is unknown here. Provide WOKELO_BASE_URL and the "
                    f"endpoint path ({self._path_env}) — or a WOKELO_OPENAPI_PATH spec — "
                    "before executing.")
        if not self.path:
            return f"{self._path_env} (endpoint path) is not configured for {self.name}"
        return None

    def available(self) -> bool:
        return self._why_unavailable() is None

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        why = self._why_unavailable()
        if why is not None:
            raise ProviderUnavailable(f"{self.name} unavailable (fail-closed): {why}")
        t0 = time.monotonic()
        url = self.base_url.rstrip("/") + "/" + self.path.lstrip("/")
        body = {"query": query, "limit": limit, **opts}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['WOKELO_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        rows = data.get("results") or data.get("data") or ([data] if data else [])
        results = [_to_result(r, i) for i, r in enumerate(rows[:limit])]
        return SearchResponse(provider=self.name, query=query, results=tuple(results),
                              cost=self.cost_per_call, latency_s=time.monotonic() - t0)


class WokeloResearch(_WokeloBase):
    name = "wokelo_research"
    _path_env = "WOKELO_RESEARCH_PATH"


class WokeloCompanyLookup(_WokeloBase):
    name = "wokelo_company_lookup"
    _path_env = "WOKELO_COMPANY_PATH"


def _to_result(r: dict[str, Any], i: int) -> SearchResult:
    if not isinstance(r, dict):
        r = {"snippet": str(r)}
    return SearchResult(
        title=r.get("title") or r.get("name", ""),
        url=r.get("url") or r.get("source", ""),
        snippet=(r.get("summary") or r.get("snippet") or r.get("description") or "")[:2000],
        rank=i, extra={"provider": "wokelo", "tool_family": "specialized_research"})


def wokelo_research(**kw) -> WokeloResearch:
    return WokeloResearch(**kw)


def wokelo_company_lookup(**kw) -> WokeloCompanyLookup:
    return WokeloCompanyLookup(**kw)
