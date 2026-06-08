"""Monid adapters (optional) — an *agentic tool-discovery* arm, not a search engine.

Monid discovers tools for a task, inspects a tool's schema/pricing/docs, and (when
explicitly allowed) runs a tool with structured input. For regimes-probe this lets
the contextual bandit learn a *tool-discovery* policy as one arm among many — it is
NOT an ordinary web-search provider.

Requires ``MONID_API_KEY``. ``monid_run`` is stateful/paid and is DISABLED by
default (gated behind ``allow_stateful_or_paid_tools``). All calls go through the
recording cache via the live invoker; nothing here runs in cheap/diverse mode
unless explicitly enabled.

Docs: https://docs.monid.ai/  (e.g. POST https://api.monid.ai/v1/discover with
Authorization: Bearer <key>).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from decimal import Decimal
from typing import Any, Optional

from regimes_probe.tools.base import (
    ProviderUnavailable, SearchProvider, SearchResponse, SearchResult, ToolDisabled)

_BASE = os.environ.get("MONID_BASE_URL", "https://api.monid.ai/v1")


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    key = os.environ["MONID_API_KEY"]
    req = urllib.request.Request(
        f"{_BASE}/{path.lstrip('/')}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


class _MonidBase(SearchProvider):
    deterministic = False

    def available(self) -> bool:
        return bool(os.environ.get("MONID_API_KEY"))


class MonidDiscover(_MonidBase):
    """Discover candidate tools for a task. family=agentic_discovery."""

    name = "monid_discover"
    cost_per_call = Decimal("0.004")

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        if not self.available():
            raise ProviderUnavailable("monid_discover requires MONID_API_KEY")
        t0 = time.monotonic()
        data = _post("discover", {"query": query, "limit": limit})
        rows = data.get("tools") or data.get("results") or data.get("data") or []
        results = [_tool_to_result(row, i) for i, row in enumerate(rows[:limit])]
        return SearchResponse(provider=self.name, query=query, results=tuple(results),
                              cost=self.cost_per_call, latency_s=time.monotonic() - t0)


class MonidInspect(_MonidBase):
    """Inspect a discovered tool's schema/pricing/docs. ``query`` = tool/endpoint id."""

    name = "monid_inspect"
    cost_per_call = Decimal("0.002")

    def search(self, query: str, *, limit: int = 1, **opts: Any) -> SearchResponse:
        if not self.available():
            raise ProviderUnavailable("monid_inspect requires MONID_API_KEY")
        t0 = time.monotonic()
        # tool/endpoint id passed as the query; exact endpoint may evolve.
        data = _post("inspect", {"tool_id": query, **opts})
        results = (_tool_to_result(data, 0),) if data else ()
        return SearchResponse(provider=self.name, query=query, results=results,
                              cost=self.cost_per_call, latency_s=time.monotonic() - t0)


class MonidRun(_MonidBase):
    """Run a discovered tool with structured input. STATEFUL/PAID — disabled by default."""

    name = "monid_run"
    cost_per_call = Decimal("0")   # unknown; tool-dependent

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled

    def search(self, query: str, *, limit: int = 1, **opts: Any) -> SearchResponse:
        if not self.enabled:
            raise ToolDisabled(
                "monid_run is disabled by default (stateful/paid). Enable explicitly "
                "with --allow-stateful-or-paid-tools; never used in cheap/diverse modes.")
        if not self.available():
            raise ProviderUnavailable("monid_run requires MONID_API_KEY")
        t0 = time.monotonic()
        data = _post("run", {"tool_id": query, "input": opts.get("input", {})})
        results = (_tool_to_result(data, 0),) if data else ()
        return SearchResponse(provider=self.name, query=query, results=results,
                              cost=self.cost_per_call, latency_s=time.monotonic() - t0)


def _tool_to_result(row: dict[str, Any], i: int) -> SearchResult:
    """Normalize a Monid tool record into an evidence-like SearchResult."""
    name = row.get("name") or row.get("tool") or row.get("id") or ""
    desc = row.get("description") or row.get("summary") or ""
    pricing = row.get("pricing") or row.get("price") or ""
    schema = row.get("input_schema") or row.get("schema") or {}
    schema_summary = ", ".join(sorted(schema.get("properties", {}).keys())) if isinstance(schema, dict) else ""
    snippet = f"{desc}".strip()
    if pricing:
        snippet += f" | pricing: {pricing}"
    if schema_summary:
        snippet += f" | inputs: {schema_summary}"
    return SearchResult(
        title=name, url=row.get("url") or row.get("docs") or row.get("source") or "",
        snippet=snippet, rank=i,
        extra={"provider": "monid", "tool_family": "agentic_discovery",
               "tool_id": row.get("id"), "pricing": pricing,
               "schema_summary": schema_summary})


def monid_discover() -> MonidDiscover:
    return MonidDiscover()


def monid_inspect() -> MonidInspect:
    return MonidInspect()


def monid_run(*, enabled: bool = False) -> MonidRun:
    return MonidRun(enabled=enabled)
