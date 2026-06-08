"""ActiveGraph tool boundary + recording / replay tool invokers.

This module is the *only* place a search/fetch provider is invoked during a
recorded run. Each invocation emits a ``tool.requested`` / ``tool.responded``
event pair and creates a ``tool_call`` object — exactly the recorded-tool
pattern ActiveGraph uses (CONTRACT v0.7), so replay reads responses back from
the log instead of re-invoking the network.

  * :class:`RecordingInvoker` — live: calls the provider, records the pair.
  * :class:`ReplayInvoker`    — deterministic: replays recorded responses in
                                order, asserting the request matches. No network.

The provider call (the actual I/O) happens here, never inside a behavior body.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

from regimes_probe.activegraph_pack.events import Events
from regimes_probe.activegraph_pack.objects import Objects
from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult


def response_to_payload(resp: SearchResponse) -> dict[str, Any]:
    """Serialize a response for the event log (replayable)."""
    return resp.to_dict()


def payload_to_response(payload: dict[str, Any]) -> SearchResponse:
    """Reconstruct a :class:`SearchResponse` from a recorded payload."""
    results = tuple(SearchResult(**r) for r in payload.get("results", []))
    return SearchResponse(
        provider=payload["provider"],
        query=payload["query"],
        results=results,
        cost=Decimal(str(payload.get("cost", "0"))),
        latency_s=float(payload.get("latency_s", 0.0)),
        error=payload.get("error"),
        error_meta=payload.get("error_meta"),
    )


class RecordingInvoker:
    """Invoke providers and record each call as an event pair + tool_call object."""

    def __init__(self, providers: dict[str, SearchProvider], log: "EventLog",
                 *, attempt_id: Optional[str] = None) -> None:
        self.providers = providers
        self.log = log
        self.attempt_id = attempt_id
        self.recorded: list[dict[str, Any]] = []

    def call(self, tool: str, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        req = self.log.emit(
            Events.TOOL_REQUESTED,
            {"tool": tool, "query": query, "opts": opts, "limit": limit,
             "attempt_id": self.attempt_id},
        )
        from regimes_probe.tools.base import safe_search
        # safe_search converts provider/API errors into a recorded FAILED response
        # (config/preflight errors still raise); a tool error never crashes the run.
        resp = safe_search(self.providers[tool], query, limit=limit, **opts)   # the only I/O
        payload = response_to_payload(resp)
        self.log.emit(
            Events.TOOL_RESPONDED,
            {"tool": tool, "query": query, "n_results": len(resp.results),
             "cost": str(resp.cost), "failed": resp.failed, "error": resp.error,
             "response": payload, "attempt_id": self.attempt_id},
            caused_by=req.id,
        )
        self.log.add_object(
            Objects.TOOL_CALL,
            {"tool": tool, "query": query, "n_results": len(resp.results),
             "cost": str(resp.cost), "failed": resp.failed, "attempt_id": self.attempt_id},
            caused_by=req.id,
        )
        self.recorded.append({"tool": tool, "query": query, "response": payload})
        return resp


class ReplayInvoker:
    """Replay recorded tool responses in order. Deterministic, no network."""

    def __init__(self, recorded: list[dict[str, Any]]) -> None:
        self._recorded = list(recorded)
        self._i = 0

    def call(self, tool: str, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        if self._i >= len(self._recorded):
            raise AssertionError("replay exhausted: more calls than recorded")
        rec = self._recorded[self._i]
        self._i += 1
        if rec["tool"] != tool or rec["query"] != query:
            raise AssertionError(
                f"replay divergence at call {self._i - 1}: "
                f"recorded ({rec['tool']!r},{rec['query']!r}) != "
                f"live ({tool!r},{query!r})"
            )
        return payload_to_response(rec["response"])


# Optional: expose providers as real @tool-decorated callables for a future
# Runtime-driven integration. Import is lazy so the pack works without the
# ActiveGraph tool registry being required at import time.
def register_provider_tools(providers: dict[str, SearchProvider]) -> list[Any]:
    """Register each provider as an ActiveGraph ``@tool``. Best-effort."""
    try:
        from activegraph.tools import tool as ag_tool
    except Exception:
        return []
    tools = []
    for name, provider in providers.items():
        def _body(args, ctx, _p=provider):  # pragma: no cover - runtime path
            resp = _p.search(args["query"], limit=args.get("limit", 5))
            return response_to_payload(resp)
        tools.append(
            ag_tool(name=name, description=f"search via {name}",
                    deterministic=getattr(provider, "deterministic", False),
                    cost_per_call=getattr(provider, "cost_per_call", Decimal("0")))(_body)
        )
    return tools


# Late import to avoid a cycle (behaviors imports tools).
from regimes_probe.activegraph_pack.behaviors import EventLog  # noqa: E402
