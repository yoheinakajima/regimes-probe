"""Live provider + answerer builders (env-gated, cache-wrapped, opt-in).

Builds the OpenAI Responses answerer, the OpenAI ``web_search`` tool adapter, the
page-fetch adapter, and any keyed independent search adapters already in
``regimes_probe.tools``. Every live call flows through a
:class:`~regimes_probe.live.cache.RecordingCache`, and every wrapper carries an
``armed`` flag: when not armed (dry-run) any attempt to actually call a provider
raises instead of spending.

Credentials are read from environment variables only and never stored.
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Optional

from regimes_probe.activegraph_pack.tools import payload_to_response
from regimes_probe.agent.answerer import CandidateAnswer
from regimes_probe.agent.evidence import EvidenceObservation
from regimes_probe.live.cache import RecordingCache, ReplayMiss
from regimes_probe.tools.base import (
    ProviderUnavailable, SearchProvider, SearchResponse, safe_search)

# Env var each adapter needs (None = no key required). Derived from the central
# tool metadata registry so there is a single source of truth.
from regimes_probe.tools.metadata import TOOL_META as _TOOL_META

_ADAPTER_ENV = {name: meta.requires_api_key for name, meta in _TOOL_META.items()}


class NotArmed(RuntimeError):
    """Raised when a live call is attempted in dry-run (un-armed) mode."""


class CachedProvider(SearchProvider):
    """Wrap a live provider so calls are cached/replayable and spend-guarded."""

    def __init__(self, inner: SearchProvider, cache: RecordingCache, *, armed: bool) -> None:
        self.inner = inner
        self.cache = cache
        self.armed = armed
        self.name = inner.name
        self.cost_per_call = getattr(inner, "cost_per_call", Decimal("0"))
        self.deterministic = False
        self.is_fetch = getattr(inner, "is_fetch", False)

    def available(self) -> bool:
        return self.inner.available()

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        meta = {"query": query, "limit": limit, "opts": opts}
        h = self.cache.request_hash(self.name, self.name, meta)
        entry = self.cache.get(h)
        if entry and self.cache.mode in ("auto", "replay"):
            self.cache.hits += 1
            return payload_to_response(entry["response"])
        if self.cache.mode == "replay":
            raise ReplayMiss(f"{self.name}: no cached response for this request (replay mode)")
        if not self.armed:
            raise NotArmed(f"{self.name}: refusing to call provider in dry-run "
                           "(pass --execute to arm live calls)")
        # API/network errors become a recorded FAILED response (not a crash);
        # config/preflight errors still raise. safe_search is the boundary.
        resp = safe_search(self.inner, query, limit=limit, **opts)   # the only network
        self.cache.calls += 1
        if self.cache.mode != "off":               # cache failures too, so reruns/replay
            self.cache.store(h, provider=self.name, name=self.name,   # don't re-trigger them
                             request_meta=meta, response_payload=resp.to_dict())
        return resp


# ----------------------------------------------------------------- answerers
class _BaseLiveAnswerer:
    model: str
    cache: RecordingCache
    armed: bool
    prompt_name: str

    def available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return True

    def _call_openai(self, instructions: str, user_input: str) -> str:
        from openai import OpenAI  # local import: never at module load
        client = OpenAI()
        resp = client.responses.create(model=self.model, instructions=instructions,
                                        input=user_input)
        return (getattr(resp, "output_text", "") or "").strip()

    def _cached_or_call(self, meta: dict[str, Any], instructions: str,
                        user_input: str) -> str:
        h = self.cache.request_hash("openai_responses", self.model, meta)
        entry = self.cache.get(h)
        if entry and self.cache.mode in ("auto", "replay"):
            self.cache.hits += 1
            return entry["response"].get("text", "")
        if self.cache.mode == "replay":
            raise ReplayMiss(f"answerer: no cached response (replay mode)")
        if not self.armed:
            raise NotArmed("answerer: refusing to call model in dry-run "
                           "(pass --execute to arm live calls)")
        text = self._call_openai(instructions, user_input)
        self.cache.calls += 1
        if self.cache.mode != "off":
            self.cache.store(h, provider="openai_responses", name=self.model,
                             request_meta=meta, response_payload={"text": text})
        return text


class LiveAnswerer(_BaseLiveAnswerer):
    """Evidence-grounded answerer for the search conditions (OpenAI Responses)."""

    name = "live_answerer"
    prompt_name = "answerer"

    def __init__(self, model: str, cache: RecordingCache, *, armed: bool) -> None:
        self.model = model
        self.cache = cache
        self.armed = armed

    def answer(self, observations: list[EvidenceObservation], *, item=None) -> CandidateAnswer:
        from regimes_probe.agent import prompts
        supports = [o for o in observations if o.supports]
        evidence = [{"url": o.url, "snippet": o.snippet} for o in supports]
        if not evidence:
            return CandidateAnswer(None, 0, 0.0, [])
        meta = {"model": self.model, "prompt": prompts.fingerprint("answerer"),
                "question": (item.question if item else ""), "evidence": evidence}
        instr = prompts.get("answerer").content
        user = (f"Question: {item.question if item else ''}\n\nEvidence:\n"
                + "\n".join(f"- ({e['url']}) {e['snippet']}" for e in evidence))
        text = self._cached_or_call(meta, instr, user)
        ans = None if (not text or text.strip().upper() == "ABSTAIN") else text.strip()
        auth = max((o.source_authority for o in supports), default=0.0)
        return CandidateAnswer(ans, len(supports), auth, [o.url for o in supports])


class LiveClosedBookAnswerer(_BaseLiveAnswerer):
    """Closed-book answerer (no tools): intrinsic-knowledge estimate."""

    name = "live_closed_book_answerer"
    prompt_name = "closed_book"

    def __init__(self, model: str, cache: RecordingCache, *, armed: bool) -> None:
        self.model = model
        self.cache = cache
        self.armed = armed

    def answer(self, observations: list[EvidenceObservation], *, item=None) -> CandidateAnswer:
        from regimes_probe.agent import prompts
        meta = {"model": self.model, "prompt": prompts.fingerprint("closed_book"),
                "question": (item.question if item else "")}
        text = self._cached_or_call(meta, prompts.get("closed_book").content,
                                    f"Question: {item.question if item else ''}")
        ans = None if (not text or text.strip().upper() == "ABSTAIN") else text.strip()
        return CandidateAnswer(ans, 0, 1.0 if ans else 0.0, ["intrinsic:closed_book"])


# ----------------------------------------------------------------- builders
def _build_inner(name: str, *, web_search_model: str = "gpt-5.4-mini",
                 web_search_context: str = "low",
                 allow_stateful_or_paid: bool = False,
                 enable_browserish: bool = False) -> Optional[SearchProvider]:
    if name in ("openai_web_search", "openai_web_search_low_context"):
        from regimes_probe.tools.openai_web_search import (
            openai_web_search, openai_web_search_low_context)
        # model/context come from the CALLER (CLI/config), never hardcoded here.
        if name.endswith("low_context"):
            return openai_web_search_low_context(model=web_search_model)
        return openai_web_search(model=web_search_model, context_size=web_search_context)
    # --- Firecrawl ---
    if name == "firecrawl_search":
        from regimes_probe.tools.firecrawl import firecrawl_search
        return firecrawl_search()
    if name == "firecrawl_scrape":
        from regimes_probe.tools.firecrawl import firecrawl_scrape
        return firecrawl_scrape()
    if name == "firecrawl_interact":   # browser-like; only callable if enabled
        from regimes_probe.tools.firecrawl import firecrawl_interact
        return firecrawl_interact(enabled=enable_browserish)
    # --- Monid (agentic tool discovery) ---
    if name == "monid_discover":
        from regimes_probe.tools.monid import monid_discover
        return monid_discover()
    if name == "monid_inspect":
        from regimes_probe.tools.monid import monid_inspect
        return monid_inspect()
    if name == "monid_run":            # stateful/paid; only callable if allowed
        from regimes_probe.tools.monid import monid_run
        return monid_run(enabled=allow_stateful_or_paid)
    # --- Wokelo (specialized research; fails closed without base url/path) ---
    if name == "wokelo_research":
        from regimes_probe.tools.wokelo import wokelo_research
        return wokelo_research()
    if name == "wokelo_company_lookup":
        from regimes_probe.tools.wokelo import wokelo_company_lookup
        return wokelo_company_lookup()
    if name == "page_fetch":
        from regimes_probe.tools.page_fetch import PageFetch
        return PageFetch()
    if name == "brave_search":
        from regimes_probe.tools.brave_search import BraveSearch
        return BraveSearch()
    if name == "tavily_search":
        from regimes_probe.tools.tavily_search import TavilySearch
        return TavilySearch()
    if name == "exa_search":
        from regimes_probe.tools.exa_search import ExaSearch
        return ExaSearch()
    if name == "serper_search":
        from regimes_probe.tools.serper_search import SerperSearch
        return SerperSearch()
    if name in ("news_search", "official_domain_search", "generic_web_search"):
        from regimes_probe.tools.generic_web_search import (
            GenericWebSearch, news_search, official_domain_search)
        if name == "news_search":
            return news_search()
        if name == "official_domain_search":
            return official_domain_search()
        return GenericWebSearch(name="generic_web_search")
    return None


def build_live_providers(tool_names: list[str], *, cache: RecordingCache, armed: bool,
                         web_search_model: str = "gpt-5.4-mini",
                         web_search_context: str = "low",
                         allow_stateful_or_paid: bool = False,
                         enable_browserish: bool = False) -> dict[str, SearchProvider]:
    """Construct + cache-wrap the requested live providers (no network at build).

    Each provider becomes a separate bandit arm. ``web_search_model``/
    ``web_search_context`` apply to the OpenAI hosted adapter only. Stateful/paid
    (``monid_run``) and browser-like (``firecrawl_interact``) tools are built in a
    DISABLED state unless their allow flag is set.
    """
    providers: dict[str, SearchProvider] = {}
    for name in tool_names:
        inner = _build_inner(name, web_search_model=web_search_model,
                             web_search_context=web_search_context,
                             allow_stateful_or_paid=allow_stateful_or_paid,
                             enable_browserish=enable_browserish)
        if inner is None:
            continue
        providers[name] = CachedProvider(inner, cache, armed=armed)
    return providers


def build_live_answerer(kind: str, *, model: str, cache: RecordingCache, armed: bool):
    if kind == "closed_book":
        return LiveClosedBookAnswerer(model, cache, armed=armed)
    return LiveAnswerer(model, cache, armed=armed)


def build_task_frame_model_fn(model: str, cache: RecordingCache, *, armed: bool):
    """A cached/replayable ``model_fn(prompt)->str`` for the LLM task-frame parser.

    Routes through the same :class:`RecordingCache` as the answerer: a dry-run
    (un-armed) raises :class:`NotArmed`, a replay miss raises :class:`ReplayMiss`,
    and a successful call is recorded. The parser catches any exception and falls
    back to the deterministic parser (recording ``fallback_reason``). The model is
    NOT called to answer the question — only to parse it into task state.
    """
    def _model_fn(prompt_text: str) -> str:
        from regimes_probe.agent import prompts
        meta = {"model": model, "prompt": prompts.fingerprint("task_frame_parser"),
                "input": prompt_text}
        h = cache.request_hash("openai_responses_task_frame", model, meta)
        entry = cache.get(h)
        if entry and cache.mode in ("auto", "replay"):
            cache.hits += 1
            return entry["response"].get("text", "")
        if cache.mode == "replay":
            raise ReplayMiss("task_frame_parser: no cached response (replay mode)")
        if not armed:
            raise NotArmed("task_frame_parser: refusing to call model in dry-run "
                           "(pass --execute to arm live calls)")
        from openai import OpenAI  # local import: never at module load
        client = OpenAI()
        resp = client.responses.create(model=model, input=prompt_text)
        text = (getattr(resp, "output_text", "") or "").strip()
        cache.calls += 1
        if cache.mode != "off":
            cache.store(h, provider="openai_responses_task_frame", name=model,
                        request_meta=meta, response_payload={"text": text})
        return text
    return _model_fn


def build_frontier_model_fn(model: str, cache: RecordingCache, *, armed: bool):
    """A cached/replayable ``model_fn(prompt)->str`` for the LLM frontier proposer.

    Same discipline as the task-frame parser: dry-run raises :class:`NotArmed`, replay
    miss raises :class:`ReplayMiss`, success is recorded. The model PROPOSES research
    actions only — it never answers; deterministic code validates/scores/executes."""
    def _model_fn(prompt_text: str) -> str:
        from regimes_probe.agent import prompts
        meta = {"model": model, "prompt": prompts.fingerprint("frontier_planner"),
                "input": prompt_text}
        h = cache.request_hash("openai_responses_frontier", model, meta)
        entry = cache.get(h)
        if entry and cache.mode in ("auto", "replay"):
            cache.hits += 1
            return entry["response"].get("text", "")
        if cache.mode == "replay":
            raise ReplayMiss("frontier_planner: no cached response (replay mode)")
        if not armed:
            raise NotArmed("frontier_planner: refusing to call model in dry-run "
                           "(pass --execute to arm live calls)")
        from openai import OpenAI  # local import: never at module load
        client = OpenAI()
        resp = client.responses.create(model=model, input=prompt_text, temperature=0)
        text = (getattr(resp, "output_text", "") or "").strip()
        cache.calls += 1
        if cache.mode != "off":
            cache.store(h, provider="openai_responses_frontier", name=model,
                        request_meta=meta, response_payload={"text": text})
        return text
    return _model_fn


def build_evidence_model_fn(model: str, cache: RecordingCache, *, armed: bool):
    """A cached/replayable ``model_fn(prompt)->str`` for the LLM evidence interpreter.

    Same discipline as the frontier proposer: dry-run raises :class:`NotArmed`, replay
    miss raises :class:`ReplayMiss`, success is recorded. The model only RE-CLASSIFIES a
    single result's source role from bounded snippets — it never answers."""
    def _model_fn(prompt_text: str) -> str:
        meta = {"model": model, "task": "evidence_source_role", "input": prompt_text}
        h = cache.request_hash("openai_responses_evidence", model, meta)
        entry = cache.get(h)
        if entry and cache.mode in ("auto", "replay"):
            cache.hits += 1
            return entry["response"].get("text", "")
        if cache.mode == "replay":
            raise ReplayMiss("evidence_interpreter: no cached response (replay mode)")
        if not armed:
            raise NotArmed("evidence_interpreter: refusing to call model in dry-run "
                           "(pass --execute to arm live calls)")
        from openai import OpenAI  # local import: never at module load
        client = OpenAI()
        resp = client.responses.create(model=model, input=prompt_text, temperature=0)
        text = (getattr(resp, "output_text", "") or "").strip()
        cache.calls += 1
        if cache.mode != "off":
            cache.store(h, provider="openai_responses_evidence", name=model,
                        request_meta=meta, response_payload={"text": text})
        return text
    return _model_fn


def missing_keys(tool_names: list[str], *, answer_model_needs_openai: bool = True) -> list[str]:
    """Env-var NAMES that are required but absent (for execute/strict errors)."""
    needed: set[str] = set()
    if answer_model_needs_openai:
        needed.add("OPENAI_API_KEY")
    for name in tool_names:
        env = _ADAPTER_ENV.get(name)
        if env:
            needed.add(env)
    return sorted(k for k in needed if not os.environ.get(k))
