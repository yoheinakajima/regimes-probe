"""The budgeted epistemic search loop (the agent hot path).

For one question the loop:

  question → signature → routing_plan (Level 1) → for each step:
  query_plan (Level 2) → tool call → evidence → candidate → verification →
  stop/continue decision (Level 2) — all under a hard budget cap.

The loop is **gold-free**: it never sees the answer. It produces an
:class:`AttemptTrace` of everything it did (answer-free except the transient
``asserts`` on observations, which the cold-path reward/grader use and which is
never persisted to policy memory).

Tool calls go through a :class:`ToolInvoker` so the ActiveGraph pack can record
each as a ``tool.requested`` / ``tool.responded`` event pair for replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional, Protocol

from regimes_probe.agent.answerer import CandidateAnswer, DeterministicAnswerer
from regimes_probe.agent.evidence import EvidenceObservation, score_observation
from regimes_probe.datasets.base import Item
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.query_policy import QueryPolicy
from regimes_probe.policy.router import Router
from regimes_probe.policy.signatures import QuerySignature
from regimes_probe.policy.stopping_policy import StoppingPolicy
from regimes_probe.policy.verification_policy import (
    VerificationConfig,
    VerificationState,
    verify,
)
from regimes_probe.tools.base import SearchProvider, SearchResponse


class ToolInvoker(Protocol):
    """How the loop calls a tool. Implementations may record events."""

    def call(self, tool: str, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        ...


class LoopRecorder:
    """Hooks the loop calls at each seam so events can be emitted in causal
    order. The default is a no-op; the ActiveGraph pack supplies a recorder
    that emits ``routing_plan.created``, ``query_plan.created``,
    ``evidence.observed``, ``candidate_answer.created``,
    ``verification.completed`` and ``stop_decision.created`` events.
    """

    def on_routing_plan(self, plan: dict[str, Any]) -> None: ...
    def on_query_plan(self, step: int, plan: dict[str, Any]) -> None: ...
    def on_evidence(self, step: int, observations: list[EvidenceObservation]) -> None: ...
    def on_candidate(self, step: int, candidate: "CandidateAnswer") -> None: ...
    def on_verification(self, step: int, vstate: VerificationState) -> None: ...
    def on_stop(self, step: int, decision: dict[str, Any]) -> None: ...


class DirectInvoker:
    """Default invoker: call the provider directly (used in unit tests)."""

    def __init__(self, providers: dict[str, SearchProvider]) -> None:
        self.providers = providers

    def call(self, tool: str, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        # safe_search turns provider/API errors into a recorded failed response
        # (config/preflight errors still raise) so a tool error never crashes the run.
        from regimes_probe.tools.base import safe_search
        return safe_search(self.providers[tool], query, limit=limit, **opts)


@dataclass
class CallRecord:
    call_index: int
    tool: str
    query_arm: str
    query: str
    cost: float
    latency: float
    observations: list[EvidenceObservation]
    stop_arm: str
    supported: bool
    failed: bool = False
    error_type: Optional[str] = None
    status_code: Optional[int] = None
    query_text_hash: str = ""
    clue_ids: list[str] = field(default_factory=list)
    query_quality: float = 0.0
    n_query_candidates: int = 0
    n_query_candidates_dropped: int = 0
    query_candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def contaminated_results(self) -> int:
        return sum(1 for o in self.observations
                   if getattr(o, "benchmark_contaminated", False))

    def public_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "tool": self.tool,
            "query_arm": self.query_arm,
            "query": self.query,
            "query_text_hash": self.query_text_hash,
            "clue_ids": list(self.clue_ids),
            "query_quality": self.query_quality,
            "n_query_candidates": self.n_query_candidates,
            "n_query_candidates_dropped": self.n_query_candidates_dropped,
            "query_candidates": self.query_candidates,
            "cost": self.cost,
            "latency": self.latency,
            "stop_arm": self.stop_arm,
            "supported": self.supported,
            "failed": self.failed,
            "error_type": self.error_type,
            "status_code": self.status_code,
            "contaminated_results": self.contaminated_results,
            "observations": [o.to_public_dict() for o in self.observations],
        }


@dataclass
class AttemptTrace:
    attempt_id: str
    item_id: str
    signature: QuerySignature
    routing_plan: dict[str, Any]
    calls: list[CallRecord]
    candidate: CandidateAnswer
    final_answer: Optional[str]
    vstate: VerificationState
    budget: int
    mode: str

    @property
    def tool_calls(self) -> int:
        return len(self.calls)

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def total_latency(self) -> float:
        return sum(c.latency for c in self.calls)

    def tools_used(self) -> list[str]:
        return [c.tool for c in self.calls]

    def query_arms_used(self) -> list[str]:
        return [c.query_arm for c in self.calls]

    def stop_arms(self) -> list[str]:
        return [c.stop_arm for c in self.calls]


@dataclass
class SearchLoopConfig:
    available_tools: list[str]
    budget: int
    explore: bool = False
    as_of: str = "2026-06-01"
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    enable_query_decomposition: bool = False


class SearchLoop:
    """Drive one question to a final answer under budget."""

    def __init__(
        self,
        router: Router,
        query_policy: QueryPolicy,
        stopping_policy: StoppingPolicy,
        *,
        answerer: Optional[DeterministicAnswerer] = None,
        invoker_factory=None,
    ) -> None:
        self.router = router
        self.query_policy = query_policy
        self.stopping_policy = stopping_policy
        self.answerer = answerer or DeterministicAnswerer()
        self._invoker_factory = invoker_factory

    def run(
        self,
        item: Item,
        signature: QuerySignature,
        memory: PolicyMemory,
        invoker: ToolInvoker,
        providers: dict[str, SearchProvider],
        config: SearchLoopConfig,
        *,
        attempt_id: str,
        recorder: Optional[LoopRecorder] = None,
    ) -> AttemptTrace:
        from regimes_probe.tools.metadata import first_hop_tools, followup_tools

        rec = recorder or LoopRecorder()
        tool_costs = {n: getattr(p, "cost_per_call", Decimal("0")) for n, p in providers.items()}
        # First-hop arms are query tools (search family); follow-up tools
        # (page_fetch/scrape) operate on a URL from evidence and must NOT be
        # routed as a first-hop arm. The router only ranks first-hop tools.
        first_hop = first_hop_tools(config.available_tools) or list(config.available_tools)
        followup = [t for t in followup_tools(config.available_tools) if t in providers]
        routing_plan = self.router.route(
            signature, memory, first_hop,
            budget=config.budget, tool_costs=tool_costs, explore=config.explore,
            salt=f"{attempt_id}:route",
        )
        rec.on_routing_plan(routing_plan.to_dict())
        tool_seq = routing_plan.sequence or list(first_hop)
        # The follow-up tool used by the fetch_page mechanism (prefer page_fetch).
        fetch_tool = ("page_fetch" if "page_fetch" in followup
                      else (followup[0] if followup else None))
        fetch_available = fetch_tool is not None
        known_domain = item.meta.get("known_domain")
        freshness_sensitive = bool(signature.features.get("freshness_sensitive"))

        calls: list[CallRecord] = []
        observations: list[EvidenceObservation] = []
        candidate = CandidateAnswer(None, 0, 0.0, [])
        vstate = VerificationState()
        pending_fetch_url: Optional[str] = None
        step = 0

        while len(calls) < config.budget:
            query_text_hash, clue_ids = "", []
            q_quality, n_cand, n_dropped, q_cands = 0.0, 0, 0, []
            if pending_fetch_url is not None and fetch_available:
                tool = fetch_tool
                query = pending_fetch_url
                query_arm = "fetch"
                opts: dict[str, Any] = {}
                pending_fetch_url = None
            else:
                tool = tool_seq[step] if step < len(tool_seq) else tool_seq[-1]
                qplan = self.query_policy.formulate(
                    signature,
                    bandit=memory.bandits["query"],
                    neighbor=memory.estimate("query", signature.embedding),
                    explore=config.explore,
                    known_domain=known_domain,
                    salt=f"{attempt_id}:q{step}",
                    decompose=config.enable_query_decomposition,
                    tool=tool,
                )
                query, query_arm, opts = qplan.query, qplan.arm, qplan.opts
                query_text_hash, clue_ids = qplan.query_text_hash, qplan.clue_ids
                ex = qplan.explanation
                if ex.get("decompose") is True:
                    q_quality = ex.get("selected_query_quality", 0.0)
                    q_cands = ex.get("candidate_queries", [])
                    n_cand = len(q_cands)
                    n_dropped = ex.get("dropped_count", 0)
                rec.on_query_plan(step, qplan.to_dict())

            response = invoker.call(tool, query, limit=5, **opts)
            ci = len(calls)
            obs = [
                score_observation(
                    r, item, call_index=ci, tool=tool, query_arm=query_arm,
                    as_of=config.as_of,
                )
                for r in response.results
            ]
            call_failed = bool(response.failed)
            if call_failed:
                # provider/API error -> recorded failed observation, agent continues.
                obs.append(EvidenceObservation.failure(
                    call_index=ci, tool=tool, query_arm=query_arm, response=response))
            observations.extend(obs)
            supported = any(o.supports for o in obs)
            rec.on_evidence(step, obs)

            candidate = self.answerer.answer(observations, item=item)
            rec.on_candidate(step, candidate)
            vstate = verify(
                candidate.to_dict() if candidate.answer else None,
                [o.to_verify_dict() for o in observations],
                freshness_sensitive=freshness_sensitive,
                config=config.verification,
            )
            rec.on_verification(step, vstate)

            decision = self.stopping_policy.decide(
                signature.cluster_key,
                vstate,
                calls_used=ci + 1,
                budget=config.budget,
                fetch_available=fetch_available and any(o.fetchable for o in observations),
                bandit=memory.bandits["stop"],
                neighbor=memory.estimate("stop", signature.embedding),
                explore=config.explore,
                salt=f"{attempt_id}:stop{step}",
            )
            rec.on_stop(step, decision.to_dict())

            calls.append(
                CallRecord(
                    call_index=ci,
                    tool=tool,
                    query_arm=query_arm,
                    query=query,
                    cost=float(response.cost),
                    latency=float(response.latency_s),
                    observations=obs,
                    stop_arm=decision.arm,
                    supported=supported,
                    failed=call_failed,
                    error_type=(response.error_meta or {}).get("error_type") if call_failed else None,
                    status_code=(response.error_meta or {}).get("status_code") if call_failed else None,
                    query_text_hash=query_text_hash,
                    clue_ids=clue_ids,
                    query_quality=q_quality,
                    n_query_candidates=n_cand,
                    n_query_candidates_dropped=n_dropped,
                    query_candidates=q_cands,
                )
            )

            if decision.stop:
                break
            if decision.arm == "fetch_page":
                fetch_targets = [o for o in observations if o.fetchable]
                if fetch_targets:
                    pending_fetch_url = fetch_targets[0].url
            step += 1

        # Recompute once after the loop so a zero-budget (closed-book) attempt,
        # which never entered the loop body, still produces a candidate.
        candidate = self.answerer.answer(observations, item=item)
        final_answer = candidate.answer
        return AttemptTrace(
            attempt_id=attempt_id,
            item_id=item.id,
            signature=signature,
            routing_plan=routing_plan.to_dict(),
            calls=calls,
            candidate=candidate,
            final_answer=final_answer,
            vstate=vstate,
            budget=config.budget,
            mode="explore" if config.explore else "exploit",
        )
