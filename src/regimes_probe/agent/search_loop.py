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
        provider = self.providers[tool]
        return provider.search(query, limit=limit, **opts)


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

    def public_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "tool": self.tool,
            "query_arm": self.query_arm,
            "query": self.query,
            "cost": self.cost,
            "latency": self.latency,
            "stop_arm": self.stop_arm,
            "supported": self.supported,
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
        rec = recorder or LoopRecorder()
        tool_costs = {n: getattr(p, "cost_per_call", Decimal("0")) for n, p in providers.items()}
        routing_plan = self.router.route(
            signature, memory, config.available_tools,
            budget=config.budget, tool_costs=tool_costs, explore=config.explore,
            salt=f"{attempt_id}:route",
        )
        rec.on_routing_plan(routing_plan.to_dict())
        tool_seq = routing_plan.sequence or list(config.available_tools)
        fetch_available = "page_fetch" in providers
        known_domain = item.meta.get("known_domain")
        freshness_sensitive = bool(signature.features.get("freshness_sensitive"))

        calls: list[CallRecord] = []
        observations: list[EvidenceObservation] = []
        candidate = CandidateAnswer(None, 0, 0.0, [])
        vstate = VerificationState()
        pending_fetch_url: Optional[str] = None
        step = 0

        while len(calls) < config.budget:
            if pending_fetch_url is not None and fetch_available:
                tool = "page_fetch"
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
                )
                query, query_arm, opts = qplan.query, qplan.arm, qplan.opts
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
            observations.extend(obs)
            supported = any(o.supports for o in obs)
            rec.on_evidence(step, obs)

            candidate = self.answerer.answer(observations)
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
                )
            )

            if decision.stop:
                break
            if decision.arm == "fetch_page":
                fetch_targets = [o for o in observations if o.fetchable]
                if fetch_targets:
                    pending_fetch_url = fetch_targets[0].url
            step += 1

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
