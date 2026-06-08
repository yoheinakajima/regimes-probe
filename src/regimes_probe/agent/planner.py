"""The agent facade: signature → plans → search loop.

``EpistemicAgent`` wires the policy seams together and exposes a single
``attempt`` call. It owns no answer knowledge; it consults policy memory and the
contextual bandits to plan, then drives :class:`SearchLoop`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.agent.answerer import ClosedBookAnswerer
from regimes_probe.agent.search_loop import (
    AttemptTrace,
    DirectInvoker,
    SearchLoop,
    SearchLoopConfig,
    ToolInvoker,
)
from regimes_probe.datasets.base import Item
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.query_policy import QueryPolicy
from regimes_probe.policy.router import Router, RouterConfig
from regimes_probe.policy.signatures import QuerySignature, SignatureExtractor
from regimes_probe.policy.stopping_policy import StopConfig, StoppingPolicy
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.tools.base import SearchProvider


@dataclass
class AgentConfig:
    available_tools: list[str]
    query_mode: str = "learned"          # fixed | llm | learned
    stop_mode: str = "learned"           # always_full | first_candidate | learned
    as_of: str = "2026-06-01"
    router: RouterConfig = field(default_factory=RouterConfig)
    stop: StopConfig = field(default_factory=StopConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)


class EpistemicAgent:
    def __init__(self, config: AgentConfig, *, extractor: Optional[SignatureExtractor] = None,
                 answerer=None) -> None:
        self.config = config
        self.extractor = extractor or SignatureExtractor()
        self.router = Router(config.router)
        self.query_policy = QueryPolicy(config.query_mode)
        self.stopping_policy = StoppingPolicy(config.stop_mode, config.stop)
        self.answerer = answerer
        self.loop = SearchLoop(self.router, self.query_policy, self.stopping_policy,
                               answerer=answerer)

    def signature(self, item: Item) -> QuerySignature:
        return self.extractor.compute(item.question)

    def attempt(
        self,
        item: Item,
        memory: PolicyMemory,
        providers: dict[str, SearchProvider],
        *,
        budget: int,
        explore: bool,
        attempt_id: str,
        invoker: Optional[ToolInvoker] = None,
        signature: Optional[QuerySignature] = None,
        recorder=None,
    ) -> AttemptTrace:
        sig = signature or self.signature(item)
        invoker = invoker or DirectInvoker(providers)
        loop_cfg = SearchLoopConfig(
            available_tools=self.config.available_tools,
            budget=budget,
            explore=explore,
            as_of=self.config.as_of,
            verification=self.config.verification,
        )
        return self.loop.run(
            item, sig, memory, invoker, providers, loop_cfg,
            attempt_id=attempt_id, recorder=recorder,
        )


def build_closed_book_agent(config: AgentConfig, knowledge: dict[str, str]) -> "EpistemicAgent":
    """An agent that answers from intrinsic knowledge only (run with budget 0)."""
    return EpistemicAgent(config, answerer=ClosedBookAnswerer(knowledge))
