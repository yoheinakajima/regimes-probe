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


def _dedupe_keep(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for s in seq:
        k = s.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def _best_read_target(observations, scraped_urls, no_progress_domains):
    """Pick the best unread, non-contaminated, non-failed evidence page to read."""
    from urllib.parse import urlparse
    from regimes_probe.agent.reading_policy import normalize_url, _is_social
    best, best_key = None, (-1.0, 0)
    for o in observations:
        url = getattr(o, "url", "") or ""
        if (getattr(o, "failed", False) or not url
                or getattr(o, "benchmark_contaminated", False)):
            continue
        host = (urlparse(url).hostname or "").lower()
        if not host or _is_social(host) or host in no_progress_domains:
            continue
        if normalize_url(url) in scraped_urls:
            continue
        key = (float(getattr(o, "source_authority", 0.0)), 1 if o.supports else 0)
        if key > best_key:
            best, best_key = o, key
    return best


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
    # query decomposition debug (candidate queries before selection)
    query_quality: float = 0.0
    n_query_candidates: int = 0
    n_query_candidates_dropped: int = 0
    query_candidates: list[dict[str, Any]] = field(default_factory=list)
    # iterative clue resolution (staged search)
    stage: int = 1
    parent_query_id: Optional[int] = None
    candidate_entities: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate: Optional[str] = None
    selection_reason: Optional[str] = None
    evidence_improved: bool = False
    # candidate-hypothesis policy
    target_roles: list[str] = field(default_factory=list)
    selected_role: Optional[str] = None
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    sticky_penalty: float = 0.0
    no_progress: bool = False
    # Level 3 evidence reading (page_fetch vs firecrawl_scrape)
    scrape: dict[str, Any] = field(default_factory=dict)
    # Level 4 task-frame action + evidence record
    task_action: dict[str, Any] = field(default_factory=dict)
    evidence_record: dict[str, Any] = field(default_factory=dict)
    # Level 5 frontier controller: the frontier_action this tool call executed
    frontier_action_id: Optional[str] = None
    # Level 5c: the LLM frontier proposal this tool call was translated from (if any)
    llm_frontier_proposal_id: Optional[str] = None

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
            "query_quality": round(float(self.query_quality), 3),
            "n_query_candidates": self.n_query_candidates,
            "n_query_candidates_dropped": self.n_query_candidates_dropped,
            "query_candidates": self.query_candidates,
            "stage": self.stage,
            "parent_query_id": self.parent_query_id,
            "candidate_entities": self.candidate_entities,
            "selected_candidate": self.selected_candidate,
            "selection_reason": self.selection_reason,
            "evidence_improved": self.evidence_improved,
            "target_roles": list(self.target_roles),
            "selected_role": self.selected_role,
            "rejected_candidates": self.rejected_candidates,
            "sticky_penalty": round(float(self.sticky_penalty), 3),
            "no_progress": self.no_progress,
            "scrape": self.scrape,
            "task_action": self.task_action,
            "evidence_record": self.evidence_record,
            "frontier_action_id": self.frontier_action_id,
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


def _with_read_accounting(cf: dict[str, Any], read_acct: dict[str, int]) -> dict[str, Any]:
    """Attach Level-5f read-path accounting to the candidate-frontier trace (so metrics can
    aggregate it). A no-op when the slate layer was skipped."""
    if isinstance(cf, dict) and cf and not cf.get("skipped"):
        cf["read_accounting"] = dict(read_acct)
    return cf


def _frontier_trace(frontier, mode: str, skip_reason: str, *, uses_layer: bool,
                    had_frame: bool, controller: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Candidate-frontier trace payload: the slate debug, or a bounded skip record."""
    if frontier is not None:
        out = frontier.to_debug()
        out["controller"] = controller or {}
        return out
    if uses_layer and mode in ("direct_answer_possible", "simple_lookup"):
        return {"skipped": True, "skipped_candidate_slate_reason": f"epistemic_mode={mode}"}
    if had_frame:
        return {"skipped": True,
                "skipped_candidate_slate_reason": skip_reason or "no_frontier"}
    return {}


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
    task_frame: dict[str, Any] = field(default_factory=dict)
    hypothesis_summary: dict[str, Any] = field(default_factory=dict)
    frame_coverage: dict[str, Any] = field(default_factory=dict)
    task_frame_parse: dict[str, Any] = field(default_factory=dict)
    epistemic_mode: dict[str, Any] = field(default_factory=dict)
    candidate_frontier: dict[str, Any] = field(default_factory=dict)
    llm_frontier: dict[str, Any] = field(default_factory=dict)

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
    enable_iterative_clue_resolution: bool = False
    enable_task_frame: bool = False
    enable_llm_task_frame_parser: bool = False
    auto_epistemic_mode: bool = False
    force_task_frame: bool = False
    disable_direct_answer: bool = False
    enable_frontier_controller: bool = False
    enable_llm_frontier_repair: bool = False
    enable_llm_frontier_planner: bool = False
    enable_llm_evidence_interpreter: bool = False
    enable_llm_evidence_judge: bool = False
    scrape_fallback_to_page_fetch: bool = True
    allow_social_scrape: bool = False


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
        task_frame_parser=None,
        llm_frontier=None,
        evidence_interpreter=None,
        evidence_judge=None,
    ) -> None:
        self.router = router
        self.query_policy = query_policy
        self.stopping_policy = stopping_policy
        self.answerer = answerer or DeterministicAnswerer()
        self._invoker_factory = invoker_factory
        self.task_frame_parser = task_frame_parser
        self.llm_frontier = llm_frontier
        #: optional shared (cache, model_fn) for the LLM evidence-interpreter hook; a fresh
        #: per-attempt EvidenceInterpreter is built from it so stats stay per-attempt.
        self.evidence_interpreter = evidence_interpreter
        #: optional shared (cache, model_fn) for the LLM evidence JUDGE (Level 5f).
        self.evidence_judge = evidence_judge

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
        # Level 3 reading tools (URL-only follow-up): page_fetch (cheap) and
        # firecrawl_scrape (richer/paid). Neither is a first-hop arm.
        page_fetch_available = "page_fetch" in providers
        scrape_available = "firecrawl_scrape" in providers
        reading_tools = [t for t in ("page_fetch", "firecrawl_scrape") if t in providers]
        fallback_tool = ("page_fetch" if page_fetch_available
                         else (followup[0] if followup else None))
        fetch_available = bool(reading_tools) or fallback_tool is not None
        # cross-provider corroboration: host -> set of first-hop tools that returned it
        domain_providers: dict[str, set] = {}
        scraped_urls: set[str] = set()
        no_progress_domains: set[str] = set()
        pending_read = None          # an EvidenceObservation to read next
        force_page_fetch = False     # set after a firecrawl_scrape failure (fallback)
        known_domain = item.meta.get("known_domain")
        freshness_sensitive = bool(signature.features.get("freshness_sensitive"))

        # Epistemic escalation controller: decide how much machinery this question
        # warrants. When --auto-epistemic-mode is OFF the explicit flags win (and the
        # decision is recorded for audit only); when ON, the decision sets the
        # effective task-frame / iterative / decomposition flags for THIS item.
        from regimes_probe.agent.epistemic_mode import (
            decide_epistemic_mode, explicit_mode_decision)
        if config.auto_epistemic_mode:
            epistemic_decision = decide_epistemic_mode(
                item.question, budget=config.budget,
                force_task_frame=config.force_task_frame,
                disable_direct_answer=config.disable_direct_answer)
            epistemic_decision.applied = True
            eff_task_frame = epistemic_decision.use_task_frame or config.force_task_frame
            eff_iterative = epistemic_decision.use_iterative or config.enable_iterative_clue_resolution
            eff_decompose = (epistemic_decision.use_decomposition
                             or config.enable_query_decomposition)
        else:
            eff_task_frame = config.enable_task_frame or config.force_task_frame
            eff_iterative = config.enable_iterative_clue_resolution
            eff_decompose = config.enable_query_decomposition
            epistemic_decision = explicit_mode_decision(
                task_frame=eff_task_frame, iterative=eff_iterative, decomposition=eff_decompose)

        # Iterative clue resolution (staged search): clue spans + answer-shape to
        # chase intermediate entities found in earlier results.
        iterative = eff_iterative
        clue_spans: list[str] = []
        answer_shape: list[str] = []
        clue_terms: list[str] = []
        target_roles: list[str] = []
        intermediate_roles: list[str] = []
        beam = None
        if iterative:
            from regimes_probe.policy.query_decomposition import (
                answer_shape_terms, extract_clues)
            from regimes_probe.agent.clue_resolution import (
                HypothesisBeam, infer_target_roles)
            _cl = extract_clues(item.question)
            clue_spans = _dedupe_keep(_cl.phrase_spans + _cl.entities)
            answer_shape = answer_shape_terms(item.question)
            clue_terms = [t for s in clue_spans for t in s.split()]
            target_roles, intermediate_roles = infer_target_roles(item.question)
            beam = HypothesisBeam(beam_size=3)
        used_clue_spans: set[str] = set()
        stage = 1
        prev_vscore = 0.0
        current_selected_norm: Optional[str] = None

        # Level 4 task frame: constraint-satisfaction state over latent slots. When
        # enabled it drives query/read/stop via an action planner (supersedes the
        # iterative beam). Clue terms feed the answer/read gating.
        task_frame = eff_task_frame
        frame = htable = planner = None
        frame_parse_meta: dict[str, Any] = {}
        if task_frame:
            from regimes_probe.agent.hypothesis_table import HypothesisTable
            from regimes_probe.agent.action_planner import ActionPlanner, frame_read_value
            from regimes_probe.agent.llm_task_frame import build_task_frame
            frame, _pm = build_task_frame(
                item.id, item.question,
                use_llm=config.enable_llm_task_frame_parser,
                parser=self.task_frame_parser)
            frame_parse_meta = _pm.to_dict()
            htable = HypothesisTable(frame)
            planner = ActionPlanner(frame, htable)
            if not clue_terms:
                clue_terms = [t for c in frame.constraints for t in c.normalized_terms]
            if not answer_shape:
                answer_shape = list(frame.answer_shape_hints)

        # Candidate-slate / frontier layer (multi-hop): maintain per-slot candidate
        # pools + frontier scheduling. Skipped for easy/direct/simple epistemic modes.
        frontier = None
        slate_skipped_reason = ""
        # Frontier controller (optional): when enabled, the frontier scheduler DRIVES
        # tool/action selection; otherwise it stays shadow (recommendation only).
        frontier_controller = config.enable_frontier_controller and task_frame
        ctrl = {"frontier_controller_used": False, "old_planner_fallback_count": 0,
                "frontier_action_execution_success_count": 0,
                "frontier_action_execution_failure_count": 0,
                "tool_calls_from_frontier_actions": 0,
                "shadow_agreements": 0, "shadow_total": 0, "shadow_disagree_reasons": [],
                "first_action_discriminative": None, "first_query_generic": None}
        # LLM frontier proposer (Level 5c): optional, repairs a generic deterministic
        # query OR proposes the next action from a bounded ActiveGraph state card.
        lf_mode = ("planner" if config.enable_llm_frontier_planner
                   else ("repair" if config.enable_llm_frontier_repair else ""))
        llm_frontier = (self.llm_frontier if (task_frame and frame is not None and lf_mode
                                              and self.llm_frontier is not None) else None)
        lf_steps: list[dict[str, Any]] = []
        lf_no_progress_queries: list[str] = []
        # Level 5f-A: read-path accounting (URL-backed reads + page_fetch->scrape fallback).
        read_acct: dict[str, int] = {}
        read_fallback_pending = False
        _MIN_READ_CHARS = 1
        if task_frame and frame is not None:
            _mode = epistemic_decision.selected_epistemic_mode
            if _mode in ("direct_answer_possible", "simple_lookup"):
                slate_skipped_reason = f"epistemic_mode={_mode}"
            else:
                from regimes_probe.agent.candidate_frontier import CandidateFrontier
                # Level 5d: a per-attempt evidence interpreter (deterministic by default;
                # LLM source-role re-classification only when enabled + a shared cache/model
                # is injected, so replay/dry-run make no model call).
                interp = None
                judge = None
                # Level 5f: a per-attempt evidence JUDGE (decides candidate/slot/constraint
                # support fit) built from the shared cache/model so stats stay per-attempt and
                # replay/dry-run make no model call.
                if config.enable_llm_evidence_judge and self.evidence_judge is not None:
                    from regimes_probe.agent.evidence_judge import EvidenceJudge
                    ej = self.evidence_judge
                    judge = EvidenceJudge(
                        model_fn=getattr(ej, "model_fn", None), cache=getattr(ej, "cache", None),
                        model=getattr(ej, "model", "deterministic"),
                        replay_only=getattr(ej, "replay_only", False), enabled=True)
                if (config.enable_llm_evidence_interpreter or judge is not None):
                    from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
                    ei = self.evidence_interpreter
                    interp = EvidenceInterpreter(
                        model_fn=getattr(ei, "model_fn", None), cache=getattr(ei, "cache", None),
                        model=getattr(ei, "model", "deterministic"),
                        replay_only=getattr(ei, "replay_only", False),
                        enabled_llm=bool(config.enable_llm_evidence_interpreter
                                         and self.evidence_interpreter is not None),
                        judge=judge)
                frontier = CandidateFrontier(frame, attempt_id=attempt_id, item_id=item.id,
                                             interpreter=interp)
                ctrl["frontier_controller_used"] = bool(config.enable_frontier_controller)

        calls: list[CallRecord] = []
        observations: list[EvidenceObservation] = []
        candidate = CandidateAnswer(None, 0, 0.0, [])
        vstate = VerificationState()
        task_actions: list[dict[str, Any]] = []
        terminal_action: dict[str, Any] = {}
        step = 0

        while len(calls) < config.budget:
            query_text_hash, clue_ids = "", []
            stage_info: dict[str, Any] = {}
            scrape_info: dict[str, Any] = {}
            task_action_info: dict[str, Any] = {}
            evidence_record_info: dict[str, Any] = {}
            current_selected_norm = None
            cur_lf_meta: dict[str, Any] | None = None    # LLM-frontier meta for this step
            # Decide whether/how to READ a pending URL (Level 3) before anything else.
            # The reading policy (tool choice + gating/dedup) engages only when a
            # scrape tool is available or in iterative mode; otherwise the legacy
            # page_fetch-only follow-up is preserved unchanged.
            read_dec = None
            legacy_read = False
            if pending_read is not None and reading_tools and not (scrape_available or iterative):
                legacy_read = page_fetch_available or bool(reading_tools)
            elif pending_read is not None and reading_tools:
                from regimes_probe.agent.reading_policy import select_reading_tool
                read_dec = select_reading_tool(
                    url=pending_read.url, title=getattr(pending_read, "title", ""),
                    snippet=pending_read.snippet,
                    source_authority=float(getattr(pending_read, "source_authority", 0.0)),
                    contaminated=bool(getattr(pending_read, "benchmark_contaminated", False)),
                    unresolved_clue_terms=clue_terms,
                    answer_shape=answer_shape if iterative else [],
                    cross_provider_domains={d for d, ts in domain_providers.items()
                                            if len(ts) >= 2},
                    page_fetch_available=page_fetch_available,
                    scrape_available=scrape_available, scraped_urls=scraped_urls,
                    no_progress_domains=no_progress_domains,
                    allow_social=config.allow_social_scrape,
                    prefer_page_fetch=force_page_fetch,
                    force_read=bool(getattr(pending_read, "fetchable", False)))
                if read_dec.tool is None:                    # gated out -> do not read
                    pending_read = None
                    read_dec = None
            beam_sel = beam.select() if (iterative and beam is not None and calls) else None
            read_target_obs = None

            # Frontier CONTROLLER step: when enabled, the frontier scheduler drives this
            # step's tool/action (every executed call links to a frontier_action id).
            controller_handled = False
            step_plan = None
            if frontier is not None:
                step_plan = frontier.propose_step_action(
                    observations=observations, budget_remaining=config.budget - len(calls),
                    reading_tools=bool(reading_tools), scraped_urls=scraped_urls,
                    no_progress_domains=no_progress_domains,
                    page_fetch_available=page_fetch_available, scrape_available=scrape_available,
                    allow_social=config.allow_social_scrape, force_page_fetch=force_page_fetch)
                # LLM frontier proposer: validate/score/select a constraint-grounded
                # action that replaces (planner) or repairs (repair) the generic one.
                if llm_frontier is not None:
                    from regimes_probe.agent.llm_frontier import build_research_state_card
                    card = build_research_state_card(
                        frame, frontier, question=item.question,
                        epistemic_mode=epistemic_decision.selected_epistemic_mode,
                        available_tools=config.available_tools,
                        remaining_budget=config.budget - len(calls),
                        memory_access_mode=("policy_memory" if memory.has_priors() else "no_memory")
                        if hasattr(memory, "has_priors") else "unknown",
                        failed_queries=[c.query for c in calls if c.failed],
                        no_progress_queries=lf_no_progress_queries, det_plan=step_plan)
                    lf_plan, lf_meta = llm_frontier.plan_step(
                        card, step_plan, frame=frame, frontier=frontier, mode=lf_mode,
                        observations=observations, scraped_urls=scraped_urls,
                        no_progress_domains=no_progress_domains,
                        page_fetch_available=page_fetch_available, scrape_available=scrape_available,
                        allow_social=config.allow_social_scrape, force_page_fetch=force_page_fetch,
                        failed_queries=lf_no_progress_queries + [c.query for c in calls if c.failed])
                    lf_meta["step"] = step
                    lf_steps.append(lf_meta)
                    if lf_plan is not None:
                        step_plan = lf_plan
                        cur_lf_meta = lf_meta          # for post-execution evidence integrity
            if frontier_controller and step_plan is not None and step_plan.executable:
                if ctrl["first_action_discriminative"] is None and step_plan.kind == "search":
                    ctrl["first_action_discriminative"] = step_plan.is_discriminative_constraint
                    ctrl["first_query_generic"] = step_plan.is_generic_query
                if step_plan.kind in ("answer", "abstain"):
                    task_action_info = step_plan.action_info()
                    task_actions.append(task_action_info)
                    terminal_action = task_action_info
                    frontier.record_execution(step_plan.frontier_action_id, success=True,
                                              kind=step_plan.kind)
                    break
                if step_plan.kind == "search":
                    tool = step_plan.tool or (tool_seq[step] if step < len(tool_seq) else tool_seq[-1])
                    query, query_arm, opts = step_plan.query, step_plan.query_arm, {}
                    task_action_info = step_plan.action_info()
                    task_actions.append(task_action_info)
                    controller_handled = True
                elif step_plan.kind == "read":
                    o, rd = step_plan.read_obs, step_plan.read_decision
                    tool, query, query_arm, opts = rd.tool, o.url, rd.query_arm, {}
                    read_target_obs = o
                    force_page_fetch = False
                    scrape_info = {"read_tool": tool, "scrape_url": query, "scrape_provider": tool,
                                   "scrape_selected_reason": rd.reason, "is_scrape": rd.is_scrape}
                    task_action_info = step_plan.action_info()
                    task_actions.append(task_action_info)
                    controller_handled = True
            elif frontier_controller and step_plan is not None and not step_plan.executable:
                frontier.record_unexecutable(step_plan.frontier_action_id or "n/a",
                                             reason=step_plan.reason)
                ctrl["old_planner_fallback_count"] += 1

            if controller_handled:
                pass                                    # tool/query set by the controller
            elif task_frame:
                from urllib.parse import urlparse as _up
                from regimes_probe.agent.reading_policy import (
                    normalize_url as _nurl, select_reading_tool as _srt)
                from regimes_probe.agent.action_planner import frame_read_value
                cross = {d for d, ts in domain_providers.items() if len(ts) >= 2}
                chosen_read = None
                if reading_tools:
                    for o in observations:
                        if getattr(o, "failed", False) or not getattr(o, "url", ""):
                            continue
                        host = (_up(o.url).hostname or "").lower()
                        if _nurl(o.url) in scraped_urls or host in no_progress_domains:
                            continue
                        rv = frame_read_value(o, frame, htable, cross_provider=(host in cross),
                                              allow_social=config.allow_social_scrape)
                        if not rv.should_read:
                            continue
                        rd = _srt(url=o.url, title=getattr(o, "title", ""), snippet=o.snippet,
                                  source_authority=float(getattr(o, "source_authority", 0.0)),
                                  contaminated=False, unresolved_clue_terms=clue_terms,
                                  answer_shape=answer_shape, cross_provider_domains=cross,
                                  page_fetch_available=page_fetch_available,
                                  scrape_available=scrape_available, scraped_urls=scraped_urls,
                                  no_progress_domains=no_progress_domains,
                                  allow_social=config.allow_social_scrape,
                                  prefer_page_fetch=force_page_fetch, force_read=True)
                        if rd.tool:
                            chosen_read = (o, rv, rd)
                            break
                if chosen_read is not None:
                    o, rv, rd = chosen_read
                    tool, query, query_arm, opts = rd.tool, o.url, rd.query_arm, {}
                    read_target_obs = o
                    force_page_fetch = False
                    scrape_info = {"read_tool": tool, "scrape_url": query, "scrape_provider": tool,
                                   "scrape_selected_reason": rd.reason, "is_scrape": rd.is_scrape}
                    task_action_info = {"action_id": planner._next_id(),
                                        "kind": "read_url_for_constraint",
                                        "target_slot_id": rv.target_slot_id,
                                        "tested_constraint_ids": rv.tested_constraint_ids,
                                        "hypothesis_id": rv.hypothesis_id,
                                        "query_text_preview": query[:160], "query_arm": query_arm,
                                        "read_value": rv.to_dict()}
                    task_actions.append(task_action_info)
                else:
                    action = planner.plan(budget_remaining=config.budget - len(calls),
                                          reading_available=bool(reading_tools))
                    task_action_info = action.to_dict()
                    task_actions.append(task_action_info)
                    if action.kind in ("answer_if_supported", "abstain_if_no_path",
                                       "abstain_if_blocked"):
                        terminal_action = task_action_info
                        break
                    tool = tool_seq[step] if step < len(tool_seq) else tool_seq[-1]
                    query = action.query or signature.question
                    query_arm, opts = action.query_arm, {}
            elif read_dec is not None:
                tool = read_dec.tool
                query = pending_read.url
                query_arm = read_dec.query_arm
                opts = {}
                scrape_info = {"read_tool": tool, "scrape_url": query,
                               "scrape_provider": tool,
                               "scrape_selected_reason": read_dec.reason,
                               "is_scrape": read_dec.is_scrape}
                read_target_obs = pending_read
                pending_read = None
                force_page_fetch = False
            elif legacy_read:
                # Legacy page_fetch-only follow-up (no gating/dedup/scrape stats).
                tool = "page_fetch" if page_fetch_available else followup[0]
                query = pending_read.url
                query_arm = "fetch"
                opts = {}
                pending_read = None
            elif beam_sel is not None:
                # Stage >= 2: carry the best ROLE-COMPATIBLE candidate hypothesis
                # forward (beam handles anti-sticky / forced exploration).
                from regimes_probe.agent.clue_resolution import compose_followup_from_hypothesis
                fq = compose_followup_from_hypothesis(
                    beam_sel.hypothesis, clue_spans=clue_spans,
                    used_clue_spans=used_clue_spans, answer_shape=answer_shape, beam=beam)
                tool = tool_seq[step] if step < len(tool_seq) else tool_seq[-1]
                query, query_arm, opts = fq.query, fq.query_arm, {}
                if fq.clue_used:
                    used_clue_spans.add(fq.clue_used.lower())
                stage += 1
                current_selected_norm = beam_sel.hypothesis.normalized_text
                stage_info = {
                    "stage": stage, "parent_query_id": calls[-1].call_index,
                    "candidate_entities": [h.to_dict() for h in
                                           list(beam.pool.values())[:6]],
                    "selected_candidate": fq.selected_candidate,
                    "selected_role": beam_sel.hypothesis.role,
                    "selection_reason": f"{fq.reason} | {beam_sel.reason}",
                    "rejected_candidates": beam_sel.rejected,
                    "sticky_penalty": beam_sel.hypothesis.sticky_penalty,
                    "target_roles": list(target_roles)}
            else:
                tool = tool_seq[step] if step < len(tool_seq) else tool_seq[-1]
                qplan = self.query_policy.formulate(
                    signature,
                    bandit=memory.bandits["query"],
                    neighbor=memory.estimate("query", signature.embedding),
                    explore=config.explore,
                    known_domain=known_domain,
                    salt=f"{attempt_id}:q{step}",
                    decompose=eff_decompose,
                    tool=tool,
                )
                query, query_arm, opts = qplan.query, qplan.arm, qplan.opts
                query_text_hash, clue_ids = qplan.query_text_hash, qplan.clue_ids
                ex = qplan.explanation
                if ex.get("decompose") is True:
                    stage_info = {
                        "query_quality": ex.get("selected_query_quality", 0.0),
                        "n_query_candidates": len(ex.get("candidate_queries", [])),
                        "n_query_candidates_dropped": ex.get("dropped_count", 0),
                        "query_candidates": ex.get("candidate_queries", [])}
                rec.on_query_plan(step, qplan.to_dict())

            # SHADOW comparison (controller OFF): record the frontier's recommendation
            # vs. the planner's actual action for this task-frame step.
            if (task_frame and frontier is not None and not frontier_controller
                    and step_plan is not None and not controller_handled):
                actual_kind = task_action_info.get("kind", "")
                rec_kind = step_plan.planner_kind
                agree = (actual_kind == rec_kind
                         or (actual_kind.startswith("search") and rec_kind.startswith("search")))
                ctrl["shadow_total"] += 1
                if agree:
                    ctrl["shadow_agreements"] += 1
                else:
                    ctrl["shadow_disagree_reasons"].append(f"{rec_kind}!={actual_kind or 'none'}")
                if task_action_info:
                    task_action_info["frontier_recommended_action"] = step_plan.action_type
                    task_action_info["planner_actual_action"] = actual_kind
                    task_action_info["action_agreement"] = agree

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

            # Cross-provider corroboration: which first-hop tools surfaced each host.
            if read_dec is None and not call_failed:
                from urllib.parse import urlparse as _urlparse
                for o in obs:
                    h = (_urlparse(o.url or "").hostname or "").lower()
                    if h:
                        domain_providers.setdefault(h, set()).add(tool)

            candidate = self.answerer.answer(observations, item=item)
            rec.on_candidate(step, candidate)
            vstate = verify(
                candidate.to_dict() if candidate.answer else None,
                [o.to_verify_dict() for o in observations],
                freshness_sensitive=freshness_sensitive,
                config=config.verification,
            )
            rec.on_verification(step, vstate)

            # A read is possible if there is a known multihop target (fetchable) or
            # — in iterative mode — any readable URL from evidence so far.
            readable = any(getattr(o, "fetchable", False) for o in observations) or (
                iterative and any(getattr(o, "url", "") and not getattr(o, "failed", False)
                                  for o in observations))
            decision = self.stopping_policy.decide(
                signature.cluster_key,
                vstate,
                calls_used=ci + 1,
                budget=config.budget,
                fetch_available=fetch_available and readable,
                bandit=memory.bandits["stop"],
                neighbor=memory.estimate("stop", signature.embedding),
                explore=config.explore,
                salt=f"{attempt_id}:stop{step}",
            )
            rec.on_stop(step, decision.to_dict())

            # Did this call improve evidence vs before it? (Used by the iterative
            # metrics, the evidence-progress gate, and recorded per call.)
            evidence_improved = bool(vstate.score > prev_vscore + 1e-9
                                     or (supported and not any(c.supported for c in calls)))
            prev_vscore = max(prev_vscore, vstate.score)
            no_progress_flag = False

            # Level 3 read outcome + fail-closed fallback to page_fetch.
            if scrape_info:
                from urllib.parse import urlparse as _urlparse
                from regimes_probe.agent.reading_policy import normalize_url
                read_obs = next((o for o in obs if not getattr(o, "failed", False)), None)
                text_l = (read_obs.snippet.lower() if read_obs else "")
                host = (_urlparse(query or "").hostname or "").lower()
                scrape_info.update({
                    "scrape_success": (not call_failed),
                    "scrape_chars": len(read_obs.snippet) if read_obs else 0,
                    "evidence_added_by_scrape": bool(not call_failed and (evidence_improved or supported)),
                    "answer_shape_found_after_scrape": bool(
                        read_obs and answer_shape and any(s.lower() in text_l for s in answer_shape)),
                    "unresolved_clues_supported_after_scrape": (
                        sum(1 for s in clue_spans if s.lower() in text_l) if read_obs else 0),
                    "scrape_failure_type": ((response.error_meta or {}).get("error_type")
                                            if call_failed else None),
                    "scrape_cost_estimate": float(response.cost),
                })
                # how many chars did the read actually return? a zero-byte/empty fetch is
                # NOT a successful read (Level 5f-A).
                read_chars = sum(len((getattr(o, "snippet", "") or ""))
                                 + len((getattr(o, "title", "") or "")) for o in obs
                                 if not getattr(o, "failed", False))
                read_zero = (call_failed or read_chars < _MIN_READ_CHARS)
                scrape_info["read_chars"] = read_chars
                scrape_info["read_zero_chars"] = read_zero
                if read_zero:
                    read_acct["read_failed_zero_chars_count"] = \
                        read_acct.get("read_failed_zero_chars_count", 0) + 1
                # resolve the outcome of a previously-scheduled fallback read.
                if read_fallback_pending:
                    key = ("read_fallback_success_count" if not read_zero
                           else "read_failed_after_fallback_count")
                    read_acct[key] = read_acct.get(key, 0) + 1
                    read_fallback_pending = False
                if call_failed and scrape_info.get("is_scrape") and \
                        config.scrape_fallback_to_page_fetch and page_fetch_available:
                    # fail closed: retry the SAME url with the basic fetch next step.
                    pending_read = read_target_obs
                    force_page_fetch = True
                    scrape_info["fallback_to_page_fetch"] = True
                    read_fallback_pending = True
                    read_acct["read_fallback_attempted_count"] = \
                        read_acct.get("read_fallback_attempted_count", 0) + 1
                elif read_zero and not scrape_info.get("is_scrape") and scrape_available \
                        and read_target_obs is not None:
                    # page_fetch returned zero chars -> fall back ONCE to firecrawl_scrape on
                    # the SAME url (Level 5f-A read fallback policy).
                    pending_read = read_target_obs
                    force_page_fetch = False
                    scrape_info["fallback_to_firecrawl_scrape"] = True
                    read_fallback_pending = True
                    read_acct["read_fallback_attempted_count"] = \
                        read_acct.get("read_fallback_attempted_count", 0) + 1
                else:
                    scraped_urls.add(normalize_url(query))
                    if (call_failed or not evidence_improved) and host:
                        no_progress_domains.add(host)

            if iterative and beam is not None:
                from regimes_probe.agent.clue_resolution import build_hypotheses
                if current_selected_norm is not None:
                    beam.record_progress(current_selected_norm, evidence_improved)
                    no_progress_flag = not evidence_improved
                beam.observe(build_hypotheses(
                    observations, question=item.question, target_roles=target_roles,
                    intermediate_roles=intermediate_roles, clue_terms=clue_terms,
                    stage_found=stage))

            # Level 4: fold this call's evidence into the hypothesis table.
            if task_frame and htable is not None:
                ev = htable.ingest_evidence(
                    obs, source_tool=tool, stage=ci + 1,
                    read_depth=(2 if scrape_info.get("is_scrape") else (1 if scrape_info else 0)),
                    action_id=task_action_info.get("action_id"))
                htable.reject_contradicted()
                evidence_record_info = ev.to_dict()
                progressed = getattr(ev, "_progressed", False)
                if not progressed:
                    if scrape_info and ev.domain:
                        no_progress_domains.add(ev.domain)
                    if query and query not in lf_no_progress_queries and not scrape_info:
                        lf_no_progress_queries.append(query)   # feeds the LLM state card
                    best = htable.best_hypothesis()
                    if best is not None:
                        htable.note_no_progress(best.hypothesis_id)
                # Fold the same evidence into the candidate-slate frontier (multi-hop)
                # and let the frontier scheduler record its next action by info gain.
                if frontier is not None:
                    # DIRECTED ingest: when an LLM proposal drove this call, link the
                    # evidence to the proposal's SELECTED slot/constraints (req 3).
                    lf_pid = task_action_info.get("llm_frontier_proposal_id")
                    fev = frontier.ingest_evidence(
                        obs, source_tool=tool, stage=ci + 1,
                        read_depth=(2 if scrape_info.get("is_scrape") else (1 if scrape_info else 0)),
                        action_id=task_action_info.get("action_id"),
                        directed_slot_id=(task_action_info.get("target_slot_id") if lf_pid else None),
                        directed_constraint_ids=(task_action_info.get("tested_constraint_ids")
                                                 if lf_pid else None),
                        proposal_id=lf_pid,
                        read_candidate_id=(task_action_info.get("candidate_id")
                                           if scrape_info else None))
                    if not progressed:
                        for s in frontier.slates:
                            frontier.note_no_progress_for_slate(s)
                    # When the controller drove this call, record its execution outcome.
                    # For an LLM-frontier action, success requires progress on the SELECTED
                    # slot/constraint — NOT an arbitrary unrelated candidate (req 7).
                    if controller_handled and step_plan is not None:
                        pc = getattr(fev, "progress_components", {}) or {}
                        if lf_pid:
                            ok = (not call_failed) and bool(
                                pc.get("selected_slot_candidate_count", 0) > 0
                                or pc.get("selected_constraint_support_count", 0) > 0
                                or pc.get("hypothesis_score_delta", 0.0) > 0)
                        else:
                            ok = (not call_failed) and float(
                                getattr(fev, "evidence_progress_score", 0.0)) > 0
                        frontier.record_execution(step_plan.frontier_action_id, success=ok,
                                                  kind=step_plan.kind,
                                                  evidence_progress=getattr(fev, "evidence_progress_score", 0.0))
                        ctrl["tool_calls_from_frontier_actions"] += 1
                        if ok:
                            ctrl["frontier_action_execution_success_count"] += 1
                        else:
                            ctrl["frontier_action_execution_failure_count"] += 1
                        # Post-execution evidence-linkage integrity (req 2/3): did the
                        # evidence actually attach to the proposal's slot/constraints?
                        if lf_pid and cur_lf_meta is not None:
                            sel_slot = task_action_info.get("target_slot_id")
                            sel_cons = set(task_action_info.get("tested_constraint_ids") or [])
                            linked_slot = sel_slot in (getattr(fev, "supports_slot_ids", []) or [])
                            linked_cons = bool(sel_cons & set(
                                getattr(fev, "supports_constraint_ids", []) or []))
                            integ = dict(cur_lf_meta.get("integrity", {}))
                            integ["evidence_linked_to_selected_slot"] = linked_slot
                            integ["evidence_linked_to_selected_constraints"] = (
                                linked_cons or not sel_cons)
                            cur_lf_meta["integrity"] = integ
                            cur_lf_meta["progress_components"] = dict(pc)
                            cur_lf_meta["execution_success"] = ok
                    frontier.select_frontier_action(
                        budget_remaining=config.budget - len(calls),
                        reading_available=bool(reading_tools))

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
                    query_quality=stage_info.get("query_quality", 0.0),
                    n_query_candidates=stage_info.get("n_query_candidates", 0),
                    n_query_candidates_dropped=stage_info.get("n_query_candidates_dropped", 0),
                    query_candidates=stage_info.get("query_candidates", []),
                    stage=stage_info.get("stage", 1),
                    parent_query_id=stage_info.get("parent_query_id"),
                    candidate_entities=stage_info.get("candidate_entities", []),
                    selected_candidate=stage_info.get("selected_candidate"),
                    selection_reason=stage_info.get("selection_reason"),
                    evidence_improved=evidence_improved,
                    target_roles=list(target_roles),
                    selected_role=stage_info.get("selected_role"),
                    rejected_candidates=stage_info.get("rejected_candidates", []),
                    sticky_penalty=stage_info.get("sticky_penalty", 0.0),
                    no_progress=no_progress_flag,
                    scrape=scrape_info,
                    task_action=task_action_info,
                    frontier_action_id=task_action_info.get("frontier_action_id"),
                    llm_frontier_proposal_id=task_action_info.get("llm_frontier_proposal_id"),
                    evidence_record=evidence_record_info,
                )
            )

            if decision.stop and not task_frame:
                break
            if decision.arm == "fetch_page" and reading_tools and not task_frame:
                # Choose a URL to read: a known multihop target first, else (in
                # iterative mode) the best unread, non-contaminated evidence page.
                target = next((o for o in observations if getattr(o, "fetchable", False)), None)
                if target is None and iterative:
                    target = _best_read_target(observations, scraped_urls, no_progress_domains)
                if target is not None:
                    pending_read = target
            step += 1

        # Recompute once after the loop so a zero-budget (closed-book) attempt,
        # which never entered the loop body, still produces a candidate.
        candidate = self.answerer.answer(observations, item=item)
        final_answer = candidate.answer

        # Strict answer-support gate (Level 4): a final answer counts as
        # constraint-supported only when evidence tied to the hypothesis supports
        # the target — never from slot existence alone.
        support = None
        if task_frame and htable is not None:
            from regimes_probe.agent.action_planner import evaluate_answer_support
            support = evaluate_answer_support(frame, htable)
        frame_coverage: dict[str, Any] = {}
        if task_frame and htable is not None:
            # SAFETY invariant (req 4): a blocking constraint may be marked resolved ONLY
            # when a provider/cached evidence event supports it — never from prompt text.
            _resolved_no_ev = sum(
                1 for c in frame.constraints
                if c.status == "resolved" and not getattr(c, "supporting_evidence_ids", [])
                and (getattr(c, "blocks_answer_if_unresolved", False)
                     or getattr(c, "required", False) or getattr(c, "priority", "") == "high"))
            frame_coverage = dict(
                htable.coverage(),
                final_answer_supported_by_constraints=bool(support and support.supported),
                answer_support_gate=bool(support and support.supported),
                missing_support_reasons=(support.missing_support_reasons if support else []),
                terminal_action=terminal_action.get("kind", ""),
                n_actions=len(task_actions),
                initial_blocking_constraint_resolved_without_evidence_count=_resolved_no_ev)
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
            task_frame=(frame.to_dict() if task_frame and frame is not None else {}),
            hypothesis_summary=(htable.to_debug() if task_frame and htable is not None else {}),
            frame_coverage=frame_coverage,
            task_frame_parse=(frame_parse_meta if task_frame else {}),
            epistemic_mode=epistemic_decision.to_dict(),
            candidate_frontier=_with_read_accounting(_frontier_trace(
                frontier, epistemic_decision.selected_epistemic_mode, slate_skipped_reason,
                uses_layer=(config.enable_task_frame or config.auto_epistemic_mode
                            or config.force_task_frame),
                had_frame=(task_frame and frame is not None), controller=ctrl), read_acct),
            llm_frontier=({"enabled": True, "mode": lf_mode,
                           "model": self.llm_frontier.model if self.llm_frontier else "",
                           "steps": lf_steps,
                           "totals": self.llm_frontier.stats() if self.llm_frontier else {}}
                          if llm_frontier is not None else (
                              {"enabled": False,
                               "skipped_reason": (f"epistemic_mode={epistemic_decision.selected_epistemic_mode}"
                                                  if (lf_mode and task_frame and frame is not None)
                                                  else "")}
                              if lf_mode else {})),
        )
