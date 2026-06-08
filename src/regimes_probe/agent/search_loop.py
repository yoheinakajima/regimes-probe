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
    task_frame: dict[str, Any] = field(default_factory=dict)
    hypothesis_summary: dict[str, Any] = field(default_factory=dict)
    frame_coverage: dict[str, Any] = field(default_factory=dict)

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

        # Iterative clue resolution (staged search): clue spans + answer-shape to
        # chase intermediate entities found in earlier results.
        iterative = config.enable_iterative_clue_resolution
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
        task_frame = config.enable_task_frame
        frame = htable = planner = None
        if task_frame:
            from regimes_probe.agent.task_frame import parse_task_frame
            from regimes_probe.agent.hypothesis_table import HypothesisTable
            from regimes_probe.agent.action_planner import ActionPlanner, frame_read_value
            frame = parse_task_frame(item.id, item.question)
            htable = HypothesisTable(frame)
            planner = ActionPlanner(frame, htable)
            if not clue_terms:
                clue_terms = [t for c in frame.constraints for t in c.normalized_terms]
            if not answer_shape:
                answer_shape = list(frame.answer_shape_hints)

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
            if task_frame:
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
                    if action.kind in ("answer_if_supported", "abstain_if_no_path"):
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
                    decompose=config.enable_query_decomposition,
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
                if call_failed and scrape_info.get("is_scrape") and \
                        config.scrape_fallback_to_page_fetch and page_fetch_available:
                    # fail closed: retry the SAME url with the basic fetch next step.
                    # Do NOT mark it read yet, so the page_fetch retry is allowed.
                    pending_read = read_target_obs
                    force_page_fetch = True
                    scrape_info["fallback_to_page_fetch"] = True
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
                if not getattr(ev, "_progressed", False):
                    if scrape_info and ev.domain:
                        no_progress_domains.add(ev.domain)
                    best = htable.best_hypothesis()
                    if best is not None:
                        htable.note_no_progress(best.hypothesis_id)

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
            frame_coverage=(dict(htable.coverage(),
                                 final_answer_supported_by_constraints=bool(
                                     terminal_action.get("kind") == "answer_if_supported"),
                                 terminal_action=terminal_action.get("kind", ""),
                                 n_actions=len(task_actions))
                            if task_frame and htable is not None else {}),
        )
