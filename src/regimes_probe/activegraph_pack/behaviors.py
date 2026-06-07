"""Event log + deterministic recording behaviors + replay check.

The event log is the source of truth; the graph is a deterministic projection.
:class:`EventLog` uses the real ActiveGraph ``Graph`` when the package is
installed (the native path) and otherwise a faithful, equivalent fallback so the
whole project runs with only the standard library + PyYAML.

Every function in this module is a *behavior body*: deterministic, with no
``random`` / ``datetime.now`` / ``uuid`` / network I/O. Time comes from a frozen
clock and ids from a monotonic generator; the only I/O in a recorded run is the
provider call inside :class:`~regimes_probe.activegraph_pack.tools.RecordingInvoker`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from regimes_probe.activegraph_pack.events import Events
from regimes_probe.activegraph_pack.objects import Objects
from regimes_probe.activegraph_pack.relations import Relations

try:  # native path
    from activegraph import Event, FrozenClock, Graph, IDGen
    _HAS_ACTIVEGRAPH = True
except Exception:  # pragma: no cover - exercised only without the optional dep
    _HAS_ACTIVEGRAPH = False


# --------------------------------------------------------------------------
# Fallback graph (faithful subset of ActiveGraph semantics)
# --------------------------------------------------------------------------
class _FallbackEvent:
    __slots__ = ("id", "type", "payload", "actor", "caused_by", "timestamp")

    def __init__(self, id, type, payload, actor, caused_by, timestamp):  # noqa: A002
        self.id, self.type, self.payload = id, type, payload
        self.actor, self.caused_by, self.timestamp = actor, caused_by, timestamp

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "payload": self.payload,
                "actor": self.actor, "caused_by": self.caused_by, "timestamp": self.timestamp}


class _FallbackGraph:
    """Append-only log + deterministic projection. Mirrors ActiveGraph CONTRACT #2."""

    def __init__(self, run_id: str = "run_local", timestamp: str = "2026-05-15T10:32:01Z") -> None:
        self.run_id = run_id
        self._ts = timestamp
        self._ec = self._oc = self._rc = 0
        self._events: list[_FallbackEvent] = []
        self._objects: dict[str, dict[str, Any]] = {}
        self._relations: dict[str, dict[str, Any]] = {}

    @property
    def events(self):
        return list(self._events)

    def _emit(self, type, payload, actor, caused_by):  # noqa: A002
        self._ec += 1
        ev = _FallbackEvent(f"evt_{self._ec:03d}", type, payload, actor, caused_by, self._ts)
        self._events.append(ev)
        # project
        if type == "object.created":
            o = payload["object"]
            self._objects[o["id"]] = o
        elif type == "relation.created":
            r = payload["relation"]
            self._relations[r["id"]] = r
        return ev

    def emit_domain(self, type, payload, actor="agent", caused_by=None):  # noqa: A002
        return self._emit(type, payload, actor, caused_by)

    def add_object(self, type, data, *, actor="agent", caused_by=None):  # noqa: A002
        self._oc += 1
        oid = f"{type}#{self._oc}"
        obj = {"id": oid, "type": type, "data": data, "version": 1, "provenance": {"actor": actor}}
        self._emit("object.created", {"object": obj, "id": oid}, actor, caused_by)
        return oid

    def add_relation(self, source, target, type, data=None, *, actor="agent", caused_by=None):  # noqa: A002
        self._rc += 1
        rid = f"rel_{self._rc:03d}"
        rel = {"id": rid, "source": source, "target": target, "type": type,
               "data": data or {}, "provenance": {"actor": actor}}
        self._emit("relation.created", {"relation": rel, "id": rid, "source": source,
                                        "target": target}, actor, caused_by)
        return rid

    def objects(self, type=None):  # noqa: A002
        return [o for o in self._objects.values() if type is None or o["type"] == type]

    def relations(self, type=None):  # noqa: A002
        return [r for r in self._relations.values() if type is None or r["type"] == type]


class EventLog:
    """Thin uniform wrapper over the ActiveGraph graph (or the fallback)."""

    def __init__(self, run_id: str = "regimes-probe-run", timestamp: str = "2026-05-15T10:32:01Z") -> None:
        self.run_id = run_id
        self.native = _HAS_ACTIVEGRAPH
        if _HAS_ACTIVEGRAPH:
            self._g = Graph(ids=IDGen(), clock=FrozenClock(timestamp), run_id=run_id)
        else:  # pragma: no cover
            self._g = _FallbackGraph(run_id=run_id, timestamp=timestamp)

    # ---- emit a domain (non object/relation) event ----
    def emit(self, type: str, payload: dict[str, Any], *, actor: str = "agent",
             caused_by: Optional[str] = None):  # noqa: A002
        if _HAS_ACTIVEGRAPH:
            ev = Event(id=self._g.ids.event(), type=type, payload=payload, actor=actor,
                       caused_by=caused_by, timestamp=self._g.clock.now())
            return self._g.emit(ev)
        return self._g.emit_domain(type, payload, actor=actor, caused_by=caused_by)  # type: ignore[attr-defined]

    def add_object(self, type: str, data: dict[str, Any], *, actor: str = "agent",
                   caused_by: Optional[str] = None) -> str:  # noqa: A002
        obj = self._g.add_object(type, data, actor=actor, caused_by=caused_by)
        return obj.id if _HAS_ACTIVEGRAPH else obj

    def add_relation(self, source: str, target: str, type: str, data: Optional[dict[str, Any]] = None,
                     *, actor: str = "agent", caused_by: Optional[str] = None) -> str:  # noqa: A002
        rel = self._g.add_relation(source, target, type, data or {}, actor=actor, caused_by=caused_by)
        return rel.id if _HAS_ACTIVEGRAPH else rel

    # ---- read ----
    @property
    def events(self) -> list[Any]:
        return self._g.events

    def event_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._g.events]

    def objects(self, type: Optional[str] = None) -> list[dict[str, Any]]:  # noqa: A002
        if _HAS_ACTIVEGRAPH:
            return [o.to_dict() for o in self._g.objects(type=type)]
        return self._g.objects(type=type)  # type: ignore[attr-defined]

    def relations(self, type: Optional[str] = None) -> list[dict[str, Any]]:  # noqa: A002
        if _HAS_ACTIVEGRAPH:
            return [r.to_dict() for r in self._g.relations(type=type)]
        return self._g.relations(type=type)  # type: ignore[attr-defined]

    @property
    def graph(self):
        return self._g


# --------------------------------------------------------------------------
# Replay check (determinism of the projection from the log)
# --------------------------------------------------------------------------
@dataclass
class ReplayReport:
    n_events: int
    n_objects: int
    n_relations: int
    projection_matches: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_events": self.n_events,
            "n_objects": self.n_objects,
            "n_relations": self.n_relations,
            "projection_matches": self.projection_matches,
            "detail": self.detail,
        }


def replay_check(log: EventLog) -> ReplayReport:
    """Re-project the event log into a fresh graph and confirm it matches.

    The log is the source of truth; replaying its events must reproduce the same
    object/relation projection byte-for-byte. Uses ActiveGraph's replay path
    when available, otherwise the fallback projector.
    """
    events = log.events
    live_objects = {o["id"]: o for o in log.objects()}
    live_relations = {r["id"]: r for r in log.relations()}

    if _HAS_ACTIVEGRAPH:
        fresh = Graph(ids=IDGen(), clock=FrozenClock(), run_id=log.run_id + "-replay")
        for ev in events:
            fresh._replay_event(ev)
        replay_objects = {o.id: o.to_dict() for o in fresh.all_objects()}
        replay_relations = {r.id: r.to_dict() for r in fresh.all_relations()}
    else:  # pragma: no cover
        fg = _FallbackGraph(run_id=log.run_id + "-replay")
        for ev in events:
            fg._emit(ev.type, ev.payload, ev.actor, ev.caused_by)
        replay_objects = {o["id"]: o for o in fg.objects()}
        replay_relations = {r["id"]: r for r in fg.relations()}

    obj_match = set(live_objects) == set(replay_objects)
    rel_match = set(live_relations) == set(replay_relations)
    matches = obj_match and rel_match
    detail = "ok" if matches else f"objects_match={obj_match} relations_match={rel_match}"
    return ReplayReport(
        n_events=len(events),
        n_objects=len(live_objects),
        n_relations=len(live_relations),
        projection_matches=matches,
        detail=detail,
    )


# --------------------------------------------------------------------------
# Recording behaviors: emit the canonical event sequence for one run.
# Each function is deterministic; the only I/O is the provider call inside the
# RecordingInvoker (imported lazily to avoid an import cycle with tools.py).
# --------------------------------------------------------------------------
from regimes_probe.agent.search_loop import LoopRecorder  # noqa: E402


class _AGLoopRecorder(LoopRecorder):
    """Emits planning/evidence/verification/stop events in causal order."""

    def __init__(self, log: EventLog, attempt_obj_id: str) -> None:
        self.log = log
        self.attempt_obj_id = attempt_obj_id
        self.evidence_obj_ids: list[str] = []

    def on_routing_plan(self, plan: dict[str, Any]) -> None:
        ev = self.log.emit(Events.ROUTING_PLAN_CREATED,
                           {"sequence": plan["sequence"], "attempt": self.attempt_obj_id})
        oid = self.log.add_object(Objects.ROUTING_PLAN,
                                  {"sequence": plan["sequence"],
                                   "explanation": plan.get("explanation", {})}, caused_by=ev.id)
        self.log.add_relation(oid, self.attempt_obj_id, Relations.PLAN_FOR_ATTEMPT, caused_by=ev.id)

    def on_query_plan(self, step: int, plan: dict[str, Any]) -> None:
        ev = self.log.emit(Events.QUERY_PLAN_CREATED,
                           {"step": step, "arm": plan["arm"], "query": plan["query"]})
        self.log.add_object(Objects.QUERY_PLAN,
                            {"step": step, "arm": plan["arm"], "query": plan["query"]},
                            caused_by=ev.id)

    def on_evidence(self, step: int, observations) -> None:
        for o in observations:
            ev = self.log.emit(Events.EVIDENCE_OBSERVED, {"step": step, **o.to_public_dict()})
            if o.supports:
                oid = self.log.add_object(Objects.EVIDENCE_OBSERVATION,
                                          o.to_public_dict(), caused_by=ev.id)
                self.evidence_obj_ids.append(oid)

    def on_candidate(self, step: int, candidate) -> None:
        self.log.emit(Events.CANDIDATE_ANSWER_CREATED,
                      {"step": step, "has_answer": candidate.answer is not None,
                       "support_count": candidate.support_count})

    def on_verification(self, step: int, vstate) -> None:
        self.log.emit(Events.VERIFICATION_COMPLETED, {"step": step, **vstate.to_dict()})

    def on_stop(self, step: int, decision: dict[str, Any]) -> None:
        self.log.emit(Events.STOP_DECISION_CREATED, {"step": step, **decision})


def record_run_start(log: EventLog, *, dataset_version: str, condition: str,
                     config: dict[str, Any]) -> str:
    """Emit ``benchmark.started`` and create the ``benchmark_run`` object."""
    ev = log.emit(Events.BENCHMARK_STARTED,
                  {"dataset_version": dataset_version, "condition": condition, "config": config})
    return log.add_object(Objects.BENCHMARK_RUN,
                          {"dataset_version": dataset_version, "condition": condition,
                           "config": config}, caused_by=ev.id)


def record_attempt(log: EventLog, run_obj_id: str, agent, item, memory, providers, *,
                   budget: int, explore: bool, attempt_id: str,
                   signature=None) -> dict[str, Any]:
    """Record the full hot path for one item. Returns the trace + ids."""
    from regimes_probe.activegraph_pack.tools import RecordingInvoker  # lazy: avoid cycle

    qev = log.emit(Events.ITEM_QUEUED, {"item_id": item.id})
    item_obj = log.add_object(Objects.BENCHMARK_ITEM, item.public_dict(), caused_by=qev.id)
    aev = log.emit(Events.ATTEMPT_STARTED,
                   {"item_id": item.id, "budget": budget, "explore": explore,
                    "attempt_id": attempt_id})
    attempt_obj = log.add_object(Objects.QUESTION_ATTEMPT,
                                 {"item_id": item.id, "budget": budget,
                                  "mode": "explore" if explore else "exploit"}, caused_by=aev.id)
    log.add_relation(attempt_obj, item_obj, Relations.ATTEMPT_FOR_ITEM, caused_by=aev.id)

    sig = signature or agent.signature(item)
    sev = log.emit(Events.SIGNATURE_CREATED,
                   {"cluster_key": sig.cluster_key, "norm_hash": sig.norm_hash,
                    "features": {k: v for k, v in sig.features.items() if v}})
    sig_obj = log.add_object(Objects.QUERY_SIGNATURE,
                             {"cluster_key": sig.cluster_key, "norm_hash": sig.norm_hash,
                              "features": sig.features, "lexical": sig.lexical}, caused_by=sev.id)
    log.add_relation(sig_obj, attempt_obj, Relations.SIGNATURE_FOR_ATTEMPT, caused_by=sev.id)

    recorder = _AGLoopRecorder(log, attempt_obj)
    invoker = RecordingInvoker(providers, log, attempt_id=attempt_id)
    trace = agent.attempt(item, memory, providers, budget=budget, explore=explore,
                          attempt_id=attempt_id, invoker=invoker, signature=sig,
                          recorder=recorder)

    fev = log.emit(Events.FINAL_ANSWER_CREATED,
                   {"has_answer": trace.final_answer is not None,
                    "tool_calls": trace.tool_calls})
    final_obj = log.add_object(Objects.FINAL_ANSWER,
                               {"has_answer": trace.final_answer is not None,
                                "support_count": trace.candidate.support_count,
                                "tool_calls": trace.tool_calls}, caused_by=fev.id)
    for eid in recorder.evidence_obj_ids:
        log.add_relation(final_obj, eid, Relations.ANSWER_SUPPORTED_BY_EVIDENCE, caused_by=fev.id)

    return {"trace": trace, "attempt_obj": attempt_obj, "final_obj": final_obj,
            "signature": sig, "recorded": invoker.recorded}


def record_grade_and_reward(log: EventLog, rec: dict[str, Any], item, *, weights,
                            freshness_sensitive: bool):
    """Emit grade + reward events and create grade_result / reward objects."""
    from regimes_probe.eval.grader import grade as grade_fn, normalize_answer
    from regimes_probe.eval.reward import compute_rewards

    trace = rec["trace"]
    grade = grade_fn(item, trace.final_answer)
    gev = log.emit(Events.GRADE_COMPLETED,
                   {"item_id": item.id, "correct": grade.correct, "abstained": grade.abstained,
                    "method": grade.method})
    grade_obj = log.add_object(Objects.GRADE_RESULT, grade.to_dict(), caused_by=gev.id)
    log.add_relation(grade_obj, rec["final_obj"], Relations.GRADE_FOR_ANSWER, caused_by=gev.id)

    gold_norms = [normalize_answer(g) for g in item.gold_answers() if g]
    reward = compute_rewards(trace, correct=grade.correct, gold_norms=gold_norms,
                             weights=weights, freshness_sensitive=freshness_sensitive)
    rev = log.emit(Events.REWARD_COMPUTED,
                   {"attempt_reward": round(reward.attempt_reward, 6), "flags": reward.flags})
    trace_reward_obj = log.add_object(Objects.TRACE_REWARD, reward.to_dict(), caused_by=rev.id)
    log.add_relation(trace_reward_obj, rec["attempt_obj"], Relations.REWARD_FOR_ATTEMPT, caused_by=rev.id)
    for tool, r in reward.arm_rewards.get("tool", {}).items():
        log.add_object(Objects.TOOL_REWARD, {"tool": tool, "reward": round(r, 6)}, caused_by=rev.id)
    return grade, reward


def record_policy_update(log: EventLog, memory, signature, attempt_id: str, reward, trace,
                         *, correct: bool) -> None:
    """Apply the policy update (memory.observe) and emit ``policy_update.applied``."""
    memory.observe(signature, attempt_id, reward.arm_rewards,
                   correct=correct, tool_calls=trace.tool_calls)
    log.emit(Events.POLICY_UPDATE_APPLIED,
             {"attempt_id": attempt_id, "cluster_key": signature.cluster_key,
              "families": list(reward.arm_rewards.keys())})
