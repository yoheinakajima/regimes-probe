"""Export a compact, secret-free ActiveGraph projection for a completed run.

``graph_projection.json`` is the standardized, typed object/relation graph of a
run — the thing a reviewer (or an offline forked ablation) reads instead of the
transient in-memory objects. Nodes are *projected* from already-recorded,
bounded, answer-free data: the per-condition outcomes, the bounded debug records
(themselves projections of the event log), the frozen memory snapshot, and the
run-level verdict/regime/claim objects.

Bounding rules (so the file stays compact and leak-free):
  * Every attempt gets compact ``question_attempt``/``answer_attempt``/
    ``grade_result``/``reward_assignment`` (+ ``failure_regime`` when failed) nodes.
  * Heavy sub-nodes (``query_plan``/``tool_call``/``tool_response``/
    ``evidence_observation``) are expanded only for the first ``detail_limit``
    attempts; all previews are length-bounded and contain no full pages.
  * The graph carries model *predictions* (bounded) but NOT gold answers — gold
    lives only in the audit ``debug_questions.jsonl``. Secrets never appear.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from regimes_probe.activegraph_pack.objects import (
    Objects, PROJECTION_OBJECTS)
from regimes_probe.activegraph_pack.relations import (
    Relations, PROJECTION_RELATIONS)

_MAX_FRAGMENT_NODES = 200


def _attempt_id(condition: str, budget: int, item_id: str) -> str:
    return f"{condition}-b{budget}-{item_id}"


class _GraphBuilder:
    def __init__(self) -> None:
        self.objects: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []
        self._oids: set[str] = set()
        self._rc = 0

    def obj(self, oid: str, type_: str, data: dict[str, Any]) -> str:
        if oid not in self._oids:
            self._oids.add(oid)
            self.objects.append({"id": oid, "type": type_, "data": data})
        return oid

    def has(self, oid: str) -> bool:
        return oid in self._oids

    def rel(self, source: str, target: str, type_: str,
            data: Optional[dict[str, Any]] = None) -> None:
        self._rc += 1
        self.relations.append({"id": f"rel_{self._rc:05d}", "source": source,
                               "target": target, "type": type_, "data": data or {}})


def build_graph_projection(
    run_id: str,
    *,
    runs,
    debug_records,
    eligibility_verdict: dict[str, Any],
    snapshot: Optional[dict[str, Any]] = None,
    meta: Optional[dict[str, Any]] = None,
    replay: Optional[dict[str, Any]] = None,
    failure_regimes: Optional[list[dict[str, Any]]] = None,
    claim_candidates: Optional[list[dict[str, Any]]] = None,
    detail_limit: int = 60,
) -> dict[str, Any]:
    """Assemble the standardized graph projection dict (compact, secret-free)."""
    meta = meta or {}
    snapshot = snapshot or {}
    replay = replay or {}
    g = _GraphBuilder()

    run_node = g.obj(f"benchmark_run#{run_id}", Objects.BENCHMARK_RUN, {
        "run_id": run_id, "dataset": meta.get("dataset"),
        "dataset_version": meta.get("dataset_version"),
        "conditions_present": eligibility_verdict.get("conditions_present", []),
        "budgets": meta.get("budgets", []),
    })

    # Per (condition, budget) outcome lookup for reward/flags (debug lacks reward).
    out_map: dict[tuple[str, int, str], Any] = {}
    for run in runs:
        for o in run.outcomes:
            out_map[(run.condition, run.budget, o.item_id)] = o

    regime_by_attempt = {fr["attempt_id"]: fr for fr in (failure_regimes or [])}

    detailed = 0
    for dr in debug_records:
        d = dr.to_dict() if hasattr(dr, "to_dict") else dict(dr)
        cond, bud, item_id = d["condition"], int(d["budget"]), d["item_id"]
        aid = _attempt_id(cond, bud, item_id)
        a_node = f"question_attempt#{aid}"
        item_node = g.obj(f"benchmark_item#{item_id}", Objects.BENCHMARK_ITEM,
                          {"item_id": item_id, "question_preview": d.get("question_preview", "")})
        g.obj(a_node, Objects.QUESTION_ATTEMPT, {
            "attempt_id": aid, "item_id": item_id, "condition": cond, "budget": bud,
            "correct": bool(d.get("correct")), "abstained": bool(d.get("abstained")),
            "tool_calls": len(d.get("tool_sequence", [])), "n_results": d.get("n_results", 0),
            "regime": d.get("regime", ""), "regime_names": d.get("regime_names", []),
        })
        g.rel(a_node, item_node, Relations.ATTEMPT_FOR_ITEM)

        # answer_attempt + grade_result + reward_assignment (every attempt).
        ans_node = g.obj(f"answer_attempt#{aid}", Objects.ANSWER_ATTEMPT, {
            "has_answer": bool(d.get("prediction_preview")),
            "prediction_preview": d.get("prediction_preview", ""),
            "abstained": bool(d.get("abstained")),
            "support_found": bool(d.get("support_found")),
        })
        grade_node = g.obj(f"grade_result#{aid}", Objects.GRADE_RESULT, {
            "correct": bool(d.get("correct")), "abstained": bool(d.get("abstained")),
        })
        g.rel(grade_node, ans_node, Relations.GRADE_FOR_ANSWER)
        o = out_map.get((cond, bud, item_id))
        rew_node = g.obj(f"reward_assignment#{aid}", Objects.REWARD_ASSIGNMENT, {
            "attempt_reward": round(float(getattr(o, "attempt_reward", 0.0)), 6) if o else None,
            "first_tool_hit": bool(getattr(o, "first_tool_hit", False)) if o else None,
            "found_hit": bool(d.get("found_hit")),
        })
        g.rel(rew_node, a_node, Relations.REWARD_FOR_ATTEMPT)

        # failure_regime (only when the attempt failed).
        fr = regime_by_attempt.get(aid)
        if fr is not None:
            fr_node = g.obj(f"failure_regime#{aid}", Objects.FAILURE_REGIME, fr)
            g.rel(fr_node, a_node, Relations.REGIME_FOR_ATTEMPT)

        # Heavy sub-graph only for the first `detail_limit` attempts.
        if detailed < detail_limit:
            detailed += 1
            plan_node = g.obj(f"routing_plan#{aid}", Objects.ROUTING_PLAN,
                              {"tool_sequence": d.get("tool_sequence", [])})
            g.rel(plan_node, a_node, Relations.PLAN_FOR_ATTEMPT)
            first_response: Optional[str] = None
            first_call: Optional[str] = None
            for i, c in enumerate(d.get("calls", [])):
                tc = g.obj(f"tool_call#{aid}#{i}", Objects.TOOL_CALL, {
                    "tool": c.get("tool"), "query_preview": c.get("query_preview", ""),
                    "n_results": c.get("n_results", 0), "failed": bool(c.get("failed")),
                    "status_code": c.get("status_code"), "error_type": c.get("error_type"),
                })
                qp = g.obj(f"query_plan#{aid}#{i}", Objects.QUERY_PLAN, {
                    "step": c.get("call_index", i), "query_arm": c.get("query_arm"),
                    "query_preview": c.get("query_preview", ""), "tool": c.get("tool"),
                })
                tr = g.obj(f"tool_response#{aid}#{i}", Objects.TOOL_RESPONSE, {
                    "tool": c.get("tool"), "n_results": c.get("n_results", 0),
                    "failed": bool(c.get("failed")), "status_code": c.get("status_code"),
                    "error_type": c.get("error_type"),
                })
                g.rel(qp, tc, Relations.QUERY_FOR_TOOL_CALL)
                g.rel(tr, tc, Relations.RESPONSE_FOR_TOOL_CALL)
                if first_call is None:
                    first_call, first_response = tc, tr
                if not c.get("failed") and first_response is None:
                    first_response = tr
            if first_call is not None:
                g.rel(rew_node, first_call, Relations.REWARD_FROM_TOOL_CALL)
            for j, ev in enumerate(d.get("evidence", [])):
                ev_node = g.obj(f"evidence_observation#{aid}#{j}", Objects.EVIDENCE_OBSERVATION, {
                    "title_preview": ev.get("title_preview", ""), "url": ev.get("url", ""),
                    "snippet_preview": ev.get("snippet_preview", ""),
                    "supports": bool(ev.get("supports")),
                    "source_authority": ev.get("source_authority"),
                })
                if first_response is not None:
                    g.rel(ev_node, first_response, Relations.EVIDENCE_FROM_TOOL_RESPONSE)
                if ev.get("supports"):
                    g.rel(ans_node, ev_node, Relations.ANSWER_SUPPORTED_BY_EVIDENCE)

            # --- Level 4 task-frame subgraph (when present) ---
            tf = d.get("task_frame") or {}
            if tf:
                pm = d.get("task_frame_parse") or {}
                tf_node = g.obj(f"task_frame#{aid}", Objects.TASK_FRAME, {
                    "n_target_slots": len(tf.get("target_answer_slots", [])),
                    "n_latent_slots": len(tf.get("latent_slots", [])),
                    "n_constraints": len(tf.get("constraints", [])),
                    "parse_quality": tf.get("parse_quality"),
                    "parser_used": pm.get("parser_used", "deterministic"),
                    "prompt_version": pm.get("prompt_version", ""),
                    "prompt_hash": pm.get("prompt_hash", ""),
                    "fallback_reason": pm.get("fallback_reason", ""),
                    "known_context_terms": tf.get("known_context_terms", [])[:8]})
                g.rel(tf_node, a_node, Relations.FRAME_FOR_ATTEMPT)
                # Epistemic escalation decision (front-door mode selection).
                em = d.get("epistemic_mode") or {}
                if em.get("selected_epistemic_mode"):
                    em_node = g.obj(f"epistemic_mode_decision#{aid}", Objects.EPISTEMIC_MODE_DECISION, {
                        "selected_epistemic_mode": em.get("selected_epistemic_mode"),
                        "escalation_reason": em.get("escalation_reason"),
                        "skipped_heavy_parser_reason": em.get("skipped_heavy_parser_reason"),
                        "auto": em.get("auto"), "applied": em.get("applied")})
                    g.rel(em_node, a_node, Relations.EPISTEMIC_MODE_FOR_ATTEMPT)
                for s in (tf.get("target_answer_slots", []) + tf.get("latent_slots", [])):
                    s_node = g.obj(f"latent_slot#{aid}#{s['slot_id']}", Objects.LATENT_SLOT, {
                        "slot_name": s.get("slot_name"), "slot_role": s.get("slot_role"),
                        "is_target_answer_slot": s.get("is_target_answer_slot"),
                        "is_intermediate_slot": s.get("is_intermediate_slot")})
                    g.rel(s_node, tf_node, Relations.SLOT_IN_FRAME)
                for con in tf.get("constraints", []):
                    # Open-world SEMANTIC constraint: keep the raw label/facets AND the
                    # derived affordances (the planner branches on the latter).
                    c_node = g.obj(f"constraint#{aid}#{con['constraint_id']}",
                                   Objects.SEMANTIC_CONSTRAINT, {
                        "constraint_type": con.get("constraint_type"),
                        "semantic_label": con.get("semantic_label"),
                        "semantic_facets": con.get("semantic_facets", []),
                        "affordances": con.get("affordances", []),
                        "required": con.get("required"), "priority": con.get("priority"),
                        "blocks_answer_if_unresolved": con.get("blocks_answer_if_unresolved"),
                        "status": con.get("status"),
                        "specificity_score": con.get("specificity_score"),
                        "text_span": con.get("text_span", "")[:120]})
                    for sid in con.get("applies_to", []):
                        g.rel(c_node, f"latent_slot#{aid}#{sid}", Relations.CONSTRAINT_APPLIES_TO_SLOT)
                    for facet in con.get("semantic_facets", [])[:8]:
                        f_node = g.obj(f"constraint_facet#{aid}#{facet}", Objects.CONSTRAINT_FACET,
                                       {"facet": facet})
                        g.rel(c_node, f_node, Relations.CONSTRAINT_HAS_FACET)
                    for aff in con.get("affordances", []):
                        af_node = g.obj(f"operational_affordance#{aid}#{aff}",
                                        Objects.OPERATIONAL_AFFORDANCE, {"affordance": aff})
                        g.rel(c_node, af_node, Relations.CONSTRAINT_HAS_AFFORDANCE)
                    # an unresolved blocking constraint blocks the answer.
                    if (con.get("status") != "resolved"
                            and (con.get("blocks_answer_if_unresolved")
                                 or "can_block_answer" in con.get("affordances", []))):
                        g.rel(c_node, tf_node, Relations.UNRESOLVED_CONSTRAINT_BLOCKS_ANSWER)
                # epistemic actions + evidence records (per call)
                for i, c in enumerate(d.get("calls", [])):
                    ta = c.get("task_action") or {}
                    if ta.get("kind"):
                        act_node = g.obj(f"epistemic_action#{aid}#{i}", Objects.EPISTEMIC_ACTION, {
                            "kind": ta.get("kind"), "query_arm": ta.get("query_arm"),
                            "query_text_preview": ta.get("query_text_preview", ""),
                            "expected_information_gain": ta.get("expected_information_gain")})
                        if ta.get("target_slot_id"):
                            g.rel(act_node, f"latent_slot#{aid}#{ta['target_slot_id']}",
                                  Relations.ACTION_TARGETS_SLOT)
                        for cid in ta.get("tested_constraint_ids", []):
                            g.rel(act_node, f"constraint#{aid}#{cid}", Relations.ACTION_TESTS_CONSTRAINT)
                        if "read_value" in ta:
                            rv_node = g.obj(f"read_value_decision#{aid}#{i}",
                                            Objects.READ_VALUE_DECISION, ta["read_value"])
                            g.rel(act_node, rv_node, Relations.ACTION_TESTS_CONSTRAINT)
                    er = c.get("evidence_record") or {}
                    if er.get("evidence_id"):
                        er_node = g.obj(f"evidence_record#{aid}#{er['evidence_id']}",
                                        Objects.EVIDENCE_RECORD, {
                            "source_tool": er.get("source_tool"), "domain": er.get("domain"),
                            "evidence_progress_score": er.get("evidence_progress_score"),
                            "read_depth": er.get("read_depth")})
                        for cid in er.get("supports_constraint_ids", []):
                            g.rel(er_node, f"constraint#{aid}#{cid}", Relations.EVIDENCE_SUPPORTS_CONSTRAINT)
                        for sid in er.get("supports_slot_ids", []):
                            g.rel(er_node, f"latent_slot#{aid}#{sid}", Relations.EVIDENCE_SUPPORTS_SLOT)
                # hypotheses
                hs = d.get("hypothesis_summary") or {}
                for h in hs.get("top_hypotheses", []):
                    h_node = g.obj(f"hypothesis#{aid}#{h['hypothesis_id']}", Objects.HYPOTHESIS, {
                        "support_score": h.get("support_score"),
                        "confidence_score": h.get("confidence_score"),
                        "coverage_score": h.get("coverage_score"), "active": h.get("active")})
                    for sid, ctext in (h.get("slot_assignments") or {}).items():
                        sa_node = g.obj(f"slot_assignment#{aid}#{h['hypothesis_id']}#{sid}",
                                        Objects.SLOT_ASSIGNMENT,
                                        {"slot_id": sid, "candidate_text": str(ctext)[:80]})
                        g.rel(h_node, sa_node, Relations.HYPOTHESIS_ASSIGNS_CANDIDATE)
                        g.rel(h_node, f"latent_slot#{aid}#{sid}", Relations.HYPOTHESIS_ASSIGNS_SLOT)
                    if h.get("support_score", 0) > 0:
                        g.rel(h_node, tf_node, Relations.HYPOTHESIS_SUPPORTED_BY_EVIDENCE)
                    cov = d.get("frame_coverage", {})
                    if cov.get("final_answer_supported_by_constraints"):
                        g.rel(ans_node, h_node, Relations.ANSWER_SUPPORTED_BY_HYPOTHESIS)
                        # an explicit answer→hypothesis→slot→evidence→constraint path.
                        asp = g.obj(f"answer_support_path#{aid}", Objects.ANSWER_SUPPORT_PATH, {
                            "hypothesis_id": h.get("hypothesis_id"),
                            "answer_support_gate": cov.get("answer_support_gate"),
                            "missing_support_reasons": cov.get("missing_support_reasons", [])})
                        g.rel(ans_node, asp, Relations.ANSWER_SUPPORTED_BY_PATH)
                for h in hs.get("rejected_hypotheses", []):
                    hr = g.obj(f"hypothesis#{aid}#{h['hypothesis_id']}", Objects.HYPOTHESIS,
                               {"active": False, "rejection_reason": h.get("rejection_reason")})
                    g.rel(hr, tf_node, Relations.HYPOTHESIS_REJECTED_BY_EVIDENCE)

    # --- run-level objects: memory_snapshot + policy_fragment lineage ---
    fragments = snapshot.get("fragments", {}) if snapshot else {}
    if snapshot:
        ms_node = g.obj(f"memory_snapshot#{run_id}", Objects.MEMORY_SNAPSHOT, {
            "n_fragments": len(fragments), "n_traces": len(snapshot.get("traces", [])),
            "memory_snapshot_leakage_pass": eligibility_verdict.get("memory_snapshot_leakage_pass"),
        })
        for ck, frag in list(fragments.items())[:_MAX_FRAGMENT_NODES]:
            pf_node = g.obj(f"policy_fragment#{ck}", Objects.POLICY_FRAGMENT, {
                "fragment_id": frag.get("fragment_id", ck), "cluster_key": ck,
                "support_count": frag.get("support_count", 0),
                "correct_count": frag.get("correct_count", 0),
                "best_tool": frag.get("best_tool"), "best_query_arm": frag.get("best_query_arm"),
                "top_tool_rewards": frag.get("top_tool_rewards", []),
                "top_query_rewards": frag.get("top_query_rewards", []),
                "source_trace_ids": frag.get("source_trace_ids", []),
                "source_attempt_ids": frag.get("source_attempt_ids", []),
            })
            g.rel(ms_node, pf_node, Relations.MEMORY_SNAPSHOT_CONTAINS_FRAGMENT)
            # lineage edge: fragment <- traces (ids carried in edge data, answer-free)
            g.rel(pf_node, ms_node, Relations.POLICY_FRAGMENT_FROM_TRACES,
                  {"source_trace_ids": frag.get("source_trace_ids", [])})

    # --- eligibility_verdict object ---
    ev_node = g.obj(f"eligibility_verdict#{run_id}", Objects.ELIGIBILITY_VERDICT,
                    dict(eligibility_verdict))
    g.rel(ev_node, run_node, Relations.ELIGIBILITY_FOR_RUN)

    # --- report + claim_candidate objects ---
    report_node = g.obj(f"report#{run_id}", Objects.REPORT, {
        "run_id": run_id,
        "structurally_valid": eligibility_verdict.get("structurally_valid"),
        "headline_eligible_memory_claim": eligibility_verdict.get("headline_eligible_memory_claim"),
    })
    for i, cc in enumerate(claim_candidates or []):
        cc_node = g.obj(f"claim_candidate#{run_id}#{i}", Objects.CLAIM_CANDIDATE, cc)
        g.rel(cc_node, report_node, Relations.CLAIM_SUPPORTED_BY_ARTIFACT,
              {"artifacts": cc.get("supporting_artifacts", [])})

    counts: dict[str, int] = {}
    for o in g.objects:
        counts[o["type"]] = counts.get(o["type"], 0) + 1
    rel_counts: dict[str, int] = {}
    for r in g.relations:
        rel_counts[r["type"]] = rel_counts.get(r["type"], 0) + 1

    return {
        "schema_version": "1",
        "run_id": run_id,
        "object_types": list(PROJECTION_OBJECTS),
        "relation_types": list(PROJECTION_RELATIONS),
        "detail_limit": detail_limit,
        # provenance: the projection is derived from this event-log replay summary.
        "event_log": {
            "n_events": replay.get("n_events"),
            "n_objects": replay.get("n_objects"),
            "n_relations": replay.get("n_relations"),
            "projection_matches": replay.get("projection_matches"),
        },
        "object_counts": counts,
        "relation_counts": rel_counts,
        "objects": g.objects,
        "relations": g.relations,
    }


def default_claim_candidates(eligibility_verdict: dict[str, Any]) -> list[dict[str, Any]]:
    """Conservative, structural claim candidates derived from a verdict.

    Never includes a performance claim; those are gated by
    ``headline_eligible_memory_claim`` and produced by scripts/generate_claims.py.
    """
    v = eligibility_verdict or {}
    out: list[dict[str, Any]] = []
    if v.get("replay_pass"):
        out.append({"claim": "The graph is a deterministic projection of the event log.",
                    "kind": "structural",
                    "supporting_artifacts": ["replay_check.md", "graph_projection.json"]})
    if v.get("memory_snapshot_leakage_pass"):
        out.append({"claim": "Frozen policy memory contains no answer text.",
                    "kind": "structural",
                    "supporting_artifacts": ["memory_snapshot.json", "report.json"]})
    if v.get("split_disjoint"):
        out.append({"claim": "OPTIMIZE and CONFIRM splits are disjoint.",
                    "kind": "structural", "supporting_artifacts": ["run_manifest.json"]})
    if v.get("budget_enforced"):
        out.append({"claim": "Tool-call budgets were enforced.",
                    "kind": "structural", "supporting_artifacts": ["per_question.csv"]})
    return out


def write_graph_projection(run_dir: str | Path, projection: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "graph_projection.json"
    path.write_text(json.dumps(projection, indent=2, sort_keys=False), encoding="utf-8")
    return path
