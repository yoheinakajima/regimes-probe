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


def _project_frontier(g: "_GraphBuilder", aid: str, tf_node: str, cf: dict[str, Any],
                      calls: Optional[list] = None) -> None:
    """Project the candidate-slate / frontier subgraph (Level 5)."""
    cand_node = {}  # candidate_id -> node id
    for slate in cf.get("slates", []):
        sl_id = g.obj(f"candidate_slate#{aid}#{slate['slate_id']}", Objects.CANDIDATE_SLATE, {
            "slot_id": slate.get("slot_id"), "slot_role": slate.get("slot_role"),
            "slot_descriptor": slate.get("slot_descriptor"),
            "slate_confidence": slate.get("slate_confidence"),
            "n_active": len(slate.get("active_candidate_ids", [])),
            "n_confirmed": len(slate.get("confirmed_candidate_ids", [])),
            "n_rejected": len(slate.get("rejected_candidate_ids", []))})
        g.rel(sl_id, f"latent_slot#{aid}#{slate['slot_id']}", Relations.SLATE_FOR_SLOT)
        for c in slate.get("top_candidates", []):
            cid = g.obj(f"slot_candidate#{aid}#{c['candidate_id']}", Objects.SLOT_CANDIDATE, {
                "candidate_text_preview": c.get("candidate_text_preview"),
                "normalized_text_hash": c.get("normalized_text_hash"),
                "inferred_role": c.get("inferred_role"), "slot_id": c.get("slot_id"),
                "status": c.get("status"), "status_reason": c.get("status_reason"),
                "evidence_score": c.get("evidence_score"),
                "source_domains": c.get("source_domains", []),
                "constraints_supported": c.get("constraints_supported", []),
                "constraints_contradicted": c.get("constraints_contradicted", [])})
            cand_node[c["candidate_id"]] = cid
            g.rel(cid, sl_id, Relations.CANDIDATE_IN_SLATE)
            g.rel(cid, f"latent_slot#{aid}#{c['slot_id']}", Relations.CANDIDATE_ASSIGNED_TO_SLOT)
            st = g.obj(f"candidate_status#{aid}#{c['candidate_id']}", Objects.CANDIDATE_STATUS,
                       {"status": c.get("status"), "status_reason": c.get("status_reason")})
            g.rel(cid, st, Relations.EVIDENCE_UPDATES_CANDIDATE_STATUS)
            for con in c.get("constraints_supported", []):
                g.rel(cid, f"constraint#{aid}#{con}", Relations.CANDIDATE_SUPPORTS_CONSTRAINT)
            for con in c.get("constraints_contradicted", []):
                g.rel(cid, f"constraint#{aid}#{con}", Relations.CANDIDATE_CONTRADICTS_CONSTRAINT)
            if c.get("status") == "rejected":
                rj = g.obj(f"candidate_rejection#{aid}#{c['candidate_id']}",
                           Objects.CANDIDATE_REJECTION, {"reason": c.get("status_reason")})
                g.rel(cid, rj, Relations.CANDIDATE_REJECTED_BY_EVIDENCE)
            if c.get("status") == "confirmed":
                pr = g.obj(f"candidate_promotion#{aid}#{c['candidate_id']}",
                           Objects.CANDIDATE_PROMOTION, {"reason": c.get("status_reason")})
                g.rel(cid, pr, Relations.CANDIDATE_CONFIRMED_BY_EVIDENCE)
            if c.get("duplicate_of"):
                mg = g.obj(f"candidate_merge#{aid}#{c['candidate_id']}", Objects.CANDIDATE_MERGE,
                           {"merged_into": c.get("duplicate_of")})
                g.rel(cid, mg, Relations.CANDIDATE_MERGED_INTO)
    action_nodes: dict[str, str] = {}
    for h in cf.get("top_hypotheses", []):
        hid = g.obj(f"hypothesis_state#{aid}#{h['hypothesis_id']}", Objects.HYPOTHESIS_STATE, {
            "support_score": h.get("support_score"), "coverage_score": h.get("coverage_score"),
            "source_diversity_score": h.get("source_diversity_score"),
            "confidence_score": h.get("confidence_score"), "active": h.get("active")})
        g.rel(hid, tf_node, Relations.FRAME_FOR_ATTEMPT)
        if not h.get("active"):
            g.rel(hid, tf_node, Relations.HYPOTHESIS_REJECTED_BY_CONSTRAINT)
        # which frontier action last updated this hypothesis.
        lfa = h.get("last_frontier_action_id")
        if lfa:
            g.rel(hid, f"frontier_action#{aid}#{lfa}",
                  Relations.HYPOTHESIS_UPDATED_AFTER_FRONTIER_ACTION)
    for a in cf.get("frontier_actions", []):
        plan = a.get("plan") or {}
        an = g.obj(f"frontier_action#{aid}#{a['action_id']}", Objects.FRONTIER_ACTION, {
            "action_type": a.get("action_type"), "target_slot_id": a.get("target_slot_id"),
            "candidate_id": a.get("candidate_id"),
            "expected_information_gain": a.get("expected_information_gain"),
            "estimated_cost": a.get("estimated_cost"), "selected": a.get("selected"),
            "executed": bool(plan.get("executed")),
            "execution_success": plan.get("execution_success"),
            "selected_reason": a.get("selected_reason"),
            "rejected_reason": a.get("rejected_reason")})
        action_nodes[a["action_id"]] = an
        if a.get("target_slot_id"):
            g.rel(an, f"latent_slot#{aid}#{a['target_slot_id']}", Relations.ACTION_TARGETS_SLOT)
        if a.get("candidate_id") and a["candidate_id"] in cand_node:
            rel = (Relations.ACTION_EXPANDS_CANDIDATE
                   if a.get("action_type") == "expand_candidate_to_dependent_slot"
                   else Relations.ACTION_TESTS_CANDIDATE)
            g.rel(an, cand_node[a["candidate_id"]], rel)
        for con in a.get("constraint_ids", []):
            g.rel(an, f"constraint#{aid}#{con}", Relations.ACTION_TESTS_CONSTRAINT)
        if a.get("selected"):
            fd = g.obj(f"frontier_decision#{aid}#{a['action_id']}", Objects.FRONTIER_DECISION,
                       {"selected_reason": a.get("selected_reason")})
            g.rel(an, fd, Relations.FRONTIER_ACTION_SELECTED_BECAUSE)
            fs = g.obj(f"frontier_score#{aid}#{a['action_id']}", Objects.FRONTIER_SCORE,
                       {"expected_information_gain": a.get("expected_information_gain"),
                        "estimated_cost": a.get("estimated_cost")})
            g.rel(an, fs, Relations.FRONTIER_ACTION_SELECTED_BECAUSE)
    # Controller: each executed tool call links to the frontier action that drove it.
    # The driving action may predate the final action list (the frontier regenerates
    # every step), so materialize a node for it from the call's task_action.
    for i, c in enumerate(calls or []):
        fa_id = c.get("frontier_action_id")
        if not fa_id:
            continue
        node = action_nodes.get(fa_id)
        if node is None:
            ta = c.get("task_action") or {}
            node = g.obj(f"frontier_action#{aid}#{fa_id}", Objects.FRONTIER_ACTION, {
                "action_type": ta.get("frontier_action_type"), "executed": True,
                "target_slot_id": ta.get("target_slot_id"),
                "selected_reason": ta.get("selected_reason"),
                "driven_by": "frontier_controller"})
            action_nodes[fa_id] = node
        g.rel(f"epistemic_action#{aid}#{i}", node, Relations.TOOL_CALL_FROM_FRONTIER_ACTION)


def _project_llm_frontier(g: "_GraphBuilder", aid: str, tf_node: str, a_node: str,
                          lf: dict[str, Any], calls: Optional[list] = None) -> None:
    """Project the LLM frontier proposer subgraph (Level 5c): state cards, proposals,
    validations, selections, and the tool-call link."""
    sel_by_id: dict[str, str] = {}      # proposal_id -> proposal node
    for si, st in enumerate(lf.get("steps", [])):
        card_hash = st.get("card_hash", "")
        sc = g.obj(f"research_state_card#{aid}#{si}", Objects.RESEARCH_STATE_CARD, {
            "card_hash": card_hash, "mode": st.get("mode"),
            "prompt_version": st.get("prompt_version"), "prompt_hash": st.get("prompt_hash"),
            "model": st.get("model"), "cache_hit": st.get("cache_hit"),
            "fallback_reason": st.get("fallback_reason", "")})
        pr = g.obj(f"llm_frontier_prompt#{aid}#{si}", Objects.LLM_FRONTIER_PROMPT,
                   {"prompt_version": st.get("prompt_version"), "prompt_hash": st.get("prompt_hash")})
        g.rel(sc, pr, Relations.PROPOSAL_BASED_ON_STATE_CARD)
        for p in st.get("proposals", []):
            pid = p.get("proposal_id")
            pn = g.obj(f"llm_frontier_proposal#{aid}#{si}#{pid}", Objects.LLM_FRONTIER_PROPOSAL, {
                "action_type": p.get("action_type"), "target_slot_id": p.get("target_slot_id"),
                "candidate_id": p.get("candidate_id"), "status": p.get("status"),
                "score": p.get("score"), "proposed_query": p.get("proposed_query"),
                "avoids_generic_query": p.get("avoids_generic_query")})
            g.rel(pn, sc, Relations.PROPOSAL_BASED_ON_STATE_CARD)
            if p.get("target_slot_id"):
                g.rel(pn, f"latent_slot#{aid}#{p['target_slot_id']}", Relations.PROPOSAL_TARGETS_SLOT)
            for cid in p.get("constraint_ids", []):
                g.rel(pn, f"constraint#{aid}#{cid}", Relations.PROPOSAL_TESTS_CONSTRAINT)
            if p.get("candidate_id"):
                g.rel(pn, f"slot_candidate#{aid}#{p['candidate_id']}", Relations.PROPOSAL_USES_CANDIDATE)
            if p.get("status") == "rejected":
                v = g.obj(f"llm_frontier_validation#{aid}#{si}#{pid}", Objects.LLM_FRONTIER_VALIDATION,
                          {"rejection_reason": p.get("rejection_reason")})
                g.rel(pn, v, Relations.PROPOSAL_REJECTED_BECAUSE)
            sel_by_id[str(pid)] = pn
        selp = st.get("selected", {})
        if selp.get("proposal_id"):
            integ = st.get("integrity", {}) or {}
            pc = st.get("progress_components", {}) or {}
            # The selection node exposes the proposal->action->evidence audit fields
            # (req 8): proposal vs executed slot/constraints, integrity, normalized tool,
            # repair trigger reason, and progress components — all replay-reconstructable.
            sn = g.obj(f"llm_frontier_selection#{aid}#{si}", Objects.LLM_FRONTIER_SELECTION, {
                "proposal_id": selp.get("proposal_id"), "score": selp.get("score"),
                "proposal_slot_id": st.get("proposal_slot_id"),
                "executed_slot_id": st.get("executed_slot_id"),
                "proposal_constraint_ids": st.get("proposal_constraint_ids", []),
                "executed_constraint_ids": st.get("executed_constraint_ids", []),
                "integrity_passed": st.get("integrity_passed"),
                "repair_trigger_reason": st.get("repair_trigger_reason", ""),
                "normalized_tool": st.get("normalized_tool", ""),
                "normalized_tool_family": st.get("normalized_tool_family", ""),
                "progress_components": pc,
                "evidence_linked_to_selected_slot": integ.get("evidence_linked_to_selected_slot"),
                "evidence_linked_to_selected_constraints":
                    integ.get("evidence_linked_to_selected_constraints")})
            pn = sel_by_id.get(str(selp.get("proposal_id")))
            if pn:
                g.rel(pn, sn, Relations.PROPOSAL_SELECTED_FOR_ACTION)
                # evidence actually attached to the selected slot -> auditable link.
                exslot = st.get("executed_slot_id")
                if integ.get("evidence_linked_to_selected_slot") and exslot:
                    g.rel(pn, f"latent_slot#{aid}#{exslot}",
                          Relations.EVIDENCE_LINKED_TO_LLM_PROPOSAL)
                if (exslot and pc.get("selected_slot_candidate_count", 0)
                        and st.get("execution_success")):
                    g.rel(pn, f"latent_slot#{aid}#{exslot}",
                          Relations.CANDIDATE_PROMOTED_FROM_LLM_FRONTIER)
    # each tool call driven by an LLM proposal links to its proposal node.
    for i, c in enumerate(calls or []):
        fa = c.get("frontier_action_id") or ""
        if fa.startswith("lfp_"):
            pid = fa[len("lfp_"):]
            for si in range(len(lf.get("steps", []))):
                node = f"llm_frontier_proposal#{aid}#{si}#{pid}"
                if g.has(node):
                    g.rel(f"epistemic_action#{aid}#{i}", node,
                          Relations.TOOL_CALL_FROM_LLM_FRONTIER_PROPOSAL)
                    break


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
                    "id_mapping": tf.get("id_mapping", {}),
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
                # known_context_term constants (NOT answers) — projected once per frame.
                kct_nodes: dict[str, str] = {}
                for term in tf.get("known_context_terms", [])[:12]:
                    kt = g.obj(f"known_context_term#{aid}#{term}", Objects.KNOWN_CONTEXT_TERM,
                               {"term": str(term)[:80]})
                    g.rel(kt, tf_node, Relations.KNOWN_CONTEXT_NOT_ANSWER)
                    kct_nodes[str(term)] = kt
                for s in (tf.get("target_answer_slots", []) + tf.get("latent_slots", [])):
                    status = s.get("slot_status", "unbound_variable")
                    # a slot is a VARIABLE until evidence binds it; record its status.
                    s_node = g.obj(f"latent_slot#{aid}#{s['slot_id']}", Objects.SLOT_VARIABLE, {
                        "slot_name": s.get("slot_name"), "slot_role": s.get("slot_role"),
                        "raw_slot_id": s.get("raw_slot_id", ""),
                        "slot_status": status, "bound_value": s.get("bound_value", ""),
                        "is_target_answer_slot": s.get("is_target_answer_slot"),
                        "is_intermediate_slot": s.get("is_intermediate_slot")})
                    g.rel(s_node, tf_node, Relations.SLOT_IN_FRAME)
                    # descriptor object: the raw text describing the unknown.
                    desc = s.get("descriptor_text") or s.get("slot_name")
                    if desc:
                        d_node = g.obj(f"slot_descriptor#{aid}#{s['slot_id']}",
                                       Objects.SLOT_DESCRIPTOR, {"descriptor_text": str(desc)[:120]})
                        g.rel(s_node, d_node, Relations.SLOT_HAS_DESCRIPTOR)
                    # binding status object (unbound until evidence proposes a candidate).
                    bs_node = g.obj(f"binding_status#{aid}#{s['slot_id']}", Objects.BINDING_STATUS,
                                    {"slot_status": status})
                    if status == "unbound_variable" and s.get("is_target_answer_slot"):
                        g.rel(s_node, bs_node, Relations.TARGET_SLOT_UNBOUND_UNTIL_EVIDENCE)
                    # context terms this descriptor references.
                    for ref in s.get("known_context_refs", [])[:6]:
                        kt = kct_nodes.get(str(ref))
                        if kt:
                            g.rel(s_node, kt, Relations.SLOT_DEPENDS_ON_CONTEXT)
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
                        # a candidate_binding is the evidence-proposed value for a slot
                        # VARIABLE (this is when a slot stops being unbound).
                        cb_node = g.obj(f"candidate_binding#{aid}#{h['hypothesis_id']}#{sid}",
                                        Objects.CANDIDATE_BINDING,
                                        {"slot_id": sid, "candidate_text": str(ctext)[:80],
                                         "from_evidence": True})
                        g.rel(f"latent_slot#{aid}#{sid}", cb_node, Relations.SLOT_BOUND_BY_CANDIDATE)
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

                # --- Level 5 candidate-slate / frontier subgraph ---
                cf = d.get("candidate_frontier") or {}
                if cf and not cf.get("skipped"):
                    _project_frontier(g, aid, tf_node, cf, d.get("calls", []))

                # --- Level 5c LLM frontier proposer subgraph ---
                lf = d.get("llm_frontier") or {}
                if lf.get("enabled"):
                    _project_llm_frontier(g, aid, tf_node, a_node, lf, d.get("calls", []))

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
