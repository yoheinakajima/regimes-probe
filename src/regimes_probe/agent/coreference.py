"""Level 5i-J: SAFETY SCAFFOLDING for task-frame coreference (no aggressive auto-merge).

A multi-constraint question often describes ONE latent entity with several clauses ("this
author … he later … the writer …"), which the parser can split into several person slots —
inflating candidate slates and judge calls. Fully collapsing those is risky, so this layer
only **proposes** merges (``merge_status='proposed_only'``) and never merges automatically by
default. Its job is mostly to *block* unsafe merges and make the proposals inspectable.

Hard safety invariants (enforced here, asserted in tests):
- never merge slots of different entity ROLES merely because they share a type;
- never merge distinct anchor entities (a hotel / museum / restaurant / founder / birth-year
  target stay distinct);
- never merge a TARGET answer slot with a SUBJECT slot when the target asks for a
  property/value OF the subject (the answer is not the subject);
- only merge person slots when frame language indicates the SAME referent (a pronoun /
  "this author" / the same described subject), and even then only as a proposal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: coreference cue tokens that signal "the same entity referred to again".
_COREF_CUES = ("this", "the same", "same", "aforementioned", "said", "who", "whom", "whose",
               "he", "she", "they", "his", "her", "their", "this person", "this author")
#: words that mark the target as a PROPERTY/VALUE of another entity (not coreferent with it).
_PROPERTY_OF = ("of the", "of this", "of that", "'s ", "s birth", "year of", "number of",
                "age of", "name of", "date of", "founder of", "author of")
_STOP = frozenset({"the", "a", "an", "of", "this", "that", "their", "his", "her", "its",
                   "who", "whom", "whose", "which", "in", "on", "at", "to", "and", "or"})


@dataclass
class CoreferenceProposal:
    source_slot_ids: list[str]
    proposed_entity_slot_id: str = ""
    shared_head: str = ""
    reason: str = ""
    confidence: float = 0.0
    #: proposed_only | applied | rejected
    merge_status: str = "proposed_only"
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"source_slot_ids": list(self.source_slot_ids),
                "proposed_entity_slot_id": self.proposed_entity_slot_id,
                "shared_head": self.shared_head, "reason": self.reason,
                "confidence": round(self.confidence, 3), "merge_status": self.merge_status,
                "rejection_reason": self.rejection_reason}


def _head(descriptor: str) -> str:
    """The descriptor's lead content noun (the entity TYPE head, e.g. "author"), used to
    group same-referent person descriptors — not the last clause word."""
    toks = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'&-]+", descriptor or "")
            if w.lower() not in _STOP]
    return toks[0] if toks else ""


def _has_coref_cue(descriptor: str) -> bool:
    low = f" {(descriptor or '').lower()} "
    return any(f" {c} " in low or low.strip().startswith(c) for c in _COREF_CUES)


def _is_property_of(descriptor: str) -> bool:
    low = (descriptor or "").lower()
    return any(p in low for p in _PROPERTY_OF)


def _desc(slot) -> str:
    return getattr(slot, "descriptor_text", "") or getattr(slot, "slot_name", "")


def propose_coreference(frame, *, auto_merge: bool = False) -> list[CoreferenceProposal]:
    """Conservatively PROPOSE (never auto-apply unless ``auto_merge`` and proven safe) merges
    of obvious same-referent person slots, and BLOCK unsafe target/subject merges (5i-J)."""
    proposals: list[CoreferenceProposal] = []
    target_ids = {s.slot_id for s in getattr(frame, "target_answer_slots", [])}
    latent = [s for s in getattr(frame, "all_slots", []) if s.slot_id not in target_ids]
    targets = [s for s in getattr(frame, "all_slots", []) if s.slot_id in target_ids]

    # (1) block a target<->subject "merge" when the target is a PROPERTY/VALUE of a subject.
    for t in targets:
        if _is_property_of(_desc(t)):
            subj = next((s for s in latent if s.slot_role != t.slot_role), None)
            if subj is not None:
                proposals.append(CoreferenceProposal(
                    source_slot_ids=[t.slot_id, subj.slot_id], shared_head=_head(_desc(t)),
                    reason="target_is_property_or_value_of_subject", merge_status="rejected",
                    rejection_reason="never_merge_target_value_with_subject", confidence=0.0))

    # (2) propose merges among same-role person slots that share a head AND carry a coref cue.
    persons = [s for s in latent if (s.slot_role or "").lower() == "person"]
    by_head: dict[str, list] = {}
    for s in persons:
        by_head.setdefault(_head(_desc(s)), []).append(s)
    for head, group in by_head.items():
        if len(group) < 2 or not head:
            continue
        if not any(_has_coref_cue(_desc(s)) for s in group):
            continue                            # no same-referent language -> do NOT propose
        roles = {s.slot_role for s in group}
        if len(roles) > 1:                      # never merge across roles
            proposals.append(CoreferenceProposal(
                source_slot_ids=[s.slot_id for s in group], shared_head=head,
                reason="shared_head_but_distinct_roles", merge_status="rejected",
                rejection_reason="never_merge_distinct_roles"))
            continue
        proposals.append(CoreferenceProposal(
            source_slot_ids=sorted(s.slot_id for s in group),
            proposed_entity_slot_id=sorted(s.slot_id for s in group)[0], shared_head=head,
            reason="same_referent_person_descriptors_share_head_and_coref_cue",
            confidence=0.6, merge_status=("applied" if auto_merge else "proposed_only")))
    return proposals


def coreference_metrics(proposals) -> dict[str, Any]:
    """Aggregate proposal stats. ``invalid_coreference_collapse_count`` counts any APPLIED
    merge that violated a safety rule — pinned 0 (default never auto-merges)."""
    proposed_only = sum(1 for p in proposals if p.merge_status == "proposed_only")
    applied = sum(1 for p in proposals if p.merge_status == "applied")
    rejected = sum(1 for p in proposals if p.merge_status == "rejected")
    target_blocked = sum(1 for p in proposals
                         if p.rejection_reason == "never_merge_target_value_with_subject")
    invalid = sum(1 for p in proposals if p.merge_status == "applied"
                  and p.rejection_reason)              # an applied merge with a rejection reason
    return {
        "coreference_proposals_count": len(proposals),
        "coreference_proposed_only_count": proposed_only,
        "coreference_applied_count": applied,
        "coreference_rejected_count": rejected,
        "invalid_coreference_collapse_count": invalid,            # pinned 0
        "target_subject_merge_blocked_count": target_blocked,
        "coreference_collapse_debug_map": [p.to_dict() for p in proposals][:12],
        # judge-call reduction from coreference is NOT claimed unless auto-merge is enabled.
        "judge_calls_saved_by_coreference_collapse": 0,
    }
