"""Regimes-style improvement layer: detect failure regimes, propose bounded
policy updates, gate through OPTIMIZE then CONFIRM, promote auditably."""

from __future__ import annotations

from regimes_probe.regimes.action_space import (
    Mutation,
    PolicyConfigBundle,
    safe_mutations,
)
from regimes_probe.regimes.detectors import REGIMES, detect_regimes, dominant_regimes, label_outcome
from regimes_probe.regimes.gates import (
    GateResult,
    PromotionDecision,
    confirm_gate,
    optimize_gate,
)
from regimes_probe.regimes.hypothesize import (
    PolicyUpdateProposal,
    apply_update,
    propose_update,
)
from regimes_probe.regimes.runner import (
    RegimesLoopResult,
    build_agent,
    evaluate_bundle,
    run_regimes_loop,
)

__all__ = [
    "Mutation", "PolicyConfigBundle", "safe_mutations",
    "REGIMES", "detect_regimes", "dominant_regimes", "label_outcome",
    "GateResult", "PromotionDecision", "confirm_gate", "optimize_gate",
    "PolicyUpdateProposal", "apply_update", "propose_update",
    "RegimesLoopResult", "build_agent", "evaluate_bundle", "run_regimes_loop",
]
