"""Policy layer: signatures, embeddings, memory, bandit, and the policy seams.

This is the pure-Python, deterministic core that the agent loop drives and the
ActiveGraph pack records. Nothing here performs network I/O.
"""

from __future__ import annotations

from regimes_probe.policy.contextual_bandit import (
    ArmStats,
    BanditParams,
    ContextualBandit,
    ScoredArm,
)
from regimes_probe.policy.consolidation import consolidate
from regimes_probe.policy.embeddings import HashEmbedder, cosine
from regimes_probe.policy.memory import (
    FAMILIES,
    FrozenMemoryError,
    PolicyMemory,
    PolicyMemorySnapshot,
    TraceRecord,
)
from regimes_probe.policy.policy_fragment import (
    PolicyFragment,
    assert_no_answer_leakage,
)
from regimes_probe.policy.query_policy import QUERY_ARMS, QueryPlan, QueryPolicy
from regimes_probe.policy.router import Router, RouterConfig, RoutingPlan
from regimes_probe.policy.signatures import (
    EPISTEMIC_FEATURES,
    QuerySignature,
    SignatureExtractor,
)
from regimes_probe.policy.stopping_policy import (
    STOP_ARMS,
    StopConfig,
    StopDecision,
    StoppingPolicy,
)
from regimes_probe.policy.verification_policy import (
    VerificationConfig,
    VerificationState,
    verify,
)

__all__ = [
    "ArmStats", "BanditParams", "ContextualBandit", "ScoredArm",
    "consolidate", "HashEmbedder", "cosine",
    "FAMILIES", "FrozenMemoryError", "PolicyMemory", "PolicyMemorySnapshot", "TraceRecord",
    "PolicyFragment", "assert_no_answer_leakage",
    "QUERY_ARMS", "QueryPlan", "QueryPolicy",
    "Router", "RouterConfig", "RoutingPlan",
    "EPISTEMIC_FEATURES", "QuerySignature", "SignatureExtractor",
    "STOP_ARMS", "StopConfig", "StopDecision", "StoppingPolicy",
    "VerificationConfig", "VerificationState", "verify",
]
