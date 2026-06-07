"""Evaluation: split, grading, reward, metrics, significance, reporting, harness."""

from __future__ import annotations

from regimes_probe.eval.grader import GradeResult, grade, normalize_answer
from regimes_probe.eval.harness import (
    ConditionResult,
    experience_phase,
    run_condition,
)
from regimes_probe.eval.metrics import AttemptOutcome, compute_metrics
from regimes_probe.eval.report import ConditionRun, write_full_report
from regimes_probe.eval.reward import RewardResult, RewardWeights, compute_rewards
from regimes_probe.eval.significance import (
    BootstrapCI,
    McNemarResult,
    bootstrap_correct_per_tool_call,
    mcnemar,
    paired_diff_ci,
)
from regimes_probe.eval.split import Split, build_split, partition

__all__ = [
    "GradeResult", "grade", "normalize_answer",
    "ConditionResult", "experience_phase", "run_condition",
    "AttemptOutcome", "compute_metrics",
    "ConditionRun", "write_full_report",
    "RewardResult", "RewardWeights", "compute_rewards",
    "BootstrapCI", "McNemarResult", "bootstrap_correct_per_tool_call", "mcnemar", "paired_diff_ci",
    "Split", "build_split", "partition",
]
