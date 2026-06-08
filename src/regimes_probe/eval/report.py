"""Report artifacts: report.json, summary.md, CSVs, replay_check.md.

Outputs the full ``results/{run_id}/`` set documented in ``docs/REPORTING.md``.
All claims in a summary must be grounded in these committed artifacts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from regimes_probe.eval.metrics import AttemptOutcome, compute_metrics
from regimes_probe.regimes.detectors import detect_regimes, dominant_regimes, label_outcome


@dataclass
class ConditionRun:
    """One (condition, budget) cell of a report."""

    condition: str
    budget: int
    outcomes: list[AttemptOutcome]

    @property
    def metrics(self) -> dict[str, Any]:
        return compute_metrics(self.outcomes)


def _write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _budget_curve_rows(runs: list[ConditionRun]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        m = run.metrics
        rows.append({
            "budget": run.budget,
            "condition": run.condition,
            "accuracy": round(m.get("accuracy", 0.0), 4),
            "correct_per_tool_call": round(m.get("correct_per_tool_call", 0.0), 4),
            "mean_tool_calls": round(m.get("mean_tool_calls", 0.0), 3),
            "first_tool_hit_rate": round(m.get("first_tool_hit_rate", 0.0), 4),
            "n": m.get("n", 0),
        })
    return rows


def _per_question_rows(runs: list[ConditionRun]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for o in run.outcomes:
            row = o.to_row()
            row["regime"] = label_outcome(o)
            rows.append(row)
    return rows


def _reward_rows(snapshot: dict[str, Any], family: str) -> list[dict[str, Any]]:
    rows = []
    bandit = (snapshot or {}).get("bandits", {}).get(family, {})
    for cluster, arms in bandit.get("ctx", {}).items():
        for arm, st in arms.items():
            rows.append({
                "signature_cluster": cluster,
                "arm": arm,
                "mean_reward": round(st.get("mean", 0.0), 4),
                "count": st.get("n", 0),
                "w": round(st.get("w", 0.0), 3),
            })
    rows.sort(key=lambda r: (r["signature_cluster"], -r["mean_reward"]))
    return rows


def write_full_report(
    run_id: str,
    *,
    runs: list[ConditionRun],
    snapshot: Optional[dict[str, Any]] = None,
    replay: Optional[dict[str, Any]] = None,
    promotions: Optional[list[dict[str, Any]]] = None,
    significance: Optional[dict[str, Any]] = None,
    meta: Optional[dict[str, Any]] = None,
    eligibility: Optional[dict[str, Any]] = None,
    same_conditions: Optional[dict[str, Any]] = None,
    condition_specs: Optional[dict[str, Any]] = None,
    results_root: str | Path = "results",
) -> Path:
    """Write every artifact and return the run directory."""
    run_dir = Path(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = meta or {}
    promotions = promotions or []
    snapshot = snapshot or {}

    # --- CSVs ---
    _write_csv(run_dir / "budget_curve.csv",
               ["budget", "condition", "accuracy", "correct_per_tool_call",
                "mean_tool_calls", "first_tool_hit_rate", "n"],
               _budget_curve_rows(runs))
    pq_rows = _per_question_rows(runs)
    if pq_rows:
        _write_csv(run_dir / "per_question.csv", list(pq_rows[0].keys()), pq_rows)
    _write_csv(run_dir / "tool_rewards.csv",
               ["signature_cluster", "arm", "mean_reward", "count", "w"],
               _reward_rows(snapshot, "tool"))
    _write_csv(run_dir / "query_rewards.csv",
               ["signature_cluster", "arm", "mean_reward", "count", "w"],
               _reward_rows(snapshot, "query"))
    sv_rows = _reward_rows(snapshot, "stop") + _reward_rows(snapshot, "verify")
    _write_csv(run_dir / "stop_verify_rewards.csv",
               ["signature_cluster", "arm", "mean_reward", "count", "w"], sv_rows)

    # --- JSON artifacts ---
    (run_dir / "memory_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    (run_dir / "policy_updates.json").write_text(json.dumps(promotions, indent=2), encoding="utf-8")

    elig = eligibility or {}
    conditions_present = elig.get("conditions_present") or sorted({r.condition for r in runs})
    report = {
        "run_id": run_id,
        "meta": meta,
        # Two distinct verdicts (see eval/eligibility.py):
        "structurally_valid": elig.get("structurally_valid",
                                       elig.get("mechanism_ok", False)),
        "headline_eligible_memory_claim": elig.get("headline_eligible_memory_claim",
                                                   elig.get("headline_eligible", False)),
        "headline_eligibility_reasons": elig.get("headline_eligibility_reasons",
                                                 elig.get("reasons", [])),
        "conditions_present": conditions_present,
        # back-compat alias (== headline_eligible_memory_claim):
        "headline_eligible": elig.get("headline_eligible_memory_claim",
                                      elig.get("headline_eligible", False)),
        "eligibility": elig,
        "same_conditions": same_conditions or {},
        "condition_specs": condition_specs or {},
        "conditions": [
            {"condition": r.condition, "budget": r.budget, "metrics": r.metrics}
            for r in runs
        ],
        "significance": significance or {},
        "promotions": promotions,
        "replay_check": replay or {},
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- replay_check.md ---
    (run_dir / "replay_check.md").write_text(_replay_md(replay or {}), encoding="utf-8")

    # --- summary.md ---
    (run_dir / "summary.md").write_text(
        _summary_md(run_id, runs, meta, promotions, significance or {}, replay or {},
                    eligibility or {}, same_conditions or {}),
        encoding="utf-8",
    )
    return run_dir


def _replay_md(replay: dict[str, Any]) -> str:
    if not replay:
        return "# Replay check\n\n_No replay performed._\n"
    ok = replay.get("projection_matches")
    return (
        "# Replay check\n\n"
        f"- events: {replay.get('n_events')}\n"
        f"- objects: {replay.get('n_objects')}\n"
        f"- relations: {replay.get('n_relations')}\n"
        f"- projection_matches: **{ok}**\n"
        f"- detail: {replay.get('detail')}\n\n"
        f"{'PASS — the graph is a deterministic projection of the log.' if ok else 'FAIL'}\n"
    )


def _summary_md(run_id, runs, meta, promotions, significance, replay,
                eligibility=None, same_conditions=None) -> str:
    eligibility = eligibility or {}
    same_conditions = same_conditions or {}
    lines: list[str] = []
    A = lines.append
    A(f"# regimes-probe run `{run_id}`\n")

    # Eligibility banner — the most important lines in the report.
    present = eligibility.get("conditions_present") or sorted({r.condition for r in runs})
    has_policy = "policy_memory" in present
    A("## Eligibility\n")
    A(f"- **structurally_valid: {eligibility.get('structurally_valid', eligibility.get('mechanism_ok'))}**")
    A(f"- **headline_eligible_memory_claim: "
      f"{eligibility.get('headline_eligible_memory_claim', eligibility.get('headline_eligible'))}**")
    A(f"- dataset_is_real: {eligibility.get('dataset_is_real')}")
    A(f"- conditions_present: {present}")
    reasons = eligibility.get("headline_eligibility_reasons") or eligibility.get("reasons") or []
    if reasons:
        A("- reasons NOT headline-eligible (memory claim):")
        for r in reasons:
            A(f"  - {r}")
    # The same-conditions statement only makes sense when BOTH compared
    # conditions ran. A plumbing run (no policy_memory) must not claim it.
    if has_policy and same_conditions and not same_conditions.get("not_applicable"):
        A(f"- same_conditions (no_memory_search vs policy_memory) ok: "
          f"{same_conditions.get('ok')} (unexpected diffs: {same_conditions.get('unexpected_diffs')})")
    elif not has_policy:
        A("- same_conditions: N/A — policy_memory was not run, so there is no memory "
          "comparison to validate.")
    A("")

    # Headline: best paired improvement on the main metric if present.
    headline = meta.get("headline") or _auto_headline(runs)
    A(f"## Headline result\n\n{headline}\n")

    A("## Models, providers, tools\n")
    A(f"- answer_model: `{meta.get('answer_model', 'n/a')}`")
    A(f"- search_baseline: `{meta.get('search_baseline', 'n/a')}`")
    A(f"- tools_enabled: {meta.get('tools_enabled', [])}")
    A(f"- embedder: `{meta.get('embedder', 'hash_embedder')}`\n")

    A("## Benchmark / split\n")
    A(f"- dataset: `{meta.get('dataset', 'n/a')}`  version: `{meta.get('dataset_version', 'n/a')}`")
    A(f"- split: {meta.get('split', {})}")
    A(f"- budget caps: {meta.get('budgets', [])}")
    A(f"- memory condition: `{meta.get('memory_condition', 'n/a')}`")
    A(f"- policy condition: `{meta.get('policy_condition', 'n/a')}`\n")

    A("## Metrics by condition and budget\n")
    A("| condition | budget | accuracy | correct_per_tool_call | mean_calls | first_tool_hit |")
    A("|---|---|---|---|---|---|")
    for r in runs:
        m = r.metrics
        A(f"| {r.condition} | {r.budget} | {m.get('accuracy',0):.3f} | "
          f"{m.get('correct_per_tool_call',0):.3f} | {m.get('mean_tool_calls',0):.2f} | "
          f"{m.get('first_tool_hit_rate',0):.3f} |")
    A("")

    A("## Efficiency & epistemic-error rates\n")
    A("| condition | budget | over_search | false_stop | stale_err | evidence_gain/call |")
    A("|---|---|---|---|---|---|")
    for r in runs:
        m = r.metrics
        A(f"| {r.condition} | {r.budget} | {m.get('over_search_rate',0):.3f} | "
          f"{m.get('false_stop_rate',0):.3f} | {m.get('stale_source_error_rate',0):.3f} | "
          f"{m.get('evidence_gain_per_call',0):.3f} |")
    A("")

    if significance:
        A("## Statistical tests\n")
        A("```json")
        A(json.dumps(significance, indent=2))
        A("```\n")

    A("## Failure regimes (dominant on evaluated set)\n")
    dom = dominant_regimes([o for r in runs for o in r.outcomes])
    A("\n".join(f"- {k}: {v}" for k, v in dom.most_common(8)) or "- none detected")
    A("")

    A("## Promotions\n")
    if promotions:
        for p in promotions:
            acc = p.get("accepted")
            A(f"- {'ACCEPTED' if acc else 'REJECTED'}: {p.get('update', {})}")
    else:
        A("- none")
    A("")

    A("## Replay\n")
    A(f"- projection_matches: **{replay.get('projection_matches')}** "
      f"({replay.get('n_events')} events)\n")

    A("## Limitations\n")
    for lim in meta.get("limitations", _default_limitations()):
        A(f"- {lim}")
    A("")

    A("## What is NOT claimed\n")
    for nc in meta.get("not_claimed", _default_not_claimed()):
        A(f"- {nc}")
    A("")
    return "\n".join(lines)


def _auto_headline(runs: list[ConditionRun]) -> str:
    by = {(r.condition, r.budget): r.metrics for r in runs}
    conds = {r.condition for r in runs}
    base_name = "no_memory_search" if "no_memory_search" in conds else "no_memory"
    if {base_name, "policy_memory"} <= conds:
        budgets = sorted({r.budget for r in runs if r.condition in (base_name, "policy_memory")})
        parts = []
        for b in budgets:
            base = by.get((base_name, b), {}).get("correct_per_tool_call", 0.0)
            pol = by.get(("policy_memory", b), {}).get("correct_per_tool_call", 0.0)
            parts.append(f"budget {b}: {base:.3f} → {pol:.3f}")
        return (f"Frozen policy memory vs {base_name} baseline, correct_per_tool_call — "
                + "; ".join(parts) + ".")
    return "See metrics table."


def _default_limitations() -> list[str]:
    return [
        "Results below are on the synthetic harness unless a real dataset is named above.",
        "The synthetic answerer models a competent extractor; it isolates retrieval/epistemic policy, not extraction.",
        "Live provider results are non-deterministic; only fixture-backed runs are byte-reproducible.",
    ]


def _default_not_claimed() -> list[str]:
    return [
        "No claim that base model weights changed (they do not).",
        "No claim that benchmark answers are stored in policy memory (they are not).",
        "No claim of improvement on any dataset not listed in this report.",
    ]
