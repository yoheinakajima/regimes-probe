"""Call/cost estimation for a live run (no network, no prices hard-coded).

Estimates how many answerer / grader / tool calls a run would make, so the cost
of a live run is known *before* spending anything. Dollar costs are reported as
``"unknown"`` unless per-call prices are explicitly supplied via config — vendor
prices are never baked in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CostEstimate:
    n_optimize: int
    n_confirm: int
    budgets: list[int]
    passes: int
    experience_budget: int
    # call counts
    answerer_calls: int
    grader_calls: int
    judge_calls: int
    worst_case_tool_calls: int
    worst_case_search_calls: int
    worst_case_page_fetch_calls: int
    per_condition: dict[str, Any]
    # money (optional)
    prices: dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: Any = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_optimize": self.n_optimize,
            "n_confirm": self.n_confirm,
            "budgets": self.budgets,
            "passes": self.passes,
            "experience_budget": self.experience_budget,
            "answerer_calls": self.answerer_calls,
            "grader_calls": self.grader_calls,
            "judge_calls": self.judge_calls,
            "worst_case_tool_calls": self.worst_case_tool_calls,
            "worst_case_search_calls": self.worst_case_search_calls,
            "worst_case_page_fetch_calls": self.worst_case_page_fetch_calls,
            "per_condition": self.per_condition,
            "prices": self.prices,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    def render(self) -> str:
        lines = [
            f"  optimize/confirm items : {self.n_optimize} / {self.n_confirm}",
            f"  budgets                : {self.budgets}   passes: {self.passes}",
            f"  answerer calls         : {self.answerer_calls}",
            f"  grader calls (exact)   : {self.grader_calls}",
            f"  judge calls (LLM)      : {self.judge_calls}",
            f"  worst-case tool calls  : {self.worst_case_tool_calls}",
            f"    ├─ search (≤)        : {self.worst_case_search_calls}",
            f"    └─ page_fetch (≤)    : {self.worst_case_page_fetch_calls}",
            f"  estimated cost (USD)   : {self.estimated_cost_usd}",
        ]
        return "\n".join(lines)


def estimate_calls(
    *,
    n_optimize: int,
    n_confirm: int,
    budgets: list[int],
    passes: int = 4,
    experience_budget: int = 5,
    judge: str = "exact",                 # "exact" | "llm"
    search_conditions: int = 3,           # no_memory_search, policy_memory, random_memory
    prices: Optional[dict[str, Any]] = None,
) -> CostEstimate:
    """Estimate worst-case call counts for the standard 4-condition pipeline.

    Attempts = experience (n_optimize * passes) + confirm conditions. Confirm
    runs closed_book once (budget 0) plus ``search_conditions`` per budget.
    Worst-case tool calls assume every attempt uses its full budget.
    """
    sum_budgets = sum(budgets)
    exp_attempts = n_optimize * passes
    confirm_search_attempts = n_confirm * search_conditions * len(budgets)
    closed_book_attempts = n_confirm                      # budget 0, no tools
    total_attempts = exp_attempts + confirm_search_attempts + closed_book_attempts

    # One answerer call per attempt (the extractor/answer step).
    answerer_calls = total_attempts
    # Exact-match grading is free of model calls; an LLM judge is one per graded
    # attempt (every attempt is graded for reward).
    grader_calls = total_attempts
    judge_calls = total_attempts if judge == "llm" else 0

    # Worst-case tool calls: experience at experience_budget, confirm search
    # conditions at their budgets, closed_book = 0.
    worst_exp_tools = n_optimize * passes * experience_budget
    worst_confirm_tools = n_confirm * search_conditions * sum_budgets
    worst_case_tool_calls = worst_exp_tools + worst_confirm_tools
    # Search vs page_fetch is policy-dependent; each is bounded above by the
    # total (they share the same budget), so report the bound for each.
    worst_case_search_calls = worst_case_tool_calls
    worst_case_page_fetch_calls = worst_case_tool_calls

    per_condition = {
        "experience": {"attempts": exp_attempts, "max_tool_calls": worst_exp_tools},
        "closed_book": {"attempts": closed_book_attempts, "max_tool_calls": 0},
        "search_conditions": {"attempts": confirm_search_attempts,
                              "max_tool_calls": worst_confirm_tools,
                              "names": ["no_memory_search", "policy_memory", "random_memory"]},
    }

    prices = prices or {}
    cost: Any = "unknown"
    if prices:
        # Only compute if EVERY referenced price is provided; otherwise stay
        # "unknown" rather than guess.
        try:
            cost = (
                answerer_calls * float(prices["answerer_per_call"])
                + judge_calls * float(prices.get("judge_per_call", 0.0))
                + worst_case_tool_calls * float(prices["tool_per_call"])
            )
            cost = round(cost, 4)
        except (KeyError, TypeError, ValueError):
            cost = "unknown"

    return CostEstimate(
        n_optimize=n_optimize, n_confirm=n_confirm, budgets=list(budgets), passes=passes,
        experience_budget=experience_budget, answerer_calls=answerer_calls,
        grader_calls=grader_calls, judge_calls=judge_calls,
        worst_case_tool_calls=worst_case_tool_calls,
        worst_case_search_calls=worst_case_search_calls,
        worst_case_page_fetch_calls=worst_case_page_fetch_calls,
        per_condition=per_condition, prices=prices, estimated_cost_usd=cost,
    )
