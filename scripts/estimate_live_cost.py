#!/usr/bin/env python
"""Estimate the call/cost footprint of a live run (no network, no provider calls).

    python scripts/estimate_live_cost.py --optimize 10 --confirm 20 --budgets 1,3
    python scripts/estimate_live_cost.py --search-provider-mode openai-hosted --web-search-model gpt-5.5

Shows the answerer model, the OpenAI web_search model/context, the enabled tools
(bandit arms), worst-case calls per provider/tool, and WARNS if gpt-5.5 +
openai_web_search is selected. Dollar cost is "unknown" unless `pricing:` is set
in config (vendor prices are never hard-coded).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_argparser, load_config

from regimes_probe.live.runner import ALL_CONDITIONS, estimate_live
from regimes_probe.live.settings import resolve_live_settings


def main() -> int:
    ap = base_argparser("Estimate live-run call/cost footprint.")
    ap.add_argument("--optimize", type=int, default=10)
    ap.add_argument("--confirm", type=int, default=20)
    ap.add_argument("--budgets", default="1,3")
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    ap.add_argument("--search-provider-mode", default=None,
                    choices=["cheap", "diverse", "openai-hosted"])
    ap.add_argument("--tools", default=None)
    ap.add_argument("--answer-model", default=None)
    ap.add_argument("--web-search-model", default=None)
    ap.add_argument("--web-search-context-size", default=None)
    ap.add_argument("--disable-openai-web-search", action="store_true")
    ap.add_argument("--enable-agentic-tool-discovery", action="store_true")
    ap.add_argument("--enable-scrape-tools", action="store_true")
    ap.add_argument("--enable-browserish-tools", action="store_true")
    ap.add_argument("--allow-stateful-or-paid-tools", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    live = cfg.get("live", {})
    mem = cfg.get("memory", {})
    budgets = [int(b) for b in str(args.budgets).replace(",", " ").split()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    settings = resolve_live_settings(
        mode=args.search_provider_mode or live.get("search_provider_mode", "cheap"),
        cli_tools=([t.strip() for t in args.tools.split(",") if t.strip()] if args.tools else None),
        answer_model=args.answer_model or live.get("answer_model", "gpt-5.4-mini"),
        web_search_model=args.web_search_model or live.get("web_search_model", "gpt-5.4-mini"),
        web_search_context_size=(args.web_search_context_size
                                 or live.get("web_search_context_size", "low")),
        disable_openai_web_search=args.disable_openai_web_search,
        enable_agentic_tool_discovery=args.enable_agentic_tool_discovery,
        enable_scrape_tools=args.enable_scrape_tools,
        enable_browserish_tools=args.enable_browserish_tools,
        allow_stateful_or_paid_tools=args.allow_stateful_or_paid_tools,
    )
    est = estimate_live(
        conditions, budgets, n_opt=args.optimize, n_con=args.confirm,
        passes=args.passes or mem.get("experience_passes", 4),
        exp_budget=mem.get("experience_budget", 5),
        judge=cfg.get("grading", {}).get("judge", "exact"), settings=settings.to_dict())

    print("Live-run call/cost estimate (no providers called):\n")
    print(f"  provider_mode          : {est['provider_mode']}")
    print(f"  answerer model         : {est['answer_model']}")
    print(f"  web_search model/ctx   : {est['web_search_model']} / {est['web_search_context_size']}"
          f"  (enabled={est['openai_web_search_enabled']})")
    print(f"  enabled tools (arms)   : {est['enabled_tools']}")
    print(f"  provider classes       : {est.get('provider_classes')}")
    print(f"  flags                  : agentic_discovery={est.get('agentic_tool_discovery_enabled')} "
          f"scrape={est.get('scrape_tools_enabled')} browserish={est.get('browserish_tools_enabled')} "
          f"stateful_or_paid={est.get('stateful_or_paid_tools_allowed')}")
    print(f"  stateful arms          : "
          f"{[n for n, m in est.get('tools_meta', {}).items() if m.get('stateful')] or 'none'}")
    print(f"  optimize/confirm       : {est['n_optimize']} / {est['n_confirm']}   "
          f"budgets={est['budgets']}  passes={est['passes']}")
    print(f"  answerer calls         : {est['answerer_calls']}")
    print(f"  judge calls (LLM)      : {est['judge_calls']}")
    print(f"  worst-case tool calls  : {est['worst_case_tool_calls']}")
    print(f"  max calls by tool      : {est['max_calls_by_tool']}")
    print(f"  estimated cost (USD)   : {est['estimated_cost_usd']}")
    for n in settings.notes:
        print(f"  note: {n}")
    for w in est.get("warnings", []):
        print(f"  ⚠️  {w}")
    if est["estimated_cost_usd"] == "unknown":
        print("\n  (set `pricing:` in config for a dollar estimate; vendor prices not hard-coded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
