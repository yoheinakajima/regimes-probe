#!/usr/bin/env python
"""Validate readiness for a LIVE benchmark run WITHOUT calling any provider.

    python scripts/validate_live_readiness.py            # warns on gaps
    python scripts/validate_live_readiness.py --strict   # gaps become failures

Checks (all offline):
  - benchmark dataset path / HF availability config
  - tool provider config presence (config/tools.yaml or tools.example.yaml)
  - env var NAMES expected (values only required with --strict)
  - output directories writable
  - model / tool names present and recognized
  - budget caps well-formed
  - OPTIMIZE/CONFIRM split disjointness (built, not just configured)
  - memory freezes before CONFIRM (confirm_uses_frozen_snapshot / confirm_updates_memory)

Never prints secret values. Never opens a network connection.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml

from _common import DEFAULT_CONFIG, build_synthetic, load_config
from regimes_probe.eval.split import build_split
from regimes_probe.tools import ADAPTER_REGISTRY

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_ENV_VARS = ["OPENAI_API_KEY", "BRAVE_SEARCH_API_KEY", "TAVILY_API_KEY",
             "EXA_API_KEY", "SERPER_API_KEY"]


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    def render(self) -> str:
        out = []
        for status, name, detail in self.rows:
            mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[status]
            out.append(f"  [{mark}] {status:<4} {name}" + (f" — {detail}" if detail else ""))
        return "\n".join(out)

    @property
    def n_fail(self) -> int:
        return sum(1 for s, *_ in self.rows if s == FAIL)

    @property
    def n_warn(self) -> int:
        return sum(1 for s, *_ in self.rows if s == WARN)


def _load_tools_config() -> tuple[dict, str]:
    for name in ("tools.yaml", "tools.example.yaml"):
        p = ROOT / "config" / name
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")), name
    return {}, "(none)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate live benchmark readiness (no calls).")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    ap.add_argument("--strict", action="store_true", help="treat missing keys/deps as failures")
    args = ap.parse_args()
    strict = args.strict
    gap = FAIL if strict else WARN
    r = Report()

    cfg = load_config(args.config)
    tools_cfg, tools_file = _load_tools_config()

    # 1. dataset config / availability
    ds = cfg.get("dataset", {})
    r.add(PASS, "dataset.primary", ds.get("primary", "?"))
    r.add(PASS, "dataset.secondary", ds.get("secondary", "?"))
    try:
        import datasets  # noqa: F401
        r.add(PASS, "huggingface 'datasets' importable", "LiveBrowseComp HF load possible")
    except Exception:
        r.add(gap, "huggingface 'datasets' importable",
              "pip install 'regimes-probe[datasets]' OR provide local_jsonl")
    bc_csv = ds.get("browsecomp_csv")
    if bc_csv and Path(bc_csv).exists():
        r.add(PASS, "BrowseComp CSV present", bc_csv)
    else:
        r.add(WARN, "BrowseComp CSV present", "set dataset.browsecomp_csv to a downloaded CSV")

    # 2. tool provider config presence
    r.add(PASS if tools_cfg else WARN, "tools config file", tools_file)
    adapters_cfg = tools_cfg.get("adapters", {}) if tools_cfg else {}
    enabled = [n for n, a in adapters_cfg.items() if a.get("enabled")]
    baseline = cfg.get("live", {}).get("search_baseline", "openai_web_search")
    if baseline in ADAPTER_REGISTRY:
        r.add(PASS, "search_baseline is a known adapter", baseline)
    else:
        r.add(FAIL, "search_baseline is a known adapter", f"unknown: {baseline}")
    indep = [n for n in enabled if n not in (baseline, "page_fetch")]
    if len(indep) >= 2:
        r.add(PASS, ">=2 independent search adapters enabled", ", ".join(indep))
    else:
        r.add(WARN, ">=2 independent search adapters enabled",
              "enable Brave/Tavily/Exa/Serper for tool diversity")

    # 3. env var names (presence only; never values)
    for var in _ENV_VARS:
        present = bool(os.environ.get(var))
        if var == "OPENAI_API_KEY":
            r.add(PASS if present else gap, f"env {var}",
                  "set" if present else "needed for default answerer + hosted baseline")
        else:
            r.add(PASS if present else WARN, f"env {var}", "set" if present else "optional")

    # 4. output directories
    out = Path(args.results_root)
    try:
        out.mkdir(parents=True, exist_ok=True)
        r.add(PASS, "results dir writable", str(out))
    except Exception as e:
        r.add(FAIL, "results dir writable", str(e))

    # 5. model / tool names present
    live = cfg.get("live", {})
    for key in ("answer_model", "cheaper_answer_model", "embedder", "search_baseline"):
        r.add(PASS if live.get(key) else WARN, f"live.{key}", str(live.get(key)))
    bad_tools = [t for t in live.get("tools_enabled", []) if t not in ADAPTER_REGISTRY]
    r.add(PASS if not bad_tools else FAIL, "live.tools_enabled are known adapters",
          "ok" if not bad_tools else f"unknown: {bad_tools}")

    # 6. budget caps
    budgets = cfg.get("budgets", [])
    ok_b = (isinstance(budgets, list) and budgets and all(isinstance(b, int) and b > 0 for b in budgets)
            and budgets == sorted(budgets))
    r.add(PASS if ok_b else FAIL, "budget caps well-formed", str(budgets))

    # 7. split disjointness (actually build one)
    try:
        items, *_ = build_synthetic(cfg)
        sp = cfg.get("split", {})
        split = build_split(items, confirm_fraction=sp.get("confirm_fraction", 0.4),
                            salt=sp.get("salt", "regimes-probe-v0"), mode=sp.get("mode", "hash"))
        split.assert_disjoint()
        r.add(PASS, "OPTIMIZE/CONFIRM split disjoint",
              f"OPT={len(split.optimize_ids)} CONFIRM={len(split.confirm_ids)} mode={split.mode}")
    except Exception as e:
        r.add(FAIL, "OPTIMIZE/CONFIRM split disjoint", str(e))

    # 8. memory freezes before CONFIRM
    mem = cfg.get("memory", {})
    frozen_ok = mem.get("confirm_uses_frozen_snapshot") and not mem.get("confirm_updates_memory")
    r.add(PASS if frozen_ok else FAIL, "memory frozen before CONFIRM",
          f"frozen={mem.get('confirm_uses_frozen_snapshot')} updates={mem.get('confirm_updates_memory')}")

    print("Live-readiness validation (no providers called):\n")
    print(r.render())
    print(f"\n{r.n_fail} failures, {r.n_warn} warnings."
          + ("  (strict)" if strict else "  (run with --strict to gate on gaps)"))
    print("\nNOTE: this validates configuration only. It does NOT prove a live run "
          "will succeed,\nand it makes NO benchmark claim.")
    return 1 if r.n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
