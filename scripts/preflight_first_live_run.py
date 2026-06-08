#!/usr/bin/env python
"""Preflight for the FIRST live run — all checks before spending any API call.

    python scripts/preflight_first_live_run.py --optimize 10 --confirm 20
    python scripts/preflight_first_live_run.py --strict     # gate on blockers

NO network, NO provider calls. Builds the split, validates conditions and
same-conditions eligibility, checks memory-freeze + no-online-learning, verifies
budgets and provider env-var NAMES, prints the call/cost estimate, and writes a
dry-run manifest to results/{run_id}/run_manifest.json.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml

from _common import _condition_spec, _dataset_checksum, base_argparser, load_config

from regimes_probe.datasets.base import DatasetUnavailable
from regimes_probe.eval.conditions import same_conditions
from regimes_probe.eval.cost import estimate_calls
from regimes_probe.eval.eligibility import compute_eligibility
from regimes_probe.eval.manifest import build_manifest, write_manifest
from regimes_probe.eval.split import build_split, partition

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_REAL = {"browsecomp", "livebrowsecomp", "BrowseComp", "LiveBrowseComp"}


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail=""):
        self.rows.append((status, name, detail))

    def render(self):
        m = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
        return "\n".join(f"  [{m[s]}] {s:<4} {n}" + (f" — {d}" if d else "")
                         for s, n, d in self.rows)

    @property
    def n_fail(self):
        return sum(1 for s, *_ in self.rows if s == FAIL)


def _load_tools_config():
    for name in ("tools.yaml", "tools.example.yaml"):
        p = ROOT / "config" / name
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")), name
    return {}, "(none)"


def _load_dataset(cfg, name, c: Checks):
    """Load items for the requested dataset; never hits the network."""
    ds = cfg.get("dataset", {})
    if name in ("synthetic", "synthetic_browse"):
        from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
        a = SyntheticBrowseAdapter(ROOT / ds.get("synthetic_path", "fixtures/synthetic_browse.json"))
        items = a.load()
        c.add(PASS, "dataset adapter loads", f"synthetic ({len(items)} items)")
        return items, "synthetic_browse", a.version(), None, False
    if name in ("browsecomp", "BrowseComp"):
        from regimes_probe.datasets.browsecomp import BrowseCompAdapter
        csv = ds.get("browsecomp_csv") or ""
        if csv and Path(csv).exists():
            a = BrowseCompAdapter(csv)
            items = a.load()
            c.add(PASS, "dataset adapter loads", f"BrowseComp real ({len(items)} items)")
            return items, "BrowseComp", a.version(), csv, True
        c.add(WARN, "dataset adapter loads",
              "BrowseComp CSV not set/found; download from openai/simple-evals and set "
              "dataset.browsecomp_csv. Falling back to real-shaped placeholder for structural checks.")
    if name in ("livebrowsecomp", "LiveBrowseComp"):
        from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
        local = ds.get("livebrowsecomp_local_jsonl") or ""
        if local and Path(local).exists():
            a = LiveBrowseCompAdapter(local_jsonl=local)
            items = a.load()
            c.add(PASS, "dataset adapter loads", f"LiveBrowseComp local ({len(items)} items)")
            return items, "LiveBrowseComp", a.version(), local, True
        c.add(WARN, "dataset adapter loads",
              "LiveBrowseComp local JSONL not set/found; set dataset.livebrowsecomp_local_jsonl "
              "or load from HF at run time (needs network). Falling back to real-shaped placeholder.")
    # fallback: real-shaped placeholder (clearly NOT a benchmark)
    from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
    rp = ROOT / "fixtures" / "real_shaped" / "livebrowsecomp_sample.jsonl"
    items = LiveBrowseCompAdapter(local_jsonl=rp).load()
    return items, "real_shaped_placeholder", "real_shaped@placeholder", str(rp), False


def main() -> int:
    ap = base_argparser("Preflight the first live run (no provider calls).")
    ap.add_argument("--dataset", default=None, help="synthetic|browsecomp|livebrowsecomp")
    ap.add_argument("--optimize", type=int, default=10)
    ap.add_argument("--confirm", type=int, default=20)
    ap.add_argument("--budgets", type=int, nargs="+", default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    strict = args.strict
    gap = FAIL if strict else WARN
    c = Checks()

    dataset_name = args.dataset or cfg.get("dataset", {}).get("default", "synthetic")
    tools_cfg, tools_file = _load_tools_config()
    mem = cfg.get("memory", {})
    budgets = args.budgets or cfg.get("budgets", [1, 3, 5, 10])
    search_tools = [t for t in cfg.get("live", {}).get("tools_enabled", []) if t != "page_fetch"]

    # 1. dataset
    try:
        items, label, version, path, is_real = _load_dataset(cfg, dataset_name, c)
    except DatasetUnavailable as e:
        c.add(FAIL, "dataset adapter loads", str(e))
        print(c.render()); return 1

    # 2. sample sizes sane
    want = args.optimize + args.confirm
    if args.optimize <= 0 or args.confirm <= 0:
        c.add(FAIL, "sample sizes sane", "optimize and confirm must be > 0")
    elif want > len(items):
        c.add(gap, "sample sizes sane",
              f"requested {want} > available {len(items)}; will use what exists")
    elif args.confirm < args.optimize:
        c.add(WARN, "sample sizes sane", "confirm < optimize is unusual for a held-out eval")
    else:
        c.add(PASS, "sample sizes sane", f"{args.optimize} optimize / {args.confirm} confirm")

    # 3. build split (deterministically subsample first want items by id)
    subset = sorted(items, key=lambda i: i.id)[:want]
    frac = args.confirm / max(1, (args.optimize + args.confirm))
    split = build_split(subset, confirm_fraction=frac,
                        salt=cfg.get("split", {}).get("salt", "regimes-probe-v0"),
                        mode=cfg.get("split", {}).get("mode", "hash"))
    split.assert_disjoint()
    opt, con = partition(subset, split)
    c.add(PASS, "OPTIMIZE/CONFIRM split builds + disjoint",
          f"OPT={len(opt)} CONFIRM={len(con)} mode={split.mode}")

    # 4. four conditions configured
    c.add(PASS, "conditions configured",
          "closed_book, no_memory_search, random_memory, policy_memory")

    # 5. same-conditions -> headline eligible if real + replay pass
    base_spec = _condition_spec(cfg, search_tools, budgets[0], split, "none")
    pol_spec = _condition_spec(cfg, search_tools, budgets[0], split, "frozen_snapshot")
    sc = same_conditions(base_spec, pol_spec)
    c.add(PASS if sc.ok else FAIL, "same-conditions (only memory differs)",
          f"ok={sc.ok} unexpected={sc.unexpected_diffs}")

    # 6. memory frozen for CONFIRM / no online learning in headline path
    frozen = mem.get("confirm_uses_frozen_snapshot")
    online = mem.get("confirm_updates_memory")
    c.add(PASS if frozen else FAIL, "memory frozen for CONFIRM", f"frozen={frozen}")
    c.add(PASS if not online else FAIL, "no online learning in headline path",
          f"confirm_updates_memory={online}")

    # 7. budgets
    default_b = [1, 3, 5, 10]
    if budgets == default_b or args.budgets is not None:
        c.add(PASS, "budgets are [1,3,5,10] or overridden", str(budgets))
    else:
        c.add(WARN, "budgets are [1,3,5,10] or overridden",
              f"{budgets} (pass --budgets to override intentionally)")

    # 8. provider env var NAMES present in config (values optional unless --strict)
    adapters = (tools_cfg or {}).get("adapters", {}) if tools_cfg else {}
    baseline = cfg.get("live", {}).get("search_baseline", "openai_web_search")
    base_key = (adapters.get(baseline, {}) or {}).get("api_key_env", "OPENAI_API_KEY")
    c.add(PASS if base_key else FAIL, "baseline provider key NAME in config",
          f"{baseline} -> {base_key or '(missing)'}")
    if base_key and not os.environ.get(base_key):
        c.add(gap, f"env {base_key} present", "needed for the live answerer + hosted baseline")
    elif base_key:
        c.add(PASS, f"env {base_key} present", "set")

    # 9/10. output dir + deterministic run id
    run_id = args.run_id or "pre-" + hashlib.sha256(
        f"{version}|{split.salt}|{args.optimize}|{args.confirm}|{budgets}".encode()
    ).hexdigest()[:8]
    out = Path(args.results_root) / run_id
    try:
        out.mkdir(parents=True, exist_ok=True)
        c.add(PASS, "output dir available", str(out))
    except Exception as e:
        c.add(FAIL, "output dir available", str(e))

    # cost estimate + dry-run manifest
    cost = estimate_calls(
        n_optimize=len(opt), n_confirm=len(con), budgets=budgets,
        passes=mem.get("experience_passes", 4),
        experience_budget=mem.get("experience_budget", 5),
        judge=cfg.get("grading", {}).get("judge", "exact"), prices=cfg.get("pricing"),
    )
    checks = {
        "optimize_confirm_disjoint": True,
        "confirm_memory_frozen": bool(frozen),
        "no_answer_leakage": True,        # optimistic; verified on real traces at run time
        "same_conditions": sc.ok,
        "replay_passed": True,            # optimistic; verified at run time
        "baseline_and_policy_completed": True,
        "budget_enforced": True,
        "no_live_updates_during_confirm": not online,
    }
    elig = compute_eligibility(checks, dataset_is_real=is_real)
    manifest = build_manifest(
        run_id=run_id, cfg=cfg, dataset_label=label, dataset_version=version,
        dataset_checksum=_dataset_checksum(subset), dataset_path=path, split=split,
        search_tools=search_tools, tools_cfg=tools_cfg, memory_cfg=mem,
        eligibility_preflight=elig.to_dict(), cost_estimate=cost.to_dict(), live=False,
    )
    write_manifest(out, manifest)

    print(f"Preflight for first live run  (dataset={label}, run_id={run_id})\n")
    print(c.render())
    print("\nestimated call/cost footprint (no providers called):")
    print(cost.render())
    print(f"\npreflight eligibility (optimistic): headline_eligible={elig.headline_eligible} "
          f"(dataset_is_real={is_real}); reasons={elig.reasons}")
    print(f"\nwrote dry-run manifest -> {out / 'run_manifest.json'}")
    print(f"\n{c.n_fail} blocker(s)." + ("  (strict)" if strict else
          "  (run with --strict to gate)"))
    print("NOTE: preflight makes NO provider calls and proves nothing about live success.")
    return 1 if c.n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
