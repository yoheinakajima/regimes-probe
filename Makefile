# regimes-probe — no-key developer targets.
# Everything here runs WITHOUT API keys or network, except targets explicitly
# labelled "(live)". The "live" validation targets still make NO provider calls;
# they only validate configuration.

PY ?= python
RUN_ID ?= demo

.PHONY: help
help:
	@echo "Targets (all no-key unless noted):"
	@echo "  make test               - run the full test suite"
	@echo "  make synthetic-full     - run the complete no-key pipeline -> results/$(RUN_ID)"
	@echo "  make real-shaped-smoke  - run the real-data-shaped smoke test"
	@echo "  make ablations          - run Level1/Level2/reward/online ablations"
	@echo "  make validate-live      - validate live config (no provider calls)"
	@echo "  make preflight-live     - preflight the first live run (no provider calls)"
	@echo "  make estimate-cost      - estimate live call/cost footprint"
	@echo "  make inspect-memory     - inspect results/$(RUN_ID)/memory_snapshot.json"
	@echo "  make hash-artifacts     - hash results/$(RUN_ID) artifacts"
	@echo "  make claims             - generate conservative claim candidates"
	@echo "  make docs-check         - lint docs presence + no-overclaim"
	@echo "  make check              - test + synthetic-full + smoke + docs-check"

.PHONY: test
test:
	$(PY) -m pytest -q

.PHONY: synthetic-full
synthetic-full:
	$(PY) scripts/run_synthetic_full.py --run-id $(RUN_ID)

.PHONY: real-shaped-smoke
real-shaped-smoke:
	$(PY) -m pytest -q tests/test_real_data_shape.py

.PHONY: ablations
ablations:
	$(PY) scripts/run_ablations.py

.PHONY: validate-live
validate-live:   ## (live config only — NO provider calls)
	$(PY) scripts/validate_live_readiness.py

.PHONY: preflight-live
preflight-live:  ## (live config only — NO provider calls)
	$(PY) scripts/preflight_first_live_run.py

.PHONY: estimate-cost
estimate-cost:
	$(PY) scripts/estimate_live_cost.py

.PHONY: inspect-memory
inspect-memory:
	$(PY) scripts/inspect_memory_snapshot.py results/$(RUN_ID)/memory_snapshot.json

.PHONY: hash-artifacts
hash-artifacts:
	$(PY) scripts/hash_artifacts.py results/$(RUN_ID)

.PHONY: claims
claims:
	$(PY) scripts/generate_claims.py results/$(RUN_ID)/report.json

.PHONY: docs-check
docs-check:
	$(PY) scripts/docs_check.py

.PHONY: check
check: test docs-check
	$(PY) scripts/run_synthetic_full.py --run-id _check >/dev/null && echo "synthetic-full OK"
	$(PY) -m pytest -q tests/test_real_data_shape.py
	@echo "all no-key checks passed"
