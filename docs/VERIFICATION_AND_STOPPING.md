# VERIFICATION_AND_STOPPING.md

## Purpose (Level 2: verification & stopping)

After each tool call and its `evidence_observation`s, the policy must decide **when to
verify** and **when to stop**. These are bandit decisions
(`docs/CONTEXTUAL_BANDIT.md`) over the **stop** and **verification** arm families, scored
from trace-derived `stopping_rewards` and `verification_rewards`
(`docs/POLICY_MEMORY.md`). Budget caps `[1,3,5,10]` remain binding.

---

## Stop policy arms

| Arm | Meaning |
|---|---|
| `stop_now` | Commit the current candidate as `final_answer`; end the attempt. |
| `search_more` | Issue another search (new `query_plan`) within budget. |
| `fetch_page` | Fetch a specific page/source to deepen evidence rather than search again. |
| `corroborate` | Seek an additional independent source supporting the current candidate. |
| `contradiction_check` | Actively look for sources that would contradict the candidate. |

`stop_now` and `search_more` are the core continue/stop choices; `fetch_page`,
`corroborate`, and `contradiction_check` are deepening/verification continuations.

---

## Verification policy checks

Before committing, evaluate the verification state of the current candidate:

| Check | Question it answers |
|---|---|
| `answer_candidate_found` | Is there a `candidate_answer` at all? |
| `supporting_evidence_found` | Does at least one `evidence_observation` support it? |
| `source_authority_acceptable` | Are supporting sources authoritative enough? |
| `freshness_acceptable` | Are supporting sources recent enough for the signature? |
| `contradictions_unresolved` | Are there unresolved contradicting sources? |

These produce a `verification_state` consumed by the stop decision.

---

## Stopping reward

The stopping reward balances **accuracy/support** against **cost/latency**, using the
same terms as the global reward decomposition in `docs/CONTEXTUAL_BANDIT.md`:

```
stop_reward =   correctness_reward
              + evidence_quality_reward
              + source_authority_reward
              + freshness_reward
              - cost_penalty
              - latency_penalty
              - stale_penalty
              - contradiction_penalty
              - extra_call_penalty
```

- Stopping **early** with weak support forfeits `correctness_reward` /
  `evidence_quality_reward` (a false stop).
- Stopping **late** (continuing past sufficient support) accrues `cost_penalty`,
  `latency_penalty`, and `extra_call_penalty` (over-search).

---

## Error definitions and metrics

Each attempt records whether it false-stopped or over-searched.

- A **false stop** = `stop_now` chosen when the committed answer was wrong *and*
  available budget remained that could plausibly have corrected it (verification state
  was insufficient: missing support, unresolved contradiction, or stale evidence).
- **Over-search** = continuing (any non-`stop_now` arm) after the candidate already had
  sufficient, fresh, authoritative, contradiction-free support — incurring extra calls.

```
false_stop_rate = (# attempts with stop_too_early / answer wrong despite budget)
                  / (# attempts)

over_search_rate = (# attempts with over_search: continued past sufficient support)
                  / (# attempts)
```

These map to the failure regimes `stop_too_early` (and `verification_miss` when a needed
check was skipped) and `over_search`. `stop_too_late` is the temporal form of
over-search.

| Failure regime | Trigger |
|---|---|
| `stop_too_early` | `stop_now` with insufficient verification state; counts toward `false_stop_rate` |
| `verification_miss` | Committed without running a needed verification check |
| `contradiction_unresolved` | Stopped with `contradictions_unresolved=true` |
| `over_search` / `stop_too_late` | Continued past sufficient support; counts toward `over_search_rate` |
| `stale_evidence` | Committed with `freshness_acceptable=false` on a freshness-sensitive signature |

---

## PSEUDOCODE: stop/verify decision

```
function verify_state(candidate, evidence, features, snapshot):
    return VerificationState(
        answer_candidate_found       = (candidate != NULL),
        supporting_evidence_found    = any(e supports candidate for e in evidence),
        source_authority_acceptable  = max_authority(evidence) >= AUTHORITY_MIN,
        freshness_acceptable         = (not features.freshness_sensitive)
                                       or max_freshness(evidence) >= FRESHNESS_MIN,
        contradictions_unresolved    = has_unresolved_contradiction(evidence, candidate)
    )

function support_confidence(vstate, candidate, evidence):
    # Aggregate verification state into a scalar support score in [0,1]
    s  = 0.0
    s += W_FOUND      if vstate.answer_candidate_found
    s += W_SUPPORT    if vstate.supporting_evidence_found
    s += W_AUTHORITY  if vstate.source_authority_acceptable
    s += W_FRESH      if vstate.freshness_acceptable
    s -= W_CONTRADICT if vstate.contradictions_unresolved
    return clamp(s, 0.0, 1.0)

function decide_stop(ctx, candidate, evidence, snapshot, mode, now_tick,
                     calls_used, budget_cap):

    vstate = verify_state(candidate, evidence, ctx.features, snapshot)
    conf   = support_confidence(vstate, candidate, evidence)
    thr    = snapshot.stop_confidence_threshold(ctx)     # from policy_fragment params

    # --- Hard budget gate: never exceed the binding cap ---
    if calls_used >= budget_cap:
        return StopDecision(arm = stop_now, reason = "budget_exhausted",
                            vstate = vstate, confidence = conf)

    # --- High-confidence fast path: avoid over-search ---
    if conf >= thr and not vstate.contradictions_unresolved:
        return StopDecision(arm = stop_now, reason = "sufficient_support",
                            vstate = vstate, confidence = conf)

    # --- Otherwise let the bandit choose among continue/verify arms ---
    arms = [stop_now, search_more]
    if can_fetch(ctx):                  arms.append(fetch_page)
    if vstate.supporting_evidence_found: arms.append(corroborate)
    if ctx.features.verification_heavy or vstate.contradictions_unresolved:
        arms.append(contradiction_check)

    plan = choose_tool_plan(ctx, snapshot, mode, now_tick, available_arms = arms)

    # --- Guard against false stops when support is weak and budget remains ---
    if plan.chosen_arm == stop_now and conf < thr and calls_used < budget_cap:
        # bandit may still stop, but flag the risk for error accounting
        record_risk(stop_too_early_candidate = true)

    return StopDecision(arm = plan.chosen_arm,
                        reason = "bandit_choice",
                        vstate = vstate, confidence = conf,
                        explanation = plan.explanation)
```

During **CONFIRM** the decision runs with `mode = EXPLOIT` on the frozen snapshot
(deterministic, `confirm_updates_memory=false`); during **OPTIMIZE** exploration is
allowed (`optimize_allows_exploration=true`) and `stopping_rewards` /
`verification_rewards` are updated after grading.

---

## Post-attempt accounting

After grading (`docs/GRADING_AND_REWARD.md`), label the attempt:

```
if committed and answer_wrong and budget_remained and conf < thr:
    regime_label += stop_too_early ; false_stop = true
if continued_after(conf >= thr and support_complete):
    regime_label += over_search   ; over_searched = true
if committed and vstate.contradictions_unresolved:
    regime_label += contradiction_unresolved
if committed and not vstate.freshness_acceptable and features.freshness_sensitive:
    regime_label += stale_evidence
```

These labels feed `false_stop_rate`, `over_search_rate`, and the consolidated
`stopping_rewards` / `verification_rewards`.
