# ROUTING_POLICY.md

## Purpose (Level 1)

The routing policy decides **where to look**: it ranks available tools and produces a
bounded tool sequence (a `routing_plan`) for a `question_attempt`, under the current
budget cap. It learns from trace-derived priors via the contextual bandit
(`docs/CONTEXTUAL_BANDIT.md`) and the `policy_memory_snapshot`
(`docs/POLICY_MEMORY.md`).

> **The default router MUST NOT call an LLM.** Routing decisions come from
> nearest-neighbor trace retrieval + contextual bandit scores + cost/latency penalties.
> An LLM router may exist only as a separately-labeled baseline arm for comparison.

---

## Inputs

| Input | Description |
|---|---|
| `question_text` | Raw question (normalized for hashing/embedding) |
| `question_embedding` | Dense embedding vector |
| `features` | Generic epistemic `query_signature` features + lexical/result features |
| `memory_snapshot` | Active (OPTIMIZE) or frozen (CONFIRM) `policy_memory_snapshot` |
| `available_tools` | Tool availability set (search tools, fetch tools) |
| `budget_cap` | Remaining budget cap from `budgets=[1,3,5,10]` |

---

## Output: `routing_plan`

| Field | Description |
|---|---|
| `ranked_tools` | Tools sorted by bandit score (desc), with per-tool numeric breakdown |
| `tool_sequence` | The chosen ordered sequence to execute, length `<= budget_cap` |
| `numeric_explanation` | Per-tool mean, exploration bonus, penalties, final score |
| `mode` | `EPSILON_GREEDY`/`UCB`/`THOMPSON` (OPTIMIZE) or `EXPLOIT` (CONFIRM) |

The router emits exactly one `routing_plan` per attempt. The actual `tool_call` objects
and their `evidence_observation`s are produced during execution and fed back via
`update_reward` (OPTIMIZE only).

---

## Scoring

Routing scores reuse `score_tool` from `docs/CONTEXTUAL_BANDIT.md`. On top of the bandit
mean/exploration score, the router subtracts **cost and latency penalties** so cheaper,
faster tools win ties when expected reward is comparable. These penalties are the same
`cost_penalty` and `latency_penalty` terms from the reward decomposition, applied here as
**expected** penalties from the tool's stored stats.

```
route_score(tool) =   bandit_score(tool)                 # from score_tool
                    - EXP_COST    * expected_cost(tool)
                    - EXP_LATENCY * expected_latency(tool)
```

`first_tool_hit_reward` makes the bandit favor tools that historically surfaced usable
evidence on the **first** call for this regime/signature, which is what drives the head
of `tool_sequence`.

---

## Exploration regime gating

- **OPTIMIZE:** `optimize_allows_exploration=true` → epsilon-greedy / UCB / Thompson.
  Memory updates after execution.
- **CONFIRM:** frozen snapshot, deterministic exploitation (`EXPLOIT`, epsilon=0),
  `confirm_updates_memory=false`.

Budget caps `[1,3,5,10]` are **binding**: `len(tool_sequence) <= budget_cap`.

---

## PSEUDOCODE: full routing decision

```
function route(question_text, question_embedding, features,
               memory_snapshot, available_tools, budget_cap, mode, now_tick):

    ctx = Context(
        question_embedding = question_embedding,
        features           = features,                 # generic epistemic signature
        regime_label       = memory_snapshot.assign_regime(question_embedding, features),
        budget_cap         = budget_cap,
        tool_availability  = available_tools
    )

    # --- 1. Nearest-neighbor trace retrieval (no LLM) ---
    # score_tool() internally performs cosine-similarity-weighted NN reward blending.
    scored = []
    for tool in available_tools:
        s = score_tool(tool, ctx, memory_snapshot, mode, now_tick)   # bandit score

        # --- 2. Apply expected cost/latency penalties ---
        exp_cost    = memory_snapshot.expected_cost(ctx, tool)
        exp_latency = memory_snapshot.expected_latency(ctx, tool)
        route_score = s.score - EXP_COST * exp_cost - EXP_LATENCY * exp_latency

        scored.append(RankedTool(
            tool = tool,
            mean = s.mean,
            exploration_bonus = s.score - s.mean,        # UCB/Thompson/epsilon component
            expected_cost = exp_cost,
            expected_latency = exp_latency,
            confidence = s.confidence,
            n = s.n,
            route_score = route_score
        ))

    # --- 3. Deterministic ranking: score desc, then tool name asc ---
    ranked = sort(scored, key = (-r.route_score, r.tool))

    # --- 4. Exploration only during OPTIMIZE ---
    if mode == EPSILON_GREEDY and OPTIMIZE_ALLOWS_EXPLORATION:
        rng = seeded_rng(hash(signature_key(ctx)) ^ now_tick)
        if rng.uniform() < EPSILON and len(ranked) > 1:
            idx = 1 + (rng.randint() mod (len(ranked) - 1))
            swap(ranked, 0, idx)                          # promote an explore tool to head

    # --- 5. Build bounded tool sequence under the budget cap ---
    tool_sequence = []
    for r in ranked:
        if len(tool_sequence) >= budget_cap:             # binding cap
            break
        # Skip a near-duplicate tool unless it adds expected complementary coverage
        if adds_marginal_value(r, tool_sequence, memory_snapshot, ctx):
            tool_sequence.append(r.tool)

    # Guarantee at least one tool when budget >= 1
    if len(tool_sequence) == 0 and budget_cap >= 1 and len(ranked) > 0:
        tool_sequence = [ranked[0].tool]

    # --- 6. Numeric explanation ---
    explanation = [
        { "tool": r.tool,
          "mean": r.mean,
          "exploration_bonus": r.exploration_bonus,
          "expected_cost": r.expected_cost,
          "expected_latency": r.expected_latency,
          "route_score": r.route_score,
          "n": r.n,
          "confidence": r.confidence }
        for r in ranked
    ]

    return RoutingPlan(
        ranked_tools        = ranked,
        tool_sequence       = tool_sequence,
        numeric_explanation = explanation,
        mode                = mode,
        budget_cap          = budget_cap
    )
```

---

## Failure regimes addressed

| Regime | How routing relates |
|---|---|
| `route_miss` | Wrong tool ranked first; corrected by `first_tool_hit_reward` priors |
| `over_search` | Too many tools queued; bounded by `budget_cap` + `extra_call_penalty` |
| `under_search` | Sequence too short for `multihop-likely`; coverage via `adds_marginal_value` |

`query_miss`, verification, and stopping regimes are handled at Level 2
(`docs/QUERY_POLICY.md`, `docs/VERIFICATION_AND_STOPPING.md`).
