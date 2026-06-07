"""Contextual-bandit updates and UCB / epsilon-greedy scoring."""

from __future__ import annotations

from regimes_probe.policy.contextual_bandit import BanditParams, ContextualBandit


def test_update_moves_mean_and_count():
    # isolate local stats (no global/neighbour blend) so w == raw pull count
    b = ContextualBandit("tool", BanditParams(recency_decay=1.0, blend_global=0.0,
                                              blend_neighbor=0.0))
    for r in (1.0, 1.0, 1.0):
        b.update_reward("c", "news_search", r)
    s = b.score_tool("c", "news_search", total_w=3, explore=False)
    assert abs(s.mean - 1.0) < 1e-9
    assert s.w == 3.0


def test_ucb_exploration_bonus_decreases_with_pulls():
    b = ContextualBandit("tool", BanditParams(strategy="ucb", exploration_coeff=1.0))
    b.update_reward("c", "a", 0.5)
    low_pull = b.score_tool("c", "a", total_w=10, explore=True).explore_bonus
    for _ in range(20):
        b.update_reward("c", "a", 0.5)
    high_pull = b.score_tool("c", "a", total_w=30, explore=True).explore_bonus
    assert high_pull < low_pull


def test_exploit_picks_highest_mean_arm():
    b = ContextualBandit("tool", BanditParams(strategy="ucb"))
    for _ in range(5):
        b.update_reward("c", "good", 1.0)
        b.update_reward("c", "bad", -0.5)
    ranked = b.choose_tool_plan("c", ["good", "bad"], k=2, explore=False)
    assert ranked[0].arm == "good"


def test_scoring_is_deterministic():
    b = ContextualBandit("tool", BanditParams(strategy="ucb"))
    b.update_reward("c", "x", 0.7)
    s1 = b.score_tool("c", "x", total_w=5, explore=True, salt="z")
    s2 = b.score_tool("c", "x", total_w=5, explore=True, salt="z")
    assert s1.score == s2.score


def test_epsilon_greedy_explores_deterministically():
    p = BanditParams(strategy="epsilon_greedy", exploration_coeff=1.0)  # always explore
    b = ContextualBandit("tool", p)
    for _ in range(3):
        b.update_reward("c", "good", 1.0)
    r1 = [s.arm for s in b.choose_tool_plan("c", ["good", "bad", "mid"], k=3, explore=True, salt="s")]
    r2 = [s.arm for s in b.choose_tool_plan("c", ["good", "bad", "mid"], k=3, explore=True, salt="s")]
    assert r1 == r2  # same salt -> same shuffle


def test_round_trip_serialization():
    b = ContextualBandit("tool", BanditParams())
    b.update_reward("c", "x", 0.4)
    b2 = ContextualBandit.from_dict(b.to_dict())
    assert b2.score_tool("c", "x", total_w=2, explore=False).mean == \
        b.score_tool("c", "x", total_w=2, explore=False).mean
