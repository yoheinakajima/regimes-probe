"""The default router is LLM-free and deterministic given a memory snapshot."""

from __future__ import annotations

from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.router import Router
from regimes_probe.policy.signatures import SignatureExtractor

TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]


def _seed_memory():
    mem = PolicyMemory(BanditParams(strategy="ucb"))
    sig = SignatureExtractor().compute("Who is currently the CEO of Acme Robotics?")
    # give news_search a strong track record in this cluster
    for _ in range(5):
        mem.bandits["tool"].update_reward(sig.cluster_key, "news_search", 1.4)
        mem.bandits["tool"].update_reward(sig.cluster_key, "brave_search", -0.2)
    return mem, sig


def test_router_is_deterministic_exploit():
    mem, sig = _seed_memory()
    router = Router()
    p1 = router.route(sig, mem, TOOLS, budget=3, explore=False)
    p2 = router.route(sig, mem, TOOLS, budget=3, explore=False)
    assert p1.sequence == p2.sequence
    assert p1.ranked == p2.ranked


def test_router_prefers_learned_tool():
    mem, sig = _seed_memory()
    seq = Router().route(sig, mem, TOOLS, budget=1, explore=False).sequence
    assert seq[0] == "news_search"


def test_router_frozen_snapshot_reproduces_plan():
    mem, sig = _seed_memory()
    snap = mem.snapshot()
    frozen = PolicyMemory.from_snapshot(snap, frozen=True)
    a = Router().route(sig, mem, TOOLS, budget=3, explore=False).sequence
    b = Router().route(sig, frozen, TOOLS, budget=3, explore=False).sequence
    assert a == b


def test_router_respects_budget():
    mem, sig = _seed_memory()
    for B in (1, 2, 3, 4):
        seq = Router().route(sig, mem, TOOLS, budget=B, explore=False).sequence
        assert len(seq) == min(B, len(TOOLS))
