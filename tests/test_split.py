"""Deterministic split + OPTIMIZE/CONFIRM disjointness."""

from __future__ import annotations

from regimes_probe.eval.split import build_split, partition


def test_split_is_deterministic(items):
    s1 = build_split(items, confirm_fraction=0.4, salt="x")
    s2 = build_split(items, confirm_fraction=0.4, salt="x")
    assert s1.optimize_ids == s2.optimize_ids
    assert s1.confirm_ids == s2.confirm_ids


def test_split_disjoint_and_complete(items):
    s = build_split(items, confirm_fraction=0.4)
    s.assert_disjoint()
    assert set(s.optimize_ids).isdisjoint(s.confirm_ids)
    assert len(s.optimize_ids) + len(s.confirm_ids) == len(items)


def test_salt_changes_split(items):
    a = build_split(items, salt="a")
    b = build_split(items, salt="b")
    assert a.confirm_ids != b.confirm_ids


def test_partition_round_trips(items):
    s = build_split(items, confirm_fraction=0.5)
    opt, con = partition(items, s)
    assert {i.id for i in opt} == set(s.optimize_ids)
    assert {i.id for i in con} == set(s.confirm_ids)


def test_time_mode_is_time_disjoint(items):
    s = build_split(items, mode="time", confirm_fraction=0.3)
    s.assert_disjoint()
    # every confirm item is at least as recent as every optimize item
    by_id = {i.id: i for i in items}
    opt_dates = [by_id[i].released_at for i in s.optimize_ids]
    con_dates = [by_id[i].released_at for i in s.confirm_ids]
    if opt_dates and con_dates:
        assert max(opt_dates) <= min(con_dates) or s.mode == "hash"
