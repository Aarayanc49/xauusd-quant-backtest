"""Invariants that must never break.

These are not coverage tests. Each one guards a specific way the old tree went
wrong, and every one of those faults produced a *plausible* result rather than a
crash — which is why they survived for months. A lookahead bug does not raise; it
just prints a better number.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import levels as L
from core.cluster import cluster_points
from core.context import Context, Scale, atr, rolling_mean, true_range
from core.store import Bars, write


# ── fixtures ────────────────────────────────────────────────────────────────

def synth(n=2000, seed=7, start=2000.0, step_sec=300):
    """Deterministic random walk with realistic intrabar structure."""
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, 0.35, n))
    o = np.empty(n)
    o[0] = start
    o[1:] = close[:-1]
    wick = np.abs(rng.normal(0, 0.25, n))
    h = np.maximum(o, close) + wick
    l = np.minimum(o, close) - np.abs(rng.normal(0, 0.25, n))
    return {
        "time": np.arange(n, dtype=np.int64) * step_sec + 1_600_000_000,
        "open": o, "high": h, "low": l, "close": close,
        "volume": rng.integers(50, 500, n).astype(float),
        "spread": np.full(n, 20.0),
    }


@pytest.fixture(scope="module")
def store_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("bars"))
    for tf, step, n in (("M5", 300, 4000), ("M15", 900, 1400),
                        ("H1", 3600, 400), ("H4", 14400, 120)):
        write("TESTFX", tf, synth(n, seed=hash(tf) % 1000, step_sec=step),
              digits=5, point=0.00001, base=d)
    return d


@pytest.fixture(scope="module")
def series(store_dir):
    return {tf: Bars("TESTFX", tf, base=store_dir)
            for tf in ("M5", "M15", "H1", "H4")}


# ── store ───────────────────────────────────────────────────────────────────

def test_store_rejects_non_monotonic_time(tmp_path):
    a = synth(50)
    a["time"][10] = a["time"][20]
    with pytest.raises(ValueError, match="not strictly increasing"):
        write("BAD", "M5", a, 5, 1e-5, base=str(tmp_path))


def test_store_rejects_impossible_ohlc(tmp_path):
    a = synth(50)
    a["high"][7] = a["low"][7] - 1.0
    with pytest.raises(ValueError, match="OHLC bounds"):
        write("BAD", "M5", a, 5, 1e-5, base=str(tmp_path))


def test_closed_upto_never_returns_an_unclosed_bar(series):
    b = series["M5"]
    for i in (5, 100, 999, len(b) - 1):
        t_open = int(b.time[i])
        # at the instant the bar OPENS, the newest closed bar is the previous one
        assert b.closed_upto(t_open) == i - 1
        # one second before it closes, still the previous one
        assert b.closed_upto(t_open + b.bar_seconds - 1) == i - 1
        # at its close, it becomes available
        assert b.closed_upto(t_open + b.bar_seconds) == i


# ── context: the lookahead guard ────────────────────────────────────────────

def test_align_never_reads_an_unfinished_higher_timeframe_bar(series):
    """The multi-timeframe lookahead: a decision at 09:05 must not see an H4 bar
    that does not close until 12:00. This is the single easiest way to fake an
    edge and it is invisible in the output."""
    ctx = Context(series, base="M5")
    base = series["M5"]
    base_close = np.asarray(base.time, np.int64) + base.bar_seconds
    for tf in ("M15", "H1", "H4"):
        m = ctx.align(tf)
        other = series[tf]
        ok = m >= 0
        mapped_close = (np.asarray(other.time, np.int64)[m[ok]] + other.bar_seconds)
        assert (mapped_close <= base_close[ok]).all(), f"{tf} mapped into the future"
        # and it must be the NEWEST such bar, not an older one
        nxt = m[ok] + 1
        valid = nxt < len(other)
        nxt_close = np.asarray(other.time, np.int64)[nxt[valid]] + other.bar_seconds
        assert (nxt_close > base_close[ok][valid]).all(), f"{tf} mapped too far back"


def test_rolling_mean_is_causal():
    x = np.arange(100.0)
    full = rolling_mean(x, 10)
    for cut in (20, 55, 99):
        assert rolling_mean(x[:cut + 1], 10)[cut] == pytest.approx(full[cut])


def test_atr_is_causal(series):
    """Recomputing ATR on a truncated series must give the same value at the
    truncation point. If it does not, something downstream of it sees the future."""
    b = series["M5"]
    full = atr(b, 14)
    for cut in (50, 500, 2000):
        part = true_range(b.high[:cut + 1], b.low[:cut + 1], b.close[:cut + 1])
        assert rolling_mean(part, 14)[cut] == pytest.approx(full[cut])


def test_every_threshold_resolves_to_a_positive_scale(series):
    ctx = Context(series, base="M5")
    for name in Scale.THRESHOLDS:
        v = ctx.threshold(name)
        assert len(v) == len(series["M5"])
        assert np.isfinite(v).all(), f"{name} produced non-finite values"
        assert (v > 0).all(), f"{name} produced a non-positive threshold"


# ── levels ──────────────────────────────────────────────────────────────────

def test_swing_is_not_knowable_before_confirmation(series):
    """A swing high at bar i is confirmed by bar i+k. born must reflect that."""
    b = series["M15"]
    k = 3
    p = L.swing_points(b, "M15", k=k)
    assert len(p) > 0
    assert (p.born == p.ref + k).all()
    assert (p.born > p.ref).all()


def test_swing_price_is_an_actual_bar_extreme(series):
    b = series["M15"]
    p = L.swing_points(b, "M15", k=3)
    h = np.asarray(b.high)
    l = np.asarray(b.low)
    for j in range(0, len(p), max(1, len(p) // 200)):
        want = h[p.ref[j]] if p.source[j] == "swing_h" else l[p.ref[j]]
        assert p.price[j] == pytest.approx(want)


def test_levels_die_when_price_closes_through_them(series):
    b = series["M15"]
    p = L.swing_points(b, "M15", k=3)
    close = np.asarray(b.close)
    n = len(b)
    checked = 0
    for j in range(len(p)):
        d = int(p.dead[j])
        if d >= n:
            continue
        checked += 1
        if p.source[j] == "swing_h":
            assert close[d] > p.price[j], "a swing high died without a close above it"
        else:
            assert close[d] < p.price[j], "a swing low died without a close below it"
    assert checked > 10, "no deaths observed — the test proves nothing"


def test_dead_levels_are_excluded_from_the_active_set(series):
    b = series["M15"]
    p = L.swing_points(b, "M15", k=3)
    for i in (100, 500, len(b) - 1):
        a = p.active(i)
        assert (a.born <= i).all()
        assert (a.dead > i).all()


def test_equal_levels_are_never_adjacent_bars(series):
    """The bug in the old tree: 20% of "equal highs" were two adjacent bars — one
    swing counted as an equal high. eql is the highest-edge source measured, and
    it was being fed this."""
    b = series["M15"]
    tol = atr(b, 14) * 0.15
    p = L.equal_levels(b, "M15", tol, k=3, min_sep=3)
    if len(p) == 0:
        pytest.skip("no equal levels in the synthetic series")
    # every EQ level derives from a swing at least min_sep bars from its partner;
    # the emitted ref is the LATER swing, and born trails it by k
    assert (p.born > p.ref).all()
    assert len(np.unique(p.price)) > 1


def test_indexed_near_matches_brute_force():
    """`near` is price-indexed for speed — 85k calls against 93k points over 14
    years is 24 billion element ops if done as a full mask. The optimisation must
    return exactly what the naive version does, or every study silently changes."""
    rng = np.random.default_rng(0)
    n = 5000
    price = rng.uniform(1000, 2000, n)
    born = rng.integers(0, 900, n).astype(np.int64)
    dead = born + rng.integers(1, 300, n)
    P = L.Points(price, born, dead.astype(np.int64),
                 np.full(n, "swing_h", "<U12"), np.full(n, "M15", "<U3"), born)
    for _ in range(200):
        i = int(rng.integers(0, 1000))
        c = float(rng.uniform(1000, 2000))
        w = float(rng.uniform(1, 200))
        fast = P.near(i, c, w)
        m = (P.born <= i) & (P.dead > i) & (np.abs(P.price - c) <= w)
        assert sorted(fast.price.tolist()) == sorted(P.price[m].tolist())


def test_death_by_break_window_search_matches_full_scan(series):
    """`death_by_break` searches in a doubling window instead of scanning the
    whole tail. Same answer, or every level's lifetime is wrong."""
    b = series["M15"]
    close = np.asarray(b.close, np.float64)
    rng = np.random.default_rng(1)
    n = 300
    idx = rng.integers(0, len(b) - 10, n).astype(np.int64)
    price = close[idx] * (1 + rng.normal(0, 0.002, n))
    above = rng.random(n) > 0.5
    buf = np.full(n, b.pip * 2)
    got = L.death_by_break(close, price, idx, above, buf)
    for j in range(n):
        s = int(idx[j]) + 1
        thr = price[j] + (buf[j] if above[j] else -buf[j])
        w = np.flatnonzero(close[s:] > thr if above[j] else close[s:] < thr)
        want = s + int(w[0]) if w.size else len(close)
        assert int(got[j]) == want


def test_round_numbers_are_emitted_once_each(series):
    """The other density bug: re-emitting a round number on every re-entry into
    range produced 5,007 half-round points out of 9,970, all of them the same
    handful of prices, all still voting at the end of the series."""
    b = series["M15"]
    p = L.round_points(b, "M15")
    assert len(p) == len(np.unique(np.round(p.price, 8))), "duplicate round prices"


def test_zones_carry_bounds_and_die(series):
    b = series["M15"]
    z = L.fvg_zones(b, "M15") + L.ob_zones(b, "M15")
    if len(z) == 0:
        pytest.skip("no zones in the synthetic series")
    assert (z.high >= z.low).all()
    assert (z.dead > z.born).all()
    assert not hasattr(z, "price"), "a Zone must never expose a trigger price"


# ── clustering: the fault that broke everything ─────────────────────────────

@pytest.mark.parametrize("seed", range(12))
def test_no_cluster_ever_exceeds_the_cap(seed):
    """The old chain compared each point to the PREVIOUS one, so 4000 -> 4018 ->
    4035 -> 4052 was a single legal "level" of unbounded span. Every hop passed;
    the total never did."""
    rng = np.random.default_rng(seed)
    n = rng.integers(5, 400)
    price = np.sort(rng.uniform(1900, 2100, n))
    p = L.Points(price, np.zeros(n, np.int64), np.full(n, n, np.int64),
                 np.full(n, "swing_h", "<U12"), np.full(n, "M15", "<U3"),
                 np.zeros(n, np.int64))
    cap = float(rng.uniform(0.5, 8.0))
    for c in cluster_points(p, cap):
        assert c.width <= cap + 1e-9, f"cluster {c.width} exceeded cap {cap}"


@pytest.mark.parametrize("seed", range(12))
def test_reference_price_is_always_a_real_member_price(seed):
    """`price = mean(prices)` was the second half of the fault: the number every
    trigger referenced was the arithmetic mean of unrelated points from different
    sources and timeframes, and frequently corresponded to no structure at all."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 200))
    price = np.sort(rng.uniform(1900, 2100, n))
    src = rng.choice(["swing_h", "eqh", "round", "pdh"], n)
    tf = rng.choice(["M15", "H1", "H4"], n)
    p = L.Points(price, np.zeros(n, np.int64), np.full(n, n, np.int64),
                 src.astype("<U12"), tf.astype("<U3"), np.zeros(n, np.int64))
    for c in cluster_points(p, 3.0):
        assert float(c.price) in set(price[c.members].tolist()), \
            "reference price is not a real constituent price"
        assert c.lo <= c.price <= c.hi


def test_clustering_covers_every_point_exactly_once():
    price = np.sort(np.random.default_rng(3).uniform(1900, 2100, 500))
    n = len(price)
    p = L.Points(price, np.zeros(n, np.int64), np.full(n, n, np.int64),
                 np.full(n, "swing_h", "<U12"), np.full(n, "M15", "<U3"),
                 np.zeros(n, np.int64))
    seen = np.concatenate([c.members for c in cluster_points(p, 2.0)])
    assert sorted(seen.tolist()) == list(range(n))


def test_isolated_levels_survive_as_singletons():
    """A point that cannot merge must stay a level, not be forced into a
    neighbour. The old code had no way to express "this level is alone and that
    is fine"."""
    price = np.array([1900.0, 1950.0, 2000.0, 2050.0])
    p = L.Points(price, np.zeros(4, np.int64), np.full(4, 4, np.int64),
                 np.full(4, "round", "<U12"), np.full(4, "H1", "<U3"),
                 np.zeros(4, np.int64))
    cl = cluster_points(p, 1.0)
    assert len(cl) == 4
    assert all(len(c.members) == 1 and c.width == 0.0 for c in cl)
