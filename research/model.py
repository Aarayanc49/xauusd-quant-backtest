"""The scoring model — evidence accumulation across every feature family.

Not a gate stack. Gates multiply: six individually defensible filters took v10 to
7% of population for a break-even edge, and the anchored study then showed that
filtering a trigger that carries nothing recovers nothing. This scores instead —
every candidate gets a number, the trade rate is a threshold you choose, and no
single feature can veto.

## How it scores

For each feature, bin it and measure the average R inside each bin against the
population average. That per-bin difference is the feature's EVIDENCE, in R. A
candidate's score is the sum of the evidence its bins carry:

    score(x) = SUM over features f of   lift[f][ bin_f(x) ]

Three properties make this the right shape for this project. It is additive, so
nothing collapses the population. It is binned, so a non-monotonic relationship
(the pullback-depth result was strongly non-monotonic) survives instead of being
flattened by a linear fit. And it is readable — `--explain` prints exactly which
features paid for any given trade, which no tree ensemble would.

The obvious weakness is that summing correlated features double-counts evidence.
198 columns computed from one price series are nowhere near 198 independent
things. That is what the selection stage exists to control.

## The discipline

Overfitting is the entire risk here, so the split is three-way and strict:

    IS-A   2012 .. midpoint     fit the bin lifts
    IS-B   midpoint .. 2019     SELECT features — nothing is chosen on IS-A
    OOS    2020 .. 2026         never touched until the model is frozen

Features are added greedily and one at a time, and a feature is kept only if
adding it improves the running score's discrimination **on IS-B**, which its own
lifts were not fitted on. Then the chosen set is refitted on all of IS and
applied once to OOS.

Everything is judged against the matched random control at identical geometry
(-0.163R), never against zero, and never against the neighbouring cell.

    python -m research.model
    python -m research.model --bins 6 --max-features 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.exits import OUTCOME_KEYS  # noqa: E402
from research.universe import CACHE, CONTROL_R  # noqa: E402

# Outcome and bookkeeping columns — never inputs.
#
# `OUTCOME_KEYS` comes from core/exits.py so that adding a field to what
# `simulate` returns cannot silently make it an input. The first version of this
# file listed the exclusions by hand, missed `bars` and `exit_price`, and
# produced a 56%-win-rate out-of-sample result built on trade duration.
#
# `risk_pips` is excluded for a subtler reason: it is 1.5 x ATR in pips, and
# gold's ATR grew roughly fourfold across the sample, so it is a proxy for WHICH
# YEAR a trade happened in. A model allowed to read it learns "trade after 2020"
# and scores beautifully out of sample for a reason that is not skill.
EXCLUDE = set(OUTCOME_KEYS) | {
    "pips", "bar", "year", "day", "ts", "direction", "risk_pips",
}

MIN_BIN = 400          # a bin with fewer candidates than this is not evidence
TRADE_PCTL = 0.90      # trade the top decile by default


def load(path=CACHE):
    if not os.path.exists(path):
        raise SystemExit(f"no universe at {path} — run research.universe first")
    return json.load(open(path))


def to_matrix(rows):
    """Flat float32 matrix plus the outcome vectors."""
    keys = [k for k in rows[0]
            if k not in EXCLUDE and isinstance(rows[0][k], (int, float, bool))]
    keys.sort()
    X = np.empty((len(rows), len(keys)), np.float32)
    for j, k in enumerate(keys):
        X[:, j] = [float(x.get(k, 0.0) or 0.0) for x in rows]
    y = np.array([x["r"] for x in rows], np.float64)
    bar = np.array([x["bar"] for x in rows], np.int64)
    yr = np.array([x["year"] for x in rows], np.int32)
    return keys, X, y, bar, yr


def fit_bins(X, y, keys, nbins):
    """Quantile edges and per-bin lift for every feature, fitted on one slice."""
    base = y.mean()
    edges, lifts, usable = [], [], []
    for j in range(X.shape[1]):
        col = X[:, j]
        qs = np.unique(np.quantile(col, np.linspace(0, 1, nbins + 1)[1:-1]))
        idx = np.digitize(col, qs)
        lift = np.zeros(len(qs) + 1)
        ok = True
        for b in range(len(qs) + 1):
            m = idx == b
            if m.sum() < MIN_BIN:
                # a bin too thin to be evidence contributes nothing rather than
                # a loud number from forty trades
                lift[b] = 0.0
                continue
            lift[b] = y[m].mean() - base
        # a feature with only one populated bin carries no information
        if len(qs) < 1 or np.all(lift == 0):
            ok = False
        edges.append(qs)
        lifts.append(lift)
        usable.append(ok)
    return edges, lifts, np.array(usable)


def score_with(X, edges, lifts, cols):
    """Summed evidence for a chosen set of feature indices."""
    s = np.zeros(len(X), np.float64)
    for j in cols:
        s += lifts[j][np.digitize(X[:, j], edges[j])]
    return s


def top_mean(score, y, pctl):
    """Mean R of the top slice by score — the number the model is judged on."""
    if len(score) == 0:
        return 0.0, 0
    cut = np.quantile(score, pctl)
    m = score >= cut
    return (float(y[m].mean()) if m.any() else 0.0), int(m.sum())


def select(Xa, ya, Xb, yb, keys, nbins, max_features, pctl):
    """Greedy forward selection. Lifts fitted on A, every decision made on B."""
    edges, lifts, usable = fit_bins(Xa, ya, keys, nbins)

    solo = []
    for j in range(Xa.shape[1]):
        if not usable[j]:
            continue
        sb = lifts[j][np.digitize(Xb[:, j], edges[j])]
        m, cnt = top_mean(sb, yb, pctl)
        if cnt >= MIN_BIN:
            solo.append((j, m))
    solo.sort(key=lambda t: -t[1])

    chosen, cur = [], np.zeros(len(Xb))
    best, _ = top_mean(cur, yb, pctl)
    trail = []
    for j, _m in solo:
        if len(chosen) >= max_features:
            break
        trial = cur + lifts[j][np.digitize(Xb[:, j], edges[j])]
        got, cnt = top_mean(trial, yb, pctl)
        if got > best + 1e-4 and cnt >= MIN_BIN:
            chosen.append(j)
            cur = trial
            trail.append((keys[j], got, got - best))
            best = got
    return chosen, edges, lifts, trail, solo


def stats(y, yr, bar):
    if not len(y):
        return None
    order = np.argsort(bar)
    ys = y[order]
    m = len(ys) // 2
    per = defaultdict(list)
    for v, k in zip(y, yr):
        per[int(k)].append(v)
    pos = sum(1 for v in per.values() if np.mean(v) > 0)
    return dict(n=len(y), win=float((y > 0.05).mean()), avg=float(y.mean()),
                h1=float(ys[:m].mean()) if m else 0.0,
                h2=float(ys[m:].mean()) if m else 0.0,
                pos=pos, nyears=len(per))


def line(label, s, w=30, base=None):
    """`base` is the average of the SAME PERIOD's unfiltered population.

    Comparing an out-of-sample slice against the full-period random control
    flatters it badly: 2020-2026 is a structurally kinder regime, so the whole
    OOS universe beats the control by +0.12R before any model exists. The number
    that means something is the lift over the contemporaneous population — what
    the model added, not what the era added.
    """
    if s is None or not s["n"]:
        return f"    {label:<{w}} n=      0"
    ok = "OK " if s["h1"] > 0 and s["h2"] > 0 else "   "
    lift = "" if base is None else f"  lift {s['avg'] - base:>+6.3f}R"
    return (f"    {label:<{w}} n={s['n']:>7}  win={s['win']:>5.1%}  "
            f"avg={s['avg']:>+6.3f}R{lift}  "
            f"halves {s['h1']:>+6.3f}/{s['h2']:>+6.3f} {ok} "
            f"yrs {s['pos']}/{s['nyears']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--max-features", type=int, default=15)
    p.add_argument("--split", type=int, default=2019)
    p.add_argument("--pctl", type=float, default=TRADE_PCTL)
    a = p.parse_args(argv)

    rows = load(a.cache)
    keys, X, y, bar, yr = to_matrix(rows)
    print("=" * 122)
    print(f"  EVIDENCE MODEL — {len(rows):,} candidates, {len(keys)} features, "
          f"{a.bins} bins, matched control {CONTROL_R:+.3f}R")
    print("=" * 122)

    is_m = yr <= a.split
    oos_m = ~is_m
    bar_is = bar[is_m]
    mid = np.median(bar_is)
    a_m = is_m & (bar <= mid)
    b_m = is_m & (bar > mid)
    print(f"  IS-A  fit      {a_m.sum():>7,}   IS-B  select  {b_m.sum():>7,}"
          f"   OOS  frozen  {oos_m.sum():>7,}")

    chosen, edges, lifts, trail, solo = select(
        X[a_m], y[a_m], X[b_m], y[b_m], keys, a.bins, a.max_features, a.pctl)

    print(f"\n  strongest features ALONE (fitted on IS-A, scored on IS-B, "
          f"top {1-a.pctl:.0%}):")
    for j, m in solo[:14]:
        print(f"    {keys[j]:<34} top-slice avg {m:>+7.3f}R  "
              f"vs ctrl {m - CONTROL_R:>+7.3f}R")

    print(f"\n  greedy selection — kept only if it improves the RUNNING score "
          f"on IS-B:")
    if not trail:
        print("      nothing cleared the bar")
    for name, got, delta in trail:
        print(f"    + {name:<34} running {got:>+7.3f}R   ({delta:>+6.3f})")

    if not chosen:
        print("\n  No feature set survived selection. That is a result: on this "
              "\n  universe, nothing predicts outcome well enough to trade.")
        return 0

    # ── refit on ALL of IS, then apply once to OOS ──────────────────────────
    edges_f, lifts_f, _ = fit_bins(X[is_m], y[is_m], keys, a.bins)
    s_is = score_with(X[is_m], edges_f, lifts_f, chosen)
    s_oos = score_with(X[oos_m], edges_f, lifts_f, chosen)
    cut = float(np.quantile(s_is, a.pctl))

    print("\n" + "=" * 122)
    print(f"  OUT OF SAMPLE — model frozen on {a.split} and earlier, "
          f"threshold set on IS ({cut:+.3f})")
    print("=" * 122)
    s_all_is = stats(y[is_m], yr[is_m], bar[is_m])
    s_all_oos = stats(y[oos_m], yr[oos_m], bar[oos_m])
    base_oos = s_all_oos["avg"]
    print(line("everything, IS", s_all_is))
    print(line("everything, OOS  <- baseline", s_all_oos))
    print(f"    {'matched random control':<30} {CONTROL_R:>+27.3f}R"
          f"   (full period, for scale)")
    print()
    for name, pctl in (("top 50%", 0.50), ("top 25%", 0.75),
                       ("top 10%", 0.90), ("top 5%", 0.95), ("top 2%", 0.98)):
        c = float(np.quantile(s_is, pctl))
        m = s_oos >= c
        s = stats(y[oos_m][m], yr[oos_m][m], bar[oos_m][m])
        yrs = max(1e-9, (bar[oos_m].max() - bar[oos_m].min()) / (288 * 252))
        rate = (s["n"] / yrs) if s else 0
        print(line(f"OOS {name}", s, base=base_oos) + f"  ~{rate:,.0f}/yr")

    # How much of the in-sample discrimination actually survived? The swing
    # config retained ~50% of its lift out of sample. A model that retains a
    # small fraction of a large in-sample number is fitting noise, and the
    # ratio says so more clearly than either number alone.
    c90 = float(np.quantile(s_is, 0.90))
    is_top = y[is_m][s_is >= c90]
    lift_is = is_top.mean() - s_all_is["avg"]
    oos_top = y[oos_m][s_oos >= c90]
    lift_oos = oos_top.mean() - base_oos
    print(f"\n  discrimination at the top decile, over its own period:")
    print(f"      in-sample      {lift_is:>+7.3f}R")
    print(f"      out-of-sample  {lift_oos:>+7.3f}R"
          f"      retained {lift_oos / lift_is if lift_is else 0:>5.0%}")
    print("\n  The model is only worth having if the OOS top slice beats the "
          "SAME PERIOD's\n  unfiltered baseline by more than the per-year "
          "column's noise — not if it\n  merely beats a control measured over "
          "a harsher era.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
