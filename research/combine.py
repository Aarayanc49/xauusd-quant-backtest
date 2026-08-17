"""Stack the MEASURED filters and see if the thing turns positive.

This is the step that destroyed v10, so it is worth being explicit about why this
attempt is different. That wave stacked six filters chosen by reasoning —
tier, magnet share, zone width, room, one-fire-per-bar, PA weights — each
individually defensible, never measured alone. Survival multiplied to 0.072 and
the trade rate fell 93% for a break-even edge.

Here every filter has already been measured ALONE on 28,493 candidates over 14.6
years, against a random-entry control on identical geometry. The stacking is still
dangerous — overlapping filters can select the same trades twice, and a
combination fitted on the whole sample is fitted — so:

  * every combination reports n, and a combination that leaves too few trades is
    reported as untradeable no matter how good its average looks
  * every combination is split into halves and per-year
  * the survival rate of each added filter is printed, so the multiplicative
    collapse that killed v10 is visible while it happens rather than afterwards

    python -m research.combine
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import features as F  # noqa: E402

BASE = F.RANDOM_BASELINE

# Each filter measured alone first. Name -> predicate.
FILTERS = {
    "vol: range_pct>=.75": lambda x: x["range_pct"] >= 0.75,
    "vol: atr_pct>=.75": lambda x: x["atr_pct"] >= 0.75,
    "vol: expanding>=1.15": lambda x: x["expansion"] >= 1.15,
    "cost: spread_pct<.5": lambda x: x["spread_pct"] < 0.5,
    "time: london/ny/ovlp": lambda x: x["session"] in ("london", "ny", "overlap"),
    "level: age<200": lambda x: x["level_age"] < 200,
    "leg: 4-12 ATR": lambda x: 4.0 <= x["leg_atr"] < 12.0,
}


def stat(rows):
    if not rows:
        return None
    r = np.array([x["r"] for x in rows])
    return dict(n=len(r), win=(r > 0.05).mean(), avg=r.mean(),
                lift=r.mean() - BASE, sum=r.sum())


def halves(rows):
    t = sorted(rows, key=lambda x: x["bar"])
    m = len(t) // 2
    return stat(t[:m]), stat(t[m:])


def years(rows):
    by = defaultdict(list)
    for x in rows:
        by[x["year"]].append(x)
    return {y: stat(v) for y, v in sorted(by.items())}


def report(rows, label, all_n):
    s = stat(rows)
    if s is None or s["n"] == 0:
        print(f"  {label:<44} n=    0")
        return
    h1, h2 = halves(rows)
    ys = years(rows)
    pos = sum(1 for v in ys.values() if v and v["avg"] > 0)
    ok = "OK " if (h1 and h2 and h1["avg"] > 0 and h2["avg"] > 0) else "   "
    tradeable = "" if s["n"] >= 300 else "  (too few to trade)"
    print(f"  {label:<44} n={s['n']:>5} ({s['n']/all_n:>5.1%})  "
          f"win={s['win']:>5.1%}  avg={s['avg']:>+6.3f}R  "
          f"halves {h1['avg']:>+6.3f}/{h2['avg']:>+6.3f} {ok} "
          f"yrs {pos}/{len(ys)}{tradeable}")


def main(target_r=8.0, hold_hours=24, symbol="XAUUSD"):
    rows = F.build(symbol=symbol, target_r=target_r, hold_hours=hold_hours)
    all_n = len(rows)
    print("\n" + "=" * 118)
    print(f"  MEASURED FILTERS, ALONE — {all_n:,} candidates, 1.5 ATR stop, "
          f"2R target, funded spread")
    print("=" * 118)
    report(rows, "ALL (no filter)", all_n)
    for name, fn in FILTERS.items():
        report([x for x in rows if fn(x)], name, all_n)

    print("\n" + "=" * 118)
    print("  STACKED — added one at a time, best-first. Watch n collapse.")
    print("=" * 118)
    order = ["vol: range_pct>=.75", "cost: spread_pct<.5",
             "time: london/ny/ovlp", "vol: expanding>=1.15", "leg: 4-12 ATR"]
    cur = rows
    for name in order:
        prev = len(cur)
        cur = [x for x in cur if FILTERS[name](x)]
        surv = len(cur) / prev if prev else 0
        report(cur, f"+ {name}  (survival {surv:.0%})", all_n)
        if len(cur) < 100:
            print("    -> population exhausted; stopping")
            break

    # the surviving stack, year by year — a blended average over 14 years of a
    # market that went from $1,200 to $4,700 is not a result on its own
    best = [x for x in rows
            if FILTERS["vol: range_pct>=.75"](x)
            and FILTERS["cost: spread_pct<.5"](x)
            and FILTERS["time: london/ny/ovlp"](x)]
    print("\n" + "=" * 118)
    print(f"  THE SURVIVING STACK, PER YEAR — n={len(best):,} "
          f"({len(best)/14.6:.0f} trades/year)")
    print("=" * 118)
    eq = 0.0
    for y, s in years(best).items():
        eq += s["sum"]
        bar = "+" * int(max(0, s["avg"]) * 60) + "-" * int(max(0, -s["avg"]) * 60)
        print(f"    {y}  n={s['n']:>4}  win={s['win']:>5.1%}  "
              f"avg={s['avg']:>+6.3f}R  sum={s['sum']:>+7.1f}R  "
              f"cum={eq:>+7.1f}R  {bar}")

    print("\n" + "=" * 118)
    print("  THE VOLATILITY REGIME SPLIT — the operator's point, measured")
    print("=" * 118)
    for lo, hi, lbl in ((0.0, 0.33, "low-vol third"),
                        (0.33, 0.66, "mid-vol third"),
                        (0.66, 1.01, "high-vol third")):
        sub = [x for x in rows if lo <= x["range_pct"] < hi]
        report(sub, lbl, all_n)
    print("\n  per-year, high-vol third only:")
    hv = [x for x in rows if x["range_pct"] >= 0.66]
    for y, s in years(hv).items():
        if s:
            print(f"    {y}  n={s['n']:>5}  win={s['win']:>5.1%}  "
                  f"avg={s['avg']:>+6.3f}R")
    return 0


if __name__ == "__main__":
    import sys as _s
    _t = float(_s.argv[1]) if len(_s.argv) > 1 else 8.0
    _sym = _s.argv[2] if len(_s.argv) > 2 else "XAUUSD"
    raise SystemExit(main(_t, 24, _sym))
