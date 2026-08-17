"""The INTRADAY trader — fixed 20-30 pip stop, ~100 pip target, closed same day.

A second, separate system from the swing trader we just validated. The operator's
spec, and the reasoning behind it is sound: a $5k or $10k account has no leverage
headroom, so it needs a small absolute stop and a target it can actually reach
inside a session.

    stop     20-30 pips   ($2-3 on gold)
    target   ~100 pips    ($10 on gold)
    hold     intraday, closed by session end
    R:R      ~4:1

## The thing this has to survive

A FIXED pip stop means something completely different in each regime, and gold is
the worst instrument for it:

    2013   ATR(M15) ~ 20 pips   ->  a 25p stop is ~1.2 ATR   (roomy)
    2026   ATR(M15) ~ 100 pips  ->  a 25p stop is ~0.25 ATR  (inside noise)

Everything else in this project is ATR-scaled precisely because of that. So the
honest expectation is that a fixed-pip intraday system worked in the low-
volatility years and stopped working as gold's range exploded — and the per-year
table is the only way to see it. If that is what the data says, the fix is an
ATR-scaled stop that happens to be ~25 pips in 2013 conditions, not a hard 25.

    python -m research.intraday
    python -m research.intraday --sweep      # SL x TP grid
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import features as F  # noqa: E402


def intraday_filter(x) -> bool:
    """The swing filters minus the ones that only make sense over 24h.

    Volatility and spread stay — both were the strongest separators and both
    apply even more to a tight-stop system, because a 25-pip stop on a 3-pip
    spread is paying 12% of its risk in cost before it starts.
    """
    return (x["spread_pct"] < 0.5
            and x["session"] in ("london", "ny", "overlap")
            and x["h4_pullback"] >= 0.4)


def stat(rows):
    if not rows:
        return None
    r = np.array([x["r"] for x in rows])
    pips = np.array([x["r"] * x["risk_pips"] for x in rows])
    t = sorted(rows, key=lambda x: x["bar"])
    m = len(t) // 2
    return dict(n=len(r), win=(r > 0.05).mean(), avg=r.mean(),
                pips=pips.mean(), total_pips=pips.sum(), sum=r.sum(),
                h1=np.mean([x["r"] for x in t[:m]]) if m else 0,
                h2=np.mean([x["r"] for x in t[m:]]) if m else 0)


def per_year(rows, label):
    by = defaultdict(list)
    for x in rows:
        by[x["year"]].append(x)
    print(f"\n  {label} — per year")
    print(f"    {'yr':<6}{'n':>6}{'win':>8}{'avg R':>9}{'avg pips':>10}"
          f"{'total pips':>12}{'sum R':>9}")
    pos = 0
    for y in sorted(by):
        s = stat(by[y])
        pos += s["avg"] > 0
        print(f"    {y:<6}{s['n']:>6}{s['win']:>7.1%}{s['avg']:>+9.3f}"
              f"{s['pips']:>+10.1f}{s['total_pips']:>+12.0f}{s['sum']:>+9.1f}")
    print(f"    -> {pos} of {len(by)} years positive")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--sl", type=float, default=25.0, help="stop in pips")
    p.add_argument("--tp", type=float, default=100.0, help="target in pips")
    p.add_argument("--hours", type=int, default=8, help="max hold, intraday")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--from-year", type=int, default=None)
    a = p.parse_args(argv)

    if a.sweep:
        print("\n" + "=" * 96)
        print(f"  INTRADAY GRID — {a.symbol}, fixed pip stop/target, "
              f"{a.hours}h max hold")
        print("=" * 96)
        tps = (50, 75, 100, 150)
        print(f"  {'SL':<8}" + "".join(f"{('TP ' + str(t) + 'p'):>19}" for t in tps))
        for sl in (20, 25, 30, 40, 60):
            cells = []
            for tp in tps:
                rows = F.build(symbol=a.symbol, stop_pips=sl, target_pips=tp,
                               hold_hours=a.hours, quiet=True)
                sel = [x for x in rows if intraday_filter(x)
                       and (a.from_year is None or x["year"] >= a.from_year)]
                s = stat(sel)
                cells.append(f"{s['avg']:>+7.3f}R {s['win']:>5.1%} n{s['n']:<5}"
                             if s else "  -")
            print(f"  {sl:<8}" + "".join(f"{c:>19}" for c in cells))
        return 0

    rows = F.build(symbol=a.symbol, stop_pips=a.sl, target_pips=a.tp,
                   hold_hours=a.hours)
    sel = [x for x in rows if intraday_filter(x)
           and (a.from_year is None or x["year"] >= a.from_year)]
    print("\n" + "=" * 96)
    print(f"  INTRADAY TRADER — {a.symbol}   SL {a.sl:g}p / TP {a.tp:g}p "
          f"(R:R 1:{a.tp/a.sl:.1f})   max hold {a.hours}h")
    print("=" * 96)
    for lbl, rws in (("ALL candidates", rows), ("filtered", sel)):
        s = stat(rws)
        print(f"  {lbl:<18} n={s['n']:>6} win={s['win']:>5.1%} "
              f"avg={s['avg']:>+6.3f}R ({s['pips']:>+5.1f} pips) "
              f"total {s['total_pips']:>+8.0f} pips  "
              f"halves {s['h1']:>+6.3f}/{s['h2']:>+6.3f}")
    per_year(sel, "filtered")

    # the regime question this design lives or dies on
    print("\n  by volatility regime (is a fixed pip stop regime-dependent?):")
    for lo, hi, lbl in ((0.0, 0.33, "low ATR third"), (0.33, 0.66, "mid"),
                        (0.66, 1.01, "high ATR third")):
        sub = [x for x in sel if lo <= x["atr_pct"] < hi]
        s = stat(sub)
        if s:
            print(f"    {lbl:<18} n={s['n']:>6} win={s['win']:>5.1%} "
                  f"avg={s['avg']:>+6.3f}R ({s['pips']:>+5.1f} pips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
