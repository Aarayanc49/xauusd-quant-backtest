"""Test the strategies that actually have published, out-of-sample evidence.

The intraday level/breakout family we built returns 14.1% CAGR against buy-and-
hold's 12.4% at the same drawdown. That is not an edge worth running software for.
Before building anything else, this tests the two approaches with the strongest
documented evidence, on our own 14.6 years of gold.

**1. Time-series momentum** (Moskowitz, Ooi & Pedersen, JFE 2012). Long if the
past k-month return is positive, short if negative, sized inversely to volatility.
Documented across 58 futures including COMEX gold, 1985-2009, portfolio Sharpe
~1.1. This is the single best-evidenced systematic strategy in the literature, and
it is the opposite of what we built: it holds for months and trades with the trend
rather than fading into it.

**2. Market intraday momentum** (Gao, Han, Li & Zhou, JFE 2018). The first
half-hour return predicts the last half-hour return, and — the part that matters
here — the effect is **stronger on high-volatility and high-volume days**. Our own
feature study independently found day-range percentile to be the single strongest
separator we measured. Two completely different methods pointing at the same
conditioning variable is worth taking seriously.

Both are tested with the same honesty as everything else: per year, both halves,
and against buy-and-hold on identical dates.

    python -m research.tsmom
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discover import load_series  # noqa: E402


def perf(ret, label, years, dd_floor=1e-9):
    """Compounded stats for a daily return series."""
    eq = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100
    total = (eq[-1] - 1) * 100
    cagr = (eq[-1] ** (1 / years) - 1) * 100
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    print(f"  {label:<34}{total:>10,.0f}%{cagr:>8.1f}%{dd:>8.1f}%"
          f"{cagr/max(dd,dd_floor):>8.2f}{sharpe:>8.2f}")
    return dict(total=total, cagr=cagr, dd=dd, sharpe=sharpe, eq=eq)


def main():
    series = load_series("XAUUSD")
    d1 = series["D1"]
    t = np.asarray(d1.time, np.int64)
    c = np.asarray(d1.close, np.float64)
    n = len(c)
    years = (t[-1] - t[0]) / (365.25 * 86400)
    ret = np.zeros(n)
    ret[1:] = c[1:] / c[:-1] - 1

    print("\n" + "=" * 92)
    print(f"  GOLD D1  {np.datetime64(int(t[0]),'s')} .. {np.datetime64(int(t[-1]),'s')}"
          f"   {n:,} days, {years:.1f} years")
    print("=" * 92)
    print(f"  {'':<34}{'total':>10}{'CAGR':>8}{'maxDD':>8}{'ret/DD':>8}{'Sharpe':>8}")

    bh = perf(ret[1:], "BUY & HOLD", years)

    # ── time-series momentum ────────────────────────────────────────────────
    # signal = sign of the trailing k-day return, applied to the NEXT day.
    # The shift is what keeps it causal; without it this is a lookahead machine
    # that prints a Sharpe of 8 and means nothing.
    print()
    best = None
    for k in (20, 60, 120, 180, 250):
        sig = np.zeros(n)
        sig[k:] = np.sign(c[k:] / c[:-k] - 1)
        pos = np.zeros(n)
        pos[1:] = sig[:-1]                    # yesterday's signal, today's return
        r = pos[1:] * ret[1:]
        p = perf(r, f"TSMOM {k}d (long/short)", years)
        if best is None or p["sharpe"] > best[1]["sharpe"]:
            best = (k, p, pos)

    # long-only variant — shorting gold has been a losing side for 14 years
    print()
    for k in (60, 120, 250):
        sig = np.zeros(n)
        sig[k:] = (c[k:] / c[:-k] - 1) > 0
        pos = np.zeros(n)
        pos[1:] = sig[:-1]
        perf(pos[1:] * ret[1:], f"TSMOM {k}d (long/flat)", years)

    # ── volatility-scaled TSMOM ─────────────────────────────────────────────
    # MOP size positions inversely to trailing volatility so each bet carries the
    # same risk. On an instrument whose daily range went 124p -> 1,322p this is
    # not a refinement, it is the difference between a strategy and a lottery.
    print()
    k, _, pos = best
    vol = np.zeros(n)
    for i in range(60, n):
        vol[i] = ret[i - 60:i].std()
    target = np.nanmedian(vol[vol > 0])
    scale = np.zeros(n)
    good = vol > 0
    scale[good] = np.clip(target / vol[good], 0.25, 3.0)
    sc = np.zeros(n)
    sc[1:] = scale[:-1]
    perf(pos[1:] * sc[1:] * ret[1:], f"TSMOM {k}d vol-scaled", years)

    # ── the conditioning both literatures agree on ──────────────────────────
    # Gao/Han/Li/Zhou find intraday momentum is stronger on volatile days; our own
    # feature study found day-range percentile the strongest separator. Test the
    # simplest version: TSMOM but only on days following elevated volatility.
    hi = np.zeros(n, bool)
    for i in range(250, n):
        hi[i] = vol[i] >= np.percentile(vol[i - 250:i][vol[i - 250:i] > 0], 60)
    gate = np.zeros(n)
    gate[1:] = hi[:-1]
    perf(pos[1:] * sc[1:] * gate[1:] * ret[1:],
         f"TSMOM {k}d vol-scaled, hi-vol only", years)

    # ── halves + per-year on the best variant ───────────────────────────────
    r = pos[1:] * sc[1:] * ret[1:]
    m = len(r) // 2
    print("\n  two-halves (vol-scaled TSMOM):")
    for lbl, seg in (("H1", r[:m]), ("H2", r[m:])):
        eq = np.cumprod(1 + seg)
        print(f"    {lbl}  total {(eq[-1]-1)*100:>+8.1f}%   "
              f"Sharpe {seg.mean()/seg.std()*np.sqrt(252):>5.2f}")

    yr = t[1:].astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    print("\n  per year   (vol-scaled TSMOM vs buy & hold):")
    print(f"    {'yr':<6}{'TSMOM':>10}{'B&H':>10}")
    for y in sorted(set(yr.tolist())):
        m2 = yr == y
        a = (np.prod(1 + r[m2]) - 1) * 100
        b = (np.prod(1 + ret[1:][m2]) - 1) * 100
        print(f"    {y:<6}{a:>+9.1f}%{b:>+9.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
