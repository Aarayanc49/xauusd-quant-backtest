"""The benchmark nobody ran: does the strategy beat just OWNING the gold?

The operator's question, and it is the correct one. Gold went from roughly $1,600
in 2012 to $4,700 in 2026. Any active strategy on gold has to clear that bar
before it is worth the effort, the risk and the software — and if it does not,
the honest answer is to say so rather than keep tuning.

This compares, on identical data and identical dates:

    BUY AND HOLD      buy at the start, hold to the end, no leverage
    B&H LEVERAGED     same, sized so its worst drawdown matches the strategy's,
                      which is the only fair way to compare two things with
                      different risk
    STRATEGY          the surviving config, compounding, no withdrawals

Risk-adjusted comparison matters because an unlevered buy-and-hold that returns
13%/yr with a 45% drawdown is NOT better than a strategy returning 11%/yr with a
29% one. Return per unit of drawdown is the comparison that survives scrutiny.

    python -m research.benchmark
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discover import load_series  # noqa: E402
from research import accounts as A  # noqa: E402
from research import features as F  # noqa: E402


def buy_hold(bars, lo_t, hi_t, capital=10_000.0):
    """Unlevered long gold over the same window, with its real drawdown."""
    t = np.asarray(bars.time, np.int64)
    i = int(np.searchsorted(t, lo_t, "left"))
    j = int(np.searchsorted(t, hi_t, "right")) - 1
    c = np.asarray(bars.close, np.float64)[i:j + 1]
    if len(c) < 2:
        return None
    units = capital / c[0]
    eq = units * c
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    years = (t[j] - t[i]) / (365.25 * 86400)
    return dict(start=c[0], end=c[-1], final=eq[-1], years=years,
                total_pct=(eq[-1] / capital - 1) * 100,
                cagr=((eq[-1] / capital) ** (1 / years) - 1) * 100,
                max_dd=dd.max() * 100, eq=eq)


def main(start_year=None):
    rows = F.build()
    trades = [x for x in rows if A.surviving(x)]
    trades.sort(key=lambda x: x["bar"])
    if start_year:
        trades = [x for x in trades if x["year"] >= start_year]

    series = load_series("XAUUSD")
    bars = series["M5"]
    t = np.asarray(bars.time, np.int64)
    lo_t, hi_t = int(t[trades[0]["bar"]]), int(t[trades[-1]["bar"]])

    bh = buy_hold(bars, lo_t, hi_t)
    spec = A.Spec("NORMAL 10k", 10_000, 0.40, funded=False, risk_pct=1.0)
    res = A.simulate(trades, spec)
    yrs = bh["years"]
    strat_cagr = ((res.final / spec.base) ** (1 / yrs) - 1) * 100

    print("\n" + "=" * 92)
    print(f"  BENCHMARK — {np.datetime64(lo_t, 's')} .. {np.datetime64(hi_t, 's')}"
          f"   ({yrs:.1f} years)")
    print("=" * 92)
    print(f"  gold went ${bh['start']:,.0f} -> ${bh['end']:,.0f}  "
          f"({bh['total_pct']:+,.0f}%)")
    print()
    print(f"  {'':<26}{'final':>14}{'total':>12}{'CAGR':>9}{'maxDD':>9}{'ret/DD':>9}")
    print(f"  {'BUY & HOLD (unlevered)':<26}{bh['final']:>14,.0f}"
          f"{bh['total_pct']:>11,.0f}%{bh['cagr']:>8.1f}%{bh['max_dd']:>8.1f}%"
          f"{bh['cagr']/bh['max_dd']:>9.2f}")
    print(f"  {'STRATEGY (1% risk)':<26}{res.final:>14,.0f}"
          f"{(res.final/spec.base-1)*100:>11,.0f}%{strat_cagr:>8.1f}%"
          f"{res.max_dd_pct:>8.1f}%{strat_cagr/max(res.max_dd_pct,1e-9):>9.2f}")

    # match the risk: lever buy-and-hold until its drawdown equals the strategy's
    lev = res.max_dd_pct / bh["max_dd"]
    lev_cagr = ((1 + bh["total_pct"] / 100 * lev) ** (1 / yrs) - 1) * 100
    print(f"  {'B&H levered to same DD':<26}{10_000*(1+bh['total_pct']/100*lev):>14,.0f}"
          f"{bh['total_pct']*lev:>11,.0f}%{lev_cagr:>8.1f}%"
          f"{res.max_dd_pct:>8.1f}%{lev_cagr/max(res.max_dd_pct,1e-9):>9.2f}"
          f"   (x{lev:.2f})")

    print("\n  VERDICT:", end=" ")
    if strat_cagr / max(res.max_dd_pct, 1e-9) > bh["cagr"] / bh["max_dd"]:
        print("strategy wins on return per unit of drawdown")
    else:
        print("BUY AND HOLD WINS. The strategy is not worth its complexity.")

    # per-year head to head — a single CAGR hides everything that matters
    print("\n  per year:")
    print(f"    {'yr':<6}{'gold %':>10}{'strategy R':>13}{'strategy %':>13}")
    close = np.asarray(bars.close, np.float64)
    years_list = sorted({x["year"] for x in trades})
    for y in years_list:
        sub = [x for x in trades if x["year"] == y]
        m = (t >= np.datetime64(f"{y}-01-01").astype("datetime64[s]").astype(int)) & \
            (t < np.datetime64(f"{y+1}-01-01").astype("datetime64[s]").astype(int))
        idx = np.flatnonzero(m)
        gold = ((close[idx[-1]] / close[idx[0]] - 1) * 100) if len(idx) > 1 else 0.0
        rsum = sum(x["r"] for x in sub)
        print(f"    {y:<6}{gold:>+9.1f}%{rsum:>+13.1f}{rsum:>+12.1f}%")
    print("\n  (strategy % at 1% risk per trade ~= its R sum)")
    return 0


if __name__ == "__main__":
    import sys as _s
    _y = int(_s.argv[1]) if len(_s.argv) > 1 else None
    raise SystemExit(main(_y))
