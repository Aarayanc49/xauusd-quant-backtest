"""Random-entry control — is the harness fair?

A negative result is only worth believing if the measuring instrument is honest.
If random entries with the SAME stop/target geometry and the SAME costs come back
at roughly breakeven-minus-cost, the harness is fair and a strategy that scores
below that is genuinely bad. If random entries also lose heavily, the harness is
broken and every conclusion drawn from it is worthless.

The old project never ran this. It is the cheapest possible protection against
spending months tuning against a bug, and it takes seconds.

Expectation for a fair harness, entering at random with a 1R stop and a 2R target:
    win rate  ->  1/(1+2) = 33.3%  minus a little for the stop-first tie rule
    avg R     ->  slightly negative, by roughly the spread cost
Anything much worse than that means the simulator, not the strategy, is the problem.

    python -m research.control
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.exits import NO_TRAIL, Plan, simulate  # noqa: E402
from core.discover import load_series  # noqa: E402
from core.context import Context  # noqa: E402
from research.engine import FUNDED_SPREAD_MULT  # noqa: E402


def run(symbol="XAUUSD", n=4000, target_r=2.0, stop_atr=1.5, seed=0,
        spread_mult=FUNDED_SPREAD_MULT):
    series = load_series(symbol)
    bars, m1 = series["M5"], series["M1"]
    ctx = Context(series, base="M5")
    atr = ctx.threshold("stop") / 1.50
    pip = bars.pip

    m1_at = np.searchsorted(np.asarray(m1.time, np.int64),
                            np.asarray(bars.time, np.int64) + bars.bar_seconds,
                            "left")
    half_m1 = (m1.spread_pips() * pip * spread_mult) / 2.0
    sp_base = bars.spread_pips() * pip * spread_mult

    rng = np.random.default_rng(seed)
    lo, hi = 300, len(bars) - 300
    picks = rng.integers(lo, hi, n)
    out = []
    for i in picks:
        i = int(i)
        a = float(atr[i])
        if not np.isfinite(a) or a <= 0:
            continue
        buy = bool(rng.random() > 0.5)
        d = "buy" if buy else "sell"
        entry = float(bars.close[i]) + (sp_base[i] / 2 if buy else -sp_base[i] / 2)
        risk = stop_atr * a
        stop = entry - risk if buy else entry + risk
        target = entry + target_r * risk if buy else entry - target_r * risk
        s = int(m1_at[i])
        e = min(s + 24 * 60, len(m1))
        if e - s < 2:
            continue
        plan = Plan(entry=entry, stop=stop, direction=d, risk=risk,
                    target=target, ladder=NO_TRAIL)
        out.append(simulate(plan, m1.high[s:e], m1.low[s:e], m1.close[s:e],
                            half_m1[s:e]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--n", type=int, default=4000)
    a = p.parse_args(argv)

    print("=" * 84)
    print("  RANDOM-ENTRY CONTROL — is the harness fair?")
    print("=" * 84)
    print(f"  {'target':<9}{'n':>7}{'win%':>8}{'expected':>10}{'avg R':>9}"
          f"{'payoff':>8}{'timeouts':>10}")
    for tr in (1.0, 1.5, 2.0, 3.0):
        rows = run(a.symbol, n=a.n, target_r=tr)
        if not rows:
            continue
        r = np.array([x["r"] for x in rows])
        w = r[r > 0.05]
        l = r[r < -0.05]
        payoff = (w.mean() / abs(l.mean())) if len(w) and len(l) else float("nan")
        expect = 1.0 / (1.0 + tr)
        tmo = sum(1 for x in rows if x["exit_reason"] == "timeout") / len(rows)
        print(f"  {tr:<9.1f}{len(r):>7}{(r > 0.05).mean():>7.1%}{expect:>10.1%}"
              f"{r.mean():>+9.3f}{payoff:>8.2f}{tmo:>10.1%}")
    print("\n  A fair harness lands win% near expected and avg R slightly negative")
    print("  (the spread). Much worse than that = the simulator is the problem,")
    print("  not the strategy.")

    # ── the cost curve ──────────────────────────────────────────────────────
    # Spread is a fixed number of PIPS. Its cost measured in R therefore falls as
    # the stop widens. This is the single most actionable number in the project:
    # it says exactly how much of the edge the broker takes at each stop size.
    print("\n" + "=" * 84)
    print("  COST CURVE — spread cost in R, by stop size (random entry, 2R target)")
    print("=" * 84)
    print(f"  {'stop':<12}{'n':>7}{'win%':>8}{'avg R':>10}{'cost vs 0-spread':>20}")
    for satr in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        paid = run(a.symbol, n=a.n, target_r=2.0, stop_atr=satr)
        free = run(a.symbol, n=a.n, target_r=2.0, stop_atr=satr, spread_mult=0.0)
        if not paid:
            continue
        rp = np.array([x["r"] for x in paid])
        rf = np.array([x["r"] for x in free])
        print(f"  {satr:<12.2f}{len(rp):>7}{(rp > 0.05).mean():>7.1%}"
              f"{rp.mean():>+10.3f}{rp.mean() - rf.mean():>+20.3f}")
    print("\n  Cost in R is roughly proportional to 1/stop. A strategy whose gross")
    print("  edge is smaller than this row is unprofitable no matter how it is tuned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
