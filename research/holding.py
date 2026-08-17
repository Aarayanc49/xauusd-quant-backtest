"""How much edge is thrown away by exiting early?

The operator trades gold discretionarily and beats the bot. That gap is
information, and the most likely explanation is structural rather than skill:

    the bot     max 24h hold, 2R target, ~198 trades/year
    a human     holds for days or weeks, lets winners run

2024 is the case in point. Gold rose 27.1% and the strategy lost 41.5R. A person
holding long made money without doing anything clever. No intraday system with a
24-hour clock can participate in that, because the move takes months.

This sweeps hold time and target together on the SAME entries, so the question
"is the entry bad, or is the exit throwing the move away?" gets a direct answer.

    python -m research.holding
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.discover import load_series  # noqa: E402
from core.exits import NO_TRAIL, Plan, simulate  # noqa: E402
from research import accounts as A  # noqa: E402
from research import features as F  # noqa: E402
from research.engine import FUNDED_SPREAD_MULT  # noqa: E402

HOURS = (1, 2, 4, 6, 12, 24, 72)
TARGETS = (2.0, 3.0, 5.0, 8.0, None)


def main():
    rows = F.build()
    trades = [x for x in rows if A.surviving(x)]
    trades.sort(key=lambda x: x["bar"])

    series = load_series("XAUUSD")
    bars, m1 = series["M5"], series["M1"]
    pip = bars.pip
    m1_at = np.searchsorted(np.asarray(m1.time, np.int64),
                            np.asarray(bars.time, np.int64) + bars.bar_seconds,
                            "left")
    half_m1 = (m1.spread_pips() * pip * FUNDED_SPREAD_MULT) / 2.0

    print("\n" + "=" * 100)
    print(f"  HOLD TIME x TARGET — same {len(trades):,} entries, 1.5 ATR stop, "
          f"no trail")
    print("=" * 100)
    print(f"  {'hold':<10}" + "".join(
        f"{('TP ' + (f'{t:g}R' if t else 'none')):>17}" for t in TARGETS))

    best = None
    for hrs in HOURS:
        cells = []
        for tgt in TARGETS:
            rs = []
            for t in trades:
                d = t["direction"]
                buy = d == "buy"
                i = t["bar"]
                risk = float(t["risk_pips"]) * pip
                entry = float(t["entry"]) if "entry" in t else None
                # rebuild the plan from the recorded stop distance
                px = float(bars.close[i]) + (1 if buy else -1) * 0.0
                stop = px - risk if buy else px + risk
                target = None
                if tgt is not None:
                    target = px + tgt * risk if buy else px - tgt * risk
                a = int(m1_at[i])
                b = min(a + hrs * 60, len(m1))
                if b - a < 2:
                    continue
                out = simulate(Plan(entry=px, stop=stop, direction=d, risk=risk,
                                    target=target, ladder=NO_TRAIL),
                               m1.high[a:b], m1.low[a:b], m1.close[a:b],
                               half_m1[a:b])
                rs.append(out["r"])
            r = np.array(rs)
            avg = r.mean() if len(r) else 0.0
            wr = (r > 0.05).mean() if len(r) else 0.0
            cells.append(f"{avg:>+8.3f}R {wr:>5.1%}")
            if best is None or avg > best[0]:
                best = (avg, hrs, tgt, r)
        lbl = f"{hrs}h" if hrs < 48 else f"{hrs//24}d"
        print(f"  {lbl:<10}" + "".join(f"{c:>17}" for c in cells))

    avg, hrs, tgt, r = best
    print(f"\n  BEST: hold {hrs}h ({hrs/24:.0f}d), target "
          f"{('%gR' % tgt) if tgt else 'none'} -> {avg:+.3f}R/trade  "
          f"total {r.sum():+.0f}R over {len(r):,} trades")
    m = len(r) // 2
    print(f"  halves: {r[:m].mean():+.3f}R / {r[m:].mean():+.3f}R")
    print("\n  Compare: the shipped config (24h, 2R) returns +0.074R/trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
