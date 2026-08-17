"""The DAY TRADER — the validated entry, flat by the close.

Everything else has been tried. The time-anchored liquidity family returned a
gross edge of -0.007R over 521,606 events, and a scoring model over 198 features
on an unbiased intraday universe retained 16% of its in-sample discrimination
and was still negative out of sample. What has never been tested properly is the
one entry in this project that measures positive out of sample:

    structure break (BOS/CHoCH) -> impulse leg -> price retraces into
    0.5-0.618 of that leg -> enter on the retrace

The earlier intraday attempt used those entries and returned +0.041R, which
looked like a verdict on the entry. It was not. It paired them with a **fixed
25-pip stop**, and the median 1.5-ATR stop on gold ran 24 pips in 2012-2019 but
69 pips in 2020-2026 — so that stop is roughly 0.5 ATR in the modern market and
pays ~0.54R of spread before the trade begins. It was a verdict on the stop.

So this asks the question cleanly: **ATR-scaled stop, same entries, flat by the
close.** If it fails here it fails for a reason about holding period rather than
a reason about cost, and the day-trader idea is genuinely finished.

## Flat by the close

A day trader does not carry risk overnight, so the hold is
`min(max_hold, time until the session closes)`. That matters more than it
sounds: a signal at 19:30 UTC gets 90 minutes, not four hours, and averaging
those together is what hides a working strategy inside a broken one. Trades with
less than `min_window` minutes left are not taken at all rather than being taken
with no room.

## Speed

Re-simulates the cached candidate set (`data/_feat8.json`, 28,493 entries with
their bar index and direction) under new geometry rather than re-deriving levels
and structure per cell. The entries do not depend on the exit, so a full
hold x target grid costs one pass over M1 per cell instead of rebuilding the
whole book sixteen times.

    python -m research.daytrade --sweep
    python -m research.daytrade --hold 240 --target-r 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.context import Context  # noqa: E402
from core.discover import load_series  # noqa: E402
from core.exits import NO_TRAIL, Plan, simulate  # noqa: E402
from core.strategy import SWING, SWING_NOBIAS  # noqa: E402
from research.engine import FUNDED_SPREAD_MULT  # noqa: E402

FEAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "_feat8.json")

SESSION_CLOSE_HOUR = 21      # NY close, UTC. Flat by here, always.
MIN_WINDOW = 30              # minutes; below this the trade is not taken


class Book:
    """Cached candidates plus the price arrays needed to re-simulate them."""

    def __init__(self, symbol="XAUUSD", base="M5", feat=FEAT,
                 spread_mult=FUNDED_SPREAD_MULT):
        self.rows = json.load(open(feat))
        self.symbol = symbol
        series = load_series(symbol)
        bars, m1 = series[base], series["M1"]
        self.bars, self.m1, self.pip = bars, m1, bars.pip
        ctx = Context(series, base=base)
        self.atr = ctx.threshold("stop") / 1.50
        self.close = np.asarray(bars.close, np.float64)
        t = np.asarray(bars.time, np.int64)
        # Epoch seconds, not the bar index. `bar` is a per-symbol M5 position and
        # is meaningless across instruments — merging six books on it would
        # interleave 2013 gold with 2021 EURUSD and quietly invent a portfolio.
        self.t = t
        self.hour = ((t // 3600) % 24).astype(np.int8)
        self.minute = ((t % 3600) // 60).astype(np.int8)
        self.m1_at = np.searchsorted(np.asarray(m1.time, np.int64),
                                     t + bars.bar_seconds, "left")
        self.half_m1 = (m1.spread_pips() * self.pip * spread_mult) / 2.0
        self.sp = bars.spread_pips() * self.pip * spread_mult

    def minutes_left(self, i):
        """Minutes from this bar's close until the session close."""
        return (SESSION_CLOSE_HOUR - int(self.hour[i])) * 60 - int(self.minute[i])


def run(book: Book, stop_atr=1.5, target_r=3.0, max_hold=240,
        flat_by_close=True, rows=None):
    """Re-simulate the cached entries under one intraday geometry."""
    out = []
    for x in (rows if rows is not None else book.rows):
        i = int(x["bar"])
        a = float(book.atr[i])
        if not np.isfinite(a) or a <= 0:
            continue
        left = book.minutes_left(i)
        hold = min(max_hold, left) if flat_by_close else max_hold
        if hold < MIN_WINDOW:
            continue
        buy = x["direction"] == "buy"
        entry = book.close[i] + (book.sp[i] / 2 if buy else -book.sp[i] / 2)
        risk = stop_atr * a
        p1 = int(book.m1_at[i])
        p2 = min(p1 + hold, len(book.m1))
        if p2 - p1 < 2:
            continue
        r = simulate(
            Plan(entry=entry,
                 stop=entry - risk if buy else entry + risk,
                 direction=x["direction"], risk=risk,
                 target=entry + target_r * risk if buy else entry - target_r * risk,
                 ladder=NO_TRAIL),
            book.m1.high[p1:p2], book.m1.low[p1:p2], book.m1.close[p1:p2],
            book.half_m1[p1:p2])
        out.append({**x, **r, "risk_pips": risk / book.pip,
                    "pips": r["r"] * risk / book.pip, "hold_min": hold,
                    "symbol": book.symbol, "entry_price": float(entry),
                    "bar_utc": int(book.t[i])})
    return out


def stat(rows):
    if not rows:
        return None
    r = np.array([x["r"] for x in rows])
    p = np.array([x["pips"] for x in rows])
    t = sorted(rows, key=lambda x: x["bar"])
    m = len(t) // 2
    ys = defaultdict(list)
    for x in rows:
        ys[x["year"]].append(x["r"])
    pos = sum(1 for v in ys.values() if np.mean(v) > 0)
    return dict(n=len(r), win=float((r > 0.05).mean()), avg=float(r.mean()),
                pips=float(p.mean()), sum=float(r.sum()),
                h1=float(np.mean([x["r"] for x in t[:m]])) if m else 0.0,
                h2=float(np.mean([x["r"] for x in t[m:]])) if m else 0.0,
                pos=pos, nyears=len(ys))


def line(label, s, w=28):
    if s is None or not s["n"]:
        return f"    {label:<{w}} n=    0"
    ok = "OK " if s["h1"] > 0 and s["h2"] > 0 else "   "
    return (f"    {label:<{w}} n={s['n']:>5}  win={s['win']:>5.1%}  "
            f"avg={s['avg']:>+6.3f}R {s['pips']:>+6.1f}p  "
            f"halves {s['h1']:>+6.3f}/{s['h2']:>+6.3f} {ok} "
            f"yrs {s['pos']}/{s['nyears']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stop-atr", type=float, default=1.5)
    p.add_argument("--target-r", type=float, default=3.0)
    p.add_argument("--hold", type=int, default=240)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--split", type=int, default=2019)
    a = p.parse_args(argv)

    print("  loading book ...", flush=True)
    book = Book()
    filt = SWING.select(book.rows)
    print(f"  {len(book.rows):,} candidates, {len(filt):,} pass the validated "
          f"filters\n")

    if a.sweep:
        for label, rows in (("ALL CANDIDATES", book.rows),
                            ("VALIDATED FILTERS", filt)):
            print("=" * 118)
            print(f"  DAY-TRADE GRID — {label}, {a.stop_atr} ATR stop, "
                  f"flat by {SESSION_CLOSE_HOUR}:00 UTC")
            print("=" * 118)
            tgts = (1.5, 2.0, 3.0, 5.0, 8.0)
            print(f"  {'max hold':<11}" + "".join(f"{('TP ' + str(t) + 'R'):>21}"
                                                  for t in tgts))
            for hold in (60, 120, 240, 480):
                cells = []
                for tr in tgts:
                    s = stat(run(book, a.stop_atr, tr, hold, rows=rows))
                    cells.append(
                        f"{s['avg']:>+6.3f}R {s['pos']}/{s['nyears']}y n{s['n']:<5}"
                        if s else "   -")
                print(f"  {str(hold) + 'm':<11}" + "".join(f"{c:>21}" for c in cells))
            print()
        return 0

    for label, rows in (("all candidates", book.rows),
                        ("validated filters", filt),
                        ("no-bias filters", SWING_NOBIAS.select(book.rows))):
        res = run(book, a.stop_atr, a.target_r, a.hold, rows=rows)
        print("=" * 118)
        print(f"  {label.upper()} — {a.stop_atr} ATR stop, {a.target_r}R target, "
              f"max {a.hold}m, flat by {SESSION_CLOSE_HOUR}:00")
        print("=" * 118)
        print(line("full 2012-2026", stat(res)))
        print(line(f"in-sample <= {a.split}",
                   stat([x for x in res if x["year"] <= a.split])))
        print(line(f"FORWARD > {a.split}",
                   stat([x for x in res if x["year"] > a.split])))
        print("\n    per year:")
        for y in sorted({x["year"] for x in res}):
            print(line(f"  {y}", stat([x for x in res if x["year"] == y])))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
