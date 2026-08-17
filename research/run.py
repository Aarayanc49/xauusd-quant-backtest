"""Drive the engine.

    python -m research.run                       # default config, full window
    python -m research.run --ladder legacy       # reproduce the old exit
    python -m research.run --compare             # every ladder, side by side
    python -m research.run --spread 1.0          # raw feed spread, not funded

`--compare` is the point of the whole rewrite: it runs the same entries through
different exits and prints the arithmetic for each. On the old harness that was
half a day per variant.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.context import Context  # noqa: E402
from core.discover import build_points, load_series  # noqa: E402
from core.exits import LEGACY_LADDER, NO_TRAIL, Ladder  # noqa: E402
from research import engine, report  # noqa: E402

LADDERS = {
    "default": Ladder(),
    "legacy": LEGACY_LADDER,
    "none": NO_TRAIL,
    # deliberately late: nothing until the trade is clear of the losers' median
    # MAE (0.95R), then trail wide
    "late": Ladder(((1.00, 0.00), (1.50, 0.60), (2.20, 1.30), (3.20, 2.20)),
                   name="late"),
    # tight-but-legal: arms at the winners' p90 MAE rather than p75
    "p90": Ladder(((0.80, 0.00), (1.10, 0.40), (1.50, 0.80),
                   (2.00, 1.25), (3.00, 2.10)), name="p90"),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--ladder", default="default", choices=sorted(LADDERS))
    p.add_argument("--spread", type=float, default=engine.FUNDED_SPREAD_MULT,
                   help="multiplier on the feed's own spread column")
    p.add_argument("--target-r", type=float, default=None)
    p.add_argument("--cooldown", type=int, default=12)
    p.add_argument("--min-depth-atr", type=float, default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--brief", action="store_true")
    a = p.parse_args(argv)

    series = load_series(a.symbol)
    if not series:
        raise SystemExit(f"no data for {a.symbol} — run research.fetch first")
    ctx = Context(series, base="M5")
    print(f"building levels for {a.symbol} ...", flush=True)
    pts = build_points(series, ctx, "M5")
    print(f"  {len(pts):,} level points\n", flush=True)
    pre = {"ctx": ctx, "points": pts}

    def cfg_for(name):
        return engine.Config(
            symbol=a.symbol, start=a.start, end=a.end,
            spread_mult=a.spread, ladder=LADDERS[name],
            target_r=a.target_r, cooldown_bars=a.cooldown,
            min_depth_atr=a.min_depth_atr,
            # A ladder changes how long trades last, so under one-position-at-a-
            # time each variant SELECTS A DIFFERENT TRADE SET and the comparison
            # silently stops being controlled. For the exit sweep every event
            # becomes a trade, so the entries are genuinely identical and only the
            # exit differs. Overlapping positions are not how the account will
            # trade — that is what the single-ladder run is for.
            one_at_a_time=not a.compare)

    names = sorted(LADDERS) if a.compare else [a.ladder]
    results = {}
    for name in names:
        res = engine.run(cfg_for(name), series=series, prebuilt=pre)
        results[name] = res
        if not a.compare:
            print(report.summary(res, f"{a.symbol}  ladder={name}",
                                 detail=not a.brief))
            print(report.split_halves(res))

    if a.compare:
        print("=" * 92)
        print("  SAME ENTRIES, DIFFERENT EXITS   (overlap allowed, so n is identical)")
        print("=" * 92)
        print(f"  {'ladder':<10} {'n':>6} {'win':>7} {'payoff':>7} {'need':>7} "
              f"{'edge':>7} {'avg R':>8} {'total':>9} {'maxDD':>8} {'ret/DD':>7}")
        for name, res in results.items():
            a_ = report.arithmetic(res.trades)
            if not a_:
                print(f"  {name:<10}  no trades")
                continue
            dd, final, _ = report.drawdown(res.trades)
            print(f"  {name:<10} {a_['n']:>6} {a_['win_rate']:>6.1%} "
                  f"{a_['payoff']:>7.2f} {a_['need']:>7.2f} {a_['edge']:>+7.2f} "
                  f"{a_['avg_r']:>+8.3f} {a_['sum_r']:>+9.1f} {dd:>+8.1f} "
                  f"{(final/abs(dd)) if dd else 0:>7.2f}")
        secs = sum(r.seconds for r in results.values())
        print(f"\n  {len(results)} full backtests in {secs:.1f}s "
              f"— the old harness needed ~150 min for ONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
