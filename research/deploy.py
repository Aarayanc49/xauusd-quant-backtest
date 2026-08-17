"""What each strategy actually does to an account — backtest and forward test.

Expectancy and account survival are different claims. The old v10 config earned
+1,101 pips and drew down -648, which on a $5k account is blown while showing a
profit. So every strategy that survives the R-level checks has to come through
here before it means anything in dollars.

## The correction this module exists to make

`research/model.py` reports the top 50% of the swing universe at 1,275 trades/yr
and +0.146R, which multiplies out to ~186R/yr against the hand-picked config's
~85R/yr. That number is arithmetic, not money, because **those trades overlap**.
The swing config holds for up to 24 hours; a real account cannot carry 1,275
overlapping positions a year at 1% risk each.

So `--one-at-a-time` (the default) walks the trade stream in time order and skips
any entry that arrives while a position is still open — which is how the account
will actually trade. The difference between the two modes is the honest cost of
the overlap, and it is reported for both strategies rather than only the new one.
The previously published SWING account figures did NOT enforce this either, so
they are re-run here on the same footing.

## Backtest vs forward test

    backtest       2012-2026, everything. For the model this is PARTLY
                   IN-SAMPLE — the bins and the threshold were fitted on
                   2012-2019 — so it is a description, not evidence.
    forward test   2020-2026 only, with the model frozen on 2019 and earlier
                   and the threshold set on that period. This is the number
                   that counts.

    python -m research.deploy --strategy both
    python -m research.deploy --strategy model --pctl 0.5 --overlap
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy import SWING, SWING_STRICT  # noqa: E402
from research import accounts as AC  # noqa: E402
from research import model as MD  # noqa: E402

FEAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "_feat8.json")
BARS_PER_M5 = 5          # the cache simulated on M1, so `bars` is minutes


def position_rule(trades, mode="one"):
    """Filter a trade stream down to what an account could actually have held.

    `bars` is the trade's length in M1 bars (minutes); `bar` is the M5 index of
    the entry, so the exit lands at bar + bars/5.

      one           strictly one position at a time. Right for a 24h-hold swing
                    book, where a second entry is usually the same idea again.
      no_opposite   concurrent positions allowed, but never in opposite
                    directions. Right for a day-trade book: short holds do not
                    monopolise the slot the way an overnight position does, and
                    the thing actually worth forbidding is paying spread twice
                    to hold a flat net position against yourself.
      overlap       no limit. Unrealistic; kept for contrast, because the gap
                    between this and the others is the honest cost of the rule.
    """
    if mode == "overlap":
        return sorted(trades, key=lambda x: x["bar"])
    out, open_pos = [], []
    for t in sorted(trades, key=lambda x: x["bar"]):
        b = int(t["bar"])
        open_pos = [p for p in open_pos if p[0] > b]
        if mode == "one":
            if open_pos:
                continue
        elif any(p[1] != t["direction"] for p in open_pos):
            continue
        out.append(t)
        open_pos.append((b + math.ceil(int(t.get("bars", 1)) / BARS_PER_M5),
                         t["direction"]))
    return out


def one_at_a_time(trades):
    return position_rule(trades, "one")


def model_trades(rows, split=2019, pctl=0.5, bins=8, max_features=15):
    """Fit the evidence model on <= split, return (scores, threshold, keys)."""
    keys, X, y, bar, yr = MD.to_matrix(rows)
    is_m = yr <= split
    bar_is = bar[is_m]
    mid = np.median(bar_is)
    a_m = is_m & (bar <= mid)
    b_m = is_m & (bar > mid)
    chosen, _e, _l, trail, _s = MD.select(
        X[a_m], y[a_m], X[b_m], y[b_m], keys, bins, max_features, 0.90)
    if not chosen:
        raise SystemExit("model selected no features")
    edges, lifts, _ = MD.fit_bins(X[is_m], y[is_m], keys, bins)
    score = MD.score_with(X, edges, lifts, chosen)
    cut = float(np.quantile(score[is_m], pctl))
    return score, cut, [keys[j] for j in chosen], trail


def summarise(trades, label):
    if not trades:
        print(f"  {label:<28} no trades")
        return None
    r = np.array([x["r"] for x in trades])
    years = (trades[-1]["day"] - trades[0]["day"]) / 365.25
    print(f"  {label:<28} n={len(r):>5} ({len(r)/max(years,1e-9):>5.0f}/yr)  "
          f"win={(r > 0.05).mean():>5.1%}  avg={r.mean():>+6.3f}R  "
          f"total={r.sum():>+7.0f}R  ({r.sum()/max(years,1e-9):>+5.0f}R/yr)")
    return r


def run_accounts(trades, risk, trailing, title):
    print(f"\n  {title}")
    print(f"  {'account':<13}{'trades':>8}{'skip':>6}{'capped':>8}{'breach':>8}"
          f"{'passes':>8}{'withdrawn':>12}{'final':>12}{'DD%':>7}")
    years = (trades[-1]["day"] - trades[0]["day"]) / 365.25
    for spec in AC.specs(risk, trailing):
        res = AC.simulate(trades, spec)
        fees = (res.breaches + 1) * spec.fee if spec.funded else 0.0
        tot = res.withdrawn + (res.final - spec.base) - fees
        print(f"  {spec.name:<13}{res.trades:>8}{res.skipped_daily:>6}"
              f"{res.capped:>8}{res.breaches:>8}{res.passes:>8}"
              f"{res.withdrawn:>12,.0f}{res.final:>12,.0f}{res.max_dd_pct:>7.1f}")
        if spec.funded:
            print(f"  {'':13}-> {res.passes} passed, {res.breaches} blown, "
                  f"net ${tot:,.0f} over {years:.1f}y  (${tot/max(years,1e-9):,.0f}/yr)")
        else:
            cagr = ((res.final / spec.base) ** (1 / max(years, 1e-9)) - 1) * 100
            print(f"  {'':13}-> compounded {(res.final/spec.base-1)*100:+,.0f}%  "
                  f"({cagr:+.1f}%/yr) at {res.max_dd_pct:.1f}% max DD")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="both",
                   choices=("swing", "strict", "model", "both", "daytrade"))
    p.add_argument("--rule", default="one",
                   choices=("one", "no_opposite", "overlap"))
    p.add_argument("--dt-hold", type=int, default=480)
    p.add_argument("--dt-target", type=float, default=8.0)
    p.add_argument("--pctl", type=float, default=0.50)
    p.add_argument("--split", type=int, default=2019)
    p.add_argument("--risk", type=float, default=1.0)
    p.add_argument("--trailing", action="store_true")
    p.add_argument("--overlap", action="store_true",
                   help="allow concurrent positions (unrealistic; for contrast)")
    p.add_argument("--feat", default=FEAT)
    a = p.parse_args(argv)

    import json
    rows = json.load(open(a.feat))
    print("=" * 112)
    rule = "overlap" if a.overlap else a.rule
    RULE_LABEL = {"one": "ONE POSITION AT A TIME",
                  "no_opposite": "NO OPPOSITE POSITIONS",
                  "overlap": "OVERLAP ALLOWED"}
    print(f"  DEPLOYMENT — {len(rows):,} candidates, risk {a.risk}%/trade, "
          f"{RULE_LABEL[rule]}")
    print("=" * 112)

    sets = {}
    if a.strategy == "daytrade":
        from research import daytrade as DT
        book = DT.Book(feat=a.feat)
        sel = DT.run(book, stop_atr=1.5, target_r=a.dt_target,
                     max_hold=a.dt_hold, rows=SWING.select(rows))
        sets[f"DAY TRADER {a.dt_hold}m/{a.dt_target:g}R"] = sel
    if a.strategy in ("swing", "both"):
        sets["SWING"] = SWING.select(rows)
    if a.strategy in ("strict", "both"):
        sets["SWING-STRICT"] = SWING_STRICT.select(rows)
    if a.strategy in ("model", "both"):
        score, cut, feats, trail = model_trades(rows, a.split, a.pctl)
        sets[f"MODEL top {1-a.pctl:.0%}"] = [x for x, s in zip(rows, score)
                                             if s >= cut]
        print(f"\n  model frozen on <= {a.split}; {len(feats)} features, "
              f"threshold {cut:+.3f}")
        print("    " + ", ".join(feats[:8])
              + (" ..." if len(feats) > 8 else ""))

    for name, sel in sets.items():
        print("\n" + "=" * 112)
        print(f"  {name}")
        print("=" * 112)
        for period, flt in (("BACKTEST 2012-2026", lambda x: True),
                            ("FORWARD  2020-2026", lambda x: x["year"] > a.split)):
            t = position_rule([x for x in sel if flt(x)], rule)
            if not t:
                continue
            print()
            summarise(t, period)
            run_accounts(t, a.risk, a.trailing, period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
