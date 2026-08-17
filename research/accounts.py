"""Account simulation — funded challenges and normal accounts, 2012-2026.

Runs the surviving strategy config through real account mechanics: position
sizing under a lot ceiling, a daily loss limit, a maximum drawdown limit, staged
profit targets, breach-and-restart, and withdrawals.

## Why this is a separate study from the strategy

The 14-year strategy result is +0.074R per trade over 2,854 trades. That number
says nothing about whether a $5,000 funded challenge survives, because a prop
account does not care about expectancy — it cares about the WORST SEQUENCE. The
old tree learned this the expensive way: its v10 config earned +1,101 pips and
drew down -648, which on a $5k account at 0.10 lots is $648 against a $500 total
cap. **Profitable and blown at the same time.** Zero days breached the daily stop;
cumulative drawdown was the killer.

So the questions here are different from "does the strategy work":
    does the account survive to the payout?
    how often does it breach, and how much does restarting cost?
    what does it actually pay per year?

## Contract maths

XAUUSD standard lot = 100 oz, so a $1 price move is $100 per lot. A pip is $0.10,
therefore **1.00 lot = $10 per pip**, 0.10 lot = $1 per pip. Every dollar figure
below derives from that.

## Rules modelled

  * risk-based sizing, capped by the account's lot ceiling
  * daily loss limit stops trading for the rest of that UTC day
  * funded: a max-loss breach fails the account, which then restarts from base
  * funded: staged targets (phase 1, then phase 2), then live
  * withdrawal at a profit threshold, resetting the balance to base
  * normal accounts compound and never fail — they just take the drawdown

Costs are already inside each trade's R (per-bar spread at the funded multiplier),
so no cost is applied twice here.

## Forward test

`--from-year` restricts the trade stream to a period. Run it at 2020 and this
becomes the account-level version of `research/walkforward.py`: what a $5k funded
challenge and a $10k normal account would have done over the years the config was
NOT chosen on. Expectancy surviving out of sample and the ACCOUNT surviving out
of sample are different claims — the old v10 config was profitable and blown at
the same time — so both get asked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The filter set lives in core/strategy.py and nowhere else. It was duplicated
# here — with its own copy of the thresholds — which is precisely how the
# validated config stopped being reproducible from the tree.
from core.strategy import SPECS as STRATS  # noqa: E402
from core.strategy import SWING  # noqa: E402
from research import features as F  # noqa: E402

USD_PER_PIP_PER_LOT = 10.0        # XAUUSD: 100oz lot, $0.10 pip
LOT_STEP = 0.01

FEAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "_feat8.json")


@dataclass
class Spec:
    name: str
    base: float
    max_lot: float
    funded: bool
    risk_pct: float = 1.0
    daily_loss_pct: float = 2.0
    max_loss_pct: float = 10.0        # funded only
    phase1_pct: float = 8.0
    phase2_pct: float = 5.0
    withdraw_pct: float = 20.0
    trailing_dd: bool = False
    fee: float = 0.0                  # cost of each challenge attempt


@dataclass
class Result:
    spec: Spec
    equity: list = field(default_factory=list)
    withdrawn: float = 0.0
    breaches: int = 0
    passes: int = 0                   # full challenges completed (both phases)
    phase_passes: int = 0
    trades: int = 0
    skipped_daily: int = 0
    capped: int = 0                   # trades where the lot ceiling bound
    final: float = 0.0
    max_dd_pct: float = 0.0           # WITHIN a run — resets do not count as loss
    worst_day_pct: float = 0.0
    days_traded: int = 0
    runs: int = 0                     # challenge attempts started


def simulate(trades, spec: Spec) -> Result:
    """Walk the trade stream through one account's rules."""
    r = Result(spec=spec)
    bal = spec.base
    peak = spec.base
    phase = 1 if spec.funded else 0    # 0 = normal/live, 1 = phase1, 2 = phase2
    phase_start = bal
    day = None
    day_start = bal
    day_locked = False
    seen_days = set()

    for t in trades:
        if t["day"] != day:
            day = t["day"]
            day_start = bal
            day_locked = False
            seen_days.add(day)
        if day_locked:
            r.skipped_daily += 1
            continue

        # ── size ────────────────────────────────────────────────────────────
        # risk a fixed % of CURRENT balance on this trade's own stop distance,
        # then clamp to the account's lot ceiling
        risk_usd = bal * spec.risk_pct / 100.0
        stop_pips = float(t["risk_pips"])
        if stop_pips <= 0:
            continue
        lot = risk_usd / (stop_pips * USD_PER_PIP_PER_LOT)
        lot = np.floor(lot / LOT_STEP) * LOT_STEP
        if lot > spec.max_lot:
            lot = spec.max_lot
            r.capped += 1
        if lot < LOT_STEP:
            continue

        pnl = float(t["r"]) * stop_pips * USD_PER_PIP_PER_LOT * lot
        bal += pnl
        r.trades += 1
        r.equity.append(bal)
        peak = max(peak, bal)
        # Drawdown measured WITHIN the current run only. `peak` is reset on every
        # breach, phase pass and withdrawal, so the balance dropping from
        # base+20% back to base at a payout is not counted as a 17% loss — which
        # is what made the first version of this report show funded accounts
        # "drawing down" 23% under a 10% max-loss rule they never actually broke.
        r.max_dd_pct = max(r.max_dd_pct, (peak - bal) / peak * 100.0)

        # ── daily loss limit ────────────────────────────────────────────────
        day_dd = (day_start - bal) / day_start * 100.0
        r.worst_day_pct = max(r.worst_day_pct, day_dd)
        if day_dd >= spec.daily_loss_pct:
            day_locked = True

        # ── funded rules ────────────────────────────────────────────────────
        if spec.funded:
            # Max-loss reference. Firms differ and it matters a great deal:
            #   static   — measured from the INITIAL balance. Profit banked in
            #              the account becomes buffer, so a winning account can
            #              take a big peak-to-valley without breaching.
            #   trailing — measured from the PEAK. Every new high tightens the
            #              floor, so the same drawdown kills you.
            ref = peak if spec.trailing_dd else spec.base
            if (ref - bal) >= spec.base * spec.max_loss_pct / 100.0:
                r.breaches += 1
                r.runs += 1
                bal = spec.base            # buy a new challenge, start over
                peak = day_start = bal
                phase, phase_start = 1, bal
                day_locked = True
                continue
            target = (spec.phase1_pct if phase == 1 else
                      spec.phase2_pct if phase == 2 else None)
            if target is not None and (bal - phase_start) >= phase_start * target / 100.0:
                r.phase_passes += 1
                if phase == 1:
                    phase, phase_start, bal = 2, spec.base, spec.base
                else:
                    r.passes += 1
                    r.runs += 1
                    phase, phase_start, bal = 0, spec.base, spec.base
                peak = day_start = bal
                continue

        # ── withdrawal ──────────────────────────────────────────────────────
        # funded: only once live. normal: the operator asked these be left to
        # compound, so they never withdraw.
        if spec.funded and phase == 0:
            if (bal - spec.base) >= spec.base * spec.withdraw_pct / 100.0:
                r.withdrawn += bal - spec.base
                bal = spec.base
                # `day_start` must move with the balance or the payout drop is
                # scored as an intraday loss — that is what produced a "worst
                # day" of 15.8% under a 2% daily rule in the first run.
                peak = day_start = bal

    r.final = bal
    r.days_traded = len(seen_days)
    return r


def specs(risk_pct: float, trailing: bool = False) -> list:
    """The account set the operator asked for.

    Funded lot ceilings scale with size; normal accounts get double the funded
    ceiling at the same balance, per the operator's instruction.
    """
    # Challenge fees. Every blown account is re-bought, and with 12-26 breaches
    # over 14 years this is not a rounding error — on a trailing-drawdown 5k it
    # is over a third of gross profit. Figures are typical FundingPips-class
    # pricing; override if the operator's are different.
    funded = [("FUNDED 5k", 5_000, 0.10, 40), ("FUNDED 10k", 10_000, 0.20, 70),
              ("FUNDED 25k", 25_000, 0.40, 150), ("FUNDED 50k", 50_000, 0.80, 250),
              ("FUNDED 100k", 100_000, 1.00, 450)]
    normal = [("NORMAL 10k", 10_000, 0.40), ("NORMAL 50k", 50_000, 1.60)]
    out = [Spec(n, b, l, True, risk_pct=risk_pct, trailing_dd=trailing, fee=f)
           for n, b, l, f in funded]
    out += [Spec(n, b, l, False, risk_pct=risk_pct) for n, b, l in normal]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--risk", type=float, default=1.0, help="%% of balance per trade")
    p.add_argument("--target-r", type=float, default=8.0,
                   help="take-profit in R; the 2R default was cutting the tail")
    p.add_argument("--hold-hours", type=int, default=24)
    p.add_argument("--trailing", action="store_true",
                   help="max loss measured from PEAK instead of initial balance")
    p.add_argument("--strict", action="store_true",
                   help="also require H4+H1 bias agreement (fewer, better trades)")
    p.add_argument("--all-trades", action="store_true",
                   help="skip the surviving-config filters (the full population)")
    p.add_argument("--spec", default=None, choices=sorted(STRATS),
                   help="which frozen config from core/strategy.py")
    p.add_argument("--from-year", type=int, default=None,
                   help="forward test: only trade from this year on")
    p.add_argument("--to-year", type=int, default=None)
    p.add_argument("--cache", default=FEAT,
                   help="cached feature set; ignored if the geometry differs")
    a = p.parse_args(argv)

    # The cache was built at 8R / 24h. Any other geometry has to be rebuilt.
    if (a.target_r == 8.0 and a.hold_hours == 24
            and a.cache and os.path.exists(a.cache)):
        print(f"  reading cached candidates from {os.path.basename(a.cache)} ...")
        rows = json.load(open(a.cache))
    else:
        rows = F.build(target_r=a.target_r, hold_hours=a.hold_hours)

    strat = STRATS[a.spec] if a.spec else (
        STRATS["swing-strict"] if a.strict else SWING)
    trades = rows if a.all_trades else strat.select(rows)
    if a.from_year is not None:
        trades = [x for x in trades if x["year"] >= a.from_year]
    if a.to_year is not None:
        trades = [x for x in trades if x["year"] <= a.to_year]
    trades.sort(key=lambda x: x["bar"])
    if not trades:
        raise SystemExit("no trades")

    years = (trades[-1]["day"] - trades[0]["day"]) / 365.25
    r_arr = np.array([x["r"] for x in trades])
    fwd = "  ** FORWARD TEST — config was not chosen on this period **" if \
        a.from_year and a.from_year >= 2020 else ""
    print("\n" + "=" * 112)
    print(f"  BACKTEST {trades[0]['ts'][:10]} .. {trades[-1]['ts'][:10]}   "
          f"{years:.1f} years{fwd}")
    print("=" * 112)
    print(f"  config      : "
          f"{'ALL CANDIDATES' if a.all_trades else strat.name}"
          f"  (no candle-pattern requirement)")
    print(f"  trades      : {len(trades):,}  ({len(trades)/years:.0f}/yr)")
    print(f"  win rate    : {(r_arr > 0.05).mean():.1%}")
    print(f"  expectancy  : {r_arr.mean():+.4f}R    total {r_arr.sum():+.1f}R")
    print(f"  risk/trade  : {a.risk:.2f}% of balance, capped by each account's lot ceiling")
    print(f"  median stop : {np.median([x['risk_pips'] for x in trades]):.0f} pips")

    print("\n" + "=" * 112)
    print(f"  {'account':<13}{'lot cap':>8}{'trades':>8}{'skip':>6}{'capped':>8}"
          f"{'breach':>8}{'phase':>7}{'passes':>8}{'withdrawn':>12}"
          f"{'final':>12}{'DD%':>7}{'wDay%':>7}")
    print("=" * 112)
    for spec in specs(a.risk, a.trailing):
        res = simulate(trades, spec)
        fees = (res.breaches + 1) * spec.fee if spec.funded else 0.0
        tot = res.withdrawn + (res.final - spec.base) - fees
        print(f"  {spec.name:<13}{spec.max_lot:>8.2f}{res.trades:>8}"
              f"{res.skipped_daily:>6}{res.capped:>8}{res.breaches:>8}"
              f"{res.phase_passes:>7}{res.passes:>8}{res.withdrawn:>12,.0f}"
              f"{res.final:>12,.0f}{res.max_dd_pct:>7.1f}{res.worst_day_pct:>7.1f}")
        if spec.funded:
            print(f"  {'':13}-> {res.passes} full challenge(s) passed, "
                  f"{res.breaches} blown, net ${tot:,.0f} over {years:.1f}y "
                  f"(${tot/years:,.0f}/yr)  after ${fees:,.0f} in fees")
        else:
            growth = (res.final / spec.base - 1) * 100
            cagr = ((res.final / spec.base) ** (1 / years) - 1) * 100 if years else 0
            print(f"  {'':13}-> compounded {growth:+,.0f}%  "
                  f"({cagr:+.1f}%/yr), never reset")
    print("\n  Daily-loss stops skipped trades; 'capped' counts trades where the")
    print("  lot ceiling bound before the risk %% did. Funded accounts restart at")
    print("  base on a max-loss breach and after each payout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
