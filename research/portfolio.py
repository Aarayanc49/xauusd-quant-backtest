"""Multi-symbol account simulation — one book, six instruments, real constraints.

`research/accounts.py` walks a single symbol's trade stream with a flat lot
ceiling. That is not what a funded account does. This runs every symbol's day
trader simultaneously through one account and enforces the things that actually
decide whether it survives:

    position sizing     risk % of CURRENT balance on THIS trade's stop, in the
                        instrument's own pip value
    per-symbol lot cap  gold is capped at 0.10 on the operator's account
    margin              a position is refused if free margin cannot carry it
    daily loss limit    breached on the UTC day, stops trading for that day
    max loss limit      breached against the initial balance, fails the account
    no opposite         never long and short the same symbol at once

## Why floating loss is modelled, not just closed P&L

A prop firm checks its drawdown rules against EQUITY, which includes open
positions. A book holding six correlated instruments can be well inside its
limits on closed trades and still breach on floating loss — and gold, EURUSD,
GBPUSD and the two indices are not independent. `accounts.py` measures closed
balance only and therefore understates breaches.

Each trade carries `mae_r`, its worst excursion in R, so the floating loss of the
open book is bounded by the sum of `mae_r x risk_usd` over open positions. That
is a WORST CASE — it assumes every position hits its own low simultaneously,
which is pessimistic — so both figures are reported: breaches on closed balance,
and breaches on worst-case equity. The truth is between them and the gap is a
statement about how correlated the book is.

## Costs

Spread is already inside each trade's R at the funded multiplier. Nothing is
charged twice here. Swap is not modelled — the day trader is flat by 21:00 UTC
every day and never pays it.

    python -m research.portfolio --from-year 2020
    python -m research.portfolio --balance 10000 --risk 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.contracts import CONTRACTS, caps_for, stepped_caps  # noqa: E402
from research.buildbooks import SYMBOLS, path_for  # noqa: E402

# Positions are tracked in EPOCH SECONDS, never in bar indices — a bar index is
# a per-symbol position and comparing gold's bar 40,000 with EURUSD's would
# interleave different years into one imaginary account.
MARGIN_UTIL = 0.50      # never commit more than half of free margin to new risk


@dataclass
class Rules:
    base: float = 10_000.0
    risk_pct: float = 1.0
    daily_loss_pct: float = 2.0
    max_loss_pct: float = 10.0
    phase1_pct: float = 8.0
    phase2_pct: float = 5.0
    withdraw_pct: float = 20.0
    funded: bool = True
    trailing_dd: bool = False
    fee: float = 70.0        # a 10k challenge re-buy
    lot_caps: dict = field(default_factory=dict)
    leverage: float = 30.0
    # Lot ceilings scale with the CURRENT balance rather than the opening one.
    # Fixing them at the starting balance is what made the first compounding run
    # misleading: 3,454 of 3,596 trades hit a cap as the account grew, so
    # position size stopped tracking equity and the reported CAGR was really an
    # artefact of the early years. A live account's size limits move with it.
    dynamic_caps: bool = True
    # Stepped sizing: raise the ceiling by `lot_step` once per `lot_step_per`
    # dollars of growth, instead of scaling continuously.
    stepped: bool = False
    lot_step: float = 0.05
    lot_step_per: float = 5_000.0
    # Normal accounts: take income out at a threshold rather than compounding
    # without limit. withdraw_at=20000, withdraw_amount=5000 means "every time
    # the balance reaches $20k, take $5k off the table".
    withdraw_at: float = 0.0
    withdraw_amount: float = 0.0
    # Portfolio-level margin ceiling. Without this the book stacked 13 concurrent
    # positions to 99.5% of balance at 1:30 — a margin call one adverse tick
    # away. Per-trade limits alone do not bound the BOOK.
    max_margin_pct: float = 40.0


@dataclass
class Out:
    trades: int = 0
    skipped_daily: int = 0
    skipped_margin: int = 0
    skipped_opposite: int = 0
    skipped_lot: int = 0
    capped: int = 0
    breaches: int = 0
    breaches_equity: int = 0
    passes: int = 0
    phase_passes: int = 0
    withdrawn: float = 0.0
    final: float = 0.0
    max_dd_pct: float = 0.0
    max_dd_equity_pct: float = 0.0
    peak_margin_pct: float = 0.0
    max_concurrent: int = 0
    by_symbol: dict = field(default_factory=lambda: defaultdict(
        lambda: {"n": 0, "pnl": 0.0, "win": 0, "r": 0.0}))
    by_year: dict = field(default_factory=lambda: defaultdict(
        lambda: {"n": 0, "pnl": 0.0, "win": 0, "start": 0.0, "end": 0.0,
                 "peak": 0.0, "dd": 0.0, "skipped": 0}))


def simulate(trades, rules: Rules) -> Out:
    """Walk one merged, time-ordered trade stream through one account."""
    o = Out()
    bal = peak = rules.base
    phase = 1 if rules.funded else 0
    phase_start = bal
    day = None
    day_start = bal
    day_locked = False
    open_pos = []      # (exit_bar, symbol, direction, risk_usd, mae_r, margin)

    for t in sorted(trades, key=lambda x: (x["bar_utc"], x["symbol"])):
        b = int(t["bar_utc"])                    # epoch seconds
        open_pos = [p for p in open_pos if p[0] > b]

        if t["day"] != day:
            day, day_start, day_locked = t["day"], bal, False
        if day_locked:
            o.skipped_daily += 1
            continue

        sym = t["symbol"]
        c = CONTRACTS.get(sym)
        if c is None:
            continue
        if any(p[1] == sym and p[2] != t["direction"] for p in open_pos):
            o.skipped_opposite += 1
            continue

        px = float(t["entry_price"])
        per_pip = c.usd_per_pip(px)
        stop_pips = float(t["risk_pips"])
        if stop_pips <= 0 or per_pip <= 0:
            continue

        # ── size ────────────────────────────────────────────────────────────
        risk_usd = bal * rules.risk_pct / 100.0
        lot = risk_usd / (stop_pips * per_pip)
        if rules.stepped:
            caps = stepped_caps(bal, rules.base, rules.lot_step,
                                rules.lot_step_per)
        elif rules.dynamic_caps:
            caps = caps_for(bal)
        else:
            caps = rules.lot_caps
        cap = caps.get(sym, 0.10)
        if lot > cap:
            lot, o.capped = cap, o.capped + 1

        # ── margin, bounded at the BOOK level, not just per trade ───────────
        used = sum(p[5] for p in open_pos)
        ceiling = bal * rules.max_margin_pct / 100.0
        room = max(0.0, ceiling - used)
        if room <= 0:
            o.skipped_margin += 1
            continue
        mpl = c.margin_per_lot(px, rules.leverage)
        max_by_margin = min(room, max(0.0, bal - used) * MARGIN_UTIL) / max(mpl, 1e-9)
        if lot > max_by_margin:
            lot = max_by_margin
        lot = math.floor(lot / c.lot_step) * c.lot_step
        if lot < c.min_lot:
            o.skipped_margin += 1
            continue

        margin = lot * mpl
        real_risk = lot * stop_pips * per_pip
        pnl = float(t["r"]) * stop_pips * per_pip * lot

        bal += pnl
        o.trades += 1
        o.by_symbol[sym]["n"] += 1
        o.by_symbol[sym]["pnl"] += pnl
        o.by_symbol[sym]["win"] += 1 if float(t["r"]) > 0.05 else 0
        o.by_symbol[sym]["r"] += float(t["r"])

        y = int(t["year"])
        yr_row = o.by_year[y]
        if yr_row["n"] == 0:
            yr_row["start"] = bal - pnl
            yr_row["peak"] = bal - pnl
        yr_row["n"] += 1
        yr_row["pnl"] += pnl
        yr_row["win"] += 1 if float(t["r"]) > 0.05 else 0
        yr_row["end"] = bal
        yr_row["peak"] = max(yr_row["peak"], bal)
        yr_row["dd"] = max(yr_row["dd"],
                           (yr_row["peak"] - bal) / max(yr_row["peak"], 1e-9) * 100.0)

        peak = max(peak, bal)
        o.max_dd_pct = max(o.max_dd_pct, (peak - bal) / peak * 100.0)
        o.peak_margin_pct = max(o.peak_margin_pct,
                                (used + margin) / max(bal, 1e-9) * 100.0)

        # `bars` is the trade's length in M1 bars, i.e. minutes
        open_pos.append((b + int(t.get("bars", 1)) * 60,
                         sym, t["direction"], real_risk,
                         abs(float(t.get("mae_r", 0.0))), margin))
        o.max_concurrent = max(o.max_concurrent, len(open_pos))

        # ── worst-case equity, including floating loss on the open book ─────
        floating = sum(p[3] * p[4] for p in open_pos)
        equity_wc = bal - floating
        o.max_dd_equity_pct = max(o.max_dd_equity_pct,
                                  (peak - equity_wc) / peak * 100.0)

        # ── daily loss ──────────────────────────────────────────────────────
        if (day_start - bal) / day_start * 100.0 >= rules.daily_loss_pct:
            day_locked = True

        # ── max loss ────────────────────────────────────────────────────────
        if rules.funded:
            ref = peak if rules.trailing_dd else rules.base
            hard = rules.base * rules.max_loss_pct / 100.0
            if (ref - equity_wc) >= hard:
                o.breaches_equity += 1
            if (ref - bal) >= hard:
                o.breaches += 1
                bal = rules.base
                peak = day_start = bal
                phase, phase_start = 1, bal
                day_locked = True
                open_pos = []
                continue
            tgt = (rules.phase1_pct if phase == 1 else
                   rules.phase2_pct if phase == 2 else None)
            if tgt is not None and (bal - phase_start) >= phase_start * tgt / 100.0:
                o.phase_passes += 1
                if phase == 1:
                    phase, phase_start, bal = 2, rules.base, rules.base
                else:
                    o.passes += 1
                    phase, phase_start, bal = 0, rules.base, rules.base
                peak = day_start = bal
                open_pos = []
                continue
            if phase == 0 and (bal - rules.base) >= rules.base * rules.withdraw_pct / 100.0:
                o.withdrawn += bal - rules.base
                bal = rules.base
                peak = day_start = bal
        elif rules.withdraw_at and bal >= rules.withdraw_at:
            # Normal account taking income: at the threshold, pull a fixed
            # amount off the table. `peak` and `day_start` must move with the
            # balance or the withdrawal is scored as a drawdown and a daily loss.
            take = min(rules.withdraw_amount, bal)
            o.withdrawn += take
            bal -= take
            # Every high-water mark must step down with the balance, or taking
            # money out is scored as losing it. Missing this on the per-year
            # peak produced yearly drawdowns of 42% against a true figure of
            # 24% — the same fault accounts.py documents for funded payouts.
            peak = min(peak, bal)
            day_start = min(day_start, bal)
            yr_row["peak"] = min(yr_row["peak"], bal)
            yr_row["withdrawn"] = yr_row.get("withdrawn", 0.0) + take

    o.final = bal
    return o


def load_books(symbols, hold, target_r, stop_atr, from_year, to_year):
    """Re-simulate each symbol's cached candidates under day-trade geometry."""
    from core.strategy import SWING
    from research import daytrade as DT

    all_rows = []
    for s in symbols:
        p = path_for(s)
        if not os.path.exists(p) and s == "XAUUSD":
            # gold's book predates the per-symbol naming and is the same thing
            alt = os.path.join(os.path.dirname(p), "_feat8.json")
            if os.path.exists(alt):
                p = alt
        if not os.path.exists(p):
            print(f"  {s:<8} no book — run research.buildbooks")
            continue
        try:
            book = DT.Book(symbol=s, feat=p)
        except Exception as e:
            print(f"  {s:<8} skipped: {e}")
            continue
        sel = SWING.select(book.rows)
        res = DT.run(book, stop_atr=stop_atr, target_r=target_r,
                     max_hold=hold, rows=sel)
        for r in res:
            r["symbol"] = s
            if from_year and r["year"] < from_year:
                continue
            if to_year and r["year"] > to_year:
                continue
            all_rows.append(r)
        n = sum(1 for r in res if not from_year or r["year"] >= from_year)
        print(f"  {s:<8} {len(book.rows):>7,} cands -> {len(sel):>6,} filtered "
              f"-> {n:>5,} in window")
    return all_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--balance", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--from-year", type=int, default=2020)
    ap.add_argument("--to-year", type=int, default=None)
    ap.add_argument("--hold", type=int, default=480)
    ap.add_argument("--target-r", type=float, default=8.0)
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--trailing", action="store_true")
    ap.add_argument("--normal", action="store_true", help="no funded rules")
    ap.add_argument("--leverage", type=float, default=30.0,
                    help="account leverage; the operator's real accounts are 1:30")
    ap.add_argument("--static-caps", action="store_true",
                    help="freeze lot ceilings at the opening balance")
    ap.add_argument("--stepped", action="store_true",
                    help="raise lot ceilings in fixed increments as it grows")
    ap.add_argument("--lot-step", type=float, default=0.05)
    ap.add_argument("--lot-step-per", type=float, default=5_000.0)
    ap.add_argument("--withdraw-at", type=float, default=0.0,
                    help="normal accounts: withdraw when balance reaches this")
    ap.add_argument("--withdraw-amount", type=float, default=0.0)
    ap.add_argument("--max-margin", type=float, default=40.0,
                    help="portfolio margin ceiling, %% of balance")
    a = ap.parse_args(argv)

    print("=" * 112)
    print(f"  PORTFOLIO — day trader, {a.hold}m max hold flat by 21:00, "
          f"{a.stop_atr} ATR stop, {a.target_r}R target")
    print("=" * 112)
    rows = load_books(a.symbols, a.hold, a.target_r, a.stop_atr,
                      a.from_year, a.to_year)
    if not rows:
        raise SystemExit("no trades")
    rows.sort(key=lambda x: x["bar_utc"])
    yrs = (rows[-1]["day"] - rows[0]["day"]) / 365.25
    r = np.array([x["r"] for x in rows])
    print(f"\n  merged book: {len(rows):,} trades over {yrs:.1f}y "
          f"({len(rows)/yrs:,.0f}/yr)  win={(r > 0.05).mean():.1%}  "
          f"avg={r.mean():+.3f}R")

    rules = Rules(base=a.balance, risk_pct=a.risk, funded=not a.normal,
                  trailing_dd=a.trailing, lot_caps=caps_for(a.balance),
                  leverage=a.leverage, dynamic_caps=not a.static_caps,
                  stepped=a.stepped, lot_step=a.lot_step,
                  lot_step_per=a.lot_step_per,
                  withdraw_at=a.withdraw_at, withdraw_amount=a.withdraw_amount,
                  max_margin_pct=a.max_margin)
    sizing = (f"stepped +{a.lot_step:.2f} lots per ${a.lot_step_per:,.0f}"
              if a.stepped else
              "frozen at open" if a.static_caps else "scale with balance")
    print(f"  leverage 1:{a.leverage:g}   sizing: {sizing}   "
          f"portfolio margin ceiling {a.max_margin:g}%")
    if a.withdraw_at:
        print(f"  withdraw ${a.withdraw_amount:,.0f} whenever balance reaches "
              f"${a.withdraw_at:,.0f}")
    o = simulate(rows, rules)
    fees = (o.breaches + 1) * rules.fee if rules.funded else 0.0
    tot = o.withdrawn + (o.final - rules.base) - fees

    print("\n" + "=" * 112)
    print(f"  {'FUNDED' if rules.funded else 'NORMAL'} ${a.balance:,.0f}   "
          f"risk {a.risk}%/trade   {a.from_year}-{a.to_year or 2026}")
    print("=" * 112)
    print(f"  trades taken        {o.trades:>8,}   of {len(rows):,} signals")
    print(f"  skipped: daily-lock {o.skipped_daily:>8,}   "
          f"margin {o.skipped_margin:,}   opposite {o.skipped_opposite:,}")
    print(f"  lot-capped          {o.capped:>8,}   "
          f"max concurrent positions {o.max_concurrent}")
    print(f"  peak margin used    {o.peak_margin_pct:>7.1f}% of balance")
    print(f"  max DD (closed)     {o.max_dd_pct:>7.1f}%")
    print(f"  max DD (worst-case equity, incl. floating) "
          f"{o.max_dd_equity_pct:>6.1f}%")
    if rules.funded:
        print(f"  breaches (closed)   {o.breaches:>8}   "
              f"(worst-case equity would breach {o.breaches_equity})")
        print(f"  phases passed       {o.phase_passes:>8}   "
              f"full challenges {o.passes}")
        print(f"  withdrawn           ${o.withdrawn:>10,.0f}")
        print(f"  NET                 ${tot:>10,.0f}  over {yrs:.1f}y  "
              f"= ${tot/yrs:,.0f}/yr   after ${fees:,.0f} fees")
    else:
        total = o.final + o.withdrawn
        cagr = ((max(total, 1e-9) / rules.base) ** (1 / yrs) - 1) * 100
        print(f"  withdrawn           ${o.withdrawn:>10,.0f}  "
              f"(${o.withdrawn/yrs:,.0f}/yr income)")
        print(f"  final balance       ${o.final:>10,.0f}")
        print(f"  TOTAL (bal+drawn)   ${total:>10,.0f}  "
              f"({(total/rules.base-1)*100:+,.0f}%, {cagr:+.1f}%/yr)")

    print(f"\n  {'symbol':<9}{'trades':>8}{'win':>7}{'avg R':>8}"
          f"{'net $':>12}{'$/yr':>10}{'share':>8}")
    tot_pnl = sum(v["pnl"] for v in o.by_symbol.values()) or 1.0
    for s, v in sorted(o.by_symbol.items(), key=lambda kv: -kv[1]["pnl"]):
        print(f"  {s:<9}{v['n']:>8,}{v['win']/max(v['n'],1):>7.1%}"
              f"{v['r']/max(v['n'],1):>+8.3f}{v['pnl']:>12,.0f}"
              f"{v['pnl']/yrs:>10,.0f}{v['pnl']/tot_pnl:>8.1%}")

    print(f"\n  YEAR BY YEAR")
    print(f"  {'year':<7}{'trades':>8}{'win':>7}{'start $':>11}{'end $':>11}"
          f"{'P&L $':>11}{'drawn $':>10}{'return':>9}{'maxDD':>8}")
    for y in sorted(o.by_year):
        v = o.by_year[y]
        drawn = v.get("withdrawn", 0.0)
        # return counts money taken out, or a year that paid you looks flat
        ret = ((v["end"] + drawn) / v["start"] - 1) * 100 if v["start"] else 0.0
        print(f"  {y:<7}{v['n']:>8,}{v['win']/max(v['n'],1):>7.1%}"
              f"{v['start']:>11,.0f}{v['end']:>11,.0f}{v['pnl']:>11,.0f}"
              f"{drawn:>10,.0f}{ret:>+9.1f}%{v['dd']:>7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
