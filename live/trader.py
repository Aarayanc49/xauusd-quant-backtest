"""The autonomous day trader.

Runs the validated config live:

    entry     structure break -> impulse leg -> 0.5-0.618 retrace
    filters   range_pct >= 0.75, spread_pct < 0.5, london/ny/overlap,
              h4_pullback >= 0.4                      (core/strategy.SWING)
    stop      1.5 x ATR(M15)          target  8R
    hold      max 480 min, ALWAYS flat by 21:00 UTC
    rules     no opposite positions per symbol, portfolio margin ceiling,
              daily loss limit, stepped lot sizing

## What this run is for

Two weeks is ~29 trades at this trade rate. That is far too small to say
anything about edge — the confidence interval on a 36% win rate over 29 trades
covers everything from broken to excellent. It is not an edge test.

It IS the only way to test the things a backtest cannot:

  * does a signal fire live at the same bar the backtest says it should
  * do fills match the modelled spread, or is there slippage the cost model
    never charged
  * does sizing, margin and the flat-by-close rule behave under real conditions
  * does anything crash at 03:00 on a Tuesday

Everything needed to answer those is journalled — including every signal that was
REJECTED and why, because a live run that silently stops taking trades looks
identical to a quiet market.

    python -m live.trader                  # run it
    python -m live.trader --dry-run        # decide and journal, place nothing
    python -m live.trader --once           # single pass, for checking
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.contracts import CONTRACTS, stepped_caps  # noqa: E402
from core.strategy import SWING  # noqa: E402
from live import signals as SIG  # noqa: E402
from live.broker import Broker  # noqa: E402

SYMBOLS = ["XAUUSD", "USDJPY", "EURUSD", "GBPUSD", "USTEC", "US500"]

STOP_ATR = 1.5
TARGET_R = 8.0
MAX_HOLD_MIN = 480
FLAT_BY_HOUR = 21          # UTC
TRADE_FROM_HOUR = 7
RISK_PCT = 1.0
MAX_MARGIN_PCT = 40.0
DAILY_LOSS_PCT = 2.0
MAX_CONCURRENT = 8

JOURNAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "live_journal.jsonl")


def utcnow():
    return datetime.now(timezone.utc)


def log(rec: dict, path=JOURNAL):
    rec["t"] = utcnow().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


class Trader:
    def __init__(self, broker, dry_run=False, risk_pct=RISK_PCT):
        self.b = broker
        self.dry = dry_run
        self.risk_pct = risk_pct
        self.seen = set()          # (symbol, bar_time) already acted on
        self.day = None
        self.day_start_balance = None
        self.day_locked = False
        self.start_balance = broker.account().balance
        self._state = None         # last journalled loop state, for _note

    def _note(self, state: str, **extra):
        """Journal a loop state, but only when it CHANGES.

        Every `return` in `pass_once` used to be silent, so a loop that spent
        six hours outside its trading window and a loop that had crashed wrote
        exactly the same thing: nothing. That is the single hardest failure to
        notice, because the expected output of a quiet day is also nothing.

        Logging it on every pass would bury the signals under thousands of
        heartbeat lines, so only transitions are written. The journal then reads
        as a state timeline — `window_closed` at 21:00, `scanning` at 07:00 —
        and a gap in it means the process itself stopped.
        """
        if state == self._state:
            return
        self._state = state
        log({"ev": "state", "state": state, **extra})

    # ── risk gates ──────────────────────────────────────────────────────────

    def _roll_day(self, acc):
        d = utcnow().date().isoformat()
        if d != self.day:
            self.day = d
            self.day_start_balance = acc.equity
            self.day_locked = False
            log({"ev": "day_start", "day": d, "equity": acc.equity})

    def _daily_locked(self, acc) -> bool:
        if self.day_locked:
            return True
        if self.day_start_balance:
            dd = (self.day_start_balance - acc.equity) / self.day_start_balance * 100
            if dd >= DAILY_LOSS_PCT:
                self.day_locked = True
                log({"ev": "daily_lock", "drawdown_pct": round(dd, 2),
                     "equity": acc.equity})
                return True
        return False

    def _margin_used_pct(self, acc) -> float:
        return (acc.margin / acc.equity * 100.0) if acc.equity else 100.0

    # ── sizing ──────────────────────────────────────────────────────────────

    def size(self, sym, acc, stop_pips, price) -> tuple:
        """(lot, reason). Risk %, clamped by stepped cap, then by margin room."""
        c = CONTRACTS[sym]
        per_pip = c.usd_per_pip(price)
        if per_pip <= 0 or stop_pips <= 0:
            return 0.0, "bad_pip_value"
        risk_usd = acc.equity * self.risk_pct / 100.0
        lot = risk_usd / (stop_pips * per_pip)

        caps = stepped_caps(acc.equity, base=10_000.0)
        cap = caps.get(sym, 0.10)
        capped = lot > cap
        lot = min(lot, cap)

        ceiling = acc.equity * MAX_MARGIN_PCT / 100.0
        room = ceiling - acc.margin
        if room <= 0:
            return 0.0, "margin_ceiling"
        mpl = self.b.margin_for(sym, 1.0, True, price)
        if mpl and mpl > 0:
            lot = min(lot, room / mpl)

        info = self.b.mt5.symbol_info(sym)
        step = info.volume_step or 0.01
        lot = math.floor(lot / step) * step
        lot = round(lot, 2)
        if lot < (info.volume_min or 0.01):
            return 0.0, "below_min_lot"
        return lot, ("lot_cap" if capped else "risk_pct")

    # ── position management ─────────────────────────────────────────────────

    def manage(self):
        """Close anything past its hold limit or the session cutoff."""
        now = utcnow()
        flat_time = now.hour >= FLAT_BY_HOUR or now.hour < TRADE_FROM_HOUR
        for p in self.b.positions():
            age_min = (now.timestamp() - p.time) / 60.0
            why = None
            if flat_time:
                why = "session_close"
            elif age_min >= MAX_HOLD_MIN:
                why = "max_hold"
            if not why:
                continue
            if self.dry:
                log({"ev": "would_close", "symbol": p.symbol,
                     "ticket": p.ticket, "why": why,
                     "age_min": round(age_min, 1), "profit": p.profit})
                continue
            f = self.b.close(p, why)
            log({"ev": "close", "symbol": p.symbol, "ticket": p.ticket,
                 "why": why, "ok": f.ok, "price": f.price,
                 "age_min": round(age_min, 1), "profit": p.profit,
                 "comment": f.comment})

    # ── one pass ────────────────────────────────────────────────────────────

    def pass_once(self):
        acc = self.b.account()
        if acc is None:
            log({"ev": "error", "why": "no account info"})
            return
        self._roll_day(acc)
        self.manage()

        now = utcnow()
        if not (TRADE_FROM_HOUR <= now.hour < FLAT_BY_HOUR):
            self._note("window_closed", hour=now.hour,
                       opens=TRADE_FROM_HOUR, closes=FLAT_BY_HOUR)
            return
        if self._daily_locked(acc):
            self._note("daily_locked", equity=round(acc.equity, 2),
                       day_start=round(self.day_start_balance or 0, 2))
            return

        open_pos = self.b.positions()
        if len(open_pos) >= MAX_CONCURRENT:
            self._note("max_concurrent", open=len(open_pos),
                       cap=MAX_CONCURRENT)
            return
        margin_pct = self._margin_used_pct(acc)
        if margin_pct >= MAX_MARGIN_PCT:
            self._note("margin_ceiling", margin_pct=round(margin_pct, 1),
                       cap=MAX_MARGIN_PCT)
            return
        self._note("scanning", symbols=len(SYMBOLS))

        by_symbol = {}
        for p in open_pos:
            by_symbol.setdefault(p.symbol, []).append(p)

        for sym in SYMBOLS:
            if not self.b.ensure_symbol(sym):
                continue
            try:
                sigs = SIG.scan(self.b, sym, lookback_bars=2)
            except Exception as e:
                log({"ev": "scan_error", "symbol": sym,
                     "err": f"{type(e).__name__}: {e}"})
                continue
            for s in sigs:
                key = (sym, s["bar_time"])
                if key in self.seen:
                    continue
                self.seen.add(key)
                if not s["passes"]:
                    # name the filters that actually blocked it, and by how
                    # much. "why: filters" forced the reader to redo four
                    # comparisons by hand to learn anything from the line.
                    blocked = [{"f": v["name"], "v": v["value"],
                                "miss": (None if v["margin"] is None
                                         else round(v["margin"], 4))}
                               for v in SWING.verdict(s) if not v["passed"]]
                    log({"ev": "signal", "symbol": sym, "taken": False,
                         "why": "filters", "direction": s["direction"],
                         "blocked_by": [b["f"] for b in blocked],
                         "blocked": blocked,
                         "range_pct": round(s["range_pct"], 3),
                         "spread_pct": round(s["spread_pct"], 3),
                         "h4_pullback": round(s["h4_pullback"], 3),
                         "session": s["session"]})
                    continue
                self._try_enter(sym, s, acc, by_symbol.get(sym, []))

    def _try_enter(self, sym, s, acc, existing):
        is_buy = s["direction"] == "buy"
        mt5 = self.b.mt5
        for p in existing:
            if (p.type == mt5.POSITION_TYPE_BUY) != is_buy:
                log({"ev": "signal", "symbol": sym, "taken": False,
                     "why": "opposite_open", "direction": s["direction"]})
                return

        c = CONTRACTS[sym]
        tick = self.b.tick(sym)
        if tick is None:
            return
        price = tick.ask if is_buy else tick.bid
        stop_pips = float(s["risk_pips"])
        lot, reason = self.size(sym, acc, stop_pips, price)
        if lot <= 0:
            log({"ev": "signal", "symbol": sym, "taken": False, "why": reason,
                 "direction": s["direction"], "stop_pips": round(stop_pips, 1)})
            return

        dist = stop_pips * c.pip
        sl = price - dist if is_buy else price + dist
        tp = price + TARGET_R * dist if is_buy else price - TARGET_R * dist

        rec = {"ev": "signal", "symbol": sym, "taken": True,
               "direction": s["direction"], "lot": lot, "sizing": reason,
               "stop_pips": round(stop_pips, 1), "intended_price": price,
               "sl": sl, "tp": tp,
               "spread_now": round(self.b.spread_pips(sym, c.pip), 2),
               "range_pct": round(s["range_pct"], 3),
               "spread_pct": round(s["spread_pct"], 3),
               "h4_pullback": round(s["h4_pullback"], 3),
               "session": s["session"], "equity": acc.equity}
        if self.dry:
            rec["ev"] = "would_enter"
            log(rec)
            return
        f = self.b.market(sym, is_buy, lot, sl, tp, f"dt_{sym}")
        rec.update({"ok": f.ok, "fill_price": f.price, "ticket": f.ticket,
                    "slippage_pips": round(f.slippage / c.pip, 2) if f.ok else None,
                    "broker": f.comment})
        log(rec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--risk", type=float, default=RISK_PCT)
    ap.add_argument("--interval", type=int, default=60,
                    help="seconds between passes")
    ap.add_argument("--allow-live", action="store_true")
    a = ap.parse_args(argv)

    b = Broker()
    acc = b.connect(allow_live=a.allow_live)
    print(f"connected  {acc.login}@{acc.server}  balance ${acc.balance:,.2f}  "
          f"leverage 1:{acc.leverage}  {'DRY RUN' if a.dry_run else 'LIVE ORDERS'}")
    log({"ev": "start", "login": acc.login, "server": acc.server,
         "balance": acc.balance, "leverage": acc.leverage,
         "dry_run": a.dry_run, "symbols": SYMBOLS, "risk_pct": a.risk})

    t = Trader(b, dry_run=a.dry_run, risk_pct=a.risk)
    try:
        while True:
            try:
                t.pass_once()
            except Exception as e:
                log({"ev": "error", "err": f"{type(e).__name__}: {e}",
                     "tb": traceback.format_exc()[-800:]})
            if a.once:
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        log({"ev": "stop", "reason": "interrupt"})
    finally:
        b.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
