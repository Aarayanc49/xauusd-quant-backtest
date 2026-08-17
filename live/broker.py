"""MT5 connection and order execution.

Thin, deliberately. Every decision lives in `core/`; this file knows how to talk
to the terminal and nothing else. The old tree's connector had strategy logic
inside it and that is why nothing in it could be tested without a live terminal.

## Safety

`connect()` refuses to proceed unless the account reports DEMO. That check is
not a formality — this module places real orders, and the only thing separating
a demo experiment from a live one is which account the terminal happens to be
logged into. Pass `allow_live=True` deliberately if that is ever wanted.

Every order carries `MAGIC` so the trader can find its own positions and never
touches anything opened by hand or by another system.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAGIC = 20260816
DEVIATION = 20          # max slippage in points we will accept on a market order

TF = {}                 # filled on connect, needs the mt5 module


@dataclass
class Fill:
    ok: bool
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    comment: str = ""
    requested: float = 0.0

    @property
    def slippage(self) -> float:
        return self.price - self.requested if self.price and self.requested else 0.0


class Broker:
    def __init__(self):
        import MetaTrader5 as mt5
        self.mt5 = mt5
        self.info = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def connect(self, allow_live: bool = False):
        mt5 = self.mt5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError("no account info — is the terminal logged in?")
        is_demo = acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        if not is_demo and not allow_live:
            raise RuntimeError(
                f"REFUSING TO TRADE: account {acc.login} is not a demo "
                f"(trade_mode={acc.trade_mode}). Pass allow_live=True to override.")
        if not acc.trade_allowed:
            raise RuntimeError("terminal reports trading is not allowed — "
                               "check 'Algo Trading' is enabled")
        global TF
        TF = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
              "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
              "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
              "D1": mt5.TIMEFRAME_D1}
        self.info = acc
        return acc

    def connect_readonly(self):
        """Attach to the terminal for READING only — no order path is used.

        `connect` refuses a non-demo account and a terminal with algo trading
        switched off, because it is the entry point for code that places
        orders. Both checks are wrong for an observer: `live/explain.py` needs
        to read bars and account state from whatever account is loaded, and
        refusing to *explain* a live account is the opposite of safe — it hides
        the reasoning exactly where it matters most.

        Nothing reachable from here can send an order. Anything that places one
        must go through `connect`.
        """
        mt5 = self.mt5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError("no account info — is the terminal logged in?")
        global TF
        TF = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
              "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
              "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
              "D1": mt5.TIMEFRAME_D1}
        self.info = acc
        return acc

    def shutdown(self):
        self.mt5.shutdown()

    # ── market data ─────────────────────────────────────────────────────────

    def bars(self, symbol: str, tf: str, n: int):
        """The last `n` CLOSED bars. The forming bar is dropped.

        MT5 returns the in-progress bar as the newest row. Feeding that to a
        decision is a one-bar lookahead in the only place it actually matters —
        live — and it is the classic way a backtest and a live loop silently
        stop being the same system.
        """
        r = self.mt5.copy_rates_from_pos(symbol, TF[tf], 0, n + 1)
        if r is None or len(r) < 2:
            return None
        return r[:-1]

    def tick(self, symbol: str):
        return self.mt5.symbol_info_tick(symbol)

    def spread_pips(self, symbol: str, pip: float) -> float:
        t = self.tick(symbol)
        if t is None:
            return float("nan")
        return (t.ask - t.bid) / pip

    def ensure_symbol(self, symbol: str) -> bool:
        i = self.mt5.symbol_info(symbol)
        if i is None:
            return False
        if not i.visible:
            return self.mt5.symbol_select(symbol, True)
        return True

    # ── account ─────────────────────────────────────────────────────────────

    def account(self):
        return self.mt5.account_info()

    def positions(self, symbol: str | None = None):
        p = (self.mt5.positions_get(symbol=symbol) if symbol
             else self.mt5.positions_get())
        return [x for x in (p or []) if x.magic == MAGIC]

    def margin_for(self, symbol: str, lot: float, is_buy: bool, price: float):
        return self.mt5.order_calc_margin(
            self.mt5.ORDER_TYPE_BUY if is_buy else self.mt5.ORDER_TYPE_SELL,
            symbol, lot, price)

    # ── execution ───────────────────────────────────────────────────────────

    def market(self, symbol: str, is_buy: bool, lot: float,
               sl: float, tp: float, comment: str = "") -> Fill:
        mt5 = self.mt5
        t = self.tick(symbol)
        if t is None:
            return Fill(False, comment="no tick")
        px = t.ask if is_buy else t.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": px,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": DEVIATION,
            "magic": MAGIC,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(symbol),
        }
        r = mt5.order_send(req)
        if r is None:
            return Fill(False, comment=f"order_send None {mt5.last_error()}",
                        requested=px)
        ok = r.retcode == mt5.TRADE_RETCODE_DONE
        return Fill(ok, ticket=getattr(r, "order", 0), price=getattr(r, "price", 0.0),
                    volume=getattr(r, "volume", 0.0),
                    comment=f"retcode={r.retcode} {getattr(r, 'comment', '')}",
                    requested=px)

    def close(self, pos, comment: str = "flat") -> Fill:
        mt5 = self.mt5
        t = self.tick(pos.symbol)
        if t is None:
            return Fill(False, comment="no tick")
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        px = t.bid if is_buy else t.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": px,
            "deviation": DEVIATION,
            "magic": MAGIC,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(pos.symbol),
        }
        r = mt5.order_send(req)
        if r is None:
            return Fill(False, comment=f"order_send None {mt5.last_error()}")
        return Fill(r.retcode == mt5.TRADE_RETCODE_DONE,
                    ticket=pos.ticket, price=getattr(r, "price", 0.0),
                    volume=getattr(r, "volume", 0.0),
                    comment=f"retcode={r.retcode}", requested=px)

    def _filling(self, symbol: str):
        """Pick a fill mode the symbol actually supports.

        Brokers differ and an unsupported mode is rejected with an unhelpful
        retcode. IOC is the common case; FOK and RETURN are the fallbacks.
        """
        mt5 = self.mt5
        i = mt5.symbol_info(symbol)
        mode = getattr(i, "filling_mode", 0) if i else 0
        if mode & 2:
            return mt5.ORDER_FILLING_IOC
        if mode & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN
