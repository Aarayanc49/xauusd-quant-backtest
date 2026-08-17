"""Live signal generation — the same code path the research used.

The single most important property of this file is that it does not contain a
strategy. It fetches a window of bars, wraps them so they look like the store's
`Bars`, and hands them to `research.features.build` — the exact function that
produced the candidate book every measured result in this project rests on.

The old tree had a separate live path, and its live behaviour diverged from its
backtest in at least two measured ways (the level cache was popped every bar in
backtest and never live; the tier gate therefore graded a different cluster set).
Both were invisible until someone read the code. One implementation is the only
defence.

## The window

Percentile features need history: `range_pct` ranks against 60 days, `atr_pct`
against 250. So the window has to be long enough for those to mean the same
thing live as in the backtest. 30,000 M5 bars is ~104 trading days, which covers
`range_pct` properly and gives `atr_pct` a shorter — but consistent — base than
the full-history backtest. That difference is the main known
backtest-versus-live gap and it is written down here rather than discovered
later.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store import PIP  # noqa: E402
from core.strategy import SWING  # noqa: E402
from research import features as F  # noqa: E402

WINDOW = {"M1": 6000, "M5": 30_000, "M15": 12_000,
          "H1": 6_000, "H4": 3_000, "D1": 800}
NEEDED = ("M1", "M5", "M15", "H1", "H4")


@dataclass
class _Meta:
    point: float


class LiveBars:
    """In-memory stand-in for `core.store.Bars`, built from MT5 rates."""

    __slots__ = ("symbol", "tf", "time", "open", "high", "low", "close",
                 "volume", "spread", "pip", "meta", "_secs")

    def __init__(self, symbol, tf, rates, point, secs):
        self.symbol, self.tf = symbol, tf
        self.time = np.asarray(rates["time"], np.int64)
        self.open = np.asarray(rates["open"], np.float64)
        self.high = np.asarray(rates["high"], np.float64)
        self.low = np.asarray(rates["low"], np.float64)
        self.close = np.asarray(rates["close"], np.float64)
        self.volume = np.asarray(rates["tick_volume"], np.float64)
        self.spread = np.asarray(rates["spread"], np.float64)
        self.pip = PIP.get(symbol.upper(), 0.0001)
        self.meta = _Meta(point=point)
        self._secs = secs

    def __len__(self):
        return len(self.time)

    @property
    def bar_seconds(self):
        return self._secs

    def spread_pips(self, lo=0, hi=None):
        hi = len(self) if hi is None else hi
        return self.spread[lo:hi] * self.meta.point / self.pip

    def t_str(self, i):
        return np.datetime64(int(self.time[i]), "s").astype(str)


SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
           "H1": 3600, "H4": 14400, "D1": 86400}


def fetch_series(broker, symbol: str) -> dict | None:
    """Pull every timeframe the pipeline needs, closed bars only."""
    info = broker.mt5.symbol_info(symbol)
    if info is None:
        return None
    out = {}
    for tf in NEEDED:
        r = broker.bars(symbol, tf, WINDOW[tf])
        if r is None or len(r) < 500:
            return None
        out[tf] = LiveBars(symbol, tf, r, info.point, SECONDS[tf])
    return out


def scan(broker, symbol: str, spread_mult: float = 1.0,
         lookback_bars: int = 2) -> list:
    """Signals on the bars that just closed, already passing the SWING filters.

    `spread_mult` is 1.0 live because the spread column is the real spread being
    paid — the 1.67 multiplier in research exists to model a funded account's
    wider quotes from demo data, and applying it to live quotes would
    double-count.
    """
    series = fetch_series(broker, symbol)
    if series is None:
        return []
    n = len(series["M5"])
    rows = F.build(symbol=symbol, base="M5", series=series, quiet=True,
                   spread_mult=spread_mult, target_r=8.0, hold_hours=1,
                   since_bar=max(0, n - lookback_bars))
    out = []
    for r in rows:
        r["symbol"] = symbol
        r["passes"] = SWING.passes(r)
        r["bar_time"] = int(series["M5"].time[r["bar"]])
        out.append(r)
    return out
