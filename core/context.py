"""Volatility and regime context — the scale every threshold is expressed in.

Design rule 2: no fixed-pip constant anywhere in the decision path. The old tree
used a 5p zone half-width, a 20p cluster tolerance, a 3p sweep minimum, a 60p stop
and a fixed-pip trail ladder on a market whose average daily range ran 124p in
2017 and 1,322p in 2026. That single mismatch manufactured the strongest apparent
"regime signal" in the whole 14-year study: the stop was volatility-scaled and the
ladder was not, so rung 1 armed at 0.67R in one year and 0.22R in another. The
regime correlation was self-inflicted.

Everything here is vectorized and causal. `f(i)` uses bars 0..i inclusive and
never i+1. `assert_causal` in tests/test_context.py checks that by recomputing on
truncated inputs — a claim this important should not rest on reading the code.
"""
from __future__ import annotations

import numpy as np

from .store import Bars


def true_range(high, low, close) -> np.ndarray:
    """TR[i] = max(h-l, |h-prev_c|, |l-prev_c|). TR[0] = h-l."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    prev = np.empty_like(close)
    prev[0] = close[0]
    prev[1:] = close[:-1]
    return np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))


def rolling_mean(x, n: int) -> np.ndarray:
    """Causal rolling mean. Positions with fewer than n samples use the expanding
    mean rather than NaN, so warmup bars carry a usable — if noisier — scale
    instead of silently disabling every threshold that divides by it."""
    x = np.asarray(x, dtype=np.float64)
    c = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - n + 1)
    return (c[i + 1] - c[lo]) / (i - lo + 1)


def atr(bars: Bars, n: int = 14, lo: int = 0, hi: int | None = None) -> np.ndarray:
    """ATR(n) in PRICE units, aligned to bar index. atr[i] is legal to use in a
    decision taken at bar i's close."""
    hi = len(bars) if hi is None else hi
    tr = true_range(bars.high[lo:hi], bars.low[lo:hi], bars.close[lo:hi])
    return rolling_mean(tr, n)


def atr_pips(bars: Bars, n: int = 14, lo: int = 0, hi: int | None = None) -> np.ndarray:
    return atr(bars, n, lo, hi) / bars.pip


def realised_vol(close, n: int = 20) -> np.ndarray:
    """Rolling stdev of log returns, annualised on a 24h/365d basis. Used only for
    regime labelling and reporting — never as a threshold scale, because it says
    nothing about the size of the move a stop has to survive."""
    close = np.asarray(close, dtype=np.float64)
    r = np.zeros_like(close)
    r[1:] = np.log(close[1:] / np.clip(close[:-1], 1e-12, None))
    m = rolling_mean(r, n)
    v = rolling_mean(r * r, n) - m * m
    return np.sqrt(np.clip(v, 0.0, None)) * np.sqrt(365 * 24 * 60)


def efficiency_ratio(close, n: int = 20) -> np.ndarray:
    """Kaufman efficiency: |net move| / sum|move| over n bars. 1.0 = a clean
    trend, ~0 = pure chop. The 14-year study found ER uncorrelated with per-trade
    result (+0.005) — it is kept for slicing reports, not for gating."""
    close = np.asarray(close, dtype=np.float64)
    step = np.zeros_like(close)
    step[1:] = np.abs(np.diff(close))
    denom = rolling_mean(step, n) * n
    net = np.zeros_like(close)
    idx = np.arange(len(close))
    back = np.maximum(0, idx - n)
    net = np.abs(close - close[back])
    return np.where(denom > 1e-12, net / np.clip(denom, 1e-12, None), 0.0)


# ── the scale bundle ────────────────────────────────────────────────────────

class Scale:
    """Per-bar volatility scale for one timeframe, with named threshold helpers.

    Thresholds are declared as ATR fractions in ONE place so the mapping is
    auditable. The multipliers below are the V11 plan's proposals; every one of
    them is a hypothesis to be measured, not a fitted value, and none may be
    tuned without a run that shows the change surviving a two-halves split.
    """

    # name -> (timeframe, atr multiple)
    #
    # The cluster width cap is deliberately expressed against the STOP, not as an
    # independent ATR multiple, because "precise" only means anything relative to
    # the risk being taken. stop = 1.5 x ATR(M15), so a cap of 0.25 x stop is
    # 0.375 x ATR(M15). The first attempt used 0.25 x ATR(H1) — which came out at
    # 57 pips on live gold, roughly the whole stop, and produced a p50 cluster
    # width of 48p: the cap held, and the levels were still blobs. A cap price can
    # travel a quarter of the stop across is the loosest thing worth calling a
    # level.
    THRESHOLDS = {
        "zone_half_width":  ("M15", 0.15),   # was a flat 5p
        "cluster_tolerance": ("M15", 0.15),  # was a flat 20p
        "cluster_width_cap": ("M15", 0.375),  # 0.25 x stop. was unbounded in practice
        "sweep_min":        ("M5",  0.10),   # was a flat 3p
        "stop":             ("M15", 1.50),   # was a flat 60p
        "proximity":        ("H1",  0.50),   # was a flat 35p
        "contested":        ("H1",  0.35),   # was a flat 25p
    }

    __slots__ = ("bars", "n", "_atr")

    def __init__(self, bars: Bars, n: int = 14):
        self.bars = bars
        self.n = n
        self._atr = atr(bars, n)

    @property
    def atr(self) -> np.ndarray:
        return self._atr

    def pips(self, i: int) -> float:
        return float(self._atr[i]) / self.bars.pip

    def at(self, i: int, mult: float) -> float:
        """`mult` ATRs in PRICE units at bar i."""
        return float(self._atr[i]) * mult

    def pips_at(self, i: int, mult: float) -> float:
        return self.at(i, mult) / self.bars.pip


class Context:
    """Multi-timeframe scale bundle. Built once per symbol per study; every
    threshold in the decision path resolves through it.

    Timeframes are aligned by TIME, not by index: `align(tf, i)` maps bar i of the
    base timeframe to the newest bar of `tf` that is fully CLOSED at bar i's close.
    Aligning by index instead is the standard multi-timeframe lookahead — it lets
    a decision at 09:05 read an H4 bar that does not finish until 12:00.
    """

    def __init__(self, series: dict[str, Bars], base: str = "M5", n: int = 14):
        if base not in series:
            raise KeyError(f"base timeframe {base} missing from series")
        self.series = series
        self.base = base
        self.scales = {tf: Scale(b, n) for tf, b in series.items()}
        self._maps: dict[str, np.ndarray] = {}

    @property
    def bars(self) -> Bars:
        return self.series[self.base]

    def align(self, tf: str) -> np.ndarray:
        """Index map: for each base bar i, the index of the newest `tf` bar fully
        closed at base bar i's close. -1 where no such bar exists yet."""
        if tf in self._maps:
            return self._maps[tf]
        b = self.bars
        other = self.series[tf]
        # base bar i closes at time[i] + bar_seconds; the other bar is closed when
        # its own open + its own bar_seconds is at or before that instant.
        base_close = np.asarray(b.time, dtype=np.int64) + b.bar_seconds
        other_close = np.asarray(other.time, dtype=np.int64) + other.bar_seconds
        m = np.searchsorted(other_close, base_close, side="right") - 1
        self._maps[tf] = m.astype(np.int64)
        return self._maps[tf]

    def threshold(self, name: str) -> np.ndarray:
        """Threshold `name` in PRICE units, per base bar. Resolves the declared
        timeframe through `align`, so a threshold sourced from H1 changes only
        when an H1 bar actually closes."""
        tf, mult = Scale.THRESHOLDS[name]
        if tf == self.base:
            return self.scales[tf].atr * mult
        m = self.align(tf)
        src = self.scales[tf].atr
        out = np.full(len(self.bars), np.nan)
        ok = m >= 0
        out[ok] = src[m[ok]] * mult
        # Before the first close of the higher timeframe, fall back to the base
        # timeframe's own ATR scaled by the bar-length ratio — a rough but
        # bounded stand-in that keeps warmup bars tradeable rather than NaN.
        if (~ok).any():
            ratio = self.series[tf].bar_seconds / self.bars.bar_seconds
            out[~ok] = self.scales[self.base].atr[~ok] * np.sqrt(ratio) * mult
        return out

    def threshold_pips(self, name: str) -> np.ndarray:
        return self.threshold(name) / self.bars.pip


# ── regime labelling (reporting only) ───────────────────────────────────────

def label_regime(bars: Bars, n: int = 20) -> np.ndarray:
    """Coarse per-bar regime for SLICING results, never for gating.

    The 14-year study rejected the trendiness hypothesis outright — the choppy
    half of the sample ran -0.059R/trade and the trendy half -0.081R, so the bot
    does not lose because gold trends. Regime here exists to check that a change
    helps everywhere rather than in one market state.
    """
    close = np.asarray(bars.close, dtype=np.float64)
    er = efficiency_ratio(close, n)
    a = atr(bars, n)
    a_slow = rolling_mean(a, n * 5)
    expanding = a > a_slow * 1.10

    out = np.full(len(close), 2, dtype=np.int8)      # 2 = range
    idx = np.arange(len(close))
    back = np.maximum(0, idx - n)
    up = close > close[back]
    out[(er >= 0.35) & up] = 0                       # 0 = trend_up
    out[(er >= 0.35) & ~up] = 1                      # 1 = trend_dn
    out[(er < 0.35) & expanding] = 3                 # 3 = impulse
    return out


REGIME_NAMES = {0: "trend_up", 1: "trend_dn", 2: "range", 3: "impulse"}
