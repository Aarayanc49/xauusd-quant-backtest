"""Volume and microstructure — the only input here that is not a function of OHLC.

Every study in this project so far has measured a transformation of open, high,
low and close. The `volume` column has been in the store since the first fetch
and no code has ever read it. That makes this the largest genuinely unexplored
surface available without fetching new data, and it is the standard explanation
for the failure that motivated it: 521,606 level-interaction events returned a
gross edge of -0.007R, and "the break that matters is the one on volume" is the
first thing anyone would say about that result.

## Is it usable?

Measured on XAUUSD M5, 2012-2026, before building anything on it:

    zero-volume bars   0.00%
    median             322 ticks   (154-412 across individual years)
    corr(log volume, log range)   +0.669

Strongly positive correlation with range is the sanity check — tick volume that
did not track movement would be a broken column. USTEC fails this (corr +0.152,
max 8.9M against a median of 671) so its volume is not trustworthy; USDJPY is
fine at +0.520.

## The trap this module exists to avoid

Tick volume has a large, fixed hour-of-day shape — gold's median M5 volume runs
~105 at 23:00 UTC and ~720 at 17:00. A naive z-score of raw volume is therefore
mostly a clock, and "high volume" would silently mean "it is the afternoon".
That matters enormously here, because the one real gradient the failed intraday
study found was **cost**, which is also a function of the hour. A feature that
re-encodes the hour would rediscover the spread gradient and look like a new
signal.

So the headline feature is `rel_hour`: this bar's volume against the trailing
median volume **for this hour of day specifically**. A value of 2.0 means twice
the participation normal for this time — which is a statement about today, not
about the clock.

Everything is causal: bar i uses bars up to and including i, never i+1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .context import rolling_mean


@dataclass
class Flow:
    """Per-bar volume and microstructure state, aligned to the base bar index."""
    volume: np.ndarray
    rel_hour: np.ndarray        # volume / trailing median volume FOR THIS HOUR
    rel_recent: np.ndarray      # volume / trailing mean over the last day
    vol_pct: np.ndarray         # trailing percentile rank, 0..1
    vol_trend: np.ndarray       # mean(5) / mean(20) — participation building?
    per_range: np.ndarray       # ticks per ATR of range — effort vs result
    range_per_tick: np.ndarray  # the inverse: how far price moves per tick
    delta: np.ndarray           # signed participation proxy, in ATR-volume units
    delta_cum: np.ndarray       # cumulative signed proxy within the fx day
    divergence: np.ndarray      # new extreme on falling participation

    def row(self, i: int, prefix: str = "vol_") -> dict:
        return {
            f"{prefix}rel_hour": float(self.rel_hour[i]),
            f"{prefix}rel_recent": float(self.rel_recent[i]),
            f"{prefix}pct": float(self.vol_pct[i]),
            f"{prefix}trend": float(self.vol_trend[i]),
            f"{prefix}per_range": float(self.per_range[i]),
            f"{prefix}delta": float(self.delta[i]),
            f"{prefix}delta_cum": float(self.delta_cum[i]),
            f"{prefix}divergence": float(self.divergence[i]),
        }


def _trailing_hour_median(v: np.ndarray, hour: np.ndarray,
                          samples: int = 240) -> np.ndarray:
    """Trailing mean volume for each bar's OWN hour of day.

    `samples` counts bars of that hour, not bars overall — 240 M5 bars at a given
    hour is 20 trading days of that hour. Computed by slicing the series per
    hour, running a causal rolling mean within the slice, and scattering back, so
    bar i is normalised only against earlier bars sharing its hour.
    """
    out = np.ones(len(v))
    for h in range(24):
        idx = np.flatnonzero(hour == h)
        if idx.size < 10:
            continue
        m = rolling_mean(v[idx], samples)
        # shift by one so bar i is compared against history STRICTLY before it
        m_prev = np.empty_like(m)
        m_prev[0] = m[0]
        m_prev[1:] = m[:-1]
        out[idx] = np.maximum(m_prev, 1e-9)
    return out


def _causal_roll(x: np.ndarray, window: int, op) -> np.ndarray:
    """Rolling max/min over the trailing `window` bars, inclusive of bar i.

    `sliding_window_view` gives a strided view rather than a copy, so this costs
    one reduction pass and no extra memory. The first `window-1` bars have no
    full window and take the expanding result instead of NaN, matching
    `context.rolling_mean` — a warmup bar should carry a noisier value, not
    disable every feature that reads it.
    """
    n = len(x)
    if n <= window:
        return op.accumulate(x)
    view = np.lib.stride_tricks.sliding_window_view(x, window)
    out = np.empty(n, np.float64)
    out[:window - 1] = op.accumulate(x[:window - 1])
    out[window - 1:] = op.reduce(view, axis=1)
    return out


def _pct_rank(x: np.ndarray, window: int, grid: int = 4000) -> np.ndarray:
    """Causal trailing percentile, sampled on a grid and interpolated.

    Same approach as core/session.py — an exact rolling rank over a million bars
    is far too slow for a study that reruns constantly, and the interpolation
    error is irrelevant against bucket widths of 0.25.
    """
    n = len(x)
    if n < 50:
        return np.full(n, 0.5)
    step = max(1, n // grid)
    pts = np.arange(0, n, step)
    exact = np.empty(len(pts))
    for k, i in enumerate(pts):
        s = max(0, i - window)
        seg = x[s:i + 1]
        exact[k] = (seg < x[i]).mean() if len(seg) > 1 else 0.5
    return np.interp(np.arange(n), pts, exact)


def build(bars, atr: np.ndarray, fx_day: np.ndarray | None = None) -> Flow:
    """Compute the full flow state for a series."""
    v = np.asarray(bars.volume, np.float64)
    h = np.asarray(bars.high, np.float64)
    l = np.asarray(bars.low, np.float64)
    c = np.asarray(bars.close, np.float64)
    t = np.asarray(bars.time, np.int64)
    n = len(v)
    a = np.maximum(np.asarray(atr, np.float64), 1e-12)
    hour = ((t // 3600) % 24).astype(np.int8)

    bars_per_day = max(1, 86400 // bars.bar_seconds)

    hour_norm = _trailing_hour_median(v, hour)
    rel_hour = v / hour_norm

    recent = np.maximum(rolling_mean(v, bars_per_day), 1e-9)
    rel_recent = v / recent

    vol_pct = _pct_rank(v, bars_per_day * 60)
    vol_trend = rolling_mean(v, 5) / np.maximum(rolling_mean(v, 20), 1e-9)

    rng_atr = np.maximum((h - l) / a, 1e-6)
    per_range = v / rng_atr                 # ticks spent per ATR of movement
    range_per_tick = rng_atr / np.maximum(v, 1e-9)

    # Signed participation. Where a bar closes inside its own range is the
    # cheapest available proxy for whether buyers or sellers finished in
    # control; weighting it by volume turns it into a flow estimate. This is a
    # PROXY — real delta needs tick data with bid/ask, which a retail feed does
    # not carry. Named `delta` for what it approximates, not what it is.
    close_loc = (c - l) / np.maximum(h - l, 1e-12)
    delta = rel_hour * (2.0 * close_loc - 1.0)

    # cumulative within the trading day — is the session net bought or sold?
    if fx_day is None:
        fx_day = (t - 21 * 3600) // 86400
    edges = np.flatnonzero(np.concatenate(([True], np.diff(fx_day) != 0)))
    delta_cum = np.empty(n)
    for k in range(len(edges)):
        s = edges[k]
        e = edges[k + 1] if k + 1 < len(edges) else n
        delta_cum[s:e] = np.cumsum(delta[s:e])

    # Divergence: price makes a new local extreme while participation falls. The
    # value is signed — positive when a new HIGH comes on fading flow, negative
    # when a new LOW does — so one column carries both sides of the exhaustion
    # claim and a model can read direction from its sign.
    look = max(2, bars_per_day // 8)
    roll_hi = _causal_roll(h, look, np.maximum)
    roll_lo = _causal_roll(l, look, np.minimum)
    fading = 1.0 - vol_trend
    divergence = np.where(h >= roll_hi, fading,
                          np.where(l <= roll_lo, -fading, 0.0))

    return Flow(volume=v, rel_hour=rel_hour, rel_recent=rel_recent,
                vol_pct=vol_pct, vol_trend=vol_trend, per_range=per_range,
                range_per_tick=range_per_tick, delta=delta,
                delta_cum=delta_cum, divergence=divergence)
