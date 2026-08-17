"""VWAP — the reference price that is actually traded against.

Missing from this tree entirely, which is a real gap: VWAP is the benchmark
institutional intraday execution is measured against, and it is the one line on
an intraday chart that a large share of participants genuinely care about.

## Why this is not just another level

The anchored-liquidity study killed 32 kinds of reference price — Asia high,
yesterday's low, the previous H1 candle's extreme — at a gross edge of -0.007R
over 521,606 events. It would be reasonable to assume VWAP dies with them. Two
things make it a different object:

  1. **It is volume-weighted.** Every level in the failed study was a price that
     something touched once. VWAP is where business was actually done, weighted
     by how much. It is the only reference in this project derived from the
     volume column at all.
  2. **It is a moving average, not a boundary.** The failed family asked "does
     price reverse at this line". The VWAP question is different: "how far from
     fair value is price, and does distance predict reversion". That is a
     continuous, two-sided claim about a distribution, and it fails or succeeds
     independently of anything measured so far.

## What is computed

    session VWAP     anchored to the fx day (21:00 UTC), the standard one
    london VWAP      anchored to 07:00 — what the European book is measured on
    ny VWAP          anchored to 16:00
    bands            +/- 1 and 2 volume-weighted standard deviations

and, the features a model actually reads, all in ATRs so they mean the same
thing in a 124-pip year and a 1,322-pip one:

    dev              signed distance from session VWAP
    band_pos         where price sits in the band, 0 = lower, 1 = upper
    slope            is VWAP itself rising or falling

## Causality

VWAP at bar i uses bars from the anchor up to and including i. It is a running
cumulative, so there is no lookahead available even by accident — but the bands
are computed from the same running sums rather than from the completed session,
which is the mistake that would quietly leak the day's outcome into a morning
decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VWAP:
    """Per-bar VWAP state, aligned to the base bar index."""
    session: np.ndarray         # running session VWAP price
    london: np.ndarray
    ny: np.ndarray
    upper1: np.ndarray          # session VWAP +1 sigma
    lower1: np.ndarray
    upper2: np.ndarray
    lower2: np.ndarray

    dev: np.ndarray             # (close - session VWAP) / ATR, signed
    dev_london: np.ndarray
    band_pos: np.ndarray        # 0 at -1 sigma, 1 at +1 sigma, unclipped
    slope: np.ndarray           # VWAP change over the last hour, in ATRs

    def row(self, i: int, prefix: str = "vwap_") -> dict:
        return {
            f"{prefix}dev": float(self.dev[i]),
            f"{prefix}dev_london": float(self.dev_london[i]),
            f"{prefix}band_pos": float(self.band_pos[i]),
            f"{prefix}slope": float(self.slope[i]),
        }


def _anchored(price: np.ndarray, vol: np.ndarray, starts: np.ndarray,
              n: int, want_sigma: bool = False):
    """Running volume-weighted mean (and variance) from each anchor in `starts`.

    `starts` is the index of every anchor bar. Between anchors the sums simply
    accumulate, so the value at bar i is the VWAP of [anchor .. i] — which is
    what a screen shows intraday, not the completed-session figure.
    """
    pv = price * vol
    c_pv = np.concatenate(([0.0], np.cumsum(pv)))
    c_v = np.concatenate(([0.0], np.cumsum(vol)))
    c_pv2 = np.concatenate(([0.0], np.cumsum(price * price * vol)))

    base = np.zeros(n, np.int64)
    for k in range(len(starts)):
        s = starts[k]
        e = starts[k + 1] if k + 1 < len(starts) else n
        base[s:e] = s

    idx = np.arange(n) + 1
    v_sum = np.maximum(c_v[idx] - c_v[base], 1e-9)
    mean = (c_pv[idx] - c_pv[base]) / v_sum
    if not want_sigma:
        return mean, None
    m2 = (c_pv2[idx] - c_pv2[base]) / v_sum
    sigma = np.sqrt(np.clip(m2 - mean * mean, 0.0, None))
    return mean, sigma


def build(bars, atr: np.ndarray, fx_day: np.ndarray | None = None,
          hour: np.ndarray | None = None) -> VWAP:
    """Compute session, London and NY VWAPs with bands."""
    h = np.asarray(bars.high, np.float64)
    l = np.asarray(bars.low, np.float64)
    c = np.asarray(bars.close, np.float64)
    v = np.maximum(np.asarray(bars.volume, np.float64), 1e-9)
    t = np.asarray(bars.time, np.int64)
    n = len(c)
    a = np.maximum(np.asarray(atr, np.float64), 1e-12)

    # Typical price, the conventional VWAP input — a bar's close alone
    # over-weights wherever the last tick landed.
    tp = (h + l + c) / 3.0

    if hour is None:
        hour = ((t // 3600) % 24).astype(np.int8)
    if fx_day is None:
        fx_day = (t - 21 * 3600) // 86400

    day_starts = np.flatnonzero(np.concatenate(([True], np.diff(fx_day) != 0)))
    sess, sigma = _anchored(tp, v, day_starts, n, want_sigma=True)

    def session_anchor(open_hour):
        """First bar at or after `open_hour` within each fx day."""
        marks = np.zeros(n, bool)
        for k in range(len(day_starts)):
            s = day_starts[k]
            e = day_starts[k + 1] if k + 1 < len(day_starts) else n
            w = np.flatnonzero(hour[s:e] >= open_hour)
            if w.size:
                marks[s + int(w[0])] = True
        st = np.flatnonzero(marks)
        return st if st.size else day_starts

    lon, _ = _anchored(tp, v, session_anchor(7), n)
    ny, _ = _anchored(tp, v, session_anchor(16), n)

    # slope over the last hour, in ATRs
    step = max(1, 3600 // bars.bar_seconds)
    prev = np.empty(n)
    prev[:step] = sess[:step]
    prev[step:] = sess[:-step]
    slope = (sess - prev) / a

    band = np.maximum(sigma, 1e-9)
    return VWAP(
        session=sess, london=lon, ny=ny,
        upper1=sess + band, lower1=sess - band,
        upper2=sess + 2 * band, lower2=sess - 2 * band,
        dev=(c - sess) / a,
        dev_london=(c - lon) / a,
        band_pos=(c - (sess - band)) / (2 * band),
        slope=slope)
