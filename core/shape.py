"""Absolute candle anatomy — numbers, not names.

The operator's instruction, and it is a sharper idea than it first sounds:

    "i want a absolute candle understanding not pattern but body wick stuff not
     just ratio where it wicked what was candle look how was structure"

That is a different request from `core/candles.py`, which detects NAMED shapes
(pin, engulf, turtle soup) and was largely a dead end: all 35 patterns in the
library were built and measured, and **"no pattern at all" beat most named
shapes**. The reason is visible once you write the detectors down — a name is a
threshold applied to a continuous quantity, and the threshold throws away the
part that carried the information. `pin` is `lower_wick > 0.55 * range`; the
market does not know about 0.55.

So this module emits the CONTINUOUS quantities and never buckets them:

    how big was the body, in ATRs and as a fraction of range
    how long was each wick, separately, in ATRs and as a fraction
    WHERE did it close inside its own range (0 = on the low, 1 = on the high)
    where did it open
    did it take out the prior candle's high, its low, or both
    did it take one out and close back inside      <- the liquidity event
    did it gap from the prior close
    is its range expanding or contracting against the prior candle
    how many candles in a row have closed the same way

`reclaim_high` / `reclaim_low` are the ones the intraday system is built around.
"Swept the prior high and closed back below it" is a complete statement about who
just lost money and has to get out, and unlike `turtle_soup` it carries the depth
of the sweep and the strength of the reclaim as numbers alongside it, so research
can find the level rather than inheriting 0.15 from a docstring.

Everything is causal: row i describes candle i using candles 0..i only, and is
legal to act on at candle i's CLOSE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# The anatomy fields, in one place, so consumers can iterate them without
# restating the list (core/mtf.py expands every one of these onto the base bar
# index). SHORT_NAME keeps the emitted feature names readable once they are
# prefixed by a timeframe — `h1_prev_up_wick_atr`, not
# `h1_prev_upper_wick_atr`.
SHAPE_FIELDS = (
    "body_atr", "body_frac", "range_atr", "upper_wick_atr", "lower_wick_atr",
    "upper_wick_frac", "lower_wick_frac", "close_loc", "open_loc", "direction",
    "took_prev_high", "took_prev_low", "reclaim_high", "reclaim_low",
    "sweep_depth_high", "sweep_depth_low", "engulf", "inside", "outside",
    "gap_atr", "range_ratio", "close_run", "body_sum3",
)

SHORT_NAME = {
    "upper_wick_atr": "up_wick_atr", "lower_wick_atr": "dn_wick_atr",
    "upper_wick_frac": "up_wick_frac", "lower_wick_frac": "dn_wick_frac",
    "direction": "dir", "sweep_depth_high": "sweep_hi_atr",
    "sweep_depth_low": "sweep_lo_atr",
}


@dataclass
class Shape:
    """Per-candle anatomy. All arrays index-aligned with the source series."""
    # size
    body_atr: np.ndarray        # signed: + closed up, - closed down
    body_frac: np.ndarray       # |body| / range, 0..1
    range_atr: np.ndarray
    upper_wick_atr: np.ndarray
    lower_wick_atr: np.ndarray
    upper_wick_frac: np.ndarray
    lower_wick_frac: np.ndarray

    # where it closed / opened inside its own range
    close_loc: np.ndarray       # 0 = at the low, 1 = at the high
    open_loc: np.ndarray
    direction: np.ndarray       # int8 +1 / -1 / 0

    # relationship to the previous candle
    took_prev_high: np.ndarray  # bool
    took_prev_low: np.ndarray
    reclaim_high: np.ndarray    # took the high AND closed back below it
    reclaim_low: np.ndarray
    sweep_depth_high: np.ndarray  # how far beyond the prior high, in ATRs
    sweep_depth_low: np.ndarray
    engulf: np.ndarray          # int8 +1 bull, -1 bear, 0
    inside: np.ndarray          # bool  range contained by the prior candle
    outside: np.ndarray         # bool  took both prior extremes
    gap_atr: np.ndarray         # open - prev close, in ATRs
    range_ratio: np.ndarray     # range / prev range

    # persistence
    close_run: np.ndarray       # int8 consecutive closes in the same direction
    body_sum3: np.ndarray       # signed sum of the last 3 bodies, in ATRs

    def __len__(self) -> int:
        return len(self.body_atr)

    def row(self, i: int, prefix: str = "") -> dict:
        """One candle's anatomy as a flat feature dict, for the studies."""
        return {
            f"{prefix}body_atr": float(self.body_atr[i]),
            f"{prefix}body_frac": float(self.body_frac[i]),
            f"{prefix}range_atr": float(self.range_atr[i]),
            f"{prefix}up_wick_atr": float(self.upper_wick_atr[i]),
            f"{prefix}dn_wick_atr": float(self.lower_wick_atr[i]),
            f"{prefix}up_wick_frac": float(self.upper_wick_frac[i]),
            f"{prefix}dn_wick_frac": float(self.lower_wick_frac[i]),
            f"{prefix}close_loc": float(self.close_loc[i]),
            f"{prefix}dir": int(self.direction[i]),
            f"{prefix}took_prev_high": bool(self.took_prev_high[i]),
            f"{prefix}took_prev_low": bool(self.took_prev_low[i]),
            f"{prefix}reclaim_high": bool(self.reclaim_high[i]),
            f"{prefix}reclaim_low": bool(self.reclaim_low[i]),
            f"{prefix}sweep_hi_atr": float(self.sweep_depth_high[i]),
            f"{prefix}sweep_lo_atr": float(self.sweep_depth_low[i]),
            f"{prefix}engulf": int(self.engulf[i]),
            f"{prefix}inside": bool(self.inside[i]),
            f"{prefix}outside": bool(self.outside[i]),
            f"{prefix}gap_atr": float(self.gap_atr[i]),
            f"{prefix}range_ratio": float(self.range_ratio[i]),
            f"{prefix}close_run": int(self.close_run[i]),
            f"{prefix}body_sum3": float(self.body_sum3[i]),
        }


def read(bars, atr: np.ndarray) -> Shape:
    """Compute the full anatomy of every candle in `bars`.

    `atr` must be causal and aligned to the bar index — it is the scale that makes
    a 40-pip body mean the same thing in 2013 and 2026.
    """
    o = np.asarray(bars.open, np.float64)
    h = np.asarray(bars.high, np.float64)
    l = np.asarray(bars.low, np.float64)
    c = np.asarray(bars.close, np.float64)
    n = len(o)
    a = np.maximum(np.asarray(atr, np.float64), 1e-12)

    rng = np.maximum(h - l, 1e-12)
    body = c - o
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    prev = np.arange(n) - 1
    prev[0] = 0
    first = np.zeros(n, bool)
    first[0] = True
    ph, pl, pc, po = h[prev], l[prev], c[prev], o[prev]
    prng = np.maximum(ph - pl, 1e-12)

    took_hi = (h > ph) & ~first
    took_lo = (l < pl) & ~first
    # The liquidity event: ran the stops beyond the prior extreme, then closed
    # back on the origin side. Depth is kept as a number so research can find the
    # threshold instead of inheriting one.
    reclaim_hi = took_hi & (c < ph)
    reclaim_lo = took_lo & (c > pl)

    direction = np.zeros(n, np.int8)
    direction[c > o] = 1
    direction[c < o] = -1

    engulf = np.zeros(n, np.int8)
    both = took_hi & took_lo
    engulf[both & (c > o) & (pc < po)] = 1
    engulf[both & (c < o) & (pc > po)] = -1

    # consecutive closes in the same direction — a plain run-length, computed
    # without a Python loop over 5M bars
    run = np.zeros(n, np.int8)
    same = np.zeros(n, bool)
    same[1:] = direction[1:] == direction[:-1]
    cnt = 0
    d_prev = 0
    out = np.empty(n, np.int8)
    for i in range(n):
        d = direction[i]
        cnt = cnt + 1 if (d != 0 and d == d_prev) else (1 if d != 0 else 0)
        out[i] = min(cnt, 127)
        d_prev = d
    run = out

    b_atr = body / a
    body_sum3 = b_atr.copy()
    body_sum3[1:] += b_atr[:-1]
    body_sum3[2:] += b_atr[:-2]

    return Shape(
        body_atr=b_atr,
        body_frac=np.abs(body) / rng,
        range_atr=rng / a,
        upper_wick_atr=upper / a,
        lower_wick_atr=lower / a,
        upper_wick_frac=upper / rng,
        lower_wick_frac=lower / rng,
        close_loc=(c - l) / rng,
        open_loc=(o - l) / rng,
        direction=direction,
        took_prev_high=took_hi,
        took_prev_low=took_lo,
        reclaim_high=reclaim_hi,
        reclaim_low=reclaim_lo,
        sweep_depth_high=np.where(took_hi, (h - ph) / a, 0.0),
        sweep_depth_low=np.where(took_lo, (pl - l) / a, 0.0),
        engulf=engulf,
        inside=(h <= ph) & (l >= pl) & ~first,
        outside=both,
        gap_atr=np.where(first, 0.0, (o - pc) / a),
        range_ratio=rng / prng,
        close_run=run,
        body_sum3=body_sum3)


def wick_touched(bars, i: int, price: float, side: str, atr_i: float) -> float:
    """How far this candle's wick reached BEYOND `price`, in ATRs. 0 if it did not.

    "Where it wicked" only means something relative to a reference. A 0.4-ATR
    upper wick is noise in the middle of a range and a rejection if it is sitting
    on yesterday's high.
    """
    if not np.isfinite(price) or atr_i <= 0:
        return 0.0
    if side == "high":
        return max(0.0, float(bars.high[i]) - price) / atr_i
    return max(0.0, price - float(bars.low[i])) / atr_i
