"""Session, time-of-day, candle opens, and volatility state.

Three separate things the old tree treated as one vague notion of "context", none
of which ever gated the engine that traded:

  * **Session and time of day.** `market_clock` had kill zones and conviction
    multipliers; ScanTrader used the conviction as a size scalar and never as a
    filter. The measured session gradient from the 14-year study was real and
    monotonic — rollover -0.40R/trade, asia -0.32, london -0.22, ldn/ny -0.08,
    ny pm -0.08, late -0.06 — and nothing acted on it.

  * **Candle opens.** `level_read._inject_open_levels` treated the current D1/H4/H1
    OPEN as a magnet level, and `_update_opening_range` captured a 4H opening
    range. Both rendered to a terminal and gated nothing. The open of a session is
    where the day's positioning starts from, and distance from it is a cheap,
    genuinely independent state variable.

  * **Volatility state.** The single most important measured fact in the whole
    project is that gold's average daily range ran 124 pips in 2017 and 1,322 in
    2026 at $1,200 and $4,700. Everything here is expressed as a PERCENTILE of the
    instrument's own recent history rather than an absolute, so a "high volatility
    bar" means the same thing in both years.

Spread is included because the study proved it decides profitability: random-entry
cost is -0.19R at a 1.5-ATR stop and scales as 1/stop, against a measured gross
edge of +0.147R. Hour-of-day spread is therefore a first-class strategy variable,
not an execution detail.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .context import rolling_mean

# UTC session bounds. Gold's liquidity follows FX, so these are the FX sessions.
SESSIONS = {
    "rollover": (21, 23),     # thin, and where the measured spread doubles
    "asia": (23, 7),
    "london": (7, 12),
    "overlap": (12, 16),      # London/NY — the deepest book of the day
    "ny": (16, 21),
}
SESSION_ID = {"rollover": 0, "asia": 1, "london": 2, "overlap": 3, "ny": 4}
SESSION_NAME = {v: k for k, v in SESSION_ID.items()}


@dataclass
class State:
    """Per-bar context. All arrays index-aligned with the base series."""
    hour: np.ndarray            # int8  UTC hour
    session: np.ndarray         # int8  SESSION_ID
    minutes_into: np.ndarray    # int16 minutes since this session opened
    dow: np.ndarray             # int8  0=Mon .. 6=Sun

    atr_pct: np.ndarray         # float percentile of ATR vs trailing year
    expansion: np.ndarray       # float ATR(5) / ATR(20) — >1 expanding
    range_pct: np.ndarray       # float today's range so far vs its own history

    spread_pips: np.ndarray     # float per-bar broker spread
    spread_pct: np.ndarray      # float percentile vs trailing history

    from_d1_open: np.ndarray    # float distance from the D1 open, in ATRs
    from_h4_open: np.ndarray    # float
    from_h1_open: np.ndarray    # float

    or_high: np.ndarray         # float opening range of the current session
    or_low: np.ndarray
    or_broken: np.ndarray       # int8  0 none, +1 above, -1 below

    def session_name(self, i: int) -> str:
        return SESSION_NAME.get(int(self.session[i]), "?")


def _pct_rank(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing percentile of each value within its own recent history.

    Causal by construction — bar i is ranked only against bars before it. Uses a
    coarse histogram rather than a sort per bar: an exact rolling rank over 1M
    bars is O(n * window log window) and this study reruns constantly.
    """
    n = len(x)
    out = np.full(n, 0.5)
    if n < 50:
        return out
    lo, hi = np.nanpercentile(x[np.isfinite(x)], [0.5, 99.5])
    if not np.isfinite(lo) or hi <= lo:
        return out
    nb = 256
    b = np.clip(((x - lo) / (hi - lo) * nb).astype(np.int32), 0, nb - 1)
    counts = np.zeros(nb + 1, np.float64)
    total = 0.0
    for i in range(n):
        if total > 0:
            out[i] = counts[b[i]] / total
        counts[b[i]] += 1.0
        total += 1.0
        if i >= window:
            counts[b[i - window]] -= 1.0
            total -= 1.0
        # cumulative-below needs a running structure; approximate with the
        # bin's own share plus everything under it, recomputed cheaply
        if i % 512 == 0 and total > 0:
            cum = np.cumsum(counts[:nb])
            below = cum / max(total, 1.0)
            out[i] = below[b[i]]
    # smooth pass: recompute exactly on a coarse grid and interpolate
    step = max(1, n // 4000)
    grid = np.arange(0, n, step)
    exact = np.empty(len(grid))
    for k, i in enumerate(grid):
        s = max(0, i - window)
        seg = x[s:i + 1]
        exact[k] = (seg < x[i]).mean() if len(seg) > 1 else 0.5
    return np.interp(np.arange(n), grid, exact)


def build(bars, atr: np.ndarray, ctx=None, server_offset: int = 0) -> State:
    """Compute the full per-bar context for a series."""
    t = np.asarray(bars.time, np.int64)
    n = len(t)
    hour = (((t + server_offset * 3600) // 3600) % 24).astype(np.int8)
    dow = (((t // 86400) + 4) % 7).astype(np.int8)     # 1970-01-01 was a Thursday

    sess = np.full(n, SESSION_ID["asia"], np.int8)
    for name, (a, b) in SESSIONS.items():
        m = (hour >= a) & (hour < b) if a < b else ((hour >= a) | (hour < b))
        sess[m] = SESSION_ID[name]

    # minutes since the session flipped
    minutes = np.zeros(n, np.int16)
    flip = np.flatnonzero(np.concatenate(([True], np.diff(sess) != 0)))
    for k in range(len(flip)):
        s = flip[k]
        e = flip[k + 1] if k + 1 < len(flip) else n
        minutes[s:e] = np.clip((t[s:e] - t[s]) // 60, 0, 32767)

    a = np.asarray(atr, np.float64)
    bars_per_day = max(1, 86400 // bars.bar_seconds)
    atr_pct = _pct_rank(a, bars_per_day * 250)
    expansion = np.divide(rolling_mean(a, 5), np.maximum(rolling_mean(a, 20), 1e-12))

    high = np.asarray(bars.high, np.float64)
    low = np.asarray(bars.low, np.float64)
    day = t // 86400
    day_edge = np.flatnonzero(np.concatenate(([True], np.diff(day) != 0)))
    day_rng = np.zeros(n)
    d1_open = np.zeros(n)
    for k in range(len(day_edge)):
        s = day_edge[k]
        e = day_edge[k + 1] if k + 1 < len(day_edge) else n
        day_rng[s:e] = np.maximum.accumulate(high[s:e]) - np.minimum.accumulate(low[s:e])
        d1_open[s:e] = bars.open[s]
    range_pct = _pct_rank(day_rng, bars_per_day * 60)

    sp = bars.spread_pips()
    spread_pct = _pct_rank(sp, bars_per_day * 60)

    def tf_open(seconds):
        edge = np.flatnonzero(np.concatenate(([True], np.diff(t // seconds) != 0)))
        out = np.zeros(n)
        for k in range(len(edge)):
            s = edge[k]
            e = edge[k + 1] if k + 1 < len(edge) else n
            out[s:e] = bars.open[s]
        return out

    h4_open = tf_open(4 * 3600)
    h1_open = tf_open(3600)
    close = np.asarray(bars.close, np.float64)
    safe_a = np.maximum(a, 1e-12)

    # opening range of each session, and whether it has been taken
    or_hi = np.full(n, np.nan)
    or_lo = np.full(n, np.nan)
    or_brk = np.zeros(n, np.int8)
    or_bars = max(1, 30 * 60 // bars.bar_seconds)      # first 30 minutes
    for k in range(len(flip)):
        s = flip[k]
        e = flip[k + 1] if k + 1 < len(flip) else n
        w = min(s + or_bars, e)
        hi_v = float(high[s:w].max()) if w > s else np.nan
        lo_v = float(low[s:w].min()) if w > s else np.nan
        or_hi[s:e] = hi_v
        or_lo[s:e] = lo_v
        if w < e and np.isfinite(hi_v):
            up = np.flatnonzero(close[w:e] > hi_v)
            dn = np.flatnonzero(close[w:e] < lo_v)
            fu = w + up[0] if up.size else n + 1
            fd = w + dn[0] if dn.size else n + 1
            if fu < fd and fu < e:
                or_brk[fu:e] = 1
            elif fd < e:
                or_brk[fd:e] = -1

    return State(hour=hour, session=sess, minutes_into=minutes, dow=dow,
                 atr_pct=atr_pct, expansion=expansion, range_pct=range_pct,
                 spread_pips=sp, spread_pct=spread_pct,
                 from_d1_open=(close - d1_open) / safe_a,
                 from_h4_open=(close - h4_open) / safe_a,
                 from_h1_open=(close - h1_open) / safe_a,
                 or_high=or_hi, or_low=or_lo, or_broken=or_brk)
