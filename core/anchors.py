"""Time-anchored reference levels — the intraday system's raw material.

The swing trader is STRUCTURE-anchored: find a break, measure the leg, wait for a
retrace into it. That works over 24h holds and it does not transfer to intraday,
which was measured directly — the same setup on a 25-pip stop and a 100-pip
target returns +0.041R, essentially nothing, because spread cost scales as 1/stop
and a 25-pip stop against a 3-6 pip funded spread pays 12-24% of its own risk
before it starts.

So the intraday system is anchored to the CLOCK instead. Its reference prices are
not discovered by an algorithm looking for structure; they are known in advance
every single day:

    the Asia session's high and low        (23:00-07:00 UTC)
    yesterday's high, low and close
    last week's high and low
    the day's open, and each session's open
    the first 30-60 minutes of London / NY

Why this is a different hypothesis rather than a reskin of the same one:

  * These levels are the same for every participant, which is the entire reason
    stops pool at them. A discovered "level" is a property of the discovering
    algorithm; yesterday's high is a property of the market.
  * They are known BEFORE the session starts, so the strategy is not waiting for
    a pattern to appear. It waits for a specific time.
  * Hour-of-day was already the strongest single separator in the feature study
    (hour 18 +0.163R, hour 11 +0.161R) and nothing in the tree acted on it.
  * The old tree had `session_sweep` (Power of Three) fully written and never
    wired to anything.

## The day boundary

Everything is indexed to an **fx_day that starts at 21:00 UTC**, not midnight.
That is not cosmetic: a UTC-midnight day cuts the Asia session in half, so
"today's Asia range" would span two day indices and every sweep test would be
wrong at the boundary. Starting at 21:00 puts one complete session cycle —
rollover, Asia, London, overlap, NY — inside each day index, in order.

## Causality

An anchor is NaN until the moment it is genuinely knowable. `asia_high` is NaN
for every bar before Asia closes at 07:00, so a London decision cannot read a
range that had not finished forming. This is the multi-timeframe lookahead that
manufactures enormous fake edge, and it is the reason `Context.align` exists;
the same discipline applies here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The fx day starts at the rollover, 21:00 UTC.
DAY_START_HOUR = 21
ASIA = (23, 7)          # UTC, crosses midnight
LONDON_OPEN = 7
NY_OPEN = 16


@dataclass
class Anchors:
    """Per-base-bar time-anchored levels. All arrays index-aligned with `bars`.

    NaN means "not knowable yet at this bar", never "missing data".
    """
    fx_day: np.ndarray          # int64  day index, boundary at 21:00 UTC
    hour: np.ndarray            # int8   UTC hour

    asia_high: np.ndarray       # float  NaN until Asia closes
    asia_low: np.ndarray
    asia_mid: np.ndarray
    asia_range_atr: np.ndarray  # float  Asia range measured in base ATRs

    pd_high: np.ndarray         # float  previous fx day, known from bar 1 of today
    pd_low: np.ndarray
    pd_close: np.ndarray
    pw_high: np.ndarray         # float  previous week
    pw_low: np.ndarray

    day_open: np.ndarray        # float  price at 21:00
    lon_open: np.ndarray        # float  price at 07:00, NaN before
    ny_open: np.ndarray         # float  price at 16:00, NaN before

    # running sweep state within the current fx day
    asia_hi_taken: np.ndarray   # bool   Asia high has traded through today
    asia_lo_taken: np.ndarray
    pd_hi_taken: np.ndarray
    pd_lo_taken: np.ndarray
    asia_first: np.ndarray      # int8   +1 high taken first, -1 low first, 0 neither

    def __len__(self) -> int:
        return len(self.fx_day)


def _prev_day_extremes(day_idx, day_edge, high, low, close, n):
    """Yesterday's high/low/close, published across the whole of today.

    Uses the PREVIOUS fx day's completed extremes, so it is knowable from the
    first bar of the current day onward. The first day of the series has no
    predecessor and stays NaN.
    """
    pdh = np.full(n, np.nan)
    pdl = np.full(n, np.nan)
    pdc = np.full(n, np.nan)
    for k in range(1, len(day_edge)):
        ps = day_edge[k - 1]
        pe = day_edge[k]
        s = day_edge[k]
        e = day_edge[k + 1] if k + 1 < len(day_edge) else n
        if pe > ps:
            pdh[s:e] = high[ps:pe].max()
            pdl[s:e] = low[ps:pe].min()
            pdc[s:e] = close[pe - 1]
    return pdh, pdl, pdc


def build(bars, atr: np.ndarray) -> Anchors:
    """Compute every time anchor for a base series. Vectorized per day."""
    t = np.asarray(bars.time, np.int64)
    n = len(t)
    high = np.asarray(bars.high, np.float64)
    low = np.asarray(bars.low, np.float64)
    close = np.asarray(bars.close, np.float64)
    open_ = np.asarray(bars.open, np.float64)
    a = np.maximum(np.asarray(atr, np.float64), 1e-12)

    hour = ((t // 3600) % 24).astype(np.int8)
    fx_day = (t - DAY_START_HOUR * 3600) // 86400
    day_edge = np.flatnonzero(np.concatenate(([True], np.diff(fx_day) != 0)))

    asia_hi = np.full(n, np.nan)
    asia_lo = np.full(n, np.nan)
    day_open = np.full(n, np.nan)
    lon_open = np.full(n, np.nan)
    ny_open = np.full(n, np.nan)
    hi_taken = np.zeros(n, bool)
    lo_taken = np.zeros(n, bool)
    pdh_taken = np.zeros(n, bool)
    pdl_taken = np.zeros(n, bool)
    first = np.zeros(n, np.int8)

    pd_high, pd_low, pd_close = _prev_day_extremes(
        fx_day, day_edge, high, low, close, n)

    for k in range(len(day_edge)):
        s = day_edge[k]
        e = day_edge[k + 1] if k + 1 < len(day_edge) else n
        if e <= s:
            continue
        day_open[s:e] = open_[s]
        h = hour[s:e]

        # ── Asia range, published only once Asia has closed ──────────────────
        # The window is 23:00-07:00. Inside an fx day that starts at 21:00 those
        # are simply the bars with hour >= 23 or hour < 7, and every one of them
        # precedes the 07:00 bars of the same day index.
        in_asia = (h >= ASIA[0]) | (h < ASIA[1])
        post = np.flatnonzero(h >= ASIA[1])          # 07:00 onward
        if in_asia.any() and post.size:
            w = np.flatnonzero(in_asia)
            # only bars strictly before the first post-Asia bar count
            w = w[w < post[0]]
            if w.size:
                ah = float(high[s:e][w].max())
                al = float(low[s:e][w].min())
                p0 = s + int(post[0])
                asia_hi[p0:e] = ah
                asia_lo[p0:e] = al

                # sweep state, from the first post-Asia bar onward
                seg_hi = high[p0:e]
                seg_lo = low[p0:e]
                up = np.flatnonzero(seg_hi > ah)
                dn = np.flatnonzero(seg_lo < al)
                fu = int(up[0]) if up.size else 10 ** 9
                fd = int(dn[0]) if dn.size else 10 ** 9
                if up.size:
                    hi_taken[p0 + fu:e] = True
                if dn.size:
                    lo_taken[p0 + fd:e] = True
                if fu < fd:
                    first[p0 + fu:e] = 1
                elif fd < fu:
                    first[p0 + fd:e] = -1

        # ── session opens ────────────────────────────────────────────────────
        for hr, arr in ((LONDON_OPEN, lon_open), (NY_OPEN, ny_open)):
            w = np.flatnonzero(h >= hr)
            if w.size:
                arr[s + int(w[0]):e] = open_[s + int(w[0])]

        # ── prior-day sweep state ────────────────────────────────────────────
        ph, pl = pd_high[s], pd_low[s]
        if np.isfinite(ph):
            up = np.flatnonzero(high[s:e] > ph)
            if up.size:
                pdh_taken[s + int(up[0]):e] = True
        if np.isfinite(pl):
            dn = np.flatnonzero(low[s:e] < pl)
            if dn.size:
                pdl_taken[s + int(dn[0]):e] = True

    # ── previous week ────────────────────────────────────────────────────────
    week = (t - DAY_START_HOUR * 3600) // (7 * 86400)
    w_edge = np.flatnonzero(np.concatenate(([True], np.diff(week) != 0)))
    pw_high = np.full(n, np.nan)
    pw_low = np.full(n, np.nan)
    for k in range(1, len(w_edge)):
        ps, pe = w_edge[k - 1], w_edge[k]
        s = w_edge[k]
        e = w_edge[k + 1] if k + 1 < len(w_edge) else n
        if pe > ps:
            pw_high[s:e] = high[ps:pe].max()
            pw_low[s:e] = low[ps:pe].min()

    return Anchors(
        fx_day=fx_day, hour=hour,
        asia_high=asia_hi, asia_low=asia_lo,
        asia_mid=(asia_hi + asia_lo) / 2.0,
        asia_range_atr=(asia_hi - asia_lo) / a,
        pd_high=pd_high, pd_low=pd_low, pd_close=pd_close,
        pw_high=pw_high, pw_low=pw_low,
        day_open=day_open, lon_open=lon_open, ny_open=ny_open,
        asia_hi_taken=hi_taken, asia_lo_taken=lo_taken,
        pd_hi_taken=pdh_taken, pd_lo_taken=pdl_taken,
        asia_first=first)
