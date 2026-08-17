"""Market structure — BOS, CHoCH, impulse legs, and the pullback zone.

This is the half of the old tree that was designed four times and wired zero
times. `market_structure._detect_structure_breaks` produced the events;
`confirmation_stack`, `signal_advisor`, `reaction_engine` and four of the eight
strategies in `strategies/` all consumed them. None of those ran in production.
What shipped fired on the sweep candle's close with a fixed 60-pip stop.

## The distinction that matters

    BOS   — a swing is broken in the direction of the prevailing trend.
            Continuation. The trade is the RETEST of the broken level.
    CHoCH — a swing is broken AGAINST the prevailing trend. Order flow has
            flipped. The trade is the retest in the NEW direction.

Both are breakout trades. They differ only in what the trend was beforehand,
which is why one engine can serve both and the regime decides which is allowed.

## Why the leg matters more than the event

An event says "structure broke". It does not say where to enter. Entering at the
break is chasing — the old engine's whole problem. What makes the trade is the
IMPULSE LEG the break produced and the retracement into it:

    origin ──────────────────► extreme        the leg
                      ◄──────                 the pullback
                   0.5 .. 0.618               where you enter

`reaction_engine`'s own journal is the evidence: trades that used the pullback
ran 51.4% win / +0.086R against 42.8% / -0.073R for those that did not, and its
`safe_bos` trigger (wait for the confirmed break) ran +0.254R against `fast`
(-0.139R). Three independent knobs, all pointing the same way, none of them a
direction bet.

Everything here is causal: an event is stamped at the bar whose CLOSE confirms
it, never at the swing it broke.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .levels import death_by_break, find_swings

BOS, CHOCH = "bos", "choch"
UP, DOWN = "up", "down"


@dataclass
class Breaks:
    """Structure break events, sorted by the bar that confirmed them."""
    bar: np.ndarray          # int64 — bar whose CLOSE broke the level
    level: np.ndarray        # float64 — the swing price that was broken
    direction: np.ndarray    # <U4 'up' | 'down'
    kind: np.ndarray         # <U5 'bos' | 'choch'
    origin: np.ndarray       # float64 — start of the impulse leg
    origin_bar: np.ndarray   # int64
    extreme: np.ndarray      # float64 — furthest the leg ran after the break
    extreme_bar: np.ndarray  # int64

    _F = ("bar", "level", "direction", "kind", "origin", "origin_bar",
          "extreme", "extreme_bar")

    def __len__(self) -> int:
        return len(self.bar)

    def leg(self) -> np.ndarray:
        """Signed leg size in price units."""
        return np.abs(self.extreme - self.origin)

    def pullback_zone(self, lo_frac: float = 0.50, hi_frac: float = 0.618):
        """The retracement band of each leg — where an entry is looked for.

        Returned as (near, far) where `near` is the shallower retrace. For an up
        leg both are below the extreme; for a down leg, above it.
        """
        size = self.leg()
        up = self.direction == UP
        near = np.where(up, self.extreme - size * lo_frac, self.extreme + size * lo_frac)
        far = np.where(up, self.extreme - size * hi_frac, self.extreme + size * hi_frac)
        return near, far

    def select(self, m) -> "Breaks":
        return Breaks(*[getattr(self, f)[m] for f in self._F])


def find_breaks(bars, k: int = 3, buffer_atr: float = 0.05,
                atr: np.ndarray | None = None) -> Breaks:
    """Every BOS and CHoCH over the series.

    A swing high confirmed at bar i+k is broken at the first later bar whose
    CLOSE exceeds it by `buffer_atr` ATRs. That buffer is the difference between
    a break and a spread blip; the old detector had none and counted any close
    beyond the level, which on gold means every wick-driven tick.

    Kind is assigned by the trend prevailing at the moment of the break, tracked
    forward through the event sequence — a break up while the trend is down is a
    CHoCH, the same break while the trend is up is a BOS.
    """
    from .context import atr as _atr
    if atr is None:
        atr = _atr(bars, 14)
    n = len(bars)
    hi_i, lo_i = find_swings(bars.high, bars.low, k)
    if len(hi_i) == 0 or len(lo_i) == 0:
        return Breaks(*[np.empty(0, t) for t in
                        (np.int64, np.float64, "<U4", "<U5", np.float64,
                         np.int64, np.float64, np.int64)])

    high = np.asarray(bars.high, np.float64)
    low = np.asarray(bars.low, np.float64)
    buf = np.asarray(atr, np.float64) * buffer_atr

    # when is each swing broken?
    hp, lp = high[hi_i], low[lo_i]
    h_born, l_born = hi_i + k, lo_i + k
    h_break = death_by_break(bars.close, hp, h_born, np.ones(len(hp), bool),
                             buf[np.clip(h_born, 0, n - 1)])
    l_break = death_by_break(bars.close, lp, l_born, np.zeros(len(lp), bool),
                             buf[np.clip(l_born, 0, n - 1)])

    # one event per swing that actually broke, merged and time-ordered
    ev_bar = np.concatenate([h_break, l_break])
    ev_lvl = np.concatenate([hp, lp])
    ev_dir = np.concatenate([np.full(len(hp), UP), np.full(len(lp), DOWN)])
    ev_swing = np.concatenate([hi_i, lo_i])
    alive = ev_bar < n
    ev_bar, ev_lvl, ev_dir, ev_swing = (ev_bar[alive], ev_lvl[alive],
                                        ev_dir[alive], ev_swing[alive])
    if not len(ev_bar):
        return Breaks(*[np.empty(0, t) for t in
                        (np.int64, np.float64, "<U4", "<U5", np.float64,
                         np.int64, np.float64, np.int64)])
    order = np.argsort(ev_bar, kind="stable")
    ev_bar, ev_lvl, ev_dir, ev_swing = (ev_bar[order], ev_lvl[order],
                                        ev_dir[order], ev_swing[order])

    # Collapse repeats: several stacked swings can break on the same bar. Keep
    # the FURTHEST level broken — that is the one that actually defined
    # structure — rather than whichever sorted first.
    keep = np.ones(len(ev_bar), bool)
    for j in range(1, len(ev_bar)):
        if ev_bar[j] == ev_bar[j - 1] and ev_dir[j] == ev_dir[j - 1]:
            prev = j - 1
            while not keep[prev]:
                prev -= 1
            better = (ev_lvl[j] > ev_lvl[prev] if ev_dir[j] == UP
                      else ev_lvl[j] < ev_lvl[prev])
            if better:
                keep[prev] = False
            else:
                keep[j] = False
    ev_bar, ev_lvl, ev_dir, ev_swing = (ev_bar[keep], ev_lvl[keep],
                                        ev_dir[keep], ev_swing[keep])

    # trend walk -> bos vs choch
    kinds = np.empty(len(ev_bar), "<U5")
    trend = None
    for j in range(len(ev_bar)):
        d = ev_dir[j]
        kinds[j] = CHOCH if (trend is not None and trend != d) else BOS
        trend = d

    # ── the impulse leg ─────────────────────────────────────────────────────
    # origin  = the opposing swing that preceded the broken one (where the leg
    #           started), so the leg spans the actual move, not just the break.
    # extreme = how far the leg ran before retracing halfway back. Measured
    #           forward from the break, which is legal because the entry that
    #           uses it happens after the retrace.
    origin = np.empty(len(ev_bar))
    origin_bar = np.empty(len(ev_bar), np.int64)
    extreme = np.empty(len(ev_bar))
    extreme_bar = np.empty(len(ev_bar), np.int64)

    # A leg is bounded: `max_leg_bars` caps the forward scan. Without it each
    # event accumulates over the whole remaining series, which is O(n) per event
    # and quadratic over the run — tolerable at 40k bars, hours at 1M. A leg that
    # has not half-retraced within this many bars is not an impulse any entry
    # would still be waiting on.
    max_leg_bars = 400
    # searchsorted beats boolean masking for "the swing just before this bar"
    lo_conf = lo_i + k
    hi_conf = hi_i + k

    for j in range(len(ev_bar)):
        b = int(ev_bar[j])
        e_hi = min(b + max_leg_bars, n)
        if ev_dir[j] == UP:
            p = int(np.searchsorted(lo_i, ev_swing[j], "left")) - 1
            while p >= 0 and lo_conf[p] > b:
                p -= 1
            o_bar = int(lo_i[p]) if p >= 0 else max(0, b - 50)
            origin[j], origin_bar[j] = low[o_bar], o_bar
            run = high[b:e_hi]
            peak = np.maximum.accumulate(run)
            trigger = peak - (peak - origin[j]) * 0.5
            done = np.flatnonzero(low[b:e_hi] <= trigger)
            end = int(done[0]) if len(done) else len(run) - 1
            e = int(np.argmax(run[:end + 1]))
            extreme[j], extreme_bar[j] = run[e], b + e
        else:
            p = int(np.searchsorted(hi_i, ev_swing[j], "left")) - 1
            while p >= 0 and hi_conf[p] > b:
                p -= 1
            o_bar = int(hi_i[p]) if p >= 0 else max(0, b - 50)
            origin[j], origin_bar[j] = high[o_bar], o_bar
            run = low[b:e_hi]
            trough = np.minimum.accumulate(run)
            trigger = trough + (origin[j] - trough) * 0.5
            done = np.flatnonzero(high[b:e_hi] >= trigger)
            end = int(done[0]) if len(done) else len(run) - 1
            e = int(np.argmin(run[:end + 1]))
            extreme[j], extreme_bar[j] = run[e], b + e

    return Breaks(bar=ev_bar.astype(np.int64), level=ev_lvl.astype(np.float64),
                  direction=ev_dir.astype("<U4"), kind=kinds,
                  origin=origin, origin_bar=origin_bar,
                  extreme=extreme, extreme_bar=extreme_bar)


def trend_state(bars, breaks: Breaks) -> np.ndarray:
    """Per-bar prevailing trend from the break sequence: +1 up, -1 down, 0 none.

    Steps only at a confirmed break, so it can be read at any bar without
    peeking. This is the `regime_engine` idea reduced to the one thing that
    measured: which way structure last broke.
    """
    out = np.zeros(len(bars), np.int8)
    if len(breaks) == 0:
        return out
    for j in range(len(breaks)):
        b = int(breaks.bar[j])
        out[b:] = 1 if breaks.direction[j] == UP else -1
    return out
