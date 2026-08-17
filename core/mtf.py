"""Every timeframe's candle state, at every base bar.

The operator's description of what a decision actually needs:

    "if trading m15 break then we need look h1 h4 m5 everyone how that break
     might take with m15 ... whats the previous candle did both h4 h1 m15 m30
     then whats likely to break whats the momentum in every time frame"

Nothing in this tree could answer that. `core/bias.py` gives H4/H1 *structure
direction* — a single -1/0/+1 — which was the discovery that moved the swing
system, but it is one number per timeframe. It cannot say the H1 candle is 40
minutes old, has already taken the previous H1 high, and is closing on its lows.
That is a completely different and much richer statement, and it is the one an
intraday decision is made from.

## What this computes, per timeframe, per base bar

    PREVIOUS closed candle    open/high/low/close + full anatomy (core/shape.py)
    FORMING candle            its open, its running high/low so far, how many
                              minutes old it is, and whether it has ALREADY
                              taken the previous candle's high or low
    location                  where price sits relative to that candle's open
                              and relative to the previous candle's range

The forming-candle state is the part that does not exist anywhere else and is
most of the point. "The H1 has taken the prior H1 low and is now back above it,
22 minutes in" is a complete intraday setup description; "h1_dir = -1" is not.

## Grouping, and why it is done this way

Base (M5) bars are grouped into each higher timeframe's candles:

  * **M15 / M30** by wall-clock bucket (`t // 900`, `t // 1800`). Exact and
    unambiguous — these boundaries are :00/:15/:30/:45 for every broker alive.
  * **H1 / H4** by `Context.align`, i.e. the BROKER's own candle boundaries.
    H4 is convention-dependent (gold's H4 does not start at 00:00 UTC on most
    servers) so bucketing it by `t // 14400` would silently invent candles that
    no chart shows. `core/store.py` refuses to resample for the same reason.

Both produce a non-decreasing group id, and one shared routine handles the rest.

## Causality

`prev_*` is the last FULLY CLOSED candle of that timeframe. `cur_*` is the
forming candle's state using base bars up to and including i — which is legal,
because those bars have closed even though their parent candle has not. What is
never available is the forming candle's final high/low/close. Reading those is
the multi-timeframe lookahead that manufactures enormous fake edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import shape as SH

# The timeframes an intraday decision reads. M5 is the base and is described by
# core/shape.py directly, so it is not repeated here.
TFS = ("M15", "M30", "H1", "H4")
TF_SECONDS = {"M15": 900, "M30": 1800, "H1": 3600, "H4": 14400}


def _dense(g: np.ndarray, t: np.ndarray | None = None,
           max_gap: int | None = None) -> np.ndarray:
    """Renumber a non-decreasing group id to 0,1,2,... with no gaps.

    `max_gap` additionally starts a new group wherever the base series itself
    jumps — which is not cosmetic. Grouping H1 purely by `Context.align` makes
    the whole weekend one "forming candle": no H1 closes between Friday's last
    bar and Monday's first, so Monday morning's running high was coming back as
    the max of Friday afternoon AND Monday. It showed up as a forming candle
    4,740 minutes old. Any gap longer than a few base bars ends the candle.
    """
    change = np.diff(g) != 0
    if t is not None and max_gap:
        change |= np.diff(t) > max_gap
    return np.cumsum(np.concatenate(([0], change.astype(np.int64))))


def _grouped_cummax(x: np.ndarray, g: np.ndarray, big: float) -> np.ndarray:
    """Running max within each group, vectorized.

    Offsetting each group by `big` makes every value in group k exceed every
    value in group k-1, so a plain accumulate cannot carry a stale maximum
    across a boundary. Subtracting the offset afterwards recovers the real
    number. Same trick with the sign flipped gives the running min. This matters
    because the alternative — a Python loop over ~370,000 M15 groups — is the
    kind of thing that made the old harness run at 12 bars/sec.
    """
    return np.maximum.accumulate(x + g * big) - g * big


def _grouped_cummin(x: np.ndarray, g: np.ndarray, big: float) -> np.ndarray:
    return -(np.maximum.accumulate(-x + g * big) - g * big)


@dataclass
class TFState:
    """One timeframe's candle state, aligned to the base bar index."""
    tf: str
    group: np.ndarray         # which candle of this TF each base bar belongs to
    # the last fully closed candle of this timeframe
    prev_open: np.ndarray
    prev_high: np.ndarray
    prev_low: np.ndarray
    prev_close: np.ndarray
    prev_shape: dict          # feature name -> array, from core/shape.py
    # the candle currently forming
    cur_open: np.ndarray
    cur_high: np.ndarray      # running, base bars 0..i only
    cur_low: np.ndarray
    minutes_into: np.ndarray  # how old the forming candle is
    frac_into: np.ndarray     # 0..1 through the candle — is it about to close?
    took_prev_high: np.ndarray   # forming candle has already run the prior high
    took_prev_low: np.ndarray
    # location, all in base ATRs
    from_cur_open: np.ndarray    # signed: where price is vs this candle's open
    from_prev_high: np.ndarray
    from_prev_low: np.ndarray
    prev_range_pos: np.ndarray   # 0 = at prev low, 1 = at prev high

    def row(self, i: int) -> dict:
        p = self.tf.lower() + "_"
        out = {
            f"{p}from_open": float(self.from_cur_open[i]),
            f"{p}from_prev_high": float(self.from_prev_high[i]),
            f"{p}from_prev_low": float(self.from_prev_low[i]),
            f"{p}prev_range_pos": float(self.prev_range_pos[i]),
            f"{p}minutes_into": int(self.minutes_into[i]),
            f"{p}frac_into": float(self.frac_into[i]),
            f"{p}took_prev_high": bool(self.took_prev_high[i]),
            f"{p}took_prev_low": bool(self.took_prev_low[i]),
        }
        for k, arr in self.prev_shape.items():
            v = arr[i]
            out[f"{p}prev_{k}"] = (int(v) if arr.dtype.kind in "ib"
                                   else float(v))
        return out


def _state_for(tf: str, g_raw: np.ndarray, o, h, l, c, t, atr, base_seconds):
    """Build one timeframe's state from base bars grouped by `g_raw`."""
    n = len(o)
    g = _dense(np.asarray(g_raw, np.int64), t, max_gap=base_seconds * 3)
    starts = np.flatnonzero(np.concatenate(([True], np.diff(g) != 0)))
    ends = np.concatenate((starts[1:], [n]))

    span = float(np.nanmax(h) - np.nanmin(l))
    big = (span if np.isfinite(span) and span > 0 else 1.0) * 2.0 + 1.0

    # ── per-group aggregates (the candles themselves) ───────────────────────
    g_open = o[starts]
    g_high = np.maximum.reduceat(h, starts)
    g_low = np.minimum.reduceat(l, starts)
    g_close = c[ends - 1]

    # ── the previous CLOSED candle, published across the forming one ────────
    # group k's bars see candle k-1 as the last closed one.
    pv_o = np.concatenate(([np.nan], g_open[:-1]))[g]
    pv_h = np.concatenate(([np.nan], g_high[:-1]))[g]
    pv_l = np.concatenate(([np.nan], g_low[:-1]))[g]
    pv_c = np.concatenate(([np.nan], g_close[:-1]))[g]

    # ── anatomy of that previous candle ─────────────────────────────────────
    # shape.read wants a Bars-like object; the candle series is exactly the
    # per-group aggregates, and its ATR scale is the base ATR at each group's
    # start so the numbers stay comparable across timeframes.
    class _Series:
        """Minimal duck-typed bar series — shape.read only reads OHLC."""
        __slots__ = ("open", "high", "low", "close")

    s = _Series()
    s.open, s.high, s.low, s.close = g_open, g_high, g_low, g_close
    sh = SH.read(s, np.maximum(atr[starts], 1e-12))

    # Anatomy of candle k-1, published across candle k. `sh` is indexed by
    # candle, so shift by one and then expand back onto the base bar index.
    prev_shape = {}
    for name in SH.SHAPE_FIELDS:
        full = np.asarray(getattr(sh, name))
        shifted = np.concatenate((full[:1], full[:-1]))
        prev_shape[SH.SHORT_NAME.get(name, name)] = shifted[g]

    # ── the forming candle ──────────────────────────────────────────────────
    cur_open = g_open[g]
    cur_high = _grouped_cummax(h, g, big)
    cur_low = _grouped_cummin(l, g, big)
    t_start = t[starts][g]
    secs = (t - t_start) + base_seconds
    minutes = np.clip(secs // 60, 0, 32767).astype(np.int16)
    frac = np.clip(secs / float(TF_SECONDS.get(tf, base_seconds)), 0.0, 1.0)

    a = np.maximum(atr, 1e-12)
    prng = np.where(np.isfinite(pv_h - pv_l) & (pv_h - pv_l > 1e-12),
                    pv_h - pv_l, np.nan)

    return TFState(
        tf=tf, group=g,
        prev_open=pv_o, prev_high=pv_h, prev_low=pv_l, prev_close=pv_c,
        prev_shape=prev_shape,
        cur_open=cur_open, cur_high=cur_high, cur_low=cur_low,
        minutes_into=minutes, frac_into=frac,
        took_prev_high=np.nan_to_num(cur_high > pv_h, nan=0).astype(bool),
        took_prev_low=np.nan_to_num(cur_low < pv_l, nan=0).astype(bool),
        from_cur_open=(c - cur_open) / a,
        from_prev_high=(c - pv_h) / a,
        from_prev_low=(c - pv_l) / a,
        prev_range_pos=np.where(np.isfinite(prng), (c - pv_l) / prng, np.nan))


def build(bars, ctx, atr: np.ndarray, tfs=TFS) -> dict:
    """Every timeframe's candle state, keyed by timeframe name.

    `ctx` supplies the broker's real H1/H4 boundaries via `align`. M15/M30 are
    bucketed by wall clock, which is exact for those two.
    """
    t = np.asarray(bars.time, np.int64)
    o = np.asarray(bars.open, np.float64)
    h = np.asarray(bars.high, np.float64)
    l = np.asarray(bars.low, np.float64)
    c = np.asarray(bars.close, np.float64)
    a = np.asarray(atr, np.float64)

    out = {}
    for tf in tfs:
        if tf in ("M15", "M30"):
            g = t // TF_SECONDS[tf]
        else:
            if tf not in ctx.series:
                continue
            g = np.asarray(ctx.align(tf), np.int64)
            # before the first close of that timeframe there is no candle yet;
            # clamp so the grouping stays non-decreasing and those bars simply
            # carry NaN previous-candle values.
            g = np.maximum(g, -1)
        out[tf] = _state_for(tf, g, o, h, l, c, t, a, bars.bar_seconds)
    return out


def row(states: dict, i: int) -> dict:
    """Flatten every timeframe's state at bar i into one feature dict."""
    d = {}
    for st in states.values():
        d.update(st.row(i))
    return d
