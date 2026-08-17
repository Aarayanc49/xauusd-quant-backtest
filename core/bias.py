"""Multi-timeframe bias — the cascade the operator actually trades.

In the operator's words: gold finds a bias, then you take the intraday position
inside it. H4 and H1 set the direction from their own swing structure and
pullbacks; M15/M5/M1 are where you find the entry once you know which way the
market is trying to go. Support and resistance on their own mean little — "what
makes it support, nothing really" — the direction is what makes a level tradeable.

**Neither the old tree nor this rebuild has ever done that.** `regime_engine`
computed exactly this (7 phases with buy_allowed/sell_allowed per phase) and
ScanTrader explicitly ignored it — "Top-down is CONTEXT, not a veto." My rebuild
reproduced the same omission by accident: `Rules.require_trend_agreement` defaults
to False and the one ablation labelled "trend agreement" tested BOS-vs-CHoCH, not
higher-timeframe direction. So the single thing the operator says makes the money
is the one axis that has never been measured here.

## What this computes, per base bar

    h4_dir, h1_dir     -1 / 0 / +1 from each timeframe's own swing structure
    aligned            both agree and the trade direction matches
    h4_pos, h1_pos     where price sits inside that timeframe's current leg,
                       0 = at the leg's origin, 1 = at its extreme. A pullback
                       is a LOW number in the direction of the bias.
    h4_pullback        price has retraced into 0.3-0.7 of the H4 leg — the
                       "buy the dip in an uptrend" location
    stack              how many of H4/H1 agree with the trade (0, 1 or 2)

Everything resolves through `Context.align`, so a decision on an M5 bar can only
read higher-timeframe bars that have actually closed. Aligning by index instead is
the classic multi-timeframe lookahead and it manufactures enormous fake edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .context import Context
from .structure import find_breaks, trend_state


@dataclass
class Bias:
    """Per-base-bar multi-timeframe state."""
    h4_dir: np.ndarray        # int8  -1 / 0 / +1
    h1_dir: np.ndarray
    h4_pos: np.ndarray        # float 0..1 position within the current H4 leg
    h1_pos: np.ndarray
    h4_leg_atr: np.ndarray    # float size of the current H4 leg, in base ATRs

    def stack_for(self, i: int, direction: str) -> int:
        """How many higher timeframes agree with `direction` at bar i (0-2)."""
        want = 1 if direction == "buy" else -1
        return int(self.h4_dir[i] == want) + int(self.h1_dir[i] == want)

    def aligned(self, direction: str) -> np.ndarray:
        want = 1 if direction == "buy" else -1
        return (self.h4_dir == want) & (self.h1_dir == want)


def build(series: dict, ctx: Context, base: str = "M5",
          tfs=("H4", "H1")) -> Bias:
    """Compute the bias cascade for every base bar."""
    n = len(ctx.bars)
    out = {}

    for tf in tfs:
        if tf not in series:
            out[tf] = (np.zeros(n, np.int8), np.full(n, 0.5), np.zeros(n))
            continue
        b = series[tf]
        atr_tf = ctx.scales[tf].atr
        brk = find_breaks(b, k=3, atr=atr_tf)
        tr = trend_state(b, brk)               # per-TF-bar trend, causal

        # Where is price inside the CURRENT leg on this timeframe? A pullback in
        # an uptrend is price low within a leg that is pointing up — which is the
        # thing the operator is looking for and the thing an M5-only engine
        # cannot see at all.
        hi = np.asarray(b.high, np.float64)
        lo = np.asarray(b.low, np.float64)
        cl = np.asarray(b.close, np.float64)
        origin = np.full(len(b), np.nan)
        extreme = np.full(len(b), np.nan)
        for j in range(len(brk)):
            s = int(brk.bar[j])
            e = int(brk.bar[j + 1]) if j + 1 < len(brk) else len(b)
            origin[s:e] = brk.origin[j]
            extreme[s:e] = brk.extreme[j]
        span = extreme - origin
        with np.errstate(invalid="ignore", divide="ignore"):
            pos = np.where(np.abs(span) > 1e-12, (cl - origin) / span, 0.5)
        pos = np.clip(np.nan_to_num(pos, nan=0.5), -0.5, 1.5)
        leg = np.abs(np.nan_to_num(span))

        # map TF -> base bars, causally
        m = ctx.align(tf)
        ok = m >= 0
        dir_b = np.zeros(n, np.int8)
        pos_b = np.full(n, 0.5)
        leg_b = np.zeros(n)
        dir_b[ok] = tr[m[ok]]
        pos_b[ok] = pos[m[ok]]
        leg_b[ok] = leg[m[ok]]
        out[tf] = (dir_b, pos_b, leg_b)

    h4d, h4p, h4l = out.get("H4", (np.zeros(n, np.int8), np.full(n, 0.5), np.zeros(n)))
    h1d, h1p, _ = out.get("H1", (np.zeros(n, np.int8), np.full(n, 0.5), np.zeros(n)))
    base_atr = ctx.scales[base].atr
    return Bias(h4_dir=h4d, h1_dir=h1d, h4_pos=h4p, h1_pos=h1p,
                h4_leg_atr=np.divide(h4l, np.maximum(base_atr, 1e-12)))


def pullback_depth(pos: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """How deep the retracement is, 0 = at the extreme, 1 = back at the origin.

    Expressed so it means the same thing in both directions: in an uptrend
    (direction +1) `pos` near 1 is at the highs, so depth = 1 - pos. In a
    downtrend the leg runs the other way and `pos` is already measured from its
    own origin, so the same formula holds.
    """
    return np.clip(1.0 - pos, 0.0, 1.5)
