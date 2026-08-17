"""The complete candlestick pattern library — vectorised, ATR-scaled.

The old tree read five shapes (pin, engulfing, displacement, turtle soup,
spring) and the shipped engine collapsed even those into a single
`strength >= 5` boolean used by one playbook that fired 33 times in 14 years.
This is the full classical set so the study can find out which shapes, if any,
actually carry information on gold — instead of assuming the five somebody
happened to implement are the right five.

## Rules this file follows

**Every threshold is an ATR fraction.** A "long body" on a 124-pip day and a
1,322-pip day are not the same number of pips, and hard-coding one is the single
mistake that runs through the entire old codebase.

**No pattern carries a strength number.** The classical literature ranks these
shapes; that ranking has never been checked on gold M5 and every invented score
in the old tree measured |rho| < 0.08 against outcome. Patterns are LABELS here.
`research/reaction.py` and the stage ablation decide which ones pay.

**Context is not baked in.** A hammer and a hanging man are the same candle; only
the preceding trend differs. Rather than guess the trend inside the detector,
both are emitted as one shape and the caller supplies context — otherwise the
detector silently makes a directional call the study cannot see.

Bullish and bearish variants are returned separately because the trade direction
is the whole point; a pattern that only appears in one direction's book is a
finding, not a bug.
"""
from __future__ import annotations

import numpy as np

# ── shape thresholds, all in ATR or as ratios of the bar's own range ─────────
DOJI_BODY = 0.08          # body <= this * range  -> doji
SMALL_BODY = 0.30
LONG_BODY = 0.60
LONG_BODY_ATR = 0.70      # and the body must be this many ATRs to be "long"
LONG_WICK = 0.55          # wick >= this * range
TINY_WICK = 0.08          # marubozu tolerance
NEAR_EQ_ATR = 0.10        # tweezers / matching highs: within this many ATRs

BULL, BEAR = "bull", "bear"


def _shift(a, k):
    """a[i-k], edge-padded. Causal: only looks BACKWARD."""
    out = np.empty_like(a)
    if k <= 0:
        return a.copy()
    out[:k] = a[0]
    out[k:] = a[:-k]
    return out


class Patterns:
    """Every pattern as a boolean mask, keyed `name` -> array.

    Split into `.bull` and `.bear`. A bar may match several; `names_at` returns
    all of them so the study can slice on combinations rather than a priority
    order somebody guessed.
    """

    def __init__(self, bull: dict, bear: dict, meta: dict):
        self.bull = bull
        self.bear = bear
        self.meta = meta

    @property
    def all_names(self):
        return sorted(set(self.bull) | set(self.bear))

    def mask(self, name: str, direction: str) -> np.ndarray:
        d = self.bull if direction in ("buy", BULL) else self.bear
        return d.get(name)

    def names_at(self, i: int, direction: str) -> list:
        d = self.bull if direction in ("buy", BULL) else self.bear
        return [k for k, m in d.items() if m[i]]

    def any_at(self, i: int, direction: str) -> bool:
        d = self.bull if direction in ("buy", BULL) else self.bear
        return any(m[i] for m in d.values())

    def count(self) -> dict:
        return {k: int(self.bull.get(k, np.zeros(1)).sum()
                       + self.bear.get(k, np.zeros(1)).sum())
                for k in self.all_names}


def read(bars, atr: np.ndarray) -> Patterns:
    """Detect every pattern across the whole series."""
    o = np.asarray(bars.open, np.float64)
    h = np.asarray(bars.high, np.float64)
    l = np.asarray(bars.low, np.float64)
    c = np.asarray(bars.close, np.float64)
    a = np.asarray(atr, np.float64)
    n = len(o)

    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    br = body / rng
    up_w = h - np.maximum(o, c)
    dn_w = np.minimum(o, c) - l
    bull_bar = c > o
    bear_bar = c < o
    top = np.maximum(o, c)
    bot = np.minimum(o, c)
    mid = (o + c) / 2.0
    eq = a * NEAR_EQ_ATR

    # previous bars
    o1, h1, l1, c1 = _shift(o, 1), _shift(h, 1), _shift(l, 1), _shift(c, 1)
    o2, h2, l2, c2 = _shift(o, 2), _shift(h, 2), _shift(l, 2), _shift(c, 2)
    b1, b2 = _shift(body, 1), _shift(body, 2)
    br1 = _shift(br, 1)
    bull1, bear1 = _shift(bull_bar, 1), _shift(bear_bar, 1)
    bull2, bear2 = _shift(bull_bar, 2), _shift(bear_bar, 2)
    top1, bot1 = _shift(top, 1), _shift(bot, 1)
    mid1 = _shift(mid, 1)

    valid = np.ones(n, bool)
    valid[:3] = False        # first bars have no history

    long_body = (br >= LONG_BODY) & (body >= a * LONG_BODY_ATR)
    long_body1 = _shift(long_body, 1)
    long_body2 = _shift(long_body, 2)
    doji = br <= DOJI_BODY
    doji1 = _shift(doji, 1)
    small = br <= SMALL_BODY
    small1 = _shift(small, 1)

    bull_p, bear_p, meta = {}, {}, {}

    # ══ SINGLE BAR ══════════════════════════════════════════════════════════
    bull_p["doji"] = bear_p["doji"] = doji & valid
    bull_p["dragonfly_doji"] = doji & (dn_w >= rng * LONG_WICK) & (up_w <= rng * TINY_WICK) & valid
    bear_p["gravestone_doji"] = doji & (up_w >= rng * LONG_WICK) & (dn_w <= rng * TINY_WICK) & valid
    bull_p["long_legged_doji"] = bear_p["long_legged_doji"] = (
        doji & (up_w >= rng * 0.3) & (dn_w >= rng * 0.3) & valid)

    # hammer / hanging man share a shape; shooting star / inverted hammer likewise.
    # Emitted once each, direction assigned by which way the rejection points.
    bull_p["hammer"] = (dn_w >= rng * LONG_WICK) & small & (up_w <= rng * 0.2) & valid
    bear_p["shooting_star"] = (up_w >= rng * LONG_WICK) & small & (dn_w <= rng * 0.2) & valid

    bull_p["marubozu"] = bull_bar & (br >= 1 - 2 * TINY_WICK) & (body >= a * LONG_BODY_ATR) & valid
    bear_p["marubozu"] = bear_bar & (br >= 1 - 2 * TINY_WICK) & (body >= a * LONG_BODY_ATR) & valid

    bull_p["spinning_top"] = bear_p["spinning_top"] = (
        small & ~doji & (up_w >= rng * 0.25) & (dn_w >= rng * 0.25) & valid)

    bull_p["belt_hold"] = bull_bar & (dn_w <= rng * TINY_WICK) & long_body & valid
    bear_p["belt_hold"] = bear_bar & (up_w <= rng * TINY_WICK) & long_body & valid

    # displacement: a big committed body. The one shape the old tree relied on.
    bull_p["displacement"] = bull_bar & (br > 0.70) & (body > a) & valid
    bear_p["displacement"] = bear_bar & (br > 0.70) & (body > a) & valid

    # ══ TWO BAR ═════════════════════════════════════════════════════════════
    # engulfing: full RANGE engulf, as the old tree correctly required — a
    # body-only engulf is common noise
    full_engulf = (h > h1) & (l < l1) & valid
    bull_p["engulfing"] = full_engulf & bull_bar & bear1
    bear_p["engulfing"] = full_engulf & bear_bar & bull1

    inside = (h <= h1) & (l >= l1) & valid
    bull_p["harami"] = inside & bull_bar & bear1 & long_body1 & ~doji
    bear_p["harami"] = inside & bear_bar & bull1 & long_body1 & ~doji
    bull_p["harami_cross"] = inside & doji & bear1 & long_body1
    bear_p["harami_cross"] = inside & doji & bull1 & long_body1

    bull_p["tweezer_bottom"] = (np.abs(l - l1) <= eq) & bull_bar & bear1 & valid
    bear_p["tweezer_top"] = (np.abs(h - h1) <= eq) & bear_bar & bull1 & valid

    bull_p["piercing"] = (bull_bar & bear1 & long_body1 & (o < l1)
                          & (c > mid1) & (c < o1) & valid)
    bear_p["dark_cloud"] = (bear_bar & bull1 & long_body1 & (o > h1)
                            & (c < mid1) & (c > o1) & valid)

    bull_p["kicker"] = bull_bar & bear1 & (o > o1) & (l > h1) & valid
    bear_p["kicker"] = bear_bar & bull1 & (o < o1) & (h < l1) & valid

    bull_p["matching_low"] = (np.abs(c - c1) <= eq) & bear1 & bear_bar & valid
    bear_p["matching_high"] = (np.abs(c - c1) <= eq) & bull1 & bull_bar & valid

    bull_p["separating_lines"] = bull_bar & bear1 & (np.abs(o - o1) <= eq) & valid
    bear_p["separating_lines"] = bear_bar & bull1 & (np.abs(o - o1) <= eq) & valid

    # ══ THREE BAR ═══════════════════════════════════════════════════════════
    # morning star: long down, small-bodied gap-ish middle, long up closing past
    # the midpoint of bar 1
    star_mid = small1
    bull_p["morning_star"] = (valid & bear2 & long_body2 & star_mid
                              & bull_bar & long_body
                              & (c > (o2 + c2) / 2.0) & (bot1 < bot))
    bear_p["evening_star"] = (valid & bull2 & long_body2 & star_mid
                              & bear_bar & long_body
                              & (c < (o2 + c2) / 2.0) & (top1 > top))
    bull_p["morning_doji_star"] = bull_p["morning_star"] & doji1
    bear_p["evening_doji_star"] = bear_p["evening_star"] & doji1

    # abandoned baby: the star gaps away on BOTH sides — rare and strict
    bull_p["abandoned_baby"] = (bull_p["morning_star"] & doji1
                                & (h1 < l2) & (h1 < l))
    bear_p["abandoned_baby"] = (bear_p["evening_star"] & doji1
                                & (l1 > h2) & (l1 > h))

    bull_p["three_soldiers"] = (valid & bull_bar & bull1 & bull2
                                & (c > c1) & (c1 > c2)
                                & (o < c1) & (o > o1) & long_body & long_body1)
    bear_p["three_crows"] = (valid & bear_bar & bear1 & bear2
                             & (c < c1) & (c1 < c2)
                             & (o > c1) & (o < o1) & long_body & long_body1)

    inside2 = (h1 <= h2) & (l1 >= l2)
    bull_p["three_inside_up"] = valid & bear2 & inside2 & bull1 & bull_bar & (c > h2)
    bear_p["three_inside_down"] = valid & bull2 & inside2 & bear1 & bear_bar & (c < l2)

    outside2 = (h1 > h2) & (l1 < l2)
    bull_p["three_outside_up"] = valid & bear2 & outside2 & bull1 & bull_bar & (c > c1)
    bear_p["three_outside_down"] = valid & bull2 & outside2 & bear1 & bear_bar & (c < c1)

    bull_p["tri_star"] = bear_p["tri_star"] = valid & doji & doji1 & _shift(doji, 2)

    # ══ SWEEP SHAPES (level-free versions; the level-aware ones live in sweep.py)
    depth = a * 0.15
    bull_p["turtle_soup"] = valid & (l < l1 - depth) & (c > l1) & bull_bar
    bear_p["turtle_soup"] = valid & (h > h1 + depth) & (c < h1) & bear_bar
    bull_p["spring"] = valid & (l < l1) & (c > h1) & bull_bar
    bear_p["upthrust"] = valid & (h > h1) & (c < l1) & bear_bar

    meta["stop_long"] = np.minimum(l, l1) - a * 0.10
    meta["stop_short"] = np.maximum(h, h1) + a * 0.10
    meta["body_atr"] = body / np.maximum(a, 1e-12)
    meta["range_atr"] = rng / np.maximum(a, 1e-12)

    # drop anything that never fires — keeps the study tables readable
    bull_p = {k: v for k, v in bull_p.items() if v.any()}
    bear_p = {k: v for k, v in bear_p.items() if v.any()}
    return Patterns(bull_p, bear_p, meta)


# Classical groupings, for slicing results. Membership only — no weights.
REVERSAL = ("hammer", "shooting_star", "engulfing", "harami", "harami_cross",
            "piercing", "dark_cloud", "morning_star", "evening_star",
            "morning_doji_star", "evening_doji_star", "abandoned_baby",
            "three_inside_up", "three_inside_down", "three_outside_up",
            "three_outside_down", "tweezer_bottom", "tweezer_top",
            "turtle_soup", "spring", "upthrust", "kicker",
            "dragonfly_doji", "gravestone_doji")
CONTINUATION = ("marubozu", "belt_hold", "displacement", "three_soldiers",
                "three_crows", "separating_lines")
INDECISION = ("doji", "long_legged_doji", "spinning_top", "tri_star",
              "matching_low", "matching_high")
