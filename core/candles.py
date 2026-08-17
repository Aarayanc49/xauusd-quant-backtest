"""The market candle read — and the tight structural stop it buys.

Ported from the old tree's `confirmation.py`, which was the best-reasoned module
in that codebase and was reduced in the shipped engine to a single
`strength >= 5` boolean used by one playbook that fired 33 times in 14 years.

Its own docstring stated the rule the shipped engine then ignored:

    SL RULE: Stop goes behind the confirmation candle's extreme.
    Not behind some HTF level 400 pips away. BEHIND THE CANDLE.

ScanTrader used a flat 60 pips on every trade instead. That single substitution is
most of why R:R was never better than 1:1.5 — a fixed stop cannot produce a good
reward ratio, because the reward is structural and the risk is arbitrary.

## What each pattern is claiming

Every pattern here is a claim about *who just lost money and has to get out*:

  * **turtle soup** — price ran a level, tripped the stops resting beyond it, and
    immediately closed back. The trapped breakout traders are the fuel. Strength 6:
    it is the only pattern that requires a liquidity event to have happened.
  * **engulfing** — the entire prior bar's range was taken out and reversed in one
    bar. Commitment, not drift.
  * **displacement** — a body larger than `min_body_atr` ATRs with little wick.
    Institutional size leaves this footprint; retail does not.
  * **spring / upthrust** — Wyckoff. Broke the zone, closed back through the far
    side. Stronger than a pin because the close proves the reclaim.
  * **pin** — a long rejection wick. Weakest, because a wick alone is a claim
    without a close behind it.

## What changed in the port

  1. **Every threshold is an ATR fraction.** The original used `2 * pip_value`
     buffers and a flat `max_sl_pips = 80`. Same fault as everywhere else: one
     constant cannot serve a 124-pip day and a 1,322-pip day.
  2. **Vectorized over the whole series.** The original scanned the last 5 candles
     on every call, per zone, per direction — fine live, useless for research.
  3. **`strength` is kept as a LABEL, not a score.** The old tree's ranking by
     strength was never validated against outcome, and every other invented score
     in that codebase measured |rho| < 0.08 against result. Which patterns
     actually pay is a question for research/reaction.py, not a constant here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pattern names and the old tree's strength labels. Retained for slicing results
# by pattern; deliberately NOT summed into a score.
PIN = "pin"
ENGULF = "engulf"
DISPLACEMENT = "displacement"
TURTLE_SOUP = "turtle_soup"
SPRING = "spring"           # upthrust is the bearish mirror, same name

STRENGTH = {TURTLE_SOUP: 6, ENGULF: 5, DISPLACEMENT: 5, SPRING: 5, PIN: 4}


@dataclass
class Signals:
    """Per-bar boolean pattern masks plus the structural stop each one implies.

    `stop_long` / `stop_short` are the price a stop would sit at IF that bar were
    taken in that direction — the candle's own extreme plus a buffer. NaN where no
    pattern fired. This is the number the whole exercise is about.
    """
    bull: dict           # pattern -> bool mask
    bear: dict
    stop_long: np.ndarray
    stop_short: np.ndarray

    def any_bull(self) -> np.ndarray:
        return np.logical_or.reduce(list(self.bull.values()))

    def any_bear(self) -> np.ndarray:
        return np.logical_or.reduce(list(self.bear.values()))

    def name_at(self, i: int, direction: str) -> str:
        d = self.bull if direction == "buy" else self.bear
        hits = [k for k, m in d.items() if m[i]]
        if not hits:
            return ""
        return max(hits, key=lambda k: STRENGTH.get(k, 0))


def read(bars, atr: np.ndarray, buffer_atr: float = 0.10,
         pin_wick: float = 0.55, pin_body: float = 0.35,
         displ_body_ratio: float = 0.70, displ_body_atr: float = 1.00,
         soup_depth_atr: float = 0.15) -> Signals:
    """Detect every confirmation pattern across a whole series.

    `atr` must be causal and aligned to the bar index. Patterns reference bar
    i and i-1 only, so a mask at i is legal to act on at bar i's close.

    Note `displ_body_atr` defaults to 1.00 where the original used 1.5. The
    original was scanning M5 with an M5 ATR and rarely fired; 1.0 ATR of body with
    70% body ratio is already an unusual bar. This is a parameter, not a belief —
    sweep it.
    """
    o = np.asarray(bars.open, np.float64)
    h = np.asarray(bars.high, np.float64)
    l = np.asarray(bars.low, np.float64)
    c = np.asarray(bars.close, np.float64)
    n = len(o)
    a = np.asarray(atr, np.float64)
    buf = a * buffer_atr

    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    body_ratio = body / rng
    up_wick = h - np.maximum(o, c)
    dn_wick = np.minimum(o, c) - l
    is_bull = c > o
    is_bear = c < o

    prev = np.arange(n) - 1
    prev[0] = 0
    ph, pl, po, pc = h[prev], l[prev], o[prev], c[prev]
    first = np.zeros(n, bool)
    first[0] = True

    bull, bear = {}, {}

    # ── pin ─────────────────────────────────────────────────────────────────
    bull[PIN] = (dn_wick > rng * pin_wick) & (body_ratio < pin_body)
    bear[PIN] = (up_wick > rng * pin_wick) & (body_ratio < pin_body)

    # ── engulfing ───────────────────────────────────────────────────────────
    # True RANGE engulf, as the original had it: the bar must take out the prior
    # bar's high AND low, not merely its body. A body-only engulf is common noise.
    engulfed = (h > ph) & (l < pl) & ~first
    bull[ENGULF] = engulfed & is_bull & (pc < po)
    bear[ENGULF] = engulfed & is_bear & (pc > po)

    # ── displacement ────────────────────────────────────────────────────────
    displ = (body_ratio > displ_body_ratio) & (body > a * displ_body_atr)
    bull[DISPLACEMENT] = displ & is_bull
    bear[DISPLACEMENT] = displ & is_bear

    # ── turtle soup / spring / upthrust ─────────────────────────────────────
    # These need a LEVEL to sweep, so the level-relative versions live in
    # `at_level` below. What can be said series-wide is the sweep of the PRIOR
    # BAR's extreme, which is the minimal liquidity event: the stops sitting under
    # the last bar's low were taken and price closed back above.
    depth = a * soup_depth_atr
    bull[TURTLE_SOUP] = (~first & (l < pl - depth) & (c > pl) & is_bull)
    bear[TURTLE_SOUP] = (~first & (h > ph + depth) & (c < ph) & is_bear)

    # ── spring / upthrust vs the prior bar's range ──────────────────────────
    bull[SPRING] = (~first & (l < pl) & (c > ph) & is_bull)
    bear[SPRING] = (~first & (h > ph) & (c < pl) & is_bear)

    # ── the structural stop ─────────────────────────────────────────────────
    # Below the LOWER of this bar and the prior bar, for a long. The original
    # called this the ICT model and it matters: the prior bar is often the sweep
    # candle whose wick is the true invalidation, and stopping only behind the
    # confirmation bar puts the stop inside the move that just happened.
    stop_long = np.minimum(l, np.where(first, l, pl)) - buf
    stop_short = np.maximum(h, np.where(first, h, ph)) + buf

    return Signals(bull=bull, bear=bear, stop_long=stop_long, stop_short=stop_short)


def at_level(bars, atr: np.ndarray, level: float, lo: int, hi: int,
             soup_depth_atr: float = 0.15) -> tuple:
    """Level-relative turtle soup and spring/upthrust over [lo, hi).

    This is the version that matters — a sweep of the PRIOR BAR is a minor event,
    a sweep of a real level is the trade. Returns (bull_mask, bear_mask,
    stop_long, stop_short) as offsets within the window.
    """
    h = np.asarray(bars.high[lo:hi], np.float64)
    l = np.asarray(bars.low[lo:hi], np.float64)
    o = np.asarray(bars.open[lo:hi], np.float64)
    c = np.asarray(bars.close[lo:hi], np.float64)
    a = np.asarray(atr[lo:hi], np.float64)
    depth = a * soup_depth_atr

    # swept the level and closed back on the origin side, with a body confirming
    bull = (l < level - depth) & (c > level) & (c > o)
    bear = (h > level + depth) & (c < level) & (c < o)

    n = len(h)
    prev = np.arange(n) - 1
    prev[0] = 0
    stop_long = np.minimum(l, l[prev]) - a * 0.10
    stop_short = np.maximum(h, h[prev]) + a * 0.10
    return bull, bear, stop_long, stop_short
