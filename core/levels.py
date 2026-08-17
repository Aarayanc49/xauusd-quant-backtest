"""Level sources — and the type split the old tree never made.

Design rule 1: **a level is a PRICE, a zone is a REGION, and they never share a
struct.** In the old tree nine sources pushed into one dict: seven emitted a fixed
10-pip line, while FVGs and order blocks (32% of all points) emitted their own
width — FVG median 37p, max 931p. Downstream code could not tell them apart, so a
931-pip FVG entered clustering as a "level" and dragged a third of the day's range
into the zone it produced.

Here:
  * `Points` are prices. An entry triggers off one, and only off one.
  * `Zones` are regions. They may confirm or veto; they can never define a
    trigger price.

Everything is computed ONCE over a whole series and returned as flat arrays with
a `born` index — the bar at which the level first becomes knowable. A study or a
backtest at bar i takes the slice `born <= i`. That is what makes the research
engine fast, and it is also what makes lookahead structurally hard: a swing high
at bar 100 confirmed by 5 bars carries born=105, so no decision before bar 105 can
see it. Detectors that cannot state an honest `born` do not belong in this file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .context import rolling_mean
from .store import Bars

# ── source taxonomy ─────────────────────────────────────────────────────────
# The DECISION / MAGNET split is Osler (2003, J.Finance): take-profit orders
# cluster AT round numbers and reverse price, stop orders cluster JUST BEYOND them
# and cascade. Fading a magnet is fading into a stop cascade.
#
# The old tree's journal measured that split and then largely ignored it:
#   DECISION  D1 untested +78.6 · D1 fib +62.2 · H4 swing +36.6 · round +33.9
#   MAGNET    liquidity pool -35.3 (19% win, n=134) · PDH/PDL -21.2
#
# But the context-free version of that measurement is invalid — sources do
# opposite things in different regimes. `liquidity_pool` splits +0.978 in a
# pullback and -0.638 in an impulse: during an impulse the pools get run, during a
# pullback they hold. Weights come from research/reaction.py, conditioned on
# regime. The labels below are taxonomy only. They carry no strength.

DECISION = "decision"
MAGNET = "magnet"

SOURCE_CLASS = {
    "swing_h": DECISION, "swing_l": DECISION,
    "eqh": MAGNET, "eql": MAGNET,
    "round": DECISION, "half_round": DECISION,
    "pdh": MAGNET, "pdl": MAGNET, "pwh": MAGNET, "pwl": MAGNET,
    "session_h": MAGNET, "session_l": MAGNET,
    "fib": DECISION,
    "ob": DECISION, "fvg": DECISION,       # zone sources
}


@dataclass
class Points:
    """Flat arrays of price levels. All same length, index-aligned.

    `dead` is as important as `born`. The old tree had no concept of a level
    expiring — `discover_levels` re-derived everything from scratch each scan and
    bounded the result only by distance from price (300 pips). Carried into a
    born-indexed store that becomes a silent disaster: every prior-day high ever
    printed stays "active" forever, and a round number coming back into range
    emits a fresh point each time, so the active set grows without limit and
    confluence counts measure how long the study has been running.

    A level dies when the thing that made it a level stops being true — price
    closes decisively through it, the gap fills, the period it summarised is two
    periods stale. Detectors that cannot state an honest death must emit
    `dead = n` (immortal) deliberately, not by omission.
    """
    price: np.ndarray        # float64
    born: np.ndarray         # int64 — first bar index at which this is knowable
    dead: np.ndarray         # int64 — first bar at which it no longer applies
    source: np.ndarray       # <U12
    tf: np.ndarray           # <U3
    ref: np.ndarray          # int64 — bar index the level derives from (<= born)

    _F = ("price", "born", "dead", "source", "tf", "ref")

    def __len__(self) -> int:
        return len(self.price)

    def __add__(self, other: "Points") -> "Points":
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other
        return Points(*[np.concatenate([getattr(self, f), getattr(other, f)])
                        for f in self._F])

    def active(self, i: int) -> "Points":
        """The subset knowable AND still applicable at bar i."""
        m = (self.born <= i) & (self.dead > i)
        return Points(*[getattr(self, f)[m] for f in self._F])

    def near(self, i: int, price: float, within: float) -> "Points":
        """Active at bar i and within `within` price units of `price`. A level
        200 pips away is not a candidate for anything and only dilutes counts.

        Price-indexed: a full-array mask costs O(all points) per call, and the
        research loop calls this once per cluster refresh — 85k times over 14
        years against 93k points, which is 24 billion element ops. Sorting by
        price once and binary-searching the window makes it O(log n + window).
        """
        idx = self._price_index()
        p = self.price[idx]
        lo = int(np.searchsorted(p, price - within, "left"))
        hi = int(np.searchsorted(p, price + within, "right"))
        if hi <= lo:
            return Points.empty()
        sel = idx[lo:hi]
        keep = (self.born[sel] <= i) & (self.dead[sel] > i)
        sel = sel[keep]
        return Points(*[getattr(self, f)[sel] for f in self._F])

    def _price_index(self) -> np.ndarray:
        cached = getattr(self, "_pidx", None)
        if cached is None or len(cached) != len(self.price):
            cached = np.argsort(self.price, kind="stable")
            object.__setattr__(self, "_pidx", cached)
        return cached

    def sort_by_born(self) -> "Points":
        o = np.argsort(self.born, kind="stable")
        return Points(*[getattr(self, f)[o] for f in self._F])

    @staticmethod
    def empty() -> "Points":
        return Points(np.empty(0), np.empty(0, np.int64), np.empty(0, np.int64),
                      np.empty(0, "<U12"), np.empty(0, "<U3"), np.empty(0, np.int64))


@dataclass
class Zones:
    """Flat arrays of price regions. `dead` is the bar at which the region is
    invalidated (filled / mitigated / traded through), or len(bars) if still
    live at the end of the series."""
    high: np.ndarray
    low: np.ndarray
    born: np.ndarray
    dead: np.ndarray
    source: np.ndarray
    tf: np.ndarray
    side: np.ndarray         # <U7 'bullish' | 'bearish'

    _F = ("high", "low", "born", "dead", "source", "tf", "side")

    def __len__(self) -> int:
        return len(self.high)

    def __add__(self, other: "Zones") -> "Zones":
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other
        return Zones(*[np.concatenate([getattr(self, f), getattr(other, f)])
                       for f in self._F])

    def active(self, i: int) -> "Zones":
        m = (self.born <= i) & (self.dead > i)
        return Zones(*[getattr(self, f)[m] for f in self._F])

    @staticmethod
    def empty() -> "Zones":
        return Zones(np.empty(0), np.empty(0), np.empty(0, np.int64),
                     np.empty(0, np.int64), np.empty(0, "<U12"),
                     np.empty(0, "<U3"), np.empty(0, "<U7"))


# ── death ───────────────────────────────────────────────────────────────────

def death_by_break(close: np.ndarray, price: np.ndarray, born: np.ndarray,
                   above: np.ndarray, buffer: np.ndarray) -> np.ndarray:
    """First bar after `born` at which price closes decisively through a level.

    `above=True` means the level is only meaningful while price stays below it (a
    swing high, an EQH, a PDH) so a close above it by more than `buffer` kills it.
    A close through a level is the market saying that price is no longer resting
    there — which is exactly the fact the old tree lost when it re-derived every
    level from scratch each scan and let broken structure keep voting.

    O(n log n) via per-level search over the running extreme of closes; a naive
    per-level scan of the tail is O(n*m) and dominated a 7M-bar series.
    """
    close = np.asarray(close, np.float64)
    n = len(close)
    out = np.full(len(price), n, np.int64)
    if len(price) == 0:
        return out

    # running max / min of closes from each index forward, computed once
    run_max = np.maximum.accumulate(close[::-1])[::-1]
    run_min = np.minimum.accumulate(close[::-1])[::-1]

    for j in range(len(price)):
        b = int(born[j]) + 1
        if b >= n:
            continue
        thr = float(price[j]) + (float(buffer[j]) if above[j] else -float(buffer[j]))
        # cheap reject: if the level is never breached after birth, it never dies
        if above[j]:
            if run_max[b] <= thr:
                continue
        else:
            if run_min[b] >= thr:
                continue
        # Doubling window search. `flatnonzero(close[b:] > thr)` scans the whole
        # tail every time, which is O(n) per level and quadratic over a series —
        # fine at 40k bars, fatal at 1M. Most levels break within a few hundred
        # bars, so search a small window first and grow only when needed.
        w = 512
        while True:
            e = min(b + w, n)
            seg = close[b:e]
            hit = np.argmax(seg > thr) if above[j] else np.argmax(seg < thr)
            found = (seg[hit] > thr) if above[j] else (seg[hit] < thr)
            if found:
                out[j] = b + int(hit)
                break
            if e >= n:
                break
            w *= 4
    return out


# ── swings ──────────────────────────────────────────────────────────────────

def find_swings(high, low, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Fractal swing indices: bar i is a swing high if its high is the strict max
    of [i-k, i+k]. Returns (high_idx, low_idx).

    A swing at i is not KNOWABLE until bar i+k has closed. Callers must set
    born = i + k. Ties are excluded (strict max/min) so a flat double top does not
    register twice — that is what the EQH detector is for.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = len(high)
    if n < 2 * k + 1:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    win = 2 * k + 1
    sh = np.lib.stride_tricks.sliding_window_view(high, win)
    sl = np.lib.stride_tricks.sliding_window_view(low, win)
    centre = np.arange(k, n - k)

    is_h = np.argmax(sh, axis=1) == k
    is_l = np.argmin(sl, axis=1) == k
    # strictness: the centre must beat every neighbour, not merely tie for max
    uniq_h = (sh == sh[:, [k]]).sum(axis=1) == 1
    uniq_l = (sl == sl[:, [k]]).sum(axis=1) == 1

    return centre[is_h & uniq_h].astype(np.int64), centre[is_l & uniq_l].astype(np.int64)


def swing_points(bars: Bars, tf: str, k: int = 3,
                 buffer: np.ndarray | None = None) -> Points:
    """Swing highs and lows as level POINTS, priced at the swing extreme itself —
    never a mean. A swing dies when a close breaks through it by `buffer` (default
    0.10 ATR); after that it is broken structure, not a level."""
    from .context import atr as _atr
    hi_i, lo_i = find_swings(bars.high, bars.low, k)
    if len(hi_i) + len(lo_i) == 0:
        return Points.empty()
    if buffer is None:
        buffer = _atr(bars, 14) * 0.10

    price = np.concatenate([np.asarray(bars.high)[hi_i], np.asarray(bars.low)[lo_i]])
    ref = np.concatenate([hi_i, lo_i]).astype(np.int64)
    src = np.concatenate([np.full(len(hi_i), "swing_h"), np.full(len(lo_i), "swing_l")])
    above = np.concatenate([np.ones(len(hi_i), bool), np.zeros(len(lo_i), bool)])
    born = ref + k
    dead = death_by_break(bars.close, price, born, above,
                          np.asarray(buffer)[np.clip(born, 0, len(bars) - 1)])
    return Points(price.astype(np.float64), born, dead, src.astype("<U12"),
                  np.full(len(ref), tf, "<U3"), ref).sort_by_born()


# ── equal highs / lows ──────────────────────────────────────────────────────

def equal_levels(bars: Bars, tf: str, tol: np.ndarray, k: int = 3,
                 min_sep: int = 3, max_sep: int = 60) -> Points:
    """EQH / EQL — done properly.

    The old implementation appended EVERY pair of bars whose highs sat within a
    flat 5-pip tolerance, then took the first five. Measured over 40 windows of
    M15: 28 pairs generated per scan, the five actually used always came from the
    oldest fifth of the window, and 20% of them were ADJACENT bars — one swing
    counted as an equal high. `eql` is the strongest source in the whole reaction
    study (+1.261 context-free, +2.433 in range regimes) and it was being fed
    arbitrary stale noise.

    What an equal high actually is: two or more distinct SWING highs, separated by
    enough bars to be separate events, agreeing on a price within an ATR-scaled
    tolerance. Not two adjacent bars. The emitted price is the EXTREME of the
    agreeing swings — the price stops actually sit beyond — not their mean.

    `tol` is a per-bar price tolerance array (see Context.threshold).
    """
    hi_i, lo_i = find_swings(bars.high, bars.low, k)
    out = Points.empty()

    for idx, col, name, take in ((hi_i, np.asarray(bars.high), "eqh", np.max),
                                 (lo_i, np.asarray(bars.low), "eql", np.min)):
        if len(idx) < 2:
            continue
        prices = col[idx]
        pr, bo, rf = [], [], []
        # Pair each swing with LATER swings only; the level is born when the
        # second one is confirmed, because that is when the agreement exists.
        for a in range(len(idx) - 1):
            for b in range(a + 1, len(idx)):
                sep = idx[b] - idx[a]
                if sep < min_sep:
                    continue
                if sep > max_sep:
                    break
                t = float(tol[min(idx[b], len(tol) - 1)])
                if abs(prices[a] - prices[b]) <= t:
                    pr.append(float(take([prices[a], prices[b]])))
                    bo.append(int(idx[b] + k))
                    rf.append(int(idx[b]))
                    break        # one EQ event per swing; the nearest match wins
        if pr:
            price = np.asarray(pr, np.float64)
            born = np.asarray(bo, np.int64)
            above = np.full(len(pr), name == "eqh")
            buf = np.asarray(tol)[np.clip(born, 0, len(tol) - 1)]
            out = out + Points(price, born,
                               death_by_break(bars.close, price, born, above, buf),
                               np.full(len(pr), name, "<U12"),
                               np.full(len(pr), tf, "<U3"), np.asarray(rf, np.int64))
    return out.sort_by_born()


# ── round numbers ───────────────────────────────────────────────────────────

def round_points(bars: Bars, tf: str, step: float | None = None) -> Points:
    """Big figures and half figures.

    ONE point per round price for the whole series — born the first bar price
    comes within range, immortal thereafter. A round number is not invalidated by
    being traded through; $4,700 is still $4,700 next week.

    That "emit once" is deliberate and was the first thing to go wrong here. The
    initial version emitted a fresh point on every re-entry into range, which on
    gold produced 5,007 half-round points out of 9,970 total — half the level
    universe was one price restated hundreds of times, and because the active set
    only grows, every one of them was still voting at the end of the series.

    Gold steps $50 (big) / $25 (half): at $4,700 a $10 grid puts a "round number"
    inside almost every hour's range, which makes the source meaningless. The
    reaction study measured `big_figure` at +0.058 context-free and +1.376 in
    trend_up, and that is a genuine $50-class level, not every ten dollars.
    """
    pip = bars.pip
    if step is None:
        step = 25.0 if pip >= 0.1 else 0.0050          # $25 gold, 50 pips FX
    close = np.asarray(bars.close, dtype=np.float64)
    n = len(close)
    grid = np.arange(np.floor(close.min() / step) * step,
                     np.ceil(close.max() / step) * step + step, step)
    if grid.size == 0:
        return Points.empty()

    pr, bo, src = [], [], []
    for g in grid:
        # first bar whose range touches this price — that is when it is in play
        w = np.flatnonzero((np.asarray(bars.low) <= g) & (np.asarray(bars.high) >= g))
        born = int(w[0]) if w.size else 0
        is_big = abs(g / (step * 2) - round(g / (step * 2))) < 1e-9
        pr.append(float(g))
        bo.append(born)
        src.append("round" if is_big else "half_round")
    return Points(np.asarray(pr, np.float64), np.asarray(bo, np.int64),
                  np.full(len(pr), n, np.int64),          # immortal
                  np.asarray(src, "<U12"), np.full(len(pr), tf, "<U3"),
                  np.asarray(bo, np.int64)).sort_by_born()


# ── prior period extremes ───────────────────────────────────────────────────

def period_points(bars: Bars, tf: str, period: str = "D",
                  live_periods: int = 2) -> Points:
    """Previous day / week high and low (PDH/PDL, PWH/PWL).

    Born at the FIRST bar of the following period — the moment the prior period is
    complete and its extremes are fixed. Not at the extreme's own bar, which would
    let a decision use a high the session had not yet finished making.

    Dies at whichever comes first: a decisive close through it, or `live_periods`
    later. "Yesterday's high" stops meaning anything once it is last Tuesday's.
    """
    t = np.asarray(bars.time, dtype=np.int64)
    span = 86400 if period == "D" else 7 * 86400
    bucket = t // span
    edges = np.flatnonzero(np.concatenate(([True], np.diff(bucket) != 0)))
    if len(edges) < 2:
        return Points.empty()

    high = np.asarray(bars.high, np.float64)
    low = np.asarray(bars.low, np.float64)
    n = len(bars)
    pr, bo, rf, src, expire, above = [], [], [], [], [], []
    for j in range(1, len(edges)):
        s, e = edges[j - 1], edges[j]
        born = int(e)                       # first bar of the NEXT period
        stop = int(edges[min(j + live_periods, len(edges) - 1)]) \
            if j + live_periods < len(edges) else n
        hi_i = int(s + np.argmax(high[s:e]))
        lo_i = int(s + np.argmin(low[s:e]))
        pr += [float(high[hi_i]), float(low[lo_i])]
        bo += [born, born]
        rf += [hi_i, lo_i]
        expire += [stop, stop]
        above += [True, False]
        src += (["pdh", "pdl"] if period == "D" else ["pwh", "pwl"])

    price = np.asarray(pr, np.float64)
    born_a = np.asarray(bo, np.int64)
    buf = np.full(len(pr), bars.pip * 5.0)
    broken = death_by_break(bars.close, price, born_a, np.asarray(above), buf)
    dead = np.minimum(broken, np.asarray(expire, np.int64))
    return Points(price, born_a, dead, np.asarray(src, "<U12"),
                  np.full(len(pr), tf, "<U3"), np.asarray(rf, np.int64))


# ── zones: order blocks and fair value gaps ─────────────────────────────────

def fvg_zones(bars: Bars, tf: str, min_gap_atr: float = 0.05,
              min_body_atr: float = 1.0) -> Zones:
    """Three-bar fair value gaps, with an explicit death index.

    A gap is born when bar i+1 closes (the pattern needs all three bars) and dies
    at the bar that fully fills it. The old tree tracked `status` but never a
    death BAR, so a gap filled 400 bars ago still entered clustering as live
    structure.
    """
    from .context import atr as _atr
    h = np.asarray(bars.high, np.float64)
    l = np.asarray(bars.low, np.float64)
    o = np.asarray(bars.open, np.float64)
    c = np.asarray(bars.close, np.float64)
    n = len(h)
    a = _atr(bars, 14)
    if n < 4:
        return Zones.empty()

    i = np.arange(1, n - 1)
    body = np.abs(c[i] - o[i])
    min_gap = np.maximum(3 * bars.pip, a[i] * min_gap_atr)

    bull = (l[i + 1] > h[i - 1] + min_gap) & (body >= a[i] * min_body_atr)
    bear = (h[i + 1] < l[i - 1] - min_gap) & (body >= a[i] * min_body_atr)

    rows = []
    for mask, side in ((bull, "bullish"), (bear, "bearish")):
        idx = i[mask]
        if not len(idx):
            continue
        if side == "bullish":
            top, bot = l[idx + 1], h[idx - 1]
        else:
            top, bot = l[idx - 1], h[idx + 1]
        born = idx + 1
        # death = first bar after birth that closes the gap completely
        dead = np.full(len(idx), n, np.int64)
        for j, (b, lo_p, hi_p) in enumerate(zip(born, bot, top)):
            after = (l[b + 1:] <= lo_p) if side == "bullish" else (h[b + 1:] >= hi_p)
            w = np.flatnonzero(after)
            if w.size:
                dead[j] = int(b + 1 + w[0])
        rows.append(Zones(top.astype(np.float64), bot.astype(np.float64),
                          born.astype(np.int64), dead,
                          np.full(len(idx), "fvg", "<U12"),
                          np.full(len(idx), tf, "<U3"),
                          np.full(len(idx), side, "<U7")))
    out = Zones.empty()
    for r in rows:
        out = out + r
    return out


def ob_zones(bars: Bars, tf: str, impulse_atr: float = 1.5,
             body_ratio: float = 0.5) -> Zones:
    """Order blocks: the last opposing candle before a two-bar displacement.

    Born at the bar that completes the displacement, dies when price closes
    through the block's far side (structure broken), not merely on a touch — the
    old detector conflated "tested" with "mitigated" and dropped blocks that were
    still working.
    """
    from .context import atr as _atr
    o = np.asarray(bars.open, np.float64)
    c = np.asarray(bars.close, np.float64)
    h = np.asarray(bars.high, np.float64)
    l = np.asarray(bars.low, np.float64)
    n = len(o)
    if n < 5:
        return Zones.empty()
    a = _atr(bars, 14)

    body = c - o
    rng = np.maximum(h - l, 1e-12)
    strong_up = (body > 0) & (np.abs(body) / rng > body_ratio)
    strong_dn = (body < 0) & (np.abs(body) / rng > body_ratio)

    i = np.arange(0, n - 3)
    move_up = body[i + 1] + body[i + 2]
    up = strong_up[i + 1] & strong_up[i + 2] & (move_up >= a[i] * impulse_atr) & (body[i] < 0)
    dn = strong_dn[i + 1] & strong_dn[i + 2] & (-move_up >= a[i] * impulse_atr) & (body[i] > 0)

    rows = []
    for mask, side in ((up, "bullish"), (dn, "bearish")):
        idx = i[mask]
        if not len(idx):
            continue
        born = idx + 2
        dead = np.full(len(idx), n, np.int64)
        for j, b in enumerate(born):
            k = idx[j]
            after = (c[b + 1:] < l[k]) if side == "bullish" else (c[b + 1:] > h[k])
            w = np.flatnonzero(after)
            if w.size:
                dead[j] = int(b + 1 + w[0])
        rows.append(Zones(h[idx].astype(np.float64), l[idx].astype(np.float64),
                          born.astype(np.int64), dead,
                          np.full(len(idx), "ob", "<U12"),
                          np.full(len(idx), tf, "<U3"),
                          np.full(len(idx), side, "<U7")))
    out = Zones.empty()
    for r in rows:
        out = out + r
    return out
