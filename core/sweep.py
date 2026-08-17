"""The trigger — a stop run that failed, resolved against ONE price.

The strategy idea, stripped to its core and unchanged from the old tree because
the idea was never the problem:

    Price reacts at levels where resting orders sit. Find such a level, wait for
    price to run the stops just beyond it and fail, then enter in the direction of
    the failure with a stop beyond the sweep.

That requires a PRECISE level, which `core/cluster.py` now delivers (p90 width 21p
against a 160p stop, down from 239p against 60p). This module is where the old
tree's fifth and most damaging fault lived:

    level_read.apply_close  resolved SWEPT / HELD / BROKEN against zone_high /
                            zone_low — the band EDGES
    playbook.trigger_sweep_v fired against cluster['price'] — the band CENTRE

Two different prices on the same object, 24 pips apart at the median and 119 at
the p90, firing on different candles. So "the level was swept, therefore fade the
sweep" was never the sequence the code executed. The docstring even stated the
choice — *"cluster center, never the zone edge"* — so it was deliberate, and it is
defensible for a 10p zone and incoherent for a 49p one.

Here there is exactly one reference: `Cluster.price`, a real constituent's price.
State and trigger both resolve against it, and the sweep depth that fired is
recorded on the event so nothing downstream has to recompute it.

Two further fixes carried in:

  * **A sweep has a minimum depth again.** The old code's comment read "Sweeps
    have NO minimum-pips threshold: a sweep is valid on candle anatomy, not on how
    deep the wick went." That let a one-tick pierce — pure spread noise — count as
    a stop run. Depth is now `0.10 x ATR(M5)`, scaled like everything else.
  * **HELD is a finding, not an else-branch.** In the old state machine any candle
    that touched the band and closed inside it was HELD, so on a 238p band
    essentially every interacting candle was "defended". HELD here requires price
    to have actually traded into the level and closed away from it on the origin
    side.

Everything is vectorized over a bar window. The engine never loops bars looking up
levels; it loops LEVELS and asks each one where it fired, which is what makes a
year of M5 resolve in under a second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Event kinds. `direction` is what we would TRADE, i.e. the direction of the
# failure, not the direction of the pierce.
SWEEP_V = "sweep_v"              # one bar: pierce and reject, same candle
SWEEP_RECLAIM = "sweep_reclaim"  # two bars: close beyond, then close back
BUY, SELL = "buy", "sell"

# Level lifecycle. Reported for study and for the entry filter; not a score.
UNTESTED, HELD, SWEPT, BROKEN = "untested", "held", "swept", "broken"


@dataclass
class Events:
    """Flat arrays of trigger events, one row per (bar, level) firing."""
    bar: np.ndarray          # int64 — bar index whose CLOSE fires the trade
    level: np.ndarray        # int64 — index into the caller's level list
    price: np.ndarray        # float64 — the level price, the single reference
    direction: np.ndarray    # <U4
    kind: np.ndarray         # <U14
    depth: np.ndarray        # float64 — how far past the level the wick ran, price units
    extreme: np.ndarray      # float64 — the wick extreme; the stop goes beyond THIS
    entry: np.ndarray        # float64 — the close that fires it

    _F = ("bar", "level", "price", "direction", "kind", "depth", "extreme", "entry")

    def __len__(self) -> int:
        return len(self.bar)

    def __add__(self, other: "Events") -> "Events":
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other
        return Events(*[np.concatenate([getattr(self, f), getattr(other, f)])
                        for f in self._F])

    def sort_by_bar(self) -> "Events":
        o = np.argsort(self.bar, kind="stable")
        return Events(*[getattr(self, f)[o] for f in self._F])

    @staticmethod
    def empty() -> "Events":
        return Events(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0),
                      np.empty(0, "<U4"), np.empty(0, "<U14"), np.empty(0),
                      np.empty(0), np.empty(0))


def _cols(o, h, l, c, lo, hi):
    s = slice(lo, hi)
    return (np.asarray(o[s], np.float64), np.asarray(h[s], np.float64),
            np.asarray(l[s], np.float64), np.asarray(c[s], np.float64))


def sweep_v(o, h, l, c, price: float, sweep_min, lo: int, hi: int):
    """One-bar V-sweep against `price`, over bars [lo, hi).

    Upside pierce -> SELL:
        1. the wick ran past the level by at least `sweep_min`
        2. the body opened AND closed back on the trade side of the level
        3. the rejection wick dominates: >= the body and >= the opposite wick

    Returns (offsets, direction, depth, extreme) where offsets are relative to lo.
    Condition 2 is why this anatomy can never match a sweep-and-reclaim — the body
    must OPEN on the trade side — which is exactly why `sweep_reclaim` exists.
    """
    O, H, L, C = _cols(o, h, l, c, lo, hi)
    if len(O) == 0:
        return (np.empty(0, np.int64),) * 1 + (np.empty(0, "<U4"),) + (np.empty(0),) * 2
    sm = np.asarray(sweep_min[lo:hi], np.float64)

    body = np.abs(C - O)
    up_wick = H - np.maximum(O, C)
    dn_wick = np.minimum(O, C) - L

    sell = ((H > price + sm) & (O <= price) & (C < price)
            & (up_wick >= body) & (up_wick >= dn_wick))
    buy = ((L < price - sm) & (O >= price) & (C > price)
           & (dn_wick >= body) & (dn_wick >= up_wick))

    idx = np.flatnonzero(sell | buy)
    if idx.size == 0:
        return (np.empty(0, np.int64), np.empty(0, "<U4"),
                np.empty(0), np.empty(0))
    is_sell = sell[idx]
    direction = np.where(is_sell, SELL, BUY).astype("<U4")
    extreme = np.where(is_sell, H[idx], L[idx])
    depth = np.where(is_sell, H[idx] - price, price - L[idx])
    return idx.astype(np.int64), direction, depth, extreme


def sweep_reclaim(o, h, l, c, price: float, sweep_min, lo: int, hi: int):
    """Two-bar sweep-and-reclaim: the class the one-bar anatomy structurally
    cannot match.

    For an upside sweep -> SELL:
        1. context bar (i-2) CLOSED below the level — the level was respected
           before the sweep leg, which is what separates a stop run from a
           sustained breakout
        2. sweep leg (i-1) body-CLOSED above the level
        3. reclaim bar (i) closes back below with a down body

    No wick-dominance test on the reclaim bar: the reclaim body IS the rejection.
    The event fires on bar i's close, and `extreme` spans both legs, because the
    stop has to sit beyond the whole excursion and not just the final bar's wick.
    """
    O, H, L, C = _cols(o, h, l, c, lo, hi)
    n = len(O)
    if n < 3:
        return (np.empty(0, np.int64), np.empty(0, "<U4"),
                np.empty(0), np.empty(0))
    sm = np.asarray(sweep_min[lo:hi], np.float64)

    i = np.arange(2, n)
    ctx, leg, rec = C[i - 2], C[i - 1], C[i]

    sell = (ctx <= price) & (leg > price + sm[i - 1]) & (rec < price) & (rec < O[i])
    buy = (ctx >= price) & (leg < price - sm[i - 1]) & (rec > price) & (rec > O[i])

    hit = np.flatnonzero(sell | buy)
    if hit.size == 0:
        return (np.empty(0, np.int64), np.empty(0, "<U4"),
                np.empty(0), np.empty(0))
    at = i[hit]
    is_sell = sell[hit]
    direction = np.where(is_sell, SELL, BUY).astype("<U4")
    span_hi = np.maximum(H[at - 1], H[at])
    span_lo = np.minimum(L[at - 1], L[at])
    extreme = np.where(is_sell, span_hi, span_lo)
    depth = np.where(is_sell, span_hi - price, price - span_lo)
    return at.astype(np.int64), direction, depth, extreme


def find_events(bars, levels, sweep_min: np.ndarray,
                born: np.ndarray, dead: np.ndarray,
                allow_reclaim: bool = True,
                cooldown: int = 0) -> Events:
    """Every trigger every level produced, over each level's own live window.

    This is the inversion that makes the engine fast: rather than stepping bars
    and re-deriving the level set at each one (the old harness spent 76ms per bar
    in `discover_levels`, 97% of its pipeline), each level is asked once where it
    fired across its whole life. Cost is O(sum of level lifetimes), not
    O(bars x levels).

    `cooldown` suppresses repeat fires from the same level within N bars, so a
    level chopping around does not emit ten events on consecutive candles. The old
    tree solved this with a one-take-per-level-per-UTC-day map, which is both
    coarser and clock-dependent.
    """
    o, h, l, c = bars.open, bars.high, bars.low, bars.close
    n = len(bars)
    out = Events.empty()

    for j, p in enumerate(np.asarray(levels, np.float64)):
        lo = int(born[j])
        hi = int(min(dead[j], n))
        if hi - lo < 3:
            continue

        offs, dirs, deps, exts, kinds = [], [], [], [], []
        a, d, dp, ex = sweep_v(o, h, l, c, float(p), sweep_min, lo, hi)
        if a.size:
            offs.append(a); dirs.append(d); deps.append(dp); exts.append(ex)
            kinds.append(np.full(a.size, SWEEP_V, "<U14"))
        if allow_reclaim:
            a, d, dp, ex = sweep_reclaim(o, h, l, c, float(p), sweep_min, lo, hi)
            if a.size:
                offs.append(a); dirs.append(d); deps.append(dp); exts.append(ex)
                kinds.append(np.full(a.size, SWEEP_RECLAIM, "<U14"))
        if not offs:
            continue

        off = np.concatenate(offs)
        order = np.argsort(off, kind="stable")
        off = off[order]
        dirv = np.concatenate(dirs)[order]
        depv = np.concatenate(deps)[order]
        extv = np.concatenate(exts)[order]
        kndv = np.concatenate(kinds)[order]

        # A bar matching both anatomies journals as the stricter kind (V first,
        # since its order/close conditions are a superset of the reclaim's).
        keep = np.concatenate(([True], np.diff(off) != 0))
        off, dirv, depv, extv, kndv = (off[keep], dirv[keep], depv[keep],
                                       extv[keep], kndv[keep])

        if cooldown > 0 and off.size > 1:
            sel = [0]
            for t in range(1, off.size):
                if off[t] - off[sel[-1]] >= cooldown:
                    sel.append(t)
            sel = np.asarray(sel)
            off, dirv, depv, extv, kndv = (off[sel], dirv[sel], depv[sel],
                                           extv[sel], kndv[sel])

        bar = off + lo
        out = out + Events(
            bar=bar.astype(np.int64),
            level=np.full(bar.size, j, np.int64),
            price=np.full(bar.size, float(p)),
            direction=dirv, kind=kndv, depth=depv, extreme=extv,
            entry=np.asarray(c, np.float64)[bar],
        )
    return out.sort_by_bar()


# ── lifecycle, for study and for filtering ──────────────────────────────────

def resolve_state(o, h, l, c, price: float, sweep_min, break_min,
                  lo: int, hi: int) -> np.ndarray:
    """Per-bar lifecycle label over [lo, hi), against the SAME single reference
    the trigger uses.

    HELD is a positive finding here: the bar must have traded into the level
    (its range contains the price) and closed away from it on the side it
    approached from, without piercing far enough to be a sweep or closing far
    enough to be a break. In the old state machine HELD was the else branch, so on
    a wide band it meant only "price was inside the blob".
    """
    O, H, L, C = _cols(o, h, l, c, lo, hi)
    n = len(O)
    out = np.full(n, UNTESTED, "<U9")
    if n == 0:
        return out
    sm = np.asarray(sweep_min[lo:hi], np.float64)
    bm = np.asarray(break_min[lo:hi], np.float64)

    touched = (L <= price) & (H >= price)
    broke = np.abs(C - price) > bm
    swept = (((H > price + sm) & (C < price))
             | ((L < price - sm) & (C > price)))

    out[touched & broke] = BROKEN
    out[touched & swept] = SWEPT
    out[touched & ~broke & ~swept] = HELD
    return out
