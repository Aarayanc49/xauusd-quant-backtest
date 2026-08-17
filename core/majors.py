"""Major levels — the ones worth breaking.

The measured problem this fixes: `find_breaks` produced **110,561 structure
breaks over 14.6 years** — roughly 30 per day on M5. A break of a minor swing
that formed 40 minutes ago is not "resistance breaking"; it is noise with a name.
Trading all of them gave a +0.147R gross edge that the spread ate.

A MAJOR level is not a swing. It is a price that has been **repeatedly tested and
repeatedly defended**, and the market's willingness to defend it is the only
evidence that anyone's orders are actually there. That makes majorness a measured
quantity, not a label:

    touches      how many distinct times price came to it and left
    respects     how many of those it survived
    respect_rate respects / touches
    age          how long it has been a level
    tf_rank      the highest timeframe that produced a member
    categories   how many INDEPENDENT source families agree on it

The old tree had two thirds of this and used none of it properly:

  * `key_levels._compute_reactions` already counted touches and computed a
    respect rate over 100 H1 bars — then multiplied an INVENTED strength by
    `(0.5 + respect_rate)` and discarded the raw counts.
  * `top_down_sr` required ">=3 sources from >=2 categories", which is the right
    independence test, and was never wired to anything.
  * `zone_registry` tracked touch counts and lifecycle across restarts, for the
    eight strategies that never ran.

Nothing here invents a weight. Every field is a count or a ratio, and
`research/reaction.py` decides which of them actually predicts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cluster import TF_RANK

# A touch has to LEAVE before it can count again, otherwise one long consolidation
# at a level registers as fifty touches. Price must clear the band by this many
# tolerances before the level is armed for another touch.
_REARM = 1.5


@dataclass
class Majors:
    """Per-track majorness evidence. Index-aligned with the tracks passed in."""
    price: np.ndarray
    born: np.ndarray
    dead: np.ndarray
    touches: np.ndarray        # int32 — distinct approach-and-leave events
    respects: np.ndarray       # int32 — touches that did not break it
    first_touch: np.ndarray    # int64 — bar of the first touch (-1 if never)
    age: np.ndarray            # int64 — bars from born to dead (or series end)
    tf_rank: np.ndarray        # int8  — highest member timeframe
    categories: np.ndarray     # int8  — distinct independent source families
    n_members: np.ndarray      # int16

    _F = ("price", "born", "dead", "touches", "respects", "first_touch",
          "age", "tf_rank", "categories", "n_members")

    def __len__(self) -> int:
        return len(self.price)

    @property
    def respect_rate(self) -> np.ndarray:
        return np.divide(self.respects, self.touches,
                         out=np.full(len(self), 0.5), where=self.touches > 0)

    def score(self) -> np.ndarray:
        """A single ordering for convenience ONLY — never a gate.

        Deliberately crude and equally weighted. The old tree's fatal habit was
        inventing a weighted score and then tuning the weights; every such score
        it built measured |rho| < 0.08 against outcome. Use the raw fields in
        filters and let the study decide.
        """
        return (np.log1p(self.touches) * self.respect_rate
                + self.categories * 0.5 + self.tf_rank * 0.25)

    def select(self, m) -> "Majors":
        return Majors(*[getattr(self, f)[m] for f in self._F])


# Independent source families. Two swings agreeing is one observation restated;
# a swing agreeing with a round number and a prior-day high is three different
# reasons for orders to rest at that price.
_FAMILY = {
    "swing_h": "swing", "swing_l": "swing",
    "eqh": "equal", "eql": "equal",
    "round": "round", "half_round": "round",
    "pdh": "period", "pdl": "period", "pwh": "period", "pwl": "period",
}


def _families(sources) -> int:
    return len({_FAMILY.get(str(s), "other") for s in sources})


def measure(bars, tracks, atr: np.ndarray, tol_atr: float = 0.20,
            horizon: int = 24, points=None) -> Majors:
    """Count touches and respects for every track over its own lifetime.

    A touch is an approach into the band followed by a departure. It counts as a
    RESPECT when price leaves on the side it arrived from within `horizon` bars,
    and as a break otherwise — which is also where the track dies, so respects are
    simply the touches that happened before death.

    Vectorised per track over its own window; total work is the sum of track
    lifetimes, not tracks x bars.
    """
    n = len(bars)
    high = np.asarray(bars.high, np.float64)
    low = np.asarray(bars.low, np.float64)
    close = np.asarray(bars.close, np.float64)

    m = len(tracks.price)
    touches = np.zeros(m, np.int32)
    respects = np.zeros(m, np.int32)
    first_touch = np.full(m, -1, np.int64)

    for j in range(m):
        p = float(tracks.price[j])
        lo = int(tracks.born[j])
        hi = int(min(tracks.dead[j], n))
        if hi - lo < 2:
            continue
        tol = float(atr[min(lo, n - 1)]) * tol_atr
        if tol <= 0:
            continue

        inband = (low[lo:hi] <= p + tol) & (high[lo:hi] >= p - tol)
        if not inband.any():
            continue
        # rising edges = distinct approaches; require a clear departure between
        outside = (low[lo:hi] > p + tol * _REARM) | (high[lo:hi] < p - tol * _REARM)
        armed = True
        cnt = 0
        first = -1
        for t in range(hi - lo):
            if armed and inband[t]:
                cnt += 1
                if first < 0:
                    first = lo + t
                armed = False
            elif not armed and outside[t]:
                armed = True
        touches[j] = cnt
        first_touch[j] = first
        # every touch before the track died is a touch the level survived
        respects[j] = max(0, cnt - 1) if tracks.dead[j] < n else cnt

    # composition from the underlying points, when available
    tf_rank = np.zeros(m, np.int8)
    cats = np.ones(m, np.int8)
    n_mem = np.ones(m, np.int16)
    if points is not None and hasattr(tracks, "member_sources"):
        for j, (srcs, tfs) in enumerate(zip(tracks.member_sources, tracks.member_tfs)):
            cats[j] = _families(srcs)
            n_mem[j] = len(srcs)
            tf_rank[j] = max((TF_RANK.get(str(t), 0) for t in tfs), default=0)
    elif hasattr(tracks, "n_members"):
        n_mem = np.asarray(tracks.n_members, np.int16)

    dead = np.minimum(tracks.dead, n)
    return Majors(price=np.asarray(tracks.price, np.float64),
                  born=np.asarray(tracks.born, np.int64),
                  dead=np.asarray(dead, np.int64),
                  touches=touches, respects=respects, first_touch=first_touch,
                  age=(dead - tracks.born).astype(np.int64),
                  tf_rank=tf_rank, categories=cats, n_members=n_mem)


def is_major(mj: Majors, min_touches: int = 2, min_respect: float = 0.5,
             min_categories: int = 1, min_age: int = 0,
             min_tf_rank: int = 0) -> np.ndarray:
    """Boolean mask of tracks that qualify as major under the given bar.

    Every threshold is an argument because every one of them is a hypothesis.
    The defaults are deliberately permissive so the study can tighten them and
    watch the trade count and the edge move together — the old tree set six such
    thresholds at once by reasoning and cut its population to 7% for no gain.
    """
    return ((mj.touches >= min_touches)
            & (mj.respect_rate >= min_respect)
            & (mj.categories >= min_categories)
            & (mj.age >= min_age)
            & (mj.tf_rank >= min_tf_rank))


def major_breaks(breaks, mj: Majors, atr: np.ndarray, tol_atr: float = 0.30,
                 mask: np.ndarray | None = None):
    """Keep only the breaks that broke a MAJOR level, and tag which one.

    Returns (filtered_breaks, level_index) where level_index maps each surviving
    break to its row in `mj`, so the level's touch count and respect rate travel
    with the trade into the journal and can be sliced on later.
    """
    if mask is None:
        mask = np.ones(len(mj), bool)
    idx = np.flatnonzero(mask)
    if not len(idx) or len(breaks) == 0:
        return breaks.select(np.zeros(len(breaks), bool)), np.empty(0, np.int64)

    prices = mj.price[idx]
    order = np.argsort(prices)
    sp, sidx = prices[order], idx[order]

    keep = np.zeros(len(breaks), bool)
    lvl = np.full(len(breaks), -1, np.int64)
    for j in range(len(breaks)):
        b = int(breaks.bar[j])
        tol = float(atr[min(b, len(atr) - 1)]) * tol_atr
        p = float(breaks.level[j])
        a = int(np.searchsorted(sp, p - tol, "left"))
        z = int(np.searchsorted(sp, p + tol, "right"))
        if z <= a:
            continue
        # among nearby majors, require the level to be live at the break
        cand = sidx[a:z]
        live = cand[(mj.born[cand] <= b) & (mj.dead[cand] >= b)]
        if not len(live):
            continue
        best = live[np.argmin(np.abs(mj.price[live] - p))]
        keep[j] = True
        lvl[j] = best
    return breaks.select(keep), lvl[keep]
