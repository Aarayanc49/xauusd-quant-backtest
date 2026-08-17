"""Clustering with a hard width cap — the fix for the fault that broke everything.

The old `cluster_levels` walked a chain:

    for pt in sorted_points[1:]:
        if abs(pt.price - current[-1].price) <= tolerance:   # the PREVIOUS point
            current.append(pt)

Each hop was legal and the total span was unbounded. With ~52 points spread over
~537p and a 20p tolerance, 84% of adjacent gaps sat inside tolerance, so the chain
walked 4000 -> 4018 -> 4035 -> 4052 -> 4070 without ever terminating. Measured
output: median cluster 49p wide, p75 114p, p90 239p, max 931p — against a 60p
stop. On a p75 cluster price could travel the entire stop distance without leaving
the "level".

Then `_build_cluster` set `price = mean(prices)` and `zone = union(member zones)`,
so the number every trigger referenced was the arithmetic mean of four-plus
unrelated points from different sources and timeframes. It frequently corresponded
to no structure whatsoever.

Two rules fix it:

  1. **A hard width cap.** A merge that would push the cluster wider than the cap
     is refused, full stop. Points that cannot merge stay singletons — which is
     correct and useful, not a failure. A cluster wider than the cap is never
     emitted; if the code cannot produce a precise level there, it says so rather
     than averaging five things together.

  2. **The reference price is a real constituent's price.** Never a mean. With
     measured weights it is the strongest member's price; with uniform weights it
     is the highest-timeframe, longest-established member. Either way it is a
     price something is actually resting at.

Success criterion for this module, checkable without simulating a single trade:
p90 cluster width must fall from 239p to under the cap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .levels import SOURCE_CLASS, DECISION, MAGNET, Points

# Timeframe seniority for picking the reference price when weights are uniform.
TF_RANK = {"M1": 0, "M5": 1, "M15": 2, "M30": 3, "H1": 4, "H4": 5, "D1": 6, "W1": 7}


@dataclass
class Cluster:
    """A precise level with an audit trail back to what made it."""
    price: float             # a REAL constituent price, never a mean
    lo: float                # tightest bound of the agreeing members
    hi: float
    members: np.ndarray      # indices into the Points that produced it
    sources: tuple           # source names, in member order
    weight: float            # summed member weight
    n_decision: int
    n_magnet: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def magnet_share(self) -> float:
        t = self.n_decision + self.n_magnet
        return (self.n_magnet / t) if t else 0.0

    def __repr__(self) -> str:
        return (f"<Cluster {self.price:.2f} w={self.width:.2f} "
                f"n={len(self.members)} {'/'.join(sorted(set(self.sources)))}>")


def cluster_points(points: Points, cap: float, weights: dict | None = None,
                   dedup: float = 0.0) -> list[Cluster]:
    """Agglomerate `points` into clusters no wider than `cap` (price units).

    `weights` maps source name -> float; missing sources weigh 1.0. Supply the
    output of research/reaction.py here once it has run — until then uniform
    weights are honest and a guessed weight is not.

    `dedup` collapses members closer together than this before clustering, so a
    source that emits the same price repeatedly (a round number coming back into
    range, a swing re-confirmed) does not inflate confluence counts. This is the
    other half of the old tree's confluence problem: order blocks, FVGs, fibs and
    swings are all computed from the SAME OHLC series, so "5 confluences" was
    often one swing high expressed five ways — one fact counted five times, and
    exactly the kind of cluster that scored highest.
    """
    n = len(points)
    if n == 0:
        return []
    w = weights or {}
    order = np.argsort(points.price, kind="stable")
    price = np.asarray(points.price, dtype=np.float64)[order]

    out: list[Cluster] = []
    start = 0
    for i in range(1, n + 1):
        # close the run when adding point i would exceed the cap, or at the end
        if i == n or (price[i] - price[start]) > cap:
            out.append(_build(points, order[start:i], w, dedup))
            start = i
    return out


def _build(points: Points, member_idx: np.ndarray, weights: dict,
           dedup: float) -> Cluster:
    """Assemble one cluster. `member_idx` indexes into `points`."""
    pr = np.asarray(points.price, np.float64)[member_idx]
    src = np.asarray(points.source)[member_idx]
    tfs = np.asarray(points.tf)[member_idx]
    born = np.asarray(points.born, np.int64)[member_idx]

    # Collapse near-duplicate prices from the same source before counting, so
    # confluence measures agreement between DIFFERENT observations rather than one
    # observation restated.
    if dedup > 0 and len(pr) > 1:
        keep = np.ones(len(pr), bool)
        seen: dict[str, list[float]] = {}
        for j in np.argsort(pr):
            s = str(src[j])
            if any(abs(pr[j] - q) <= dedup for q in seen.get(s, ())):
                keep[j] = False
            else:
                seen.setdefault(s, []).append(float(pr[j]))
        if keep.any():
            member_idx, pr, src, tfs, born = (member_idx[keep], pr[keep],
                                              src[keep], tfs[keep], born[keep])

    wts = np.array([float(weights.get(str(s), 1.0)) for s in src])

    # Reference price: the strongest member's own price. Ties broken by timeframe
    # seniority, then by age (an older level has survived more of the tape).
    rank = np.lexsort((born, [TF_RANK.get(str(t), 0) for t in tfs], wts))
    best = rank[-1]

    classes = [SOURCE_CLASS.get(str(s), DECISION) for s in src]
    return Cluster(
        price=float(pr[best]),
        lo=float(pr.min()), hi=float(pr.max()),
        members=member_idx,
        sources=tuple(str(s) for s in src),
        weight=float(wts.sum()),
        n_decision=sum(1 for c in classes if c == DECISION),
        n_magnet=sum(1 for c in classes if c == MAGNET),
    )


def nearest(clusters: list[Cluster], price: float, within: float | None = None):
    """Closest cluster to `price`, or None. `within` is a price-unit limit."""
    if not clusters:
        return None
    d = [abs(c.price - price) for c in clusters]
    j = int(np.argmin(d))
    if within is not None and d[j] > within:
        return None
    return clusters[j]


def anatomy(clusters: list[Cluster], pip: float) -> dict:
    """Width and composition percentiles — the number that proves this module
    works. The old chain produced p50 49p, p75 114p, p90 239p, max 931p."""
    if not clusters:
        return {}
    w = np.array([c.width for c in clusters]) / pip
    m = np.array([len(c.members) for c in clusters])
    return {
        "n": len(clusters),
        "width_p50": float(np.percentile(w, 50)),
        "width_p75": float(np.percentile(w, 75)),
        "width_p90": float(np.percentile(w, 90)),
        "width_p95": float(np.percentile(w, 95)),
        "width_max": float(w.max()),
        "over_35p": float((w >= 35).mean()),
        "over_100p": float((w >= 100).mean()),
        "over_200p": float((w >= 200).mean()),
        "members_p50": float(np.percentile(m, 50)),
        "members_max": int(m.max()),
        "singletons": float((m == 1).mean()),
    }
