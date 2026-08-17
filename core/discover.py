"""Level discovery — the whole pipeline from bars to tradeable level tracks.

Replaces `methodology.discover_levels` + `cluster_levels` + `classify_cluster`,
which between them were ~250 lines producing a 49-pip median blob priced at the
mean of unrelated points.

The structural difference is not the algorithms, it is WHEN they run. The old
pipeline re-derived every level from scratch on every scan — `discover_levels`
alone cost 75.9ms per bar, 97% of the level pipeline, and even with a 30-minute
structural cache it stayed at 23.5ms. That is what capped the harness at 12
bars/sec and 150 minutes per backtest year, and that in turn is why four
consecutive tuning waves shipped on assumptions nobody could afford to check.

Here every source is computed ONCE over the whole series with honest `born` and
`dead` indices, and clustering runs on a cadence rather than per bar. Levels do not
change meaningfully between two M5 bars; pretending they might cost the project a
year of iteration speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import levels as L
from .cluster import cluster_points
from .context import Context
from .store import Bars

# Which sources are read from which timeframe. Deliberately narrow — every source
# here has to earn its place in research/reaction.py before it gates anything, and
# the old tree's nine-source, everything-in list is exactly how 52 points ended up
# spread over 537 pips with nothing to tell them apart.
PLAN = [
    ("H4", "swing", dict(k=3)),
    ("H1", "swing", dict(k=3)),
    ("M15", "swing", dict(k=4)),
    ("H1", "equal", dict(k=3)),
    ("M15", "equal", dict(k=4)),
    ("H1", "round", {}),
    ("M15", "period", dict(period="D")),
    ("H1", "period", dict(period="W")),
]


def build_points(series: dict[str, Bars], ctx: Context, base: str) -> L.Points:
    """Every level point, expressed on the BASE timeframe's index."""
    out = L.Points.empty()
    for tf, kind, kw in PLAN:
        if tf not in series:
            continue
        b = series[tf]
        if kind == "swing":
            p = L.swing_points(b, tf, **kw)
        elif kind == "equal":
            p = L.equal_levels(b, tf, ctx.scales[tf].atr * 0.15, **kw)
        elif kind == "round":
            p = L.round_points(b, tf, **kw)
        elif kind == "period":
            p = L.period_points(b, tf, **kw)
        else:
            continue
        if len(p):
            out = out + remap_points(p, tf, base, series)
    return out.sort_by_born()


def build_zones(series: dict[str, Bars], base: str) -> L.Zones:
    out = L.Zones.empty()
    for tf in ("H4", "H1", "M15"):
        if tf not in series:
            continue
        for z in (L.fvg_zones(series[tf], tf), L.ob_zones(series[tf], tf)):
            if len(z):
                out = out + remap_zones(z, tf, base, series)
    return out


def _mapper(tf: str, base: str, series: dict):
    src, dst = series[tf], series[base]
    st = np.asarray(src.time, np.int64)
    dst_close = np.asarray(dst.time, np.int64) + dst.bar_seconds

    def to_base(idx):
        past = idx >= len(src)
        t = st[np.clip(idx, 0, len(src) - 1)] + src.bar_seconds
        m = np.searchsorted(dst_close, t, side="left")
        return np.where(past, len(dst), m).astype(np.int64)

    return to_base, len(dst)


def remap_points(p: L.Points, tf: str, base: str, series: dict) -> L.Points:
    """Translate born/dead from `tf` bar indices to `base` bar indices.

    A level born on `tf` bar j becomes knowable on the first base bar whose close
    falls at or after that tf bar's close. Anything mapping past the end of the
    base series is DROPPED, not clamped — clamping resurrects levels at the final
    bar and quietly biases the tail of every study.
    """
    if tf == base:
        return p
    to_base, n = _mapper(tf, base, series)
    nb, nd = to_base(p.born), to_base(p.dead)
    keep = nb < n
    return L.Points(p.price[keep], nb[keep], nd[keep], p.source[keep],
                    p.tf[keep], nb[keep])


def remap_zones(z: L.Zones, tf: str, base: str, series: dict) -> L.Zones:
    if tf == base:
        return z
    to_base, n = _mapper(tf, base, series)
    nb, nd = to_base(z.born), to_base(z.dead)
    keep = nb < n
    return L.Zones(z.high[keep], z.low[keep], nb[keep], nd[keep],
                   z.source[keep], z.tf[keep], z.side[keep])


# ── tracks ──────────────────────────────────────────────────────────────────

@dataclass
class Tracks:
    """Clustered levels as persistent objects with a life span.

    A track is one price that stayed a level across consecutive re-clusterings.
    Because the reference price is a real constituent's price rather than a mean,
    it is STABLE — it only changes when the strongest member dies — which is what
    makes tracking by exact price sound. Under the old mean-of-chain pricing the
    reference jittered every scan as points drifted in and out of the chain, and
    no such tracking was possible.
    """
    price: np.ndarray
    born: np.ndarray
    dead: np.ndarray
    weight: np.ndarray        # summed member weight at first sighting
    n_members: np.ndarray
    n_decision: np.ndarray
    n_magnet: np.ndarray
    width: np.ndarray         # cluster extent in price units

    _F = ("price", "born", "dead", "weight", "n_members",
          "n_decision", "n_magnet", "width")

    def __len__(self) -> int:
        return len(self.price)

    @property
    def magnet_share(self) -> np.ndarray:
        t = self.n_decision + self.n_magnet
        return np.divide(self.n_magnet, t, out=np.zeros(len(self)), where=t > 0)


def level_tracks(points: L.Points, ctx: Context, every: int = 12,
                 reach_mult: float = 6.0, weights: dict | None = None) -> Tracks:
    """Cluster on a cadence and stitch the results into tracks.

    `every` is in base bars — 12 M5 bars is one hour. Re-clustering per bar would
    cost 39,530 clusterings per year to describe a level set that changes a few
    times a day.

    Only levels within `reach_mult` proximity thresholds of price are clustered:
    a level 400 pips away is not a candidate for anything and only dilutes the
    counts that later get measured.
    """
    b = ctx.bars
    n = len(b)
    cap = ctx.threshold("cluster_width_cap")
    tol = ctx.threshold("cluster_tolerance")
    prox = ctx.threshold("proximity")
    close = np.asarray(b.close, np.float64)

    live: dict[float, dict] = {}
    done: list[dict] = []

    for i in range(0, n, every):
        act = points.near(i, float(close[i]), float(prox[i]) * reach_mult)
        seen = set()
        if len(act) >= 1:
            for c in cluster_points(act, float(cap[i]), weights=weights,
                                    dedup=float(tol[i]) * 0.25):
                key = round(float(c.price), 6)
                seen.add(key)
                cur = live.get(key)
                if cur is None:
                    live[key] = {
                        "price": float(c.price), "born": i, "last": i,
                        "weight": c.weight, "n_members": len(c.members),
                        "n_decision": c.n_decision, "n_magnet": c.n_magnet,
                        "width": c.width,
                    }
                else:
                    cur["last"] = i
        # a track that was not re-emitted this pass has ended; it dies at the
        # start of the interval it went missing in, never at the end, so no
        # decision can act on a level after it stopped being one
        for key in [k for k in live if k not in seen]:
            t = live.pop(key)
            t["dead"] = min(t["last"] + every, n)
            done.append(t)

    for t in live.values():
        t["dead"] = n
        done.append(t)

    if not done:
        return Tracks(*[np.empty(0) for _ in Tracks._F])
    return Tracks(
        price=np.array([t["price"] for t in done], np.float64),
        born=np.array([t["born"] for t in done], np.int64),
        dead=np.array([t["dead"] for t in done], np.int64),
        weight=np.array([t["weight"] for t in done], np.float64),
        n_members=np.array([t["n_members"] for t in done], np.int64),
        n_decision=np.array([t["n_decision"] for t in done], np.int64),
        n_magnet=np.array([t["n_magnet"] for t in done], np.int64),
        width=np.array([t["width"] for t in done], np.float64),
    )


def load_series(symbol: str, tfs=("M1", "M5", "M15", "H1", "H4", "D1"),
                base: str | None = None) -> dict[str, Bars]:
    out = {}
    for tf in tfs:
        try:
            out[tf] = Bars(symbol, tf, base=base)
        except FileNotFoundError:
            pass
    return out
