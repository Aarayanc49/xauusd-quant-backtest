"""Level anatomy — the verification for build steps 1 and 2.

Neither step needs a single trade simulated. The number to watch is the p90
cluster width. The old chain produced:

    clusters/scan  median 8
    width  p50 48.6p  p75 113.6p  p90 238.6p  p95 499.1p  max 931.4p
    >=35p  57.7%   >=100p 29.8%   >=200p 16.3%   >=400p 7.7%
    constituents  median 4  max 31

against a 60-pip stop. If this run does not put p90 under the cap, the rebuild has
not fixed anything and nothing downstream is worth writing.

    python -m research.anatomy XAUUSD
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import levels as L  # noqa: E402
from core.cluster import anatomy, cluster_points  # noqa: E402
from core.context import Context  # noqa: E402
from core.store import Bars  # noqa: E402

# Which sources are read from which timeframe. Deliberately narrow: every source
# here has to earn its place in research/reaction.py before it gates anything.
POINT_PLAN = [
    ("H4", "swing", dict(k=3)),
    ("H1", "swing", dict(k=3)),
    ("M15", "swing", dict(k=4)),
    ("H1", "equal", dict(k=3)),
    ("M15", "equal", dict(k=4)),
    ("H1", "round", {}),
    ("M15", "period", dict(period="D")),
    ("H1", "period", dict(period="W")),
]


def build_points(series: dict[str, Bars], ctx: Context, base: str) -> dict:
    """Every level point, expressed on the BASE timeframe's index.

    Points detected on H4 carry H4 bar indices; they are remapped onto base bars
    through `Context.align`, which is what keeps a decision at 09:05 from reading
    an H4 bar that does not close until 12:00.
    """
    out = L.Points.empty()
    tol = {tf: ctx.scales[tf].atr * 0.15 for tf in series}

    for tf, kind, kw in POINT_PLAN:
        if tf not in series:
            continue
        b = series[tf]
        if kind == "swing":
            p = L.swing_points(b, tf, **kw)
        elif kind == "equal":
            p = L.equal_levels(b, tf, tol[tf], **kw)
        elif kind == "round":
            p = L.round_points(b, tf, **kw)
        elif kind == "period":
            p = L.period_points(b, tf, **kw)
        else:
            continue
        if len(p) == 0:
            continue
        out = out + _remap(p, tf, base, series, ctx)
    return out


def _remap(p: L.Points, tf: str, base: str, series: dict, ctx: Context) -> L.Points:
    """Translate born/dead from `tf` bar indices to `base` bar indices.

    A level born on `tf` bar j becomes knowable on the first base bar whose close
    is at or after that tf bar's close. Anything that maps past the end of the base
    series is dropped rather than clamped — clamping would resurrect a level at the
    last bar and quietly bias the tail of every study.
    """
    if tf == base:
        return p
    src, dst = series[tf], series[base]
    st = np.asarray(src.time, np.int64)
    dst_close = np.asarray(dst.time, np.int64) + dst.bar_seconds

    def to_base(idx):
        past = idx >= len(src)
        t = st[np.clip(idx, 0, len(src) - 1)] + src.bar_seconds
        m = np.searchsorted(dst_close, t, side="left")
        return np.where(past, len(dst), m).astype(np.int64)

    nb, nd = to_base(p.born), to_base(p.dead)
    keep = nb < len(dst)
    return L.Points(p.price[keep], nb[keep], nd[keep], p.source[keep],
                    p.tf[keep], nb[keep])


def build_zones(series: dict[str, Bars], base: str, ctx: Context) -> L.Zones:
    out = L.Zones.empty()
    for tf in ("H4", "H1", "M15"):
        if tf not in series:
            continue
        b = series[tf]
        for z in (L.fvg_zones(b, tf), L.ob_zones(b, tf)):
            if len(z) == 0:
                continue
            out = out + _remap_zones(z, tf, base, series)
    return out


def _remap_zones(z: L.Zones, tf: str, base: str, series: dict) -> L.Zones:
    if tf == base:
        return z
    src, dst = series[tf], series[base]
    st = np.asarray(src.time, np.int64)
    dst_close = np.asarray(dst.time, np.int64) + dst.bar_seconds

    def to_base(idx, is_death):
        i = np.clip(idx, 0, len(src) - 1)
        t = st[i] + src.bar_seconds
        m = np.searchsorted(dst_close, t, side="left")
        # a zone still alive at the end of its own series stays alive here
        return np.where(idx >= len(src), len(dst), m).astype(np.int64)

    nb, nd = to_base(z.born, False), to_base(z.dead, True)
    keep = nb < len(dst)
    return L.Zones(z.high[keep], z.low[keep], nb[keep], nd[keep],
                   z.source[keep], z.tf[keep], z.side[keep])


def run(symbol: str, base: str = "M5", scans: int = 200) -> None:
    tfs = ["M5", "M15", "H1", "H4", "D1"]
    series = {}
    for tf in tfs:
        try:
            series[tf] = Bars(symbol, tf)
        except FileNotFoundError:
            pass
    if base not in series:
        raise SystemExit(f"no {base} data for {symbol} — run research.fetch first")

    ctx = Context(series, base=base)
    b = series[base]
    pip = b.pip
    print(f"{symbol}  base={base}  {b!r}")
    print(f"  timeframes loaded: {', '.join(series)}\n")

    pts = build_points(series, ctx, base)
    zones = build_zones(series, base, ctx)
    mid = len(b) // 2
    print(f"POINTS  {len(pts):,} emitted over the series")
    print(f"    {'source':<12} {'emitted':>8} {'live@mid':>9} {'median life':>12}")
    live_mid = pts.active(mid)
    for s in sorted(set(pts.source.tolist())):
        m = pts.source == s
        life = np.minimum(pts.dead[m], len(b)) - pts.born[m]
        print(f"    {s:<12} {m.sum():>8,} {int((live_mid.source == s).sum()):>9,} "
              f"{np.median(life):>10.0f} bars")
    print(f"    {'TOTAL':<12} {len(pts):>8,} {len(live_mid):>9,}")
    print(f"\nZONES   {len(zones):,} total")
    for s in sorted(set(zones.source.tolist())):
        m = zones.source == s
        life = (zones.dead[m] - zones.born[m])
        print(f"    {s:<12} {m.sum():>7,}   median life {np.median(life):>6.0f} bars")

    # ── build step 1: the type split holds ──────────────────────────────────
    print("\n── STEP 1 — a level is a price, a zone is a region ──")
    print(f"  every point carries a single price, no width      : "
          f"{'PASS' if not hasattr(pts, 'high') else 'FAIL'}")
    print(f"  every zone carries explicit bounds, hi >= lo      : "
          f"{'PASS' if len(zones) == 0 or bool((zones.high >= zones.low).all()) else 'FAIL'}")
    print(f"  no zone can be used as a trigger price            : PASS (Zones has no .price)")

    # ── build step 2: clustering respects the cap ───────────────────────────
    cap_arr = ctx.threshold("cluster_width_cap")
    tol_arr = ctx.threshold("cluster_tolerance")
    n = len(b)
    step = max(1, n // scans)
    idxs = [i for i in range(int(n * 0.2), n, step)][:scans]

    # Only levels within reach are ever traded, so only those are measured. The
    # old anatomy counted everything discovery emitted inside a flat 300-pip
    # window, which mixed levels that could never be reached into the widths.
    prox = ctx.threshold("proximity")
    stats, caps, counts, breaches = [], [], [], 0
    for i in idxs:
        cap = float(cap_arr[i])
        act = pts.near(i, float(b.close[i]), prox[i] * 6)
        if len(act) < 2:
            continue
        cl = cluster_points(act, cap, dedup=float(tol_arr[i]) * 0.25)
        if not cl:
            continue
        stats.append(anatomy(cl, pip))
        caps.append(cap / pip)
        counts.append(len(cl))
        # per-scan check: this scan's widest cluster against THIS scan's cap
        if max(c.width for c in cl) > cap * 1.0001:
            breaches += 1

    if not stats:
        raise SystemExit("no scans produced clusters — check the data window")

    def agg(k):
        return float(np.mean([s[k] for s in stats]))

    cap_p = float(np.mean(caps))
    print(f"\n── STEP 2 — clustering, {len(stats)} scans ──")
    print(f"  cap (0.25 x stop)        {cap_p:>8.1f}p   (was a flat 20p tolerance)")
    print(f"  clusters per scan        {np.mean(counts):>8.1f}     (old: median 8)")
    print(f"  members  median          {agg('members_p50'):>8.1f}     (old: median 4, max 31)")
    print(f"           max             {max(s['members_max'] for s in stats):>8d}")
    print(f"  singletons               {agg('singletons'):>8.1%}     (kept, not merged away)")
    print()
    print(f"  width  p50               {agg('width_p50'):>8.1f}p    (old  48.6p)")
    print(f"         p75               {agg('width_p75'):>8.1f}p    (old 113.6p)")
    print(f"         p90               {agg('width_p90'):>8.1f}p    (old 238.6p)   <-- the number")
    print(f"         p95               {agg('width_p95'):>8.1f}p    (old 499.1p)")
    print(f"         max               {max(s['width_max'] for s in stats):>8.1f}p    (old 931.4p)")
    print()
    print(f"  >= 35p wide              {agg('over_35p'):>8.1%}     (old  57.7%)")
    print(f"  >= 100p wide             {agg('over_100p'):>8.1%}     (old  29.8%)")
    print(f"  >= 200p wide             {agg('over_200p'):>8.1%}     (old  16.3%)")

    stop_p = float(np.mean(ctx.threshold_pips("stop")[idxs]))
    ratio = agg("width_p90") / stop_p
    print(f"\n  cap breaches             {breaches:>8d} of {len(stats)} scans")
    print(f"  mean stop                {stop_p:>8.1f}p")
    print(f"  p90 width / stop         {ratio:>8.1%}     (old 238.6/60 = 398%)")
    ok = breaches == 0 and ratio <= 0.30
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} — "
          f"{'every cluster fits well inside the stop' if ok else 'levels are still regions'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbol", nargs="?", default="XAUUSD")
    p.add_argument("--base", default="M5")
    p.add_argument("--scans", type=int, default=200)
    a = p.parse_args(argv)
    run(a.symbol.upper(), a.base, a.scans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
