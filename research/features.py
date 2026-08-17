"""Which features actually separate outcome? Measured, not assumed.

The old project's method was: think of a filter, reason about why it should work,
ship it. Six of those stacked multiplicatively cut the trade rate to 7% of
population for a break-even edge, and the 14-year decomposition later found every
computed feature sitting at |rho| < 0.08 against result.

This does the opposite. It generates a large, deliberately UNDER-filtered
candidate set, attaches every feature the engine can compute, runs them all
through **identical trade geometry**, and then asks which features move the
number. Fixed geometry matters: if each candidate got its own structural stop and
R:R gate, a "feature" could look predictive purely because it correlates with
having a tighter stop.

Baseline to beat, established by research/control.py on the same 14.6 years:

    random entry, 1.5 ATR stop, 2R target  ->  30.8% win, -0.181R

Anything at or below that is noise wearing a name.

    python -m research.features
    python -m research.features --top 25       # widest separators only
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bias as BI  # noqa: E402
from core import majors as MJ  # noqa: E402
from core import patterns as PT  # noqa: E402
from core import session as SE  # noqa: E402
from core.context import Context  # noqa: E402
from core.discover import build_points, level_tracks, load_series  # noqa: E402
from core.exits import NO_TRAIL, Plan, simulate  # noqa: E402
from core.structure import find_breaks  # noqa: E402
from research.engine import FUNDED_SPREAD_MULT  # noqa: E402

STOP_ATR = 1.5
TARGET_R = 2.0
RANDOM_BASELINE = -0.181          # research/control.py, same data, same geometry


def build(symbol="XAUUSD", base="M5", pb_near=0.5, pb_far=0.618, pb_max=48,
          spread_mult=FUNDED_SPREAD_MULT, quiet=False,
          target_r=TARGET_R, hold_hours=24,
          stop_pips=None, target_pips=None, series=None, since_bar=None):
    """Generate the candidate book.

    `series` lets a caller supply its own bars instead of reading the store —
    which is how `live/signals.py` runs. That parameter exists specifically so
    the live loop calls THIS function rather than reimplementing the entry
    logic: a second implementation is how a backtest and a live system quietly
    stop being the same system, and it is exactly what went wrong in the old
    tree. `since_bar` then keeps only candidates at or after a bar index, so
    live can ask "is there a signal on the bar that just closed".
    """
    series = load_series(symbol) if series is None else series
    bars, m1 = series[base], series["M1"]
    ctx = Context(series, base=base)
    atr = ctx.threshold("stop") / 1.50
    pip = bars.pip

    _p = (lambda *x, **k: None) if quiet else print
    _p("  levels ...", flush=True)
    pts = build_points(series, ctx, base)
    tracks = level_tracks(pts, ctx, every=12)
    _p(f"    {len(pts):,} points -> {len(tracks):,} tracks", flush=True)

    _p("  majorness ...", flush=True)
    mj = MJ.measure(bars, tracks, atr)
    _p(f"    touches: median {np.median(mj.touches):.0f}  "
          f"max {mj.touches.max()}   "
          f"respect_rate median {np.median(mj.respect_rate):.2f}", flush=True)

    _p("  structure ...", flush=True)
    brk = find_breaks(bars, k=3, atr=atr)
    _p(f"    {len(brk):,} breaks", flush=True)

    _p("  patterns ...", flush=True)
    pat = PT.read(bars, atr)
    _p(f"    {len(pat.all_names)} distinct patterns", flush=True)

    _p("  session/volatility ...", flush=True)
    st = SE.build(bars, atr)

    _p("  H4/H1 bias cascade ...", flush=True)
    bi = BI.build(series, ctx, base)

    _p("  major-break filter ...", flush=True)
    mb, lvl_idx = MJ.major_breaks(brk, mj, atr, mask=MJ.is_major(mj, min_touches=1))
    _p(f"    {len(mb):,} of {len(brk):,} breaks landed on a tracked level",
          flush=True)

    # ── candidates: break -> leg -> pullback -> entry ────────────────────────
    high = np.asarray(bars.high, np.float64)
    low = np.asarray(bars.low, np.float64)
    close = np.asarray(bars.close, np.float64)
    n = len(bars)
    m1_at = np.searchsorted(np.asarray(m1.time, np.int64),
                            np.asarray(bars.time, np.int64) + bars.bar_seconds,
                            "left")
    half_m1 = (m1.spread_pips() * pip * spread_mult) / 2.0
    sp_base = bars.spread_pips() * pip * spread_mult
    tsec = np.asarray(bars.time, np.int64)
    year = (tsec.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)
    # UTC day index — the unit a prop firm's daily-loss rule is enforced on
    day = tsec // 86400

    rows = []
    for j in range(len(mb)):
        up = mb.direction[j] == "up"
        d = "buy" if up else "sell"
        e_bar = int(mb.extreme_bar[j])
        ext = float(mb.extreme[j])
        leg = abs(ext - float(mb.origin[j]))
        if leg <= 0:
            continue
        near = ext - leg * pb_near if up else ext + leg * pb_near
        far = ext - leg * pb_far if up else ext + leg * pb_far
        s, e = e_bar + 1, min(e_bar + 1 + pb_max, n)
        if e <= s:
            continue
        inz = ((low[s:e] <= near) & (high[s:e] >= far) if up
               else (high[s:e] >= near) & (low[s:e] <= far))
        w = np.flatnonzero(inz)
        if not w.size:
            continue
        i = s + int(w[0])
        if since_bar is not None and i < since_bar:
            continue
        a = float(atr[i])
        if not np.isfinite(a) or a <= 0:
            continue

        entry = close[i] + (sp_base[i] / 2 if d == "buy" else -sp_base[i] / 2)
        risk = (stop_pips * pip) if stop_pips else (STOP_ATR * a)
        stop = entry - risk if d == "buy" else entry + risk
        if target_pips:
            tdist = target_pips * pip
        elif target_r is not None:
            tdist = target_r * risk
        else:
            tdist = None
        target = None if tdist is None else (entry + tdist if d == "buy"
                                             else entry - tdist)
        p1, p2 = int(m1_at[i]), min(int(m1_at[i]) + hold_hours * 60, len(m1))
        if p2 - p1 < 2:
            continue
        out = simulate(Plan(entry=entry, stop=stop, direction=d, risk=risk,
                            target=target, ladder=NO_TRAIL),
                       m1.high[p1:p2], m1.low[p1:p2], m1.close[p1:p2],
                       half_m1[p1:p2])

        li = int(lvl_idx[j])
        names = pat.names_at(i, d)
        rows.append({
            "bar": i, "year": int(year[i]), "day": int(day[i]),
            "ts": str(np.datetime64(int(tsec[i]), "s")),
            "direction": d, "kind": str(mb.kind[j]),
            # what the trade actually risked, so account sizing can be exact
            "risk_pips": risk / pip,
            # majorness of the broken level
            "touches": int(mj.touches[li]), "respects": int(mj.respects[li]),
            "respect_rate": float(mj.respect_rate[li]),
            "categories": int(mj.categories[li]),
            "level_age": int(mj.age[li]),
            "n_members": int(mj.n_members[li]),
            # leg / geometry
            "leg_atr": leg / a,
            "pullback_bars": i - e_bar,
            # context
            "hour": int(st.hour[i]), "session": st.session_name(i),
            "dow": int(st.dow[i]),
            "minutes_into": int(st.minutes_into[i]),
            "atr_pct": float(st.atr_pct[i]),
            "expansion": float(st.expansion[i]),
            "range_pct": float(st.range_pct[i]),
            "spread_pct": float(st.spread_pct[i]),
            "spread_pips": float(st.spread_pips[i]),
            "from_d1_open": float(st.from_d1_open[i]),
            "from_h4_open": float(st.from_h4_open[i]),
            "or_broken": int(st.or_broken[i]),
            # the multi-timeframe cascade the operator actually trades
            "h4_dir": int(bi.h4_dir[i]), "h1_dir": int(bi.h1_dir[i]),
            "htf_stack": bi.stack_for(i, d),
            "h4_pullback": float(BI.pullback_depth(bi.h4_pos[i:i+1],
                                                   bi.h4_dir[i:i+1])[0]),
            "h4_leg_atr": float(bi.h4_leg_atr[i]),
            "patterns": names,
            **out,
        })
    _p(f"    {len(rows):,} candidates\n", flush=True)
    return rows


# ── slicing ─────────────────────────────────────────────────────────────────

def stat(rows):
    if not rows:
        return None
    r = np.array([x["r"] for x in rows])
    return dict(n=len(r), win=(r > 0.05).mean(), avg=r.mean(),
                lift=r.mean() - RANDOM_BASELINE)


def show(rows, label, keyfn, minn=60, top=None):
    b = defaultdict(list)
    for x in rows:
        for k in (keyfn(x) if isinstance(keyfn(x), list) else [keyfn(x)]):
            b[k].append(x)
    out = []
    for k, v in b.items():
        s = stat(v)
        if s and s["n"] >= minn:
            out.append((k, s))
    if not out:
        return
    out.sort(key=lambda kv: -kv[1]["lift"])
    if top:
        out = out[:top] + ([("...", None)] if len(out) > top else [])
    print(f"\n  {label}   (baseline {RANDOM_BASELINE:+.3f}R)")
    for k, s in out:
        if s is None:
            print("    ...")
            continue
        flag = "  <<<" if s["lift"] > 0.15 and s["n"] >= 100 else ""
        print(f"    {str(k)[:26]:<26} n={s['n']:>5}  win={s['win']:>5.1%}  "
              f"avg={s['avg']:>+6.3f}R  lift={s['lift']:>+6.3f}R{flag}")


def bucket(v, edges):
    for i, e in enumerate(edges):
        if v < e:
            return f"<{e}"
    return f">={edges[-1]}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--top", type=int, default=None)
    a = p.parse_args(argv)

    rows = build(a.symbol)
    if not rows:
        raise SystemExit("no candidates")

    print("=" * 92)
    print(f"  FEATURE STUDY — {len(rows):,} candidates, identical geometry "
          f"({STOP_ATR} ATR stop, {TARGET_R}R target, no trail)")
    print("=" * 92)
    s = stat(rows)
    print(f"  ALL                        n={s['n']:>5}  win={s['win']:>5.1%}  "
          f"avg={s['avg']:>+6.3f}R  lift={s['lift']:>+6.3f}R")

    show(rows, "by level touches", lambda x: bucket(x["touches"], [1, 2, 3, 5, 8]))
    show(rows, "by respect rate", lambda x: bucket(x["respect_rate"], [0.5, 0.7, 0.85, 0.95]))
    show(rows, "by source categories", lambda x: x["categories"])
    show(rows, "by level age (bars)", lambda x: bucket(x["level_age"], [50, 200, 800, 3000]))
    show(rows, "by break kind", lambda x: x["kind"])
    show(rows, "by leg size (ATR)", lambda x: bucket(x["leg_atr"], [2, 4, 7, 12]))
    show(rows, "by session", lambda x: x["session"])
    show(rows, "by hour UTC", lambda x: f"{x['hour']:02d}", minn=100, top=a.top)
    show(rows, "by day of week", lambda x: x["dow"])
    show(rows, "by ATR percentile", lambda x: bucket(x["atr_pct"], [0.25, 0.5, 0.75, 0.9]))
    show(rows, "by expansion", lambda x: bucket(x["expansion"], [0.9, 1.0, 1.15, 1.4]))
    show(rows, "by day-range percentile", lambda x: bucket(x["range_pct"], [0.25, 0.5, 0.75]))
    show(rows, "by spread percentile", lambda x: bucket(x["spread_pct"], [0.25, 0.5, 0.75, 0.9]))
    show(rows, "by opening-range state", lambda x: x["or_broken"])
    show(rows, "by distance from D1 open (ATR)",
         lambda x: bucket(round(x["from_d1_open"], 1), [-3, -1, 0, 1, 3]))
    show(rows, "by direction", lambda x: x["direction"])
    # The cascade the operator says is where the money is: H4/H1 set the bias,
    # the lower timeframes find the pullback inside it.
    show(rows, "by HTF STACK (H4+H1 agreeing)", lambda x: x["htf_stack"])
    show(rows, "by H4 direction vs trade",
         lambda x: ("with H4" if x["h4_dir"] == (1 if x["direction"] == "buy" else -1)
                    else "against H4" if x["h4_dir"] != 0 else "H4 flat"))
    show(rows, "by H1 direction vs trade",
         lambda x: ("with H1" if x["h1_dir"] == (1 if x["direction"] == "buy" else -1)
                    else "against H1" if x["h1_dir"] != 0 else "H1 flat"))
    show(rows, "by H4 pullback depth",
         lambda x: bucket(round(x["h4_pullback"], 2), [0.2, 0.4, 0.6, 0.8]))
    show(rows, "by H4 leg size (base ATRs)",
         lambda x: bucket(round(x["h4_leg_atr"]), [5, 15, 40, 100]))
    show(rows, "by CANDLE PATTERN", lambda x: x["patterns"] or ["(none)"],
         minn=40, top=a.top)

    # per-year sanity on the whole population
    print("\n  per year:")
    for y in sorted({x["year"] for x in rows}):
        sy = stat([x for x in rows if x["year"] == y])
        print(f"    {y}  n={sy['n']:>5}  win={sy['win']:>5.1%}  "
              f"avg={sy['avg']:>+6.3f}R  lift={sy['lift']:>+6.3f}R")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
