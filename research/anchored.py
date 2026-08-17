"""The INTRADAY candidate set — time-anchored events, measured the honest way.

A separate system from the swing trader, with a different objective function, and
the difference is the whole point.

## Why intraday cannot be the swing system on a shorter leash

Two measured facts settle the design before any code runs.

**1. The tail is where the swing edge lives.** Same entries, different target:
2R = +0.074R, 8R = +0.499R. Eighty-five percent of that edge is past 2R. A
system that is flat within the hour cannot reach it. So intraday has to earn its
money somewhere else.

**2. A tight stop is a cost decision, not a risk decision.** Spread cost scales
as 1/stop: -0.37R at 0.75 ATR, -0.18R at 1.5 ATR, -0.10R at 3 ATR. And the
median 1.5-ATR stop on gold was **24 pips in 2012-2019 and 69 pips in 2020-2026**.
A 20-30 pip stop was right for the market it was learned in; today it is ~0.5 ATR
and hands roughly **0.54R per trade** to the broker before the trade starts. That
is the entire explanation for the earlier intraday attempt returning +0.041R.

So: intraday means **holding for an hour, not risking 25 pips.** The stop stays
ATR-scaled; the hold gets short.

## The objective

    swing      89-134 trades/yr  x  +0.65R   =  ~60-90R/yr
    intraday  600-1200 trades/yr x  +0.10R   =  ~60-120R/yr

Intraday does not need a large edge. It needs a small positive one at high
frequency. Filtering this down to +0.6R per trade would rebuild the swing system
by accident, at 40 trades a year.

## Anchors, not discovered levels

Every trigger here fires off a price the CLOCK puts on the chart — the Asia
range, yesterday's high, the previous H1 candle's low, the session's opening
range. That is a different hypothesis from the sweep/fade family that measured
-0.191R (identical to random), which faded levels an ALGORITHM had discovered. A
discovered level is a property of the discovering algorithm; yesterday's high is
a property of the market, and it is where stops actually rest.

## The one idea being tested

At every anchor there are two opposite trades — fade the sweep, or join the
break. The old tree always faded, and measured noise. Here both are emitted as
separate events and the arbitration is left to the H4 bias, which is the axis
that measured strongly and was independently re-picked by the walk-forward:

    fade when the sweep runs AGAINST the H4 bias
    go   when the break runs WITH it

That is a hypothesis, not a filter. `--report` prints event x direction x bias so
it either separates or it does not.

## Method

Deliberately UNDER-filtered, exactly as `research/features.py` was: every event
fires, identical geometry for all of them, every feature attached, and the
question "which of these separates outcome" is answered afterwards rather than
assumed. Per-year and two-halves on every claim.

    python -m research.anchored                  # build + report
    python -m research.anchored --hold 60 --target-r 2
    python -m research.anchored --sweep          # hold x target grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import anchors as AN  # noqa: E402
from core import bias as BI  # noqa: E402
from core import mtf as MT  # noqa: E402
from core import session as SE  # noqa: E402
from core import shape as SH  # noqa: E402
from core.context import Context  # noqa: E402
from core.discover import load_series  # noqa: E402
from core.exits import NO_TRAIL, Plan, simulate  # noqa: E402
from research.engine import FUNDED_SPREAD_MULT  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "_anchored.json")

# Intraday defaults. The stop stays ATR-scaled for the cost reason above; the
# hold is the operator's own constraint ("i dont think of trading for more than
# ... holding a trade over an hour").
STOP_ATR = 1.0
TARGET_R = 2.0
HOLD_MIN = 60
TRADING_HOURS = (7, 21)      # London open .. NY close. Never Asia, never rollover.


def _first_in_group(cond: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Mask marking the FIRST true bar of `cond` within each group of `g`.

    One fire per anchor instance. Without this a swept level re-triggers on
    every subsequent bar it is still beyond, which would silently weight the
    sample towards whichever days trended hardest.
    """
    idx = np.flatnonzero(cond)
    out = np.zeros(len(cond), bool)
    if idx.size:
        gg = g[idx]
        out[idx[np.concatenate(([True], gg[1:] != gg[:-1]))]] = True
    return out


def build_events(bars, an, st, sess, close, high, low, atr):
    """Every time-anchored trigger, as (name, direction, mask) tuples.

    `direction` is the trade the event implies. A SWEEP is a fade — price ran the
    level and closed back, so the trade is away from the sweep. A BREAK is
    continuation — price closed through and stayed. Both are emitted; which one
    pays, and under what bias, is the question being measured.
    """
    day = an.fx_day
    ev = []

    def add(name, direction, cond, group):
        ev.append((name, direction, _first_in_group(cond, group)))

    # ── daily anchors ───────────────────────────────────────────────────────
    for label, lvl, taken in (("asia", an.asia_high, an.asia_hi_taken),
                              ("pd", an.pd_high, an.pd_hi_taken)):
        ok = np.isfinite(lvl)
        # swept and rejected: the level has been run at some point today and this
        # bar closes back underneath it
        add(f"{label}_hi_sweep", "sell", ok & taken & (close < lvl), day)
        # broken and holding: first close beyond it
        add(f"{label}_hi_break", "buy", ok & (close > lvl), day)

    for label, lvl, taken in (("asia", an.asia_low, an.asia_lo_taken),
                              ("pd", an.pd_low, an.pd_lo_taken)):
        ok = np.isfinite(lvl)
        add(f"{label}_lo_sweep", "buy", ok & taken & (close > lvl), day)
        add(f"{label}_lo_break", "sell", ok & (close < lvl), day)

    # previous week — one fire per week, not per day, or a level that stays
    # broken all week re-triggers every session
    week = day // 7
    okh, okl = np.isfinite(an.pw_high), np.isfinite(an.pw_low)
    add("pw_hi_sweep", "sell", okh & (high > an.pw_high) & (close < an.pw_high), week)
    add("pw_hi_break", "buy", okh & (close > an.pw_high), week)
    add("pw_lo_sweep", "buy", okl & (low < an.pw_low) & (close > an.pw_low), week)
    add("pw_lo_break", "sell", okl & (close < an.pw_low), week)

    # ── session opening range ───────────────────────────────────────────────
    orh, orl = sess.or_high, sess.or_low
    ok = np.isfinite(orh) & np.isfinite(orl) & (sess.minutes_into > 30)
    add("or_hi_break", "buy", ok & (close > orh), day)
    add("or_lo_break", "sell", ok & (close < orl), day)
    add("or_hi_sweep", "sell", ok & (high > orh) & (close < orh), day)
    add("or_lo_sweep", "buy", ok & (low < orl) & (close > orl), day)

    # ── inter-candle: the trade-rate engine ─────────────────────────────────
    # The previous H1/H4/M15 candle's extreme is resting liquidity that refreshes
    # every hour rather than once a day, which is where 600-1,200 trades/yr comes
    # from. `took_prev_*` is running state on the FORMING candle, so a sweep is
    # "this candle already ran the last one's high, and we have closed back".
    for tf in ("M15", "M30", "H1", "H4"):
        s = st.get(tf)
        if s is None:
            continue
        g = s.group
        p = tf.lower()
        okh = np.isfinite(s.prev_high)
        okl = np.isfinite(s.prev_low)
        add(f"{p}_hi_sweep", "sell", okh & s.took_prev_high & (close < s.prev_high), g)
        add(f"{p}_lo_sweep", "buy", okl & s.took_prev_low & (close > s.prev_low), g)
        add(f"{p}_hi_break", "buy", okh & (close > s.prev_high), g)
        add(f"{p}_lo_break", "sell", okl & (close < s.prev_low), g)

    return ev


class Prep:
    """Everything expensive, computed once.

    Separating event GENERATION from trade SIMULATION is not tidiness — it is
    what makes the hold-vs-target question askable. The events do not depend on
    the geometry, so re-deriving anchors, four timeframes of candle state and
    half a million triggers for every cell of a sweep would be re-running the
    only slow part of the study sixteen times over.
    """
    __slots__ = ("bars", "m1", "atr", "pip", "events", "hours_ok", "close",
                 "year", "tsec", "m1_at", "half_m1", "sp_base", "an", "st",
                 "m5", "sess", "bi")


def prepare(symbol="XAUUSD", base="M5", spread_mult=FUNDED_SPREAD_MULT,
            quiet=False) -> Prep:
    _p = (lambda *x, **k: None) if quiet else print
    series = load_series(symbol)
    bars, m1 = series[base], series["M1"]
    ctx = Context(series, base=base)
    atr = ctx.threshold("stop") / 1.50
    pip = bars.pip

    _p("  anchors ...", flush=True)
    an = AN.build(bars, atr)
    _p("  timeframes ...", flush=True)
    st = MT.build(bars, ctx, atr)
    _p("  base candle anatomy ...", flush=True)
    m5 = SH.read(bars, atr)
    _p("  session/volatility ...", flush=True)
    sess = SE.build(bars, atr)
    _p("  H4/H1 bias ...", flush=True)
    bi = BI.build(series, ctx, base)

    close = np.asarray(bars.close, np.float64)
    high = np.asarray(bars.high, np.float64)
    low = np.asarray(bars.low, np.float64)
    n = len(bars)
    tsec = np.asarray(bars.time, np.int64)
    year = (tsec.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)
    m1_at = np.searchsorted(np.asarray(m1.time, np.int64),
                            tsec + bars.bar_seconds, "left")
    half_m1 = (m1.spread_pips() * pip * spread_mult) / 2.0
    sp_base = bars.spread_pips() * pip * spread_mult

    _p("  events ...", flush=True)
    events = build_events(bars, an, st, sess, close, high, low, atr)

    # Trade only the liquid part of the day. Rollover measured -0.40R with double
    # spread and Asia -0.32R; a system that is flat 21:00-07:00 is not a filter,
    # it is a refusal to pay for the privilege of being there.
    hours_ok = (an.hour >= TRADING_HOURS[0]) & (an.hour < TRADING_HOURS[1])

    p = Prep()
    (p.bars, p.m1, p.atr, p.pip, p.events, p.hours_ok, p.close, p.year,
     p.tsec, p.m1_at, p.half_m1, p.sp_base, p.an, p.st, p.m5, p.sess, p.bi) = (
        bars, m1, atr, pip, events, hours_ok, close, year, tsec, m1_at,
        half_m1, sp_base, an, st, m5, sess, bi)
    return p


def run_geometry(p: Prep, stop_atr=STOP_ATR, target_r=TARGET_R,
                 hold_min=HOLD_MIN, every=1, quiet=False):
    """Simulate every prepared event under one stop/target/hold geometry.

    `every` subsamples the event stream for sweeps — 500k events x 16 cells is a
    lot of simulation for a question that a tenth of the sample answers to three
    decimal places.
    """
    _p = (lambda *x, **k: None) if quiet else print
    (bars, m1, atr, pip, events, hours_ok, close, year, tsec, m1_at,
     half_m1, sp_base, an, st, m5, sess, bi) = (
        p.bars, p.m1, p.atr, p.pip, p.events, p.hours_ok, p.close, p.year,
        p.tsec, p.m1_at, p.half_m1, p.sp_base, p.an, p.st, p.m5, p.sess, p.bi)

    rows = []
    for name, direction, mask in events:
        fire = np.flatnonzero(mask & hours_ok)
        if every > 1:
            fire = fire[::every]
        for i in fire:
            i = int(i)
            a = float(atr[i])
            if not np.isfinite(a) or a <= 0:
                continue
            buy = direction == "buy"
            entry = close[i] + (sp_base[i] / 2 if buy else -sp_base[i] / 2)
            risk = stop_atr * a
            stop = entry - risk if buy else entry + risk
            target = (entry + target_r * risk) if buy else (entry - target_r * risk)
            p1 = int(m1_at[i])
            p2 = min(p1 + hold_min, len(m1))
            if p2 - p1 < 2:
                continue
            out = simulate(Plan(entry=entry, stop=stop, direction=direction,
                                risk=risk, target=target, ladder=NO_TRAIL),
                           m1.high[p1:p2], m1.low[p1:p2], m1.close[p1:p2],
                           half_m1[p1:p2])

            kind = "sweep" if name.endswith("_sweep") else "break"
            anchor = name.rsplit("_", 2)[0]
            h4d = int(bi.h4_dir[i])
            h1d = int(bi.h1_dir[i])
            want = 1 if buy else -1
            r = {
                "bar": i, "year": int(year[i]), "day": int(tsec[i] // 86400),
                "ts": str(np.datetime64(int(tsec[i]), "s")),
                "event": name, "anchor": anchor, "kind": kind,
                "direction": direction,
                "risk_pips": risk / pip,
                "pips": None,          # filled below
                # context
                "hour": int(an.hour[i]), "session": sess.session_name(i),
                "dow": int(sess.dow[i]),
                "atr_pct": float(sess.atr_pct[i]),
                "range_pct": float(sess.range_pct[i]),
                "expansion": float(sess.expansion[i]),
                "spread_pct": float(sess.spread_pct[i]),
                "spread_pips": float(sess.spread_pips[i]),
                # the bias arbitration this whole design rests on
                "h4_dir": h4d, "h1_dir": h1d,
                "with_h4": h4d == want, "with_h1": h1d == want,
                "htf_stack": int(h4d == want) + int(h1d == want),
                "h4_pullback": float(BI.pullback_depth(bi.h4_pos[i:i + 1],
                                                       bi.h4_dir[i:i + 1])[0]),
                # anchor geometry
                "asia_range_atr": float(an.asia_range_atr[i])
                if np.isfinite(an.asia_range_atr[i]) else -1.0,
                "day_open_dist": float(sess.from_d1_open[i]),
                **out,
            }
            r["pips"] = r["r"] * r["risk_pips"]
            r.update(m5.row(i, prefix="m5_"))
            r.update(MT.row(st, i))
            rows.append(r)

    rows.sort(key=lambda x: x["bar"])
    _p(f"    {len(rows):,} candidates\n", flush=True)
    return rows


def build(symbol="XAUUSD", base="M5", stop_atr=STOP_ATR, target_r=TARGET_R,
          hold_min=HOLD_MIN, spread_mult=FUNDED_SPREAD_MULT, quiet=False):
    """Prepare and simulate in one call — the single-geometry path."""
    p = prepare(symbol, base, spread_mult, quiet)
    return run_geometry(p, stop_atr, target_r, hold_min, quiet=quiet)


# ── reporting ───────────────────────────────────────────────────────────────

def stat(rows):
    if not rows:
        return None
    r = np.array([x["r"] for x in rows])
    p = np.array([x["pips"] for x in rows])
    t = sorted(rows, key=lambda x: x["bar"])
    m = len(t) // 2
    ys = defaultdict(list)
    for x in rows:
        ys[x["year"]].append(x["r"])
    pos = sum(1 for v in ys.values() if np.mean(v) > 0)
    return dict(n=len(r), win=float((r > 0.05).mean()), avg=float(r.mean()),
                pips=float(p.mean()), sum_pips=float(p.sum()),
                h1=float(np.mean([x["r"] for x in t[:m]])) if m else 0.0,
                h2=float(np.mean([x["r"] for x in t[m:]])) if m else 0.0,
                pos=pos, nyears=len(ys))


def line(label, s, w=30):
    if s is None or not s["n"]:
        return f"    {label:<{w}} n=     0"
    ok = "OK " if s["h1"] > 0 and s["h2"] > 0 else "   "
    return (f"    {label:<{w}} n={s['n']:>6}  win={s['win']:>5.1%}  "
            f"avg={s['avg']:>+6.3f}R {s['pips']:>+6.1f}p  "
            f"tot={s['sum_pips']:>+8.0f}p  "
            f"halves {s['h1']:>+6.3f}/{s['h2']:>+6.3f} {ok} "
            f"yrs {s['pos']}/{s['nyears']}")


def show(rows, label, keyfn, minn=150, base=None):
    b = defaultdict(list)
    for x in rows:
        b[keyfn(x)].append(x)
    out = [(k, stat(v)) for k, v in b.items() if len(v) >= minn]
    if not out:
        return
    out.sort(key=lambda kv: -kv[1]["avg"])
    print(f"\n  {label}")
    for k, s in out:
        print(line(str(k), s))


def report(rows):
    all_s = stat(rows)
    print("\n" + "=" * 124)
    print(f"  INTRADAY ANCHORED EVENTS — {len(rows):,} candidates, "
          f"{STOP_ATR} ATR stop, {TARGET_R}R target, {HOLD_MIN} min max hold, "
          f"funded spread")
    print("=" * 124)
    print(line("ALL", all_s))

    show(rows, "by EVENT", lambda x: x["event"], minn=200)
    show(rows, "by anchor", lambda x: x["anchor"])
    show(rows, "by kind (fade the sweep vs join the break)", lambda x: x["kind"])

    # ── the design's central question ───────────────────────────────────────
    print("\n" + "=" * 124)
    print("  THE ARBITRATION — does H4 bias decide whether to fade or to go?")
    print("=" * 124)
    for kind in ("sweep", "break"):
        sub = [x for x in rows if x["kind"] == kind]
        print(f"\n  {kind.upper()}S")
        print(line("  all", stat(sub)))
        print(line("  with H4", stat([x for x in sub if x["with_h4"]])))
        print(line("  against H4", stat([x for x in sub if not x["with_h4"]])))
        print(line("  both TFs agree", stat([x for x in sub if x["htf_stack"] == 2])))
        print(line("  neither agrees", stat([x for x in sub if x["htf_stack"] == 0])))

    show(rows, "by session", lambda x: x["session"])
    show(rows, "by hour UTC", lambda x: f"{x['hour']:02d}")
    show(rows, "by spread percentile",
         lambda x: ("<0.25" if x["spread_pct"] < 0.25 else
                    "0.25-0.5" if x["spread_pct"] < 0.5 else
                    "0.5-0.75" if x["spread_pct"] < 0.75 else ">=0.75"))
    show(rows, "by day-range percentile",
         lambda x: ("<0.25" if x["range_pct"] < 0.25 else
                    "0.25-0.5" if x["range_pct"] < 0.5 else
                    "0.5-0.75" if x["range_pct"] < 0.75 else ">=0.75"))
    show(rows, "by M5 trigger-candle body (ATR)",
         lambda x: ("<0.25" if abs(x["m5_body_atr"]) < 0.25 else
                    "0.25-0.5" if abs(x["m5_body_atr"]) < 0.5 else
                    "0.5-1.0" if abs(x["m5_body_atr"]) < 1.0 else ">=1.0"))
    show(rows, "by M5 close location in its own range",
         lambda x: ("0.0-0.25" if x["m5_close_loc"] < 0.25 else
                    "0.25-0.5" if x["m5_close_loc"] < 0.5 else
                    "0.5-0.75" if x["m5_close_loc"] < 0.75 else "0.75-1.0"))
    show(rows, "by H1 minutes into the forming candle",
         lambda x: f"{int(x['h1_minutes_into']) // 15 * 15}-"
                   f"{int(x['h1_minutes_into']) // 15 * 15 + 15}")
    show(rows, "by previous H1 candle body (ATR)",
         lambda x: ("<-1" if x["h1_prev_body_atr"] < -1 else
                    "-1..-0.3" if x["h1_prev_body_atr"] < -0.3 else
                    "-0.3..0.3" if x["h1_prev_body_atr"] < 0.3 else
                    "0.3..1" if x["h1_prev_body_atr"] < 1 else ">=1"))
    show(rows, "by previous H4 candle direction",
         lambda x: {1: "H4 up", -1: "H4 down", 0: "H4 doji"}[x["h4_prev_dir"]])

    print("\n  per year:")
    for y in sorted({x["year"] for x in rows}):
        print(line(str(y), stat([x for x in rows if x["year"] == y])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--stop-atr", type=float, default=STOP_ATR)
    p.add_argument("--target-r", type=float, default=TARGET_R)
    p.add_argument("--hold", type=int, default=HOLD_MIN, help="max hold, minutes")
    p.add_argument("--sweep", action="store_true", help="hold x target grid")
    p.add_argument("--cache", action="store_true", help="write the candidate set")
    a = p.parse_args(argv)

    if a.sweep:
        # The question this grid exists to answer: are the ENTRIES worthless, or
        # is the ONE-HOUR HOLD what makes them worthless? Those have opposite
        # implications, and at a 60-minute cap 41% of trades time out, so the
        # cap is a live suspect. Same events in every cell — only the geometry
        # moves.
        p = prepare(a.symbol)
        holds = (30, 60, 240, 1440)
        targets = (1.5, 2.0, 4.0, 8.0)
        print("\n" + "=" * 116)
        print(f"  GEOMETRY GRID — {a.symbol}, {a.stop_atr} ATR stop, "
              f"identical events in every cell")
        print("=" * 116)
        print(f"  {'hold':<10}" + "".join(f"{('TP ' + str(t) + 'R'):>26}"
                                          for t in targets))
        for hold in holds:
            cells = []
            for tr in targets:
                rows = run_geometry(p, stop_atr=a.stop_atr, target_r=tr,
                                    hold_min=hold, every=8, quiet=True)
                s = stat(rows)
                cells.append(
                    f"{s['avg']:>+7.3f}R {s['pips']:>+6.1f}p {s['pos']}/{s['nyears']}yr"
                    if s else "   -")
            lbl = f"{hold}m" if hold < 1440 else "24h"
            print(f"  {lbl:<10}" + "".join(f"{c:>26}" for c in cells))
        print("\n  Compare each cell against the matched random control at the")
        print("  same geometry — a cell is only interesting if it beats that,")
        print("  not if it is merely less negative than its neighbours.")
        return 0

    rows = build(a.symbol, stop_atr=a.stop_atr, target_r=a.target_r,
                 hold_min=a.hold)
    if not rows:
        raise SystemExit("no candidates")
    if a.cache:
        with open(CACHE, "w") as f:
            json.dump(rows, f)
        print(f"  cached -> {CACHE}")
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
