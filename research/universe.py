"""The unbiased candidate universe — every family's features, no pre-selection.

## Why there are no "setups" here

Every study so far picked a trigger first and measured second: a structure break,
a sweep of a level, a break of a range. That embeds the answer in the question —
if the trigger carries nothing, no amount of downstream filtering recovers it,
which is exactly what happened to the anchored family (521,606 events, gross edge
-0.007R).

So this generator has no trigger. It samples the clock on a fixed grid and emits
BOTH directions at every sample. What separates a trade from a non-trade is left
entirely to the model in `research/model.py`. That is the difference between a
gate stack and a methodology: gates multiply and collapse — six of them took v10
to 7% of population for a break-even edge — whereas a score ranks, and the trade
rate becomes a threshold you choose rather than a survival rate you suffer.

## Direction signing, which is the subtle part

Emitting both directions doubles the sample and would ordinarily double the
parameters too: the model would have to learn "VWAP deviation is bullish when
negative" and separately "VWAP deviation is bearish when positive". Instead every
DIRECTIONAL feature is signed so that **positive always means favourable for this
candidate's direction**. A sell candidate sees the negated VWAP deviation, the
negated H4 direction, the negated flow delta.

Two subtleties that are easy to get wrong and were handled explicitly:

  * **Wicks swap rather than negate.** For a long, the lower wick is the
    rejection; for a short it is the upper. `up_wick` and `dn_wick` exchange
    values on sells rather than changing sign.
  * **Close location reflects.** `close_loc` of 0.9 (closed on its high) is
    favourable for a long and unfavourable for a short, so sells see `1 - x`.

Anything genuinely symmetric — ATR percentile, spread percentile, volume
relative to its own hour, hour of day — is left alone. Signing a symmetric
feature would make the buy and sell rows mirror images and destroy the
information.

## Families represented

    A clock       hour, weekday, day of month, minutes to the London gold fixes
    B volatility  ATR and range percentiles, expansion, Asia range width
    C reversion   VWAP deviation, band position, slope, distance from day open
    D momentum    returns over five lookbacks, candle runs, body sums
    E flow        volume vs its OWN hour, delta proxy, cumulative delta, divergence
    F cross-asset USDJPY (14.6y) and USTEC (2022+) returns and divergence
    G structure   H4/H1 direction and pullback depth, full per-timeframe candle state

    python -m research.universe --every 6 --cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import anchors as AN  # noqa: E402
from core import bias as BI  # noqa: E402
from core import flow as FL  # noqa: E402
from core import mtf as MT  # noqa: E402
from core import session as SE  # noqa: E402
from core import shape as SH  # noqa: E402
from core import vwap as VW  # noqa: E402
from core.context import Context  # noqa: E402
from core.discover import load_series  # noqa: E402
from core.exits import NO_TRAIL, Plan, simulate  # noqa: E402
from research.engine import FUNDED_SPREAD_MULT  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "_universe.json")

# Geometry. 1.5 ATR rather than the operator's 25 pips for the cost reason
# measured to death elsewhere: spread costs -0.18R at 1.5 ATR and -0.54R at 0.5.
# The 60-minute cap is the operator's own constraint. Matched random control at
# exactly this geometry is -0.163R — that is the number any model must beat.
STOP_ATR = 1.5
TARGET_R = 2.0
HOLD_MIN = 60
CONTROL_R = -0.163
TRADING_HOURS = (7, 21)

# Features whose sign flips for a sell candidate.
DIRECTIONAL = {
    "vwap_dev", "vwap_dev_london", "vwap_slope", "from_d1_open", "from_h4_open",
    "h4_dir", "h1_dir", "vol_delta", "vol_delta_cum", "vol_divergence",
    "ret_15m", "ret_1h", "ret_4h", "ret_1d", "ret_1w",
    "usdjpy_ret_1h", "usdjpy_ret_1d", "ustec_ret_1h", "ustec_ret_1d",
    "m5_body_atr", "m5_gap_atr", "m5_body_sum3", "m5_dir", "m5_engulf",
    "band_pos_c",
}
# Features that reflect around a midpoint (0..1 quantities) for a sell.
REFLECTED = {"m5_close_loc", "m5_open_loc"}
# Wick pairs that EXCHANGE on a sell rather than negate.
WICK_PAIRS = (("m5_up_wick_atr", "m5_dn_wick_atr"),
              ("m5_up_wick_frac", "m5_dn_wick_frac"))


def _sign_row(row: dict, buy: bool) -> dict:
    """Re-express a feature row from the point of view of the trade's direction."""
    if buy:
        return row
    out = dict(row)
    for k, v in row.items():
        base = k
        if base in DIRECTIONAL or base.endswith(("_body_atr", "_dir", "_engulf",
                                                 "_gap_atr", "_body_sum3",
                                                 "_from_open", "_from_prev_high",
                                                 "_from_prev_low")):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[base] = -v
        elif base in REFLECTED or base.endswith(("_close_loc", "_open_loc",
                                                 "_prev_range_pos")):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[base] = 1.0 - v
    for a, b in WICK_PAIRS:
        if a in row and b in row:
            out[a], out[b] = row[b], row[a]
    # per-timeframe wick pairs from core/mtf.py
    for tf in ("m15", "m30", "h1", "h4"):
        for a_suf, b_suf in (("up_wick_atr", "dn_wick_atr"),
                             ("up_wick_frac", "dn_wick_frac")):
            a, b = f"{tf}_prev_{a_suf}", f"{tf}_prev_{b_suf}"
            if a in row and b in row:
                out[a], out[b] = row[b], row[a]
        for a_suf, b_suf in (("took_prev_high", "took_prev_low"),):
            a, b = f"{tf}_{a_suf}", f"{tf}_{b_suf}"
            if a in row and b in row:
                out[a], out[b] = row[b], row[a]
    return out


def _returns(close: np.ndarray, atr: np.ndarray, bars_per: dict) -> dict:
    """Trailing returns over several horizons, expressed in ATRs."""
    n = len(close)
    a = np.maximum(atr, 1e-12)
    out = {}
    for name, k in bars_per.items():
        prev = np.empty(n)
        prev[:k] = close[:k]
        prev[k:] = close[:-k]
        out[name] = (close - prev) / a
    return out


def _align_other(bars, other_bars):
    """Index map from base bars to the newest CLOSED bar of another symbol.

    Cross-asset lookahead is the easiest fake edge in this whole registry: an
    index-for-index join lets a decision at 09:05 read a bar that has not
    finished. Matched on close time, same discipline as Context.align.
    """
    base_close = np.asarray(bars.time, np.int64) + bars.bar_seconds
    oth_close = np.asarray(other_bars.time, np.int64) + other_bars.bar_seconds
    return np.searchsorted(oth_close, base_close, side="right") - 1


def build(symbol="XAUUSD", base="M5", every=6, stop_atr=STOP_ATR,
          target_r=TARGET_R, hold_min=HOLD_MIN,
          spread_mult=FUNDED_SPREAD_MULT, quiet=False):
    _p = (lambda *x, **k: None) if quiet else print
    series = load_series(symbol)
    bars, m1 = series[base], series["M1"]
    ctx = Context(series, base=base)
    atr = ctx.threshold("stop") / 1.50
    pip = bars.pip
    n = len(bars)

    _p("  anchors ...", flush=True)
    an = AN.build(bars, atr)
    _p("  session ...", flush=True)
    sess = SE.build(bars, atr)
    _p("  timeframes ...", flush=True)
    st = MT.build(bars, ctx, atr)
    _p("  candle anatomy ...", flush=True)
    m5 = SH.read(bars, atr)
    _p("  flow ...", flush=True)
    fl = FL.build(bars, atr, an.fx_day)
    _p("  vwap ...", flush=True)
    vw = VW.build(bars, atr, an.fx_day, an.hour)
    _p("  bias ...", flush=True)
    bi = BI.build(series, ctx, base)

    close = np.asarray(bars.close, np.float64)
    tsec = np.asarray(bars.time, np.int64)
    year = (tsec.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)
    dom = ((tsec.astype("datetime64[s]").astype("datetime64[D]")
            - tsec.astype("datetime64[s]").astype("datetime64[M]")).astype(int) + 1)
    per_hour = max(1, 3600 // bars.bar_seconds)
    rets = _returns(close, atr, {"ret_15m": max(1, per_hour // 4),
                                 "ret_1h": per_hour,
                                 "ret_4h": per_hour * 4,
                                 "ret_1d": per_hour * 24,
                                 "ret_1w": per_hour * 24 * 5})

    # ── cross-asset, causally aligned ───────────────────────────────────────
    cross = {}
    for other in ("USDJPY", "USTEC"):
        if other == symbol:
            continue
        try:
            os_ = load_series(other)
        except Exception:
            continue
        ob = os_.get(base)
        if ob is None:
            continue
        _p(f"  cross-asset {other} ...", flush=True)
        m = _align_other(bars, ob)
        oc = np.asarray(ob.close, np.float64)
        oatr = np.maximum(np.asarray(
            Context(os_, base=base).threshold("stop") / 1.5, np.float64), 1e-12)
        orets = _returns(oc, oatr, {"1h": per_hour, "1d": per_hour * 24})
        ok = m >= 0
        for k, arr in orets.items():
            col = np.zeros(n)
            col[ok] = arr[m[ok]]
            cross[f"{other.lower()}_ret_{k}"] = col

    m1_at = np.searchsorted(np.asarray(m1.time, np.int64),
                            tsec + bars.bar_seconds, "left")
    half_m1 = (m1.spread_pips() * pip * spread_mult) / 2.0
    sp_base = bars.spread_pips() * pip * spread_mult

    # ── the sampling grid ───────────────────────────────────────────────────
    hours_ok = (an.hour >= TRADING_HOURS[0]) & (an.hour < TRADING_HOURS[1])
    grid = np.flatnonzero(hours_ok)[::every]
    grid = grid[(grid > 5000) & (grid < n - 20)]
    _p(f"  {len(grid):,} sample bars x 2 directions ...", flush=True)

    rows = []
    for i in grid:
        i = int(i)
        a = float(atr[i])
        if not np.isfinite(a) or a <= 0:
            continue
        # minutes to the London gold fixes — the one gold-specific clock feature
        mins = int(an.hour[i]) * 60 + int((tsec[i] % 3600) // 60)
        base_row = {
            "hour": int(an.hour[i]), "dow": int(sess.dow[i]), "dom": int(dom[i]),
            "mins_to_am_fix": abs(mins - 630), "mins_to_pm_fix": abs(mins - 900),
            "minutes_into_session": int(sess.minutes_into[i]),
            "atr_pct": float(sess.atr_pct[i]),
            "range_pct": float(sess.range_pct[i]),
            "expansion": float(sess.expansion[i]),
            "spread_pct": float(sess.spread_pct[i]),
            "asia_range_atr": float(an.asia_range_atr[i])
            if np.isfinite(an.asia_range_atr[i]) else -1.0,
            "from_d1_open": float(sess.from_d1_open[i]),
            "from_h4_open": float(sess.from_h4_open[i]),
            "h4_dir": int(bi.h4_dir[i]), "h1_dir": int(bi.h1_dir[i]),
            "h4_pullback": float(BI.pullback_depth(bi.h4_pos[i:i + 1],
                                                   bi.h4_dir[i:i + 1])[0]),
        }
        base_row.update({k: float(v[i]) for k, v in rets.items()})
        base_row.update({k: float(v[i]) for k, v in cross.items()})
        base_row.update(fl.row(i))
        base_row.update(vw.row(i))
        base_row.update(m5.row(i, prefix="m5_"))
        base_row.update(MT.row(st, i))

        p1 = int(m1_at[i])
        p2 = min(p1 + hold_min, len(m1))
        if p2 - p1 < 2:
            continue
        risk = stop_atr * a
        for buy in (True, False):
            d = "buy" if buy else "sell"
            entry = close[i] + (sp_base[i] / 2 if buy else -sp_base[i] / 2)
            out = simulate(
                Plan(entry=entry,
                     stop=entry - risk if buy else entry + risk,
                     direction=d, risk=risk,
                     target=entry + target_r * risk if buy else entry - target_r * risk,
                     ladder=NO_TRAIL),
                m1.high[p1:p2], m1.low[p1:p2], m1.close[p1:p2], half_m1[p1:p2])
            r = _sign_row(base_row, buy)
            r.update({"bar": i, "year": int(year[i]),
                      "day": int(tsec[i] // 86400),
                      "ts": str(np.datetime64(int(tsec[i]), "s")),
                      "direction": d, "risk_pips": risk / pip, **out})
            r["pips"] = r["r"] * r["risk_pips"]
            rows.append(r)

    _p(f"    {len(rows):,} candidates, "
       f"{len(rows[0]) if rows else 0} columns\n", flush=True)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--every", type=int, default=6,
                   help="sample every Nth base bar (6 = every 30 min)")
    p.add_argument("--stop-atr", type=float, default=STOP_ATR)
    p.add_argument("--target-r", type=float, default=TARGET_R)
    p.add_argument("--hold", type=int, default=HOLD_MIN)
    p.add_argument("--cache", default=CACHE)
    a = p.parse_args(argv)

    rows = build(a.symbol, every=a.every, stop_atr=a.stop_atr,
                 target_r=a.target_r, hold_min=a.hold)
    if not rows:
        raise SystemExit("no candidates")
    r = np.array([x["r"] for x in rows])
    print(f"  universe   n={len(r):,}  win={(r > 0.05).mean():.1%}  "
          f"avg={r.mean():+.4f}R   (matched control {CONTROL_R:+.3f}R)")
    with open(a.cache, "w") as f:
        json.dump(rows, f)
    print(f"  cached -> {a.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
