"""The backtest engine. One year in seconds, not 150 minutes.

The old harness drove the live production stack against a replay connector, which
was faithful and honest and ran at ~12 bars/sec: 150 minutes per year, 2.5 hours
of wall clock for fourteen years across fourteen parallel workers. At that price a
hypothesis costs an afternoon, so hypotheses stopped being tested — four
consecutive tuning waves shipped on reasoning rather than measurement, and the
14-year run that finally checked them found every one had been fitted to noise.

**Speed is a correctness feature.** That is the whole argument for this rewrite.

The inversion that buys it: the old engine stepped bars and re-derived the level
set at each one (75.9ms/bar in `discover_levels`). This one computes every level
once with a life span, asks each level where it fired across that life, and then
makes a single pass over the resulting events. Cost goes from O(bars x discovery)
to O(sum of level lifetimes) plus O(events x trade length).

## Fidelity

Fill conventions are inherited from the old harness, which got them right, and
every one is pessimistic:

  * entry pays the full spread, charged per bar from the feed's own spread column
  * when one M1 bar spans both stop and target, the STOP is taken — intra-minute
    ordering is unknowable from OHLC
  * a trail rung armed by a bar takes effect from the NEXT bar
  * MFE/MAE come from bar high/low, not close
  * exits pay the spread as well

Known divergences from live, each of which makes the result LESS optimistic or is
flagged where it cannot:

  * no tick stream, so nothing resembling order flow contributes. In the old tree
    that degraded to 'n/a' — the same thing it did live whenever the tick buffer
    was cold.
  * slippage beyond the spread is not modelled. On a stop run this is optimistic,
    and it is the one place the result flatters itself.
  * fills are resolved on M1, not ticks. A 1-minute bar that spans the stop is
    assumed to fill at the stop.

## Costs

Spread comes from the broker's own per-bar column, never a flat number. The old
harness charged a flat 2.9p until this was fixed, against a measured median of
1.80p that rises to 4.40p at rollover — and the operator's funded account quotes
3-12p. `spread_mult` scales the measured shape; 1.67 maps the measured 1.80-6.40p
range onto 3.0-10.7p and reproduces the funded quote.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

import numpy as np

from core import sweep as SW
from core.context import Context
from core.discover import build_points, level_tracks, load_series
from core.exits import Ladder, build_plan, simulate
from core.store import Bars

# Maps the measured spread shape onto the funded account's 3-12p quote.
FUNDED_SPREAD_MULT = 1.67


@dataclass
class Config:
    symbol: str = "XAUUSD"
    base: str = "M5"
    start: str | None = None
    end: str | None = None

    # costs
    spread_mult: float = FUNDED_SPREAD_MULT
    spread_flat_pips: float | None = None    # override the feed column entirely

    # discovery
    recluster_every: int = 12                # base bars between re-clusterings
    reach_mult: float = 6.0
    weights: dict | None = None              # from research/reaction.py

    # trigger
    allow_reclaim: bool = True
    cooldown_bars: int = 12                  # per level, suppress repeat fires

    # risk / exits
    ladder: Ladder = field(default_factory=Ladder)
    beyond_atr: float = 0.25
    min_stop_atr: float = 0.75
    max_stop_atr: float = 2.50
    target_r: float | None = None
    max_hold_bars: int = 24 * 60             # M1 bars; 24h, matching the old EXPIRE_HOURS

    # portfolio
    one_at_a_time: bool = True
    max_trades_per_day: int | None = None

    # filters — every one of these is a hypothesis to be measured, and all are
    # OFF by default. The old tree stacked six individually-defensible gates
    # multiplicatively and cut its trade rate to 7% of population for no edge.
    min_members: int = 0
    max_magnet_share: float | None = None
    min_depth_atr: float | None = None
    hours_utc: tuple | None = None


@dataclass
class Result:
    trades: list
    bars: int
    seconds: float
    n_levels: int
    n_events: int
    rejected: dict
    cfg: Config

    @property
    def r(self) -> np.ndarray:
        return np.array([t["r"] for t in self.trades]) if self.trades else np.empty(0)


def run(cfg: Config, series: dict[str, Bars] | None = None,
        prebuilt: dict | None = None) -> Result:
    t0 = _time.perf_counter()
    series = series or load_series(cfg.symbol)
    if cfg.base not in series or "M1" not in series:
        raise SystemExit(f"{cfg.symbol}: need at least M1 and {cfg.base} in the store")

    b = series[cfg.base]
    m1 = series["M1"]
    pip = b.pip
    ctx = (prebuilt or {}).get("ctx") or Context(series, base=cfg.base)

    # ── window ──────────────────────────────────────────────────────────────
    lo, hi = 0, len(b)
    if cfg.start:
        lo = max(lo, int(np.searchsorted(b.time, _epoch(cfg.start), "left")))
    if cfg.end:
        hi = min(hi, int(np.searchsorted(b.time, _epoch(cfg.end), "left")))
    # fills need M1 coverage; refuse to score a window the M1 series does not span
    m1_lo, m1_hi = int(m1.time[0]), int(m1.time[-1])
    lo = max(lo, int(np.searchsorted(b.time, m1_lo, "left")))
    hi = min(hi, int(np.searchsorted(b.time, m1_hi, "right")))
    if hi - lo < 100:
        raise SystemExit("window too small or not covered by M1 data")

    # ── levels ──────────────────────────────────────────────────────────────
    pts = (prebuilt or {}).get("points")
    if pts is None:
        pts = build_points(series, ctx, cfg.base)
    tracks = level_tracks(pts, ctx, every=cfg.recluster_every,
                          reach_mult=cfg.reach_mult, weights=cfg.weights)

    # ── triggers ────────────────────────────────────────────────────────────
    sweep_min = ctx.threshold("sweep_min")
    keep = (tracks.dead > lo) & (tracks.born < hi)
    ev = SW.find_events(
        b, tracks.price[keep],
        sweep_min=sweep_min,
        born=np.maximum(tracks.born[keep], lo),
        dead=np.minimum(tracks.dead[keep], hi),
        allow_reclaim=cfg.allow_reclaim,
        cooldown=cfg.cooldown_bars,
    )
    tk = {f: getattr(tracks, f)[keep] for f in tracks._F}

    # ── filters ─────────────────────────────────────────────────────────────
    atr_m15 = ctx.threshold("stop") / 1.50          # back out raw ATR(M15)
    rejected: dict = {}
    mask = np.ones(len(ev), bool)

    def drop(name, m):
        nonlocal mask
        n = int((mask & ~m).sum())
        if n:
            rejected[name] = rejected.get(name, 0) + n
        mask = mask & m

    if len(ev):
        if cfg.min_members > 0:
            drop("members", tk["n_members"][ev.level] >= cfg.min_members)
        if cfg.max_magnet_share is not None:
            tot = tk["n_decision"][ev.level] + tk["n_magnet"][ev.level]
            share = np.divide(tk["n_magnet"][ev.level], tot,
                              out=np.zeros(len(ev)), where=tot > 0)
            drop("magnet_share", share <= cfg.max_magnet_share)
        if cfg.min_depth_atr is not None:
            drop("depth", ev.depth >= cfg.min_depth_atr * atr_m15[ev.bar])
        if cfg.hours_utc is not None:
            drop("hour", np.isin(b.hour_utc()[ev.bar], list(cfg.hours_utc)))

    idx = np.flatnonzero(mask)

    # ── spread ──────────────────────────────────────────────────────────────
    if cfg.spread_flat_pips is not None:
        sp_base = np.full(len(b), cfg.spread_flat_pips * pip)
        sp_m1 = np.full(len(m1), cfg.spread_flat_pips * pip)
    else:
        sp_base = b.spread_pips() * pip * cfg.spread_mult
        sp_m1 = m1.spread_pips() * pip * cfg.spread_mult
        # A dead spread column would silently make every trade free, which is the
        # most flattering bug available. mult=0 is the deliberate frictionless
        # baseline and is allowed; a zero column at any other multiplier is a
        # broken feed and must not be traded through.
        if float(np.nanmax(sp_base)) <= 0 and cfg.spread_mult != 0:
            raise SystemExit("spread column is empty — pass spread_flat_pips "
                             "explicitly rather than trading for free")
    half_m1 = sp_m1 / 2.0

    # M1 index at each base bar's close: where the fill walk starts
    m1_at = np.searchsorted(np.asarray(m1.time, np.int64),
                            np.asarray(b.time, np.int64) + b.bar_seconds, "left")

    # ── portfolio pass ──────────────────────────────────────────────────────
    trades: list = []
    busy_until = -1
    day_of = np.asarray(b.time, np.int64) // 86400
    per_day: dict = {}

    for k in idx:
        bar = int(ev.bar[k])
        if cfg.one_at_a_time and bar < busy_until:
            rejected["in_trade"] = rejected.get("in_trade", 0) + 1
            continue
        if cfg.max_trades_per_day:
            d = int(day_of[bar])
            if per_day.get(d, 0) >= cfg.max_trades_per_day:
                rejected["day_cap"] = rejected.get("day_cap", 0) + 1
                continue

        direction = str(ev.direction[k])
        # entry pays the full spread: buy at ask, sell at bid
        raw_close = float(ev.entry[k])
        entry = raw_close + (sp_base[bar] / 2 if direction == "buy"
                             else -sp_base[bar] / 2)
        a = float(atr_m15[bar])
        if not np.isfinite(a) or a <= 0:
            continue

        plan = build_plan(entry, direction, float(ev.extreme[k]), a,
                          ladder=cfg.ladder, beyond_atr=cfg.beyond_atr,
                          min_stop_atr=cfg.min_stop_atr,
                          max_stop_atr=cfg.max_stop_atr, target_r=cfg.target_r)

        s = int(m1_at[bar])
        e = min(s + cfg.max_hold_bars, len(m1))
        if e - s < 2:
            continue
        out = simulate(plan, m1.high[s:e], m1.low[s:e], m1.close[s:e], half_m1[s:e])

        lvl = int(ev.level[k])
        trades.append({
            "bar": bar, "time": b.t_str(bar),
            "hour": int(b.hour_utc()[bar]),
            "direction": direction, "kind": str(ev.kind[k]),
            "level": float(ev.price[k]), "entry": entry,
            "stop": plan.stop, "risk_pips": plan.risk / pip,
            "depth_pips": float(ev.depth[k]) / pip,
            "depth_atr": float(ev.depth[k]) / a,
            "spread_pips": float(sp_base[bar]) / pip,
            "n_members": int(tk["n_members"][lvl]),
            "n_decision": int(tk["n_decision"][lvl]),
            "n_magnet": int(tk["n_magnet"][lvl]),
            "level_width_pips": float(tk["width"][lvl]) / pip,
            "level_age_bars": bar - int(tk["born"][lvl]),
            **out,
            "result_pips": out["r"] * plan.risk / pip,
        })
        busy_until = bar + int(np.ceil(out["bars"] / (b.bar_seconds / 60)))
        if cfg.max_trades_per_day:
            per_day[int(day_of[bar])] = per_day.get(int(day_of[bar]), 0) + 1

    return Result(trades=trades, bars=hi - lo,
                  seconds=_time.perf_counter() - t0,
                  n_levels=int(keep.sum()), n_events=len(ev),
                  rejected=rejected, cfg=cfg)


def _epoch(when) -> int:
    return int(np.datetime64(when, "s").astype("int64"))
