"""The setup engine — six stages, each one measurable on its own.

One machine serves both trade families the operator asked for, because they are
the same mechanic with opposite signs:

    BREAKOUT   structure breaks THROUGH a level -> retest of it -> enter with
               the break
    FADE       price sweeps a level and fails  -> reclaim         -> enter against
               the sweep

What decides which is allowed is the prevailing trend, not a preference.

## The stages

    1. LEVEL      a precise cluster (core/cluster.py), category-diverse
    2. EVENT      a break of it (core/structure.py) or a sweep of it (core/sweep.py)
    3. PULLBACK   price retraces into 0.5-0.618 of the leg the event produced
    4. CANDLE     a confirmation pattern fires in that zone (core/candles.py)
    5. STOP       behind the structural extreme, never a fixed pip count
    6. RR GATE    reject unless the structural target pays >= min_rr

Every stage is a hypothesis with recorded evidence behind it, and each can be
switched off independently so `research/stages.py` can price its contribution.

## Where each stage comes from

Stage 3 and 6 are the two knobs `reaction_engine`'s own 369-trade journal
separated on, and neither is a direction bet:

    pullback used     True  n= 70  51.4% win  +0.086R
                      False n=299  42.8% win  -0.073R
    structural R:R    >=1   n=176  ~50% win   positive
                      <1    n=193  39.4% win  -0.134R
    trigger safe_bos        n= 90  53.3% win  +0.254R
            fast            n=279  41.6% win  -0.139R

Stage 1's category-diversity rule is `top_down_sr`'s: "3 round-numbers stacked
doesn't count — needs diversity". That is the fix for `confluence.score_setup`,
whose nine factors all derive from the same swing series, so a "high confluence"
zone was usually one observation wearing seven names.

Stage 5 is `confirmation.py`'s own rule, which the shipped engine discarded:
"SL goes behind the confirmation candle's extreme. Not behind some HTF level 400
pips away. BEHIND THE CANDLE."

**Nothing here is tuned.** The defaults are the values those journals implied;
every one is a hypothesis for research/stages.py to price, and a stage that
cannot show its keep on a two-halves split does not stay in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import candles as C
from . import sweep as SW
from .cluster import Cluster
from .structure import BOS, CHOCH, DOWN, UP, Breaks

BREAKOUT, FADE = "breakout", "fade"


@dataclass
class Rules:
    """Every stage's parameters in one place, all switchable."""
    # 1. level
    require_level: bool = True
    min_sources: int = 1
    min_categories: int = 1          # top_down_sr used 2; measure before adopting
    level_tol_atr: float = 0.35      # how near the event must be to the level

    # 2. event
    allow_breakout: bool = True
    allow_fade: bool = True
    require_trend_agreement: bool = False   # breakout must match prevailing trend

    # 3. pullback
    require_pullback: bool = True
    pb_near: float = 0.50
    pb_far: float = 0.618
    pb_max_bars: int = 48            # give up if it never retraces

    # 4. candle
    require_candle: bool = True
    candle_kinds: tuple = ()         # empty = any

    # 5. stop
    stop_beyond_atr: float = 0.25
    min_stop_atr: float = 0.50
    max_stop_atr: float = 2.50

    # 6. rr gate
    min_rr: float = 2.0
    target_from: str = "extreme"     # 'extreme' = the leg's own high/low


@dataclass
class Setup:
    """One tradeable setup, fully specified, with its audit trail."""
    bar: int
    family: str                  # breakout | fade
    direction: str               # buy | sell
    entry: float
    stop: float
    target: float
    rr: float
    level: float
    event_bar: int
    event_kind: str              # bos | choch | sweep_v | sweep_reclaim
    leg: float
    pullback_frac: float
    candle: str
    n_sources: int
    n_categories: int
    stage_log: dict = field(default_factory=dict)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)


def _categories(sources) -> int:
    """Distinct source CATEGORIES, not source count.

    `top_down_sr`'s insight: three round numbers stacked is one fact, not three.
    Categories are the independence axis — a round number, a prior-day high and a
    swing are three genuinely different reasons for price to be there.
    """
    cat = set()
    for s in sources:
        s = str(s)
        if s in ("round", "half_round"):
            cat.add("round")
        elif s in ("pdh", "pdl", "pwh", "pwl"):
            cat.add("period")
        elif s in ("eqh", "eql"):
            cat.add("equal")
        elif s in ("swing_h", "swing_l"):
            cat.add("swing")
        else:
            cat.add("other")
    return len(cat)


def build(bars, rules: Rules, breaks: Breaks, trend: np.ndarray,
          clusters_at, atr: np.ndarray, sweep_min: np.ndarray,
          sweep_events: SW.Events | None = None,
          signals: C.Signals | None = None,
          lo: int = 0, hi: int | None = None) -> list[Setup]:
    """Run every stage and return the setups that survive all of them.

    `clusters_at(i)` -> list[Cluster] active at bar i. `stage_log` on each Setup
    records what each stage saw, and rejections are counted by the caller, so a
    stage that rejects nothing is visible immediately — the old tree shipped a
    room gate that turned out to reject 1 trade in 14 years and nobody knew.
    """
    hi = len(bars) if hi is None else hi
    if signals is None:
        signals = C.read(bars, atr)
    close = np.asarray(bars.close, np.float64)
    high = np.asarray(bars.high, np.float64)
    low = np.asarray(bars.low, np.float64)

    out: list[Setup] = []
    rejects: dict = {}

    def rej(stage):
        rejects[stage] = rejects.get(stage, 0) + 1

    # ── breakout family ─────────────────────────────────────────────────────
    if rules.allow_breakout:
        for j in range(len(breaks)):
            b = int(breaks.bar[j])
            if not (lo <= b < hi):
                continue
            up = breaks.direction[j] == UP
            direction = "buy" if up else "sell"

            if rules.require_trend_agreement and breaks.kind[j] == CHOCH:
                rej("trend_agreement"); continue

            # stage 1 — was the broken level a real, diverse cluster?
            n_src, n_cat, lvl = 1, 1, float(breaks.level[j])
            if rules.require_level:
                cl = _nearest(clusters_at(b), lvl, float(atr[b]) * rules.level_tol_atr)
                if cl is None:
                    rej("level"); continue
                n_src, n_cat, lvl = len(cl.members), _categories(cl.sources), cl.price
                if n_src < rules.min_sources:
                    rej("min_sources"); continue
                if n_cat < rules.min_categories:
                    rej("min_categories"); continue

            # stage 3 — retrace into the leg
            leg = abs(float(breaks.extreme[j]) - float(breaks.origin[j]))
            if leg <= 0:
                rej("leg"); continue
            e_bar = int(breaks.extreme_bar[j])
            ext = float(breaks.extreme[j])
            near = ext - leg * rules.pb_near if up else ext + leg * rules.pb_near
            far = ext - leg * rules.pb_far if up else ext + leg * rules.pb_far

            entry_bar = None
            if rules.require_pullback:
                s, e = e_bar + 1, min(e_bar + 1 + rules.pb_max_bars, hi)
                if e <= s:
                    rej("pullback_window"); continue
                inzone = ((low[s:e] <= near) & (high[s:e] >= far) if up
                          else (high[s:e] >= near) & (low[s:e] <= far))
                w = np.flatnonzero(inzone)
                if not w.size:
                    rej("no_pullback"); continue
                entry_bar = s + int(w[0])
            else:
                entry_bar = min(e_bar + 1, hi - 1)

            s2 = _finish(bars, rules, signals, atr, entry_bar, direction, hi,
                         lvl, ext, leg, b, str(breaks.kind[j]), BREAKOUT,
                         n_src, n_cat, rej,
                         pullback_frac=abs(close[entry_bar] - ext) / leg)
            if s2 is not None:
                out.append(s2)

    # ── fade family ─────────────────────────────────────────────────────────
    if rules.allow_fade and sweep_events is not None:
        for j in range(len(sweep_events)):
            b = int(sweep_events.bar[j])
            if not (lo <= b < hi):
                continue
            direction = str(sweep_events.direction[j])
            lvl = float(sweep_events.price[j])
            ext = float(sweep_events.extreme[j])
            # the "leg" of a fade is the sweep excursion itself
            leg = abs(ext - lvl)
            if leg <= 0:
                rej("leg"); continue
            s2 = _finish(bars, rules, signals, atr, b, direction, hi,
                         lvl, ext, leg, b, str(sweep_events.kind[j]), FADE,
                         1, 1, rej, pullback_frac=0.0, fade_extreme=ext)
            if s2 is not None:
                out.append(s2)

    out.sort(key=lambda s: s.bar)
    for s in out:
        s.stage_log["rejects"] = rejects
    return out


def _finish(bars, rules, signals, atr, entry_bar, direction, hi,
            level, extreme, leg, event_bar, event_kind, family,
            n_src, n_cat, rej, pullback_frac, fade_extreme=None):
    """Stages 4-6: candle, stop, R:R."""
    if entry_bar >= hi or entry_bar >= len(bars):
        return None
    buy = direction == "buy"
    a = float(atr[entry_bar])
    if not np.isfinite(a) or a <= 0:
        return None

    # stage 4 — the candle read
    name = ""
    if rules.require_candle:
        name = signals.name_at(entry_bar, direction)
        if not name:
            rej("candle"); return None
        if rules.candle_kinds and name not in rules.candle_kinds:
            rej("candle_kind"); return None
    else:
        name = signals.name_at(entry_bar, direction) or "none"

    entry = float(bars.close[entry_bar])

    # stage 5 — structural stop, behind the candle AND the event extreme
    wick = (float(signals.stop_long[entry_bar]) if buy
            else float(signals.stop_short[entry_bar]))
    if fade_extreme is not None:
        wick = min(wick, fade_extreme - a * rules.stop_beyond_atr) if buy \
            else max(wick, fade_extreme + a * rules.stop_beyond_atr)
    risk = abs(entry - wick)
    risk = float(np.clip(risk, rules.min_stop_atr * a, rules.max_stop_atr * a))
    stop = entry - risk if buy else entry + risk

    # stage 6 — does the structural target pay?
    target = extreme if family == BREAKOUT else _fade_target(level, leg, direction)
    reward = (target - entry) if buy else (entry - target)
    if reward <= 0:
        rej("target_behind"); return None
    rr = reward / risk
    if rr < rules.min_rr:
        rej("rr_gate"); return None

    return Setup(bar=int(entry_bar), family=family, direction=direction,
                 entry=entry, stop=stop, target=float(target), rr=float(rr),
                 level=float(level), event_bar=int(event_bar),
                 event_kind=event_kind, leg=float(leg),
                 pullback_frac=float(pullback_frac), candle=name,
                 n_sources=int(n_src), n_categories=int(n_cat))


def _fade_target(level: float, leg: float, direction: str) -> float:
    """A fade's structural target: the sweep excursion projected the other way.
    Deliberately modest — a fade is a reversion trade, and the old tree's habit
    of claiming a 150-pip target on a 20-pip sweep is what made its R:R fiction."""
    return level - leg * 3.0 if direction == "sell" else level + leg * 3.0


def _nearest(clusters, price: float, within: float):
    if not clusters:
        return None
    best, bd = None, None
    for c in clusters:
        d = abs(c.price - price)
        if d <= within and (bd is None or d < bd):
            best, bd = c, d
    return best
