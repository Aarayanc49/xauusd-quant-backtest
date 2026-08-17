"""Price every stage of the setup engine, independently, per year.

Two questions, both of which the old project got wrong by never asking:

  1. **What does each stage actually buy?** The old tree stacked six
     individually-defensible gates and cut its trade rate to 7% of population
     for a break-even edge — and never measured any of them alone. One of them
     (the room gate) turned out to reject 1 trade in 14 years.

  2. **Does it hold in every regime?** Gold ran a 124-pip average daily range in
     2017 and 1,322 in 2026, at $1,200 and $4,700. A single blended number over
     that span is not a result, it is an average of different instruments. Every
     table here is per-year, and every claim is checked on a two-halves split.

    python -m research.stages              # ablation + per-year
    python -m research.stages --family breakout
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import candles as C  # noqa: E402
from core import setups as S  # noqa: E402
from core import sweep as SW  # noqa: E402
from core.cluster import cluster_points  # noqa: E402
from core.context import Context  # noqa: E402
from core.discover import build_points, load_series  # noqa: E402
from core.exits import NO_TRAIL, Ladder, Plan, simulate  # noqa: E402
from core.structure import find_breaks, trend_state  # noqa: E402
from research.engine import FUNDED_SPREAD_MULT  # noqa: E402


class Book:
    """Everything derived once, reused by every ablation."""

    def __init__(self, symbol="XAUUSD", base="M5"):
        self.series = load_series(symbol)
        if base not in self.series or "M1" not in self.series:
            raise SystemExit(f"{symbol}: need M1 and {base} in the store")
        self.base = base
        self.bars = self.series[base]
        self.m1 = self.series["M1"]
        self.pip = self.bars.pip
        self.ctx = Context(self.series, base=base)
        self.atr = self.ctx.threshold("stop") / 1.50          # raw ATR(M15)
        self.sweep_min = self.ctx.threshold("sweep_min")
        self.cap = self.ctx.threshold("cluster_width_cap")
        self.tol = self.ctx.threshold("cluster_tolerance")
        self.prox = self.ctx.threshold("proximity")

        print("  building levels ...", flush=True)
        self.points = build_points(self.series, self.ctx, base)
        print(f"    {len(self.points):,} points", flush=True)

        print("  finding structure ...", flush=True)
        self.breaks = find_breaks(self.bars, k=3, atr=self.atr)
        self.trend = trend_state(self.bars, self.breaks)
        print(f"    {len(self.breaks):,} breaks "
              f"({int((self.breaks.kind == 'bos').sum()):,} BOS / "
              f"{int((self.breaks.kind == 'choch').sum()):,} CHoCH)", flush=True)

        print("  reading candles ...", flush=True)
        self.signals = C.read(self.bars, self.atr)

        # cluster cache on an hourly cadence — levels do not change per bar
        self._cl_cache: dict = {}
        self._every = 12

        print("  finding sweeps ...", flush=True)
        self.sweeps = self._sweep_events()
        print(f"    {len(self.sweeps):,} sweep events", flush=True)

        # M1 alignment + costs
        self.m1_at = np.searchsorted(
            np.asarray(self.m1.time, np.int64),
            np.asarray(self.bars.time, np.int64) + self.bars.bar_seconds, "left")
        self.half_m1 = (self.m1.spread_pips() * self.pip * FUNDED_SPREAD_MULT) / 2.0
        self.spread_base = self.bars.spread_pips() * self.pip * FUNDED_SPREAD_MULT
        self.year = (np.asarray(self.bars.time, np.int64)
                     .astype("datetime64[s]").astype("datetime64[Y]")
                     .astype(int) + 1970)

    def clusters_at(self, i: int):
        key = i // self._every
        c = self._cl_cache.get(key)
        if c is None:
            j = key * self._every
            act = self.points.near(j, float(self.bars.close[j]),
                                   float(self.prox[j]) * 6)
            c = cluster_points(act, float(self.cap[j]),
                               dedup=float(self.tol[j]) * 0.25) if len(act) else []
            self._cl_cache[key] = c
        return c

    def _sweep_events(self):
        """Sweeps of clustered levels, using the same track machinery as engine."""
        from core.discover import level_tracks
        tk = level_tracks(self.points, self.ctx, every=self._every)
        return SW.find_events(self.bars, tk.price, self.sweep_min,
                              tk.born, tk.dead, cooldown=12)

    # ── run one rule set ────────────────────────────────────────────────────

    def run(self, rules: S.Rules, ladder: Ladder, one_at_a_time=True) -> list:
        st = S.build(self.bars, rules, self.breaks, self.trend,
                     self.clusters_at, self.atr, self.sweep_min,
                     sweep_events=self.sweeps, signals=self.signals)
        trades, busy = [], -1
        for s in st:
            if one_at_a_time and s.bar < busy:
                continue
            buy = s.direction == "buy"
            entry = s.entry + (self.spread_base[s.bar] / 2 if buy
                               else -self.spread_base[s.bar] / 2)
            risk = abs(entry - s.stop)
            if risk <= 0:
                continue
            plan = Plan(entry=entry, stop=s.stop, direction=s.direction,
                        risk=risk, target=s.target, ladder=ladder)
            a = int(self.m1_at[s.bar])
            b = min(a + 24 * 60, len(self.m1))
            if b - a < 2:
                continue
            out = simulate(plan, self.m1.high[a:b], self.m1.low[a:b],
                           self.m1.close[a:b], self.half_m1[a:b])
            trades.append({
                "bar": s.bar, "year": int(self.year[s.bar]),
                "family": s.family, "kind": s.event_kind,
                "direction": s.direction, "candle": s.candle,
                "rr_planned": s.rr, "risk_pips": risk / self.pip,
                "n_sources": s.n_sources, "n_categories": s.n_categories,
                **out,
            })
            busy = s.bar + int(np.ceil(out["bars"] / (self.bars.bar_seconds / 60)))
        return trades


# ── reporting ───────────────────────────────────────────────────────────────

def arith(trades):
    if not trades:
        return None
    r = np.array([t["r"] for t in trades])
    w, l = r[r > 0.05], r[r < -0.05]
    wr, lr = len(w) / len(r), len(l) / len(r)
    aw = w.mean() if len(w) else 0.0
    al = abs(l.mean()) if len(l) else 0.0
    payoff = aw / al if al > 1e-9 else float("inf")
    need = lr / wr if wr > 1e-9 else float("inf")
    return dict(n=len(r), win=wr, avg=r.mean(), sum=r.sum(),
                payoff=payoff, need=need, edge=payoff - need)


def line(label, a, width=30):
    if a is None:
        return f"  {label:<{width}} n=    0"
    return (f"  {label:<{width}} n={a['n']:>5} win={a['win']:>5.1%} "
            f"avg={a['avg']:>+6.3f}R sum={a['sum']:>+8.1f}R "
            f"payoff={a['payoff']:>5.2f} edge={a['edge']:>+6.2f}")


def halves(trades):
    t = sorted(trades, key=lambda x: x["bar"])
    m = len(t) // 2
    return arith(t[:m]), arith(t[m:])


def per_year(trades) -> str:
    ys = sorted({t["year"] for t in trades})
    out = ["\n  per year:", f"    {'yr':<6}{'n':>6}{'win':>8}{'avg R':>9}"
                            f"{'sum R':>9}{'payoff':>8}{'edge':>8}"]
    pos = 0
    for y in ys:
        a = arith([t for t in trades if t["year"] == y])
        if a is None:
            continue
        pos += a["avg"] > 0
        out.append(f"    {y:<6}{a['n']:>6}{a['win']:>7.1%}{a['avg']:>+9.3f}"
                   f"{a['sum']:>+9.1f}{a['payoff']:>8.2f}{a['edge']:>+8.2f}")
    out.append(f"    -> {pos} of {len(ys)} years positive")
    return "\n".join(out)


# ── the ablation ────────────────────────────────────────────────────────────

def ablate(book: Book, base_rules: S.Rules, ladder: Ladder, family: str) -> None:
    variants = [
        ("FULL (all stages on)", {}),
        ("no level requirement", dict(require_level=False)),
        ("no pullback (enter at break)", dict(require_pullback=False)),
        ("no candle read", dict(require_candle=False)),
        ("no R:R gate", dict(min_rr=0.0)),
        ("R:R gate >= 3", dict(min_rr=3.0)),
        ("categories >= 2", dict(min_categories=2)),
        ("sources >= 3", dict(min_sources=3)),
        ("trend agreement (BOS only)", dict(require_trend_agreement=True)),
        ("fixed stop 1.5 ATR", dict(min_stop_atr=1.5, max_stop_atr=1.5)),
    ]
    print("\n" + "=" * 104)
    print(f"  STAGE ABLATION — {family.upper()}   "
          f"(each row = FULL with ONE stage changed)")
    print("=" * 104)
    for label, override in variants:
        r = replace(base_rules, **override)
        t = book.run(r, ladder)
        a = arith(t)
        h1, h2 = halves(t) if len(t) > 20 else (None, None)
        tag = ""
        if h1 and h2:
            same = (h1["avg"] > 0) == (h2["avg"] > 0)
            tag = f"  halves {h1['avg']:+.2f}/{h2['avg']:+.2f} {'OK' if same else 'SPLIT-FAIL'}"
        print(line(label, a) + tag)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--family", default="both",
                   choices=["both", "breakout", "fade"])
    p.add_argument("--min-rr", type=float, default=2.0)
    p.add_argument("--ladder", default="default", choices=["default", "none"],
                   help="'none' isolates ENTRY quality: pure stop-vs-target, no "
                        "trail truncating winners before the target")
    p.add_argument("--stop-atr", type=float, default=None,
                   help="fixed stop in ATRs; overrides --stop. Spread cost in R "
                        "falls as 1/stop, so this is the main cost lever.")
    p.add_argument("--stop", default="structural",
                   choices=["structural", "atr"],
                   help="'atr' = fixed 1.5 ATR instead of behind the candle wick")
    p.add_argument("--no-ablate", action="store_true")
    a = p.parse_args(argv)

    book = Book(a.symbol)
    ladder = Ladder() if a.ladder == "default" else NO_TRAIL

    fams = ([("breakout", True, False), ("fade", False, True)]
            if a.family == "both" else
            [(a.family, a.family == "breakout", a.family == "fade")])

    if a.stop_atr is not None:
        stop_kw = dict(min_stop_atr=a.stop_atr, max_stop_atr=a.stop_atr)
    else:
        stop_kw = (dict(min_stop_atr=1.5, max_stop_atr=1.5)
                   if a.stop == "atr" else {})
    for name, bo, fd in fams:
        rules = S.Rules(allow_breakout=bo, allow_fade=fd, min_rr=a.min_rr,
                        **stop_kw)
        trades = book.run(rules, ladder)
        print("\n" + "=" * 104)
        print(f"  {name.upper()}   min_rr={a.min_rr}   ladder={ladder.name}   "
              f"stop={a.stop}")
        print("=" * 104)
        print(line("ALL", arith(trades)))
        if trades:
            h1, h2 = halves(trades)
            print(line("  first half", h1))
            print(line("  second half", h2))
            print(per_year(trades))
            for key, lbl in (("kind", "event kind"), ("candle", "candle"),
                             ("direction", "direction")):
                print(f"\n  by {lbl}:")
                for k in sorted({t[key] for t in trades}):
                    sub = [t for t in trades if t[key] == k]
                    if len(sub) >= 10:
                        print(line(f"    {k}", arith(sub), width=28))
        if not a.no_ablate:
            ablate(book, rules, ladder, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
