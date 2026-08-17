"""Out-of-sample test. The check every filter in this tree has so far skipped.

Everything in `core/strategy.SWING` was chosen while looking at all 14.6 years.
Per-year and two-halves checks were run on every claim, and both are real
evidence, but neither is out-of-sample: a filter picked because it works in both
halves is still a filter picked with the second half visible. The old tree died
of exactly this — four tuning waves, every one defensible in-sample, none of
them tested on data that had not already voted.

Two different questions get asked here, and they are not the same strength of
evidence:

  TEST A  Freeze SWING, run it on 2020-2026 alone.
          Weak. The config was chosen with these years visible, so this shows
          STABILITY, not independence. It can still falsify: if the config only
          worked because of 2012-2019, this is where that shows up.

  TEST B  Re-run the SELECTION PROCEDURE on 2012-2019 only, freeze whatever it
          picks, apply it untouched to 2020-2026.
          Strong. Nothing about the second period touches the choice. This is
          the honest walk-forward, and it tests the method rather than the
          config: if the procedure re-derives something close to SWING from half
          the data and that survives, the config is a property of the market
          rather than of the search.

Test B needs the selection rule written down mechanically, because "I looked at
the tables and picked the good ones" cannot be replayed on a subsample. The rule
below is the one that was actually followed last session, made explicit:
measure every candidate ALONE, keep the ones that beat the population with
enough trades to be real, then stack greedily while the average improves and the
population survives.

## Two limits of Test B, stated rather than buried

**The search space is not clean.** `candidates()` grids over the feature axes
the full-sample study surfaced. The THRESHOLDS are re-chosen on the in-sample
period only, which is the part that matters most, but the choice of which axes
to offer at all carries knowledge of the whole series. Fixing that properly
means re-deriving the axes from the in-sample period too; short of that, read
Test B as strong evidence rather than proof.

**Compare lift, not level.** The unfiltered population scores -0.212R in-sample
and +0.049R out-of-sample: the second period is structurally kinder to this
entry style, so a config can look stable while its actual edge shrinks. Every
verdict here is quoted against the CONTEMPORANEOUS unfiltered baseline for that
reason. Measured that way SWING is +1.088R in-sample and +0.602R out — real in
both, and about 45% smaller than the headline suggests.

    python -m research.walkforward
    python -m research.walkforward --split 2019 --symbol XAUUSD
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy import SWING, SWING_NOBIAS, SWING_STRICT  # noqa: E402
from research import report as RP  # noqa: E402

FEAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "_feat8.json")

# Selection rule, fixed BEFORE looking at any subsample. These are the floors
# that were in force last session (combine.py refuses to call anything with
# n<300 tradeable); writing them here is what makes the procedure replayable.
MIN_N_ALONE = 300       # a filter measured alone must leave this many trades
MIN_N_STACK = 200       # the final stack must leave this many
MIN_LIFT = 0.05         # R improvement required to accept another filter
MAX_FILTERS = 5


# ── candidate filter grid ───────────────────────────────────────────────────
# The axes the feature study measured. Values are a coarse grid, not a fine
# sweep: a fine sweep over 8 years would fit the grid to the in-sample period
# and defeat the point of the exercise.

def _sess(names):
    return lambda x: x["session"] in names


def candidates():
    C = {}
    for t in (0.25, 0.50, 0.66, 0.75, 0.90):
        C[f"range_pct >= {t}"] = (lambda t: lambda x: x["range_pct"] >= t)(t)
        C[f"atr_pct >= {t}"] = (lambda t: lambda x: x["atr_pct"] >= t)(t)
    for t in (0.25, 0.50, 0.75):
        C[f"spread_pct < {t}"] = (lambda t: lambda x: x["spread_pct"] < t)(t)
    for t in (1.00, 1.15, 1.40):
        C[f"expansion >= {t}"] = (lambda t: lambda x: x["expansion"] >= t)(t)
    C["session london/ny/ovlp"] = _sess(("london", "ny", "overlap"))
    C["session london/ovlp"] = _sess(("london", "overlap"))
    C["session ny/ovlp"] = _sess(("ny", "overlap"))
    for t in (0.2, 0.4, 0.6):
        C[f"h4_pullback >= {t}"] = (lambda t: lambda x: x["h4_pullback"] >= t)(t)
    C["h4_pullback 0.4-0.8"] = lambda x: 0.4 <= x["h4_pullback"] < 0.8
    C["htf_stack >= 1"] = lambda x: x["htf_stack"] >= 1
    C["htf_stack == 2"] = lambda x: x["htf_stack"] == 2
    C["with H4"] = lambda x: x["h4_dir"] == (1 if x["direction"] == "buy" else -1)
    C["with H1"] = lambda x: x["h1_dir"] == (1 if x["direction"] == "buy" else -1)
    for lo, hi in ((2, 7), (4, 12), (2, 12)):
        C[f"leg_atr {lo}-{hi}"] = (lambda lo, hi: lambda x: lo <= x["leg_atr"] < hi)(lo, hi)
    for t in (200, 800, 3000):
        C[f"level_age < {t}"] = (lambda t: lambda x: x["level_age"] < t)(t)
    return C


# ── stats ───────────────────────────────────────────────────────────────────

def stat(rows):
    if not rows:
        return None
    r = np.array([x["r"] for x in rows])
    t = sorted(rows, key=lambda x: x["bar"])
    m = len(t) // 2
    h1 = float(np.mean([x["r"] for x in t[:m]])) if m else 0.0
    h2 = float(np.mean([x["r"] for x in t[m:]])) if m else 0.0
    ys = defaultdict(list)
    for x in rows:
        ys[x["year"]].append(x["r"])
    pos = sum(1 for v in ys.values() if np.mean(v) > 0)
    dd, final, _ = RP.drawdown(rows)
    return dict(n=len(r), win=float((r > 0.05).mean()), avg=float(r.mean()),
                sum=float(r.sum()), h1=h1, h2=h2, pos=pos, nyears=len(ys),
                dd=dd, final=final, retdd=(final / abs(dd)) if dd else 0.0)


def line(label, s, width=34):
    if s is None or not s["n"]:
        return f"    {label:<{width}} n=    0"
    ok = "OK " if s["h1"] > 0 and s["h2"] > 0 else "   "
    return (f"    {label:<{width}} n={s['n']:>5}  win={s['win']:>5.1%}  "
            f"avg={s['avg']:>+6.3f}R  sum={s['sum']:>+7.1f}R  "
            f"halves {s['h1']:>+6.3f}/{s['h2']:>+6.3f} {ok} "
            f"yrs {s['pos']}/{s['nyears']}  ret/DD {s['retdd']:>5.2f}")


def per_year(rows, indent="      "):
    ys = defaultdict(list)
    for x in rows:
        ys[x["year"]].append(x)
    out, eq = [], 0.0
    for y in sorted(ys):
        s = stat(ys[y])
        eq += s["sum"]
        bar = ("+" * int(min(40, max(0, s["avg"]) * 30))
               or "-" * int(min(40, max(0, -s["avg"]) * 30)))
        out.append(f"{indent}{y}  n={s['n']:>4}  win={s['win']:>5.1%}  "
                   f"avg={s['avg']:>+6.3f}R  sum={s['sum']:>+7.1f}R  "
                   f"cum={eq:>+7.1f}R  {bar}")
    return "\n".join(out)


# ── the selection procedure, replayable on any subsample ────────────────────

def select_filters(rows, verbose=True):
    """Run the documented selection rule on `rows` and return the chosen stack.

    Deliberately mechanical. Every decision it makes is a comparison against a
    threshold declared at the top of this file, so running it on 2012-2019 uses
    exactly the same judgement as running it on everything — which is the only
    way the out-of-sample result means anything.
    """
    C = candidates()
    base = stat(rows)
    if verbose:
        print(f"    population           {line('', base).strip()}")

    # Stage 1 — measure every candidate ALONE.
    alone = []
    for name, fn in C.items():
        sub = [x for x in rows if fn(x)]
        s = stat(sub)
        if s and s["n"] >= MIN_N_ALONE and s["avg"] > base["avg"]:
            alone.append((name, fn, s))
    alone.sort(key=lambda t: -t[2]["avg"])
    if verbose:
        print(f"\n    {len(alone)} of {len(C)} candidates beat the population "
              f"alone with n >= {MIN_N_ALONE}:")
        for name, _, s in alone[:12]:
            print(line(name, s))
        if len(alone) > 12:
            print(f"      ... and {len(alone) - 12} more")

    # Stage 2 — greedy stack while it keeps paying and the population survives.
    chosen, cur = {}, list(rows)
    if verbose:
        print("\n    stacking (greedy, best marginal lift first):")
    while len(chosen) < MAX_FILTERS:
        best = None
        for name, fn, _ in alone:
            if name in chosen:
                continue
            sub = [x for x in cur if fn(x)]
            s = stat(sub)
            if not s or s["n"] < MIN_N_STACK:
                continue
            if s["h1"] <= 0 or s["h2"] <= 0:
                continue
            lift = s["avg"] - stat(cur)["avg"]
            if lift >= MIN_LIFT and (best is None or lift > best[3]):
                best = (name, fn, s, lift)
        if best is None:
            break
        name, fn, s, lift = best
        chosen[name] = fn
        cur = [x for x in cur if fn(x)]
        if verbose:
            print(line(f"+ {name}  (lift {lift:+.3f})", s))
    if verbose and not chosen:
        print("      nothing cleared the thresholds")
    return chosen


# ── main ────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feat", default=FEAT)
    p.add_argument("--split", type=int, default=2019,
                   help="last IN-SAMPLE year; everything after is out-of-sample")
    a = p.parse_args(argv)

    rows = json.load(open(a.feat))
    IS = [x for x in rows if x["year"] <= a.split]
    OOS = [x for x in rows if x["year"] > a.split]
    if not IS or not OOS:
        raise SystemExit("split leaves one side empty")

    yrs_is = len({x["year"] for x in IS})
    yrs_oos = len({x["year"] for x in OOS})
    print("=" * 118)
    print(f"  WALK-FORWARD — split at {a.split}")
    print("=" * 118)
    print(f"  in-sample      {min(x['year'] for x in IS)}-{a.split}   "
          f"{len(IS):,} candidates over {yrs_is} years")
    print(f"  out-of-sample  {a.split+1}-{max(x['year'] for x in OOS)}   "
          f"{len(OOS):,} candidates over {yrs_oos} years")

    # ── TEST A ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("  TEST A — the frozen config on each period")
    print("  (weak: SWING was chosen with both periods visible. Stability, not "
          "independence.)")
    print("=" * 118)
    for spec in (SWING, SWING_NOBIAS, SWING_STRICT):
        print(f"\n  {spec.name}   {spec.note}")
        print(line("full 2012-2026", stat(spec.select(rows))))
        print(line(f"in-sample <= {a.split}", stat(spec.select(IS))))
        print(line(f"OUT-OF-SAMPLE > {a.split}", stat(spec.select(OOS))))

    print("\n  unfiltered population, for reference:")
    print(line("all candidates, IS", stat(IS)))
    print(line("all candidates, OOS", stat(OOS)))

    print(f"\n  SWING per year, out-of-sample ({a.split+1} onward):")
    print(per_year(SWING.select(OOS)))

    print("\n  filter survival on the OOS period:")
    for name, before, after in SWING.survival(OOS):
        print(f"      {name:<30} {before:>6,} -> {after:>6,}   "
              f"survival {after/before if before else 0:>5.0%}")

    # ── TEST B ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print(f"  TEST B — re-select on {min(x['year'] for x in IS)}-{a.split} ONLY, "
          f"then apply untouched to {a.split+1}+")
    print("=" * 118)
    print(f"\n  running the selection procedure on the in-sample period:\n")
    chosen = select_filters(IS)

    print("\n" + "-" * 118)
    print("  what the procedure picked, knowing nothing after "
          f"{a.split}:")
    if not chosen:
        print("      (empty stack)")
    for name in chosen:
        mark = "  <- also in SWING" if any(
            name.split()[0] == k.split()[0] for k in SWING.filters) else ""
        print(f"      {name}{mark}")
    print(f"\n  SWING, for comparison:")
    for name in SWING.filters:
        print(f"      {name}")

    def apply(rs):
        return [x for x in rs if all(fn(x) for fn in chosen.values())]

    print("\n" + "-" * 118)
    print("  THE RESULT THAT COUNTS")
    print("-" * 118)
    print(line("IS-picked stack, in-sample", stat(apply(IS))))
    print(line("IS-picked stack, OUT-OF-SAMPLE", stat(apply(OOS))))
    print(line("SWING, OUT-OF-SAMPLE", stat(SWING.select(OOS))))
    print(line("no filter, OUT-OF-SAMPLE", stat(OOS)))

    # Level is misleading across periods — the unfiltered population is not the
    # same difficulty in both. Lift over the CONTEMPORANEOUS baseline is the
    # comparable number, and it is materially smaller out of sample.
    b_is, b_oos = stat(IS)["avg"], stat(OOS)["avg"]
    print(f"\n  lift over the unfiltered population OF THAT PERIOD "
          f"(baseline IS {b_is:+.3f}R, OOS {b_oos:+.3f}R):")
    for lbl, s_is, s_oos_ in (
            ("SWING", stat(SWING.select(IS)), stat(SWING.select(OOS))),
            ("IS-picked stack", stat(apply(IS)), stat(apply(OOS))),
            ("swing-strict", stat(SWING_STRICT.select(IS)),
             stat(SWING_STRICT.select(OOS)))):
        if s_is and s_oos_:
            li, lo = s_is["avg"] - b_is, s_oos_["avg"] - b_oos
            print(f"      {lbl:<20} in-sample {li:>+6.3f}R   "
                  f"out-of-sample {lo:>+6.3f}R   "
                  f"retained {lo/li if li else 0:>5.0%}")

    oos_sel = apply(OOS)
    if oos_sel:
        print(f"\n  IS-picked stack per year, out-of-sample:")
        print(per_year(oos_sel))

    # ── verdict ─────────────────────────────────────────────────────────────
    s_oos = stat(oos_sel)
    s_all = stat(OOS)
    print("\n" + "=" * 118)
    print("  READ")
    print("=" * 118)
    if s_oos is None or s_oos["n"] < MIN_N_STACK:
        print("  The in-sample procedure did not leave a tradeable population "
              "out of sample. Inconclusive rather than negative.")
    else:
        edge = s_oos["avg"] - s_all["avg"]
        print(f"  A stack chosen without seeing {a.split+1}+ scored "
              f"{s_oos['avg']:+.3f}R there against {s_all['avg']:+.3f}R "
              f"unfiltered — a lift of {edge:+.3f}R over "
              f"{s_oos['n']:,} trades, {s_oos['pos']}/{s_oos['nyears']} years "
              f"positive, halves {s_oos['h1']:+.3f}/{s_oos['h2']:+.3f}.")
        if s_oos["avg"] <= 0:
            print("  NEGATIVE out of sample. The config does not survive.")
        elif s_oos["h1"] <= 0 or s_oos["h2"] <= 0:
            print("  Positive overall but one half is negative out of sample — "
                  "not enough to build on.")
        else:
            print("  Positive out of sample in both halves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
