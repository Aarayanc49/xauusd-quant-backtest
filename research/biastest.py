"""Does the H4/H1 bias cascade actually pay? Measured on the cached feature set.

The operator's claim: gold sets a bias on H4/H1, and the intraday entry only works
when it agrees with that bias. Neither the old tree nor this rebuild ever tested
it — `regime_engine` computed it and ScanTrader ignored it; my `Rules
.require_trend_agreement` defaults to False.

Reads data/_feat8.json (28,493 candidates, 1.5 ATR stop, 8R target, 24h hold)
so it runs in a second instead of rebuilding the whole book.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "_feat8.json")


def stat(rows):
    if not rows:
        return None
    r = np.array([x["r"] for x in rows])
    t = sorted(rows, key=lambda x: x["bar"])
    m = len(t) // 2
    h1 = np.mean([x["r"] for x in t[:m]]) if m else 0.0
    h2 = np.mean([x["r"] for x in t[m:]]) if m else 0.0
    ys = defaultdict(list)
    for x in rows:
        ys[x["year"]].append(x["r"])
    pos = sum(1 for v in ys.values() if np.mean(v) > 0)
    return dict(n=len(r), win=(r > 0.05).mean(), avg=r.mean(), sum=r.sum(),
                h1=h1, h2=h2, yrs=f"{pos}/{len(ys)}")


def show(rows, label, keyfn, minn=100):
    b = defaultdict(list)
    for x in rows:
        b[keyfn(x)].append(x)
    out = [(k, stat(v)) for k, v in b.items() if len(v) >= minn]
    if not out:
        return
    out.sort(key=lambda kv: -kv[1]["avg"])
    print(f"\n  {label}")
    for k, s in out:
        ok = "OK " if s["h1"] > 0 and s["h2"] > 0 else "   "
        print(f"    {str(k)[:30]:<30} n={s['n']:>5} win={s['win']:>5.1%} "
              f"avg={s['avg']:>+6.3f}R sum={s['sum']:>+8.1f}R "
              f"halves {s['h1']:>+6.3f}/{s['h2']:>+6.3f} {ok} yrs {s['yrs']}")


def surviving(x):
    return (x["range_pct"] >= 0.75 and x["spread_pct"] < 0.5
            and x["session"] in ("london", "ny", "overlap"))


def main():
    rows = json.load(open(PATH))
    print("=" * 112)
    print(f"  H4/H1 BIAS CASCADE — {len(rows):,} candidates, 8R target, 24h hold")
    print("=" * 112)
    s = stat(rows)
    print(f"  ALL                            n={s['n']:>5} win={s['win']:>5.1%} "
          f"avg={s['avg']:>+6.3f}R  yrs {s['yrs']}")

    show(rows, "by HTF stack (how many of H4/H1 agree)", lambda x: x["htf_stack"])
    show(rows, "by H4 vs trade", lambda x: (
        "with H4" if x["h4_dir"] == (1 if x["direction"] == "buy" else -1)
        else "against H4" if x["h4_dir"] != 0 else "H4 flat"))
    show(rows, "by H1 vs trade", lambda x: (
        "with H1" if x["h1_dir"] == (1 if x["direction"] == "buy" else -1)
        else "against H1" if x["h1_dir"] != 0 else "H1 flat"))
    show(rows, "by H4 pullback depth", lambda x: (
        "<0.2 at extreme" if x["h4_pullback"] < 0.2 else
        "0.2-0.4" if x["h4_pullback"] < 0.4 else
        "0.4-0.6 golden" if x["h4_pullback"] < 0.6 else
        "0.6-0.8 deep" if x["h4_pullback"] < 0.8 else ">0.8 broken"))

    # the decisive test: bias stacked ON TOP of the surviving cost/vol filters
    print("\n" + "=" * 112)
    print("  BIAS ON TOP OF THE SURVIVING STACK")
    print("=" * 112)
    base = [x for x in rows if surviving(x)]
    b = stat(base)
    print(f"    {'surviving stack (no bias)':<30} n={b['n']:>5} win={b['win']:>5.1%} "
          f"avg={b['avg']:>+6.3f}R sum={b['sum']:>+8.1f}R "
          f"halves {b['h1']:>+6.3f}/{b['h2']:>+6.3f}     yrs {b['yrs']}")
    for lbl, fn in (
        ("+ H4 agrees", lambda x: x["h4_dir"] == (1 if x["direction"] == "buy" else -1)),
        ("+ H1 agrees", lambda x: x["h1_dir"] == (1 if x["direction"] == "buy" else -1)),
        ("+ BOTH agree (stack 2)", lambda x: x["htf_stack"] == 2),
        ("+ NEITHER agrees (stack 0)", lambda x: x["htf_stack"] == 0),
        # The standalone table says 0.2-0.4 is the WORST pullback bucket and
        # 0.6-0.8 the best, so a .2-.6 band mixes the good half with the worst
        # one. Test the band the evidence actually points at.
        ("+ H1 agrees AND pullback >=.4",
         lambda x: x["h1_dir"] == (1 if x["direction"] == "buy" else -1)
         and x["h4_pullback"] >= 0.4),
        ("+ H1 agrees AND pullback .4-.8",
         lambda x: x["h1_dir"] == (1 if x["direction"] == "buy" else -1)
         and 0.4 <= x["h4_pullback"] < 0.8),
        ("+ both agree AND pullback >=.4",
         lambda x: x["htf_stack"] == 2 and x["h4_pullback"] >= 0.4),
        ("+ pullback >=.4 only",
         lambda x: x["h4_pullback"] >= 0.4),
        ("+ drop shallow (.2-.4) only",
         lambda x: not (0.2 <= x["h4_pullback"] < 0.4)),
    ):
        sub = [x for x in base if fn(x)]
        s2 = stat(sub)
        if s2 is None:
            continue
        ok = "OK " if s2["h1"] > 0 and s2["h2"] > 0 else "   "
        print(f"    {lbl:<30} n={s2['n']:>5} win={s2['win']:>5.1%} "
              f"avg={s2['avg']:>+6.3f}R sum={s2['sum']:>+8.1f}R "
              f"halves {s2['h1']:>+6.3f}/{s2['h2']:>+6.3f} {ok} yrs {s2['yrs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
