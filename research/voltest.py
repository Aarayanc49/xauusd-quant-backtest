"""Is the volatility edge SIGNAL, or just cheaper cost?

The feature study found volatility to be the dominant separator:

    atr percentile >= 0.9      n=3,287   lift +0.176R
    day range pct  >= 0.75     n=6,231   lift +0.152R
    expansion      >= 1.4      n=  129   lift +0.387R

But the stop is 1.5 ATR, so in a high-ATR bar the stop is more PIPS, and a fixed
pip spread costs less in R. Some or all of that "edge" could be nothing but a
cheaper toll — which would still be worth having, but it is not a better setup and
it must not be described as one.

The test: run the identical candidates with cost ON and cost OFF. If the
volatility lift survives at zero cost, it is signal. If it collapses, it was the
toll all along.

    python -m research.voltest
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import features as F  # noqa: E402
from research.control import run as control_run  # noqa: E402


def band(rows, key, lo, hi):
    return [x for x in rows if lo <= x[key] < hi]


def main():
    print("building candidates WITH cost ...", flush=True)
    paid = F.build(spread_mult=F.FUNDED_SPREAD_MULT)
    print("building candidates WITHOUT cost ...", flush=True)
    free = F.build(spread_mult=0.0)

    # same candidate set, keyed by entry bar, so the comparison is exact
    fm = {x["bar"]: x for x in free}
    pairs = [(p, fm[p["bar"]]) for p in paid if p["bar"] in fm]
    print(f"\n  matched {len(pairs):,} candidates\n")

    print("=" * 96)
    print("  IS THE VOLATILITY EDGE SIGNAL OR COST?")
    print("=" * 96)
    print(f"  {'bucket':<24}{'n':>7}{'paid R':>10}{'free R':>10}"
          f"{'cost':>9}{'signal lift':>13}")

    specs = [
        ("atr_pct", [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]),
        ("range_pct", [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]),
        ("expansion", [(0.0, 0.9), (0.9, 1.0), (1.0, 1.15), (1.15, 1.4), (1.4, 99)]),
    ]
    # signal lift is measured against the ZERO-COST random baseline, so the
    # comparison is like-for-like: strategy-without-cost vs random-without-cost.
    free_base = np.mean([x["r"] for x in
                         control_run(n=4000, target_r=2.0, stop_atr=1.5,
                                     spread_mult=0.0)])
    paid_base = F.RANDOM_BASELINE
    print(f"\n  random baseline: paid {paid_base:+.3f}R   free {free_base:+.3f}R\n")

    for key, edges in specs:
        print(f"  --- {key} ---")
        for lo, hi in edges:
            pp = [p for p, f in pairs if lo <= p[key] < hi]
            ff = [f for p, f in pairs if lo <= p[key] < hi]
            if len(pp) < 60:
                continue
            rp = np.mean([x["r"] for x in pp])
            rf = np.mean([x["r"] for x in ff])
            print(f"  {f'{lo}-{hi}':<24}{len(pp):>7}{rp:>+10.3f}{rf:>+10.3f}"
                  f"{rp - rf:>+9.3f}{rf - free_base:>+13.3f}")

    print("\n  'cost' is what the spread took. 'signal lift' is the edge that")
    print("  remains once cost is removed from BOTH sides — that is the only")
    print("  column that says the setup is better rather than cheaper.")


if __name__ == "__main__":
    raise SystemExit(main())
