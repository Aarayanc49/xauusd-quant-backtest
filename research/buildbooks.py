"""Generate and cache the candidate book for each symbol.

`research/features.py` derives levels, structure, majorness, patterns, session
state and the bias cascade before it can emit a single candidate. That is the
expensive part of the whole pipeline and it does not depend on the exit, so it
is done once per symbol and cached; `research/daytrade.py` and
`research/portfolio.py` then re-simulate those same entries under whatever
geometry is being tested.

## The cached hold is deliberately tiny

The book is cached with a **1-hour** hold, which looks wrong and is not. What is
being cached is the FEATURE set — the entry bar, the direction, and the
volatility/session/bias columns the filters read. None of those depend on how
long the trade is subsequently held. The cached `r` / `mae_r` / `bars` are
overwritten by `daytrade.run`, which re-simulates every entry under whatever
geometry is being tested.

Caching at 24h instead means `simulate` walks 1,440 M1 bars per candidate to
produce a number that is immediately discarded. On USDJPY's 71,744 candidates
that is the difference between minutes and most of an hour, per symbol.

    python -m research.buildbooks
    python -m research.buildbooks --symbols EURUSD GBPUSD
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import features as F  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")
SYMBOLS = ["XAUUSD", "USDJPY", "EURUSD", "GBPUSD", "USTEC", "US500"]


def path_for(symbol):
    return os.path.join(DATA, f"_book_{symbol}.json")


def build(symbol, force=False):
    p = path_for(symbol)
    if os.path.exists(p) and not force:
        print(f"  {symbol:<8} cached")
        return p
    t0 = time.time()
    print(f"  {symbol:<8} building ...", flush=True)
    # hold_hours=1: features are hold-independent and the outcome is
    # re-simulated downstream. See the module docstring.
    rows = F.build(symbol=symbol, target_r=8.0, hold_hours=1, quiet=True)
    for r in rows:
        r["symbol"] = symbol
    with open(p, "w") as f:
        json.dump(rows, f)
    print(f"  {symbol:<8} {len(rows):>7,} candidates in {time.time()-t0:.0f}s "
          f"-> {os.path.basename(p)}", flush=True)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for s in a.symbols:
        try:
            build(s, a.force)
        except Exception as e:
            print(f"  {s:<8} FAILED: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
