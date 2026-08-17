"""Entry point for the packaged cockpit executable.

PyInstaller needs a module at the project root that owns the import graph.
`live/cockpit.py` is importable as `python -m live.cockpit` from a source
checkout; this wrapper is what `build_cockpit.py` freezes into `Cockpit.exe`.

The import block below looks redundant and is not. Most of the pipeline is
pulled in lazily inside functions — `live.explain` imports `core.discover` only
when a level rebuild actually runs, which keeps the terminal tools fast to
start. PyInstaller resolves the import graph statically, so a module that is
only ever imported inside a function body is invisible to it, and the frozen
app dies at the first refresh with ModuleNotFoundError. Naming them here is the
fix, and it is cheaper than teaching the analyser about every deferred import.

They are collected into `BUNDLED` rather than left as bare imports so that both
the reason for their presence and their use are visible in the code itself —
a linter suppression comment would hide the first and fake the second.
"""
from __future__ import annotations

import os
import sys

if getattr(sys, "frozen", False):
    # Inside the bundle, resources live next to the executable rather than
    # beside this file.
    sys.path.insert(0, os.path.dirname(sys.executable))
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import anchors, bias, candles, cluster, context, contracts
from core import discover, exits, flow, levels, majors, mtf, patterns
from core import session, shape, store, strategy, structure, sweep, vwap
from live import broker, explain, signals, trader
from research import features

from live.cockpit import main

#: Modules the cockpit reaches only through deferred imports. Listing them
#: keeps them in the frozen bundle; the tuple is never read at runtime.
BUNDLED = (
    anchors, bias, candles, cluster, context, contracts, discover, exits,
    flow, levels, majors, mtf, patterns, session, shape, store, strategy,
    structure, sweep, vwap, features, broker, explain, signals, trader,
)


if __name__ == "__main__":
    raise SystemExit(main())
