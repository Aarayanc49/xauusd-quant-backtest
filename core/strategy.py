"""The swing configuration, frozen.

This file exists because the validated result was, until now, an argument in a
transcript rather than an object in the tree. `research/combine.py` defines a
"surviving stack" of three filters; the measured +0.726R config has four, and its
geometry (8R target, 24h hold) lived as function defaults in two different
modules. Reproducing the headline number required reassembling it by hand, which
is exactly how a result rots.

One spec, one place. Research reads it, `live/` reads it, and the numbers below
are what it was measured at so a later run that disagrees is visibly a
regression rather than a new opinion.

    XAUUSD, 2012-01 .. 2026-08 (14.6y), funded spread (1.67x the feed column)
    1,282 trades (88/yr), win 29.3%, +0.726R/trade
    halves +0.750 / +0.702, 15 of 15 years positive

Nothing here is fitted in this file. Every predicate was measured ALONE on the
28,493-candidate feature study before being stacked, against a random-entry
control on identical geometry (-0.181R at a 2R target). See
`research/features.py`, `research/combine.py`, `research/biastest.py`.

## Provenance of each filter

  range_pct >= 0.75    day-range percentile. The high-vol third carried the
                       edge; the low third was negative. Volatility is what
                       makes an 8R target reachable inside 24h.
  spread_pct < 0.5     cost, measured. Spread costs -0.181R per trade at this
                       stop and scales as 1/stop, so the cheap half of the
                       distribution is not a preference, it is the difference
                       between a live edge and a dead one.
  session              london / ny / overlap. Asia and the dead hours were flat
                       to negative and pay full spread to be there.
  h4_pullback >= 0.4   depth of the retrace into the current H4 leg. The
                       0.2-0.4 bucket was the WORST in the study (-0.227R) and
                       0.6-0.8 the best (+0.163R): shallow entries are chasing
                       an extreme. This is the operator's own read of how they
                       trade, tested rather than assumed.

## What is deliberately NOT here

Full H4+H1 agreement (`htf_stack == 2`) measures better per trade (+1.412R) but
leaves 465 trades over 14.6 years — 33/yr, against an operator who trades
600-1,200. It is kept as `SWING_STRICT` for comparison, not as the default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

# ── geometry ────────────────────────────────────────────────────────────────
# Fractions of ATR and multiples of R only. No fixed-pip constant reaches the
# decision path — design rule 2, and the reason the old tree's 60p stop made
# spread 2.7x more expensive in a high-range year than it needed to be.

STOP_ATR = 1.5          # median ~44 pips on gold
TARGET_R = 8.0          # 85% of the edge lives past 2R; a 2R target scored +0.074
HOLD_HOURS = 24
TRAIL = None            # measured last in every cost regime; the exit is "don't"

SESSIONS_OK = ("london", "ny", "overlap")

_OPS = (">=", "<=", "==", ">", "<")


def _parse_filter(name: str) -> tuple:
    """Pull (field, operator, threshold) back out of a filter's own name.

    The filter names below are written as the comparison they perform
    (`"range_pct >= 0.75"`), so the name is already the machine-readable form
    and does not need a second table kept in sync with it. A name that is not a
    numeric comparison — `"session london/ny/ovlp"` — returns its field with no
    threshold, and `Spec.verdict` then reports pass/fail without a margin.
    """
    for op in _OPS:
        if op in name:
            left, _, right = name.partition(op)
            try:
                return left.strip(), op, float(right.strip())
            except ValueError:
                return left.strip(), op, None
    return name.split()[0] if name.split() else None, None, None


@dataclass(frozen=True)
class Spec:
    """A named, frozen strategy definition.

    `filters` maps a human-readable name to a predicate over one feature row
    (the dicts `research.features.build` emits). Order is the order they were
    measured and stacked in, so survival rates print in a meaningful sequence.
    """
    name: str
    filters: Mapping[str, Callable[[Mapping], bool]]
    stop_atr: float = STOP_ATR
    target_r: float = TARGET_R
    hold_hours: int = HOLD_HOURS
    trail: object = TRAIL
    note: str = ""
    # what this spec scored when it was frozen, for regression comparison
    measured: Mapping[str, float] = field(default_factory=dict)

    def passes(self, row: Mapping) -> bool:
        return all(fn(row) for fn in self.filters.values())

    def verdict(self, row: Mapping) -> list[dict]:
        """Per-filter pass/fail for ONE candidate, with how far off it was.

        `passes` answers "did it trade". This answers "why not", which is the
        question a live run actually raises: a journal line saying `filters`
        forces the reader to recompute four comparisons by hand, and nobody
        does that at 3am. Each entry carries

            name      the filter as written below
            key       the row field it reads
            value     what that field actually was
            passed    bool
            margin    signed distance to the threshold in the field's own
                      units, positive when passing. None when the filter is
                      not a numeric comparison.

        `margin` is the reason this exists rather than a bare bool: a candidate
        that missed by 0.004 and one that missed by 0.6 are different events,
        and only the first one means "the config is nearly firing here".
        """
        out = []
        for name, fn in self.filters.items():
            try:
                passed = bool(fn(row))
            except (KeyError, TypeError):
                out.append({"name": name, "key": None, "value": None,
                            "passed": False, "margin": None,
                            "error": "field missing"})
                continue
            key, op, thr = _parse_filter(name)
            value = row.get(key) if key else None
            margin = None
            if thr is not None and isinstance(value, (int, float)):
                # signed so positive always means "clear of the threshold",
                # whichever direction the comparison runs
                if op in (">=", ">"):
                    margin = float(value) - thr
                elif op in ("<=", "<"):
                    margin = thr - float(value)
                elif op == "==":
                    margin = 0.0 if passed else float(value) - thr
            out.append({"name": name, "key": key, "value": value,
                        "passed": passed, "margin": margin})
        return out

    def blocked_by(self, row: Mapping) -> list[str]:
        """Names of the filters this candidate failed. Empty means it traded."""
        return [v["name"] for v in self.verdict(row) if not v["passed"]]

    def select(self, rows: Sequence[Mapping]) -> list:
        return [x for x in rows if self.passes(x)]

    def survival(self, rows: Sequence[Mapping]) -> list:
        """(name, n_before, n_after) as each filter is added, in order.

        Printed on every run because the multiplicative collapse is what killed
        v10 — six individually defensible filters multiplied to 7% of population
        for a break-even edge. If it happens again it should be visible while it
        happens, not afterwards.
        """
        out, cur = [], list(rows)
        for name, fn in self.filters.items():
            before = len(cur)
            cur = [x for x in cur if fn(x)]
            out.append((name, before, len(cur)))
        return out


def _with_h4(x) -> bool:
    return x["h4_dir"] == (1 if x["direction"] == "buy" else -1)


# ── the validated config ────────────────────────────────────────────────────

SWING = Spec(
    name="swing",
    filters={
        "range_pct >= 0.75": lambda x: x["range_pct"] >= 0.75,
        "spread_pct < 0.5": lambda x: x["spread_pct"] < 0.5,
        "session london/ny/ovlp": lambda x: x["session"] in SESSIONS_OK,
        "h4_pullback >= 0.4": lambda x: x["h4_pullback"] >= 0.4,
    },
    note="XAUUSD 2012-2026, the config that measured 15/15 years positive",
    measured=dict(n=1282, win=0.293, avg_r=0.726, h1=0.750, h2=0.702,
                  years_positive=15, years=15),
)

# Same entries, both higher timeframes required to agree. Better per trade,
# too few trades to be the account's only strategy.
SWING_STRICT = Spec(
    name="swing-strict",
    filters={
        **SWING.filters,
        "htf_stack == 2": lambda x: x["htf_stack"] == 2,
    },
    note="H4+H1 both agreeing; +1.412R but only 33 trades/yr",
    measured=dict(n=465, avg_r=1.412, years_positive=14, years=15),
)

# The three cost/volatility filters with no bias term — what `research/combine.py`
# calls "the surviving stack". Kept so the contribution of the H4 pullback term
# is measurable rather than assumed.
SWING_NOBIAS = Spec(
    name="swing-nobias",
    filters={k: v for k, v in SWING.filters.items()
             if k != "h4_pullback >= 0.4"},
    note="cost/vol filters only, no H4 pullback term",
)

SPECS = {s.name: s for s in (SWING, SWING_STRICT, SWING_NOBIAS)}
