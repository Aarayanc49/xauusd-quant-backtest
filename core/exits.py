"""Exits — in R, and armed outside the noise. This is the strategy.

Fourteen years of measurement said no computed ENTRY feature predicts outcome:
every one of cluster confidence, scan_pa score, confluence count, touch count,
zone width, decision sources and entry displacement came back with |rho| < 0.08
against trade result. The same fourteen years said the exit decides everything:

    trail_step 0   n=760 (33.1%)   -758.3R   avg -1.00R   <- the entire loss
    trail_step 1   n=939 (40.9%)   +148.0R   avg +0.16R   <- the scratch band
    trail_step 2   n=363 (15.8%)   +240.5R   avg +0.66R
    trail_step 3   n=160 ( 7.0%)   +165.2R   avg +1.03R
    trail_step 4   n= 61 ( 2.7%)    +91.1R   avg +1.49R

So the exit is not a detail bolted onto an entry model. It IS the model.

## Why the old ladder cut winners and losers with the same blade

    TRAIL_LADDER = [(20,0), (40,18), (60,38), (85,60), (115,88), (150,120)]

Rung 1 moved the stop to breakeven once a trade was +20 pips. On a 60p stop that
is 0.33R. Against the measured heat a real trade has to survive:

    winners  median MAE 0.26R   p75 0.53R   p90 0.78R
    losers   median MAE 0.95R   p75 1.08R   p90 1.19R

**0.33R sits inside the winners' own noise band.** It did cut losses — that is why
avg loss was -0.53R instead of -1.0R — but it scratched 41% of all trades at
+0.16R. The choke: 290 trades reached 0.93R on average and banked -0.06R, handing
back 287R, with 288 of the 290 sitting at rung 1. In the live journal the same
thing measured as 50 fills that reached >=45p and banked -7.7p, giving back 3,823
pips against a journal that netted 2,092.

And because the ladder was in fixed PIPS while the stop was volatility-scaled, the
arming point drifted with the regime — 0.67R in 2017 (124p range, 30p stop) and
0.22R in 2026 (1,322p range, 90p stop). That drift *was* the "regime signal" the
study found; it was self-inflicted.

## What this module does differently

  * Everything is in R. A rung means the same thing in every year.
  * Nothing arms below `MIN_ARM_R`, which sits above the winners' p75 MAE.
  * No fixed take-profit by default. 86% of trades exited via stop or trail and
    only 14% via TP; the P&L lives in the 1.0R+ MFE bands, where a hard target
    truncates exactly the trades that pay. The old tree found this the hard way
    and raised its cap 150 -> 200 because 31 winners had closed *exactly at the
    cap* with an average MFE of 165p.

Every ladder here is a hypothesis. `research/engine.py` exists to argue with them,
and nothing ships without surviving a two-halves split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Winners' p75 MAE is 0.53R and p90 is 0.78R. An arming point below that is inside
# the heat a genuine winner still has to take, so it converts winners to scratches.
MIN_ARM_R = 0.75


@dataclass(frozen=True)
class Ladder:
    """(mfe_R, lock_R) rungs. `lock` is where the stop goes, in R from entry —
    0.0 is breakeven, negative still risks something, positive banks profit.

    Monotonic in both columns by construction; `validate` refuses anything else,
    because a non-monotonic ladder silently moves a stop backwards and the bug is
    invisible in aggregate P&L.
    """
    rungs: tuple = ((0.80, 0.00), (1.20, 0.50), (1.80, 1.00),
                    (2.50, 1.70), (3.50, 2.60))
    name: str = "default"

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if not self.rungs:
            return
        mfe = [r[0] for r in self.rungs]
        lock = [r[1] for r in self.rungs]
        if any(b <= a for a, b in zip(mfe, mfe[1:])):
            raise ValueError(f"ladder {self.name}: mfe thresholds must increase")
        if any(b < a for a, b in zip(lock, lock[1:])):
            raise ValueError(f"ladder {self.name}: locks must not decrease")
        if any(l >= m for m, l in self.rungs):
            raise ValueError(f"ladder {self.name}: a lock at or above its own "
                             f"trigger would fill instantly")
        if mfe[0] < MIN_ARM_R:
            raise ValueError(
                f"ladder {self.name}: first rung arms at {mfe[0]:.2f}R, inside the "
                f"winners' MAE band (p75 0.53R). This is the mistake that cost the "
                f"old tree 287R. Pass unsafe=True to Ladder.unchecked if you are "
                f"deliberately measuring it.")

    @staticmethod
    def unchecked(rungs, name="unchecked") -> "Ladder":
        """Build a ladder without the MIN_ARM_R guard — for reproducing the old
        behaviour in a comparison run, and for nothing else."""
        obj = object.__new__(Ladder)
        object.__setattr__(obj, "rungs", tuple(rungs))
        object.__setattr__(obj, "name", name)
        return obj

    @property
    def arrays(self) -> tuple:
        if not self.rungs:
            return np.empty(0), np.empty(0)
        a = np.asarray(self.rungs, dtype=np.float64)
        return a[:, 0], a[:, 1]


# The old ladder, converted to R on its own 60p stop, for like-for-like runs.
LEGACY_LADDER = Ladder.unchecked(
    tuple((m / 60.0, l / 60.0) for m, l in
          ((20, 0), (40, 18), (60, 38), (85, 60), (115, 88), (150, 120))),
    name="legacy_v10")

NO_TRAIL = Ladder.unchecked((), name="none")


def lock_for(mfe_r: float, ladder: Ladder) -> tuple:
    """Highest rung reached for a given MFE in R -> (step, lock_R).
    step 0 means the ladder has not armed."""
    step, lock = 0, 0.0
    for i, (thr, lk) in enumerate(ladder.rungs, start=1):
        if mfe_r >= thr:
            step, lock = i, lk
        else:
            break
    return step, lock


@dataclass
class Plan:
    """The full exit specification for one trade, fixed at entry."""
    entry: float
    stop: float
    direction: str
    risk: float                     # |entry - stop| in price units; 1R
    target: float | None = None     # None -> let the trail decide
    ladder: Ladder = field(default_factory=Ladder)

    @property
    def r_pips(self) -> float:
        return self.risk

    def r_of(self, price: float) -> float:
        """Where `price` sits in R from entry, signed by trade direction."""
        d = (price - self.entry) if self.direction == "buy" else (self.entry - price)
        return d / self.risk if self.risk else 0.0

    def price_at_r(self, r: float) -> float:
        return (self.entry + r * self.risk if self.direction == "buy"
                else self.entry - r * self.risk)


def build_plan(entry: float, direction: str, sweep_extreme: float,
               atr: float, ladder: Ladder | None = None,
               beyond_atr: float = 0.25, min_stop_atr: float = 0.75,
               max_stop_atr: float = 2.50, target_r: float | None = None) -> Plan:
    """Stop goes beyond the sweep extreme — the price the stop run reached.

    That is the structural stop the strategy idea implies: if price returns past
    the wick that just failed, the read was wrong. `beyond_atr` pads it so the
    stop is not sitting exactly where the last liquidity grab ended.

    Floored and capped, both for measured reasons:

      * FLOOR. The 14-year study tested a tighter stop directly — v10 with SL40
        returned +358p against v10's +1,101p, and its drawdown per unit earned was
        $2.82 versus $0.59. A trade that dips and then runs is saved by the ladder
        and killed by a tight stop, so a shallow sweep must not produce a hair
        trigger.
      * CAP. A deep sweep would otherwise buy an enormous stop, and since risk is
        sized off the stop, one wide trade would eat the daily budget.
    """
    lo = min_stop_atr * atr
    hi = max_stop_atr * atr
    raw = abs(entry - sweep_extreme) + beyond_atr * atr
    risk = float(np.clip(raw, lo, hi))
    stop = entry - risk if direction == "buy" else entry + risk
    target = None
    if target_r:
        target = entry + target_r * risk if direction == "buy" else entry - target_r * risk
    return Plan(entry=entry, stop=stop, direction=direction, risk=risk,
                target=target, ladder=ladder or Ladder())


def simulate(plan: Plan, high, low, close, spread_half,
             max_bars: int | None = None) -> dict:
    """Walk a trade forward over bars, applying the ladder. Pure and vectorized
    where it can be; the rung loop is sequential because a rung armed by a bar
    must not be reachable within that same bar.

    Fill conventions, all pessimistic and stated so results can be discounted —
    carried over from the old harness, which got these right:

      * when one bar spans BOTH stop and target, the STOP is taken; intra-bar
        ordering is unknowable from OHLC
      * a rung armed by a bar's excursion takes effect from the NEXT bar, so a
        rung can never be armed and hit inside the same bar
      * MFE/MAE are measured from bar high/low, not close
      * entry and exit both pay half the spread
    """
    high = np.asarray(high, np.float64)
    low = np.asarray(low, np.float64)
    close = np.asarray(close, np.float64)
    half = np.asarray(spread_half, np.float64)
    n = len(high)
    if max_bars is not None:
        n = min(n, max_bars)

    buy = plan.direction == "buy"
    stop, risk, entry = plan.stop, plan.risk, plan.entry
    step, mfe, mae = 0, 0.0, 0.0
    thr, lk = plan.ladder.arrays

    for i in range(n):
        h, l = high[i], low[i]
        fav = (h - entry) if buy else (entry - l)
        adv = (entry - l) if buy else (h - entry)
        mfe = max(mfe, fav / risk)
        mae = max(mae, adv / risk)

        hit_stop = (l <= stop) if buy else (h >= stop)
        hit_tp = plan.target is not None and (
            (h >= plan.target) if buy else (l <= plan.target))

        if hit_stop:
            px = stop - half[i] if buy else stop + half[i]
            return _close(plan, px, i, "stop", step, mfe, mae)
        if hit_tp:
            px = plan.target - half[i] if buy else plan.target + half[i]
            return _close(plan, px, i, "target", step, mfe, mae)

        # ladder armed by THIS bar takes effect from the next one
        if thr.size:
            reached = np.searchsorted(thr, mfe, side="right")
            if reached > step:
                new = plan.price_at_r(float(lk[reached - 1]))
                if (new > stop) if buy else (new < stop):
                    stop = new
                step = int(reached)

    i = n - 1
    px = close[i] - half[i] if buy else close[i] + half[i]
    return _close(plan, px, i, "timeout", step, mfe, mae)


# Every key `simulate` returns. Studies merge this dict straight into a feature
# row, so anything building a model MUST exclude all of it — these are outcomes,
# and several are numeric and innocuous-looking.
#
# `bars` is the dangerous one. It is trade duration: a trade that reaches target
# quickly is short, one that times out runs the full window. It encodes the
# result almost perfectly, and it leaked into the first run of research/model.py
# as the strongest feature in the study by a factor of ten, producing a fake
# out-of-sample 56% win rate. Import this set rather than restating it — a
# hand-written blocklist fails silently the moment a field is added here.
OUTCOME_KEYS = frozenset({
    "exit_price", "bars", "exit_reason", "trail_step", "r", "mfe_r", "mae_r",
    "outcome",
})


def _close(plan: Plan, price: float, bar: int, reason: str,
           step: int, mfe: float, mae: float) -> dict:
    r = plan.r_of(price)
    return {
        "exit_price": float(price), "bars": int(bar) + 1, "exit_reason": reason,
        "trail_step": int(step), "r": float(r),
        "mfe_r": float(mfe), "mae_r": float(mae),
        "outcome": "win" if r > 0.05 else "loss" if r < -0.05 else "scratch",
    }
