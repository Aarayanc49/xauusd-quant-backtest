"""Contract specifications — verified against the broker, not assumed.

Position sizing across instruments is where a multi-symbol backtest quietly
becomes fiction. A lot of gold, a lot of EURUSD and a lot of US500 are three
completely different amounts of risk and margin, and getting any one of them
wrong by a factor of ten changes every dollar figure in the report.

## How these were established

Every number here comes from `mt5.order_calc_profit` and `mt5.order_calc_margin`
on the live terminal, not from arithmetic on `trade_contract_size`. That
distinction mattered: MT5 reports XAUUSD `trade_tick_value = 0.10` against a
`trade_tick_size = 0.01`, which implies $1 per pip per lot. The authoritative
call returns **$10.00**, matching `contract_size x pip = 100 x 0.1`. Had the
field been trusted, every dollar result in this project would have been wrong by
10x in the safe direction.

    verified 2026-08-16, MetaQuotes-Demo, leverage 1:100

        symbol    $/pip/lot     margin/lot     pip      history
        XAUUSD       10.00          4,376      0.1      2012
        USDJPY        6.28          1,000      0.01     2012
        EURUSD       10.00          1,157      0.0001   2012
        GBPUSD       10.00          1,353      0.0001   2012
        USTEC         1.00            150      1.0      2022
        US500         1.00             39      1.0      2022

## The two that move

`USDJPY` is quote-currency JPY, so its pip value is `1000 / rate` USD and drifts
with the rate — $9.09 at 110, $6.28 at 159. It is computed per trade from the
entry price rather than frozen, because the sample spans 75-160.

Margin scales with price for everything except the JPY pair (base USD), so it is
also computed per trade. A margin figure frozen at today's gold price would
understate 2012 exposure by half.
"""
from __future__ import annotations

from dataclasses import dataclass

# Account leverage. The MetaQuotes demo runs 1:100, which is what the margin
# column above was measured at, but the operator's real accounts are **1:30** —
# the ESMA/FCA retail cap. That is not a detail: it TRIPLES margin per lot and
# turns margin from a non-binding constraint into the main one. At 1:30 a single
# 0.10-lot gold position ties up ~$1,459, roughly 15% of a $10,000 account.
LEVERAGE = 30.0


@dataclass(frozen=True)
class Contract:
    symbol: str
    pip: float               # price move of one pip
    contract_size: float     # units per 1.00 lot
    quote_ccy: str           # "USD" or the quote currency
    base_is_usd: bool        # margin notional is in USD without conversion
    min_lot: float
    lot_step: float
    max_lot_broker: float

    def usd_per_pip(self, price: float) -> float:
        """USD gained per pip of favourable move, per 1.00 lot."""
        if self.quote_ccy == "USD":
            return self.contract_size * self.pip
        # quote currency is not USD (USDJPY): the P&L accrues in JPY and
        # converts at the prevailing rate, so value per pip falls as the rate
        # rises. 100,000 x 0.01 = 1,000 JPY per pip, / rate = USD.
        return (self.contract_size * self.pip) / max(price, 1e-9)

    def margin_per_lot(self, price: float, leverage: float = None) -> float:
        """USD margin required for 1.00 lot at `price`."""
        notional = self.contract_size if self.base_is_usd \
            else self.contract_size * price
        return notional / (leverage or LEVERAGE)


CONTRACTS = {
    "XAUUSD": Contract("XAUUSD", 0.1, 100, "USD", False, 0.01, 0.01, 100),
    "EURUSD": Contract("EURUSD", 0.0001, 100_000, "USD", False, 0.01, 0.01, 500),
    "GBPUSD": Contract("GBPUSD", 0.0001, 100_000, "USD", False, 0.01, 0.01, 500),
    # base currency IS USD, so margin is a flat 100,000 / leverage and the pip
    # value is what floats
    "USDJPY": Contract("USDJPY", 0.01, 100_000, "JPY", True, 0.01, 0.01, 500),
    "USTEC": Contract("USTEC", 1.0, 1, "USD", False, 0.10, 0.10, 250),
    "US500": Contract("US500", 1.0, 1, "USD", False, 0.10, 0.10, 250),
    "US30": Contract("US30", 1.0, 1, "USD", False, 0.10, 0.10, 250),
}

# Per-symbol lot ceilings at a $10k balance.
#
# **Gold is the only genuinely capped instrument** — 0.10 lots, stated by the
# operator for their funded account. On a ~69-pip stop at $10/pip that ceiling
# allows about $69 of risk, i.e. ~0.7% of a $10k balance, so it binds slightly
# tighter than the 1% risk rule and is the real constraint on gold size.
#
# Everything else is set generously so the RISK PERCENTAGE binds instead. That
# matters: a US500 lot is $1/point against a ~15-point stop, so one lot risks
# about $15. Capping it at 1.00 lot the way gold is capped would hold the index
# to 0.15% risk per trade and silently delete it from the portfolio while
# appearing to include it. If the operator's firm imposes real per-symbol caps
# on these, put them here — `research/portfolio.py` reports how often a cap
# bound, so a wrong value shows up rather than hiding.
LOT_CAPS_10K = {
    "XAUUSD": 0.10,     # operator-specified, and the binding constraint
    "EURUSD": 0.50,
    "GBPUSD": 0.50,
    "USDJPY": 0.50,
    "USTEC": 10.00,
    "US500": 10.00,
    "US30": 10.00,
}


# Absolute size ceilings, regardless of how large the account grows.
#
# Without these, scaling lot caps linearly with balance compounds without limit
# and the backtest returns $36.9M from $10k — which is not a result, it is the
# absence of a constraint. At that balance the model was holding 369 lots of
# gold, roughly $160M notional, on a retail account. These are the sizes a
# retail/prop fill can absorb without the slippage model becoming fiction.
#
# They are deliberately generous rather than precise: the point is that the
# curve SATURATES, not that 5.0 is exactly right. Raise them if the operator
# trades size and knows their real fills.
MAX_LOT_ABS = {
    "XAUUSD": 5.0,      # ~$2.2M notional
    "EURUSD": 20.0,     # ~$2.3M notional
    "GBPUSD": 20.0,
    "USDJPY": 20.0,
    "USTEC": 100.0,
    "US500": 100.0,
    "US30": 100.0,
}


def caps_for(balance: float, base: float = 10_000.0) -> dict:
    """Lot ceilings for an account of this size.

    Scales linearly with balance so position size tracks equity the way a live
    account does, then clamps at `MAX_LOT_ABS` so compounding saturates instead
    of running to infinity.
    """
    k = balance / base
    return {s: min(round(v * k, 2), MAX_LOT_ABS.get(s, v * k))
            for s, v in LOT_CAPS_10K.items()}


def stepped_caps(balance: float, base: float = 10_000.0,
                 step: float = 0.05, per: float = 5_000.0) -> dict:
    """Lot ceilings that step up in fixed increments as the account grows.

    The operator's own sizing policy, and a far better model of how a real
    account is traded than linear scaling. Size does not track equity
    continuously — it is raised deliberately, in round increments, once the
    balance has cleared another threshold:

        gold:  0.10 at base, +`step` lots per `per` dollars of growth

    Every other instrument steps by the same proportion of its own base cap, so
    the book stays balanced rather than letting whichever symbol has the
    smallest contract quietly take over.

    This is what stops the compounding fantasy. Linear scaling produced $36.9M
    from $10k by holding 369 lots of gold; stepping is bounded by how often the
    balance actually clears a threshold, and `MAX_LOT_ABS` still caps the top.
    """
    steps = max(0.0, (balance - base) // max(per, 1e-9))
    out = {}
    for s, v in LOT_CAPS_10K.items():
        # scale each symbol's increment to its own base cap, so a 0.05 step on
        # gold (base 0.10) is a 50% increase and the indices move in proportion
        inc = step * (v / LOT_CAPS_10K["XAUUSD"])
        out[s] = min(round(v + inc * steps, 2), MAX_LOT_ABS.get(s, 1e9))
    return out
