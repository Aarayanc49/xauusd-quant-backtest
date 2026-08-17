# Measured results

Every figure here comes from a module in this repository. Nothing is estimated.
Where a result is weak, partial, or in-sample, it is labelled as such.

Reproduce any of these with the command named in its section.

---

## 1. Harness fairness — the control that comes first

`python -m research.control`

A negative result is only worth believing if the instrument is honest. Random
entries at identical stop/target geometry and identical costs:

```
random entry, 1.5 ATR stop, 2R target  ->  30.8% win, -0.181R
```

Expectation for a fair harness at a 1R stop and 2R target is `1/(1+2) = 33.3%`
minus a little for the stop-first tie rule, and an average R slightly negative by
roughly the spread cost. This is what it returns, so the simulator is fair and a
strategy scoring below it is genuinely bad.

**Every result below is judged against this number, never against zero.** At the
model's geometry the matched control is −0.163R.

---

## 2. Level precision — build verification, no trades simulated

`python -m research.anatomy XAUUSD`

The rebuild's gate. If p90 cluster width did not come under the cap, nothing
downstream was worth writing.

| Cluster width | Old | New |
|---|---:|---:|
| p50 | 48.6 p | **1.4 p** |
| p75 | 113.6 p | **10.1 p** |
| p90 | 238.6 p | **21.2 p** |
| max | 931.4 p | **150.6 p** |
| ≥ 100 p wide | 29.8% | **0.1%** |
| p90 as % of stop | 398% | **13.2%** |
| Cap breaches | n/a | **0 of 161 scans** |
| Median constituents | 4 (max 31) | **1.2 · 61% singletons** |

The membership collapse is itself a result: **confluence in the old tree was
largely an artifact of the chaining bug.** With a real width cap, most levels are
isolated.

---

## 3. Engine throughput

`python -m research.run --compare`

| | Old | New |
|---|---:|---:|
| Throughput | 12 bars/sec | **9,736 bars/sec** |
| One year | ~150 min | **seconds** |
| Five full backtests | ~12.5 hr | **11 sec** |
| Level discovery per bar | 75.9 ms (97% of runtime) | amortized once per series |
| 14 years | 2.5 hr on 14 workers | single process |

---

## 4. The exit study — identical entries, only the exit varies

`python -m research.run --compare --spread <x>`

1,158 identical entries, overlap allowed so nothing but the exit differs.
Average R per trade:

| spread × | default | late | legacy | none | p90 |
|---|---:|---:|---:|---:|---:|
| 0.00 | +0.070 | +0.063 | +0.041 | +0.117 | +0.047 |
| 1.00 | +0.043 | +0.017 | +0.004 | +0.093 | +0.015 |
| **1.67** (funded) | **+0.019** | −0.008 | −0.018 | **+0.071** | −0.004 |
| 2.50 | −0.008 | −0.034 | −0.035 | +0.046 | −0.043 |

Two readings, both measured:

1. **The old ladder is last in every cost regime.** `legacy` is the predecessor's
   ladder converted to R on its own 60p stop. It confirms from a different
   direction what the 14-year run found — the trail destroyed more than it saved.

2. **Cost is the dominant variable, not the ladder.** Every trailed variant
   crosses from positive to negative between 1.0× and 2.5× spread, at a gradient
   of about **−0.03R per spread multiple**, uniform across ladders. Against a
   gross edge of +0.07 to +0.12R, the funded spread takes more than half.

**This reframed the project.** The system does not primarily have an entry
problem or an exit problem; its gross edge per trade is small relative to its
transaction cost. Two consequences follow, both testable:

- **Make R bigger.** Spread is a fixed number of pips; its cost in R falls as the
  stop widens.
- **Trade less, and only where the expected move is large relative to cost.**

> ⚠ These figures come from a 3.5-month window on **completely unfiltered
> entries** — every sweep of every level fires, with no source weighting and no
> regime conditioning. They are a directional read on the exit and cost
> questions. **They are not evidence of an edge.**

---

## 5. Why the old exit failed — MAE decomposition

14-year trail-step decomposition:

```
trail_step 0   n=760 (33.1%)   -758.3R   avg -1.00R   <- the entire loss
trail_step 1   n=939 (40.9%)   +148.0R   avg +0.16R   <- the scratch band
trail_step 2   n=363 (15.8%)   +240.5R   avg +0.66R
trail_step 3   n=160 ( 7.0%)   +165.2R   avg +1.03R
trail_step 4   n= 61 ( 2.7%)    +91.1R   avg +1.49R
```

Against the measured heat a trade has to survive:

```
winners  median MAE 0.26R   p75 0.53R   p90 0.78R
losers   median MAE 0.95R   p75 1.08R   p90 1.19R
```

The rung armed at **0.33R**, which sits *inside the winners' own noise band*. It
did cut losses — that is why avg loss was −0.53R rather than −1.0R — but it
scratched 41% of all trades at +0.16R. A cohort of 290 trades reached 0.93R on
average and banked −0.06R, **handing back 287R**, with 288 of the 290 at rung 1.

And because the ladder was in fixed pips while the stop was volatility-scaled,
the arming point drifted with the regime — **0.67R in 2017** (124p range, 30p
stop) and **0.22R in 2026** (1,322p range, 90p stop). That drift *was* the
"regime signal" the study had found. It was self-inflicted.

---

## 6. The feature study

`python -m research.features`

28,493 candidates over 14.6 years, all run through **identical trade geometry**,
so no feature can look predictive purely by correlating with a tighter stop.
Every filter measured **alone** before any stacking.

Volatility was the dominant separator:

| Feature | n | Lift |
|---|---:|---:|
| ATR percentile ≥ 0.9 | 3,287 | +0.176 R |
| Day-range percentile ≥ 0.75 | 6,231 | +0.152 R |
| Expansion ≥ 1.4 | 129 | +0.387 R |

**Is that signal, or just a cheaper toll?** (`python -m research.voltest`) The
stop is 1.5 ATR, so a high-ATR bar has a wider stop in pips, and a fixed pip
spread costs less in R. Running identical candidates with cost on and cost off
separates the two. Cost sensitivity by stop width:

```
-0.37R @ 0.75 ATR      -0.18R @ 1.5 ATR      -0.10R @ 3 ATR
```

Retracement depth was **strongly non-monotonic** — which is why the scoring model
bins rather than fits a line:

```
0.2-0.4 bucket   -0.227R   <- the WORST in the study
0.6-0.8 bucket   +0.163R   <- the best
```

Shallow entries are chasing an extreme.

---

## 7. The frozen swing configuration

`core/strategy.py` · `python -m research.combine`

```
XAUUSD, 2012-01 .. 2026-08 (14.6y), funded spread (1.67x the feed column)
1,282 trades (88/yr) · win 29.3% · +0.726R per trade
halves +0.750 / +0.702 · 15 of 15 years positive
```

Geometry: stop 1.5 ATR (~44p median on gold), target 8R, hold 24h, **no trail**
(measured last in every cost regime — the exit is "don't").

Filters, each measured alone before stacking:

| Filter | Provenance |
|---|---|
| `range_pct >= 0.75` | Day-range percentile. The high-vol third carried the edge; the low third was negative. |
| `spread_pct < 0.5` | Cost, measured. Spread costs −0.181R at this stop and scales as 1/stop. |
| `session ∈ {london, ny, overlap}` | Asia and the dead hours were flat to negative and pay full spread to be there. |
| `h4_pullback >= 0.4` | Retracement depth into the current H4 leg — the non-monotonic result above. |

**Deliberately excluded:** full H4+H1 agreement measures better per trade
(+1.412R) but leaves 465 trades over 14.6 years — 33/yr. Kept as `SWING_STRICT`
for comparison, not as the default.

Why an 8R target: **85% of the edge lives past 2R** (2R = +0.074R, 8R =
+0.499R), and 86% of trades exit via stop or trail anyway. A hard near target
truncates exactly the trades that pay.

---

## 8. Out-of-sample — the test everything else skipped

`python -m research.walkforward`

Two questions, and they are **not the same strength of evidence**:

**Test A — freeze the config, run it on 2020–2026 alone.** *Weak.* The config was
chosen with these years visible, so this shows **stability**, not independence.
It can still falsify: if the config only worked because of 2012–2019, this is
where that shows up.

**Test B — re-run the selection procedure on 2012–2019 only, freeze whatever it
picks, apply it untouched to 2020–2026.** *Strong.* Nothing about the second
period touches the choice. This tests the **method** rather than the config.

The critical correction: **compare lift, not level.**

```
unfiltered population    -0.212R in-sample     +0.049R out-of-sample
```

The second period is structurally kinder to this entry style, so a config can
look stable while its actual edge shrinks. Measured against the contemporaneous
baseline:

```
config lift    +1.088R in-sample     +0.602R out-of-sample
                                     ~45% smaller than the headline
```

**Two stated limits.** The search space is not clean — the candidate *axes* were
surfaced by full-sample study even though the *thresholds* were re-chosen
in-sample. Read Test B as strong evidence rather than proof.

---

## 9. Negative results

These are reported as prominently as the positive ones. They are the more useful
half of the project.

### The 198-feature scoring model — `python -m research.model`

An additive, binned evidence-accumulation model (`score = Σ lift[f][bin_f(x)]`)
over an unbiased, trigger-free candidate universe. Chosen over a gate stack
because gates multiply and collapse: six individually-defensible filters took the
predecessor to **7% of population for a break-even edge**.

Strict three-way split — lifts fitted on IS-A, features selected on IS-B, OOS
never touched until frozen.

```
retained 16% of in-sample discrimination
NEGATIVE out of sample
```

### The time-anchored intraday family — `python -m research.anchored`

32 kinds of clock-anchored reference price — Asia high, yesterday's low, the
previous H1 candle's extreme:

```
521,606 events    gross edge -0.007R
```

**If the trigger carries nothing, no amount of downstream filtering recovers it.**
This is the result that motivated building a trigger-free universe.

### The candlestick pattern library — `core/patterns.py`

All 35 patterns built and measured. **"No pattern at all" beat most named
shapes.** The reason is visible once the detectors are written down: a name is a
threshold applied to a continuous quantity, and the threshold throws away the
part carrying the information. `pin` is `lower_wick > 0.55 * range`; the market
does not know about 0.55. Replaced by continuous candle anatomy (`core/shape.py`).

### Versus buy-and-hold — `python -m research.benchmark`

The comparison that has to clear before any of this is worth the software.
Compared on identical data and dates, with buy-and-hold leveraged so its worst
drawdown *matches* the strategy's — the only fair comparison between two things
with different risk:

```
intraday family   14.1% CAGR
buy and hold      12.4% CAGR    at the same drawdown
```

**That is not an edge worth running software for**, and the honest answer is to
say so rather than keep tuning.

### The structure-break population — `core/majors.py`

`find_breaks` produced **110,561 structure breaks over 14.6 years** — roughly 30
per day on M5. Trading all of them gave a +0.147R gross edge that the spread ate
entirely. A break of a minor swing that formed 40 minutes ago is not "resistance
breaking"; it is noise with a name.

### The holding-period problem — `python -m research.holding`

2024 is the case in point: **gold rose 27.1% and the strategy lost 41.5R.** No
intraday system with a 24-hour clock can participate in a move that takes months.

---

## 10. Account-level results

`python -m research.accounts` · `python -m research.portfolio` · `python -m research.deploy`

Expectancy and account survival are **different claims**. The predecessor's
config earned +1,101 pips and drew down −648, which on a $5,000 account at 0.10
lots is $648 against a $500 total cap — **profitable and blown at the same
time.** Zero days breached the daily stop; cumulative drawdown was the killer.

Modelled: risk-based sizing under a lot ceiling, daily loss limit, max-drawdown
failure, staged funded targets, breach-and-restart, withdrawals, margin refusal,
and no-opposite-positions.

**Floating loss is modelled, not just closed P&L**, because a prop firm checks
drawdown against *equity*. A book holding six correlated instruments can sit well
inside its limits on closed trades and still breach on floating loss. Both
figures are reported — breaches on closed balance and on worst-case equity — and
the gap between them is a statement about how correlated the book is.

**The overlap correction.** The model's top 50% reports 1,275 trades/yr at
+0.146R, which multiplies to ~186R/yr against the hand-picked config's ~85R/yr.
That number is arithmetic, not money, because **those trades overlap** — a real
account cannot carry 1,275 concurrent positions. Walking the stream in time order
and skipping entries that arrive while a position is open gives the honest cost
of that overlap, and the previously-published figures were re-run on the same
footing rather than only the new ones.

---

## 11. Contract specifications

`core/contracts.py` — verified against the broker's authoritative pricing calls,
not derived from reported instrument fields.

```
verified 2026-08-16, MetaQuotes-Demo, leverage 1:100

    symbol    $/pip/lot     margin/lot     pip      history
    XAUUSD       10.00          4,376      0.1      2012
    USDJPY        6.28          1,000      0.01     2012
    EURUSD       10.00          1,157      0.0001   2012
    GBPUSD       10.00          1,353      0.0001   2012
    USTEC         1.00            150      1.0      2022
    US500         1.00             39      1.0      2022
```

**This check caught a 10× error.** MT5 reports XAUUSD `trade_tick_value = 0.10`
against `trade_tick_size = 0.01`, implying $1 per pip per lot. The authoritative
call returns **$10.00**, matching `contract_size × pip = 100 × 0.1`. Had the
field been trusted, every dollar figure in this project would have been wrong by
10× — in the safe direction, which is exactly why it would never have been caught
by a result looking wrong.

`USDJPY` is quote-currency JPY, so its pip value is `1000 / rate` USD and drifts
with the rate — $9.09 at 110, $6.28 at 159. It is computed per trade from entry
price rather than frozen, because the sample spans 75–160.

---

## What none of this establishes

- **No meaningful live sample exists.** Two weeks is ~29 trades; the confidence
  interval on a 36% win rate over 29 trades covers everything from broken to
  excellent.
- **No Sharpe ratio or CAGR is claimed for the swing strategy.** The one CAGR
  figure measured (§9) is explicitly not worth running software for.
- **The 3.5-month exit window (§4) is not evidence of an edge** and is labelled
  so in the source.
- Slippage beyond spread is not modelled; on a stop run this is optimistic.
