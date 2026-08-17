# Systematic Trading Research Platform

A vectorized backtesting engine, feature-selection framework, and live execution
stack for multi-asset intraday and swing strategies — built to measure its own
assumptions rather than defend them.

```
49 modules · 11,419 lines · numpy + stdlib
6 instruments × 6 timeframes · 22.2 years of bar data (7.0M M1 bars)
9,736 bars/sec backtest throughput
```

The design brief was not "find a profitable strategy." It was **build research
infrastructure fast and honest enough that a hypothesis can be falsified in
seconds instead of an afternoon** — because the predecessor to this system
shipped four consecutive tuning waves on reasoning rather than measurement, and a
14-year run later found every one of them fitted to noise.

That framing runs through the whole codebase. Several of the headline results
here are negative, and they are reported as prominently as the positive ones.

---

## Why this exists

The previous system was measured over 14 one-year backtests — 2,296 trades,
2013–2026:

```
win rate         34.0%
avg win         +0.92R
avg loss        -0.53R
payoff           1.72
breakeven need   1.94
                 -> structurally losing by 0.22R per trade
```

Four winning years of fourteen. Decomposing it produced three findings that
determined the architecture of this rebuild:

1. **No computed entry feature predicted anything.** Spearman rank correlation
   against trade result over 299 trades came back at `|ρ| < 0.08` for every
   single one — cluster confidence, confluence count, touch count, zone width,
   entry displacement. Every feature was indistinguishable from random.

2. **The exit was destroying the edge.** 40.9% of all trades were scratched at
   +0.16R by a breakeven rung that armed at 0.33R — while the 75th-percentile
   maximum adverse excursion of a *winning* trade was 0.53R. The rung sat inside
   the noise a real winner has to survive, so it cut winners and losers with the
   same blade.

3. **The "levels" being traded were not levels.** A clustering routine compared
   each point to the *previous* point rather than a running centroid, and 84% of
   adjacent gaps sat inside the merge tolerance — so a cluster chained
   unbounded across a 537-point range and was priced at the mean of unrelated
   points. Median width was 49 points against a 60-point stop.

The full post-mortem, including two faults found by reading the code that appear
in no write-up, is in **[docs/DESIGN.md](docs/DESIGN.md)**.

---

## Architecture

```
core/       decision logic — pure, no I/O, no broker coupling, unit-testable
research/   ingestion, the vectorized engine, and the measurement studies
live/       MT5 connector, the autonomous loop, account guards
tests/      invariants, each guarding a specific historical failure mode
data/       bar store — mmap'd numpy columns (gitignored, ~5.4 GB)
```

### The engine

The predecessor drove the live production stack against a replay connector. It
was faithful, and it ran at **12 bars/sec** — 150 minutes per backtest year. At
that price a hypothesis costs an afternoon, so hypotheses stopped being tested.

**Speed is treated as a correctness feature**, and one algorithmic inversion buys
it: the old engine stepped bars and re-derived the level set at each one (75.9 ms
per bar, 97% of runtime). This one computes every level **once** with an explicit
birth and death index, asks each level where it fired across its own lifetime,
then makes a single pass over the resulting events.

```
O(bars × discovery)   ->   O(Σ level lifetimes) + O(events × trade length)
12 bars/sec           ->   9,736 bars/sec
150 min per year      ->   five full backtests in 11 seconds
```

The same `born` index that makes it fast also makes lookahead structurally hard:
a swing high at bar 100 confirmed by 5 bars carries `born = 105`, so no decision
before bar 105 can see it.

### The store

Plain `.npy` per column with `mmap_mode='r'`, one directory per
symbol-timeframe. The OS pages in only the columns and date ranges a study
actually touches — no parsing cost, no `pyarrow` dependency. Each timeframe is
fetched from the broker natively rather than resampled, because a resampled H4
disagrees with the broker's own H4 on session boundaries.

### Design rules

1. A level is a **price**. A zone is a **region**. They never share a struct.
2. **No fixed-pip constant anywhere in the decision path.** Every threshold is an
   ATR fraction — one number cannot serve a 124-point day and a 1,322-point day.
3. Clusters have a **hard width cap**, merged against the centroid.
4. The reference price is **a real constituent's price**, never a mean.
5. **One reference for state and trigger.**
6. **Weights are measured, not invented.**
7. **The exit is the strategy.** Ladders are in R, never pips.
8. **Research must be fast enough to argue with.**

---

## Results

Full tables in **[docs/RESULTS.md](docs/RESULTS.md)**. The headlines:

### Infrastructure — verified, no trades simulated

| Cluster width | Old | New |
|---|---:|---:|
| p50 | 48.6 p | **1.4 p** |
| p90 | 238.6 p | **21.2 p** |
| max | 931.4 p | **150.6 p** |
| ≥100 p wide | 29.8% | **0.1%** |
| p90 as % of stop | 398% | **13.2%** |

Zero cap breaches across 161 scans. Median cluster membership fell from 4 to 1.2,
with 61% singletons — confluence in the old tree was largely an artifact of the
chaining bug.

### Strategy — with its terms attached

A frozen swing configuration (`core/strategy.py`), selected from a
28,493-candidate feature study where every filter was measured **alone** against
a random-entry control before any were stacked:

```
XAUUSD, 2012-01 .. 2026-08 (14.6y), funded spread (1.67x the feed's own column)
1,282 trades (88/yr) · win 29.3% · +0.726R per trade
halves +0.750 / +0.702 · 15 of 15 years positive
```

Held to the honest standard, the out-of-sample picture is smaller. Because the
2020–2026 period is structurally kinder to this entry style (the unfiltered
population scores −0.212R in-sample and +0.049R out), every verdict is quoted as
**lift over the contemporaneous baseline**:

```
in-sample lift    +1.088R
out-of-sample     +0.602R      <- ~45% smaller than the headline
```

### Negative results, reported as prominently

- **A 198-feature scoring model** with a strict three-way temporal split retained
  **16%** of its in-sample discrimination and was **negative out of sample**.
- **A time-anchored intraday family** returned a gross edge of **−0.007R across
  521,606 events** — no downstream filter can recover an edge the trigger never
  carried.
- **35 candlestick patterns** were built and measured. **"No pattern at all" beat
  most named shapes** — a name is a threshold applied to a continuous quantity,
  and the threshold discards the information.
- **Against buy-and-hold at matched drawdown**, the intraday family returns 14.1%
  CAGR versus 12.4%. That is not an edge worth running software for, and the
  repo says so.

### The finding that reframed the problem

Controlled exit ablations on **identical entry sets** (1,158 entries, only the
exit varying) across a cost grid show every variant crossing from positive to
negative between 1.0× and 2.5× spread, at a uniform gradient of about
**−0.03R per spread multiple** — against a gross edge of +0.07 to +0.12R.

The system does not primarily have an entry problem or an exit problem. **Its
gross edge per trade is small relative to its transaction cost.**

---

## Methodology

Overfitting is the central risk in this domain, so the controls are explicit:

**A random-entry control.** Random entries at identical stop/target geometry and
identical costs return 30.8% win rate at −0.181R. Every result is judged against
that number, never against zero. If random entries also lost heavily, the
simulator would be the problem and every conclusion drawn from it worthless.

**A three-way temporal split.** Bin lifts are fitted on one in-sample block;
features are *selected* on a second block their own lifts never saw; 2020–2026 is
not touched until the model is frozen.

**A mechanical selection procedure.** The rule is written down so it can be
replayed on a subsample — which is what makes a genuine walk-forward possible
rather than just re-running a frozen config on later data. Both are run, and
labelled with which strength of evidence they are.

**Costs charged from the feed's own per-bar spread column**, never a flat
number. The measured shape on XAUUSD M5 (39,530 bars) is median 1.80p rising to
4.40p at rollover; `1.67×` maps that onto the 3–12p a funded account quotes.

**Pessimistic fill conventions.** Entry pays full spread; when one M1 bar spans
both stop and target the *stop* is taken; a trail rung armed by a bar takes
effect from the *next* bar; exits pay spread too. The known divergences from live
are enumerated in `research/engine.py`, each labelled optimistic or pessimistic.

---

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

The bar store is gitignored, so populate it first. With the MT5 terminal running
and logged in — and `Tools → Options → Charts → Max bars in chart` set to
`100000000`, which is what unlocks M1 back to 2004:

```bash
python -m research.fetch mt5 --symbols XAUUSD EURUSD --from 2004-01-01
```

Then:

```bash
python -m research.control            # random-entry control — is the harness fair?
python -m research.anatomy XAUUSD     # level precision, no trades simulated
python -m research.run --compare      # every exit variant, side by side
python -m research.features           # the 28,493-candidate feature study
python -m research.walkforward        # out-of-sample tests A and B
python -m research.portfolio          # six instruments through one account
pytest tests/ -v                      # 20 invariants
```

`MetaTrader5` is imported lazily and is only needed for `research.fetch` and
`live/` — every study runs against the store without it.

---

## Testing

20 invariant tests, written against specific historical failure modes rather than
for line coverage. The operating principle:

> A lookahead bug does not raise. It just prints a better number.

Causality is verified empirically rather than by inspection — indicators are
recomputed on truncated inputs and asserted unchanged, so no value at bar `i` can
depend on bar `i+1`. Others assert that no cluster exceeds its width cap, that
every reference price is a real constituent price, that clustering partitions its
input exactly once, and that a swing is never knowable before confirmation.

---

## Limitations

Stated plainly, because the alternative is someone discovering them later:

- **No meaningful live sample.** A two-week demo run is ~29 trades — far too few
  to separate a 36% win rate from noise. The live loop exists to test the things
  a backtest structurally cannot (does a signal fire on the same bar; does
  realized cost match the model; does the risk machinery bind), not to test edge.
- **The walk-forward is strong evidence, not proof.** Candidate feature *axes*
  were surfaced with full-sample knowledge even though the *thresholds* were
  re-chosen in-sample.
- **Slippage beyond the spread is not modelled.** On a stop run this is
  optimistic — the one place the result flatters itself.
- **No tick stream**, so nothing resembling order flow contributes.
- **One asset class dominates.** Most measurement is on XAUUSD; the multi-symbol
  work is more recent and less thoroughly validated.

---

## The lineage

Three iterations, each built to fix what measuring the previous one exposed. This
repository is the third.

| | Project | What it established |
|---|---|---|
| **v1** | [Gold MT5 Bot](https://github.com/Aarayanc49/gold-mt5-bot-v1) | z-score mean reversion on MT5. A fixed 2-unit stop sat inside gold's noise — **98.3% of 3,440 exits were stop-outs**. Fixed thresholds do not survive a changing volatility regime. |
| **v2** | [Trading Copilot](https://github.com/Aarayanc49/trading-copilot-v2) | 57K lines, 176 modules, a 14-year backtest driving the production stack itself. Proved the strategy structurally unprofitable at −0.22R/trade, and that **every engineered feature scored \|ρ\| < 0.10** against outcome. |
| **v3** | **XAUUSD Quant Backtest** *(this repo)* | Full rebuild on v2's findings. Every threshold ATR-scaled, a vectorized engine at **9,736 bars/sec (~800× faster)**, three-way validation split, random-entry control, levels as prices rather than regions. |

What v1 and v2 contributed directly to the design rules above: rule 2 (no fixed-pip
constant) is v1's stop failure and v2's self-inflicted "regime signal"; rule 7 (the
exit is the strategy) is v2's trail-step decomposition; rules 1, 3 and 4 are v2's
238.6-point clusters. The post-mortem in [docs/DESIGN.md](docs/DESIGN.md) is written
against v2's tree specifically.

None of the three found a durable edge. Each one narrowed down why, and the
measurement apparatus outlived the strategy every time.

---

## Disclaimer

This is research code published as a portfolio piece. All rights reserved.

Nothing here is financial advice, and **no result in this repository should be
treated as evidence of a tradeable edge.** The limitations section above is not
boilerplate — it is the honest summary of what has and has not been established.
