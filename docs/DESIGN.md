# Design rationale and post-mortem

This is the document the rebuild was designed from. It is a rebuild, not a
refactor of the predecessor — a different system, informed by what the old one
measured about itself.

---

## Why the predecessor lost

Fourteen one-year backtests, 2,296 trades, 2013–2026:

```
win rate         34.0%
avg win         +0.92R
avg loss        -0.53R
payoff           1.72
breakeven need   1.94
                 -> structurally losing by 0.22R per trade
```

Four winning years of fourteen. Fixing it requires **win rate 34% → 37%** or
**payoff 1.72 → 1.94**. Nothing else moves the number.

Where the payoff went (live journal, 390 closed trades — the backtest agrees):

```
trail_step 0   n=245   -28.46p/trade    the entire loss lives here
trail_step 1   n= 51    +3.65p          the scratch band
trail_step 2+  n= 94   +95.50p/trade    essentially zero losses
```

The breakeven rung armed at +20p. On a 60p stop that is 0.33R; in 2026, with a
volatility-scaled stop and a fixed-pip ladder, it was 0.22R. Winners' p75 MAE is
0.53R. **The rung sat inside the noise a real winner still has to survive**, so it
cut winners and losers with the same blade. 50 live fills reached ≥45p and banked
−7.7p, handing back 3,823 pips — 1.8× the journal's entire net P&L.

And nothing upstream of the exit predicted anything. Spearman rank correlation
with trade result over 299 trades:

```
cluster confidence  -0.023     scan_pa score      +0.004
confluence count    +0.039     touch count        +0.045
zone width          -0.004     entry displacement +0.079
```

Every computed entry feature is indistinguishable from random.

---

## The three mechanical faults that caused that

1. **A "level" was two incompatible objects in one struct.** Seven of nine
   sources emitted a fixed 10-pip line; FVGs and order blocks (32% of all points)
   emitted their own width — FVG median 37p, max 931p. Same struct, treated
   identically downstream.

2. **Clustering chained without bound.** `cluster_levels` compared each point to
   the *previous point*, not the centroid, and 84% of adjacent gaps sat inside the
   20p tolerance. Result: median 49p "levels", p90 238p, max 931p, with `price`
   set to the *mean* of the chain — a number corresponding to no structure at all.
   The stop was 60p. On a p75 cluster price could travel the entire stop distance
   without leaving the "level".

3. **State and trigger read different prices.** `level_read` resolved
   SWEPT/HELD/BROKEN against the zone *edges*; `playbook` fired against the
   cluster *centre*, 24–119p away. So "level was swept → fade the sweep" was never
   the sequence actually executed.

Steps 4–8 of the old chain (classify, score, gate, size) were all reasoning
carefully about a number produced by step 2, and step 2 was where the level
stopped being a level.

### Two further faults found by reading the code, not in any write-up

4. **`eql` — the highest-edge source measured (+1.261 context-free, +2.433 in
   range regimes) — was fed arbitrary input.** `_find_equal_highs_lows` appends
   every bar pair within tolerance; `discover_levels` took `[:5]`. Measured over
   40 windows: 28 pairs generated, the 5 used always came from the oldest fifth of
   the window, and 20% of them were adjacent bars — one swing, not an equal high.

5. **The tier gate graded a different cluster set than the one being traded.**
   `level_intelligence` re-ran discovery on its own 290s cache and matched by 30p
   tolerance. The backtest popped that cache every bar; live did not. The gate
   that killed ~35% of candidates behaved differently in backtest than in
   production.

---

## Design rules for this tree

1. **A level is a price. A zone is a region. They never share a struct.**
   An entry triggers off a `LevelPoint`. A `Zone` may confirm or veto, never
   define the trigger price.

2. **No fixed-pip constant anywhere in the decision path.** Every threshold is an
   ATR fraction. One number cannot serve a 124p day and a 1,322p day.

3. **Clusters have a hard width cap.** Merge against the centroid, refuse any
   merge that breaks the cap, and let isolated levels stay singletons. A cluster
   that cannot be made precise is not emitted.

4. **The reference price is a real constituent's price**, never a mean.

5. **One reference for state and trigger.** The snapshot emits the edge that was
   swept; the trigger consumes that exact price.

6. **Weights are measured, not invented.** Source strength comes from the reaction
   study, which asks "does price react here more than at a random price?" over
   years of bars without simulating a single trade.

7. **The exit is the strategy.** Ladders are in R, not pips. Nothing arms inside
   the MAE band a winner has to survive.

8. **Research must be fast enough to argue with.** The old loop ran ~12 bars/sec,
   150 minutes per year — which is why every tuning wave shipped on unmeasured
   assumptions. The engine here is vectorized; a year is seconds.

---

## Two faults surfaced during the rebuild

Neither appears in any write-up of the old system:

* **Levels never expired.** The old tree re-derived everything per scan and bounded
  it only by distance. In a born-indexed store the consequence is visible: the
  first run emitted 5,007 half-round points of 9,970 — half the level universe was
  a handful of prices restated hundreds of times, all still voting at the end of
  the series. Every source now states an honest death.

* **Confluence was largely an artifact of the chain.** With a real width cap, 61%
  of clusters are singletons and median membership is 1.2 — against the old
  "median 4, max 31". A "high confluence" zone was usually one observation wearing
  seven names, because all nine of the old confluence factors derived from the
  same swing series.

---

## Costs

Spread is charged per bar from the feed's own `spread` column, never flat. The
measured shape on XAUUSD M5 (39,530 bars):

```
overall        median 1.80p   mean 2.30p   p90 3.40p   max 53.30p
server hr 1-2  4.40p   (rollover)
server hr 22-23 2.40p
all other       1.80p
```

A funded account (FundingPips) quotes 3–12p. Scaling the measured shape by 1.67
maps 1.80–6.40p onto 3.0–10.7p and reproduces that range, so 1.67 is the default
funded multiplier. The old v10 config had a **breakeven spread of 6.3p** — it was
trading below its own cost floor for part of every session.

---

## Data

MT5 `Tools → Options → Charts → Max bars in chart = 100000000` is required; it is
what unlocks M1 back to 2004. Available per the last survey:

```
tf    bars        span                      years
M1    7,008,753   2004-06-11 .. 2026-08-10  22.16
M5    1,487,374                             22.16
M15     508,902                             22.16
H1      128,888                             22.16
H4       33,947                             22.20
D1        5,694                             22.20
```

**Data before 2013 is not traded** — the gold market changed structurally around
2012/13. Pre-2013 bars are kept for context and warmup only.
