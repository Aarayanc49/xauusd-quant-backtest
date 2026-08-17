"""Reporting — in R, always.

The format is lifted from the old tree's YEAR_BY_YEAR output, which was the single
best artefact in that project: it reported R distribution, trail steps, MFE/MAE and
capture per year, and it is what finally made the losing arithmetic undeniable
after four tuning waves had argued about pips.

One rule enforced here: **everything is in R.** The old decomposition was first
done in fixed pip buckets, which is the exact mistake the codebase was being
criticised for — a 40-pip loss means something different on a 30-pip stop than on a
90-pip one, and bucketing by pips mixes them. It was redone in R and the
conclusions changed.

The headline block always prints the breakeven arithmetic, because that is the only
number that decides whether a system works:

    payoff  = avg win / |avg loss|
    need    = (1 - win_rate) / win_rate
    edge    = payoff - need        (positive or it does not work)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _fmt(rows, label, width=22) -> str:
    if not rows:
        return f"  {label:<{width}} n=    0"
    n = len(rows)
    r = np.array([t["r"] for t in rows])
    w = int((r > 0.05).sum())
    mfe = float(np.mean([t["mfe_r"] for t in rows]))
    cap = (r.mean() / mfe) if mfe > 1e-9 else 0.0
    return (f"  {label:<{width}} n={n:>5}  win={w/n:>5.1%}  "
            f"sum={r.sum():>+8.1f}R  avg={r.mean():>+6.3f}R  "
            f"mfe={mfe:>5.2f}R  cap={cap:>+5.2f}")


def arithmetic(trades) -> dict:
    """The only numbers that decide whether a system works.

    `need` is the payoff ratio at which this win/loss mix breaks even. It is
    `loss_rate / win_rate`, NOT the textbook `(1 - win_rate) / win_rate` — those
    agree only when every trade is a win or a loss. A trailed system parks a large
    fraction of trades in the scratch band (the old tree: 31.1% within +-0.1R), and
    charging those to the loss side makes `need` far too high. It showed up here as
    a comparison whose `edge` column said losing while its `avg R` column said
    winning; the arithmetic was wrong, not the run.
    """
    if not trades:
        return {}
    r = np.array([t["r"] for t in trades])
    n = len(r)
    wins, losses = r[r > 0.05], r[r < -0.05]
    scratch = n - len(wins) - len(losses)
    wr = len(wins) / n
    lr = len(losses) / n
    aw = float(wins.mean()) if len(wins) else 0.0
    al = float(abs(losses.mean())) if len(losses) else 0.0
    avg_scr = float(r[(r >= -0.05) & (r <= 0.05)].sum() / n) if scratch else 0.0
    payoff = aw / al if al > 1e-9 else float("inf")
    # break even when wr*avg_win + scratch_contribution = lr*|avg_loss|
    need = (lr / wr) - (avg_scr / (wr * al)) if wr > 1e-9 and al > 1e-9 else float("inf")
    return {"n": n, "win_rate": wr, "loss_rate": lr,
            "scratch_rate": scratch / n, "avg_win": aw, "avg_loss": -al,
            "payoff": payoff, "need": need, "edge": payoff - need,
            "sum_r": float(r.sum()), "avg_r": float(r.mean())}


def drawdown(trades) -> tuple:
    """Max drawdown in R, and the ratio that actually mattered on a funded account.

    The old v10 config earned +1,101p and drew down -648p. On a $5k account at 0.10
    lots that drawdown is $648 against a $500 total cap — blown, while showing a
    profit. Zero days breached the $250 daily stop; CUMULATIVE drawdown was the
    killer, which is why this is reported beside every P&L number.
    """
    if not trades:
        return 0.0, 0.0, 0
    eq = peak = dd = 0.0
    run = worst_run = 0
    for t in sorted(trades, key=lambda x: x["bar"]):
        eq += t["r"]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
        run = run + 1 if t["r"] < 0 else 0
        worst_run = max(worst_run, run)
    return dd, eq, worst_run


def summary(res, title: str = "", detail: bool = True) -> str:
    t = res.trades
    L = []
    add = L.append
    add("=" * 92)
    add(f"  {title or res.cfg.symbol}   ladder={res.cfg.ladder.name}  "
        f"spread x{res.cfg.spread_mult}")
    add("=" * 92)
    rate = res.bars / max(res.seconds, 1e-9)
    add(f"  bars {res.bars:,} {res.cfg.base}   levels {res.n_levels:,}   "
        f"events {res.n_events:,}   trades {len(t):,}")
    add(f"  wall {res.seconds:.2f}s  ({rate:,.0f} bars/s "
        f"— the old harness ran 12)")

    if res.rejected:
        add("\n  events rejected:")
        for k, v in sorted(res.rejected.items(), key=lambda x: -x[1]):
            add(f"    {k:<18} {v:>6,}")

    if not t:
        add("\n  NO TRADES.")
        return "\n".join(L)

    a = arithmetic(t)
    dd, final, worst_run = drawdown(t)
    add("\n" + "-" * 92)
    add(_fmt(t, "ALL"))
    add("-" * 92)
    add(f"  win rate          {a['win_rate']:>8.1%}")
    add(f"  loss rate         {a['loss_rate']:>8.1%}")
    add(f"  scratch rate      {a['scratch_rate']:>8.1%}")
    add(f"  avg win           {a['avg_win']:>+8.2f}R")
    add(f"  avg loss          {a['avg_loss']:>+8.2f}R")
    add(f"  payoff            {a['payoff']:>8.2f}")
    add(f"  breakeven need    {a['need']:>8.2f}")
    add(f"  EDGE              {a['edge']:>+8.2f}    "
        f"{'-> profitable' if a['edge'] > 0 else '-> STRUCTURALLY LOSING'}")
    add(f"  expectancy        {a['avg_r']:>+8.3f}R per trade")
    add(f"  total             {a['sum_r']:>+8.1f}R")
    add(f"  max drawdown      {dd:>+8.1f}R    final {final:+.1f}R")
    add(f"  R earned per R of drawdown  {(final / abs(dd)) if dd else 0:>6.2f}")
    add(f"  longest losing run{worst_run:>8}")

    if not detail:
        return "\n".join(L)

    def group(key, label, fn=None, sort_by_sum=True):
        add(f"\n  by {label}:")
        b = defaultdict(list)
        for x in t:
            b[(fn or (lambda v: v))(x.get(key))].append(x)
        keys = (sorted(b, key=lambda k: -sum(y["r"] for y in b[k]))
                if sort_by_sum else sorted(b))
        for k in keys:
            add(_fmt(b[k], str(k)))

    add("\n  R distribution:")
    bands = [("full stop <=-0.9R", lambda r: r <= -0.9),
             ("-0.9..-0.1R", lambda r: -0.9 < r <= -0.1),
             ("scratch +-0.1R", lambda r: -0.1 < r < 0.1),
             ("+0.1..+0.5R", lambda r: 0.1 <= r < 0.5),
             ("+0.5..+1R", lambda r: 0.5 <= r < 1.0),
             ("+1..+2R", lambda r: 1.0 <= r < 2.0),
             (">+2R", lambda r: r >= 2.0)]
    for name, f in bands:
        rows = [x for x in t if f(x["r"])]
        pct = len(rows) / len(t)
        add(f"    {name:<20} n={len(rows):>5} ({pct:>5.1%})  "
            f"sum={sum(x['r'] for x in rows):>+8.1f}R")

    add("\n  trail step reached (0 = ladder never armed):")
    st = defaultdict(list)
    for x in t:
        st[x["trail_step"]].append(x)
    for k in sorted(st):
        add(_fmt(st[k], f"step {k}"))

    group("exit_reason", "exit reason")
    group("kind", "trigger")
    group("direction", "direction")
    group("hour", "session (UTC)",
          lambda h: ("00-07 asia" if h < 7 else "07-12 london" if h < 12
                     else "12-17 ny-am" if h < 17 else "17-21 ny-pm" if h < 21
                     else "21-24 late"),
          sort_by_sum=False)
    group("n_members", "cluster members", sort_by_sum=False)
    group("depth_atr", "sweep depth (ATR)",
          lambda d: ("<0.1" if d < 0.1 else "0.1-0.25" if d < 0.25
                     else "0.25-0.5" if d < 0.5 else "0.5-1.0" if d < 1.0 else "1.0+"),
          sort_by_sum=False)
    group("risk_pips", "stop size (pips)",
          lambda p: f"{int(p // 25) * 25}-{int(p // 25) * 25 + 25}",
          sort_by_sum=False)

    # ── the choke ───────────────────────────────────────────────────────────
    # 290 trades in the old 14-year run reached 0.93R on average and banked
    # -0.06R, handing back 287R, with 288 of them sitting at trail step 1.
    choke = [x for x in t if x["mfe_r"] >= 0.75 and x["r"] <= 0.15]
    add(f"\n  CHOKE (MFE >= 0.75R, banked <= 0.15R): n={len(choke)} "
        f"({len(choke)/len(t):.1%})")
    if choke:
        add(f"    reached {np.mean([x['mfe_r'] for x in choke]):.2f}R  "
            f"banked {np.mean([x['r'] for x in choke]):+.2f}R  "
            f"given back {sum(x['mfe_r'] - x['r'] for x in choke):+.0f}R")

    add("\n  heat taken (MAE in R):")
    for name, rows in (("winners", [x for x in t if x["r"] > 0.05]),
                       ("losers", [x for x in t if x["r"] < -0.05])):
        if rows:
            m = np.array([x["mae_r"] for x in rows])
            add(f"    {name:<8} median {np.median(m):.2f}R  "
                f"p75 {np.percentile(m, 75):.2f}R  p90 {np.percentile(m, 90):.2f}R")

    add("\n  MFE capture:")
    edges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0),
             (1.0, 1.5), (1.5, 2.5), (2.5, np.inf)]
    for lo_e, hi_e in edges:
        name = f"{lo_e}-{hi_e}R" if np.isfinite(hi_e) else f"{lo_e}R+"
        rows = [x for x in t if lo_e <= x["mfe_r"] < hi_e]
        if rows:
            r = np.array([x["r"] for x in rows])
            mf = np.array([x["mfe_r"] for x in rows])
            add(f"    {name:<12} n={len(rows):>5} ({len(rows)/len(t):>5.1%})  "
                f"avg MFE {mf.mean():.2f}  result {r.mean():>+6.2f}R  "
                f"capture {r.mean()/mf.mean() if mf.mean() else 0:>+5.2f}")

    return "\n".join(L)


def split_halves(res) -> str:
    """First half vs second half. Nothing ships without surviving this.

    The old tree's one genuinely robust finding — the dense trail ladder — was
    accepted precisely because it held in both halves on two independent datasets.
    Every finding that skipped this check later failed on the real run.
    """
    t = sorted(res.trades, key=lambda x: x["bar"])
    if len(t) < 20:
        return "  (too few trades to split)"
    mid = len(t) // 2
    out = ["\n  two-halves split:"]
    for name, rows in (("H1", t[:mid]), ("H2", t[mid:])):
        a = arithmetic(rows)
        out.append(f"    {name}  n={a['n']:>5}  win={a['win_rate']:>5.1%}  "
                   f"avg={a['avg_r']:>+6.3f}R  payoff={a['payoff']:>5.2f}  "
                   f"need={a['need']:>5.2f}  edge={a['edge']:>+6.2f}")
    return "\n".join(out)
