"""Read the live journal and say what actually happened.

The two-week demo run is not an edge test — ~29 trades cannot separate a 36%
win rate from noise. It is a test of the things a backtest structurally cannot
check, so this report is built around those rather than around P&L:

  * **Did signals fire?** A live loop that silently stops producing them looks
    exactly like a quiet market. Rejections are journalled with reasons, so the
    funnel is visible: signals seen -> passed filters -> actually placed.
  * **Do fills match the cost model?** Every measured result in this project
    charges the feed's spread column x 1.67. If live slippage plus spread runs
    materially above that, every dollar figure is optimistic and the cost curve
    needs re-fitting. This is the single most valuable number the run produces.
  * **Did the risk machinery bind?** Daily locks, margin ceilings, opposite-
    position refusals and lot caps all have live counters.

    python -m live.review
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JOURNAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "live_journal.jsonl")


def load(path=JOURNAL):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main():
    ev = load()
    if not ev:
        print("no journal yet")
        return 0

    starts = [e for e in ev if e["ev"] == "start"]
    sigs = [e for e in ev if e["ev"] in ("signal", "would_enter")]
    taken = [e for e in sigs if e.get("taken") and e.get("ok")]
    placed = [e for e in sigs if e.get("taken")]
    closes = [e for e in ev if e["ev"] == "close"]
    errs = [e for e in ev if e["ev"] in ("error", "scan_error")]

    print("=" * 92)
    print(f"  LIVE REVIEW — {len(ev):,} journal events")
    print("=" * 92)
    if starts:
        s = starts[-1]
        print(f"  account {s['login']}@{s['server']}   start balance "
              f"${s['balance']:,.2f}   leverage 1:{s['leverage']}")
        print(f"  running since {starts[0]['t']}   restarts: {len(starts)}")
    span = f"{ev[0]['t']} .. {ev[-1]['t']}"
    print(f"  window {span}")

    # ── the funnel ──────────────────────────────────────────────────────────
    print("\n  SIGNAL FUNNEL")
    print(f"    signals seen           {len(sigs):>6}")
    passed = [e for e in sigs if e.get("taken")]
    print(f"    passed the filters     {len(passed):>6}")
    print(f"    orders filled          {len(taken):>6}")
    rej = Counter(e.get("why") for e in sigs if not e.get("taken"))
    if rej:
        print("\n    rejected, by reason:")
        for why, n in rej.most_common():
            print(f"      {str(why):<22} {n:>6}")
    fails = [e for e in placed if e.get("ok") is False]
    if fails:
        print(f"\n    ORDER FAILURES        {len(fails):>6}")
        for e in fails[-5:]:
            print(f"      {e['symbol']:<8} {e.get('broker')}")

    # ── the number that matters ─────────────────────────────────────────────
    slips = [e["slippage_pips"] for e in taken
             if e.get("slippage_pips") is not None]
    if slips:
        a = np.array(slips, float)
        print("\n  EXECUTION vs THE COST MODEL")
        print(f"    fills                  {len(a):>6}")
        print(f"    slippage pips   median {np.median(a):>+6.2f}   "
              f"mean {a.mean():>+6.2f}   p90 {np.percentile(a, 90):>+6.2f}")
        sp = [e["spread_now"] for e in taken if e.get("spread_now")]
        if sp:
            b = np.array(sp, float)
            print(f"    spread at entry median {np.median(b):>6.2f} pips   "
                  f"p90 {np.percentile(b, 90):>6.2f}")
            print(f"    total entry cost median "
                  f"{np.median(b) / 2 + np.median(a):>6.2f} pips")
        print("    (research charges the feed spread x1.67; if live cost runs "
              "above\n     that, every dollar figure so far is optimistic)")

    # ── per symbol ──────────────────────────────────────────────────────────
    by = defaultdict(lambda: {"seen": 0, "taken": 0, "lots": 0.0})
    for e in sigs:
        d = by[e.get("symbol", "?")]
        d["seen"] += 1
        if e.get("taken") and e.get("ok"):
            d["taken"] += 1
            d["lots"] += float(e.get("lot") or 0)
    if by:
        print(f"\n  {'symbol':<10}{'signals':>9}{'filled':>8}{'lots':>9}")
        for s, d in sorted(by.items(), key=lambda kv: -kv[1]["seen"]):
            print(f"  {s:<10}{d['seen']:>9}{d['taken']:>8}{d['lots']:>9.2f}")

    # ── exits ───────────────────────────────────────────────────────────────
    if closes:
        print(f"\n  CLOSES  {len(closes)}")
        for why, n in Counter(e.get("why") for e in closes).most_common():
            pnl = [e.get("profit", 0.0) for e in closes if e.get("why") == why]
            print(f"    {why:<16} n={n:<4} total ${sum(pnl):>+9,.2f}")
        tot = sum(e.get("profit", 0.0) for e in closes)
        print(f"    {'ALL':<16} n={len(closes):<4} total ${tot:>+9,.2f}")

    locks = [e for e in ev if e["ev"] == "daily_lock"]
    if locks:
        print(f"\n  daily-loss locks: {len(locks)}")
    if errs:
        print(f"\n  ERRORS: {len(errs)}")
        for e in errs[-3:]:
            print(f"    {e['t']} {e.get('symbol','')} {e.get('err')}")

    # live positions, if the terminal is reachable
    try:
        from live.broker import Broker
        b = Broker()
        acc = b.connect()
        pos = b.positions()
        print(f"\n  NOW: equity ${acc.equity:,.2f} "
              f"(start ${starts[0]['balance']:,.2f}, "
              f"{(acc.equity/starts[0]['balance']-1)*100:+.2f}%)   "
              f"margin ${acc.margin:,.2f}   open {len(pos)}")
        for p in pos:
            print(f"    {p.symbol:<8} {'BUY ' if p.type == 0 else 'SELL'} "
                  f"{p.volume:>5.2f} @ {p.price_open:<10.3f} "
                  f"P&L ${p.profit:>+8.2f}")
        b.shutdown()
    except Exception as e:
        print(f"\n  (terminal not reachable: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
