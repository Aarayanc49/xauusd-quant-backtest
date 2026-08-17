"""What is the trader thinking — every decision, decomposed.

A live run that takes no trades all day is indistinguishable, from the outside,
from a live run that has silently died. `live/review.py` answers "what
happened" after the fact and is built around fills. This answers the question
that comes first and is asked far more often:

    it has been running all day and sent me nothing. What is it doing?

There are exactly four reasons this system does not trade, and until now the
journal could not tell them apart:

    1. the loop is not running, or is outside its trading window
    2. an account-level guard is standing it down (daily lock, margin, max
       concurrent positions)
    3. no candidate exists — price never produced a structure break, an impulse
       leg and a retrace into it
    4. candidates existed and the filters rejected them

Only (4) is the strategy working as designed. The other three are operational
facts that look identical in a journal that only records fills, and (1) is the
one that should page you.

## The four modes

    python -m live.explain            # what has it been thinking (journal)
    python -m live.explain --now      # gate state RIGHT NOW, per symbol
    python -m live.explain --setups   # setups on the board + what they target
    python -m live.explain --zones    # where a setup CAN form, before one does

They answer questions in the order you actually ask them. `--zones` is the
earliest: a setup here needs a level to break and then be retested, so the
levels alive right now are the complete set of prices where one can appear —
nothing triggers anywhere else. `--setups` is the next step, listing the ones
that have already formed with their entry, stop and target. `--now` says
whether the gates would let any of them through, and journal mode says what
happened when they did not.

`--now` connects read-only,
and for every symbol prints the gate state as it stands this minute — so you
can see that, say, gold's day-range percentile is 0.42 against a 0.75
requirement, and therefore **no setup on gold can trade right now no matter how
good it looks**. That is a different and much more useful statement than "no
signals today".

Journal mode reconstructs the same reasoning retroactively. The four fields the
trader records on a rejection are exactly the four the filters read, so every
past rejection can be decomposed after the fact without re-running anything.

## Margins, and why they are the interesting column

A candidate that missed a threshold by 0.004 and one that missed by 0.6 are
different events. The first means the config is nearly firing and a quiet day
is ordinary; the second means the market is nowhere near the regime this
strategy wants, and a quiet week is expected. `Spec.verdict` returns a signed
margin for exactly this, and the near-miss table below is sorted on it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy import SWING  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL = os.path.join(ROOT, "data", "live_journal.jsonl")

SYMBOLS = ["XAUUSD", "USDJPY", "EURUSD", "GBPUSD", "USTEC", "US500"]
TRADE_FROM_HOUR = 7
FLAT_BY_HOUR = 21


def _console():
    """Make stdout safe for box-drawing, or fall back to ASCII.

    The Windows console defaults to cp1252, which cannot encode `─` and raises
    UnicodeEncodeError mid-report — losing the output for a cosmetic reason.
    Reconfiguring to UTF-8 fixes it where the terminal supports it; where it
    does not, ASCII glyphs are used instead. A monitoring tool that crashes on
    its own header is worse than a plain one.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        return "─", "█"
    except (AttributeError, OSError):
        return "-", "#"


BAR, BLOCK = _console()


# ── output helpers ──────────────────────────────────────────────────────────

def head(title: str, width: int = 78):
    print()
    print(title)
    print(BAR * width)


def mark(ok: bool) -> str:
    return "pass" if ok else "FAIL"


def fmt_margin(m) -> str:
    if m is None:
        return "     ·"
    return f"{m:+7.3f}"


def fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


# ── journal ─────────────────────────────────────────────────────────────────

def load(path=JOURNAL) -> list:
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def parse_t(rec):
    t = rec.get("t")
    if not t:
        return None
    try:
        return datetime.fromisoformat(t)
    except ValueError:
        return None


def verdict_of(rec) -> list | None:
    """Recompute the per-filter verdict for a journalled rejection.

    Works on old journal lines because the trader already records exactly the
    four fields the filters read. Returns None if a field is missing, which is
    itself worth surfacing rather than guessing around.
    """
    needed = ("range_pct", "spread_pct", "session", "h4_pullback")
    if not all(k in rec for k in needed):
        return None
    return SWING.verdict(rec)


# ── journal mode ────────────────────────────────────────────────────────────

def report_journal(rows: list, day: str | None = None):
    if not rows:
        print("No journal at", JOURNAL)
        print("The trader has never written a line. It is not running, or it")
        print("has never completed a pass.")
        return

    if day:
        rows = [r for r in rows if str(r.get("t", "")).startswith(day)]
        if not rows:
            print(f"No journal entries for {day}.")
            return

    kinds = Counter(r.get("ev", "?") for r in rows)
    sigs = [r for r in rows if r.get("ev") == "signal"]
    taken = [r for r in sigs if r.get("taken")]
    rejected = [r for r in sigs if not r.get("taken")]

    times = [t for t in (parse_t(r) for r in rows) if t]
    first, last = (min(times), max(times)) if times else (None, None)

    # ── 1. is it even alive ────────────────────────────────────────────────
    head("1. LOOP STATE")
    print(f"  journal            {JOURNAL}")
    print(f"  entries            {len(rows):,}")
    if first and last:
        print(f"  first entry        {first:%Y-%m-%d %H:%M:%S} UTC")
        print(f"  last entry         {last:%Y-%m-%d %H:%M:%S} UTC")
        age = (datetime.now(timezone.utc) - last).total_seconds() / 60
        state = ("ALIVE" if age < 10 else
                 "STALE — no entry in over an hour" if age > 60 else
                 "quiet")
        print(f"  last write         {age:,.0f} min ago   -> {state}")
    print(f"  event types        " + ", ".join(f"{k}={v}" for k, v in kinds.most_common()))

    starts = [r for r in rows if r.get("ev") == "start"]
    if starts:
        s = starts[-1]
        print(f"  mode               " +
              ("DRY RUN — decides and journals, places nothing"
               if s.get("dry_run") else "LIVE — will place orders"))
        print(f"  account            {s.get('login')}@{s.get('server')}  "
              f"balance ${s.get('balance', 0):,.2f}  risk {s.get('risk_pct')}%")
        if len(starts) > 1:
            print(f"  restarts           {len(starts)} starts recorded "
                  f"— the loop has been restarted, not running continuously")

    # state transitions — why the loop was doing nothing, minute by minute
    states = [r for r in rows if r.get("ev") == "state"]
    if states:
        head("1b. STATE TIMELINE — what the loop was doing")
        prev_t = None
        for r in states:
            t = parse_t(r)
            held = ""
            if prev_t and t:
                mins = (t - prev_t).total_seconds() / 60
                held = f"  (held {mins:,.0f} min)"
            detail = ", ".join(f"{k}={v}" for k, v in r.items()
                               if k not in ("ev", "state", "t"))
            ts = f"{t:%Y-%m-%d %H:%M}" if t else "?"
            print(f"  {ts}  {r.get('state', '?'):<16s} {detail}{held}")
            prev_t = t
        print()
        print("  A gap here with no transition means the process itself was")
        print("  not running — the loop writes a line whenever its state changes.")
    else:
        head("1b. STATE TIMELINE")
        print("  No `state` events in this journal.")
        print("  This trader build predates state logging, so the periods when")
        print("  it was outside its window or stood down by a guard were never")
        print("  recorded. Restart the loop to begin capturing them.")

    # ── 2. the funnel ──────────────────────────────────────────────────────
    head("2. FUNNEL — where candidates went")
    n_scan_err = kinds.get("scan_error", 0)
    print(f"  candidates seen              {len(sigs):>6,}")
    print(f"    rejected by filters        {sum(1 for r in rejected if r.get('why') == 'filters'):>6,}")
    other = Counter(r.get("why") for r in rejected if r.get("why") != "filters")
    for why, n in other.most_common():
        print(f"    rejected: {str(why):<18s} {n:>6,}")
    print(f"    TAKEN                      {len(taken):>6,}")
    if n_scan_err:
        print(f"  scan errors                  {n_scan_err:>6,}   <- the pipeline threw")
    for ev in ("skip_all", "error"):
        if kinds.get(ev):
            print(f"  {ev:<28s} {kinds[ev]:>6,}")

    if not sigs:
        print()
        print("  No candidate was produced at all. That is reason (3): price did")
        print("  not form a structure break -> impulse -> retrace on any symbol.")
        print("  It is NOT the filters rejecting things — nothing reached them.")

    # ── 3. which filter is binding ─────────────────────────────────────────
    filt = [r for r in rejected if r.get("why") == "filters"]
    verdicts = [(r, verdict_of(r)) for r in filt]
    usable = [(r, v) for r, v in verdicts if v]
    if usable:
        head("3. WHY NOTHING TRADED — per-filter block counts")
        blocked = Counter()
        margins = defaultdict(list)
        for _, v in usable:
            for f in v:
                if not f["passed"]:
                    blocked[f["name"]] += 1
                    if f["margin"] is not None:
                        margins[f["name"]].append(f["margin"])
        print(f"  {'filter':<26s} {'blocked':>8s} {'of':>6s}  {'median miss':>12s}")
        print(f"  {BAR * 26} {BAR * 8} {BAR * 6}  {BAR * 12}")
        for name in SWING.filters:
            n = blocked.get(name, 0)
            ms = sorted(margins.get(name, []))
            med = f"{ms[len(ms) // 2]:+.3f}" if ms else "·"
            flag = "  <- binding" if n == max(blocked.values(), default=0) and n else ""
            print(f"  {name:<26s} {n:>8,} {len(usable):>6,}  {med:>12s}{flag}")

        solo = Counter()
        for _, v in usable:
            fails = [f for f in v if not f["passed"]]
            if len(fails) == 1:
                solo[fails[0]["name"]] += 1
        if solo:
            print()
            print("  Failed on ONE filter only (everything else was ready):")
            for name, n in solo.most_common():
                print(f"    {name:<26s} {n:>4,}")

        # ── 4. near misses ─────────────────────────────────────────────────
        near = []
        for r, v in usable:
            fails = [f for f in v if not f["passed"] and f["margin"] is not None]
            if len(fails) == 1:
                near.append((fails[0]["margin"], r, fails[0]))
        near.sort(key=lambda x: -x[0])
        if near:
            head("4. NEAR MISSES — one filter away, closest first")
            print(f"  {'time':<17s} {'symbol':<8s} {'dir':<5s} {'blocked by':<26s} {'miss':>8s}")
            print(f"  {BAR * 17} {BAR * 8} {BAR * 5} {BAR * 26} {BAR * 8}")
            for m, r, f in near[:15]:
                t = str(r.get("t", ""))[:16].replace("T", " ")
                print(f"  {t:<17s} {r.get('symbol', '?'):<8s} "
                      f"{r.get('direction', '?'):<5s} {f['name']:<26s} {m:>+8.3f}")
            if len(near) > 15:
                print(f"  ... and {len(near) - 15} more")

    # ── 5. per symbol ──────────────────────────────────────────────────────
    if sigs:
        head("5. PER SYMBOL")
        per = defaultdict(lambda: [0, 0])
        for r in sigs:
            per[r.get("symbol", "?")][0] += 1
            if r.get("taken"):
                per[r.get("symbol", "?")][1] += 1
        print(f"  {'symbol':<10s} {'candidates':>11s} {'taken':>7s}")
        print(f"  {BAR * 10} {BAR * 11} {BAR * 7}")
        for sym in sorted(per, key=lambda s: -per[s][0]):
            n, tk = per[sym]
            print(f"  {sym:<10s} {n:>11,} {tk:>7,}")
        silent = [s for s in SYMBOLS if s not in per]
        if silent:
            print(f"\n  produced no candidate at all: {', '.join(silent)}")

    # ── 6. hourly ──────────────────────────────────────────────────────────
    if sigs:
        head("6. ACTIVITY BY HOUR (UTC)")
        by_h = Counter()
        for r in sigs:
            t = parse_t(r)
            if t:
                by_h[t.hour] += 1
        peak = max(by_h.values(), default=1)
        for h in range(24):
            n = by_h.get(h, 0)
            inwin = TRADE_FROM_HOUR <= h < FLAT_BY_HOUR
            bar = BLOCK * int(round(24 * n / peak)) if n else ""
            note = "" if inwin else "  (outside trading window)"
            print(f"  {h:02d}:00  {n:>4,} {bar}{note}")

    print()
    print(BAR * 78)
    print("Run `python -m live.explain --now` to see the CURRENT gate state,")
    print("which tells you whether a setup could trade at all right now.")


# ── live mode ───────────────────────────────────────────────────────────────

def gate_state(broker, symbol: str) -> dict | None:
    """The four filter fields as they stand on the latest CLOSED bar.

    Computed the same way `research/features.build` computes them, from the
    same live series the trader itself reads — so this is the state the next
    candidate on this symbol would actually be judged against, not an estimate.
    """
    from core import bias as BI
    from core import session as SE
    from core.context import Context
    from live import signals as SIG

    series = SIG.fetch_series(broker, symbol)
    if series is None:
        return None
    bars = series["M5"]
    ctx = Context(series, base="M5")
    atr = ctx.scales["M5"].atr
    st = SE.build(bars, atr)
    bi = BI.build(series, ctx, "M5")
    i = len(bars) - 1
    depth = float(BI.pullback_depth(bi.h4_pos[i:i + 1], bi.h4_dir[i:i + 1])[0])
    return {
        "bar_time": bars.t_str(i),
        "range_pct": float(st.range_pct[i]),
        "spread_pct": float(st.spread_pct[i]),
        "spread_pips": float(st.spread_pips[i]),
        "session": st.session_name(i),
        "h4_pullback": depth,
        "h4_dir": int(bi.h4_dir[i]),
        "h1_dir": int(bi.h1_dir[i]),
        "atr_pct": float(st.atr_pct[i]),
    }


def report_now(symbols: list, lookback: int = 300):
    try:
        from live.broker import Broker
    except Exception as e:
        print("Cannot import the broker:", e)
        return
    from live import signals as SIG

    b = Broker()
    try:
        acc = b.connect_readonly()
    except Exception as e:
        print("Cannot reach MetaTrader 5:", e)
        print("\nThat is reason (1): if this cannot connect, neither can the")
        print("trader. Check the terminal is running and logged in.")
        return

    now = datetime.now(timezone.utc)
    head("LIVE STATE")
    print(f"  time               {now:%Y-%m-%d %H:%M:%S} UTC")
    print(f"  account            {acc.login}@{acc.server}")
    print(f"  balance / equity   ${acc.balance:,.2f} / ${acc.equity:,.2f}")
    in_window = TRADE_FROM_HOUR <= now.hour < FLAT_BY_HOUR
    print(f"  trading window     {TRADE_FROM_HOUR:02d}:00-{FLAT_BY_HOUR:02d}:00 UTC   "
          f"-> {'OPEN' if in_window else 'CLOSED — nothing can trade now'}")
    try:
        pos = b.positions()
        print(f"  open positions     {len(pos)}")
    except Exception:
        pass

    if not in_window:
        print()
        print("  The loop is outside its trading window. It is scanning nothing")
        print("  and will not open a position until the window reopens. This is")
        print("  reason (1), not a strategy decision.")

    for sym in symbols:
        head(f"{sym}")
        if not b.ensure_symbol(sym):
            print("  symbol unavailable in this terminal")
            continue
        try:
            g = gate_state(b, sym)
        except Exception as e:
            print(f"  could not build context: {type(e).__name__}: {e}")
            continue
        if g is None:
            print("  not enough history returned by the terminal")
            continue

        v = SWING.verdict(g)
        n_fail = sum(1 for f in v if not f["passed"])
        print(f"  latest closed M5 bar   {g['bar_time']}")
        print(f"  H4 dir {g['h4_dir']:+d}   H1 dir {g['h1_dir']:+d}   "
              f"ATR pctile {g['atr_pct']:.2f}   spread {g['spread_pips']:.1f}p")
        print()
        print(f"  {'gate':<26s} {'now':>9s}  {'verdict':<7s} {'margin':>8s}")
        print(f"  {BAR * 26} {BAR * 9}  {BAR * 7} {BAR * 8}")
        for f in v:
            print(f"  {f['name']:<26s} {fmt_val(f['value']):>9s}  "
                  f"{mark(f['passed']):<7s} {fmt_margin(f['margin']):>8s}")

        if n_fail == 0:
            print()
            print("  ALL GATES OPEN. A setup appearing on this symbol right now")
            print("  would be taken. Nothing is firing because no structure")
            print("  break -> impulse -> retrace has formed — reason (3).")
        else:
            blocking = [f["name"] for f in v if not f["passed"]]
            print()
            print(f"  BLOCKED by {n_fail} gate(s): {', '.join(blocking)}")
            print("  No setup on this symbol can trade until that changes,")
            print("  however good the price action looks — reason (4).")

        # recent candidates and how each was judged
        try:
            sigs = SIG.scan(b, sym, lookback_bars=lookback)
        except Exception as e:
            print(f"  (candidate scan failed: {type(e).__name__}: {e})")
            continue
        if not sigs:
            print(f"  no candidate in the last {lookback} M5 bars "
                  f"(~{lookback * 5 / 60:.0f}h)")
            continue
        fired = sum(1 for s in sigs if not SWING.blocked_by(s))
        print(f"\n  candidates in the last {lookback} M5 bars: {len(sigs)}"
              f"   would have fired: {fired}")
        print(f"  {'bar':<17s} {'dir':<5s} {'result':<10s} {'blocked by':<44s}")
        print(f"  {BAR * 17} {BAR * 5} {BAR * 10} {BAR * 44}")
        for s in sorted(sigs, key=lambda x: str(x.get("ts", "")))[-10:]:
            blocked = SWING.blocked_by(s)
            t = str(s.get("ts", ""))[:16].replace("T", " ")
            res = "WOULD FIRE" if not blocked else "rejected"
            # name the field, not the whole predicate — the thresholds are in
            # the gate table directly above and repeating them here is what
            # forced the column past the width of a terminal
            keys = [b.split()[0] for b in blocked]
            print(f"  {t:<17s} {s['direction']:<5s} {res:<10s} "
                  f"{', '.join(keys)[:44]:<44s}")

    b.shutdown()


# ── setups mode ─────────────────────────────────────────────────────────────

def _plan(trader, broker, s, acc):
    """Entry / stop / target / size for one candidate, priced right now.

    Deliberately calls the SAME `Trader.size` the live loop calls rather than
    reimplementing the arithmetic. A monitor that computes its own lot is a
    monitor that can disagree with the thing it is monitoring, which is the
    exact failure the research-vs-live split in this project exists to avoid.
    """
    from core.contracts import CONTRACTS
    from live.trader import TARGET_R
    sym = s["symbol"]
    is_buy = s["direction"] == "buy"
    c = CONTRACTS[sym]
    tick = broker.tick(sym)
    if tick is None:
        return None
    price = tick.ask if is_buy else tick.bid
    stop_pips = float(s["risk_pips"])
    dist = stop_pips * c.pip
    sl = price - dist if is_buy else price + dist
    tp = price + TARGET_R * dist if is_buy else price - TARGET_R * dist
    lot, reason = trader.size(sym, acc, stop_pips, price)
    risk_usd = lot * stop_pips * c.usd_per_pip(price) if lot > 0 else 0.0
    # Entry is the LIVE tick, so a plan for an old setup is a hybrid: a
    # structural stop measured at the setup's own bar, against a price from
    # now. That is the correct number for a setup forming this minute and a
    # misleading one for a setup from this morning, so the age is carried
    # alongside and the caller marks it.
    return {"price": price, "sl": sl, "tp": tp, "stop_pips": stop_pips,
            "target_pips": stop_pips * TARGET_R, "lot": lot,
            "size_reason": reason, "risk_usd": risk_usd,
            "reward_usd": risk_usd * TARGET_R, "pip": c.pip,
            "priced_at": "live tick"}


def _server_offset_hours(broker, symbol: str) -> float:
    """Hours the broker's clock runs ahead of UTC, measured not assumed.

    MT5 stamps bars in SERVER time and exposes no timezone for it. Most gold
    brokers run UTC+2/+3, so subtracting a bar stamp from a UTC clock produces
    a negative age and a setup from 25 minutes ago reads as "-25m ago".

    The newest CLOSED bar is by definition between one and two bar-widths old,
    so the difference between its stamp and now, rounded to the nearest hour,
    is the offset. Measuring it beats hard-coding +3: it follows the broker
    across DST, and if a terminal ever reports true UTC this returns 0 and
    nothing changes.
    """
    r = broker.bars(symbol, "M5", 2)
    if r is None or len(r) == 0:
        return 0.0
    newest = float(r[-1]["time"])
    now = datetime.now(timezone.utc).timestamp()
    return round((newest - now) / 3600.0)


def report_setups(symbols: list, lookback: int = 400, limit: int = 6):
    """Every setup the engine currently sees, with what each one targets."""
    from live.broker import Broker
    from live import signals as SIG
    from live.trader import TARGET_R, Trader

    b = Broker()
    try:
        acc = b.connect_readonly()
    except Exception as e:
        print("Cannot reach MetaTrader 5:", e)
        return
    trader = Trader(b, dry_run=True)

    now = datetime.now(timezone.utc)
    in_window = TRADE_FROM_HOUR <= now.hour < FLAT_BY_HOUR
    head("SETUPS ON THE BOARD")
    print(f"  {now:%Y-%m-%d %H:%M:%S} UTC   equity ${acc.equity:,.2f}   "
          f"window {'OPEN' if in_window else 'CLOSED'}   target {TARGET_R:.0f}R")

    total = live_n = 0
    for sym in symbols:
        if not b.ensure_symbol(sym):
            continue
        try:
            sigs = SIG.scan(b, sym, lookback_bars=lookback)
        except Exception as e:
            print(f"\n{sym}: scan failed — {type(e).__name__}: {e}")
            continue
        if not sigs:
            continue
        offset_h = _server_offset_hours(b, sym)
        sigs = sorted(sigs, key=lambda x: str(x.get("ts", "")))[-limit:]
        off = f"   (bar stamps are broker time, UTC{offset_h:+.0f})" if offset_h else ""
        head(f"{sym}   —   {len(sigs)} setup(s){off}")
        for s in sigs:
            total += 1
            blocked = SWING.blocked_by(s)
            p = _plan(trader, b, s, acc)
            age, mins = "", None
            try:
                bt = datetime.fromisoformat(str(s["ts"])).replace(
                    tzinfo=timezone.utc)
                mins = (now - bt).total_seconds() / 60 + offset_h * 60
                age = f"{mins/60:.1f}h ago" if mins > 90 else f"{mins:.0f}m ago"
            except (ValueError, KeyError):
                pass
            # The plan's entry is the LIVE tick. For a setup that formed hours
            # ago that is not the trade that was available then, and reading it
            # as one is the easiest mistake this view invites.
            stale = mins is not None and mins > 30

            verdict = "WOULD FIRE" if not blocked else "rejected"
            if not blocked:
                live_n += 1
            print(f"\n  {s['direction'].upper():<5s} {s['kind'].upper():<6s} "
                  f"{str(s['ts'])[:16].replace('T', ' ')}  {age:<9s} -> {verdict}")

            if p:
                d = 2 if p["pip"] >= 0.01 else 5
                rr = TARGET_R
                note = "  <- priced at the LIVE tick, not this bar" if stale else ""
                print(f"    entry   {p['price']:.{d}f}"
                      f"    stop {p['sl']:.{d}f}"
                      f"    target {p['tp']:.{d}f}   ({rr:.0f}R){note}")
                print(f"    risk    {p['stop_pips']:.1f} pips"
                      f"    ->  target {p['target_pips']:.1f} pips")
                if p["lot"] > 0:
                    print(f"    size    {p['lot']:.2f} lots ({p['size_reason']})"
                          f"   risking ${p['risk_usd']:,.2f}"
                          f"   to make ${p['reward_usd']:,.2f}")
                else:
                    print(f"    size    NO POSITION — {p['size_reason']}")

            print(f"    level   {s['touches']} touches / {s['respects']} respects"
                  f" ({s['respect_rate']*100:.0f}%)   age {s['level_age']} bars"
                  f"   {s['n_members']} member(s)")
            pats = ", ".join(s.get("patterns") or []) or "none"
            print(f"    leg     {s['leg_atr']:.2f} ATR   pullback {s['pullback_bars']} bars"
                  f"   patterns: {pats}")
            print(f"    htf     H4 {s['h4_dir']:+d}  H1 {s['h1_dir']:+d}"
                  f"  stack {s['htf_stack']}   h4_pullback {s['h4_pullback']:.2f}")

            gates = " | ".join(
                f"{v['name'].split()[0]} {fmt_val(v['value'])} "
                f"{'ok' if v['passed'] else 'FAIL'}" for v in SWING.verdict(s))
            print(f"    gates   {gates}")
            if blocked:
                misses = [f"{v['name']} (miss {v['margin']:+.3f})"
                          if v["margin"] is not None else v["name"]
                          for v in SWING.verdict(s) if not v["passed"]]
                print(f"    BLOCKED {'; '.join(misses)}")

            # what actually happened after — hindsight, not a forecast
            if s.get("outcome") and s.get("mfe_r") is not None:
                print(f"    since   MFE {s['mfe_r']:+.2f}R  MAE {s['mae_r']:+.2f}R"
                      f"  -> {s['outcome']} at {s['r']:+.2f}R"
                      f" ({s['exit_reason']}, 1h sim)")

    print()
    print(BAR * 78)
    print(f"  {total} setup(s) on the board, {live_n} passing every gate.")
    if not in_window:
        print("  Trading window is CLOSED — none of these can be taken now.")
    print("  'since' is HINDSIGHT from bars that have already printed, on a")
    print("  1-hour simulated hold. It is not a forecast for the live target.")
    b.shutdown()


# ── zones mode — where a setup CAN form, before it does ─────────────────────

def live_levels(broker, symbol: str, near: int = 10,
                reach: float = 6.0) -> dict | None:
    """Every level alive right now, nearest to price first.

    `--setups` is retrospective: it lists setups that have already formed. This
    is the question that comes before it — *where would one form?* A setup in
    this system needs a level to break and then be retested, so the levels
    currently alive are the complete set of places a setup can appear. Nothing
    can trigger anywhere else.

    Levels come from the same discovery pipeline the trader runs, on the same
    live series, so these are the exact prices its own next candidate would be
    built from — not a separate indicator drawn alongside it.
    """
    from core import majors as MJ
    from core.context import Context
    from core.discover import build_points, level_tracks
    from live import signals as SIG

    series = SIG.fetch_series(broker, symbol)
    if series is None:
        return None
    bars = series["M5"]
    ctx = Context(series, base="M5")
    atr = ctx.scales["M5"].atr
    pts = build_points(series, ctx, "M5")
    # reach_mult bounds how far from price discovery bothers clustering.
    # The default of 6 keeps the trader fast, but it means distant levels
    # are absent rather than merely far away, which reads as 'the tool is
    # missing levels'. Raising it here costs a slower scan and nothing else.
    tr = level_tracks(pts, ctx, reach_mult=reach)
    if len(tr) == 0:
        return {"levels": [], "price": float(bars.close[-1]),
                "atr": float(atr[-1]), "pip": bars.pip}

    i = len(bars) - 1
    mj = MJ.measure(bars, tr, atr)
    price = float(bars.close[i])
    a = float(atr[i])
    pip = bars.pip
    major = MJ.is_major(mj, min_touches=1)   # the mask features.py actually uses

    # Re-cluster at THIS bar to recover each level's composition. Tracks carry
    # only decision/magnet counts; the Cluster objects carry `sources`, which is
    # the audit trail back to what made the level — and "what is this level"
    # is the question being asked here.
    from core.cluster import cluster_points
    prox = ctx.threshold("proximity")
    cap = ctx.threshold("cluster_width_cap")
    tol = ctx.threshold("cluster_tolerance")
    act = pts.near(i, price, float(prox[i]) * reach)
    clusters = []
    if len(act):
        for c in cluster_points(act, float(cap[i]),
                                dedup=float(tol[i]) * 0.25):
            srcs = [f"{act.source[m]}({act.tf[m]})" for m in c.members]
            clusters.append((float(c.price), {
                "sources": srcs,
                # _FAMILY is the taxonomy majors.py counts categories with;
                # deriving families any other way would let this view disagree
                # with the `categories` column beside it.
                "families": sorted({MJ._FAMILY.get(str(act.source[m]),
                                                   str(act.source[m]))
                                    for m in c.members}),
                "n_decision": c.n_decision, "n_magnet": c.n_magnet,
                "magnet_share": c.magnet_share,
            }))

    def composition(level_price: float, width: float) -> dict:
        """Nearest cluster to this track, or {} if none is close enough.

        Tracks are stitched from clusterings taken every 12 bars, so a track's
        price need not appear in a clustering recomputed at the current bar —
        an exact-key lookup silently loses the composition and the level then
        reads as "made of ?". Matching on proximity keeps the audit trail.
        """
        if not clusters:
            return {}
        tolr = max(width, float(cap[i]) * 0.5, a * 0.05)
        best, gap = None, None
        for cp, meta in clusters:
            g = abs(cp - level_price)
            if gap is None or g < gap:
                best, gap = meta, g
        return best if (gap is not None and gap <= tolr) else {}

    alive = (tr.born <= i) & (tr.dead > i)
    rows = []
    for j in np.nonzero(alive)[0]:
        lp = float(tr.price[j])
        d = lp - price
        t, r = int(mj.touches[j]), int(mj.respects[j])
        k = composition(lp, float(tr.width[j]))
        rows.append({
            "price": lp,
            "dist_price": d,
            "dist_atr": d / a if a else 0.0,
            "dist_pips": d / pip,
            "above": d > 0,
            "touches": t, "respects": r,
            "respect_rate": (r / t) if t else 0.0,
            "width_pips": float(tr.width[j]) / pip,
            "age": int(mj.age[j]),
            "members": int(tr.n_members[j]),
            "categories": int(mj.categories[j]),
            "magnet_share": k.get("magnet_share", float(tr.magnet_share[j])),
            "is_major": bool(major[j]),
            "tf_rank": int(mj.tf_rank[j]),
            "sources": k.get("sources", []),
            "families": k.get("families", []),
            "n_decision": k.get("n_decision", int(tr.n_decision[j])),
            "n_magnet": k.get("n_magnet", int(tr.n_magnet[j])),
        })
    rows.sort(key=lambda x: abs(x["dist_atr"]))
    return {"levels": rows[:near], "price": price, "atr": a, "pip": pip,
            "n_alive": int(alive.sum()),
            "n_major": int((alive & major).sum())}


def report_zones(symbols: list, near: int = 8, reach: float = 6.0):
    """Where a setup can form on each symbol, before one exists."""
    from core.contracts import CONTRACTS
    from live.broker import Broker
    from live.trader import STOP_ATR, TARGET_R, Trader

    b = Broker()
    try:
        acc = b.connect_readonly()
    except Exception as e:
        print("Cannot reach MetaTrader 5:", e)
        return
    trader = Trader(b, dry_run=True)
    now = datetime.now(timezone.utc)
    in_window = TRADE_FROM_HOUR <= now.hour < FLAT_BY_HOUR

    head("LEVELS IN PLAY — where a setup can form")
    print(f"  {now:%Y-%m-%d %H:%M:%S} UTC   equity ${acc.equity:,.2f}   "
          f"window {'OPEN' if in_window else 'CLOSED'}")
    print("  A setup needs one of these levels to BREAK, then be retested.")
    print("  Nothing can trigger at a price that is not on this list.")

    for sym in symbols:
        if not b.ensure_symbol(sym):
            continue
        try:
            g = gate_state(b, sym)
            lv = live_levels(b, sym, near=near, reach=reach)
        except Exception as e:
            print(f"\n{sym}: {type(e).__name__}: {e}")
            continue
        if lv is None or g is None:
            print(f"\n{sym}: not enough history")
            continue

        c = CONTRACTS[sym]
        d = 2 if c.pip >= 0.01 else 5
        stop_pips = STOP_ATR * lv["atr"] / lv["pip"]
        lot, _ = trader.size(sym, acc, stop_pips, lv["price"])
        risk_usd = lot * stop_pips * c.usd_per_pip(lv["price"]) if lot > 0 else 0

        blocked = [v["name"] for v in SWING.verdict(g) if not v["passed"]]
        shown, total_alive = len(lv["levels"]), lv["n_alive"]
        trunc = "" if shown >= total_alive else f", showing {shown} nearest"
        head(f"{sym}   {lv['price']:.{d}f}   "
             f"ATR {lv['atr'] / lv['pip']:.1f}p   "
             f"{total_alive} levels alive{trunc}")
        if blocked:
            print(f"  GATES BLOCKED: {', '.join(blocked)}")
            print(f"  -> a setup at any level below would be REJECTED right now")
        else:
            print(f"  GATES ALL OPEN -> a setup at any level below would be TAKEN")
        print(f"  if one triggers: stop ~{stop_pips:.0f}p, target {TARGET_R:.0f}R "
              f"~{stop_pips * TARGET_R:.0f}p, {lot:.2f} lots, risk ~${risk_usd:,.0f}")

        if not lv["levels"]:
            print("  no levels alive — nothing to watch on this symbol")
            continue

        def line(r):
            """One level, read the way a trader reads it off a chart."""
            tag = "MAJOR" if r["is_major"] else "minor"
            tr_s = f"{r['touches']}t/{r['respects']}r" if r["touches"] else "untested"
            # a track can outlive the points that formed it: the swing gets
            # broken, the gap fills. The level is still real and still tested,
            # but its audit trail has expired - say so rather than print "?".
            made = "+".join(sorted(set(r["families"]))) or "(sources expired)"
            kind = ("magnet" if r["magnet_share"] > 0.5 else "decision")
            hold = f"{r['respect_rate'] * 100:.0f}%" if r["touches"] else "  -"
            return (f"  {r['price']:>11.{d}f}  {r['dist_atr']:>+6.1f}A "
                    f"{r['dist_pips']:>+8.0f}p  {tag:<5s} {tr_s:<8s}"
                    f" {hold:>4s}  {kind:<8s} {made}")

        above = [r for r in lv["levels"] if r["above"]]
        below = [r for r in lv["levels"] if not r["above"]]
        above.sort(key=lambda x: x["dist_atr"], reverse=True)
        below.sort(key=lambda x: x["dist_atr"], reverse=True)

        print()
        print(f"  {'price':>11s}  {'dist':>6s} {'':>8s}  {'grade':<5s} "
              f"{'tested':<8s} {'hold':>4s}  {'type':<8s} made of")
        print(f"  {BAR * 11}  {BAR * 6} {BAR * 8}  {BAR * 5} {BAR * 8} "
              f"{BAR * 4}  {BAR * 8} {BAR * 24}")
        for r in above:
            print(line(r))
        print(f"  {'>>> ' + format(lv['price'], '.' + str(d) + 'f'):>11s}"
              f"  {'':>6s} {'':>8s}  <<< PRICE IS HERE")
        for r in below:
            print(line(r))

        # the two that matter — nearest MAJOR level each side
        up = min((r for r in above if r["is_major"]),
                 key=lambda x: x["dist_atr"], default=None)
        dn = max((r for r in below if r["is_major"]),
                 key=lambda x: x["dist_atr"], default=None)
        print()
        print("  WHAT TO WATCH")
        for r, way in ((up, "UP"), (dn, "DOWN")):
            if r is None:
                print(f"    {way:<4s} no major level in range")
                continue
            side = "BUY" if way == "UP" else "SELL"
            print(f"    {way:<4s} {r['price']:.{d}f}"
                  f"  ({abs(r['dist_pips']):.0f}p away, {abs(r['dist_atr']):.1f} ATR)")
            print(f"         {r['touches']} tests, {r['respect_rate'] * 100:.0f}% held"
                  f"   built from "
                  f"{', '.join(r['sources'][:5]) or 'sources expired, level still tracked'}")
            print(f"         needs: close through it -> impulse leg -> retrace"
                  f" into 0.5-0.618 -> {side}")

    print()
    print(BAR * 78)
    print("  Levels are from the trader's own discovery pipeline on the same")
    print("  live series, so these are the prices its next candidate is built")
    print("  from. Stop/target/lot are ESTIMATES at the current ATR — the real")
    print("  stop is structural and is set from the leg the break actually makes.")
    print()
    print(f"  Discovery radius is --reach {reach:g} (the trader's own default is 6).")
    print("  Raising it does two things, and the second is easy to miss:")
    print("    * distant levels appear at all")
    print("    * NEAR levels gain history — a level is only tracked once price")
    print("      comes within reach, so touch counts are UNDER-reported at a")
    print("      small radius. The same level can read 3/3 at reach 6 and")
    print("      20/20 at reach 40.")
    print("  For the full picture:  --zones --all --reach 40")
    b.shutdown()


# ── cli ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Explain what the live trader is thinking.")
    ap.add_argument("--now", action="store_true",
                    help="connect read-only and show the CURRENT gate state")
    ap.add_argument("--setups", action="store_true",
                    help="every setup on the board, with entry/stop/target")
    ap.add_argument("--zones", action="store_true",
                    help="where a setup CAN form: levels alive right now")
    ap.add_argument("--near", type=int, default=8,
                    help="--zones: how many nearest levels per symbol")
    ap.add_argument("--all", action="store_true",
                    help="--zones: every level alive, not just the nearest")
    ap.add_argument("--reach", type=float, default=6.0,
                    help="--zones: discovery radius in proximity units "
                         "(default 6; raise to find distant levels)")
    ap.add_argument("--watch", type=int, metavar="SEC",
                    help="re-run every SEC seconds until interrupted")
    ap.add_argument("--limit", type=int, default=6,
                    help="--setups: most recent N setups per symbol")
    ap.add_argument("--symbol", action="append",
                    help="restrict to one symbol (repeatable)")
    ap.add_argument("--day", help="journal mode: only this UTC day, YYYY-MM-DD")
    ap.add_argument("--lookback", type=int, default=300,
                    help="--now: how many M5 bars to scan for candidates")
    a = ap.parse_args(argv)

    def once():
        if a.zones:
            report_zones(a.symbol or SYMBOLS,
                         near=10**6 if a.all else a.near,
                         reach=a.reach)
        elif a.setups:
            report_setups(a.symbol or SYMBOLS, lookback=a.lookback,
                          limit=a.limit)
        elif a.now:
            report_now(a.symbol or SYMBOLS, lookback=a.lookback)
        else:
            rows = load()
            if a.symbol:
                keep = set(a.symbol)
                rows = [r for r in rows
                        if r.get("symbol") in keep or "symbol" not in r]
            report_journal(rows, day=a.day)

    if not a.watch:
        once()
        return 0

    import time
    try:
        while True:
            # scroll rather than clear: the point of watching is to see what
            # CHANGED, and a screen wipe destroys exactly that.
            print("\n" + "=" * 78)
            print(f"= refresh {datetime.now(timezone.utc):%H:%M:%S} UTC")
            print("=" * 78)
            once()
            time.sleep(max(5, a.watch))
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
