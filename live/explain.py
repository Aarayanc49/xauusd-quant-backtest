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

## The two modes

    python -m live.explain            # what has it been thinking (journal)
    python -m live.explain --now      # what is it thinking RIGHT NOW (live)

`--now` is the one that answers the question directly. It connects read-only,
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


# ── cli ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Explain what the live trader is thinking.")
    ap.add_argument("--now", action="store_true",
                    help="connect read-only and show the CURRENT gate state")
    ap.add_argument("--symbol", action="append",
                    help="restrict to one symbol (repeatable)")
    ap.add_argument("--day", help="journal mode: only this UTC day, YYYY-MM-DD")
    ap.add_argument("--lookback", type=int, default=300,
                    help="--now: how many M5 bars to scan for candidates")
    a = ap.parse_args(argv)

    if a.now:
        report_now(a.symbol or SYMBOLS, lookback=a.lookback)
    else:
        rows = load()
        if a.symbol:
            keep = set(a.symbol)
            rows = [r for r in rows
                    if r.get("symbol") in keep or "symbol" not in r]
        report_journal(rows, day=a.day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
