"""Live cockpit — a window that watches the trader without touching it.

`live/explain.py` answers questions one command at a time. This is the thing
you leave open on a second monitor while `live/trader.py` runs: levels, gate
state and setups for every symbol, refreshing on their own.

    python -m live.cockpit
    python -m live.cockpit --refresh 15 --reach 40

## Read-only, structurally

The cockpit attaches through `Broker.connect_readonly` and never imports an
order path. It cannot place, modify or close anything — not by policy but
because nothing reachable from here can. Running it against a live account is
safe in the only sense that matters: the worst it can do is show you a stale
number.

## Why a thread, and exactly one

The MetaTrader5 module is a thin wrapper over a single terminal connection and
is not thread-safe. Every call in this file happens on ONE background worker;
the Tk main loop only ever reads finished snapshots off a queue. That split is
what keeps the window responsive while a level rebuild takes several seconds,
and it is also why there is no "refresh" button that calls MT5 directly — it
sets a flag the worker picks up instead.

Symbols are published one at a time as they finish rather than as a batch, so
the board fills in progressively instead of freezing for a full cycle. On six
symbols at a wide discovery radius a full sweep is tens of seconds, and a UI
that shows nothing until all of it lands reads as hung.

## What a dot means

Each symbol carries a status dot, and it answers the question the terminal
tools took four commands to answer:

    green   every gate open — a setup here would be TAKEN
    amber   one gate blocking — close, worth watching
    red     two or more blocking — nothing can trade here for a while
    grey    no data yet, or the symbol errored
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy import SWING  # noqa: E402
from live.explain import (JOURNAL, SYMBOLS, TRADE_FROM_HOUR,  # noqa: E402
                          FLAT_BY_HOUR, gate_state, live_levels)

# ── palette ─────────────────────────────────────────────────────────────────
BG = "#12171a"
PANEL = "#1a2126"
LINE = "#2b353c"
FG = "#dbe4e8"
DIM = "#7d8f98"
OK = "#4caf7d"
WARN = "#d9a648"
BAD = "#d0785d"
ACCENT = "#5aa9d6"
MONO = ("Consolas", 10)
MONO_S = ("Consolas", 9)
UI = ("Segoe UI", 10)
UI_B = ("Segoe UI", 10, "bold")


@dataclass
class Snap:
    symbol: str
    ok: bool = False
    err: str = ""
    price: float = 0.0
    atr_pips: float = 0.0
    h4: int = 0
    h1: int = 0
    session: str = ""
    gates: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    n_alive: int = 0
    setups: list = field(default_factory=list)
    stamp: str = ""

    @property
    def n_blocked(self) -> int:
        return sum(1 for g in self.gates if not g["passed"])


# ── worker: the only thread that talks to MT5 ───────────────────────────────

class Poller(threading.Thread):
    daemon = True

    def __init__(self, out: queue.Queue, symbols, refresh: int,
                 reach: float, near: int, lookback: int):
        super().__init__(name="mt5-poller")
        self.out, self.symbols = out, symbols
        self.refresh, self.reach = refresh, reach
        self.near, self.lookback = near, lookback
        self.stop = threading.Event()
        self.kick = threading.Event()

    def run(self):
        from live.broker import Broker
        from live import signals as SIG
        from live.explain import _plan
        from live.trader import Trader

        b = Broker()
        try:
            acc = b.connect_readonly()
        except Exception as e:
            self.out.put(("fatal", f"{type(e).__name__}: {e}"))
            return
        trader = Trader(b, dry_run=True)
        self.out.put(("account", acc))

        while not self.stop.is_set():
            try:
                acc = b.account()
                if acc is not None:
                    self.out.put(("account", acc))
            except Exception:
                pass

            for sym in self.symbols:
                if self.stop.is_set():
                    break
                s = Snap(symbol=sym,
                         stamp=datetime.now(timezone.utc).strftime("%H:%M:%S"))
                try:
                    if not b.ensure_symbol(sym):
                        raise RuntimeError("symbol not available")
                    g = gate_state(b, sym)
                    if g is None:
                        raise RuntimeError("not enough history")
                    s.price = g["price"] if "price" in g else 0.0
                    s.atr_pips = 0.0
                    s.h4, s.h1 = g["h4_dir"], g["h1_dir"]
                    s.session = g["session"]
                    s.gates = SWING.verdict(g)

                    lv = live_levels(b, sym, near=self.near, reach=self.reach)
                    if lv:
                        s.price = lv["price"]
                        s.atr_pips = lv["atr"] / lv["pip"]
                        s.levels = lv["levels"]
                        s.n_alive = lv["n_alive"]

                    try:
                        sigs = SIG.scan(b, sym, lookback_bars=self.lookback)
                    except Exception:
                        sigs = []
                    acc2 = b.account()
                    for c in sorted(sigs,
                                    key=lambda x: str(x.get("ts", "")))[-5:]:
                        p = _plan(trader, b, c, acc2) if acc2 else None
                        s.setups.append({
                            "ts": str(c.get("ts", ""))[:16].replace("T", " "),
                            "dir": c["direction"], "kind": c.get("kind", ""),
                            "blocked": SWING.blocked_by(c),
                            "plan": p,
                        })
                    s.ok = True
                except Exception as e:
                    s.err = f"{type(e).__name__}: {e}"
                self.out.put(("snap", s))

            # sleep in slices so a refresh request or quit lands promptly
            for _ in range(max(1, self.refresh) * 10):
                if self.stop.is_set() or self.kick.is_set():
                    self.kick.clear()
                    break
                time.sleep(0.1)


# ── UI ──────────────────────────────────────────────────────────────────────

class Cockpit:
    def __init__(self, root, args):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = root
        self.args = args
        self.q: queue.Queue = queue.Queue()
        self.snaps: dict = {}
        self.current = args.symbol[0] if args.symbol else SYMBOLS[0]
        self.symbols = args.symbol or SYMBOLS
        self.acc = None

        root.title("Trading Cockpit — read only")
        root.configure(bg=BG)
        root.geometry("1180x780")

        st = ttk.Style()
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("TLabel", background=BG, foreground=FG, font=UI)
        st.configure("Dim.TLabel", background=BG, foreground=DIM, font=UI)
        st.configure("Panel.TLabel", background=PANEL, foreground=FG, font=UI)
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                     foreground=FG, font=MONO_S, rowheight=20,
                     borderwidth=0)
        st.configure("Treeview.Heading", background=LINE, foreground=DIM,
                     font=("Segoe UI", 9, "bold"))
        st.map("Treeview", background=[("selected", "#26333c")])

        self._build()
        self.poller = Poller(self.q, self.symbols, args.refresh, args.reach,
                             args.near, args.lookback)
        self.poller.start()
        self.root.after(150, self._drain)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ── layout ──────────────────────────────────────────────────────────────

    def _build(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=(10, 6))
        self.l_title = ttk.Label(top, text="COCKPIT", font=("Segoe UI", 13, "bold"))
        self.l_title.pack(side="left")
        self.l_acct = ttk.Label(top, text="connecting…", style="Dim.TLabel")
        self.l_acct.pack(side="left", padx=16)
        self.l_trader = ttk.Label(top, text="", style="Dim.TLabel")
        self.l_trader.pack(side="right")
        self.b_ref = tk.Button(top, text="Refresh", command=self._kick,
                               bg=LINE, fg=FG, relief="flat", font=UI,
                               activebackground=ACCENT, cursor="hand2")
        self.b_ref.pack(side="right", padx=10)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=6)

        # left rail — symbols with status dots
        rail = ttk.Frame(body, style="Panel.TFrame", width=190)
        rail.pack(side="left", fill="y")
        rail.pack_propagate(False)
        ttk.Label(rail, text="  SYMBOLS", style="Panel.TLabel",
                  foreground=DIM, font=("Segoe UI", 9, "bold")).pack(
                      anchor="w", pady=(10, 6))
        self.rail_btns = {}
        for s in self.symbols:
            b = tk.Label(rail, text=f"  ●  {s}", bg=PANEL, fg=DIM,
                         font=UI, anchor="w", cursor="hand2", padx=8, pady=7)
            b.pack(fill="x")
            b.bind("<Button-1>", lambda _e, sym=s: self._select(sym))
            self.rail_btns[s] = b

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.l_head = ttk.Label(right, text="", font=("Consolas", 15, "bold"))
        self.l_head.pack(anchor="w")
        self.l_sub = ttk.Label(right, text="", style="Dim.TLabel", font=MONO)
        self.l_sub.pack(anchor="w", pady=(2, 8))

        self.gates = ttk.Frame(right)
        self.gates.pack(fill="x", pady=(0, 10))

        ttk.Label(right, text="LEVELS", style="Dim.TLabel",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w")
        cols = ("price", "dist", "grade", "tested", "hold", "type", "made")
        self.tree = ttk.Treeview(right, columns=cols, show="headings",
                                 height=15, selectmode="none")
        for c, w, t in (("price", 100, "price"), ("dist", 70, "dist"),
                        ("grade", 60, "grade"), ("tested", 80, "tested"),
                        ("hold", 55, "hold"), ("type", 75, "type"),
                        ("made", 250, "made of")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w")
        self.tree.tag_configure("above", foreground="#9fb6c2")
        self.tree.tag_configure("below", foreground="#9fb6c2")
        self.tree.tag_configure("major", foreground=FG)
        self.tree.tag_configure("here", background="#26333c",
                                foreground=ACCENT, font=("Consolas", 9, "bold"))
        self.tree.pack(fill="both", expand=True, pady=(4, 10))

        ttk.Label(right, text="SETUPS ON THE BOARD", style="Dim.TLabel",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.txt = tk.Text(right, height=9, bg=PANEL, fg=FG, font=MONO_S,
                           relief="flat", wrap="none", padx=10, pady=8,
                           insertbackground=FG)
        self.txt.pack(fill="both", expand=False, pady=(4, 0))
        self.txt.tag_configure("fire", foreground=OK)
        self.txt.tag_configure("rej", foreground=DIM)
        self.txt.tag_configure("hd", foreground=ACCENT)
        self.txt.configure(state="disabled")

    # ── events ──────────────────────────────────────────────────────────────

    def _kick(self):
        self.poller.kick.set()

    def _select(self, sym):
        self.current = sym
        self._paint()

    def _quit(self):
        self.poller.stop.set()
        self.root.destroy()

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "snap":
                    self.snaps[payload.symbol] = payload
                elif kind == "account":
                    self.acc = payload
                elif kind == "fatal":
                    self.l_acct.configure(text=payload, foreground=BAD)
        except queue.Empty:
            pass
        self._paint()
        self.root.after(400, self._drain)

    # ── render ──────────────────────────────────────────────────────────────

    def _paint(self):
        now = datetime.now(timezone.utc)
        inwin = TRADE_FROM_HOUR <= now.hour < FLAT_BY_HOUR
        self.l_title.configure(
            text=f"COCKPIT   {now:%H:%M:%S} UTC   "
                 f"{'WINDOW OPEN' if inwin else 'WINDOW CLOSED'}",
            foreground=FG if inwin else WARN)
        if self.acc is not None:
            self.l_acct.configure(
                text=f"{self.acc.login}@{self.acc.server}   "
                     f"equity ${self.acc.equity:,.2f}", foreground=DIM)

        # is the trader itself alive? the journal's mtime is the cheapest
        # honest signal, and needs no coupling to its process.
        try:
            age = (time.time() - os.path.getmtime(JOURNAL)) / 60
            txt, col = ((f"trader wrote {age:.0f}m ago", DIM) if age < 60
                        else (f"trader silent {age/60:.1f}h", BAD))
        except OSError:
            txt, col = ("no journal", BAD)
        self.l_trader.configure(text=txt, foreground=col)

        for sym, btn in self.rail_btns.items():
            s = self.snaps.get(sym)
            if s is None or not s.ok:
                col = DIM
            elif s.n_blocked == 0:
                col = OK
            elif s.n_blocked == 1:
                col = WARN
            else:
                col = BAD
            sel = sym == self.current
            btn.configure(fg=col, bg="#26333c" if sel else PANEL,
                          font=UI_B if sel else UI)

        s = self.snaps.get(self.current)
        if s is None:
            self.l_head.configure(text=f"{self.current}   loading…")
            return
        if not s.ok:
            self.l_head.configure(text=f"{self.current}   unavailable",
                                  foreground=BAD)
            self.l_sub.configure(text=s.err)
            return

        d = 2 if s.price > 20 else 5
        arrow = {1: "up", -1: "down", 0: "flat"}
        self.l_head.configure(text=f"{s.symbol}   {s.price:.{d}f}",
                              foreground=FG)
        self.l_sub.configure(
            text=f"ATR {s.atr_pips:.1f}p   H4 {arrow[s.h4]}  H1 {arrow[s.h1]}"
                 f"   {s.session}   {s.n_alive} levels alive   @{s.stamp}")

        for w in self.gates.winfo_children():
            w.destroy()
        for g in s.gates:
            col = OK if g["passed"] else BAD
            val = g["value"]
            val = f"{val:.2f}" if isinstance(val, float) else str(val)
            miss = ("" if g["margin"] is None or g["passed"]
                    else f"  miss {g['margin']:+.2f}")
            self.tk.Label(self.gates,
                          text=f" {g['name'].split()[0]} {val}{miss} ",
                          bg=PANEL, fg=col, font=MONO_S, padx=8, pady=4
                          ).pack(side="left", padx=(0, 6))

        self.tree.delete(*self.tree.get_children())
        above = [r for r in s.levels if r["above"]]
        below = [r for r in s.levels if not r["above"]]
        above.sort(key=lambda x: x["dist_atr"], reverse=True)
        below.sort(key=lambda x: x["dist_atr"], reverse=True)

        def row(r):
            made = "+".join(sorted(set(r["families"]))) or "(sources expired)"
            tested = f"{r['touches']}t/{r['respects']}r" if r["touches"] else "untested"
            hold = f"{r['respect_rate']*100:.0f}%" if r["touches"] else "-"
            kind = "magnet" if r["magnet_share"] > 0.5 else "decision"
            self.tree.insert("", "end", tags=("major" if r["is_major"] else "above",),
                             values=(f"{r['price']:.{d}f}",
                                     f"{r['dist_atr']:+.1f}A",
                                     "MAJOR" if r["is_major"] else "minor",
                                     tested, hold, kind, made))

        for r in above:
            row(r)
        self.tree.insert("", "end", tags=("here",),
                         values=(f"{s.price:.{d}f}", "", "<<<", "PRICE", "", "", ""))
        for r in below:
            row(r)

        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        if not s.setups:
            self.txt.insert("end", "  no candidate formed in the lookback window\n",
                            "rej")
        for c in s.setups:
            fire = not c["blocked"]
            self.txt.insert("end",
                            f"  {c['dir'].upper():<5s} {c['kind'].upper():<5s} "
                            f"{c['ts']}  ", "hd")
            self.txt.insert("end",
                            "WOULD FIRE\n" if fire else
                            f"rejected: {', '.join(b.split()[0] for b in c['blocked'])}\n",
                            "fire" if fire else "rej")
            p = c["plan"]
            if p:
                self.txt.insert(
                    "end",
                    f"        entry {p['price']:.{d}f}  stop {p['sl']:.{d}f}"
                    f"  target {p['tp']:.{d}f}   {p['stop_pips']:.0f}p risk"
                    f"   {p['lot']:.2f} lots  ${p['risk_usd']:,.0f}\n", "rej")
        self.txt.configure(state="disabled")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Live read-only trading cockpit.")
    ap.add_argument("--refresh", type=int, default=30,
                    help="seconds between full sweeps (default 30)")
    ap.add_argument("--reach", type=float, default=40.0,
                    help="level discovery radius (default 40)")
    ap.add_argument("--near", type=int, default=14,
                    help="levels shown per symbol")
    ap.add_argument("--lookback", type=int, default=300,
                    help="M5 bars scanned for setups")
    ap.add_argument("--symbol", action="append", help="restrict symbols")
    a = ap.parse_args(argv)

    try:
        import tkinter as tk
    except ImportError:
        print("tkinter is not available in this Python build.")
        print("Use the terminal view instead:  python -m live.explain --zones")
        return 1

    root = tk.Tk()
    Cockpit(root, a)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
