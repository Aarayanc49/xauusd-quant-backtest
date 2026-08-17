"""Fill the bar store — from MT5 directly, or from the old tree's CSV exports.

MT5 path requires the terminal running and logged in, and
`Tools -> Options -> Charts -> Max bars in chart` set to 100000000. Without that
setting M1 reaches ~0.3 years instead of 22; it is the single most important
piece of configuration in the project.

    python -m research.fetch mt5 --symbols XAUUSD EURUSD --from 2004-01-01
    python -m research.fetch csv "C:/.../backtest/data"   # bootstrap, no terminal

MT5's copy_rates_range refuses to return more than ~a few million bars at once
and simply gives back None when it runs out of room, so M1 is pulled in yearly
chunks and stitched. Chunks are validated for overlap rather than trusted: the
terminal will happily return a partial range after a reconnect.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import store  # noqa: E402

DEFAULT_SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
DEFAULT_TFS = ["M1", "M5", "M15", "H1", "H4", "D1"]


# ── MT5 ─────────────────────────────────────────────────────────────────────

def _mt5_timeframe(mt5, tf: str):
    return {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    }[tf]


def _resolve_symbol(mt5, want: str) -> str | None:
    """Brokers suffix symbols (XAUUSD.r, EURUSDm, XAUUSD-ECN). Match on prefix so
    the caller can ask for the plain name and still get the broker's instrument."""
    if mt5.symbol_info(want) is not None:
        return want
    for s in (mt5.symbols_get() or []):
        if s.name.upper().startswith(want.upper()):
            return s.name
    return None


def fetch_mt5(symbols, tfs, start: datetime, end: datetime | None = None,
              base: str | None = None) -> None:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()} — "
                           f"is the terminal running and logged in?")
    try:
        end = end or datetime.now(timezone.utc)
        for want in symbols:
            sym = _resolve_symbol(mt5, want)
            if sym is None:
                print(f"  {want:8} SKIP — not offered by this broker")
                continue
            info = mt5.symbol_info(sym)
            if not info.visible:
                mt5.symbol_select(sym, True)
                info = mt5.symbol_info(sym)
            print(f"\n{want} (broker: {sym})  digits={info.digits} point={info.point}")

            for tf in tfs:
                arrs = _fetch_series(mt5, sym, tf, start, end)
                if arrs is None:
                    print(f"  {tf:4} — no data")
                    continue
                meta = store.write(want, tf, arrs, info.digits, info.point, base)
                print(f"  {tf:4} {meta.n:>10,} bars  "
                      f"{np.datetime64(meta.first, 's')} .. "
                      f"{np.datetime64(meta.last, 's')}  {meta.years:.2f}y")
    finally:
        mt5.shutdown()


def _fetch_series(mt5, sym: str, tf: str, start: datetime, end: datetime):
    """Year-chunked pull, stitched and de-duplicated."""
    tfc = _mt5_timeframe(mt5, tf)
    chunks = []
    # Coarse timeframes fit in one request; only the minute series needs chunking.
    span_days = 365 if tf in ("M1",) else 365 * 5 if tf in ("M5", "M15") else 365 * 30
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=span_days), end)
        r = mt5.copy_rates_range(sym, tfc, cur, nxt)
        if r is not None and len(r):
            chunks.append(r)
        cur = nxt
    if not chunks:
        return None

    r = np.concatenate(chunks)
    t = r["time"].astype(np.int64)
    order = np.argsort(t, kind="stable")
    r, t = r[order], t[order]
    keep = np.concatenate(([True], np.diff(t) > 0))     # drop chunk-boundary dupes
    r, t = r[keep], t[keep]

    return {
        "time": t,
        "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
        "volume": r["tick_volume"],
        # real_volume is zero on most retail feeds; spread is in POINTS.
        "spread": r["spread"],
    }


# ── CSV bootstrap ───────────────────────────────────────────────────────────

def fetch_csv(src_dir: str, base: str | None = None) -> None:
    """Import the old tree's `backtest/data/{SYMBOL}_{TF}.csv` exports.

    Lets the engine be built and tested before the terminal is available. These
    files carry only a ~70-day window, so they are a correctness fixture, never a
    basis for any P&L claim.
    """
    import csv as _csv

    files = [f for f in sorted(os.listdir(src_dir)) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"no CSVs in {src_dir}")

    for fn in files:
        stem = fn[:-4]
        if "_" not in stem:
            continue
        sym, tf = stem.rsplit("_", 1)
        if tf.upper() not in store.TF_MINUTES:
            continue
        cols: dict[str, list] = {k: [] for k in
                                 ("time", "open", "high", "low", "close", "volume", "spread")}
        with open(os.path.join(src_dir, fn), newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                cols["time"].append(
                    np.datetime64(row["time"].replace(" ", "T"), "s").astype("int64"))
                for k in ("open", "high", "low", "close"):
                    cols[k].append(float(row[k]))
                cols["volume"].append(float(row.get("tick_volume") or 0))
                cols["spread"].append(float(row.get("spread") or 0))

        t = np.asarray(cols["time"], dtype=np.int64)
        order = np.argsort(t, kind="stable")
        for k in cols:
            cols[k] = np.asarray(cols[k])[order]
        keep = np.concatenate(([True], np.diff(cols["time"]) > 0))
        for k in cols:
            cols[k] = cols[k][keep]

        # The CSVs carry no symbol spec. point = 0.01 for gold / 0.00001 for FX is
        # the 3-and-5-digit convention these exports came from.
        pip = store.pip_for(sym)
        point, digits = (0.01, 2) if pip >= 0.1 else (pip / 10.0, 5)
        meta = store.write(sym, tf, cols, digits, point, base)
        print(f"  {sym} {tf:4} {meta.n:>9,} bars  "
              f"{np.datetime64(meta.first, 's')} .. {np.datetime64(meta.last, 's')}")


# ── cli ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mt5", help="pull from a running MT5 terminal")
    m.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    m.add_argument("--tfs", nargs="+", default=DEFAULT_TFS)
    m.add_argument("--from", dest="start", default="2004-01-01")
    m.add_argument("--to", dest="end", default=None)

    c = sub.add_parser("csv", help="import the old tree's CSV exports")
    c.add_argument("src")

    sub.add_parser("list", help="show what the store already holds")

    a = p.parse_args(argv)
    if a.cmd == "mt5":
        start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
        end = (datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
               if a.end else None)
        fetch_mt5(a.symbols, a.tfs, start, end)
    elif a.cmd == "csv":
        fetch_csv(a.src)
    else:
        av = store.available()
        if not av:
            print("store is empty")
        for sym, tfs in av.items():
            print(f"\n{sym}")
            for tf, mt in tfs.items():
                print(f"  {tf:4} {mt.n:>10,}  {np.datetime64(mt.first, 's')} .. "
                      f"{np.datetime64(mt.last, 's')}  {mt.years:5.2f}y")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
