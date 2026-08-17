"""Bar store — mmap'd numpy columns, one directory per (symbol, timeframe).

Why not parquet/CSV: the research engine sweeps 22 years of M1 (7M bars) many
times per session. CSV parsing dominated the old harness; parquet needs pyarrow.
Plain `.npy` per column with `mmap_mode='r'` means the OS pages in only the
columns and date ranges a study actually touches, with no dependency and no
decode cost.

    data/XAUUSD/M1/time.npy      int64   epoch SECONDS, strictly increasing
                   open.npy      float64
                   high.npy      float64
                   low.npy       float64
                   close.npy     float64
                   volume.npy    float32  tick volume
                   spread.npy    float32  broker points (NOT pips)
                   meta.json     {symbol, timeframe, digits, point, n, first, last}

`spread` is stored in broker POINTS exactly as MT5 reports it. Converting to pips
is the caller's job (`Bars.spread_pips`) because the point/pip ratio is a symbol
property, and baking it in here would silently corrupt the column if the symbol
spec ever changed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

# Minutes per bar. The store never interpolates or resamples — every timeframe is
# fetched from the broker directly, because a resampled H4 does not agree with the
# broker's H4 on session boundaries and the whole point is to see what MT5 saw.
TF_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
}

_PRICE_COLS = ("open", "high", "low", "close")
_COLS = _PRICE_COLS + ("time", "volume", "spread")

# Pip size per symbol. A "pip" here is the operator-facing unit used for every
# threshold and every report: gold moves in 0.1 increments, FX majors in 0.0001.
PIP = {
    "XAUUSD": 0.1,
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
    "USDJPY": 0.01, "GBPJPY": 0.01, "EURJPY": 0.01,
    # Index CFDs quote in index points. One point is the natural unit — using an
    # FX-style 0.0001 on an instrument trading near 20,000 would make every
    # threshold and every reported figure wrong by six orders of magnitude.
    "USTEC": 1.0, "US30": 1.0, "US500": 1.0, "DE40": 1.0, "UK100": 1.0,
}


def pip_for(symbol: str) -> float:
    """Pip size for a symbol. Unknown symbols fall back to the FX-major default
    rather than guessing from digits — an unknown symbol should be added to PIP
    deliberately, not inferred at runtime."""
    return PIP.get(symbol.upper(), 0.0001)


def root() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def path_for(symbol: str, tf: str, base: str | None = None) -> str:
    return os.path.join(base or root(), symbol.upper(), tf.upper())


@dataclass(frozen=True)
class Meta:
    symbol: str
    timeframe: str
    digits: int
    point: float
    n: int
    first: int           # epoch seconds of the first bar OPEN
    last: int            # epoch seconds of the last bar OPEN

    @property
    def years(self) -> float:
        return (self.last - self.first) / (365.25 * 86400)


class Bars:
    """Read-only view of one (symbol, timeframe) series.

    Columns are memory-mapped: constructing a Bars is O(1) regardless of series
    length, and only the slices a study touches are ever paged in.

    Every array is aligned on the same index. `time[i]` is the bar's OPEN time;
    the bar is only fully known at `time[i] + TF_MINUTES*60`. Any code that needs
    "the last bar I could legally have seen at wall-clock t" must use
    `closed_upto(t)`, which accounts for that — indexing by open time alone is the
    classic one-bar lookahead and it is the single easiest way to fake an edge.
    """

    __slots__ = ("symbol", "tf", "meta", "time", "open", "high", "low",
                 "close", "volume", "spread", "pip", "_dir")

    def __init__(self, symbol: str, tf: str, base: str | None = None):
        self.symbol = symbol.upper()
        self.tf = tf.upper()
        self._dir = path_for(self.symbol, self.tf, base)
        if not os.path.isdir(self._dir):
            raise FileNotFoundError(
                f"no bar store at {self._dir} — run research/fetch.py first")
        with open(os.path.join(self._dir, "meta.json"), encoding="utf-8") as f:
            self.meta = Meta(**json.load(f))
        for c in _COLS:
            object.__setattr__(self, c,
                               np.load(os.path.join(self._dir, f"{c}.npy"), mmap_mode="r"))
        self.pip = pip_for(self.symbol)

    def __len__(self) -> int:
        return int(self.meta.n)

    def __repr__(self) -> str:
        return (f"<Bars {self.symbol} {self.tf} n={len(self):,} "
                f"{self.t_str(0)}..{self.t_str(len(self) - 1)}>")

    # ── time helpers ────────────────────────────────────────────────────────

    @property
    def bar_seconds(self) -> int:
        return TF_MINUTES[self.tf] * 60

    def t_str(self, i: int) -> str:
        return np.datetime64(int(self.time[i]), "s").astype("datetime64[m]").astype(str)

    def index_at(self, when) -> int:
        """Index of the bar whose OPEN is at or before `when`. -1 if before the
        series starts. `when` may be an epoch int, a datetime64, or an ISO str."""
        t = _epoch(when)
        return int(np.searchsorted(self.time, t, side="right")) - 1

    def closed_upto(self, when) -> int:
        """Index of the last bar fully CLOSED at `when` — the newest bar a
        decision made at `when` is allowed to see. This is the no-lookahead
        boundary; prefer it to `index_at` everywhere in the decision path."""
        t = _epoch(when)
        return int(np.searchsorted(self.time, t - self.bar_seconds, side="right")) - 1

    def slice_between(self, start, end) -> tuple[int, int]:
        """[lo, hi) index range covering bars whose OPEN falls in [start, end)."""
        lo = int(np.searchsorted(self.time, _epoch(start), side="left"))
        hi = int(np.searchsorted(self.time, _epoch(end), side="left"))
        return lo, hi

    # ── derived columns ─────────────────────────────────────────────────────

    def spread_pips(self, lo: int = 0, hi: int | None = None) -> np.ndarray:
        """Broker spread in pips over [lo, hi). MT5 reports spread in points;
        pips = points * point / pip."""
        hi = len(self) if hi is None else hi
        return np.asarray(self.spread[lo:hi], dtype=np.float64) * self.meta.point / self.pip

    def ohlc(self, lo: int = 0, hi: int | None = None) -> np.ndarray:
        """(n, 4) float64 copy of OHLC over [lo, hi). Copies — mmap views are
        read-only and most consumers want to compute on them."""
        hi = len(self) if hi is None else hi
        return np.stack([np.asarray(getattr(self, c)[lo:hi], dtype=np.float64)
                         for c in _PRICE_COLS], axis=1)

    def hour_utc(self, lo: int = 0, hi: int | None = None) -> np.ndarray:
        """UTC hour of each bar open. Note MT5 server time is usually UTC+2/+3 —
        the spread and session studies key on SERVER hour, so use
        `hour_server` when comparing against broker-quoted session behaviour."""
        hi = len(self) if hi is None else hi
        return ((np.asarray(self.time[lo:hi], dtype=np.int64) // 3600) % 24).astype(np.int8)

    def hour_server(self, offset_hours: int, lo: int = 0, hi: int | None = None) -> np.ndarray:
        hi = len(self) if hi is None else hi
        t = np.asarray(self.time[lo:hi], dtype=np.int64) + offset_hours * 3600
        return ((t // 3600) % 24).astype(np.int8)


def _epoch(when) -> int:
    if isinstance(when, (int, np.integer)):
        return int(when)
    if isinstance(when, float):
        return int(when)
    return int(np.datetime64(when, "s").astype("int64"))


# ── writing ─────────────────────────────────────────────────────────────────

def write(symbol: str, tf: str, arrays: dict, digits: int, point: float,
          base: str | None = None) -> Meta:
    """Persist a fetched series. `arrays` must carry every column in _COLS.

    Validates monotonicity and OHLC sanity before writing anything — a store that
    silently contains an out-of-order or high<low bar will produce a plausible,
    wrong backtest, and that is the most expensive kind of bug in this project.
    """
    d = path_for(symbol, tf, base)
    os.makedirs(d, exist_ok=True)

    t = np.asarray(arrays["time"], dtype=np.int64)
    if t.size == 0:
        raise ValueError(f"{symbol} {tf}: empty series")
    if not np.all(np.diff(t) > 0):
        bad = int(np.argmin(np.diff(t))) + 1
        raise ValueError(f"{symbol} {tf}: time not strictly increasing at index {bad}")

    o, h, l, c = (np.asarray(arrays[k], dtype=np.float64) for k in _PRICE_COLS)
    if not (len(o) == len(h) == len(l) == len(c) == len(t)):
        raise ValueError(f"{symbol} {tf}: column length mismatch")
    bad = np.where((h < l) | (h < o) | (h < c) | (l > o) | (l > c))[0]
    if bad.size:
        raise ValueError(f"{symbol} {tf}: {bad.size} bars violate OHLC bounds, "
                         f"first at index {int(bad[0])}")

    np.save(os.path.join(d, "time.npy"), t)
    for k in _PRICE_COLS:
        np.save(os.path.join(d, f"{k}.npy"), np.asarray(arrays[k], dtype=np.float64))
    np.save(os.path.join(d, "volume.npy"), np.asarray(arrays["volume"], dtype=np.float32))
    np.save(os.path.join(d, "spread.npy"), np.asarray(arrays["spread"], dtype=np.float32))

    meta = Meta(symbol=symbol.upper(), timeframe=tf.upper(), digits=int(digits),
                point=float(point), n=int(t.size), first=int(t[0]), last=int(t[-1]))
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, indent=2)
    return meta


def available(base: str | None = None) -> dict:
    """{symbol: {tf: Meta}} for everything currently in the store."""
    b = base or root()
    out: dict = {}
    if not os.path.isdir(b):
        return out
    for sym in sorted(os.listdir(b)):
        sd = os.path.join(b, sym)
        if not os.path.isdir(sd):
            continue
        for tf in sorted(os.listdir(sd), key=lambda x: TF_MINUTES.get(x, 0)):
            mp = os.path.join(sd, tf, "meta.json")
            if os.path.exists(mp):
                with open(mp, encoding="utf-8") as f:
                    out.setdefault(sym, {})[tf] = Meta(**json.load(f))
    return out
