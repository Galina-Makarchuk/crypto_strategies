"""Single source of truth for market data.

Configure the project-wide dataset **once** in the ACTIVE block below, then every
notebook / script / backtest reuses the same cached candles — no per-file
symbol/interval/window literals and no redundant downloads.

Typical use (anywhere ``engine`` is importable)::

    from engine.data_configurator import load_data
    df = load_data()                 # the ACTIVE spec, cached under data/ohlcv/

Change the dataset for the whole project by editing the ACTIVE block, or pass an
explicit spec for a one-off::

    from engine.data_configurator import load_data, DataSpec
    df = load_data(DataSpec(symbol="ETHUSDT", interval="60", category="inverse"))

Cache layout (git-ignored, lives under data/)::

    data/ohlcv/<category>/<symbol>_<interval>_<window>.parquet   # the candles
    data/ohlcv/<category>/<symbol>_<interval>_<window>.json      # provenance sidecar, records the spec + fetched range + fetch time


Cache path is anchored at the repo root (Path(__file__).parents[1]), 
so it works regardless of which directory a notebook runs from.

A cache hit is reused unless refresh=True.

The returned frame matches BybitFetcher's contract exactly: a timezone-aware
(UTC) DatetimeIndex named ``timestamp`` with float columns
``open, high, low, close, volume, turnover``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .fetcher import BybitFetcher
from .core import validate_category, validate_interval

if TYPE_CHECKING:  # avoid importing the heavy backtester module at runtime
    from .backtester import BacktestResult

logger = logging.getLogger(__name__)


# ── Spec ─────────────────────────────────────────────────────────────────────
# Category validation (validate_category / VALID_CATEGORIES) lives in .core,
# shared with the fetcher and the live engine.


@dataclass(frozen=True)
class DataSpec:
    """Immutable description of *which* candles to load.

    Two mutually-exclusive windowing modes:
      * count mode — most recent ``num_candles`` (used when ``start`` is None).
      * range mode — the full ``[start, end]`` window (``end`` defaults to now);
        ``num_candles`` is ignored.
    """

    symbol: str = "BTCUSDT"
    interval: str = "15"            # any VALID_INTERVALS member
    category: str = "linear"        # "linear" | "inverse"
    num_candles: int = 800          # count mode only (when start is None)
    start: str | None = None        # ISO date/time, e.g. "2026-03-20"
    end: str | None = None          # ISO date/time; None → now

    @property
    def is_range(self) -> bool:
        return self.start is not None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            ✏️   EDIT HERE   ✏️                              ║
# ║  The ONE place to configure the project-wide dataset. Change a value,      ║
# ║  save, and re-run any notebook/script — they all call load_data().         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
ACTIVE = DataSpec(
    symbol      = "BTCUSDT",
    interval    = "15",          # 1 3 5 15 30 60 120 240 360 720 D W M
    category    = "linear",      # "linear" | "inverse"
    num_candles = 800,           # used only when start is None
    start       = None,          # e.g. "2026-03-20"  → range mode (num_candles ignored)
    end         = None,          # e.g. "2026-04-19"  → defaults to now
)
# ── end edit block ─────────────────────────────────────────────────────────


# ── Cache location (CWD-independent: anchored at the repo root) ──────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = _REPO_ROOT / "data" / "ohlcv"
RESULTS_DIR = _REPO_ROOT / "data" / "results"
LIVE_DIR = _REPO_ROOT / "data" / "live"

_INTERVAL_MINUTES: dict[str, int] = {
    "1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60,
    "120": 120, "240": 240, "360": 360, "720": 720,
    "D": 1440, "W": 10080, "M": 43200,
}


# ── public API ───────────────────────────────────────────────────────────────


def load_data(
    spec: DataSpec = ACTIVE,
    *,
    refresh: bool = False,
    cache_dir: Path | str | None = None,
    fetcher: BybitFetcher | None = None,
) -> pd.DataFrame:
    """Return OHLCV candles for ``spec``, reading the on-disk cache when possible.

    Reads the cached parquet if it exists and is still fresh; otherwise fetches
    from Bybit (via :class:`BybitFetcher`) and writes the cache. ``refresh=True``
    forces a re-download.

    Freshness: a range window with an explicit ``end`` is immutable and always
    reused; count-mode and open-ended windows are re-fetched once the cache is
    older than one bar interval.

    Args:
        spec: which candles to load (defaults to the module-level ``ACTIVE``).
        refresh: ignore any cache and re-download.
        cache_dir: override the cache root (defaults to ``CACHE_DIR``); mainly
            for tests.
        fetcher: reuse an existing fetcher (e.g. in a loop); when omitted a
            short-lived one is created and closed.
    """
    _validate(spec)
    base = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    data_path = _cache_path(base, spec)
    meta_path = data_path.with_suffix(".json")

    if not refresh and data_path.exists() and _is_fresh(spec, meta_path):
        cached = _read_cache(data_path)
        if cached is not None:
            logger.info("Loaded %d cached candles from %s", len(cached), data_path)
            return cached

    df = _ensure_contract(_fetch(spec, fetcher))
    if df.empty:
        logger.warning("Fetch returned no candles for %s; not caching.", spec)
        return df

    data_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(df, data_path)
    _write_meta(meta_path, spec, df)
    logger.info("Fetched and cached %d candles to %s", len(df), data_path)
    return df


def cache_path(spec: DataSpec = ACTIVE, cache_dir: Path | str | None = None) -> Path:
    """The parquet path where ``spec`` is (or would be) cached."""
    base = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    return _cache_path(base, spec)


def dataset_signature(spec: DataSpec = ACTIVE) -> str:
    """Stable identifier for the dataset described by ``spec``.

    Used to group saved results so every strategy analysed on the same data
    lands under one folder, e.g. ``linear_BTCUSDT_15_last800``.
    """
    return f"{spec.category}_{spec.symbol}_{spec.interval}_{_window_tag(spec)}"


def save_result(
    result: BacktestResult,
    spec: DataSpec = ACTIVE,
    *,
    name: str | None = None,
    results_dir: Path | str | None = None,
) -> Path:
    """Persist a backtest result under ``data/results/<dataset_signature>/``.

    Writes two sibling files, both keyed by ``name`` (defaults to the strategy
    name; pass it to tag results by notebook — e.g. ``name="ema_rsi"`` so the
    files say which notebook produced them, not just the bare strategy name):
      * ``<name>.json``       — summary metrics + run metadata + nested trades
        (the result object is hierarchical, so JSON keeps it in one round-trippable
        file; non-finite floats like an all-wins profit factor become ``null`` so
        the file stays valid JSON).
      * ``<name>_trades.csv`` — the flat trade log, for pandas/Excel.

    Returns the path to the JSON file.
    """
    stem = name or result.strategy_name
    base = Path(results_dir) if results_dir is not None else RESULTS_DIR
    out_dir = base / dataset_signature(spec)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = [_trade_record(t) for t in result.trades]
    payload = {
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "interval": result.interval,
        "num_bars": result.num_bars,
        "dataset_signature": dataset_signature(spec),
        "spec": asdict(spec),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "total_trades": result.total_trades,
            "suppressed_entries": result.suppressed_entries,
            "risk_sizing_fallbacks": result.risk_sizing_fallbacks,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "break_even_trades": result.break_even_trades,
            "win_rate": _finite(result.win_rate),
            "total_pnl_bps": _finite(result.total_pnl_bps),
            "avg_pnl_bps": _finite(result.avg_pnl_bps),
            "max_win_bps": _finite(result.max_win_bps),
            "max_loss_bps": _finite(result.max_loss_bps),
            "profit_factor": _finite(result.profit_factor),
            "max_drawdown_bps": _finite(result.max_drawdown_bps),
            "sharpe_approx": _finite(result.sharpe_approx),
            "initial_equity": _finite(result.initial_equity),
            "final_equity": _finite(result.final_equity),
            "total_return_pct": _finite(result.total_return_pct),
            "max_drawdown_pct": _finite(result.max_drawdown_pct),
        },
        "trades": trades,
    }

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = out_dir / f"{stem}_trades.csv"
    pd.DataFrame(trades, columns=_TRADE_COLUMNS).to_csv(csv_path, index=False)

    logger.info(
        "Saved %s result (%d trades) to %s", stem, len(trades), out_dir
    )
    return json_path


# ── internals ────────────────────────────────────────────────────────────────


_TRADE_COLUMNS = [
    "trade_id", "direction", "entry_ts", "entry_price", "exit_ts", "exit_price",
    "pnl_bps", "peak_price", "exit_reason", "duration_seconds",
    "stop_price", "notional", "pnl_currency", "equity_after",
]


def _finite(x: float) -> float | None:
    """JSON-safe float: non-finite (inf/-inf/nan) → None so the file stays valid JSON."""
    return x if math.isfinite(x) else None


def _trade_record(t) -> dict:
    """Flatten a Trade into a JSON/CSV-friendly dict (enums → values, ts → ISO)."""
    duration = t.duration
    return {
        "trade_id": t.trade_id,
        "direction": t.direction.value if t.direction is not None else None,
        "entry_ts": t.entry_ts.isoformat() if t.entry_ts is not None else None,
        "entry_price": t.entry_price,
        "exit_ts": t.exit_ts.isoformat() if t.exit_ts is not None else None,
        "exit_price": t.exit_price,
        "pnl_bps": _finite(t.pnl_bps),
        "peak_price": t.peak_price,
        "exit_reason": t.exit_reason.value if t.exit_reason is not None else None,
        "duration_seconds": duration.total_seconds() if duration is not None else None,
        "stop_price": t.stop_price,
        "notional": t.notional,
        "pnl_currency": t.pnl_currency,
        "equity_after": t.equity_after,
    }


def _validate(spec: DataSpec) -> None:
    validate_interval(spec.interval)
    validate_category(spec.category)
    if not spec.is_range and spec.num_candles <= 0:
        raise ValueError("num_candles must be positive in count mode.")


def _slug(text: str) -> str:
    """Filesystem-safe token from an ISO date/time (or any string)."""
    return "".join(c if c.isalnum() else "-" for c in text.strip())


def _window_tag(spec: DataSpec) -> str:
    if spec.is_range:
        return f"{_slug(spec.start)}_{_slug(spec.end) if spec.end else 'now'}"
    return f"last{spec.num_candles}"


def _cache_path(base: Path, spec: DataSpec) -> Path:
    name = f"{spec.symbol}_{spec.interval}_{_window_tag(spec)}.parquet"
    return base / spec.category / name


def _read_cache(data_path: Path) -> pd.DataFrame | None:
    """Read a cached parquet, treating any corruption as a cache miss.

    An interrupted write could leave a truncated file; rather than raise on the
    hot read path, we log and fall through to a re-fetch (self-healing cache).
    """
    try:
        return _ensure_contract(pd.read_parquet(data_path))
    except Exception as exc:  # noqa: BLE001 — any read failure ⇒ refetch
        logger.warning("Cached parquet %s unreadable (%s); re-fetching.", data_path, exc)
        return None


def _write_parquet_atomic(df: pd.DataFrame, data_path: Path) -> None:
    """Write parquet via a temp file + atomic rename, so an interrupted write
    can never leave a truncated cache (mirrors engine/ml/order_flow.py)."""
    tmp = data_path.with_name(data_path.name + ".part")
    df.to_parquet(tmp)
    tmp.replace(data_path)


def _fetch(spec: DataSpec, fetcher: BybitFetcher | None) -> pd.DataFrame:
    own = fetcher is None
    fetcher = fetcher or BybitFetcher()
    try:
        return fetcher.fetch_klines(
            symbol=spec.symbol,
            interval=spec.interval,
            category=spec.category,
            num_candles=spec.num_candles,
            start_time=spec.start,
            end_time=spec.end,
        )
    finally:
        if own:
            fetcher.close()


def _is_fresh(spec: DataSpec, meta_path: Path) -> bool:
    # A fully-pinned range window ([start, end] both set) is immutable.
    if spec.is_range and spec.end is not None:
        return True
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        fetched_at = pd.Timestamp(meta["fetched_at"])
    except (ValueError, KeyError, OSError):
        return False
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.tz_localize("UTC")
    age = pd.Timestamp.now(tz="UTC") - fetched_at
    return age < pd.Timedelta(minutes=_INTERVAL_MINUTES[spec.interval])


def _write_meta(meta_path: Path, spec: DataSpec, df: pd.DataFrame) -> None:
    meta = {
        **asdict(spec),
        "mode": "range" if spec.is_range else "count",
        "fetched_rows": int(len(df)),
        "fetched_start": df.index[0].isoformat() if len(df) else None,
        "fetched_end": df.index[-1].isoformat() if len(df) else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def _ensure_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Defensively guarantee the data contract after fetch or cache read.

    The fetcher already returns a UTC-aware DatetimeIndex named ``timestamp``;
    parquet round-trips it via pyarrow. This re-asserts the invariant cheaply so
    a malformed cache can never leak a naive index downstream.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df  # empty/degenerate frame — let the caller handle it
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df
