"""Order-flow features from Bybit bulk trade history (Tier 3).

Bybit publishes a tick-level trade archive at
``https://public.bybit.com/trading/{SYMBOL}/{SYMBOL}{YYYY-MM-DD}.csv.gz``.
Each daily file contains every aggressor-side trade for one symbol on one
UTC day. We download, aggregate to bar-aligned features, and cache the
small aggregates to disk so re-runs are cheap.

The big-yield feature here is **order-flow imbalance (OFI)** — signed
trade volume per bar. Empirically the single most predictive feature for
short-horizon reversals; 15m candles hide it because OHLCV only sees the
net price effect of all trades, not the buy/sell flow that produced it.

Per-bar features computed (see ``OFI_BAR_COLUMNS`` for the exact list):

- buy / sell / total USD volume
- imbalance  = (buy - sell) / (buy + sell)         ∈ [-1, +1]
- trade_count, avg_trade_usd, max_trade_usd
- large_trade_count  (proxy for institutional flow)
- buy_trade_ratio    = buy_trades / total_trades

Then ``compute_orderflow_features()`` adds rolling derivatives:

- imbalance / volume / count z-scores (50-bar window)
- cumulative volume delta (CVD) change over 10 bars

Caching strategy:
- ``data/trades/`` holds raw daily gz files (large; gitignored).
- ``data/ofi_aggregates/`` holds per-day pickle aggregates (~10 KB each; gitignored).
- Both are checked before any network call → pipeline is resumable.

The live-mode equivalent needs a Bybit WebSocket subscription to the public
trade stream, not implemented here. This module is offline-only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ── Feature schema ────────────────────────────────────────────────────────────

OFI_BAR_COLUMNS: tuple[str, ...] = (
    "ofi_buy_volume_usd",
    "ofi_sell_volume_usd",
    "ofi_total_volume_usd",
    "ofi_imbalance",
    "ofi_trade_count",
    "ofi_avg_trade_usd",
    "ofi_max_trade_usd",
    "ofi_large_trade_count",
    "ofi_buy_trade_ratio",
)

OFI_DERIVED_COLUMNS: tuple[str, ...] = (
    "ofi_imbalance_z50",
    "ofi_volume_z50",
    "ofi_count_z50",
    "ofi_cvd_change_10",
)

OFI_FEATURE_COLUMNS: tuple[str, ...] = OFI_BAR_COLUMNS + OFI_DERIVED_COLUMNS

LARGE_TRADE_USD: float = 100_000.0  # 100K USD = "large" flow threshold

BYBIT_BULK_URL = "https://public.bybit.com/trading/{symbol}/{symbol}{date}.csv.gz"


# ── Download ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DownloadResult:
    date: pd.Timestamp
    path: Path
    bytes_written: int
    from_cache: bool


def _day_url(symbol: str, date: pd.Timestamp) -> str:
    return BYBIT_BULK_URL.format(symbol=symbol, date=date.strftime("%Y-%m-%d"))


def _day_path(symbol: str, date: pd.Timestamp, raw_dir: Path) -> Path:
    return raw_dir / f"{symbol}{date.strftime('%Y-%m-%d')}.csv.gz"


def download_day(
    symbol: str,
    date: pd.Timestamp,
    raw_dir: Path,
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 1.5,
) -> DownloadResult:
    """Download one day's trade gzip. No-op if already cached.

    Raises ``FileNotFoundError`` if Bybit returns 404 for that date — common
    for the most-recent day (not yet published) and for very old dates.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = _day_path(symbol, date, raw_dir)
    if dest.exists() and dest.stat().st_size > 0:
        return DownloadResult(date=date, path=dest, bytes_written=0, from_cache=True)

    url = _day_url(symbol, date)
    sess = session or requests.Session()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with sess.get(url, stream=True, timeout=timeout) as resp:
                if resp.status_code == 404:
                    raise FileNotFoundError(f"Bybit has no trade file for {symbol} {date.date()}")
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                bytes_written = 0
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
                            bytes_written += len(chunk)
                tmp.rename(dest)
            return DownloadResult(date=date, path=dest, bytes_written=bytes_written, from_cache=False)
        except FileNotFoundError:
            raise
        except Exception as e:  # network / transient
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise RuntimeError(f"Failed to download {url}: {last_err}")


# ── Parse ─────────────────────────────────────────────────────────────────────


_CSV_DTYPES = {
    "symbol": "string",
    "side": "string",
    "size": "float64",
    "price": "float64",
    "tickDirection": "string",
    "trdMatchID": "string",
    "grossValue": "float64",
    "homeNotional": "float64",
    "foreignNotional": "float64",
}


def parse_day(path: Path) -> pd.DataFrame:
    """Read one day's gzipped trade CSV into a tz-aware UTC DataFrame.

    Keeps only the columns we need: ``timestamp`` (UTC), ``side``, ``price``,
    ``size``, ``notional_usd``. The full file is ~30–100 MB compressed; pandas
    handles it in 1–3 seconds.
    """
    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=["timestamp", "side", "size", "price", "homeNotional", "foreignNotional"],
        dtype={
            "side": "string",
            "size": "float64",
            "price": "float64",
            "homeNotional": "float64",
            "foreignNotional": "float64",
        },
    )
    # Bybit bulk timestamps are UTC epoch seconds (float). Some daily files
    # arrive sorted ascending, some descending — re-sort defensively.
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.assign(ts=ts).drop(columns=["timestamp"]).set_index("ts").sort_index()
    # foreignNotional is the USD value of the trade (size × price for BTCUSDT
    # linear perps). Fall back to size × price if it's missing or zero.
    notional = df["foreignNotional"].where(df["foreignNotional"] > 0, df["size"] * df["price"])
    return pd.DataFrame(
        {"side": df["side"], "size": df["size"], "price": df["price"], "notional_usd": notional},
        index=df.index,
    )


# ── Aggregate ─────────────────────────────────────────────────────────────────


def aggregate_to_bars(
    trades: pd.DataFrame,
    interval: str,
    *,
    large_trade_usd: float = LARGE_TRADE_USD,
) -> pd.DataFrame:
    """Bin trades to bar-aligned aggregates.

    Args:
        trades: tz-aware DataFrame from :func:`parse_day` (index = trade time).
        interval: bar size — pandas freq alias (e.g. ``"15min"``, ``"5min"``).
        large_trade_usd: threshold above which a trade counts as "large".

    Returns:
        DataFrame indexed by bar-start timestamps (left-closed, UTC) with the
        columns in :data:`OFI_BAR_COLUMNS`.
    """
    if trades.empty:
        return pd.DataFrame(columns=list(OFI_BAR_COLUMNS))

    is_buy = trades["side"].eq("Buy")
    is_sell = trades["side"].eq("Sell")
    notional = trades["notional_usd"]

    buy_vol = notional.where(is_buy, 0.0)
    sell_vol = notional.where(is_sell, 0.0)
    is_large = notional > large_trade_usd

    grouper = pd.Grouper(freq=interval, label="left", closed="left", origin="epoch")
    g = trades.groupby(grouper)

    agg = pd.DataFrame({
        "ofi_buy_volume_usd": buy_vol.groupby(grouper).sum(),
        "ofi_sell_volume_usd": sell_vol.groupby(grouper).sum(),
        "ofi_total_volume_usd": notional.groupby(grouper).sum(),
        "ofi_trade_count": g.size(),
        "ofi_max_trade_usd": notional.groupby(grouper).max(),
        "ofi_large_trade_count": is_large.groupby(grouper).sum(),
        "ofi_buy_trade_count": is_buy.groupby(grouper).sum(),
    })

    total = agg["ofi_buy_volume_usd"] + agg["ofi_sell_volume_usd"]
    agg["ofi_imbalance"] = (agg["ofi_buy_volume_usd"] - agg["ofi_sell_volume_usd"]) / total.replace(0.0, np.nan)
    agg["ofi_avg_trade_usd"] = agg["ofi_total_volume_usd"] / agg["ofi_trade_count"].replace(0, np.nan)
    agg["ofi_buy_trade_ratio"] = agg["ofi_buy_trade_count"] / agg["ofi_trade_count"].replace(0, np.nan)
    agg = agg.drop(columns=["ofi_buy_trade_count"])

    # Empty bars: total_volume = 0, trade_count = 0, imbalance = NaN.
    # Keep them in the index (we'll forward-fill at merge time so the model
    # sees "no flow" rather than dropped rows).
    agg = agg[list(OFI_BAR_COLUMNS)]
    return agg


# ── Derived (rolling) features ────────────────────────────────────────────────


def compute_derived_orderflow(bars: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-window derivatives on top of the per-bar aggregates.

    Returns a frame with all columns in :data:`OFI_FEATURE_COLUMNS`.
    """
    out = bars.copy()

    def _z(s: pd.Series, w: int = 50) -> pd.Series:
        m = s.rolling(w, min_periods=max(2, w // 4)).mean()
        sd = s.rolling(w, min_periods=max(2, w // 4)).std(ddof=0).replace(0.0, np.nan)
        return ((s - m) / sd).clip(-6.0, 6.0)

    out["ofi_imbalance_z50"] = _z(out["ofi_imbalance"], 50)
    out["ofi_volume_z50"] = _z(out["ofi_total_volume_usd"], 50)
    out["ofi_count_z50"] = _z(out["ofi_trade_count"].astype(float), 50)

    # CVD = cumulative signed volume. We use imbalance × total_volume as
    # the signed-volume proxy (avoids issues where buy/sell columns are 0
    # due to one-sided bars).
    signed = (
        out["ofi_buy_volume_usd"] - out["ofi_sell_volume_usd"]
    ).fillna(0.0)
    cvd = signed.cumsum()
    out["ofi_cvd_change_10"] = cvd - cvd.shift(10)

    return out[list(OFI_FEATURE_COLUMNS)]


# ── End-to-end orchestrator ───────────────────────────────────────────────────


def _daterange(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[pd.Timestamp]:
    cur = start.normalize()
    end = end.normalize()
    while cur <= end:
        yield cur
        cur = cur + pd.Timedelta(days=1)


def build_orderflow_features(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str = "15min",
    *,
    raw_dir: Path | str = "data/trades",
    agg_dir: Path | str = "data/ofi_aggregates",
    keep_raw: bool = True,
    on_missing: str = "skip",
    progress: bool = True,
) -> pd.DataFrame:
    """End-to-end: download → parse → aggregate → cache → merge → derive.

    Args:
        symbol: e.g. ``"BTCUSDT"``.
        start, end: any pandas-parseable timestamp (inclusive on both sides).
        interval: bar size (pandas freq alias).
        raw_dir: where to cache raw daily gz files.
        agg_dir: where to cache per-day Parquet aggregates.
        keep_raw: if False, delete the raw gz after aggregating (saves disk).
        on_missing: ``"skip"`` (warn and continue) or ``"raise"`` for 404.
        progress: log per-day progress.

    Returns:
        DataFrame indexed by bar-start timestamps (UTC, tz-aware), one row
        per bar in [start, end], with all :data:`OFI_FEATURE_COLUMNS`.
    """
    raw_dir = Path(raw_dir)
    agg_dir = Path(agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    daily_aggs: list[pd.DataFrame] = []
    session = requests.Session()
    for day in _daterange(start_ts, end_ts):
        agg_path = agg_dir / f"{symbol}_{day.strftime('%Y-%m-%d')}_{interval}.pkl"
        if agg_path.exists():
            daily_aggs.append(pd.read_pickle(agg_path))
            if progress:
                logger.info("ofi cache hit: %s", agg_path.name)
            continue
        try:
            dr = download_day(symbol, day, raw_dir, session=session)
        except FileNotFoundError as e:
            if on_missing == "raise":
                raise
            logger.warning("skipping missing day: %s", e)
            continue
        if progress:
            logger.info(
                "ofi day %s: %s %.1f MB",
                day.date(),
                "cache" if dr.from_cache else "downloaded",
                dr.bytes_written / 1e6 if not dr.from_cache else 0.0,
            )
        trades = parse_day(dr.path)
        agg = aggregate_to_bars(trades, interval=interval)
        # Restrict to bars whose START is inside this UTC day (avoid double-counting
        # the 23:45–00:00 bar in adjacent files).
        day_start = day.tz_convert("UTC") if day.tz is not None else day.tz_localize("UTC")
        day_end = day_start + pd.Timedelta(days=1)
        mask = (agg.index >= day_start) & (agg.index < day_end)
        agg = agg.loc[mask]
        agg.to_pickle(agg_path)
        daily_aggs.append(agg)
        if not keep_raw:
            dr.path.unlink(missing_ok=True)

    if not daily_aggs:
        raise RuntimeError(f"No order-flow data assembled for {symbol} [{start} → {end}]")

    bars = pd.concat(daily_aggs).sort_index()
    bars = bars[~bars.index.duplicated(keep="first")]
    return compute_derived_orderflow(bars)


def merge_orderflow_features(klines: pd.DataFrame, ofi: pd.DataFrame) -> pd.DataFrame:
    """Reindex OFI features onto a kline DataFrame's index.

    OFI bars are left-closed at the same grid as the klines (Bybit's klines
    are also left-closed UTC), so this is a clean reindex. Missing bars are
    forward-filled and then zero-filled — "no flow" is a valid state, not a
    missing-data error.
    """
    aligned = ofi.reindex(klines.index)
    # Fill empty bars: volume / counts / large = 0; imbalance / z-scores = 0
    # (neutral); CVD-change carries the last observation.
    aligned[["ofi_buy_volume_usd", "ofi_sell_volume_usd", "ofi_total_volume_usd",
             "ofi_trade_count", "ofi_max_trade_usd", "ofi_large_trade_count",
             "ofi_avg_trade_usd"]] = aligned[[
                "ofi_buy_volume_usd", "ofi_sell_volume_usd", "ofi_total_volume_usd",
                "ofi_trade_count", "ofi_max_trade_usd", "ofi_large_trade_count",
                "ofi_avg_trade_usd"]].fillna(0.0)
    aligned[["ofi_imbalance", "ofi_buy_trade_ratio",
             "ofi_imbalance_z50", "ofi_volume_z50", "ofi_count_z50"]] = aligned[[
                "ofi_imbalance", "ofi_buy_trade_ratio",
                "ofi_imbalance_z50", "ofi_volume_z50", "ofi_count_z50"]].fillna(0.0)
    aligned["ofi_cvd_change_10"] = aligned["ofi_cvd_change_10"].ffill().fillna(0.0)
    return aligned
