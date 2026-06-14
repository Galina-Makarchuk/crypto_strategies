"""The market-data provider seam.

A DataProvider is any object that fetches OHLCV candles and returns them in the
one canonical contract the whole engine consumes — a UTC-aware DataFrame with
float columns open, high, low, close, volume, turnover, indexed by timestamp,
oldest-first. Everything downstream of engine.data_configurator.load_data is
provider-blind, so swapping or adding a provider never touches the backtester,
strategies, exits, evaluation or visualization.

BybitFetcher (engine/providers/bybit.py) is the reference implementation. New providers
(e.g. engine/providers/yahoo.py) translate their vendor's payload to this
contract internally — their own pagination, rate limits, interval codes,
authentication and turnover synthesis all stay behind the seam.

The provider registry, dispatch and per-provider validation live in
engine/providers/__init__.py.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import pandas as pd

# The exact column set, in order, every provider returns.
CONTRACT_COLUMNS = ("open", "high", "low", "close", "volume", "turnover")


@runtime_checkable
class DataProvider(Protocol):
    """Structural interface for a market-data source.

    Implementations also expose class-level metadata used by the registry to
    validate a DataSpec without instantiating the provider:
      NAME: str                              — the registry key / DataSpec.provider value
      VALID_INTERVALS: frozenset[str]        — canonical interval codes it supports
      VALID_CATEGORIES: frozenset[str]|None  — product categories, or None if the
                                               provider has no product taxonomy
      DEFAULT_CATEGORY: str | None           — default category when one applies
    """

    NAME: str

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        num_candles: int = 1000,
        start_time: "str | pd.Timestamp | None" = None,
        end_time: "str | pd.Timestamp | None" = None,
        category: "str | None" = None,
    ) -> pd.DataFrame:
        ...

    def close(self) -> None:
        ...


def finalize_ohlcv(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Coerce a provider's frame to the canonical contract.

    Guarantees: a UTC-aware DatetimeIndex named timestamp (oldest-first,
    de-duplicated keeping the last row per timestamp); float64 columns
    open, high, low, close, volume, turnover in that order. turnover is
    synthesized as close * volume when the vendor doesn't supply it (a notional
    proxy), so the 7-column contract always holds. An empty/None frame returns an
    empty contract frame. Raises if a required OHLCV column is missing.

    BybitFetcher already emits the contract natively, so it does not route through
    here (keeping its output byte-for-byte). Non-crypto providers call this.
    """
    empty_index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    if df is None or len(df) == 0:
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in CONTRACT_COLUMNS},
            index=empty_index,
        )

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("provider frame must be indexed by a DatetimeIndex")
    df.index = (
        df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    )
    df.index.name = "timestamp"

    # Required input columns (turnover is synthesized if absent).
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"provider frame missing required columns: {missing}")

    if "turnover" not in df.columns:
        df["turnover"] = df["close"].astype(float) * df["volume"].astype(float)

    df = df[list(CONTRACT_COLUMNS)].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df
