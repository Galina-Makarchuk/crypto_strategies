"""Bybit kline fetcher with retry and rate-limiting.

Production considerations:
- Exponential backoff on 429/5xx via urllib3 Retry
- Token-bucket rate limiter (configurable)
- Windowed re-fetch for live mode (LiveEngine re-fetches a full window each
  tick rather than tail-appending)
- Timezone-aware UTC timestamps
- Typed return value (always a clean DataFrame)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .core import validate_interval

logger = logging.getLogger(__name__)


def _to_utc_ms(t: str | pd.Timestamp) -> int:
    """Parse a time value to a UTC-based millisecond epoch. Naive inputs are UTC."""
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, max_calls: int = 100, period_seconds: float = 5.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._timestamps: list[float] = []

    def wait(self) -> None:
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self.period]
        if len(self._timestamps) >= self.max_calls:
            sleep_for = self.period - (now - self._timestamps[0]) + 0.05
            if sleep_for > 0:
                logger.debug("Rate limiter: sleeping %.2fs", sleep_for)
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


class BybitFetcher:
    """Fetch OHLCV klines from Bybit v5 public API."""

    BASE_URL = "https://api.bybit.com"
    MAX_LIMIT = 1000

    def __init__(self, rate_limit_calls: int = 100, rate_limit_period: float = 5.0):
        self._session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))
        self._rate_limiter = RateLimiter(rate_limit_calls, rate_limit_period)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "BybitFetcher":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── internal ───────────────────────────────────────────────────────────

    def _fetch_batch(
        self,
        symbol: str,
        interval: str,
        limit: int,
        end_ms: int | None = None,
        category: str = "linear",
    ) -> list[list[str]]:
        self._rate_limiter.wait()
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, self.MAX_LIMIT),
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = self._session.get(
            f"{self.BASE_URL}/v5/market/kline", params=params, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}")

        rows = data.get("result", {}).get("list", [])
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected response shape: {type(rows)}")
        return rows

    # ── public ─────────────────────────────────────────────────────────────

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15",
        num_candles: int = 1000,
        start_time: str | pd.Timestamp | None = None,
        end_time: str | pd.Timestamp | None = None,
        category: str = "linear",
    ) -> pd.DataFrame:
        """Fetch historical klines.

        When `start_time` is set, paginate the full [start_time, end_time] window
        (end_time defaults to now); `num_candles` is ignored in this mode.
        Otherwise fetch the most recent `num_candles` ending at `end_time` (or now).

        `category` selects the Bybit product type ("linear" or "inverse").

        Time inputs accept anything `pd.Timestamp` parses (ISO strings, datetimes);
        naive values are treated as UTC.

        Returns a timezone-aware (UTC) DataFrame indexed by timestamp with
        columns: open, high, low, close, volume, turnover.
        """
        validate_interval(interval)

        start_ms = _to_utc_ms(start_time) if start_time is not None else None
        end_ms = _to_utc_ms(end_time) if end_time is not None else None
        if start_ms is not None and end_ms is not None and start_ms >= end_ms:
            raise ValueError("start_time must be before end_time")

        range_mode = start_ms is not None
        if range_mode:
            logger.info(
                "Fetching %s candles for %s [%s → %s]",
                interval, symbol, start_time, end_time or "now",
            )
        else:
            logger.info("Fetching %d × %s candles for %s …", num_candles, interval, symbol)

        all_data: list[list[str]] = []
        cursor_end_ms: int | None = end_ms
        max_pages = 500 if range_mode else 50  # safety cap

        for _ in range(max_pages):
            if range_mode:
                batch_limit = self.MAX_LIMIT
            else:
                remaining = num_candles - len(all_data)
                if remaining <= 0:
                    break
                batch_limit = remaining + 100

            batch = self._fetch_batch(symbol, interval, batch_limit, cursor_end_ms, category)
            if not batch:
                break
            all_data.extend(batch)
            oldest_ms = int(batch[-1][0])
            cursor_end_ms = oldest_ms - 1
            if range_mode and oldest_ms <= start_ms:
                break

        if not range_mode and len(all_data) < num_candles:
            logger.warning(
                "Requested %d candles but only received %d (exchange may have fewer available).",
                num_candles,
                len(all_data),
            )

        # Deduplicate by timestamp, sort ascending
        seen: set[str] = set()
        unique: list[list[str]] = []
        for row in all_data:
            ts = row[0]
            if ts not in seen:
                seen.add(ts)
                unique.append(row)
        unique.sort(key=lambda x: int(x[0]))

        if range_mode:
            unique = [r for r in unique if int(r[0]) >= start_ms]
            if end_ms is not None:
                unique = [r for r in unique if int(r[0]) <= end_ms]
        else:
            unique = unique[-num_candles:]

        df = pd.DataFrame(
            unique,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)

        if len(df) == 0:
            logger.warning("No candles in requested window.")
        else:
            logger.info("Loaded %d candles [%s → %s]", len(df), df.index[0], df.index[-1])
        return df
