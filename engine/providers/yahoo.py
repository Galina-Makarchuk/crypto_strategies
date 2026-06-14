"""yahoo — Yahoo Finance data provider (via yfinance), the non-crypto source.

Covers equity indices, commodities (oil/gold) and currency futures through Yahoo
continuous-contract and index tickers, e.g. GC=F (gold), CL=F (oil), ES=F (S&P
500 future), 6E=F (euro FX future), ^GSPC (S&P 500 index). Free and keyless —
Yahoo has no official API; yfinance handles the cookie/crumb handshake against
the unofficial endpoints, which is exactly the churn we want isolated here.

Contract: returns the canonical UTC OHLCV frame (see engine.providers.base).
Yahoo has no turnover, so finalize_ohlcv synthesizes turnover = close * volume.
Yahoo has no linear/inverse product taxonomy, so VALID_CATEGORIES is None and the
category argument is ignored.

Caveats (inherent to Yahoo, not this code): intraday history is shallow (roughly
1m ≈ 7 days, sub-hour ≈ 60 days, 1h ≈ 2 years; daily ≈ decades), and the feed is
unofficial / best-effort. Strongest for daily/EOD and recent intraday.

yfinance is imported lazily inside fetch_klines, so the registry and this module
import fine without it installed; only an actual fetch requires it.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import finalize_ohlcv

logger = logging.getLogger(__name__)

# canonical interval code -> Yahoo interval code. Canonical codes with no clean
# Yahoo equivalent (3, 120, 240, 360, 720) are intentionally unsupported.
_INTERVAL_MAP: dict[str, str] = {
    "1": "1m", "5": "5m", "15": "15m", "30": "30m", "60": "60m",
    "D": "1d", "W": "1wk", "M": "1mo",
}
# canonical -> minutes, for sizing the count-mode lookback window.
_INTERVAL_MINUTES: dict[str, int] = {
    "1": 1, "5": 5, "15": 15, "30": 30, "60": 60,
    "D": 1440, "W": 10080, "M": 43200,
}
# Yahoo's max intraday lookback (days) per interval — fetch can't exceed these.
_MAX_LOOKBACK_DAYS: dict[str, float] = {"1": 7, "5": 59, "15": 59, "30": 59, "60": 729}


class YahooProvider:
    """DataProvider backed by Yahoo Finance via yfinance."""

    NAME = "yahoo"
    VALID_INTERVALS = frozenset(_INTERVAL_MAP)
    VALID_CATEGORIES = None          # no product taxonomy (not linear/inverse)
    DEFAULT_CATEGORY = None

    def close(self) -> None:
        """No persistent session to close (yfinance manages its own)."""

    def fetch_klines(
        self,
        symbol: str = "ES=F",
        interval: str = "D",
        num_candles: int = 1000,
        start_time: "str | pd.Timestamp | None" = None,
        end_time: "str | pd.Timestamp | None" = None,
        category: "str | None" = None,   # ignored — Yahoo has no product category
    ) -> pd.DataFrame:
        if interval not in _INTERVAL_MAP:
            raise ValueError(
                f"yahoo provider does not support interval {interval!r}; "
                f"supported: {sorted(_INTERVAL_MAP)}"
            )
        yf_interval = _INTERVAL_MAP[interval]

        try:
            import yfinance as yf  # lazy — only needed for an actual fetch
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "yfinance is required for the 'yahoo' provider. Install it: "
                "pip install yfinance"
            ) from exc

        kwargs = dict(interval=yf_interval, auto_adjust=False, actions=False, progress=False)
        if start_time is not None:
            # range mode — honour the explicit window (num_candles ignored).
            kwargs["start"] = pd.Timestamp(start_time)
            if end_time is not None:
                kwargs["end"] = pd.Timestamp(end_time)
        else:
            # count mode — derive a start far enough back for num_candles bars,
            # padded for session gaps/holidays and capped at Yahoo's lookback limit.
            minutes = _INTERVAL_MINUTES[interval]
            span_days = (num_candles * minutes) / (60 * 24) * 2.0 + 5.0
            cap = _MAX_LOOKBACK_DAYS.get(interval)
            if cap is not None:
                span_days = min(span_days, cap)
            end_ts = pd.Timestamp(end_time) if end_time is not None else pd.Timestamp.utcnow()
            kwargs["end"] = end_ts
            kwargs["start"] = end_ts - pd.Timedelta(days=span_days)

        logger.info("Fetching %s %s candles from Yahoo (%s)", symbol, yf_interval, kwargs.get("start"))
        raw = yf.download(symbol, **kwargs)
        df = self._to_contract(raw)
        if start_time is None and len(df) > num_candles:
            df = df.iloc[-num_candles:]
        if len(df) == 0:
            logger.warning("Yahoo returned no candles for %s %s.", symbol, yf_interval)
        return df

    @staticmethod
    def _to_contract(raw: "pd.DataFrame | None") -> pd.DataFrame:
        """Pure conversion of a yfinance OHLCV frame to the canonical contract.

        Kept side-effect-free (no network) so it is unit-testable with a synthetic
        frame. Handles yfinance's single- or multi-ticker column shapes, drops the
        adjusted-close column, and delegates turnover synthesis / UTC / dtype /
        ordering to finalize_ohlcv.
        """
        if raw is None or len(raw) == 0:
            return finalize_ohlcv(None)

        df = raw.copy()
        # yfinance may return a column MultiIndex (field, ticker); flatten to field.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)

        cols = {}
        for c in ("open", "high", "low", "close", "volume"):
            if c not in df.columns:
                raise ValueError(
                    f"yfinance frame missing {c!r} column; got {list(df.columns)}"
                )
            cols[c] = df[c]
        out = pd.DataFrame(cols, index=df.index).dropna(subset=["open", "high", "low", "close"])
        return finalize_ohlcv(out)   # adds turnover, UTC index, float, sort, dedupe
