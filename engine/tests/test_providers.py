"""Tests for the market-data provider seam (engine.providers).

Covers the registry/dispatch, per-provider metadata + validation, the canonical
contract helper, the Yahoo adapter's pure conversion (no network), and the
data_configurator wiring (cache path + dataset signature) for a second provider.

All offline: the Yahoo provider's network fetch is never called — only its pure
_to_contract conversion and its pre-fetch interval guard are exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import providers as P
from engine.providers.base import CONTRACT_COLUMNS, finalize_ohlcv
from engine.providers.yahoo import YahooProvider
from engine.data_configurator import DataSpec, _validate, cache_path, dataset_signature


# ── Registry / dispatch ─────────────────────────────────────────────────────


class TestRegistry:
    def test_known_providers(self):
        assert set(P.PROVIDER_NAMES) == {"bybit", "yahoo"}
        assert set(P.PROVIDERS) == set(P.PROVIDER_NAMES)

    def test_make_provider_instantiates(self):
        for name in P.PROVIDER_NAMES:
            prov = P.make_provider(name)
            assert hasattr(prov, "fetch_klines") and hasattr(prov, "close")
            prov.close()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            P.provider_class("nope")
        with pytest.raises(ValueError):
            P.make_provider("nope")


# ── Per-provider metadata ───────────────────────────────────────────────────


class TestMetadata:
    def test_bybit_has_categories(self):
        assert P.provider_categories("bybit") == frozenset({"linear", "inverse"})
        assert P.default_category("bybit") == "linear"

    def test_yahoo_has_no_categories(self):
        assert P.provider_categories("yahoo") is None
        assert P.default_category("yahoo") is None

    def test_yahoo_interval_support(self):
        iv = P.provider_intervals("yahoo")
        assert {"1", "5", "15", "30", "60", "D", "W", "M"} == set(iv)
        # canonical codes without a clean Yahoo equivalent are excluded
        assert "120" not in iv and "3" not in iv


# ── resolve_category ────────────────────────────────────────────────────────


class TestResolveCategory:
    def test_bybit_passthrough_and_default(self):
        assert P.resolve_category("bybit", "inverse") == "inverse"
        assert P.resolve_category("bybit", None) == "linear"    # provider default

    def test_yahoo_always_none(self):
        assert P.resolve_category("yahoo", None) is None
        assert P.resolve_category("yahoo", "linear") is None    # category ignored


# ── Provider-aware validation ───────────────────────────────────────────────


class TestValidateSpec:
    @pytest.mark.parametrize("name,interval,category", [
        ("bybit", "15", "linear"),
        ("bybit", "D", "inverse"),
        ("yahoo", "D", None),
        ("yahoo", "15", "linear"),   # category ignored for yahoo → fine
    ])
    def test_valid(self, name, interval, category):
        P.validate_spec(name, interval, category)  # must not raise

    @pytest.mark.parametrize("name,interval,category,field", [
        ("nope", "15", "linear", "provider"),
        ("bybit", "7", "linear", "interval"),       # not a Bybit interval
        ("bybit", "15", "spot", "category"),        # not a Bybit category
        ("yahoo", "120", None, "interval"),         # unsupported by yahoo
    ])
    def test_invalid_names_the_field(self, name, interval, category, field):
        with pytest.raises(ValueError, match=field):
            P.validate_spec(name, interval, category)


# ── The canonical contract helper ───────────────────────────────────────────


class TestFinalizeOhlcv:
    def test_empty_frame_returns_empty_contract(self):
        out = finalize_ohlcv(pd.DataFrame())
        assert list(out.columns) == list(CONTRACT_COLUMNS)
        assert len(out) == 0
        assert out.index.tz is not None

    def test_synthesizes_turnover_and_normalizes(self):
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-01"])  # out of order, tz-naive
        raw = pd.DataFrame(
            {"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5],
             "close": [1.5, 2.5], "volume": [100, 200]},
            index=idx,
        )
        out = finalize_ohlcv(raw)
        assert list(out.columns) == list(CONTRACT_COLUMNS)       # turnover added, ordered
        assert str(out.index.tz) == "UTC" and out.index.name == "timestamp"
        assert out.index.is_monotonic_increasing                 # sorted ascending
        assert all(out[c].dtype == "float64" for c in CONTRACT_COLUMNS)
        # turnover = close * volume on the (now first) 2024-01-01 row
        assert out["turnover"].iloc[0] == out["close"].iloc[0] * out["volume"].iloc[0]

    def test_missing_column_raises(self):
        raw = pd.DataFrame({"open": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"]))
        with pytest.raises(ValueError):
            finalize_ohlcv(raw)


# ── Yahoo adapter: pure conversion (no network) ─────────────────────────────


def _yf_frame() -> pd.DataFrame:
    """A frame shaped like yfinance.download output (single ticker)."""
    idx = pd.date_range("2024-01-01", periods=4, freq="D")  # tz-naive, like yf daily
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0],
            "High": [2.0, 3.0, 4.0, 5.0],
            "Low": [0.5, 1.5, 2.5, 3.5],
            "Close": [1.5, 2.5, 3.5, 4.5],
            "Adj Close": [1.4, 2.4, 3.4, 4.4],
            "Volume": [100, 200, 300, 400],
        },
        index=idx,
    )


class TestYahooToContract:
    def test_single_ticker_conversion(self):
        out = YahooProvider._to_contract(_yf_frame())
        assert list(out.columns) == list(CONTRACT_COLUMNS)   # adj close dropped, turnover added
        assert str(out.index.tz) == "UTC" and out.index.name == "timestamp"
        assert out["turnover"].iloc[0] == out["close"].iloc[0] * out["volume"].iloc[0]
        assert len(out) == 4

    def test_multiindex_columns_flattened(self):
        raw = _yf_frame()
        raw.columns = pd.MultiIndex.from_product([raw.columns, ["GC=F"]])  # (field, ticker)
        out = YahooProvider._to_contract(raw)
        assert list(out.columns) == list(CONTRACT_COLUMNS)
        assert len(out) == 4

    def test_drops_nan_ohlc_rows(self):
        raw = _yf_frame()
        raw.loc[raw.index[1], "Close"] = np.nan
        out = YahooProvider._to_contract(raw)
        assert len(out) == 3   # the NaN-close row is dropped

    def test_empty_returns_empty_contract(self):
        out = YahooProvider._to_contract(pd.DataFrame())
        assert list(out.columns) == list(CONTRACT_COLUMNS) and len(out) == 0


class TestYahooFetchGuards:
    def test_unsupported_interval_raises_before_network(self):
        # interval guard fires before the lazy yfinance import, so no network.
        with pytest.raises(ValueError, match="interval"):
            YahooProvider().fetch_klines(symbol="GC=F", interval="120")


# ── data_configurator wiring for a second provider ──────────────────────────


class TestDataConfiguratorProvider:
    def test_bybit_paths_unchanged(self, tmp_path):
        spec = DataSpec()  # provider defaults to bybit
        p = cache_path(spec, cache_dir=tmp_path)
        assert p.parts[-2] == "linear"   # data/ohlcv/<category>/file — no provider segment
        assert p.name == "BTCUSDT_15_last800.parquet"
        assert dataset_signature(spec) == "linear_BTCUSDT_15_last800"

    def test_yahoo_namespaced(self, tmp_path):
        spec = DataSpec(provider="yahoo", symbol="GC=F", interval="D", num_candles=800)
        parts = cache_path(spec, cache_dir=tmp_path).parts
        assert parts[-3] == "yahoo" and parts[-2] == "_"          # data/ohlcv/yahoo/_/...
        assert dataset_signature(spec) == "yahoo_na_GC=F_D_last800"

    def test_yahoo_unsupported_interval_rejected(self):
        with pytest.raises(ValueError, match="interval"):
            _validate(DataSpec(provider="yahoo", symbol="GC=F", interval="120"))

    def test_yahoo_supported_interval_ok(self):
        _validate(DataSpec(provider="yahoo", symbol="GC=F", interval="D"))  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
