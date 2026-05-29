"""Tests for the data_configurator single-source-of-truth loader.

Run with: pytest engine/tests/test_data_configurator.py -v

All tests are offline: a fake fetcher stands in for BybitFetcher so nothing
hits the network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine.data_configurator import (
    DataSpec,
    cache_path,
    dataset_signature,
    load_data,
    save_result,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_klines(n: int = 50, seed: int = 7) -> pd.DataFrame:
    """A frame shaped exactly like BybitFetcher.fetch_klines output."""
    rng = np.random.RandomState(seed)
    closes = 100.0 + rng.randn(n).cumsum()
    # Mirror the real fetcher: a plain UTC DatetimeIndex with no `freq` set
    # (it is built from deduped API rows, not a date_range).
    idx = pd.DatetimeIndex(
        pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC").values,
        tz="UTC",
        name="timestamp",
    )
    return pd.DataFrame(
        {
            "open": closes + rng.randn(n) * 0.1,
            "high": closes + rng.uniform(0.1, 1.0, n),
            "low": closes - rng.uniform(0.1, 1.0, n),
            "close": closes,
            "volume": rng.uniform(10, 100, n),
            "turnover": rng.uniform(1e5, 1e6, n),
        },
        index=idx,
    ).astype(float)


class _FakeFetcher:
    """Records calls and returns synthetic candles instead of hitting Bybit."""

    def __init__(self, df: pd.DataFrame | None = None):
        self._df = df if df is not None else _make_klines()
        self.calls: list[dict] = []

    def fetch_klines(self, **kwargs) -> pd.DataFrame:
        self.calls.append(kwargs)
        return self._df.copy()

    def close(self) -> None:  # parity with the real fetcher
        pass


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestCachePath:
    def test_count_mode_filename(self, tmp_path):
        spec = DataSpec(symbol="BTCUSDT", interval="15", num_candles=800)
        p = cache_path(spec, cache_dir=tmp_path)
        assert p == tmp_path / "linear" / "BTCUSDT_15_last800.parquet"

    def test_range_mode_filename(self, tmp_path):
        spec = DataSpec(symbol="ETHUSDT", interval="60", start="2026-03-20", end="2026-04-19")
        p = cache_path(spec, cache_dir=tmp_path)
        assert p == tmp_path / "linear" / "ETHUSDT_60_2026-03-20_2026-04-19.parquet"

    def test_category_splits_directory(self, tmp_path):
        spec = DataSpec(symbol="BTCUSD", interval="D", category="inverse", num_candles=100)
        p = cache_path(spec, cache_dir=tmp_path)
        assert p.parent.name == "inverse"


class TestLoadAndCache:
    def test_fetch_writes_parquet_and_sidecar(self, tmp_path):
        fake = _FakeFetcher()
        spec = DataSpec(num_candles=50)
        df = load_data(spec, cache_dir=tmp_path, fetcher=fake)

        assert len(df) == 50
        assert len(fake.calls) == 1
        # category is threaded into the fetch call
        assert fake.calls[0]["category"] == "linear"

        data_path = cache_path(spec, cache_dir=tmp_path)
        assert data_path.exists()
        meta = json.loads(data_path.with_suffix(".json").read_text())
        assert meta["mode"] == "count"
        assert meta["fetched_rows"] == 50

    def test_second_call_hits_cache_no_refetch(self, tmp_path):
        fake = _FakeFetcher()
        spec = DataSpec(start="2026-01-01", end="2026-01-02")  # pinned range = immutable

        first = load_data(spec, cache_dir=tmp_path, fetcher=fake)
        second = load_data(spec, cache_dir=tmp_path, fetcher=fake)

        assert len(fake.calls) == 1  # second read served from cache
        pd.testing.assert_frame_equal(first, second)

    def test_refresh_forces_refetch(self, tmp_path):
        fake = _FakeFetcher()
        spec = DataSpec(start="2026-01-01", end="2026-01-02")

        load_data(spec, cache_dir=tmp_path, fetcher=fake)
        load_data(spec, cache_dir=tmp_path, fetcher=fake, refresh=True)

        assert len(fake.calls) == 2

    def test_contract_preserved_through_round_trip(self, tmp_path):
        fake = _FakeFetcher()
        spec = DataSpec(start="2026-01-01", end="2026-01-02")

        load_data(spec, cache_dir=tmp_path, fetcher=fake)  # writes cache
        cached = load_data(spec, cache_dir=tmp_path, fetcher=fake)  # reads cache

        assert isinstance(cached.index, pd.DatetimeIndex)
        assert str(cached.index.tz) == "UTC"
        assert cached.index.name == "timestamp"
        assert list(cached.columns) == ["open", "high", "low", "close", "volume", "turnover"]
        assert all(str(dt) == "float64" for dt in cached.dtypes)


def _age_sidecar(data_path, fetched_at: str = "2020-01-01T00:00:00+00:00") -> None:
    """Rewrite a cache's .json sidecar so it looks stale."""
    meta_path = data_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta["fetched_at"] = fetched_at
    meta_path.write_text(json.dumps(meta))


class TestCacheFreshness:
    def test_count_mode_fresh_within_interval_reuses(self, tmp_path):
        fake = _FakeFetcher()
        spec = DataSpec(num_candles=50)  # count mode
        load_data(spec, cache_dir=tmp_path, fetcher=fake)
        load_data(spec, cache_dir=tmp_path, fetcher=fake)
        assert len(fake.calls) == 1  # fetched_at ~now → fresh → cache hit

    def test_count_mode_refetches_when_stale(self, tmp_path):
        fake = _FakeFetcher()
        spec = DataSpec(num_candles=50)
        load_data(spec, cache_dir=tmp_path, fetcher=fake)
        _age_sidecar(cache_path(spec, cache_dir=tmp_path))
        load_data(spec, cache_dir=tmp_path, fetcher=fake)
        assert len(fake.calls) == 2  # stale → refetch

    def test_open_ended_range_refetches_when_stale(self, tmp_path):
        fake = _FakeFetcher()
        spec = DataSpec(start="2026-01-01", end=None)  # open-ended → age-based
        load_data(spec, cache_dir=tmp_path, fetcher=fake)
        _age_sidecar(cache_path(spec, cache_dir=tmp_path))
        load_data(spec, cache_dir=tmp_path, fetcher=fake)
        assert len(fake.calls) == 2

    def test_pinned_range_immutable_even_when_stale(self, tmp_path):
        fake = _FakeFetcher()
        spec = DataSpec(start="2026-01-01", end="2026-01-02")  # pinned → immutable
        load_data(spec, cache_dir=tmp_path, fetcher=fake)
        _age_sidecar(cache_path(spec, cache_dir=tmp_path))
        load_data(spec, cache_dir=tmp_path, fetcher=fake)
        assert len(fake.calls) == 1  # never stale


class TestCorruptionRecovery:
    def test_truncated_parquet_triggers_refetch(self, tmp_path):
        fake = _FakeFetcher()
        spec = DataSpec(start="2026-01-01", end="2026-01-02")  # pinned: _is_fresh always True
        load_data(spec, cache_dir=tmp_path, fetcher=fake)

        # Simulate an interrupted write: corrupt the parquet in place.
        data_path = cache_path(spec, cache_dir=tmp_path)
        data_path.write_bytes(b"not a parquet file")

        df = load_data(spec, cache_dir=tmp_path, fetcher=fake)
        assert len(fake.calls) == 2          # unreadable cache → self-healed via refetch
        assert len(df) == 50                 # and returned good data
        assert not data_path.with_name(data_path.name + ".part").exists()  # temp cleaned up


class TestDatasetSignature:
    def test_count_mode_signature(self):
        spec = DataSpec(symbol="BTCUSDT", interval="15", num_candles=800)
        assert dataset_signature(spec) == "linear_BTCUSDT_15_last800"

    def test_range_and_category_signature(self):
        spec = DataSpec(symbol="ETHUSDT", interval="60", category="inverse",
                        start="2026-03-20", end="2026-04-19")
        assert dataset_signature(spec) == "inverse_ETHUSDT_60_2026-03-20_2026-04-19"


def _make_result():
    """A real BacktestResult with one winning trade and an inf profit factor."""
    from datetime import datetime, timezone

    from engine.backtester import BacktestResult
    from engine.models import Direction, ExitReason, Trade

    trade = Trade(
        direction=Direction.LONG,
        entry_ts=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        entry_price=100.0,
        exit_ts=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
        exit_price=101.0,
        pnl_bps=50.0,
        peak_price=102.0,
        exit_reason=ExitReason.TAKE_PROFIT,
    )
    return BacktestResult(
        strategy_name="supertrend",
        symbol="BTCUSDT",
        interval="15",
        num_bars=800,
        total_trades=1,
        winning_trades=1,
        win_rate=1.0,
        total_pnl_bps=50.0,
        avg_pnl_bps=50.0,
        max_win_bps=50.0,
        profit_factor=float("inf"),   # all wins → non-finite
        sharpe_approx=float("inf"),
        trades=[trade],
    )


class TestSaveResult:
    def test_writes_json_and_csv(self, tmp_path):
        spec = DataSpec(num_candles=800)
        json_path = save_result(_make_result(), spec, results_dir=tmp_path)

        assert json_path == tmp_path / "linear_BTCUSDT_15_last800" / "supertrend.json"
        csv_path = json_path.with_name("supertrend_trades.csv")
        assert json_path.exists() and csv_path.exists()

    def test_json_is_valid_and_finite(self, tmp_path):
        spec = DataSpec(num_candles=800)
        raw = save_result(_make_result(), spec, results_dir=tmp_path).read_text()
        # Strictly valid JSON: no Infinity/NaN tokens.
        assert "Infinity" not in raw and "NaN" not in raw
        payload = json.loads(raw)
        assert payload["metrics"]["profit_factor"] is None   # inf → null
        assert payload["metrics"]["win_rate"] == 1.0
        assert payload["dataset_signature"] == "linear_BTCUSDT_15_last800"

    def test_trade_fields_serialized(self, tmp_path):
        spec = DataSpec(num_candles=800)
        payload = json.loads(save_result(_make_result(), spec, results_dir=tmp_path).read_text())
        t = payload["trades"][0]
        assert t["direction"] == "long"                 # enum → value
        assert t["exit_reason"] == "take_profit"
        assert t["entry_ts"] == "2026-01-01T00:00:00+00:00"
        assert t["duration_seconds"] == 3600.0          # 1h

    def test_trades_csv_loads_as_table(self, tmp_path):
        spec = DataSpec(num_candles=800)
        json_path = save_result(_make_result(), spec, results_dir=tmp_path)
        df = pd.read_csv(json_path.with_name("supertrend_trades.csv"))
        assert len(df) == 1
        assert list(df.columns)[:3] == ["trade_id", "direction", "entry_ts"]

    def test_empty_trades_csv_has_header(self, tmp_path):
        from engine.backtester import BacktestResult

        spec = DataSpec(num_candles=800)
        empty = BacktestResult(strategy_name="ema", symbol="BTCUSDT", interval="15")
        json_path = save_result(empty, spec, results_dir=tmp_path)
        df = pd.read_csv(json_path.with_name("ema_trades.csv"))
        assert len(df) == 0
        assert "trade_id" in df.columns       # header present even with no trades


class TestValidation:
    def test_bad_category_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="category"):
            load_data(DataSpec(category="spot"), cache_dir=tmp_path, fetcher=_FakeFetcher())

    def test_bad_interval_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="[Ii]nterval"):
            load_data(DataSpec(interval="7"), cache_dir=tmp_path, fetcher=_FakeFetcher())

    def test_non_positive_count_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="num_candles"):
            load_data(DataSpec(num_candles=0), cache_dir=tmp_path, fetcher=_FakeFetcher())
