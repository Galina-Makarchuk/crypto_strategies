"""Tests for LiveEngine wiring (offline — fake fetcher, no network).

Run with: pytest engine/tests/test_live.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.live import LiveEngine
from engine.strategy_configurator import StrategyConfig


class _StubStrategy:
    """Minimal strategy: LiveEngine._tick only needs name/prepare/on_bar."""

    name = "stub"

    def __init__(self):
        self.config = StrategyConfig()

    def prepare(self, df):
        return df

    def on_bar(self, i, df, state):
        return None


class _FakeFetcher:
    def __init__(self):
        self.calls: list[dict] = []

    def fetch_klines(self, **kwargs):
        self.calls.append(kwargs)
        idx = pd.DatetimeIndex(
            pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC").values,
            tz="UTC", name="timestamp",
        )
        cols = ["open", "high", "low", "close", "volume", "turnover"]
        return pd.DataFrame({c: np.arange(5, dtype=float) for c in cols}, index=idx)

    def close(self):
        pass


def _engine(tmp_path, category="linear"):
    return LiveEngine(
        strategy=_StubStrategy(),
        symbol="BTCUSDT",
        interval="15",
        category=category,
        num_candles=5,
        db_path=str(tmp_path / "state.db"),
        chart_path=str(tmp_path / "chart.html"),
    )


class TestLiveCategory:
    def test_category_stored(self, tmp_path):
        assert _engine(tmp_path, category="inverse").category == "inverse"

    def test_bad_category_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="category"):
            _engine(tmp_path, category="spot")

    def test_tick_forwards_category_to_fetcher(self, tmp_path, monkeypatch):
        monkeypatch.setattr("engine.live.build_chart", lambda *a, **k: None)
        engine = _engine(tmp_path, category="inverse")
        fake = _FakeFetcher()
        engine._fetcher = fake

        engine._tick()

        assert fake.calls[0]["category"] == "inverse"
