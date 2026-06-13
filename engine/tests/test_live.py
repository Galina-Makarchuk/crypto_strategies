"""Tests for LiveEngine wiring (offline — fake fetcher, no network).

Run with: pytest engine/tests/test_live.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.live import LiveEngine
from engine.strategy_configurator import EmaParams


class _StubStrategy:
    """Minimal strategy: LiveEngine._tick only needs name/prepare/on_bar."""

    name = "stub"

    def __init__(self):
        self.config = EmaParams()

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
        db_path=str(tmp_path / "live_records.db"),
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


class TestLiveEquityLayer:
    """Live mode runs the same compounding equity/sizing as the backtester,
    applied incrementally and persisted so the paper-equity curve survives
    restarts. (Previously live silently dropped all sizing/equity fields.)"""

    def _engine(self, tmp_path, db_name="eq.db", chart="c.html", equity=10_000.0):
        from engine.trade_configurator import TradingConfig

        return LiveEngine(
            strategy=_StubStrategy(),
            symbol="BTCUSDT",
            interval="15",
            num_candles=5,
            db_path=str(tmp_path / db_name),
            chart_path=str(tmp_path / chart),
            trading_config=TradingConfig(initial_equity=equity),
        )

    def test_closed_trade_compounds_sizes_and_persists(self, tmp_path):
        from engine.core import Direction, ExitReason

        eng = self._engine(tmp_path)
        start = eng._equity
        assert start == pytest.approx(10_000.0)  # seeded from initial_equity

        ts = pd.Timestamp("2026-01-01", tz="UTC")
        eng._state.enter(Direction.LONG, ts, 100.0)
        eng._state.exit(ts + pd.Timedelta(minutes=15), 110.0, ExitReason.SIGNAL_FLIP)
        eng._apply_equity_to_new_closures()

        trade = eng._state.closed_trades[-1]
        assert trade.notional > 0                 # was silently 0 before
        assert trade.pnl_currency > 0             # +10% move, net of cost
        assert trade.equity_after == pytest.approx(eng._equity)
        assert eng._equity > start                # equity compounded

    def test_sizing_is_idempotent_and_restart_safe(self, tmp_path):
        from engine.core import Direction, ExitReason

        eng = self._engine(tmp_path)
        ts = pd.Timestamp("2026-01-01", tz="UTC")
        eng._state.enter(Direction.LONG, ts, 100.0)
        eng._state.exit(ts + pd.Timedelta(minutes=15), 110.0, ExitReason.SIGNAL_FLIP)
        eng._apply_equity_to_new_closures()
        after_one = eng._equity
        trade = eng._state.closed_trades[-1]

        # A second pass must not double-count the same closed trade.
        eng._apply_equity_to_new_closures()
        assert eng._equity == pytest.approx(after_one)

        # Persist the trade, then a fresh engine on the same DB resumes the curve.
        eng._store.save_trade(trade)
        eng2 = self._engine(tmp_path, chart="c2.html")
        assert eng2._equity == pytest.approx(after_one)        # equity recovered
        assert trade.trade_id in eng2._sized_trade_ids         # not re-sized


class TestLiveChartTrades:
    """The live chart uses the rich trade view: closed trades in the visible
    window plus the open position's entry marker."""

    def _engine(self, tmp_path):
        return LiveEngine(
            strategy=_StubStrategy(), symbol="BTCUSDT", interval="15", num_candles=5,
            db_path=str(tmp_path / "ch.db"), chart_path=str(tmp_path / "ch.html"),
        )

    def test_chart_trades_includes_closed_in_window_and_open_entry(self, tmp_path):
        from engine.core import Direction, ExitReason

        eng = self._engine(tmp_path)
        t0 = pd.Timestamp("2026-01-01", tz="UTC")
        eng._state.enter(Direction.LONG, t0, 100.0)                         # closed
        eng._state.exit(t0 + pd.Timedelta(minutes=15), 110.0, ExitReason.SIGNAL_FLIP)
        eng._state.enter(Direction.SHORT, t0 + pd.Timedelta(minutes=30), 109.0)  # open

        ct = eng._chart_trades(window_start=t0 - pd.Timedelta(minutes=5))
        assert len(ct) == 2
        assert ct[-1] is eng._state.current_trade   # open trade -> entry marker only
        assert ct[-1].exit_ts is None

    def test_chart_trades_drops_out_of_window(self, tmp_path):
        from engine.core import Direction, ExitReason

        eng = self._engine(tmp_path)
        t0 = pd.Timestamp("2026-01-01", tz="UTC")
        eng._state.enter(Direction.LONG, t0, 100.0)
        eng._state.exit(t0 + pd.Timedelta(minutes=15), 110.0, ExitReason.SIGNAL_FLIP)

        # A window that starts after both the exit and (absent) open entry → empty.
        assert eng._chart_trades(window_start=t0 + pd.Timedelta(hours=1)) == []
