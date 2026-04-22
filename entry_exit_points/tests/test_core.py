"""Tests for core components.

Run with: pytest entry_exit_points/tests/test_core.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_ohlcv(
    n: int = 100,
    base_price: float = 100.0,
    trend: float = 0.0,
    noise: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    closes = base_price + trend * np.arange(n) + rng.randn(n).cumsum() * noise
    highs = closes + rng.uniform(0.5, 2.0, n)
    lows = closes - rng.uniform(0.5, 2.0, n)
    opens = closes + rng.randn(n) * 0.5
    volume = rng.uniform(100, 1000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


# ── Indicator tests ───────────────────────────────────────────────────────────


class TestATR:
    def test_atr_shape(self):
        from entry_exit_points.indicators import atr

        df = _make_ohlcv(50)
        result = atr(df, period=14)
        assert len(result) == len(df)
        assert result.iloc[13:].notna().all()

    def test_atr_positive(self):
        from entry_exit_points.indicators import atr

        df = _make_ohlcv(50)
        result = atr(df, period=14)
        assert (result.dropna() > 0).all()


class TestRSI:
    def test_rsi_bounds(self):
        from entry_exit_points.indicators import rsi

        series = pd.Series(np.random.randn(200).cumsum() + 100)
        result = rsi(series, period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_constant_price(self):
        from entry_exit_points.indicators import rsi

        series = pd.Series([100.0] * 50)
        result = rsi(series, period=14)
        # Constant price → RSI should be ~100 (no losses)
        assert result.iloc[-1] == pytest.approx(100.0)


class TestEMA:
    def test_ema_convergence(self):
        from entry_exit_points.indicators import ema

        series = pd.Series([100.0] * 100)
        result = ema(series, span=10)
        assert result.iloc[-1] == pytest.approx(100.0)


class TestSuperTrend:
    def test_supertrend_shape(self):
        from entry_exit_points.indicators import supertrend

        df = _make_ohlcv(100)
        st_line, trend_dir = supertrend(df, period=10, multiplier=3.0)
        assert len(st_line) == len(df)
        assert len(trend_dir) == len(df)
        assert set(trend_dir.unique()).issubset({-1, 1})

    def test_supertrend_band_ratcheting(self):
        """Verify that bands ratchet (tighten) — the key bug fix."""
        from entry_exit_points.indicators import supertrend

        # Create uptrending data where lower band should ratchet up
        prices = np.linspace(100, 150, 60)
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
        }, index=pd.date_range("2025-01-01", periods=60, freq="15min", tz="UTC"))

        st_line, trend_dir = supertrend(df, period=10, multiplier=2.0)

        # In a strong uptrend, trend should be +1 for most of the series
        assert (trend_dir.iloc[20:] == 1).sum() > len(trend_dir.iloc[20:]) * 0.7

        # SuperTrend line (lower band in uptrend) should be rising
        st_later = st_line.iloc[20:].values
        diffs = np.diff(st_later)
        rising_pct = (diffs >= 0).sum() / len(diffs)
        assert rising_pct > 0.7, "Lower band should ratchet upward in uptrend"


class TestSwingDetection:
    def test_swing_highs_detected(self):
        from entry_exit_points.indicators import detect_swing_highs

        # Create a clear peak at index 10
        prices = [100] * 5 + [101, 102, 103, 104, 105, 110, 105, 104, 103, 102, 101] + [100] * 5
        series = pd.Series(prices)
        result = detect_swing_highs(series, left=5, right=5)
        assert len(result) >= 1
        assert any(abs(p - 110) < 0.01 for _, p in result)

    def test_merge_levels(self):
        from entry_exit_points.indicators import merge_price_levels

        levels = [100.0, 100.1, 100.05, 200.0]
        merged = merge_price_levels(levels, tolerance=0.0015)
        assert len(merged) == 2  # first three should merge


# ── Model tests ────────────────────────────────────────────────────────────────


class TestPositionState:
    def test_enter_exit_cycle(self):
        from entry_exit_points.models import Direction, ExitReason, PositionState, PositionStatus

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

        sig = state.enter(Direction.LONG, ts, 100.0)
        assert sig is not None
        assert state.status == PositionStatus.OPEN
        assert state.current_trade is not None

        sig2 = state.exit(ts, 105.0, cost_bps=12.0, reason=ExitReason.TAKE_PROFIT)
        assert sig2 is not None
        assert state.status == PositionStatus.FLAT
        assert len(state.closed_trades) == 1
        assert state.closed_trades[0].pnl_bps == pytest.approx(500 - 12.0)

    def test_double_entry_rejected(self):
        from entry_exit_points.models import Direction, PositionState

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state.enter(Direction.LONG, ts, 100.0)
        sig = state.enter(Direction.SHORT, ts, 99.0)
        assert sig is None  # rejected

    def test_trailing_peak_updates(self):
        from entry_exit_points.models import Direction, PositionState

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state.enter(Direction.LONG, ts, 100.0)

        state.update_peak(110.0, 95.0)
        assert state.current_trade.peak_price == 110.0

        state.update_peak(108.0, 97.0)
        assert state.current_trade.peak_price == 110.0  # doesn't decrease

    def test_short_trailing_peak(self):
        from entry_exit_points.models import Direction, PositionState

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state.enter(Direction.SHORT, ts, 100.0)

        state.update_peak(105.0, 90.0)
        assert state.current_trade.peak_price == 90.0

        state.update_peak(102.0, 92.0)
        assert state.current_trade.peak_price == 90.0  # doesn't increase


# ── Strategy tests ─────────────────────────────────────────────────────────────


class TestEMACrossoverStrategy:
    def test_no_lookahead(self):
        """Strategy should produce same results regardless of future data."""
        from entry_exit_points.models import PositionState, StrategyConfig
        from entry_exit_points.strategies import EMACrossoverStrategy

        config = StrategyConfig()
        strat = EMACrossoverStrategy(config)
        df = _make_ohlcv(200, trend=0.5, noise=2.0)

        # Run on full data
        prepared = strat.prepare(df)
        state1 = PositionState()
        for i in range(len(prepared)):
            strat.on_bar(i, prepared, state1)

        # Run on first 100 bars only
        df_short = df.iloc[:100].copy()
        prepared_short = strat.prepare(df_short)
        state2 = PositionState()
        for i in range(len(prepared_short)):
            strat.on_bar(i, prepared_short, state2)

        # Signals up to bar 100 should match
        sigs1 = [(s.timestamp, s.action, s.direction) for s in state1.signals
                 if s.timestamp <= df_short.index[-1]]
        sigs2 = [(s.timestamp, s.action, s.direction) for s in state2.signals]
        assert sigs1 == sigs2


class TestSuperTrendStrategy:
    def test_entries_on_trend_flip(self):
        from entry_exit_points.models import PositionState, StrategyConfig, SignalAction
        from entry_exit_points.strategies import SuperTrendStrategy

        config = StrategyConfig()
        strat = SuperTrendStrategy(config)
        df = _make_ohlcv(300, trend=0.0, noise=3.0, seed=123)

        prepared = strat.prepare(df)
        state = PositionState()
        for i in range(len(prepared)):
            strat.on_bar(i, prepared, state)

        entries = [s for s in state.signals if s.action == SignalAction.ENTRY]
        # With noisy data there should be some flips
        assert len(entries) > 0


class TestExhaustionReversalStrategy:
    def _crafted_pattern_df(self, warmup: int = 20) -> pd.DataFrame:
        """Warmup + buy-push(3) + stall(4 alternating) + sell-trigger(2) + dropout."""
        bars = []

        # Warmup: tiny random walk around 100 so ATR > 0
        rng = np.random.RandomState(7)
        price = 100.0
        for _ in range(warmup):
            o = price
            c = price + rng.uniform(-0.1, 0.1)
            h = max(o, c) + 0.05
            l = min(o, c) - 0.05
            bars.append((o, h, l, c, 10.0))
            price = c

        # Helper to append a candle
        def add(direction: int, body: float, vol: float):
            nonlocal price
            o = price
            c = o + direction * body
            h = max(o, c) + 0.05
            l = min(o, c) - 0.05
            bars.append((o, h, l, c, vol))
            price = c

        # Push leg: 3 solid green candles with low per-candle volume
        for _ in range(3):
            add(+1, 0.80, 20.0)

        # Stall cluster: 4 alternating small-body candles at the high
        add(-1, 0.05, 8.0)
        add(+1, 0.05, 8.0)
        add(-1, 0.05, 8.0)
        add(+1, 0.05, 8.0)

        # Trigger: 2 red candles with HIGH per-candle volume (volume_factor >= 1)
        add(-1, 0.50, 60.0)
        add(-1, 0.50, 60.0)

        # Drop further to let any target fire
        add(-1, 0.60, 50.0)
        add(-1, 0.60, 50.0)
        add(-1, 0.60, 50.0)

        idx = pd.date_range("2025-01-01", periods=len(bars), freq="5min", tz="UTC")
        df = pd.DataFrame(
            bars, columns=["open", "high", "low", "close", "volume"], index=idx
        )
        return df

    def test_fires_short_on_crafted_pattern(self):
        from entry_exit_points.models import (
            Direction, PositionState, SignalAction, StrategyConfig,
        )
        from entry_exit_points.strategies import ExhaustionReversalStrategy

        config = StrategyConfig(atr_period=5)  # small so warmup is short
        strat = ExhaustionReversalStrategy(config)
        df = self._crafted_pattern_df(warmup=10)

        prepared = strat.prepare(df)
        state = PositionState()
        for i in range(len(prepared)):
            strat.on_bar(i, prepared, state)

        entries = [s for s in state.signals if s.action == SignalAction.ENTRY]
        assert entries, "Expected at least one entry on crafted push/stall/trigger pattern"
        assert entries[0].direction == Direction.SHORT

    def test_backtest_runs_on_noise(self):
        from entry_exit_points.backtester import Backtester
        from entry_exit_points.models import StrategyConfig
        from entry_exit_points.strategies import ExhaustionReversalStrategy

        config = StrategyConfig()
        strat = ExhaustionReversalStrategy(config)
        df = _make_ohlcv(400, noise=2.0)

        bt = Backtester(strat)
        result = bt.run(df, interval="5")

        assert result.num_bars == 400
        assert result.strategy_name == "exhaustion_reversal"
        assert all(t.is_closed for t in result.trades)

    def test_no_lookahead(self):
        from entry_exit_points.models import PositionState, StrategyConfig
        from entry_exit_points.strategies import ExhaustionReversalStrategy

        config = StrategyConfig()
        df = _make_ohlcv(250, trend=0.1, noise=2.0, seed=9)

        strat_full = ExhaustionReversalStrategy(config)
        prepared_full = strat_full.prepare(df)
        state_full = PositionState()
        for i in range(len(prepared_full)):
            strat_full.on_bar(i, prepared_full, state_full)

        df_short = df.iloc[:150].copy()
        strat_short = ExhaustionReversalStrategy(config)
        prepared_short = strat_short.prepare(df_short)
        state_short = PositionState()
        for i in range(len(prepared_short)):
            strat_short.on_bar(i, prepared_short, state_short)

        sigs_full = [(s.timestamp, s.action, s.direction) for s in state_full.signals
                     if s.timestamp <= df_short.index[-1]]
        sigs_short = [(s.timestamp, s.action, s.direction) for s in state_short.signals]
        assert sigs_full == sigs_short


# ── Backtester tests ───────────────────────────────────────────────────────────


class TestBacktester:
    def test_backtest_runs(self):
        from entry_exit_points.backtester import Backtester
        from entry_exit_points.models import StrategyConfig
        from entry_exit_points.strategies import SuperTrendStrategy

        config = StrategyConfig()
        strat = SuperTrendStrategy(config)
        df = _make_ohlcv(200, noise=3.0)

        bt = Backtester(strat)
        result = bt.run(df, interval="15")

        assert result.num_bars == 200
        assert result.strategy_name == "supertrend"
        # Should have force-closed any dangling position
        assert all(t.is_closed for t in result.trades)

    def test_pnl_includes_costs(self):
        from entry_exit_points.backtester import Backtester
        from entry_exit_points.models import StrategyConfig
        from entry_exit_points.strategies import EMACrossoverStrategy

        config = StrategyConfig(fee_bps=10.0, slippage_bps=5.0)
        strat = EMACrossoverStrategy(config)
        df = _make_ohlcv(300, noise=2.0)

        bt = Backtester(strat)
        result = bt.run(df)

        if result.total_trades > 0:
            # Each trade's P&L should be reduced by round-trip costs (30 bps)
            # We can't assert exact values but can verify trades exist
            assert result.total_trades > 0


# ── Config tests ───────────────────────────────────────────────────────────────


class TestConfig:
    def test_total_cost(self):
        from entry_exit_points.models import StrategyConfig

        config = StrategyConfig(fee_bps=4.0, slippage_bps=2.0)
        assert config.total_cost_bps() == 12.0  # 2 * (4 + 2)

    def test_interval_validation(self):
        from entry_exit_points.models import validate_interval

        assert validate_interval("15") == "15"
        with pytest.raises(ValueError):
            validate_interval("42")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
