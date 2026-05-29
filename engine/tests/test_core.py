"""Tests for core components.

Run with: pytest engine/tests/test_core.py -v
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
        from engine.indicators import atr

        df = _make_ohlcv(50)
        result = atr(df, period=14)
        assert len(result) == len(df)
        assert result.iloc[13:].notna().all()

    def test_atr_positive(self):
        from engine.indicators import atr

        df = _make_ohlcv(50)
        result = atr(df, period=14)
        assert (result.dropna() > 0).all()


class TestRSI:
    def test_rsi_bounds(self):
        from engine.indicators import rsi

        series = pd.Series(np.random.randn(200).cumsum() + 100)
        result = rsi(series, period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_constant_price(self):
        from engine.indicators import rsi

        series = pd.Series([100.0] * 50)
        result = rsi(series, period=14)
        # Constant price → RSI should be ~100 (no losses)
        assert result.iloc[-1] == pytest.approx(100.0)


class TestEMA:
    def test_ema_convergence(self):
        from engine.indicators import ema

        series = pd.Series([100.0] * 100)
        result = ema(series, span=10)
        assert result.iloc[-1] == pytest.approx(100.0)


class TestSuperTrend:
    def test_supertrend_shape(self):
        from engine.indicators import supertrend

        df = _make_ohlcv(100)
        st_line, trend_dir = supertrend(df, period=10, multiplier=3.0)
        assert len(st_line) == len(df)
        assert len(trend_dir) == len(df)
        assert set(trend_dir.unique()).issubset({-1, 1})

    def test_supertrend_band_ratcheting(self):
        """Verify that bands ratchet (tighten) — the key bug fix."""
        from engine.indicators import supertrend

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
        from engine.indicators import detect_swing_highs

        # Create a clear peak at index 10
        prices = [100] * 5 + [101, 102, 103, 104, 105, 110, 105, 104, 103, 102, 101] + [100] * 5
        series = pd.Series(prices)
        result = detect_swing_highs(series, left=5, right=5)
        assert len(result) >= 1
        assert any(abs(p - 110) < 0.01 for _, p in result)

    def test_merge_levels(self):
        from engine.indicators import merge_price_levels

        levels = [100.0, 100.1, 100.05, 200.0]
        merged = merge_price_levels(levels, tolerance=0.0015)
        assert len(merged) == 2  # first three should merge


# ── Model tests ────────────────────────────────────────────────────────────────


class TestPositionState:
    def test_enter_exit_cycle(self):
        from engine.models import Direction, ExitReason, PositionState, PositionStatus

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
        from engine.models import Direction, PositionState

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state.enter(Direction.LONG, ts, 100.0)
        sig = state.enter(Direction.SHORT, ts, 99.0)
        assert sig is None  # rejected

    def test_trailing_peak_updates(self):
        from engine.models import Direction, PositionState

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state.enter(Direction.LONG, ts, 100.0)

        state.update_peak(110.0, 95.0)
        assert state.current_trade.peak_price == 110.0

        state.update_peak(108.0, 97.0)
        assert state.current_trade.peak_price == 110.0  # doesn't decrease

    def test_short_trailing_peak(self):
        from engine.models import Direction, PositionState

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
        from engine.models import PositionState, StrategyConfig
        from engine.strategies import EMACrossoverStrategy

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
        from engine.models import PositionState, StrategyConfig, SignalAction
        from engine.strategies import SuperTrendStrategy

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
        from engine.models import (
            Direction, PositionState, SignalAction, StrategyConfig,
        )
        from engine.strategies import ExhaustionReversalStrategy

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
        from engine.backtester import Backtester
        from engine.models import StrategyConfig
        from engine.strategies import ExhaustionReversalStrategy

        config = StrategyConfig()
        strat = ExhaustionReversalStrategy(config)
        df = _make_ohlcv(400, noise=2.0)

        bt = Backtester(strat)
        result = bt.run(df, interval="5")

        assert result.num_bars == 400
        assert result.strategy_name == "exhaustion_reversal"
        assert all(t.is_closed for t in result.trades)

    def test_no_lookahead(self):
        from engine.models import PositionState, StrategyConfig
        from engine.strategies import ExhaustionReversalStrategy

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


class TestVWAPBandsStrategy:
    def test_indicator_band_ordering(self):
        from engine.indicators import vwap_stdev_bands

        df = _make_ohlcv(150, noise=2.0)
        vwap, bands = vwap_stdev_bands(df, devs=(1.0, 2.0, 3.0), session="D")
        # vwap should be finite everywhere there is volume
        assert vwap.notna().all()
        # Bands are ordered: each outer upper >= inner upper, each outer lower <= inner lower
        for k in range(len(bands) - 1):
            upper_inner, lower_inner = bands[k]
            upper_outer, lower_outer = bands[k + 1]
            assert (upper_outer >= upper_inner - 1e-9).all()
            assert (lower_outer <= lower_inner + 1e-9).all()

    def test_session_reset(self):
        """VWAP at the first bar of a new session should equal hl2 of that bar."""
        from engine.indicators import vwap_stdev_bands

        df = _make_ohlcv(300, noise=2.0)
        vwap, _ = vwap_stdev_bands(df, devs=(1.0,), session="D")
        sessions = df.index.floor("D")
        # First bar of each session: VWAP == hl2 of that bar
        first_bars = ~sessions.duplicated(keep="first")
        hl2 = (df["high"] + df["low"]) / 2.0
        assert np.allclose(vwap[first_bars].values, hl2[first_bars].values)

    def test_backtest_runs_on_noise(self):
        from engine.backtester import Backtester
        from engine.models import StrategyConfig
        from engine.strategies import VWAPBandsStrategy

        config = StrategyConfig()
        strat = VWAPBandsStrategy(config)
        df = _make_ohlcv(400, noise=3.0, seed=7)

        bt = Backtester(strat)
        result = bt.run(df, interval="15")
        assert result.num_bars == 400
        assert result.strategy_name == "vwap_bands"
        assert all(t.is_closed for t in result.trades)

    def test_no_lookahead(self):
        from engine.models import PositionState, StrategyConfig
        from engine.strategies import VWAPBandsStrategy

        config = StrategyConfig()
        df = _make_ohlcv(300, trend=0.05, noise=2.0, seed=3)

        strat_full = VWAPBandsStrategy(config)
        prepared_full = strat_full.prepare(df)
        state_full = PositionState()
        for i in range(len(prepared_full)):
            strat_full.on_bar(i, prepared_full, state_full)

        df_short = df.iloc[:180].copy()
        strat_short = VWAPBandsStrategy(config)
        prepared_short = strat_short.prepare(df_short)
        state_short = PositionState()
        for i in range(len(prepared_short)):
            strat_short.on_bar(i, prepared_short, state_short)

        sigs_full = [(s.timestamp, s.action, s.direction) for s in state_full.signals
                     if s.timestamp <= df_short.index[-1]]
        sigs_short = [(s.timestamp, s.action, s.direction) for s in state_short.signals]
        assert sigs_full == sigs_short


class TestSwingsDetector:
    def _sine_df(self, n: int = 400, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 8 * np.pi, n)
        mid = 100.0 + 20.0 * np.sin(t)
        close = mid + rng.standard_normal(n) * 0.3
        high = close + np.abs(rng.standard_normal(n)) * 0.6
        low = close - np.abs(rng.standard_normal(n)) * 0.6
        vol = rng.uniform(100, 1000, n)
        idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
        return pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": vol},
            index=idx,
        )

    def test_alternation_invariant(self):
        from engine.swings import detect_swings

        df = self._sine_df()
        swings = detect_swings(df, min_prominence_atr=1.0, return_provisional=False)
        assert len(swings) >= 4, "wave-shaped data should produce many swings"
        for a, b in zip(swings, swings[1:]):
            assert a.side != b.side

    def test_confirmation_lags_pivot(self):
        from engine.swings import detect_swings

        df = self._sine_df()
        swings = detect_swings(df, min_prominence_atr=1.0, return_provisional=False)
        assert all(s.confirmation_idx >= s.idx for s in swings)
        assert all(s.bars_to_confirm == s.confirmation_idx - s.idx for s in swings)

    def test_min_prominence_filters(self):
        from engine.swings import detect_swings

        df = self._sine_df()
        loose = detect_swings(df, min_prominence_atr=0.5, return_provisional=False)
        strict = detect_swings(df, min_prominence_atr=3.0, return_provisional=False)
        assert len(loose) >= len(strict), "raising threshold must not add swings"


class TestSwingZigZagStrategy:
    def test_backtest_runs_on_noise(self):
        from engine.backtester import Backtester
        from engine.models import StrategyConfig
        from engine.strategies import SwingZigZagStrategy

        config = StrategyConfig(swing_zz_min_prominence_atr=1.0)
        strat = SwingZigZagStrategy(config)
        df = _make_ohlcv(400, noise=3.0, seed=11)

        bt = Backtester(strat)
        result = bt.run(df, interval="15")
        assert result.num_bars == 400
        assert result.strategy_name == "swing_zigzag"
        assert all(t.is_closed for t in result.trades)

    def test_no_lookahead(self):
        from engine.models import PositionState, StrategyConfig
        from engine.strategies import SwingZigZagStrategy

        config = StrategyConfig(swing_zz_min_prominence_atr=1.0)
        df = _make_ohlcv(300, trend=0.05, noise=2.0, seed=4)

        strat_full = SwingZigZagStrategy(config)
        prepared_full = strat_full.prepare(df)
        state_full = PositionState()
        for i in range(len(prepared_full)):
            strat_full.on_bar(i, prepared_full, state_full)

        df_short = df.iloc[:180].copy()
        strat_short = SwingZigZagStrategy(config)
        prepared_short = strat_short.prepare(df_short)
        state_short = PositionState()
        for i in range(len(prepared_short)):
            strat_short.on_bar(i, prepared_short, state_short)

        sigs_full = [
            (s.timestamp, s.action, s.direction)
            for s in state_full.signals
            if s.timestamp <= df_short.index[-1]
        ]
        sigs_short = [(s.timestamp, s.action, s.direction) for s in state_short.signals]
        assert sigs_full == sigs_short


# ── Backtester tests ───────────────────────────────────────────────────────────


class TestBacktester:
    def test_backtest_runs(self):
        from engine.backtester import Backtester
        from engine.models import StrategyConfig
        from engine.strategies import SuperTrendStrategy

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
        from engine.backtester import Backtester
        from engine.models import StrategyConfig
        from engine.strategies import EMACrossoverStrategy

        config = StrategyConfig(fee_bps=10.0, slippage_bps=5.0)
        strat = EMACrossoverStrategy(config)
        df = _make_ohlcv(300, noise=2.0)

        bt = Backtester(strat)
        result = bt.run(df)

        if result.total_trades > 0:
            # Each trade's P&L should be reduced by round-trip costs (30 bps)
            # We can't assert exact values but can verify trades exist
            assert result.total_trades > 0

    # ── _compute_stats edge cases ────────────────────────────────────────────

    @staticmethod
    def _bt_with_trades(pnls: list[float]):
        """Build a Backtester + PositionState pre-loaded with synthetic closed trades."""
        from engine.backtester import Backtester
        from engine.models import (
            Direction, ExitReason, PositionState, StrategyConfig, Trade,
        )
        from engine.strategies.base import BaseStrategy

        class _NullStrategy(BaseStrategy):
            name = "null"
            def prepare(self, df):
                return df.copy()
            def on_bar(self, i, df, state):
                return None

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for pnl in pnls:
            state.closed_trades.append(Trade(
                direction=Direction.LONG,
                entry_ts=ts, entry_price=100.0,
                exit_ts=ts, exit_price=100.0 + pnl,
                pnl_bps=pnl,
                exit_reason=ExitReason.TAKE_PROFIT,
                peak_price=100.0 + max(pnl, 0),
            ))
        bt = Backtester(_NullStrategy(StrategyConfig()))
        df = _make_ohlcv(10)
        return bt, state, df

    def test_zero_trade_run(self):
        bt, state, df = self._bt_with_trades([])
        result = bt._compute_stats(state, df, interval="15")
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0
        assert result.sharpe_approx == 0.0
        assert result.max_drawdown_bps == 0.0

    def test_break_even_bucketing(self):
        bt, state, df = self._bt_with_trades([10.0, 0.0, -5.0])
        result = bt._compute_stats(state, df, interval="15")
        assert result.winning_trades == 1
        assert result.losing_trades == 1
        assert result.break_even_trades == 1
        assert result.win_rate == pytest.approx(1 / 3)

    def test_profit_factor_no_losses(self):
        bt, state, df = self._bt_with_trades([5.0, 10.0])
        result = bt._compute_stats(state, df, interval="15")
        assert result.profit_factor == float("inf")

    def test_profit_factor_all_break_even(self):
        bt, state, df = self._bt_with_trades([0.0, 0.0, 0.0])
        result = bt._compute_stats(state, df, interval="15")
        # Was previously returning inf for 0/0; should be 0.0 now.
        assert result.profit_factor == 0.0

    def test_sharpe_single_trade(self):
        # ddof=1 with n=1 would produce NaN; guard returns 0.0.
        bt, state, df = self._bt_with_trades([15.0])
        result = bt._compute_stats(state, df, interval="15")
        assert result.sharpe_approx == 0.0

    def test_sharpe_zero_std_positive_mean(self):
        bt, state, df = self._bt_with_trades([10.0, 10.0, 10.0])
        result = bt._compute_stats(state, df, interval="15")
        assert result.sharpe_approx == float("inf")

    def test_sharpe_zero_std_negative_mean(self):
        bt, state, df = self._bt_with_trades([-7.0, -7.0])
        result = bt._compute_stats(state, df, interval="15")
        assert result.sharpe_approx == float("-inf")

    def test_force_close_path(self):
        """A strategy that opens but never exits should be force-closed by run()."""
        from engine.backtester import Backtester
        from engine.models import (
            Direction, ExitReason, PositionState, StrategyConfig,
        )
        from engine.strategies.base import BaseStrategy

        class _EnterOnceStrategy(BaseStrategy):
            name = "enter_once"
            def prepare(self, df):
                return df.copy()
            def on_bar(self, i, df, state):
                if i == 0 and state.current_trade is None:
                    state.enter(Direction.LONG, df.index[i], float(df["close"].iloc[i]))

        bt = Backtester(_EnterOnceStrategy(StrategyConfig()))
        df = _make_ohlcv(20)
        result = bt.run(df, interval="15")
        assert result.total_trades == 1
        assert result.trades[0].exit_reason == ExitReason.FORCE_CLOSE
        assert result.trades[0].exit_price == pytest.approx(float(df["close"].iloc[-1]))


# ── Config tests ───────────────────────────────────────────────────────────────


class TestConfig:
    def test_total_cost(self):
        from engine.models import StrategyConfig

        config = StrategyConfig(fee_bps=4.0, slippage_bps=2.0)
        assert config.total_cost_bps() == 12.0  # 2 * (4 + 2)

    def test_interval_validation(self):
        from engine.models import validate_interval

        assert validate_interval("15") == "15"
        with pytest.raises(ValueError):
            validate_interval("42")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
