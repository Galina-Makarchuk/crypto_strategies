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
        # Constant price → no gains AND no losses → neutral 50, not overbought.
        # (RSI 100 is reserved for genuine "all gains, no losses" runs.)
        assert result.iloc[-1] == pytest.approx(50.0)

    def test_rsi_monotonic_gains_is_overbought(self):
        from engine.indicators import rsi

        # A pure uptrend (gains, no losses) is the real RSI-100 case.
        result = rsi(pd.Series(np.arange(1.0, 51.0)), period=14)
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
        from engine.core import Direction, ExitReason, PositionState, PositionStatus

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

        # Cost is now seeded on the state by the runner, not passed to exit().
        state.cost_bps = 12.0

        sig = state.enter(Direction.LONG, ts, 100.0)
        assert sig is not None
        assert state.status == PositionStatus.OPEN
        assert state.current_trade is not None

        sig2 = state.exit(ts, 105.0, reason=ExitReason.TAKE_PROFIT)
        assert sig2 is not None
        assert state.status == PositionStatus.FLAT
        assert len(state.closed_trades) == 1
        assert state.closed_trades[0].pnl_bps == pytest.approx(500 - 12.0)

    def test_double_entry_rejected(self):
        from engine.core import Direction, PositionState

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state.enter(Direction.LONG, ts, 100.0)
        sig = state.enter(Direction.SHORT, ts, 99.0)
        assert sig is None  # rejected

    def test_trailing_peak_updates(self):
        from engine.core import Direction, PositionState

        state = PositionState()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state.enter(Direction.LONG, ts, 100.0)

        state.update_peak(110.0, 95.0)
        assert state.current_trade.peak_price == 110.0

        state.update_peak(108.0, 97.0)
        assert state.current_trade.peak_price == 110.0  # doesn't decrease

    def test_short_trailing_peak(self):
        from engine.core import Direction, PositionState

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
        from engine.core import PositionState
        from engine.strategy_configurator import StrategyConfig
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
        from engine.core import PositionState, SignalAction
        from engine.strategy_configurator import StrategyConfig
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
        from engine.core import Direction, PositionState, SignalAction
        from engine.strategy_configurator import StrategyConfig
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
        from engine.strategy_configurator import StrategyConfig
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
        from engine.core import PositionState
        from engine.strategy_configurator import StrategyConfig
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
        from engine.strategy_configurator import StrategyConfig
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
        from engine.core import PositionState
        from engine.strategy_configurator import StrategyConfig
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
        from engine.swing_detector import detect_swings

        df = self._sine_df()
        swings = detect_swings(df, min_prominence_atr=1.0, return_provisional=False)
        assert len(swings) >= 4, "wave-shaped data should produce many swings"
        for a, b in zip(swings, swings[1:]):
            assert a.side != b.side

    def test_confirmation_lags_pivot(self):
        from engine.swing_detector import detect_swings

        df = self._sine_df()
        swings = detect_swings(df, min_prominence_atr=1.0, return_provisional=False)
        assert all(s.confirmation_idx >= s.idx for s in swings)
        assert all(s.bars_to_confirm == s.confirmation_idx - s.idx for s in swings)

    def test_min_prominence_filters(self):
        from engine.swing_detector import detect_swings

        df = self._sine_df()
        loose = detect_swings(df, min_prominence_atr=0.5, return_provisional=False)
        strict = detect_swings(df, min_prominence_atr=3.0, return_provisional=False)
        assert len(loose) >= len(strict), "raising threshold must not add swings"


class TestSwingFlipStrategy:
    def test_backtest_runs_on_noise(self):
        from engine.backtester import Backtester
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import SwingFlipStrategy

        config = StrategyConfig(swing_zz_min_prominence_atr=1.0)
        strat = SwingFlipStrategy(config)
        df = _make_ohlcv(400, noise=3.0, seed=11)

        bt = Backtester(strat)
        result = bt.run(df, interval="15")
        assert result.num_bars == 400
        assert result.strategy_name == "swing_flip"
        assert all(t.is_closed for t in result.trades)

    def test_no_lookahead(self):
        from engine.core import PositionState
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import SwingFlipStrategy

        config = StrategyConfig(swing_zz_min_prominence_atr=1.0)
        df = _make_ohlcv(300, trend=0.05, noise=2.0, seed=4)

        strat_full = SwingFlipStrategy(config)
        prepared_full = strat_full.prepare(df)
        state_full = PositionState()
        for i in range(len(prepared_full)):
            strat_full.on_bar(i, prepared_full, state_full)

        df_short = df.iloc[:180].copy()
        strat_short = SwingFlipStrategy(config)
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
        from engine.strategy_configurator import StrategyConfig
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
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import EMACrossoverStrategy
        from engine.trade_configurator import TradingConfig

        # Costs now live on TradingConfig and are seeded onto the state by the
        # backtester — strategies no longer carry them.
        strat = EMACrossoverStrategy(StrategyConfig())
        df = _make_ohlcv(300, noise=2.0)

        cheap = Backtester(strat, trading_config=TradingConfig(fee_bps=0.0, slippage_bps=0.0))
        pricey = Backtester(strat, trading_config=TradingConfig(fee_bps=10.0, slippage_bps=5.0))
        cheap_res = cheap.run(df)
        pricey_res = pricey.run(df)

        if cheap_res.total_trades > 0:
            # Same signals, higher costs ⇒ each trade nets exactly the round-trip
            # cost difference (2*(10+5) − 0 = 30 bps) less than the cheap run.
            assert cheap_res.total_trades == pricey_res.total_trades
            per_trade = (cheap_res.total_pnl_bps - pricey_res.total_pnl_bps) / cheap_res.total_trades
            assert per_trade == pytest.approx(30.0)

    # ── _compute_stats edge cases ────────────────────────────────────────────

    @staticmethod
    def _bt_with_trades(pnls: list[float]):
        """Build a Backtester + PositionState pre-loaded with synthetic closed trades."""
        from engine.backtester import Backtester
        from engine.core import Direction, ExitReason, PositionState, Trade
        from engine.strategy_configurator import StrategyConfig
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
        from engine.core import Direction, ExitReason, PositionState
        from engine.strategy_configurator import StrategyConfig
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
        from engine.trade_configurator import TradingConfig

        config = TradingConfig(fee_bps=4.0, slippage_bps=2.0)
        assert config.total_cost_bps() == 12.0  # 2 * (4 + 2)

    def test_interval_validation(self):
        from engine.core import validate_interval

        assert validate_interval("15") == "15"
        with pytest.raises(ValueError):
            validate_interval("42")


# ── Trade configurator tests ─────────────────────────────────────────────────


class TestTradingConfig:
    def test_costs_and_direction_gates(self):
        from engine.trade_configurator import TradeDirection, TradingConfig

        tc = TradingConfig(fee_bps=4.0, slippage_bps=2.0)
        assert tc.total_cost_bps() == 12.0
        assert tc.allows_long() and tc.allows_short()

        long_only = TradingConfig(direction=TradeDirection.LONG)
        assert long_only.allows_long() and not long_only.allows_short()

    def test_notional_sizing(self):
        from engine.trade_configurator import TradingConfig

        # 50% of equity × 2× leverage = 1.0 × equity notional.
        tc = TradingConfig(position_size_bps=5_000.0, leverage=2.0)
        assert tc.notional(10_000.0) == pytest.approx(10_000.0)

    def test_validation(self):
        from engine.trade_configurator import TradingConfig

        for bad in (
            {"initial_equity": 0},
            {"position_size_bps": 0},
            {"leverage": 0},
            {"fee_bps": -1},
            {"max_daily_loss_bps": 0},
            {"max_holding_bars": 0},
        ):
            with pytest.raises(ValueError):
                TradingConfig(**bad)


class TestDirectionGate:
    def test_short_blocked_when_long_only(self):
        from engine.core import Direction, PositionState

        state = PositionState(allow_long=True, allow_short=False)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert state.enter(Direction.SHORT, ts, 100.0) is None   # rejected
        assert state.enter(Direction.LONG, ts, 100.0) is not None  # allowed


class TestDailyLossHalt:
    def test_halts_after_daily_cap_then_resets_next_day(self):
        from engine.core import Direction, ExitReason, PositionState

        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state = PositionState(max_daily_loss_bps=100.0)
        state.enter(Direction.LONG, ts, 100.0)
        state.exit(ts, 98.0, ExitReason.STOP_LOSS)  # −200 bps realized
        assert state.closed_trades[0].pnl_bps == pytest.approx(-200.0)
        # Same-day re-entry blocked; next UTC day allowed.
        assert state.enter(Direction.LONG, ts, 100.0) is None
        next_day = datetime(2025, 1, 2, tzinfo=timezone.utc)
        assert state.enter(Direction.LONG, next_day, 100.0) is not None


class TestEquityLayer:
    def test_equity_compounds_and_fills_trade_fields(self):
        from engine.backtester import Backtester
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import SuperTrendStrategy
        from engine.trade_configurator import TradingConfig

        strat = SuperTrendStrategy(StrategyConfig())
        df = _make_ohlcv(300, noise=3.0, seed=42)
        tc = TradingConfig(initial_equity=10_000.0, position_size_bps=10_000.0)
        result = Backtester(strat, trading_config=tc).run(df)

        assert result.initial_equity == 10_000.0
        if result.total_trades > 0:
            # Last trade's running equity == reported final equity.
            assert result.trades[-1].equity_after == pytest.approx(result.final_equity)
            # final_equity is consistent with the reported total return.
            assert result.final_equity == pytest.approx(
                10_000.0 * (1 + result.total_return_pct / 100.0)
            )
            for t in result.trades:
                assert t.notional > 0
                assert t.pnl_currency == pytest.approx(t.notional * t.pnl_bps / 10_000.0)

    def test_bps_metrics_unchanged_by_equity_layer(self):
        """The equity layer is additive: bps P&L must not depend on sizing."""
        from engine.backtester import Backtester
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import SuperTrendStrategy
        from engine.trade_configurator import TradingConfig

        df = _make_ohlcv(300, noise=3.0, seed=7)
        a = Backtester(SuperTrendStrategy(StrategyConfig()),
                       trading_config=TradingConfig(initial_equity=1_000.0)).run(df)
        b = Backtester(SuperTrendStrategy(StrategyConfig()),
                       trading_config=TradingConfig(initial_equity=500_000.0, leverage=3.0)).run(df)
        assert a.total_pnl_bps == pytest.approx(b.total_pnl_bps)
        assert a.total_trades == b.total_trades


class TestMaxHolding:
    def test_force_close_after_max_bars(self):
        from engine.backtester import Backtester
        from engine.core import Direction, ExitReason
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies.base import BaseStrategy
        from engine.trade_configurator import TradingConfig

        class _EnterAndHold(BaseStrategy):
            name = "enter_hold"
            def prepare(self, df):
                return df.copy()
            def on_bar(self, i, df, state):
                if i == 0 and state.current_trade is None:
                    state.enter(Direction.LONG, df.index[i], float(df["close"].iloc[i]))

        df = _make_ohlcv(30)
        result = Backtester(
            _EnterAndHold(StrategyConfig()), trading_config=TradingConfig(max_holding_bars=5)
        ).run(df)
        assert result.total_trades == 1
        assert result.trades[0].exit_reason == ExitReason.TIME_STOP


# ── Fix A: CLI seeds from ACTIVE_TRADE (single source of truth) ───────────────


class TestCLITradeConfig:
    def _ns(self, **over):
        import argparse
        from engine import cli
        fields = [
            "initial_equity", "position_size_bps", "leverage", "fee_bps",
            "slippage_bps", "max_daily_loss_bps", "max_holding_bars", "direction",
            "sizing_mode", "risk_per_trade_bps",
        ]
        ns = argparse.Namespace(**{f: cli._UNSET for f in fields})
        for k, v in over.items():
            setattr(ns, k, v)
        return ns

    def test_inherits_active_trade_when_no_flags(self, monkeypatch):
        from engine import cli
        from engine.trade_configurator import TradeDirection, TradingConfig

        custom = TradingConfig(fee_bps=7.0, leverage=3.0, direction=TradeDirection.SHORT)
        monkeypatch.setattr(cli, "ACTIVE_TRADE", custom)
        assert cli._build_trading_config(self._ns()) == custom

    def test_flag_overrides_only_that_field(self, monkeypatch):
        from engine import cli
        from engine.trade_configurator import TradeDirection, TradingConfig

        custom = TradingConfig(fee_bps=7.0, slippage_bps=9.0, direction=TradeDirection.SHORT)
        monkeypatch.setattr(cli, "ACTIVE_TRADE", custom)
        built = cli._build_trading_config(self._ns(fee_bps=1.0, direction="both"))
        assert built.fee_bps == 1.0                       # overridden
        assert built.direction == TradeDirection.BOTH      # overridden
        assert built.slippage_bps == custom.slippage_bps   # inherited, not reset to default

    def test_sizing_flags_override(self, monkeypatch):
        from engine import cli
        from engine.trade_configurator import SizingMode, TradingConfig

        monkeypatch.setattr(cli, "ACTIVE_TRADE", TradingConfig())
        built = cli._build_trading_config(self._ns(sizing_mode="risk", risk_per_trade_bps=250.0))
        assert built.sizing_mode == SizingMode.RISK
        assert built.risk_per_trade_bps == 250.0
        assert built.position_size_bps == TradingConfig().position_size_bps  # inherited


# ── Fix B: gate suppression is observable ─────────────────────────────────────


class TestSuppressionObservable:
    def test_counter_increments_on_blocked_side(self):
        from engine.core import Direction, PositionState

        state = PositionState(allow_long=True, allow_short=False)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state.enter(Direction.SHORT, ts, 100.0)   # blocked → counted
        state.enter(Direction.LONG, ts, 100.0)    # allowed → not counted
        assert state.suppressed_entries == 1

    def test_long_only_run_surfaces_suppressions(self):
        from engine.backtester import Backtester
        from engine.core import Direction
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import SuperTrendStrategy
        from engine.trade_configurator import TradeDirection, TradingConfig

        df = _make_ohlcv(400, noise=3.0, seed=11)
        both = Backtester(SuperTrendStrategy(StrategyConfig()),
                          trading_config=TradingConfig(direction=TradeDirection.BOTH)).run(df)
        longonly = Backtester(SuperTrendStrategy(StrategyConfig()),
                              trading_config=TradingConfig(direction=TradeDirection.LONG)).run(df)
        assert both.suppressed_entries == 0
        assert all(t.direction == Direction.LONG for t in longonly.trades)
        if any(t.direction == Direction.SHORT for t in both.trades):
            assert longonly.suppressed_entries > 0


# ── Fix C: inverse-strategy × asymmetric-gate warning ─────────────────────────


class TestInverseGateWarning:
    def test_is_asymmetric(self):
        from engine.trade_configurator import TradeDirection, TradingConfig

        assert not TradingConfig(direction=TradeDirection.BOTH).is_asymmetric()
        assert TradingConfig(direction=TradeDirection.LONG).is_asymmetric()
        assert TradingConfig(direction=TradeDirection.SHORT).is_asymmetric()

    def test_warns_on_inv_plus_asymmetric(self, caplog):
        import logging
        from engine.trade_configurator import (
            TradeDirection, TradingConfig, warn_if_inverse_gated,
        )
        with caplog.at_level(logging.WARNING):
            warn_if_inverse_gated("supertrend_inv", TradingConfig(direction=TradeDirection.SHORT))
        assert any("inversion leg" in r.getMessage() for r in caplog.records)

    def test_quiet_for_base_or_symmetric(self, caplog):
        import logging
        from engine.trade_configurator import (
            TradeDirection, TradingConfig, warn_if_inverse_gated,
        )
        with caplog.at_level(logging.WARNING):
            warn_if_inverse_gated("supertrend", TradingConfig(direction=TradeDirection.SHORT))    # base strategy
            warn_if_inverse_gated("supertrend_inv", TradingConfig(direction=TradeDirection.BOTH))  # symmetric gate
        assert not any("inversion leg" in r.getMessage() for r in caplog.records)


# ── Risk-based position sizing (mode toggle, stop-where-available) ────────────


class TestRiskSizing:
    def test_risk_notional_math(self):
        from engine.trade_configurator import SizingMode, TradingConfig

        tc = TradingConfig(sizing_mode=SizingMode.RISK, risk_per_trade_bps=100.0)
        # entry 100, stop 98 → stop_frac 0.02; risk$ = 10_000*0.01 = 100; notional = 100/0.02 = 5000
        assert tc.risk_notional(10_000.0, 100.0, 98.0) == pytest.approx(5_000.0)
        assert tc.risk_notional(10_000.0, 100.0, None) is None      # no stop
        assert tc.risk_notional(10_000.0, 100.0, 100.0) is None     # zero-distance stop

    def test_validation_rejects_nonpositive_risk(self):
        from engine.trade_configurator import TradingConfig

        with pytest.raises(ValueError):
            TradingConfig(risk_per_trade_bps=0)

    def test_stop_out_loses_the_risk_budget(self):
        """A trade exiting at its stop should lose ≈ risk_per_trade_bps of equity."""
        from engine.backtester import Backtester
        from engine.core import Direction
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies.base import BaseStrategy
        from engine.trade_configurator import SizingMode, TradingConfig

        class _StopOut(BaseStrategy):
            name = "stopout"
            def prepare(self, df):
                return df.copy()
            def on_bar(self, i, df, state):
                if i == 0 and state.current_trade is None:
                    px = float(df["close"].iloc[i])
                    state.enter(Direction.LONG, df.index[i], px, stop_price=px * 0.98)

        idx = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
        closes = [100.0, 100.0, 100.0, 100.0, 98.0]   # ends 2% below the entry
        df = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
            index=idx,
        )
        tc = TradingConfig(initial_equity=10_000.0, sizing_mode=SizingMode.RISK,
                           risk_per_trade_bps=100.0, fee_bps=0.0, slippage_bps=0.0)
        res = Backtester(_StopOut(StrategyConfig()), trading_config=tc).run(df)
        assert res.total_trades == 1
        t = res.trades[0]
        assert t.stop_price == pytest.approx(98.0)
        assert t.notional == pytest.approx(5_000.0)        # risk-sized, not 100% of equity
        assert t.pnl_currency == pytest.approx(-100.0)     # lost exactly 1% (100 bps) of equity
        assert res.risk_sizing_fallbacks == 0

    def test_stopless_strategy_falls_back_to_fixed(self):
        # A strategy that opens without a stop_price (no entry stop) must fall back
        # to fixed-fraction sizing in RISK mode. (Uses a stub so it stays valid
        # regardless of which strategies adopt stop-bearing exit policies.)
        from engine.backtester import Backtester
        from engine.core import Direction
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies.base import BaseStrategy
        from engine.trade_configurator import SizingMode, TradingConfig

        class _Stopless(BaseStrategy):
            name = "stopless_nostop"
            def prepare(self, df):
                return df.copy()
            def on_bar(self, i, df, state):
                if i == 0 and state.current_trade is None:
                    state.enter(Direction.LONG, df.index[i], float(df["close"].iloc[i]))

        df = _make_ohlcv(50)
        tc = TradingConfig(initial_equity=10_000.0, sizing_mode=SizingMode.RISK)
        res = Backtester(_Stopless(StrategyConfig()), trading_config=tc).run(df)
        assert res.total_trades == 1
        assert res.risk_sizing_fallbacks == res.total_trades   # no entry stop → fell back
        assert all(t.stop_price is None for t in res.trades)
        assert res.trades[0].notional == pytest.approx(tc.notional(10_000.0))  # fixed-fraction

    def test_bps_unchanged_fixed_vs_risk(self):
        """Golden: sizing mode must not move the bps P&L."""
        from engine.backtester import Backtester
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import SuperTrendStrategy
        from engine.trade_configurator import SizingMode, TradingConfig

        df = _make_ohlcv(300, noise=3.0, seed=9)
        fixed = Backtester(SuperTrendStrategy(StrategyConfig()),
                           trading_config=TradingConfig(sizing_mode=SizingMode.FIXED)).run(df)
        risk = Backtester(SuperTrendStrategy(StrategyConfig()),
                          trading_config=TradingConfig(sizing_mode=SizingMode.RISK)).run(df)
        assert fixed.total_pnl_bps == pytest.approx(risk.total_pnl_bps)
        assert fixed.total_trades == risk.total_trades


# ── Exit policies (engine/exits.py) ───────────────────────────────────────────


class TestExitPolicies:
    @staticmethod
    def _ctx(direction, **over):
        from engine.exits import ExitContext
        base = dict(
            direction=direction, entry_price=100.0, peak_price=100.0,
            high=100.0, low=100.0, close=100.0, atr=2.0,
        )
        base.update(over)
        return ExitContext(**base)

    def test_chandelier_trails_from_peak_on_close(self):
        from engine.exits import ChandelierStop
        from engine.core import Direction, ExitReason

        p = ChandelierStop(atr_mult=3.0)
        # long, entry 100 → initial stop 100 − 3*2 = 94
        assert p.initial_stop(self._ctx(Direction.LONG)) == pytest.approx(94.0)
        # peak 110, atr 2 → trailing stop 104; close 103 < 104 → exit at close
        d = p.evaluate(self._ctx(Direction.LONG, peak_price=110.0, close=103.0))
        assert d is not None and d.reason == ExitReason.TRAILING_STOP and d.price == 103.0
        # close above the trail → stay in
        assert p.evaluate(self._ctx(Direction.LONG, peak_price=110.0, close=105.0)) is None

    def test_chandelier_short_side(self):
        # SHORT: peak_price is the low-water mark; stop sits ABOVE it and triggers
        # when close rises through it (matches supertrend's `close > peak + trail`).
        from engine.exits import ChandelierStop
        from engine.core import Direction, ExitReason

        p = ChandelierStop(atr_mult=3.0)
        # short, entry 100 → initial stop 100 + 3*2 = 106
        assert p.initial_stop(self._ctx(Direction.SHORT)) == pytest.approx(106.0)
        # low-water 90, atr 2 → trail stop 96; close 97 > 96 (adverse rise) → exit
        d = p.evaluate(self._ctx(Direction.SHORT, peak_price=90.0, close=97.0))
        assert d is not None and d.reason == ExitReason.TRAILING_STOP and d.price == 97.0
        # close still below the trail → stay in
        assert p.evaluate(self._ctx(Direction.SHORT, peak_price=90.0, close=95.0)) is None

    def test_fixed_stops_trigger_intrabar_at_level(self):
        from engine.exits import AtrStop, FixedPctStop
        from engine.core import Direction, ExitReason

        # AtrStop long: level fixed at entry 100 − 1.5*2 = 97, read back via stop_price
        assert AtrStop(1.5).initial_stop(self._ctx(Direction.LONG)) == pytest.approx(97.0)
        d = AtrStop(1.5).evaluate(self._ctx(Direction.LONG, stop_price=97.0, low=96.5))
        assert d is not None and d.reason == ExitReason.STOP_LOSS and d.price == 97.0
        # FixedPctStop short: 2% above entry = 102
        assert FixedPctStop(2.0).initial_stop(self._ctx(Direction.SHORT)) == pytest.approx(102.0)
        assert FixedPctStop(2.0).evaluate(self._ctx(Direction.SHORT, stop_price=102.0, high=101.0)) is None

    def test_structural_stop_uses_ref_then_recorded_level(self):
        from engine.exits import StructuralStop
        from engine.core import Direction, ExitReason

        p = StructuralStop()
        assert p.initial_stop(self._ctx(Direction.LONG, ref_stop=95.0)) == 95.0
        d = p.evaluate(self._ctx(Direction.LONG, stop_price=95.0, low=94.0))
        assert d is not None and d.reason == ExitReason.STOP_LOSS and d.price == 95.0

    def test_targets(self):
        from engine.exits import FixedPctTarget, RrTarget
        from engine.core import Direction, ExitReason

        # fixed % target, long 100 + 3% = 103, hit intrabar
        d = FixedPctTarget(3.0).evaluate(self._ctx(Direction.LONG, high=103.5))
        assert d is not None and d.reason == ExitReason.TAKE_PROFIT and d.price == pytest.approx(103.0)
        # RR target needs a stop: risk = |100-98| = 2, 2R → 104
        d = RrTarget(2.0).evaluate(self._ctx(Direction.LONG, stop_price=98.0, high=104.5))
        assert d is not None and d.price == pytest.approx(104.0)
        # no stop → no RR target
        assert RrTarget(2.0).evaluate(self._ctx(Direction.LONG, high=104.5)) is None
        assert RrTarget(2.0).initial_stop(self._ctx(Direction.LONG)) is None  # target has no stop

    def test_composite_is_stop_first(self):
        from engine.exits import CompositeExit, RrTarget, StructuralStop
        from engine.core import Direction, ExitReason

        stop = StructuralStop()
        # long entry 100, stop 98, 2R target 104; a bar that pierces BOTH (low 97, high 105)
        bar = self._ctx(Direction.LONG, stop_price=98.0, ref_stop=98.0, low=97.0, high=105.0)
        # stop listed first → stop wins
        d = CompositeExit(stop, RrTarget(2.0)).evaluate(bar)
        assert d.reason == ExitReason.STOP_LOSS and d.price == 98.0
        # initial_stop comes from the first policy that defines one
        assert CompositeExit(stop, RrTarget(2.0)).initial_stop(
            self._ctx(Direction.LONG, ref_stop=98.0)
        ) == 98.0

    def test_zero_or_nonfinite_level_is_not_a_real_exit(self):
        # Regression for the live 0.0-exit bug: order_block / exhaustion /
        # impulse_flag re-zero their stashed ref_stop/ref_target each prepare();
        # while a position is open in live mode that leaves the level at 0.0.
        # 0.0 (and NaN/negative) must be treated as "no level", NOT a real price —
        # otherwise a long is force-closed at price 0.0 (high >= 0 always true),
        # a phantom ~-100% trade booked as a take-profit.
        from engine.exits import (
            StructuralTarget, StructuralStop, CloseCrossTarget, CompositeExit,
        )
        from engine.core import Direction

        assert StructuralTarget().evaluate(
            self._ctx(Direction.LONG, ref_target=0.0, high=120.0)) is None
        assert StructuralStop().evaluate(
            self._ctx(Direction.SHORT, ref_stop=0.0, high=120.0)) is None
        assert CloseCrossTarget().evaluate(
            self._ctx(Direction.LONG, ref_target=0.0, close=120.0)) is None
        assert StructuralTarget().evaluate(
            self._ctx(Direction.LONG, ref_target=float("nan"), high=120.0)) is None
        # The 'structural' preset (stop-first composite) with both levels zeroed.
        comp = CompositeExit(StructuralStop(), StructuralTarget())
        assert comp.evaluate(self._ctx(Direction.LONG, ref_target=0.0, high=120.0)) is None
        # Sanity: a real positive level still fires.
        d = StructuralTarget().evaluate(
            self._ctx(Direction.LONG, ref_target=110.0, high=120.0))
        assert d is not None and d.price == pytest.approx(110.0)


class TestADX:
    def test_adx_short_frame_returns_nan_not_crash(self):
        # Regression: n <= period used to IndexError on the Wilder seed write.
        from engine.indicators import adx

        for n in (1, 5, 14):
            df = pd.DataFrame({
                "high": np.arange(1, n + 1) + 0.5,
                "low": np.arange(1, n + 1, dtype=float),
                "close": np.arange(1, n + 1) + 0.2,
            })
            out = adx(df, period=14)
            assert len(out) == n and out.isna().all()

    def test_adx_normal_frame_is_finite(self):
        from engine.indicators import adx

        n = 100
        df = pd.DataFrame({
            "high": np.arange(1, n + 1) + 0.5,
            "low": np.arange(1, n + 1, dtype=float),
            "close": np.arange(1, n + 1) + 0.2,
        })
        assert np.isfinite(adx(df, period=14).iloc[-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
