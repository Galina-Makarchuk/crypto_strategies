"""Adaptive SuperTrend strategy with ADX regime filter.

The idea:
  - ADX >= threshold  → trending market  → use normal SuperTrend (follow the trend)
  - ADX <  threshold  → ranging market   → use inverse SuperTrend (fade the trend)

This avoids the main failure mode of each approach:
  - Normal SuperTrend bleeds in ranges (constant whipsaws)
  - Inverse SuperTrend bleeds in trends (fighting momentum)

The ADX threshold (default 25) and lookback are configurable via StrategyConfig.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import adx as calc_adx
from ..indicators import atr as calc_atr
from ..indicators import supertrend as calc_supertrend
from ..models import Direction, ExitReason, PositionState
from ..strategy_configurator import StrategyConfig
from .base import BaseStrategy


class AdaptiveSuperTrendStrategy(BaseStrategy):
    name = "supertrend_adaptive"

    def __init__(self, config: StrategyConfig, adx_threshold: float = 25.0):
        super().__init__(config)
        self.adx_threshold = adx_threshold

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        st_line, trend_dir = calc_supertrend(
            df, self.config.supertrend_period, self.config.supertrend_mult
        )
        df["supertrend"] = st_line
        df["trend_dir"] = trend_dir
        df["atr"] = calc_atr(df, self.config.atr_period)
        df["adx"] = calc_adx(df, period=self.config.atr_period)
        return df

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        close = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]
        adx_val = df["adx"].iloc[i]

        trend_now = int(df["trend_dir"].iloc[i])
        trend_prev = int(df["trend_dir"].iloc[i - 1])
        flip_up = trend_now == 1 and trend_prev == -1
        flip_down = trend_now == -1 and trend_prev == 1
        has_flip = flip_up or flip_down

        # ADX not yet valid (warmup period) → skip
        if pd.isna(adx_val):
            return

        trending = adx_val >= self.adx_threshold

        # ── Update trailing stop peak ──────────────────────────────────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ─────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade
            # Signal-flip exit stays in-strategy (regime-dependent), checked first.
            if trade.direction == Direction.LONG:
                exit_flip = flip_down if trending else flip_up
            else:
                exit_flip = flip_up if trending else flip_down
            if exit_flip:
                state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                return
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                return

        # ── Entry logic ────────────────────────────────────────────────────
        if state.current_trade is not None or not has_flip:
            return

        if trending:
            # Normal: follow the trend
            if flip_up:
                state.enter(Direction.LONG, ts, close,
                            stop_price=self._entry_stop(Direction.LONG, close, atr_val))
            elif flip_down:
                state.enter(Direction.SHORT, ts, close,
                            stop_price=self._entry_stop(Direction.SHORT, close, atr_val))
        else:
            # Inverse: fade the trend
            if flip_up:
                state.enter(Direction.SHORT, ts, close,
                            stop_price=self._entry_stop(Direction.SHORT, close, atr_val))
            elif flip_down:
                state.enter(Direction.LONG, ts, close,
                            stop_price=self._entry_stop(Direction.LONG, close, atr_val))
