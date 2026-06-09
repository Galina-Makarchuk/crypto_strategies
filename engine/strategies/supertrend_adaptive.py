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
from ..core import Direction, ExitReason, PositionState
from ..strategy_configurator import StrategyConfig
from .base import BaseStrategy


class AdaptiveSuperTrendStrategy(BaseStrategy):
    name = "supertrend_adaptive"

    def __init__(self, config: StrategyConfig, exit_policy=None):
        super().__init__(config, exit_policy)
        # Regime (trending vs ranging) captured at entry. The exit-flip side is
        # decided by THIS, not the current bar's regime, so a mid-trade ADX
        # crossing can't flip a position's exit condition. None when flat (or a
        # position restored in live before this instance opened it → falls back
        # to the current regime).
        self._entry_trending: bool | None = None

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

        trending = adx_val >= self.config.adx_threshold

        # ── Update trailing stop peak ──────────────────────────────────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ─────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade
            # Signal-flip exit stays in-strategy. The flip side is decided by the
            # regime captured AT ENTRY (not this bar's), so a trend-follow long
            # keeps exiting on flip_down even after ADX falls into 'ranging'.
            entry_trending = (
                self._entry_trending if self._entry_trending is not None else trending
            )
            if trade.direction == Direction.LONG:
                exit_flip = flip_down if entry_trending else flip_up
            else:
                exit_flip = flip_up if entry_trending else flip_down
            if exit_flip:
                state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                self._entry_trending = None
                return
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                self._entry_trending = None
                return

        # ── Entry logic ────────────────────────────────────────────────────
        if state.current_trade is not None or not has_flip:
            return

        # Regime fixes both the entry side and (above) the exit-flip side.
        #   trending → follow: flip_up→LONG,  flip_down→SHORT
        #   ranging  → fade:   flip_up→SHORT, flip_down→LONG
        if trending:
            direction = Direction.LONG if flip_up else Direction.SHORT
        else:
            direction = Direction.SHORT if flip_up else Direction.LONG
        if state.enter(
            direction, ts, close,
            stop_price=self._entry_stop(direction, close, atr_val),
        ) is not None:
            self._entry_trending = trending
