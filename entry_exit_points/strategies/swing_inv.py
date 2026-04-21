"""Inverse Swing Breakout strategy.

Identical to SwingBreakoutStrategy but with flipped direction:
  - Price crosses above resistance → SHORT (fading the breakout)
  - Price crosses below support    → LONG  (buying the dip)

Mean-reversion approach: bets that breakouts will fail and price
will revert back into the range.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import (
    atr,
    detect_swing_highs,
    detect_swing_lows,
    merge_price_levels,
)
from ..models import Direction, PositionState, StrategyConfig
from .base import BaseStrategy


class InverseSwingBreakoutStrategy(BaseStrategy):
    name = "swing_inv"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._swing_highs: list[tuple[int, float]] = []
        self._swing_lows: list[tuple[int, float]] = []

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = atr(df, self.config.atr_period)
        self._swing_highs = detect_swing_highs(df["high"], self.config.left, self.config.right)
        self._swing_lows = detect_swing_lows(df["low"], self.config.left, self.config.right)
        return df

    def _active_levels(self, current_bar: int) -> tuple[list[float], list[float]]:
        confirm_offset = self.config.right
        res_prices = [p for idx, p in self._swing_highs if idx + confirm_offset < current_bar]
        sup_prices = [p for idx, p in self._swing_lows if idx + confirm_offset < current_bar]
        res = merge_price_levels(res_prices, self.config.merge_tolerance)
        sup = merge_price_levels(sup_prices, self.config.merge_tolerance)
        return res, sup

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]
        cost = self.config.total_cost_bps()

        res_levels, sup_levels = self._active_levels(i)

        # ── Update trailing stop peak ──────────────────────────────────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ─────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade
            trail = atr_val * self.config.atr_trail_mult

            if trade.direction == Direction.LONG:
                trailing_stop = trade.peak_price - trail
                # Exit long when price crosses above resistance (original would enter long)
                crossed_above_res = any(close > lvl >= prev_close for lvl in res_levels)
                if close < trailing_stop or crossed_above_res:
                    state.exit(ts, close, cost)
                    return

            elif trade.direction == Direction.SHORT:
                trailing_stop = trade.peak_price + trail
                # Exit short when price crosses below support (original would enter short)
                crossed_below_sup = any(close < lvl <= prev_close for lvl in sup_levels)
                if close > trailing_stop or crossed_below_sup:
                    state.exit(ts, close, cost)
                    return

        # ── Entry logic (INVERTED) ─────────────────────────────────────────
        if state.current_trade is not None:
            return

        # Original goes long on breakout above resistance → we go SHORT
        crossed_up = any(close > lvl >= prev_close for lvl in res_levels)
        if crossed_up:
            state.enter(Direction.SHORT, ts, close)
            return

        # Original goes short on breakdown below support → we go LONG
        crossed_down = any(close < lvl <= prev_close for lvl in sup_levels)
        if crossed_down:
            state.enter(Direction.LONG, ts, close)
