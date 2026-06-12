"""Fractal-breakout strategy (N-bar fractal pivots → merged S/R → breakout).

Detects S/R from **N-bar fractal pivots** via ``indicators.detect_swing_highs`` /
``detect_swing_lows`` (a strict ``left``/``right`` window), merged with
``indicators.merge_price_levels``. This is the *indicators-layer* level source —
distinct from ``level_breakout``, which is built on the dedicated, stateful
``engine.level_detector`` (horizontal S/R with invalidation tracking).

Key properties:
  1. No look-ahead: only uses levels confirmed ≥ `right` bars before current bar.
  2. Cross detection, not state check: entry fires only on the bar that crosses
     a level, not on every subsequent bar above it.
  3. Real trailing stop: tracks peak price since entry, exits when price falls
     ATR × mult below peak (longs) or rises above trough (shorts).
  4. Same-bar entry+exit prevention: after entering, exit logic defers to next bar.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import (
    atr,
    detect_swing_highs,
    detect_swing_lows,
    merge_price_levels,
)
from ..core import Direction, ExitReason, PositionState
from ..strategy_configurator import FractalParams
from .base import BaseStrategy


class FractalBreakoutStrategy(BaseStrategy):
    name = "fractal_breakout"

    def __init__(self, config: FractalParams, exit_policy=None):
        super().__init__(config, exit_policy)
        self._swing_highs: list[tuple[int, float]] = []
        self._swing_lows: list[tuple[int, float]] = []

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = atr(df, self.config.atr_period)

        # Pre-compute ALL swing points (indices are bar positions).
        # on_bar() filters by confirmation horizon.
        self._swing_highs = detect_swing_highs(df["high"], self.config.left, self.config.right)
        self._swing_lows = detect_swing_lows(df["low"], self.config.left, self.config.right)
        return df

    def _active_levels(self, current_bar: int) -> tuple[list[float], list[float]]:
        """Return (resistance, support) levels confirmed before `current_bar`."""
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

        res_levels, sup_levels = self._active_levels(i)

        # ── Update trailing stop peak ──────────────────────────────────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic (skip on entry bar → prevents same-bar flip) ───────
        if state.current_trade is not None:
            trade = state.current_trade
            # Trailing stop (delegated) is checked FIRST here, as before.
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                return  # don't enter on same bar as exit
            # Structural cross flip (native).
            if trade.direction == Direction.LONG:
                if any(close < lvl <= prev_close for lvl in sup_levels):
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                    return
            elif trade.direction == Direction.SHORT:
                if any(close > lvl >= prev_close for lvl in res_levels):
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                    return

        # ── Entry logic (cross detection, not state check) ────────────────
        if state.current_trade is not None:
            return  # already in position

        # Long: close crosses above any resistance level
        crossed_up = any(close > lvl >= prev_close for lvl in res_levels)
        if crossed_up:
            state.enter(Direction.LONG, ts, close,
                        stop_price=self._entry_stop(Direction.LONG, close, atr_val))
            return

        # Short: close crosses below any support level
        crossed_down = any(close < lvl <= prev_close for lvl in sup_levels)
        if crossed_down:
            state.enter(Direction.SHORT, ts, close,
                        stop_price=self._entry_stop(Direction.SHORT, close, atr_val))
