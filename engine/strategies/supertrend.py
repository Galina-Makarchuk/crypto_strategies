"""SuperTrend strategy.

Entry: trend direction flips (+1 → −1 or vice versa).
Exit:  reverse flip OR trailing ATR stop from peak.

Uses the corrected SuperTrend indicator with proper band ratcheting
and trend-state tracking (see indicators.supertrend).
"""

from __future__ import annotations

import pandas as pd

from ..indicators import atr as calc_atr
from ..indicators import supertrend as calc_supertrend
from ..core import Direction, ExitReason, PositionState
from ..strategy_configurator import StrategyConfig
from .base import BaseStrategy


class SuperTrendStrategy(BaseStrategy):
    name = "supertrend"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        st_line, trend_dir = calc_supertrend(
            df, self.config.supertrend_period, self.config.supertrend_mult
        )
        df["supertrend"] = st_line
        df["trend_dir"] = trend_dir  # +1 = up, −1 = down
        df["atr"] = calc_atr(df, self.config.atr_period)
        return df

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        close = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]

        trend_now = int(df["trend_dir"].iloc[i])
        trend_prev = int(df["trend_dir"].iloc[i - 1])
        flip_up = trend_now == 1 and trend_prev == -1
        flip_down = trend_now == -1 and trend_prev == 1

        # ── Update trailing stop peak ──────────────────────────────────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ─────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade
            # Signal-flip exit stays in-strategy (it's driven by the entry signal,
            # not a price level) and is checked first, as before.
            if (trade.direction == Direction.LONG and flip_down) or (
                trade.direction == Direction.SHORT and flip_up
            ):
                state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                return
            # Trailing stop is delegated to the injected exit policy.
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                return

        # ── Entry logic ────────────────────────────────────────────────────
        if state.current_trade is not None:
            return

        if flip_up:
            state.enter(Direction.LONG, ts, close,
                        stop_price=self._entry_stop(Direction.LONG, close, atr_val))
        elif flip_down:
            state.enter(Direction.SHORT, ts, close,
                        stop_price=self._entry_stop(Direction.SHORT, close, atr_val))
