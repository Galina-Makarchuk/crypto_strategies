"""Inverse EMA crossover + RSI filter strategy.

Identical to EMACrossoverStrategy but with flipped direction:
  - Bullish cross (fast crosses above slow) → SHORT (fading the cross)
  - Bearish cross (fast crosses below slow) → LONG  (fading the cross)

Mean-reversion approach: bets that EMA crosses mark momentum exhaustion
and price will revert rather than continue.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, ema, rsi
from ..core import Direction, ExitReason, PositionState
from ..strategy_configurator import StrategyConfig
from .base import BaseStrategy


class InverseEMACrossoverStrategy(BaseStrategy):
    name = "ema_inv"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = ema(df["close"], self.config.ema_fast)
        df["ema_slow"] = ema(df["close"], self.config.ema_slow)
        df["rsi"] = rsi(df["close"], self.config.rsi_period)
        df["atr"] = atr(df, self.config.atr_period)
        return df

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        close = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]

        ema_f = df["ema_fast"].iloc[i]
        ema_s = df["ema_slow"].iloc[i]
        ema_f_prev = df["ema_fast"].iloc[i - 1]
        ema_s_prev = df["ema_slow"].iloc[i - 1]
        rsi_val = df["rsi"].iloc[i]

        # Cross detection
        bullish_cross = ema_f > ema_s and ema_f_prev <= ema_s_prev
        bearish_cross = ema_f < ema_s and ema_f_prev >= ema_s_prev

        # ── Update trailing stop peak ──────────────────────────────────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ─────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade
            # Signal-flip exit stays in-strategy, checked first (as before):
            # the inverse strategy fades crosses, so the opposite cross flips it.
            if (trade.direction == Direction.LONG and bullish_cross) or (
                trade.direction == Direction.SHORT and bearish_cross
            ):
                state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                return
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                return

        # ── Entry logic (INVERTED) ─────────────────────────────────────────
        if state.current_trade is not None:
            return

        # RSI gates by entry direction (same knobs as the base strategy);
        # disabled when config.rsi_filter is off.
        long_ok = not self.config.rsi_filter or rsi_val < self.config.rsi_bullish
        short_ok = not self.config.rsi_filter or rsi_val > self.config.rsi_bearish

        # Original goes LONG on bullish_cross → we go SHORT
        if bullish_cross and short_ok:
            state.enter(Direction.SHORT, ts, close,
                        stop_price=self._entry_stop(Direction.SHORT, close, atr_val))
        # Original goes SHORT on bearish_cross → we go LONG
        elif bearish_cross and long_ok:
            state.enter(Direction.LONG, ts, close,
                        stop_price=self._entry_stop(Direction.LONG, close, atr_val))
