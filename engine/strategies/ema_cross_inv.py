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
from ..models import Direction, ExitReason, PositionState, StrategyConfig
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
            trail = atr_val * self.config.atr_trail_mult

            if trade.direction == Direction.LONG:
                trailing_stop = trade.peak_price - trail
                # LONG was entered on bearish_cross; a bullish_cross would
                # trigger a new SHORT in this strategy → signal flip.
                if bullish_cross:
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                    return
                if close < trailing_stop:
                    state.exit(ts, close, ExitReason.TRAILING_STOP)
                    return
            elif trade.direction == Direction.SHORT:
                trailing_stop = trade.peak_price + trail
                # SHORT was entered on bullish_cross; a bearish_cross would
                # trigger a new LONG in this strategy → signal flip.
                if bearish_cross:
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                    return
                if close > trailing_stop:
                    state.exit(ts, close, ExitReason.TRAILING_STOP)
                    return

        # ── Entry logic (INVERTED) ─────────────────────────────────────────
        if state.current_trade is not None:
            return

        # Original goes LONG on bullish_cross → we go SHORT
        if bullish_cross and rsi_val > 30:
            state.enter(Direction.SHORT, ts, close)
        # Original goes SHORT on bearish_cross → we go LONG
        elif bearish_cross and rsi_val < 70:
            state.enter(Direction.LONG, ts, close)
