"""Adaptive EMA crossover with an RSI regime switch.

The RSI-driven analogue of :class:`AdaptiveSuperTrendStrategy` (which switches
follow/fade on ADX). The EMA fast/slow cross is the raw signal; **RSI decides
whether to follow that cross or fade it**:

  - bullish cross + RSI not overbought (rsi <  rsi_bullish) → follow → LONG
  - bullish cross + RSI overbought    (rsi >= rsi_bullish) → fade   → SHORT
  - bearish cross + RSI not oversold  (rsi >  rsi_bearish) → follow → SHORT
  - bearish cross + RSI oversold      (rsi <= rsi_bearish) → fade   → LONG

So the same RSI bounds the plain ``ema`` strategy uses to *veto* an entry are
used here to *flip* it: an overbought bullish cross becomes a mean-reversion
short. RSI is the filter that controls the signal and the entry decision.

This avoids each plain variant's failure mode: riding a cross straight into an
exhausted, overextended move (the base ``ema``), or fading a cross that still has
momentum (``ema_inv``).

Exits respect the regime captured AT ENTRY (follow vs fade), so a mid-trade RSI
swing can't change a position's exit condition — mirroring AdaptiveSuperTrend.
When ``rsi_filter`` is off there is no regime to switch on, so it degenerates to
plain follow-the-cross (== the base ``ema`` strategy).
"""

from __future__ import annotations

import pandas as pd

from ..indicators import atr, ema, rsi
from ..core import Direction, ExitReason, PositionState
from .base import BaseStrategy


class AdaptiveEMACrossoverStrategy(BaseStrategy):
    name = "ema_adaptive"

    def __init__(self, config, exit_policy=None):
        super().__init__(config, exit_policy)
        # Whether the open position FOLLOWED the cross (True) or FADED it (False),
        # captured at entry so the exit side can't change mid-trade. None when flat
        # (or a position restored in live before this instance opened it → falls
        # back to follow).
        self._entry_follow: bool | None = None

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

        bullish_cross = ema_f > ema_s and ema_f_prev <= ema_s_prev
        bearish_cross = ema_f < ema_s and ema_f_prev >= ema_s_prev

        # ── Update trailing stop peak ──────────────────────────────────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ─────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade
            # Reverse-cross exit, with the side fixed by the entry regime: a
            # followed long exits on a bearish cross, a faded long on a bullish
            # cross (and vice-versa for shorts) — mirrors AdaptiveSuperTrend.
            entry_follow = self._entry_follow if self._entry_follow is not None else True
            if trade.direction == Direction.LONG:
                exit_cross = bearish_cross if entry_follow else bullish_cross
            else:
                exit_cross = bullish_cross if entry_follow else bearish_cross
            if exit_cross:
                state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                self._entry_follow = None
                return
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                self._entry_follow = None
                return

        # ── Entry logic ────────────────────────────────────────────────────
        if state.current_trade is not None or not (bullish_cross or bearish_cross):
            return

        # RSI is the regime control: follow the cross unless RSI says the move is
        # overextended, in which case fade it (enter the opposite side).
        if bullish_cross:
            overbought = self.config.rsi_filter and rsi_val >= self.config.rsi_bullish
            direction = Direction.SHORT if overbought else Direction.LONG
            follow = not overbought
        else:  # bearish_cross
            oversold = self.config.rsi_filter and rsi_val <= self.config.rsi_bearish
            direction = Direction.LONG if oversold else Direction.SHORT
            follow = not oversold

        if state.enter(
            direction, ts, close,
            stop_price=self._entry_stop(direction, close, atr_val),
        ) is not None:
            self._entry_follow = follow
