"""Inverse Level-Breakout strategy — fade breakouts of horizontal S/R levels.

The mean-reversion mirror of :class:`engine.strategies.level_breakout.\
LevelBreakoutStrategy`, on the same :mod:`engine.level_detector` level set
(see :class:`engine.strategies.level_base.LevelStrategyBase`). It bets that
breakouts fail and price reverts into the range:

  * Close clears an **overhead** level → **SHORT** (fade the up-breakout).
  * Close breaks an **underlying** level → **LONG**  (fade the breakdown).

There is no broken level to anchor against on the fade side, so the stop is a
plain ATR stop from entry (preset ``atr_stop_rr2`` = ATR stop + 2R target),
seeded via the exit policy's ``initial_stop`` so the trail distance stays the
preset's single source of truth. A native flip exits when the breakout
*continues* through a further level (the fade has been invalidated).
"""

from __future__ import annotations

import math

import pandas as pd

from ..core import Direction, ExitReason, PositionState
from .level_base import LevelStrategyBase


class InverseLevelBreakoutStrategy(LevelStrategyBase):
    name = "level_breakout_inv"

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]

        levels = self._active_levels(i)

        # ── Update trailing-stop peak ──────────────────────────────────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit (delegated stop/target first, then native continuation flip) ─
        if state.current_trade is not None:
            trade = state.current_trade
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                return
            # Native flip: the breakout continues through a further level, so the
            # fade is invalidated. SHORT faded an up-break → continuation is up;
            # LONG faded a down-break → continuation is down.
            if trade.direction == Direction.SHORT:
                if any(close > lvl >= prev_close for lvl in levels):
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                    return
            elif trade.direction == Direction.LONG:
                if any(close < lvl <= prev_close for lvl in levels):
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                    return

        # ── Entry (fade the breakout) ──────────────────────────────────────
        if state.current_trade is not None:
            return
        if not math.isfinite(atr_val) or atr_val <= 0:
            return

        buf = self.config.level_breakout_buffer_atr * atr_val

        # Up-breakout (close clears an overhead level) → SHORT.
        up = [lvl for lvl in levels if prev_close < lvl and close > lvl + buf]
        if up:
            state.enter(Direction.SHORT, ts, close,
                        stop_price=self._entry_stop(Direction.SHORT, close, atr_val))
            return

        # Breakdown (close breaks an underlying level) → LONG.
        down = [lvl for lvl in levels if prev_close > lvl and close < lvl - buf]
        if down:
            state.enter(Direction.LONG, ts, close,
                        stop_price=self._entry_stop(Direction.LONG, close, atr_val))
            return
