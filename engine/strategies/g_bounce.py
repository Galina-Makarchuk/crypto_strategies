"""Level bounce — trade a rejection off a horizontal S/R level.

Ported onto the :mod:`engine.levels` detectors (selectable via
``LevelParams.level_detector``; default ``touch_level``). The model bets a key
level HOLDS: enter in the bounce direction when a bar's extreme tests the level
within an ATR band and the close finishes back on the original side, while the
bar presses into the level (a fresh test). The stop sits just beyond the level,
the take-profit at 3R (preset ``structural_rr3``).

Entry is geometry-driven and convention-agnostic — it does not read the
detector's resistance/support labels, only each active level's position relative
to the prior close:

  * **Long**  bounce off an *underlying* level (``level < prev_close``): the bar's
    low dips into ``[level − tol, level + tol]``, the close holds above the level,
    and the low presses below the prior low. Stop = ``level − buffer``.
  * **Short** bounce off an *overhead* level (``level > prev_close``): the bar's
    high reaches into the band, the close holds below the level, and the high
    presses above the prior high. Stop = ``level + buffer``.

The structural stop (just beyond the level) doubles as the invalidation: a close
back through the level fills it intrabar, so no separate native exit is needed.

Look-ahead free via :class:`LevelStrategyBase._active_levels`. ATR
(``level_atr_period``) drives the band, buffer, and stop distance.
"""

from __future__ import annotations

import math

import pandas as pd

from ..core import Direction, PositionState
from .level_base import LevelStrategyBase


class GBounceStrategy(LevelStrategyBase):
    name = "g_bounce"

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        prev_high = df["high"].iloc[i - 1]
        prev_low = df["low"].iloc[i - 1]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]

        levels = self._active_levels(i)

        # ── In position: peak → delegated stop/target; never enter same bar ──
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)
            trade = state.current_trade
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
            return

        # ── Entry (rejection off a level) ──────────────────────────────────
        if not math.isfinite(atr_val) or atr_val <= 0:
            return

        tol = self.config.g_bounce_tol_atr * atr_val
        buf = self.config.g_bounce_stop_buffer_atr * atr_val

        # Long: underlying level tested from above and held; bar presses lower.
        longs = [
            lvl for lvl in levels
            if lvl < prev_close
            and (lvl - tol) <= low_i <= (lvl + tol)
            and close > lvl and low_i < prev_low
        ]
        if longs:
            lvl = max(longs)                       # nearest level below price (the one tested)
            stop = lvl - buf                       # structural stop just under the held level
            if stop < close:
                state.enter(Direction.LONG, ts, close, stop_price=stop)
            return

        # Short: overhead level tested from below and held; bar presses higher.
        shorts = [
            lvl for lvl in levels
            if lvl > prev_close
            and (lvl - tol) <= high_i <= (lvl + tol)
            and close < lvl and high_i > prev_high
        ]
        if shorts:
            lvl = min(shorts)                      # nearest level above price
            stop = lvl + buf
            if stop > close:
                state.enter(Direction.SHORT, ts, close, stop_price=stop)
            return
