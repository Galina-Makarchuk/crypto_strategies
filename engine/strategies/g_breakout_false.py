"""False-breakout reversal — trade a poke through a level that fails to hold.

A false breakout is the trail of a large player sweeping stops just past a level
before reversing; we trade the reversal back into the range. Three modes select
how the failed break is recognized (``g_breakout_false_mode``):

  * **single**  — one bar pokes beyond the level but closes back on the original
    side (poke depth within ``max_depth``).
  * **two_bar** — the prior bar closes beyond the level (a "real" break on paper),
    then the current bar closes back across it. More trapped traders → stronger
    reversal.
  * **complex** — a run of ``>= 3`` bars closes beyond the level, then a later bar
    finally closes back across it.

Built on the :mod:`engine.levels` detectors (selectable via
``LevelParams.level_detector``; default ``cluster_level``, whose decisive
close-through break keeps a level alive through a failed wick so the reversal is
observable). Entry is geometry-driven and convention-agnostic; the stop sits just
beyond the false-break extreme and the take-profit at 3R (preset
``structural_rr3``). The structural stop doubles as invalidation (a resumed break
fills it), so no separate native exit is needed.

Look-ahead free via :class:`LevelStrategyBase._active_levels`; the backward
scans read only bars ``<= i``. ATR(``level_atr_period``) drives depth and stop.
"""

from __future__ import annotations

import math

import pandas as pd

from ..core import Direction, PositionState
from .level_base import LevelStrategyBase


class GBreakoutFalseStrategy(LevelStrategyBase):
    name = "g_breakout_false"

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]

        # ── In position: peak → delegated stop/target; never enter same bar ──
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)
            trade = state.current_trade
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
            return

        # ── Entry (a failed break reverses) ────────────────────────────────
        if not math.isfinite(atr_val) or atr_val <= 0:
            return

        levels = self._active_levels(i)
        mode = self.config.g_breakout_false_mode
        if mode == "single":
            sig = self._single(i, df, levels, atr_val)
        elif mode == "two_bar":
            sig = self._two_bar(i, df, levels, atr_val)
        else:
            sig = self._complex(i, df, levels, atr_val)
        if sig is None:
            return

        direction, extreme = sig
        buf = self.config.g_breakout_false_stop_buffer_atr * atr_val
        close = df["close"].iloc[i]
        if direction == Direction.LONG:
            stop = extreme - buf                   # just below the false-break low
            if stop < close:
                state.enter(Direction.LONG, ts, close, stop_price=stop)
        else:
            stop = extreme + buf                   # just above the false-break high
            if stop > close:
                state.enter(Direction.SHORT, ts, close, stop_price=stop)

    # ── Mode detectors — each returns (direction, false_break_extreme) | None ──
    def _single(self, i, df, levels, atr_val):
        close = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        max_depth = self.config.g_breakout_false_max_depth_atr * atr_val
        # Poke above a level, close back below → SHORT (nearest such level).
        shorts = [lvl for lvl in levels
                  if close < lvl < high_i and (high_i - lvl) <= max_depth]
        if shorts:
            return Direction.SHORT, high_i
        # Poke below a level, close back above → LONG.
        longs = [lvl for lvl in levels
                 if low_i < lvl < close and (lvl - low_i) <= max_depth]
        if longs:
            return Direction.LONG, low_i
        return None

    def _two_bar(self, i, df, levels, atr_val):
        close = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        prev_high = df["high"].iloc[i - 1]
        prev_low = df["low"].iloc[i - 1]
        max_depth = self.config.g_breakout_false_max_depth_atr * atr_val
        # Prior bar closed above the level, current bar closes back below → SHORT.
        for lvl in levels:
            if prev_close > lvl and prev_high > lvl and close < lvl:
                extreme = max(prev_high, high_i)
                if (extreme - lvl) <= max_depth:
                    return Direction.SHORT, extreme
            if prev_close < lvl and prev_low < lvl and close > lvl:
                extreme = min(prev_low, low_i)
                if (lvl - extreme) <= max_depth:
                    return Direction.LONG, extreme
        return None

    def _complex(self, i, df, levels, atr_val):
        close = df["close"].iloc[i]
        max_depth = self.config.g_breakout_false_max_depth_atr * atr_val
        max_bars = self.config.g_breakout_false_consol_max_bars
        closes = df["close"]
        highs = df["high"]
        lows = df["low"]
        for lvl in levels:
            above, below = self._streak(closes, i, lvl, max_bars)
            # A run closed above the level, now we close back below → SHORT.
            if above >= 3 and close < lvl:
                extreme = float(highs.iloc[i - above:i].max())
                if (extreme - lvl) <= max_depth:
                    return Direction.SHORT, extreme
            if below >= 3 and close > lvl:
                extreme = float(lows.iloc[i - below:i].min())
                if (lvl - extreme) <= max_depth:
                    return Direction.LONG, extreme
        return None

    @staticmethod
    def _streak(closes, i, lvl, max_bars):
        """Consecutive bars strictly before ``i`` whose close sits beyond ``lvl``
        (above, below), each capped at ``max_bars``."""
        above = 0
        j = i - 1
        while j >= 0 and above < max_bars and closes.iloc[j] > lvl:
            above += 1
            j -= 1
        below = 0
        j = i - 1
        while j >= 0 and below < max_bars and closes.iloc[j] < lvl:
            below += 1
            j -= 1
        return above, below
