"""Level-breakout strategy — breakout through horizontal S/R levels.

Built on the :mod:`engine.levels` package — the dedicated horizontal-level
detectors, selectable via ``LevelParams.level_detector`` (pivot_level /
cluster_level / touch_level). This is the level-detector member of the ``level_*``
family — see :class:`engine.strategies.level_base.LevelStrategyBase`.

Distinct from ``fractal_breakout`` (same breakout *idea*, but its levels come from
the lighter ``indicators.detect_swing_*`` fractal pivots).

Entry is geometry-driven and convention-agnostic — it does not rely on the
detector's resistance/support labels, only on each active level's position
relative to the prior close:

  * **Long**  when the close decisively clears an *overhead* level
    (``prev_close < level``, ``close > level + buffer``).
  * **Short** when the close breaks an *underlying* level
    (``prev_close > level``, ``close < level − buffer``).

The stop is **structural**: anchored on the broken level (``level ∓ mult·ATR``)
and seeded at entry as ``Trade.stop_price``; the exit policy (preset
``structural_rr2``) checks it intrabar and takes profit at 2R, stop-first. A
native flip exits when the close crosses back through any active level.

Look-ahead free via :class:`LevelStrategyBase._active_levels` (confirmation +
causal invalidation). ATR(``level_atr_period``) drives the stop distance.
"""

from __future__ import annotations

import math

import pandas as pd

from ..core import Direction, ExitReason, PositionState
from .level_base import LevelStrategyBase


class LevelBreakoutStrategy(LevelStrategyBase):
    name = "level_breakout"

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

        # ── Exit (delegated stop/target first, then native level-cross flip) ─
        if state.current_trade is not None:
            trade = state.current_trade
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                return  # don't enter on the same bar as an exit
            # Native flip: price closes back through any active level.
            if trade.direction == Direction.LONG:
                if any(close < lvl <= prev_close for lvl in levels):
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                    return
            elif trade.direction == Direction.SHORT:
                if any(close > lvl >= prev_close for lvl in levels):
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                    return

        # ── Entry (breakout through a level) ───────────────────────────────
        if state.current_trade is not None:
            return  # already in position
        if not math.isfinite(atr_val) or atr_val <= 0:
            return

        buf = self.config.level_breakout_buffer_atr * atr_val
        mult = self.config.level_stop_atr_mult

        # Long: close clears an overhead level (was above the prior close).
        up = [lvl for lvl in levels if prev_close < lvl and close > lvl + buf]
        if up:
            broken = max(up)                       # nearest level cleared (just below close)
            stop = broken - mult * atr_val         # structural stop just under the reclaimed level
            if stop < close:
                state.enter(Direction.LONG, ts, close, stop_price=stop)
            return

        # Short: close breaks an underlying level (was below the prior close).
        down = [lvl for lvl in levels if prev_close > lvl and close < lvl - buf]
        if down:
            broken = min(down)                     # nearest level broken (just above close)
            stop = broken + mult * atr_val
            if stop > close:
                state.enter(Direction.SHORT, ts, close, stop_price=stop)
            return
