"""Range / channel fade — trade reversals between two horizontal levels.

When price trades between a level below and a level above, buy near the lower edge
and sell near the upper edge, targeting the far edge with a cushion. The channel
must be wide enough to be worth fading (``g_range_min_width_atr`` × ATR), and
entries fire only in the bottom/top zone of the channel (``g_range_entry_zone``)
so there is runway to the far edge.

Built on the :mod:`engine.levels` detectors (selectable via
``LevelParams.level_detector``; default ``cluster_level``, whose merged pivots
give clean, stable channel edges). At each bar the active levels are split around the close into the nearest
edge below (support) and above (resistance); together they define the channel.
Entry is geometry-driven and convention-agnostic:

  * **Long**  from the bottom zone: ``pos_in_channel <= entry_zone`` and the low
    tests the lower edge. Stop = ``lower − buffer``.
  * **Short** from the top zone: ``pos_in_channel >= 1 − entry_zone`` and the high
    tests the upper edge. Stop = ``upper + buffer``.

The exit is the ``structural`` preset: a structural stop (the entry stop_price,
filled intrabar beyond the near edge) plus a structural target supplied each bar
as ``ref_target`` — the far edge minus a cushion of the channel width.

Look-ahead free via :class:`LevelStrategyBase._active_levels`. ATR
(``level_atr_period``) drives the width gate, the edge-test tolerance, and the stop.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from ..core import Direction, PositionState
from .level_base import LevelStrategyBase


class GRangeStrategy(LevelStrategyBase):
    name = "g_range"

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        close = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]

        levels = self._active_levels(i)

        # ── In position: peak → structural stop + far-edge target ───────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)
            trade = state.current_trade
            ref_target = self._far_edge_target(trade.direction, levels, close)
            decision = self.exit_policy.evaluate(
                self._exit_ctx(i, df, trade, atr_val, ref_target=ref_target)
            )
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
            return

        # ── Entry (fade an edge of a wide-enough channel) ───────────────────
        if not math.isfinite(atr_val) or atr_val <= 0:
            return

        below = [lvl for lvl in levels if lvl < close]
        above = [lvl for lvl in levels if lvl > close]
        if not below or not above:
            return  # need a bracketed channel

        lower = max(below)                         # nearest edge below
        upper = min(above)                         # nearest edge above
        width = upper - lower
        cfg = self.config
        if width <= 0 or width < cfg.g_range_min_width_atr * atr_val:
            return  # channel too tight to fade

        tol = cfg.g_range_tol_atr * atr_val
        buf = cfg.g_range_stop_buffer_atr * atr_val
        pos = (close - lower) / width              # 0 at lower edge, 1 at upper edge

        # Long from the bottom of the channel.
        if pos <= cfg.g_range_entry_zone and (low_i - lower) <= tol:
            stop = lower - buf
            if stop < close:
                state.enter(Direction.LONG, ts, close, stop_price=stop)
            return

        # Short from the top of the channel.
        if pos >= 1.0 - cfg.g_range_entry_zone and (upper - high_i) <= tol:
            stop = upper + buf
            if stop > close:
                state.enter(Direction.SHORT, ts, close, stop_price=stop)
            return

    def _far_edge_target(self, direction: Direction, levels: list[float],
                         close: float) -> Optional[float]:
        """The cushioned far edge of the channel for an open trade, recomputed
        each bar from the active levels (None if the far edge has gone)."""
        cushion = self.config.g_range_target_cushion
        if direction == Direction.LONG:
            above = [lvl for lvl in levels if lvl > close]
            if not above:
                return None
            far = min(above)
            below = [lvl for lvl in levels if lvl < close]
            width = far - max(below) if below else far - close
            return far - cushion * width
        below = [lvl for lvl in levels if lvl < close]
        if not below:
            return None
        far = max(below)
        above = [lvl for lvl in levels if lvl > close]
        width = min(above) - far if above else close - far
        return far + cushion * width
