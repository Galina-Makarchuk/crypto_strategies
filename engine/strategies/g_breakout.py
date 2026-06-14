"""Squeeze breakout — breakout through a level after a tight compression.

The same breakout idea as :class:`engine.strategies.level_breakout.\
LevelBreakoutStrategy`, but gated by the compression pre-break filter that the
plain breakout omits: the approach into the level must be a tight consolidation
(small bars crawling toward the level), and any oversized ("paranormal") bar in
the approach window vetoes the trade. Big impulsive bars into a level are a
red flag for false breakouts, so they are skipped.

Built on the :mod:`engine.levels` detectors (selectable via
``LevelParams.level_detector``; default ``cluster_level``). Entry is
geometry-driven and convention-agnostic:

  * **Long**  when the close decisively clears an *overhead* level
    (``prev_close < level``, ``close > level + buffer``) after a tight approach.
  * **Short** when the close breaks an *underlying* level
    (``prev_close > level``, ``close < level − buffer``) after a tight approach.

The stop is **structural**, anchored on the broken level (``level ∓ mult·ATR``)
and seeded at entry as ``Trade.stop_price``; the exit policy (preset
``structural_rr3``) checks it intrabar and takes profit at 3R, stop-first. A
native flip exits when the close crosses back through any active level.

Look-ahead free via :class:`LevelStrategyBase._active_levels`; the approach
window inspects only bars strictly before the breakout bar. ATR
(``level_atr_period``) drives the buffer, the compression test, and the stop.
"""

from __future__ import annotations

import math

import pandas as pd

from ..core import Direction, ExitReason, PositionState
from .level_base import LevelStrategyBase


class GBreakoutStrategy(LevelStrategyBase):
    name = "g_breakout"

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

        # ── In position: peak → delegated exit, then native level-cross flip ─
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)
            trade = state.current_trade
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                return
            if trade.direction == Direction.LONG:
                if any(close < lvl <= prev_close for lvl in levels):
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
            elif trade.direction == Direction.SHORT:
                if any(close > lvl >= prev_close for lvl in levels):
                    state.exit(ts, close, ExitReason.SIGNAL_FLIP)
            return

        # ── Entry (compression then breakout) ──────────────────────────────
        if not math.isfinite(atr_val) or atr_val <= 0:
            return

        cfg = self.config
        n = cfg.g_breakout_consol_bars
        if i < n + 1:
            return

        # Inspect the n bars strictly before the breakout bar (the approach).
        approach = df.iloc[i - n:i]
        rng = approach["high"] - approach["low"]
        if rng.mean() > cfg.g_breakout_consol_max_atr * atr_val:
            return  # approach not tight enough — no squeeze
        if (rng >= cfg.g_breakout_paranormal_atr * atr_val).any():
            return  # an oversized (paranormal) bar in the approach vetoes the trade

        buf = cfg.g_breakout_buffer_atr * atr_val
        mult = cfg.g_breakout_stop_atr_mult

        # Long: close clears an overhead level (was above the prior close).
        up = [lvl for lvl in levels if prev_close < lvl and close > lvl + buf]
        if up:
            broken = max(up)                       # nearest level cleared (just below close)
            stop = broken - mult * atr_val
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
