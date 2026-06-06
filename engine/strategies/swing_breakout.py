"""Swing breakout strategy — continuation through confirmed swing pivots.

Ported from the `ema` project's ``swing_strategy.py`` *breakout* mode — the third
entry mode of the ATR-prominence ZigZag family (alongside ``swing_flip`` and
``swing_bounce``), all built on the same ``swing_detector.detect_swings``.

Where bounce *fades* a swing level, breakout *rides through* it:

  * Long  when close breaks **above** the most recent confirmed swing **high** by
    ``swing_breakout_buffer_atr × ATR``.
  * Short when close breaks **below** the most recent confirmed swing **low**.

The stop is swing-anchored on the broken level: ``swing_high − mult × ATR``
(long) / ``swing_low + mult × ATR`` (short), seeded at entry as
``Trade.stop_price``; the exit policy (default preset ``structural_rr2``) checks
it intrabar and takes profit at 2R, stop-first.

Look-ahead free: only swings with ``confirmation_idx <= i`` are active, and a swing
is invalidated once a bar closes through it (so the breakout bar consumes the
level, then drops it). Direction is the framework gate; cost/sizing follow this
engine, not the source's baked-in fees.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..swing_detector import detect_swings, wilder_atr
from ..core import Direction, PositionState
from .base import BaseStrategy


class SwingBreakoutStrategy(BaseStrategy):
    name = "swing_breakout"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config

        swings = detect_swings(
            df,
            atr_period=cfg.swing_zz_atr_period,
            min_prominence_atr=cfg.swing_zz_min_prominence_atr,
            min_bars_between=cfg.swing_zz_min_bars_between,
            vol_lookback=cfg.swing_zz_vol_lookback,
            return_provisional=False,
        )
        if cfg.swing_zz_min_score > 0:
            swings = [s for s in swings if s.score >= cfg.swing_zz_min_score]

        # The active swing low / high reference for each bar: visible at its
        # confirmation_idx, invalidated once a bar closes through it (close > swing
        # high / close < swing low). Invalidation is applied at the END of each bar
        # (so the breakout bar sees the level before it is dropped) — source parity
        # with swing_strategy.py:696-700.
        n = len(df)
        close = df["close"].to_numpy()
        active_low = np.full(n, np.nan)
        active_high = np.full(n, np.nan)
        last_low = math.nan
        last_high = math.nan
        ordered = sorted(swings, key=lambda s: s.confirmation_idx)
        ptr = 0
        for i in range(n):
            while ptr < len(ordered) and ordered[ptr].confirmation_idx <= i:
                sw = ordered[ptr]
                if sw.side == "low":
                    last_low = sw.price
                else:
                    last_high = sw.price
                ptr += 1
            active_low[i] = last_low
            active_high[i] = last_high
            if math.isfinite(last_high) and close[i] > last_high:
                last_high = math.nan
            if math.isfinite(last_low) and close[i] < last_low:
                last_low = math.nan

        df["swing_breakout_active_low"] = active_low
        df["swing_breakout_active_high"] = active_high
        df["swing_breakout_atr"] = wilder_atr(df, cfg.swing_zz_atr_period)

        self._last_exit_bar = -10**9
        return df

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        atr_val = df["swing_breakout_atr"].iloc[i]

        # ── update peak → exit (when open) ──
        if state.current_trade is not None:
            trade = state.current_trade
            state.update_peak(high_i, low_i)
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is None:
                return  # still open → no entry this bar
            state.exit(df.index[i], decision.price, decision.reason)
            self._last_exit_bar = i
            # Source runs manage → entry sequentially: an exit can be followed by a
            # same-bar re-entry (gated by the cooldown below).

        # ── entry (now flat): long tried first ──
        if not math.isfinite(atr_val) or atr_val <= 0:
            return
        if i - self._last_exit_bar < self.config.swing_breakout_min_bars_between_trades:
            return

        if self._enter_side(Direction.LONG, i, df, state):
            return
        self._enter_side(Direction.SHORT, i, df, state)

    def _enter_side(self, direction: Direction, i: int, df: pd.DataFrame, state: PositionState) -> bool:
        cfg = self.config
        close = df["close"].iloc[i]
        atr_val = df["swing_breakout_atr"].iloc[i]
        buf = cfg.swing_breakout_buffer_atr * atr_val

        if direction is Direction.LONG:
            # Break ABOVE the most recent confirmed swing high.
            ref = df["swing_breakout_active_high"].iloc[i]
            if not math.isfinite(ref):
                return False
            if not (close > ref + buf):
                return False
            stop = ref - cfg.swing_breakout_stop_atr_mult * atr_val
            if stop >= close:
                return False
        else:
            # Break BELOW the most recent confirmed swing low.
            ref = df["swing_breakout_active_low"].iloc[i]
            if not math.isfinite(ref):
                return False
            if not (close < ref - buf):
                return False
            stop = ref + cfg.swing_breakout_stop_atr_mult * atr_val
            if stop <= close:
                return False

        return state.enter(direction, df.index[i], close, stop_price=stop) is not None
