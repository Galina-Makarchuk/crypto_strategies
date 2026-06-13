"""Swing bounce strategy — mean-reversion off confirmed swing pivots.

Ported from the `ema` project's ``swing_strategy.py`` *bounce* mode (validated at
~77% bounce rate at prominence tier 1.5 in that project). It reuses our
ATR-prominence ZigZag detector (``swing_detector.detect_swings``) and trades the
*test* of the most recently confirmed swing:

  * Long  when the bar's low comes within ``test_tolerance_atr × ATR`` of the
    active swing **low** and (optionally) the bar closes back above it.
  * Short when the bar's high comes within tolerance of the active swing
    **high** and (optionally) the bar closes back below it.

The stop is *swing-anchored*: ``swing_low − mult × ATR`` (long) / ``swing_high +
mult × ATR`` (short), seeded at entry as ``Trade.stop_price``. The exit policy
(default preset ``structural_rr2``) checks that fixed stop intrabar and takes
profit at 2R — stop-first on an ambiguous bar, matching the source.

Look-ahead free: only swings whose ``confirmation_idx <= i`` are ever active, and
the entry/stop read only bar ``i``. Direction (long/short/both) is the framework
gate; the strategy emits both sides (long first) and ``state.enter`` vetoes
disallowed ones. Cost/sizing follow this engine (TradingConfig), not the source's
baked-in fees.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..swing_detector import detect_swings, wilder_atr
from ..core import Direction, PositionState
from .base import BaseStrategy


class SwingBounceStrategy(BaseStrategy):
    name = "swing_bounce"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config

        # Detection reuses the ATR-prominence ZigZag knobs (swing_zz_*), honest
        # (no trailing provisional swing), then an optional min-score filter.
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

        # The active swing low / high reference for each bar's entry: a swing
        # becomes visible at its confirmation_idx, and is *invalidated* (dropped)
        # once a bar closes through it — close > swing high, or close < swing low
        # (source parity, swing_strategy.py:696-700). Invalidation is applied at
        # the END of each bar, so it affects the next bar's entry, not this one's.
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
            active_low[i] = last_low      # reference as seen at bar i's entry
            active_high[i] = last_high
            # End-of-bar invalidation (affects bar i+1 onward).
            if math.isfinite(last_high) and close[i] > last_high:
                last_high = math.nan
            if math.isfinite(last_low) and close[i] < last_low:
                last_low = math.nan

        df["swing_bounce_active_low"] = active_low
        df["swing_bounce_active_high"] = active_high
        df["swing_bounce_atr"] = wilder_atr(df, cfg.swing_zz_atr_period)

        self._last_exit_bar = -10**9
        return df

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        atr_val = df["swing_bounce_atr"].iloc[i]

        # ── update peak → exit (when open) ──
        if state.current_trade is not None:
            trade = state.current_trade
            state.update_peak(high_i, low_i)
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is None:
                return  # still open → no entry this bar
            state.exit(df.index[i], decision.price, decision.reason)
            self._last_exit_bar = i
            # The source runs manage → entry sequentially (not if/elif), so an exit
            # can be followed by a same-bar re-entry (gated by the cooldown below).

        # ── entry (now flat): long tried first ──
        if not math.isfinite(atr_val) or atr_val <= 0:
            return
        # Cooldown after an exit (default 1 blocks same-bar re-entry; set 0 to
        # re-enable the source's same-bar exit→re-entry on one bar's OHLC).
        if i - self._last_exit_bar < self.config.swing_bounce_min_bars_between_trades:
            return

        if self._enter_side(Direction.LONG, i, df, state):
            return
        self._enter_side(Direction.SHORT, i, df, state)

    def _enter_side(self, direction: Direction, i: int, df: pd.DataFrame, state: PositionState) -> bool:
        cfg = self.config
        close = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        atr_val = df["swing_bounce_atr"].iloc[i]
        tol = cfg.swing_bounce_test_tolerance_atr * atr_val

        if direction is Direction.LONG:
            ref = df["swing_bounce_active_low"].iloc[i]
            if not math.isfinite(ref):
                return False  # no confirmed swing low yet
            touched = abs(low_i - ref) <= tol
            rejected = (not cfg.swing_bounce_require_close_rejection) or (close > ref)
            stop = ref - cfg.swing_bounce_stop_atr_mult * atr_val
            if not (touched and rejected) or stop >= close:
                return False
        else:
            ref = df["swing_bounce_active_high"].iloc[i]
            if not math.isfinite(ref):
                return False
            touched = abs(high_i - ref) <= tol
            rejected = (not cfg.swing_bounce_require_close_rejection) or (close < ref)
            stop = ref + cfg.swing_bounce_stop_atr_mult * atr_val
            if not (touched and rejected) or stop <= close:
                return False

        return state.enter(direction, df.index[i], close, stop_price=stop) is not None
