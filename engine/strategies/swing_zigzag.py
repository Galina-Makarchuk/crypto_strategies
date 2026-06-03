"""ATR-prominence ZigZag swing strategy.

Trades the reversal at each confirmed swing pivot from the swings detector:

    Confirmed swing HIGH (top is in)    → SHORT (fade the bounce)
    Confirmed swing LOW  (bottom is in) → LONG  (fade the dip)

Exit on the next opposite-side swing confirmation — which is also the next
entry signal, so the strategy flips on each confirmation. Confirmations are
indexed by ``Swing.confirmation_idx`` (the bar where the retrace cleared the
ATR-prominence threshold), so look-ahead is structurally impossible: at bar
``i`` the strategy only consumes swings with ``confirmation_idx <= i``.

Optional ATR trailing stop (Wilder-EWMA ATR at entry × ``stop_atr_mult``) as
a defensive exit in case the next swing takes too long to confirm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import Direction, ExitReason, PositionState
from ..strategy_configurator import StrategyConfig
from ..swings import detect_swings, wilder_atr
from .base import BaseStrategy


class SwingZigZagStrategy(BaseStrategy):
    name = "swing_zigzag"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        swings = detect_swings(
            df,
            atr_period=self.config.swing_zz_atr_period,
            min_prominence_atr=self.config.swing_zz_min_prominence_atr,
            min_bars_between=self.config.swing_zz_min_bars_between,
            vol_lookback=self.config.swing_zz_vol_lookback,
            return_provisional=False,
        )
        min_score = self.config.swing_zz_min_score
        if min_score > 0:
            swings = [s for s in swings if s.score >= min_score]

        n = len(df)
        signal = np.zeros(n, dtype=np.int8)              # +1 long, -1 short, 0 none
        pivot_idx = np.full(n, -1, dtype=np.int64)
        pivot_price = np.full(n, np.nan, dtype=np.float64)
        pivot_side = np.empty(n, dtype=object)
        pivot_side[:] = ""
        score_arr = np.zeros(n, dtype=np.float64)
        prominence_arr = np.zeros(n, dtype=np.float64)

        for s in swings:
            ci = s.confirmation_idx
            signal[ci] = 1 if s.side == "low" else -1
            pivot_idx[ci] = s.idx
            pivot_price[ci] = s.price
            pivot_side[ci] = s.side
            score_arr[ci] = s.score
            prominence_arr[ci] = s.prominence_atr

        df["swing_signal"] = signal
        df["swing_pivot_idx"] = pivot_idx
        df["swing_pivot_price"] = pivot_price
        df["swing_pivot_side"] = pivot_side
        df["swing_score"] = score_arr
        df["swing_prominence_atr"] = prominence_arr
        df["swing_atr"] = wilder_atr(df, self.config.swing_zz_atr_period)
        return df

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        signal = int(df["swing_signal"].iloc[i])
        close_i = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        atr_val = df["swing_atr"].iloc[i]
        ts = df.index[i]

        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ────────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade

            # Trailing stop (optional, ATR-distance from peak / trough), delegated
            # to the exit policy. Gated by use_stop + a valid ATR, and does NOT
            # return — a flip-through entry may follow on the same bar (below).
            if (
                self.config.swing_zz_use_stop
                and np.isfinite(atr_val)
                and atr_val > 0
            ):
                decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
                if decision is not None:
                    state.exit(ts, decision.price, decision.reason)

            # Opposite-side swing → flip (exit, then fall through to entry).
            if state.current_trade is not None:
                trade = state.current_trade
                flip_long = trade.direction == Direction.LONG and signal == -1
                flip_short = trade.direction == Direction.SHORT and signal == 1
                if flip_long or flip_short:
                    state.exit(ts, close_i, ExitReason.SIGNAL_FLIP)

        # ── Entry logic (flip-through allowed when flat now) ─────────────────
        if state.current_trade is None:
            if signal == 1:
                state.enter(Direction.LONG, ts, close_i)
            elif signal == -1:
                state.enter(Direction.SHORT, ts, close_i)
