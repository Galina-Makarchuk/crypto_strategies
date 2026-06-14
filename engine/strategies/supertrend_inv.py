"""Inverse SuperTrend strategy.

Identical to SuperTrendStrategy but with flipped direction:
  - Trend flips UP  → SHORT (fading the breakout)
  - Trend flips DOWN → LONG  (buying the dip)

Useful as a mean-reversion counter-trend strategy in ranging markets,
or as a hedge/pair against the standard SuperTrend.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import atr as calc_atr
from ..indicators import supertrend as calc_supertrend
from ..core import Direction, ExitReason, PositionState
from ..strategy_configurator import SupertrendParams
from .base import BaseStrategy


class InverseSuperTrendStrategy(BaseStrategy):
    name = "supertrend_inv"

    def __init__(self, config: SupertrendParams, exit_policy=None):
        # Params reach the strategy here: BaseStrategy stores `config` as
        # self.config (after type-checking it against PARAMS[name]), and prepare()
        # / on_bar() read self.config.<knob>. The annotation documents the
        # expected family and makes the import load-bearing.
        super().__init__(config, exit_policy)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        st_line, trend_dir = calc_supertrend(
            df, self.config.supertrend_period, self.config.supertrend_mult
        )
        df["supertrend"] = st_line
        df["trend_dir"] = trend_dir
        df["atr"] = calc_atr(df, self.config.atr_period)
        return df

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        close = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]

        trend_now = int(df["trend_dir"].iloc[i])
        trend_prev = int(df["trend_dir"].iloc[i - 1])
        flip_up = trend_now == 1 and trend_prev == -1
        flip_down = trend_now == -1 and trend_prev == 1

        # ── Update trailing stop peak ──────────────────────────────────────
        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ─────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade
            # Signal-flip exit stays in-strategy, checked first. Because entries
            # are inverted (down-flip -> LONG, up-flip -> SHORT), a LONG closes on
            # the next up-flip and a SHORT on the next down-flip — the opposite
            # side from base supertrend.
            if (trade.direction == Direction.LONG and flip_up) or (
                trade.direction == Direction.SHORT and flip_down
            ):
                state.exit(ts, close, ExitReason.SIGNAL_FLIP)
                return
            decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                return

        # ── Entry logic (INVERTED) ─────────────────────────────────────────
        if state.current_trade is not None:
            return

        if flip_up:
            # Original goes long here → we go SHORT
            state.enter(Direction.SHORT, ts, close,
                        stop_price=self._entry_stop(Direction.SHORT, close, atr_val))
        elif flip_down:
            # Original goes short here → we go LONG
            state.enter(Direction.LONG, ts, close,
                        stop_price=self._entry_stop(Direction.LONG, close, atr_val))
