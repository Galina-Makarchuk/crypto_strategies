"""VWAP Stdev Bands strategy — session-anchored mean reversion.

Port of the TradingView "VWAP Stdev Bands v2 Mod" idea:

    Entry: price closes outside the furthest stdev band (cross from inside
           to outside on the current bar). Long below the lower band, short
           above the upper band.
    Exit:  price closes back through the middle (VWAP).

The VWAP and its bands reset at each session boundary (default: UTC day),
matching the Pine source's `security(tickerid, "D", time)` anchor.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import vwap_stdev_bands as calc_vwap_bands
from ..models import Direction, ExitReason, PositionState, StrategyConfig
from .base import BaseStrategy


class VWAPBandsStrategy(BaseStrategy):
    name = "vwap_bands"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        vwap, bands = calc_vwap_bands(
            df,
            devs=self.config.vwap_band_devs,
            session=self.config.vwap_session,
        )
        df["vwap"] = vwap
        for k, (upper, lower) in enumerate(bands):
            df[f"vwap_upper_{k}"] = upper
            df[f"vwap_lower_{k}"] = lower
        return df

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < 1:
            return

        entry = self.config.vwap_entry_band
        upper_col = f"vwap_upper_{entry}"
        lower_col = f"vwap_lower_{entry}"

        close_i = df["close"].iloc[i]
        close_prev = df["close"].iloc[i - 1]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        vwap_i = df["vwap"].iloc[i]
        upper_i = df[upper_col].iloc[i]
        lower_i = df[lower_col].iloc[i]
        upper_prev = df[upper_col].iloc[i - 1]
        lower_prev = df[lower_col].iloc[i - 1]
        ts = df.index[i]

        if pd.isna(vwap_i) or pd.isna(upper_i) or pd.isna(lower_i):
            return

        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit: close returns to the VWAP middle ───────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade
            if trade.direction == Direction.LONG and close_i >= vwap_i:
                state.exit(ts, close_i, ExitReason.TAKE_PROFIT)
                return
            if trade.direction == Direction.SHORT and close_i <= vwap_i:
                state.exit(ts, close_i, ExitReason.TAKE_PROFIT)
                return

        if state.current_trade is not None:
            return

        # ── Entry: close crosses the furthest band ───────────────────────────
        if pd.isna(upper_prev) or pd.isna(lower_prev):
            return

        cross_below_lower = close_prev >= lower_prev and close_i < lower_i
        cross_above_upper = close_prev <= upper_prev and close_i > upper_i

        if cross_below_lower:
            state.enter(Direction.LONG, ts, close_i)
        elif cross_above_upper:
            state.enter(Direction.SHORT, ts, close_i)
