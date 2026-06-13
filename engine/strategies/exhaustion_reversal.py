"""Exhaustion-reversal strategy.

Trades local tops/bottoms where a directional push runs out of conviction.
In streak terms (a streak = a run of same-direction candles collapsed to a
signed length):

    push leg : a completed streak with |length| >= push_min_len
    stall    : N consecutive completed streaks each with |length| <= stall_max_len
    trigger  : the in-progress opposite-direction streak, |length| >= trigger_min_len
               AND per-candle volume >= push leg per-candle volume * factor

Short entry forms at the end of:  BUY push  → stall → SELL trigger
Long  entry is the mirror:         SELL push → stall → BUY  trigger

Exit rules:
  - hard stop above (short) / below (long) the push+stall extreme + ATR buffer
  - fixed R:R target
  - time stop after K bars
  - structural invalidation: a newly-completed streak in the push direction
    with |length| >= invalidation_len

No look-ahead: the streak tracker is advanced one bar at a time and only
inspects the current and prior bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..indicators import atr as calc_atr
from ..core import Direction, ExitReason, PositionState
from ..strategy_configurator import ExhaustionParams
from .base import BaseStrategy


@dataclass
class _Streak:
    start_idx: int
    end_idx: int
    direction: int
    length: int
    volume: float
    high: float
    low: float

    @property
    def per_candle_volume(self) -> float:
        return self.volume / self.length if self.length else 0.0


class _StreakTracker:
    """Incremental directional-run tracker.

    A doji candle (close == open) terminates the current streak but is not
    emitted as a streak itself, matching the notebook algorithm.
    """

    def __init__(self) -> None:
        self.completed: list[_Streak] = []
        self._dir: int = 0
        self._start: int = 0
        self._vol: float = 0.0
        self._high: float = -np.inf
        self._low: float = np.inf

    def update(
        self, i: int, dir_i: int, high: float, low: float, volume: float
    ) -> Optional[_Streak]:
        """Advance tracker by one bar. Returns any streak that just closed."""
        if self._dir == 0:
            if dir_i != 0:
                self._dir = dir_i
                self._start = i
                self._vol = volume
                self._high = high
                self._low = low
            return None

        if dir_i == self._dir:
            self._vol += volume
            self._high = max(self._high, high)
            self._low = min(self._low, low)
            return None

        completed = _Streak(
            start_idx=self._start,
            end_idx=i - 1,
            direction=self._dir,
            length=i - self._start,
            volume=self._vol,
            high=self._high,
            low=self._low,
        )
        self.completed.append(completed)

        if dir_i != 0:
            self._dir = dir_i
            self._start = i
            self._vol = volume
            self._high = high
            self._low = low
        else:
            self._dir = 0
            self._vol = 0.0
            self._high = -np.inf
            self._low = np.inf
        return completed

    @property
    def in_progress_direction(self) -> int:
        return self._dir

    @property
    def in_progress_start(self) -> int:
        return self._start

    @property
    def in_progress_volume(self) -> float:
        return self._vol

    def in_progress_length(self, current_bar: int) -> int:
        return current_bar - self._start + 1 if self._dir != 0 else 0


class ExhaustionReversalStrategy(BaseStrategy):
    """Fade exhausted directional pushes at local tops and bottoms."""

    name = "exhaustion_reversal"

    def __init__(self, config: ExhaustionParams, exit_policy=None):
        super().__init__(config, exit_policy)
        self._tracker = _StreakTracker()
        self._entry_bar_idx: int = -1
        self._target_price: float = 0.0
        self._last_push_start_idx: int = -1

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = calc_atr(df, self.config.atr_period)
        self._tracker = _StreakTracker()
        self._entry_bar_idx = -1
        self._target_price = 0.0
        self._last_push_start_idx = -1
        return df

    def _find_setup(
        self, trigger_dir: int
    ) -> Optional[tuple[_Streak, float]]:
        """Locate a valid push-stall pattern preceding the in-progress trigger.

        Returns (push_leg, extreme_price) where extreme_price is the cluster's
        high (for short trigger) or low (for long trigger). Returns None if no
        valid pattern is present.
        """
        push_min = self.config.exhaustion_push_min_len
        stall_min = self.config.exhaustion_stall_min_count
        stall_max = self.config.exhaustion_stall_max_len

        completed = self._tracker.completed
        if len(completed) < 1 + stall_min:
            return None

        push_idx: Optional[int] = None
        for k in range(len(completed) - 1, -1, -1):
            s = completed[k]
            if s.length >= push_min:
                push_idx = k
                break
            if s.length > stall_max:
                return None

        if push_idx is None:
            return None

        push = completed[push_idx]
        if push.direction * trigger_dir >= 0:
            return None

        stall_streaks = completed[push_idx + 1 :]
        if len(stall_streaks) < stall_min:
            return None
        if any(s.length > stall_max for s in stall_streaks):
            return None

        if push.start_idx == self._last_push_start_idx:
            return None

        if trigger_dir == -1:
            extreme = max([push.high] + [s.high for s in stall_streaks])
        else:
            extreme = min([push.low] + [s.low for s in stall_streaks])

        return push, extreme

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        open_i = df["open"].iloc[i]
        close_i = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        volume_i = df["volume"].iloc[i]
        ts = df.index[i]
        atr_val = df["atr"].iloc[i]

        if close_i > open_i:
            dir_i = 1
        elif close_i < open_i:
            dir_i = -1
        else:
            dir_i = 0

        completed_streak = self._tracker.update(i, dir_i, high_i, low_i, volume_i)

        if i < self.config.atr_period or np.isnan(atr_val) or atr_val <= 0:
            return

        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade

            # Fixed stop + target delegated to the exit policy (stop-first,
            # intrabar fills). Stop sits on the trade's stop_price; target is the
            # strategy's R:R level, supplied as ref_target.
            decision = self.exit_policy.evaluate(
                self._exit_ctx(i, df, trade, atr_val, ref_target=self._target_price)
            )
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                return

            inv_len = self.config.exhaustion_invalidation_len
            if completed_streak is not None and completed_streak.length >= inv_len:
                if (trade.direction == Direction.SHORT and completed_streak.direction == 1) or (
                    trade.direction == Direction.LONG and completed_streak.direction == -1
                ):
                    state.exit(ts, close_i, ExitReason.INVALIDATION)
                    return

            if i - self._entry_bar_idx >= self.config.exhaustion_time_stop_bars:
                state.exit(ts, close_i, ExitReason.TIME_STOP)
                return

        if state.current_trade is not None:
            return

        # ── Entry logic ───────────────────────────────────────────────────
        cur_dir = self._tracker.in_progress_direction
        if cur_dir == 0:
            return
        cur_len = self._tracker.in_progress_length(i)
        if cur_len < self.config.exhaustion_trigger_min_len:
            return

        setup = self._find_setup(trigger_dir=cur_dir)
        if setup is None:
            return
        push, extreme = setup

        trigger_per_candle = self._tracker.in_progress_volume / cur_len
        if trigger_per_candle < self.config.exhaustion_volume_factor * push.per_candle_volume:
            return

        buffer = self.config.exhaustion_stop_atr_mult * atr_val
        if cur_dir == -1:
            stop = extreme + buffer
            risk = stop - close_i
            if risk <= 0:
                return
            target = close_i - self.config.exhaustion_target_rr * risk
            direction = Direction.SHORT
        else:
            stop = extreme - buffer
            risk = close_i - stop
            if risk <= 0:
                return
            target = close_i + self.config.exhaustion_target_rr * risk
            direction = Direction.LONG

        sig = state.enter(direction, ts, close_i, stop_price=stop)
        if sig is not None:
            self._entry_bar_idx = i
            self._target_price = target
            self._last_push_start_idx = push.start_idx
