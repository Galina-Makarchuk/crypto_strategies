"""Base strategy interface.

The key design decision: strategies receive the full DataFrame for indicator
pre-computation, but signal generation happens bar-by-bar through `on_bar()`.
The backtester calls `on_bar(i)` with only past data visible (i.e., `df[:i+1]`
is the "known" universe).  This structurally prevents look-ahead bias.

Strategies must:
  1. Implement `prepare(df)` — compute any indicators (adds columns to a *copy*).
  2. Implement `on_bar(i, df, state)` — evaluate bar `i` and optionally call
     `state.enter()` / `state.exit()`.  MUST NOT read df beyond index `i`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ..exits import ExitContext, ExitPolicy
from ..core import Direction, PositionState, Trade
from ..strategy_configurator import exit_policy_for, params_class_for


class BaseStrategy(ABC):
    """Abstract base for all strategies."""

    name: str = "base"

    def __init__(self, config, exit_policy: Optional[ExitPolicy] = None):
        # Type-check the config against what this strategy expects (its PARAMS
        # entry). Catches the mix-up the per-strategy split can't catch by
        # itself: handing a strategy an entirely wrong — but valid — config
        # object (e.g. EMACrossoverStrategy(SupertrendParams())), which would
        # otherwise blow up later and cryptically inside prepare(). Skipped for
        # ad-hoc / test strategies not in the registry.
        expected = params_class_for(self.name)
        if expected is not None and not isinstance(config, expected):
            raise TypeError(
                f"{type(self).__name__} ({self.name!r}) expects a "
                f"{expected.__name__}, got {type(config).__name__}. "
                f"Build it from params_for({self.name!r})."
            )
        self.config = config
        # The exit/TP mechanism for this run. Defaults to the strategy's assigned
        # preset (see strategy_configurator.exit_policy_for); a caller may inject a
        # different one. Strategies that haven't been migrated to delegated exits
        # simply don't reference it — so this is inert for them.
        self.exit_policy = exit_policy if exit_policy is not None else exit_policy_for(self.name)

    def _exit_ctx(
        self,
        i: int,
        df: pd.DataFrame,
        trade: Trade,
        atr: float,
        ref_stop: Optional[float] = None,
        ref_target: Optional[float] = None,
    ) -> ExitContext:
        """Build the per-bar context an ExitPolicy needs to judge bar `i`."""
        return ExitContext(
            direction=trade.direction,
            entry_price=trade.entry_price,
            peak_price=trade.peak_price,
            high=float(df["high"].iloc[i]),
            low=float(df["low"].iloc[i]),
            close=float(df["close"].iloc[i]),
            atr=atr,
            stop_price=trade.stop_price,
            ref_stop=ref_stop,
            ref_target=ref_target,
        )

    def _entry_stop(
        self,
        direction: Direction,
        price: float,
        atr: float,
        ref_stop: Optional[float] = None,
    ) -> Optional[float]:
        """The exit policy's initial stop for a new entry (peak == entry), to seed
        ``state.enter(stop_price=…)`` so risk-based sizing can use it."""
        ctx = ExitContext(
            direction=direction, entry_price=price, peak_price=price,
            high=price, low=price, close=price, atr=atr, ref_stop=ref_stop,
        )
        return self.exit_policy.initial_stop(ctx)

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute indicators.  Returns a NEW DataFrame (no mutation)."""
        ...

    @abstractmethod
    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        """Evaluate bar `i`.  May call state.enter() / state.exit().

        Contract:
          - You may read df.iloc[0 .. i] (inclusive).
          - You MUST NOT read df.iloc[i+1 ..].
          - If state.status is OPEN, you MUST call state.update_peak(high, low)
            before checking exits — the runner does not do this for you; the
            trailing stop tracks Trade.peak_price, not the previous bar's close.
        """
        ...
