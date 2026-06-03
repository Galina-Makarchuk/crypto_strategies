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

import pandas as pd

from ..models import Direction, PositionState
from ..strategy_configurator import StrategyConfig


class BaseStrategy(ABC):
    """Abstract base for all strategies."""

    name: str = "base"

    def __init__(self, config: StrategyConfig):
        self.config = config

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
          - If state.status is OPEN, you MUST call state.update_peak() first
            (the backtester does this, but the strategy can also do it).
        """
        ...
