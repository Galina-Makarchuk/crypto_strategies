"""Shared plumbing for the level-detector strategy family.

The ``level_*`` strategies (``level_breakout`` today; ``level_bounce`` /
``level_retest`` planned) all trade the **horizontal S/R levels** produced by the
dedicated :mod:`engine.level_detector` — stateful resistance / support / pullback
levels seeded at confirmed pivots and tracked forward until *invalidated*. This is
a different, richer level source than the fractal pivots used by
``fractal_breakout`` (``indicators.detect_swing_*``).

This base computes the level set + ATR once in :meth:`prepare` and exposes the
**look-ahead-free** active levels at each bar via :meth:`_active_levels`.
Subclasses implement only ``on_bar`` (their own entry/exit geometry).

Look-ahead contract — a detector :class:`~engine.level_detector.Level` is usable
at bar ``i`` only when both hold, and both are causal (decided from data ≤ ``i``):

* **Confirmed**: ``start_idx + pivot_window <= i`` — a pivot's full symmetric
  neighbourhood must be in before it can be confirmed.
* **Alive**: ``invalidated_at is None or i <= invalidated_at`` — the detector's
  first-invalidation bar (computed from data up to that bar only), so a breakout
  bar that consumes the level can still see it, and it drops the next bar.
"""

from __future__ import annotations

import pandas as pd

from ..core import PositionState
from ..indicators import atr
from ..level_detector import detect_all_levels
from ..strategy_configurator import StrategyConfig
from .base import BaseStrategy


class LevelStrategyBase(BaseStrategy):
    """Base for strategies built on :mod:`engine.level_detector`."""

    # Detector families used as the horizontal S/R set (pullback added by config).
    _BASE_FAMILIES = ("resistance", "support")

    def __init__(self, config: StrategyConfig, exit_policy=None):
        super().__init__(config, exit_policy)
        # One tuple per detector level: (confirmation_idx, invalidated_at|None, price)
        self._levels: list[tuple[int, int | None, float]] = []

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config
        df["atr"] = atr(df, cfg.level_atr_period)

        families = detect_all_levels(
            df,
            delta_resistance=cfg.level_delta,
            delta_support=cfg.level_delta,
            delta_pullback=cfg.level_delta,
            inval_resistance=cfg.level_invalidation_candles,
            inval_support=cfg.level_invalidation_candles,
            inval_pullback=cfg.level_invalidation_candles,
            pivot_window_resistance=cfg.level_pivot_window,
            pivot_window_support=cfg.level_pivot_window,
            pivot_window_pullback=cfg.level_pivot_window,
            delta_mode=cfg.level_delta_mode,
            atr_period=cfg.level_atr_period,
        )

        chosen = list(self._BASE_FAMILIES)
        if cfg.level_use_pullback:
            chosen.append("pullback")

        pw = cfg.level_pivot_window
        self._levels = [
            (lvl.start_idx + pw, lvl.invalidated_at, float(lvl.price))
            for fam in chosen
            for lvl in families[fam]
        ]
        return df

    def _active_levels(self, i: int) -> list[float]:
        """Prices of levels confirmed by, and not yet invalidated as of, bar ``i``.

        See the module docstring for the (causal) look-ahead contract."""
        return [
            price
            for conf, inval, price in self._levels
            if conf <= i and (inval is None or i <= inval)
        ]

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:  # pragma: no cover
        raise NotImplementedError
