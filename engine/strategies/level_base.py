"""Shared plumbing for the level-detector strategy family.

The ``level_*`` strategies (``level_breakout`` today; ``level_bounce`` /
``level_retest`` planned) all trade the **horizontal S/R levels** produced by the
dedicated :mod:`engine.levels` package. The detector is selectable per run via
``LevelParams.level_detector`` (``pivot_level`` / ``cluster_level`` /
``touch_level``); :func:`engine.levels.detect_levels` dispatches on it and every
source returns the same families dict, so this base is detector-agnostic. This is
a different, richer level source than the fractal pivots used by
``fractal_breakout`` (``indicators.detect_swing_*``).

This base computes the level set + ATR once in :meth:`prepare` and exposes the
**look-ahead-free** active levels at each bar via :meth:`_active_levels`.
Subclasses implement only ``on_bar`` (their own entry/exit geometry).

Look-ahead contract — a detector :class:`~engine.levels.Level` is usable at bar
``i`` only when both hold, and both are causal (decided from data ≤ ``i``):

* **Confirmed**: ``confirmed_idx <= i`` — the level became observable by bar ``i``.
  Each source sets ``confirmed_idx`` causally (pivot_level: ``start_idx +
  pivot_window``; cluster/touch: the bar the level first became observable).
* **Alive**: ``invalidated_at is None or i <= invalidated_at`` — the detector's
  first-invalidation bar (computed from data up to that bar only), so a breakout
  bar that consumes the level can still see it, and it drops the next bar.

How it works:
The level_base uses the levels detected by engine.levels - their exact prices, unmodified.
Entries are triggered on the detected levels against those prices.
The stop is anchored on the detected level too.
The only added quantity is level_breakout_buffer_atr, which defaults to 0.0 —
so by default the cross is measured exactly at the detected level,
with the ATR buffer available only to demand a more decisive break.
On top of this, the base filters + selects, but does not alter the levels.

Three things happen, none of which change a level's price:
1. Family selection:
Keeps resistance + support (optionally pullback).
Chooses which detected levels to include.
2. Causal time-gate:
Returns only active levels: detected levels that are already confirmed and not yet invalidated as of bar i.
This selects which detected levels are visible at each bar; it doesn't move them.
3. Geometric classification:
Each visible detected level is bucketed as "overhead" or "underlying" (prev_close > lvl) by its position vs the prior close.
"""

from __future__ import annotations

import pandas as pd

from ..core import PositionState
from ..indicators import atr
from ..levels import detect_levels
from ..strategy_configurator import LevelParams
from .base import BaseStrategy


class LevelStrategyBase(BaseStrategy):
    """Base for strategies built on :mod:`engine.levels`."""

    # Detector families used as the horizontal S/R set (pullback added by config).
    _BASE_FAMILIES = ("resistance", "support")

    def __init__(self, config: LevelParams, exit_policy=None):
        super().__init__(config, exit_policy)
        # One tuple per detector level: (confirmation_idx, invalidated_at|None, price)
        self._levels: list[tuple[int, int | None, float]] = []

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config
        df["atr"] = atr(df, cfg.level_atr_period)

        # Dispatch on cfg.level_detector (pivot_level / cluster_level / touch_level);
        # every source returns the same families dict of causal Level records, with
        # cfg's detector-specific knobs mapped inside the source adapter.
        families = detect_levels(df, cfg)

        chosen = list(self._BASE_FAMILIES)
        if cfg.level_use_pullback:
            chosen.append("pullback")

        # (confirmation_idx, invalidated_at|None, price) per level. The source sets
        # confirmed_idx causally (pivot_level: start_idx + pivot_window; cluster /
        # touch: the bar the level became observable).
        self._levels = [
            (lvl.confirmed_idx, lvl.invalidated_at, float(lvl.price))
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
