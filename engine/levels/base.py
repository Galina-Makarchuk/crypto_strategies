"""Normalized level record + the shared contract for the level detectors.

This is the seam the three horizontal-level detectors live behind
(:mod:`pivot_level`, :mod:`cluster_level`, :mod:`touch_level`). A *level source*
is a pure function ``(DataFrame, LevelParams) -> dict[str, list[Level]]`` that
returns the resistance / support / pullback families of horizontal levels,
detected causally from OHLC candles.

Every :class:`Level` a source emits carries enough to be (a) gated causally by a
strategy and (b) drawn on a chart:

* ``confirmed_idx`` — first bar at which the level is observable / tradeable (its
  right neighbourhood or activation condition is satisfied using data ``<=`` this
  bar). This is the look-ahead anchor strategies gate on.
* ``invalidated_at`` — bar at which the level died (break / touch / recency), or
  ``None`` if it is still alive at the end of the data.
* ``start_idx`` — the visual pivot / seed bar, used as the left draw anchor.

:class:`~engine.strategies.level_base.LevelStrategyBase` exposes only levels that
are confirmed by, and not yet invalidated as of, the current bar; ``plot_levels``
draws each from ``start_idx`` to ``invalidated_at`` (or the chart edge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle — only needed for the type alias below
    import pandas as pd

    from ..strategy_configurator import LevelParams

# The families every source returns. ``pullback`` may be empty for a detector
# that only models two-sided S/R (cluster_level, touch_level).
FAMILIES = ("resistance", "support", "pullback")


@dataclass
class Level:
    """A single horizontal price level with a causal lifecycle.

    ``confirmed_idx`` defaults to ``start_idx`` (a level with no explicit
    confirmation lag is observable at its seed bar); the pivot detectors set the
    real confirmation bar (``start_idx + pivot_window``) explicitly.
    """

    price: float
    start_idx: int                        # visual pivot / seed bar (left draw anchor)
    cross_count: int = 0                  # candles whose [low, high] contained the level
    invalidated_at: Optional[int] = None  # bar where the level died, or None if alive
    confirmed_idx: Optional[int] = None   # first bar the level is observable / tradeable
    strength: float = 1.0                 # touch / merge weight (1.0 when not tracked)

    def __post_init__(self) -> None:
        if self.confirmed_idx is None:
            self.confirmed_idx = self.start_idx


# A level source: (candles, LevelParams) -> {family: [Level, ...]}.
LevelSource = Callable[["pd.DataFrame", "LevelParams"], "dict[str, list[Level]]"]


def tolerance(level_price: float, magnitude: float, delta_mode: str,
              atr_i: Optional[float]) -> float:
    """Resolve a tolerance band in price units from a magnitude + mode.

    Mirrors the pivot detector's tolerance semantics so all sources interpret
    ``delta_mode`` identically:

    * ``"absolute"`` — ``magnitude`` quote-currency points.
    * ``"percent"``  — ``magnitude`` percent of the level price.
    * ``"atr"``      — ``magnitude`` × ATR at the test bar.
    """
    if delta_mode == "absolute":
        return magnitude
    if delta_mode == "percent":
        return level_price * (magnitude / 100.0)
    return magnitude * (atr_i if atr_i is not None else 0.0)  # "atr"
