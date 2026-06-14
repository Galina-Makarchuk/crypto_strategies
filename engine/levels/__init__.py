"""Horizontal support/resistance level detectors.

Three interchangeable detectors live here as peers behind one contract
(:data:`engine.levels.base.LevelSource` — a pure
``(DataFrame, LevelParams) -> dict[str, list[Level]]`` function):

* ``pivot_level``   — pivot-seeded, invalidation-tracked (:mod:`pivot_level`).
* ``cluster_level`` — merge-and-break, strength-scored (:mod:`cluster_level`).
* ``touch_level``   — significance by historical touch count (:mod:`touch_level`).

A strategy selects one via the ``level_detector`` field on
:class:`~engine.strategy_configurator.LevelParams`; :func:`detect_levels`
dispatches on it. :data:`LEVEL_SOURCE_NAMES` is the single source of truth for the
selectable keys, imported by ``strategy_configurator`` to validate that field, and
pinned against the registry by :func:`_validate_level_sources` at import.
"""

from __future__ import annotations

import pandas as pd

from .base import FAMILIES, Level, LevelSource, tolerance
from .cluster_level import cluster_level_source
from .pivot_level import (
    detect_all_levels,
    detect_pullback_levels,
    detect_resistance_levels,
    detect_support_levels,
    pivot_level_source,
)
from .touch_level import touch_level_source

# key → LevelSource adapter. The keys are the values of the ``level_detector``
# config field (and the levels.ipynb section names).
LEVEL_SOURCES: dict[str, LevelSource] = {
    "pivot_level": pivot_level_source,
    "cluster_level": cluster_level_source,
    "touch_level": touch_level_source,
}

# Single source of truth for the selectable detector keys.
LEVEL_SOURCE_NAMES = tuple(LEVEL_SOURCES)


def level_source_for(name: str) -> LevelSource:
    """The adapter for a detector key. Raises on an unknown key (the config field
    is validated against :data:`LEVEL_SOURCE_NAMES`, so this is a guard)."""
    try:
        return LEVEL_SOURCES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown level detector {name!r}. Known: {sorted(LEVEL_SOURCES)}"
        ) from exc


def detect_levels(df: pd.DataFrame, cfg) -> dict[str, list[Level]]:
    """Detect levels with the detector selected by ``cfg.level_detector``.

    The single entry point both :class:`~engine.strategies.level_base.\
    LevelStrategyBase` and the levels notebook use, so all three detectors flow
    through one code path and return the same families dict."""
    return level_source_for(cfg.level_detector)(df, cfg)


def _validate_level_sources() -> None:
    """Fail at import if the registry and the selectable-name tuple drift apart."""
    if set(LEVEL_SOURCES) != set(LEVEL_SOURCE_NAMES):
        raise ValueError(
            "LEVEL_SOURCES keys must equal LEVEL_SOURCE_NAMES. "
            f"Registry: {sorted(LEVEL_SOURCES)}; names: {sorted(LEVEL_SOURCE_NAMES)}."
        )


_validate_level_sources()


__all__ = [
    "FAMILIES",
    "Level",
    "LevelSource",
    "tolerance",
    "LEVEL_SOURCES",
    "LEVEL_SOURCE_NAMES",
    "level_source_for",
    "detect_levels",
    "detect_all_levels",
    "detect_resistance_levels",
    "detect_support_levels",
    "detect_pullback_levels",
    "pivot_level_source",
    "cluster_level_source",
    "touch_level_source",
]
