"""pivot_level — pivot-seeded horizontal levels with invalidation tracking.

The project's original level detector (formerly ``engine/level_detector.py``).
Detects three families of horizontal price levels from OHLC candles. Each family
is seeded at a strict, symmetric ``pivot_window``-bar pivot, then tracked forward
until *invalidated*:

- Resistance — seeded at minimum lows (the pivot's low becomes the level);
  invalidated when a later candle's high comes within the tolerance, or the level
  is bracketed (low <= level <= high) by ``invalidation_candles`` candles.
- Support — seeded at maximum highs; invalidated on a later candle's low within
  tolerance, or bracketing.
- Pullback — seeded at minimum highs OR maximum lows; an inside bar seeds two.
  Invalidated like support (tested on the candle low).

A pivot at index ``i`` is only confirmable at iteration ``i + pivot_window`` (its
right-hand neighbourhood must be in), so detection is look-ahead free:
``start_idx`` points at the visual pivot bar, ``confirmed_idx`` is the bar the
level becomes observable (``start_idx + pivot_window``), and invalidation
tracking begins only at confirmation.

This is one of three peer detectors behind the :data:`engine.levels.LevelSource`
contract (see :mod:`engine.levels.base`); the others are :mod:`cluster_level` and
:mod:`touch_level`. The plotting helper lives in :mod:`engine.visualization`.

Tolerance modes (``delta_mode``):

- ``"absolute"`` *(default)* — a fixed quote-currency distance (symbol-specific).
- ``"percent"`` — ``delta`` percent of the level price (cross-symbol).
- ``"atr"`` — ``delta`` × Wilder ATR at the test bar (volatility-relative).

Public API: :func:`detect_resistance_levels`, :func:`detect_support_levels`,
:func:`detect_pullback_levels`, :func:`detect_all_levels`, :func:`pivot_level_source`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..swing_detector import wilder_atr
from .base import Level

_DELTA_MODES = ("absolute", "percent", "atr")


# ── Pivot masks (vectorized, O(N)) ──────────────────────────────────────────


def _pivot_high_mask(values: np.ndarray, window: int) -> np.ndarray:
    """True where ``values[i]`` is a strict pivot high over a symmetric window:
    greater than every value in ``[i-window, i-1]`` and ``[i+1, i+window]``. The
    first/last ``window`` indices are always False."""
    if window < 1:
        raise ValueError("pivot_window must be >= 1")
    n = len(values)
    if n < 2 * window + 1:
        return np.zeros(n, dtype=bool)
    s = pd.Series(values)
    left_max = s.rolling(window).max().shift(1)
    right_max = s.rolling(window).max().shift(-window)
    mask = (s > left_max) & (s > right_max)
    return mask.fillna(False).to_numpy()


def _pivot_low_mask(values: np.ndarray, window: int) -> np.ndarray:
    """Mirror of :func:`_pivot_high_mask` — strict pivot low (min, ``<``)."""
    if window < 1:
        raise ValueError("pivot_window must be >= 1")
    n = len(values)
    if n < 2 * window + 1:
        return np.zeros(n, dtype=bool)
    s = pd.Series(values)
    left_min = s.rolling(window).min().shift(1)
    right_min = s.rolling(window).min().shift(-window)
    mask = (s < left_min) & (s < right_min)
    return mask.fillna(False).to_numpy()


# ── Invalidation scan ───────────────────────────────────────────────────────


def _tolerance(level_price: float, delta: float, delta_mode: str, atr_i: Optional[float]) -> float:
    if delta_mode == "absolute":
        return delta
    if delta_mode == "percent":
        return level_price * (delta / 100.0)
    return delta * atr_i  # "atr"


def _scan_active(
    active: list[Level],
    i: int,
    candle_low: float,
    candle_high: float,
    touch_price: float,
    delta: float,
    invalidation_candles: int,
    delta_mode: str,
    atr_i: Optional[float],
) -> list[Level]:
    """Update invalidation state for each active level at candle ``i`` and return
    the still-active ones. ``touch_price`` is the OHLC value tested against the
    level (candle high for resistance, candle low for support/pullback). Returns a
    fresh list so the caller only re-scans live levels next candle."""
    surviving: list[Level] = []
    for lvl in active:
        tol = _tolerance(lvl.price, delta, delta_mode, atr_i)
        if abs(touch_price - lvl.price) <= tol:
            lvl.invalidated_at = i
            continue
        if candle_low <= lvl.price <= candle_high:
            lvl.cross_count += 1
            if lvl.cross_count >= invalidation_candles:
                lvl.invalidated_at = i
                continue
        surviving.append(lvl)
    return surviving


def _atr_array(df: pd.DataFrame, delta_mode: str, atr_period: int) -> Optional[np.ndarray]:
    if delta_mode not in _DELTA_MODES:
        raise ValueError(f"delta_mode must be one of {_DELTA_MODES}; got {delta_mode!r}")
    return wilder_atr(df, atr_period) if delta_mode == "atr" else None


# ── Detectors ───────────────────────────────────────────────────────────────


def detect_resistance_levels(
    df: pd.DataFrame,
    pivot_window: int = 1,
    delta: float = 5.0,
    invalidation_candles: int = 3,
    delta_mode: str = "absolute",
    atr_period: int = 14,
) -> list[Level]:
    """Resistance seeded at **minimum lows**; invalidated when a later candle's
    high comes within tolerance, or it is bracketed by ``invalidation_candles``."""
    levels: list[Level] = []
    n = len(df)
    if n == 0:
        return levels
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    pivots = _pivot_low_mask(lows, pivot_window)
    atr = _atr_array(df, delta_mode, atr_period)

    active: list[Level] = []
    for j in range(n):
        active = _scan_active(
            active, j, lows[j], highs[j], touch_price=highs[j],
            delta=delta, invalidation_candles=invalidation_candles,
            delta_mode=delta_mode, atr_i=(atr[j] if atr is not None else None),
        )
        confirm_idx = j - pivot_window
        if confirm_idx >= pivot_window and pivots[confirm_idx]:
            new = Level(price=float(lows[confirm_idx]), start_idx=confirm_idx, confirmed_idx=j)
            levels.append(new)
            active.append(new)
    return levels


def detect_support_levels(
    df: pd.DataFrame,
    pivot_window: int = 1,
    delta: float = 15.0,
    invalidation_candles: int = 3,
    delta_mode: str = "absolute",
    atr_period: int = 14,
) -> list[Level]:
    """Support seeded at **maximum highs**; invalidated when a later candle's low
    comes within tolerance, or it is bracketed by ``invalidation_candles``."""
    levels: list[Level] = []
    n = len(df)
    if n == 0:
        return levels
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    pivots = _pivot_high_mask(highs, pivot_window)
    atr = _atr_array(df, delta_mode, atr_period)

    active: list[Level] = []
    for j in range(n):
        active = _scan_active(
            active, j, lows[j], highs[j], touch_price=lows[j],
            delta=delta, invalidation_candles=invalidation_candles,
            delta_mode=delta_mode, atr_i=(atr[j] if atr is not None else None),
        )
        confirm_idx = j - pivot_window
        if confirm_idx >= pivot_window and pivots[confirm_idx]:
            new = Level(price=float(highs[confirm_idx]), start_idx=confirm_idx, confirmed_idx=j)
            levels.append(new)
            active.append(new)
    return levels


def detect_pullback_levels(
    df: pd.DataFrame,
    pivot_window: int = 1,
    delta: float = 25.0,
    invalidation_candles: int = 10,
    delta_mode: str = "absolute",
    atr_period: int = 14,
) -> list[Level]:
    """Pullback seeded at **minimum highs** (price = high) OR **maximum lows**
    (price = low) — an inside bar seeds two. Invalidated like support (tested on
    the candle low)."""
    levels: list[Level] = []
    n = len(df)
    if n == 0:
        return levels
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    min_high_pivots = _pivot_low_mask(highs, pivot_window)   # minimum high
    max_low_pivots = _pivot_high_mask(lows, pivot_window)    # maximum low
    atr = _atr_array(df, delta_mode, atr_period)

    active: list[Level] = []
    for j in range(n):
        active = _scan_active(
            active, j, lows[j], highs[j], touch_price=lows[j],
            delta=delta, invalidation_candles=invalidation_candles,
            delta_mode=delta_mode, atr_i=(atr[j] if atr is not None else None),
        )
        confirm_idx = j - pivot_window
        if confirm_idx >= pivot_window:
            if min_high_pivots[confirm_idx]:
                new = Level(price=float(highs[confirm_idx]), start_idx=confirm_idx, confirmed_idx=j)
                levels.append(new)
                active.append(new)
            if max_low_pivots[confirm_idx]:
                new = Level(price=float(lows[confirm_idx]), start_idx=confirm_idx, confirmed_idx=j)
                levels.append(new)
                active.append(new)
    return levels


def detect_all_levels(
    df: pd.DataFrame,
    delta_resistance: float = 5.0,
    delta_support: float = 15.0,
    delta_pullback: float = 25.0,
    inval_resistance: int = 3,
    inval_support: int = 3,
    inval_pullback: int = 10,
    pivot_window_resistance: int = 1,
    pivot_window_support: int = 1,
    pivot_window_pullback: int = 1,
    delta_mode: str = "absolute",
    atr_period: int = 14,
) -> dict[str, list[Level]]:
    """Run all three detectors with per-family params. ``delta_mode`` /
    ``atr_period`` apply to all three (absolute points, percent of level, or
    ×ATR)."""
    return {
        "resistance": detect_resistance_levels(
            df, pivot_window_resistance, delta_resistance, inval_resistance, delta_mode, atr_period,
        ),
        "support": detect_support_levels(
            df, pivot_window_support, delta_support, inval_support, delta_mode, atr_period,
        ),
        "pullback": detect_pullback_levels(
            df, pivot_window_pullback, delta_pullback, inval_pullback, delta_mode, atr_period,
        ),
    }


# ── LevelSource adapter ─────────────────────────────────────────────────────


def pivot_level_source(df: pd.DataFrame, cfg) -> dict[str, list[Level]]:
    """The :data:`engine.levels.LevelSource` adapter for ``pivot_level``: maps the
    shared :class:`~engine.strategy_configurator.LevelParams` knobs onto
    :func:`detect_all_levels`. Returns all three families; the strategy base picks
    which to use (``level_use_pullback``)."""
    return detect_all_levels(
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
