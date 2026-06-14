"""Tests for engine.levels.pivot_level — the pivot-seeded support/resistance/
pullback detector (formerly engine.level_detector). Absolute mode is a
byte-for-byte port of the `ema` project's levels.py; the percent/atr modes are
tradekit additions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.levels.pivot_level import (
    Level,
    detect_all_levels,
    detect_resistance_levels,
    detect_support_levels,
    _pivot_high_mask,
    _pivot_low_mask,
)


def _levels_df(n: int = 800) -> pd.DataFrame:
    """Deterministic drift + cycle + noise OHLC frame."""
    rng = np.random.RandomState(20240601)
    t = np.arange(n)
    close = 30000.0 + np.cumsum(rng.randn(n) * 6 + 0.05) + 500 * np.sin(t / 45.0)
    high = close + np.abs(rng.randn(n)) * 12 + 4
    low = close - np.abs(rng.randn(n)) * 12 - 4
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": 10 + np.abs(rng.randn(n)) * 5, "turnover": close * 10},
        index=idx,
    )


def test_detect_all_keys_and_counts():
    """Regression pin on the deterministic fixture (absolute mode)."""
    lv = detect_all_levels(_levels_df())
    assert set(lv) == {"resistance", "support", "pullback"}
    assert {k: len(v) for k, v in lv.items()} == {
        "resistance": 166, "support": 187, "pullback": 352,
    }


def test_level_price_anchors_and_lifecycle():
    df = _levels_df()
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    lv = detect_all_levels(df)

    # Resistance is seeded at the pivot LOW; support at the pivot HIGH.
    for r in lv["resistance"]:
        assert r.price == float(lows[r.start_idx])
    for s in lv["support"]:
        assert s.price == float(highs[s.start_idx])

    # Lifecycle: invalidation never precedes the seed, cross_count is non-negative.
    for kind in lv.values():
        for level in kind:
            assert level.cross_count >= 0
            if level.invalidated_at is not None:
                assert level.invalidated_at >= level.start_idx


def test_lookahead_free_confirmation_lag():
    """A pivot at i is only seeded at i + pivot_window, so no level can start in
    the first `pivot_window` bars and start_idx must be a real pivot."""
    df = _levels_df()
    for w in (1, 3):
        res = detect_resistance_levels(df, pivot_window=w)
        assert all(r.start_idx >= w for r in res)


def test_delta_modes_run_and_differ():
    df = _levels_df()
    absolute = detect_support_levels(df, delta=15.0, delta_mode="absolute")
    percent = detect_support_levels(df, delta=0.05, delta_mode="percent")
    atr = detect_support_levels(df, delta=0.5, delta_mode="atr", atr_period=14)
    # All three seed the same pivots (seeding is independent of the tolerance),
    # so level counts match; the tolerance only changes invalidation timing.
    assert len(absolute) == len(percent) == len(atr)
    # Tolerance mode changes *when* levels invalidate, so the invalidation
    # timestamps are not all identical across modes.
    inval = lambda ls: [l.invalidated_at for l in ls]
    assert not (inval(absolute) == inval(percent) == inval(atr))


def test_bad_delta_mode_raises():
    with pytest.raises(ValueError):
        detect_support_levels(_levels_df(50), delta_mode="nonsense")


def test_empty_and_tiny_frames():
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": []})
    assert detect_all_levels(empty) == {"resistance": [], "support": [], "pullback": []}
    # Fewer bars than a full pivot neighbourhood → no pivots, no levels.
    tiny = _levels_df(2)
    assert detect_resistance_levels(tiny, pivot_window=3) == []


def test_pivot_masks_strict_and_symmetric():
    # A clean single peak/trough at index 3 (window=1): strict pivots.
    highs = np.array([1.0, 2.0, 3.0, 9.0, 3.0, 2.0, 1.0])
    lows = np.array([9.0, 8.0, 7.0, 1.0, 7.0, 8.0, 9.0])
    assert _pivot_high_mask(highs, 1).tolist() == [False, False, False, True, False, False, False]
    assert _pivot_low_mask(lows, 1).tolist() == [False, False, False, True, False, False, False]
    # Plateau (equal neighbours) is not a strict pivot.
    flat = np.array([1.0, 5.0, 5.0, 1.0, 1.0])
    assert not _pivot_high_mask(flat, 1).any()


def test_level_dataclass_defaults():
    lv = Level(price=100.0, start_idx=5)
    assert lv.cross_count == 0 and lv.invalidated_at is None
