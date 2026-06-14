"""Tests for the pluggable level-detector package (engine.levels).

Covers the seam shared by the three detectors (pivot_level / cluster_level /
touch_level): the registry/dispatcher, the causal Level contract every source
must honour, the LevelParams.level_detector selector + namespaced knob
validation, and that level_breakout trades under every detector and obeys the
runtime causality enforcement.

pivot_level's own detection is pinned separately in test_level_detector.py; the
per-detector golden trade snapshots live in test_golden.py.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.backtester import Backtester
from engine.levels import (
    LEVEL_SOURCE_NAMES,
    LEVEL_SOURCES,
    detect_levels,
    level_source_for,
    _validate_level_sources,
)
from engine.levels.base import FAMILIES
from engine.strategies import InverseLevelBreakoutStrategy, LevelBreakoutStrategy
from engine.strategy_configurator import LevelParams, params_for


def _ohlcv(n: int = 800, seed: int = 20240601) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.8 + 0.02) + 8.0 * np.sin(t / 50.0)
    high = close + np.abs(rng.randn(n)) * 1.2 + 0.3
    low = close - np.abs(rng.randn(n)) * 1.2 - 0.3
    openp = close + rng.randn(n) * 0.5
    vol = 10.0 + np.abs(rng.randn(n)) * 5.0
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close,
         "volume": vol, "turnover": vol * close},
        index=idx,
    )


def _trade_sig(res):
    return [
        (str(t.entry_ts), round(float(t.entry_price), 8),
         str(t.exit_ts), round(float(t.pnl_bps), 8),
         t.exit_reason.value if t.exit_reason else None)
        for t in res.trades
    ]


# ── Registry / dispatcher ───────────────────────────────────────────────────


class TestRegistry:
    def test_names_match_registry(self):
        assert set(LEVEL_SOURCE_NAMES) == set(LEVEL_SOURCES)
        _validate_level_sources()  # must not raise

    def test_expected_three_detectors(self):
        assert set(LEVEL_SOURCE_NAMES) == {"pivot_level", "cluster_level", "touch_level"}

    def test_level_source_for_unknown_raises(self):
        with pytest.raises(ValueError):
            level_source_for("not_a_detector")

    def test_detect_levels_dispatches_on_config(self):
        df = _ohlcv()
        for name in LEVEL_SOURCE_NAMES:
            cfg = dataclasses.replace(params_for("level_breakout"), level_detector=name)
            fam = detect_levels(df, cfg)
            assert set(fam) == set(FAMILIES)             # every source returns all families
            assert any(fam[k] for k in fam)              # and finds at least something


# ── The causal Level contract every source must honour ──────────────────────


class TestCausalContract:
    @pytest.mark.parametrize("name", LEVEL_SOURCE_NAMES)
    def test_levels_are_causal_and_well_formed(self, name):
        df = _ohlcv()
        cfg = dataclasses.replace(params_for("level_breakout"), level_detector=name)
        fam = detect_levels(df, cfg)
        for levels in fam.values():
            for lv in levels:
                assert lv.confirmed_idx is not None
                assert lv.confirmed_idx >= lv.start_idx          # confirmation never precedes the seed
                assert 0 <= lv.start_idx < len(df)
                if lv.invalidated_at is not None:
                    assert lv.invalidated_at >= lv.confirmed_idx  # dies no earlier than it is born
                    assert lv.invalidated_at < len(df)

    def test_pullback_only_on_pivot_level(self):
        df = _ohlcv()
        base = params_for("level_breakout")
        assert detect_levels(df, dataclasses.replace(base, level_detector="pivot_level"))["pullback"]
        # the two-sided detectors model resistance/support only
        assert detect_levels(df, dataclasses.replace(base, level_detector="cluster_level"))["pullback"] == []
        assert detect_levels(df, dataclasses.replace(base, level_detector="touch_level"))["pullback"] == []


# ── LevelParams selector + namespaced knob validation ───────────────────────


class TestLevelParamsValidation:
    @pytest.mark.parametrize("name", LEVEL_SOURCE_NAMES)
    def test_each_detector_name_is_valid(self, name):
        assert dataclasses.replace(LevelParams(), level_detector=name).level_detector == name

    @pytest.mark.parametrize("factory", [
        lambda: LevelParams(level_detector="nope"),          # not a registered detector
        lambda: LevelParams(cluster_max_levels=0),           # must be positive
        lambda: LevelParams(cluster_merge_atr_mult=-1.0),    # non-negative
        lambda: LevelParams(touch_min_touches=0),            # must be positive
        lambda: LevelParams(touch_recency_bars=-1),          # non-negative
        lambda: LevelParams(touch_band_mult=-0.1),           # non-negative
    ])
    def test_bad_values_raise(self, factory):
        with pytest.raises(ValueError):
            factory()

    @pytest.mark.parametrize("factory", [
        lambda: LevelParams(touch_recency_bars=0),           # 0 = keep all
        lambda: LevelParams(cluster_break_atr_mult=0.0),     # break on any close beyond
        lambda: LevelParams(touch_min_touches=1),            # single-touch significance
    ])
    def test_valid_edges_pass(self, factory):
        factory()  # must not raise


# ── level_breakout trades under each detector and obeys runtime causality ────


class TestStrategiesUnderEachDetector:
    @pytest.mark.parametrize("name", LEVEL_SOURCE_NAMES)
    @pytest.mark.parametrize("strat_name,cls", [
        ("level_breakout", LevelBreakoutStrategy),
        ("level_breakout_inv", InverseLevelBreakoutStrategy),
    ])
    def test_runs_and_is_causal(self, name, strat_name, cls):
        df = _ohlcv()
        cfg = dataclasses.replace(params_for(strat_name), level_detector=name)
        enforced = Backtester(cls(cfg)).run(df, interval="15")
        unenforced = Backtester(cls(cfg)).run(df, interval="15", enforce_causality=False)
        # truncated-view run must match full-view run — no on_bar peeks ahead.
        assert _trade_sig(enforced) == _trade_sig(unenforced)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
