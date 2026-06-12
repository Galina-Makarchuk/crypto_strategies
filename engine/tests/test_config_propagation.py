"""Tests that pin parameter validation + propagation across the config split.

These cover the gaps the configurators previously left open:

  * StrategyConfig now validates its fields at construction (Gap 1).
  * The CLI seeds its DataSpec from ACTIVE, like it does TradingConfig (Gap 2).
  * The exit-policy catalog is validated at import; exit_policy_for never
    silently picks a default for a real strategy (Gap 4).
  * The no-look-ahead contract is *enforced* (not just documented) via
    Backtester.run(audit_lookahead=True), checked against every strategy (Gap 5).
  * --exit-preset and sizing_mode overrides actually change behaviour (Gap 6).

Run with: pytest engine/tests/test_config_propagation.py -v
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest

from engine import cli
from engine.backtester import Backtester
from engine.core import StrategyName
from engine.data_configurator import ACTIVE, DataSpec
from engine.strategy_configurator import (
    DEFAULT_EXIT,
    EXIT_PRESETS,
    PER_STRATEGY_EXIT,
    StrategyConfig,
    exit_policy_for,
)
from engine import strategy_configurator as sc
from engine.trade_configurator import SizingMode, TradingConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100 + rng.randn(n).cumsum()
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    opens = close + rng.randn(n) * 0.5
    vol = rng.uniform(100, 1000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": close,
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


# Every strategy except swing_ml (needs a trained model + its own pickle cache).
_RUNNABLE = [s.value for s in StrategyName if s.value != "swing_ml"]


# ── Gap 1: StrategyConfig validation ───────────────────────────────────────────


class TestStrategyConfigValidation:
    def test_defaults_are_valid(self):
        StrategyConfig()  # must not raise

    @pytest.mark.parametrize("kw", [
        # Only genuinely-invalid values are rejected: negatives, zero where a
        # value must be positive, bad categoricals, out-of-range indices, empty
        # required strings, and a probability outside [0, 1].
        {"atr_period": 0},
        {"atr_period": -5},
        {"ema_fast": 0},
        {"rsi_period": -1},
        {"supertrend_mult": 0},          # strictly positive
        {"supertrend_mult": -1.0},
        {"ema_touch_delta": 0},          # strictly positive
        {"ml_p_threshold": 1.5},         # probability out of [0, 1]
        {"ml_p_threshold": -0.1},
        {"rsi_bullish": -1},             # negative threshold
        {"adx_threshold": -5},           # negative threshold
        {"flag_close_pos_min": -0.1},    # negative ratio
        {"ema_touch_delta_mode": "bogus"},        # not a known mode
        {"level_delta_mode": "nope"},
        {"vwap_entry_band": 99},         # index past the bands tuple
        {"vwap_band_devs": ()},          # empty
        {"ml_model_path": ""},           # empty path
        {"swing_bounce_min_bars_between_trades": -1},     # negative cooldown
    ])
    def test_bad_values_raise(self, kw):
        with pytest.raises(ValueError):
            StrategyConfig(**kw)

    @pytest.mark.parametrize("kw", [
        {"ema_touch_period_long": None},                 # optional, None disables
        {"ema_touch_regime_filter": 200},                # optional, positive int
        {"swing_bounce_min_bars_between_trades": 0},      # zero cooldown allowed
        {"level_breakout_buffer_atr": 0.0},               # non-negative buffer
        {"flag_close_pos_min": 1.0},                      # ratio boundary
        # Free-range magnitudes: small and large values both pass, and there is
        # NO cross-field ordering or upper cap (so sweeps can roam freely).
        {"ema_fast": 1},                                  # tiny period
        {"ema_slow": 500},                                # large period
        {"ema_fast": 30, "ema_slow": 21},                 # fast >= slow allowed
        {"ob_ema_fast": 30, "ob_ema_slow": 20},           # fast >= slow allowed
        {"flag_min_cluster": 5, "flag_max_cluster": 4},   # min > max allowed
        {"rsi_bullish": 200},                             # no upper cap on RSI
        {"adx_threshold": 200},                           # no upper cap on ADX
        {"rsi_bearish": 80, "rsi_bullish": 70},           # bearish > bullish allowed
        {"flag_cluster_body_ratio": 2.5},                 # ratio > 1 allowed
    ])
    def test_valid_edges_pass(self, kw):
        StrategyConfig(**kw)  # must not raise


class TestOverrideMistakes:
    """Pins what happens when a notebook override dict has a mistake — answering
    'silently ignored or announced?' for the three distinct cases."""

    def test_unknown_key_is_announced(self):
        # A typo'd key (e.g. {"ema_fastt": 12}) — dataclasses.replace raises,
        # it is NOT silently dropped.
        with pytest.raises(TypeError):
            dataclasses.replace(StrategyConfig(), ema_fastt=12)

    def test_out_of_range_value_is_announced(self):
        # A bad value (negative / zero where invalid) raises via __post_init__.
        with pytest.raises(ValueError):
            dataclasses.replace(StrategyConfig(), supertrend_period=0)

    def test_valid_but_irrelevant_value_is_silent(self):
        # The one un-announced case: a real field with a valid value that the
        # chosen strategy simply doesn't read (e.g. ema_fast on supertrend). The
        # override lands on the config but has no effect — no error, no warning,
        # because StrategyConfig is one shared dataclass with no per-strategy
        # field map.
        cfg = dataclasses.replace(StrategyConfig(), ema_fast=12)
        assert cfg.ema_fast == 12

    def test_replace_revalidates(self):
        # dataclasses.replace (used by sweep/grid_search/notebooks) re-runs
        # __post_init__, so a bad override is caught — not silently applied.
        with pytest.raises(ValueError):
            dataclasses.replace(StrategyConfig(), atr_period=0)


# ── Gap 2: CLI DataSpec seeds from ACTIVE (mirrors TradingConfig) ───────────────


class TestCLIDataSpec:
    def _ns(self, **over):
        fields = ("symbol", "interval", "category", "candles", "start", "end")
        ns = argparse.Namespace(**{f: cli._UNSET for f in fields})
        for k, v in over.items():
            setattr(ns, k, v)
        return ns

    def test_inherits_active_when_no_flags(self, monkeypatch):
        custom = DataSpec(symbol="ETHUSDT", interval="60", category="inverse",
                          num_candles=1234)
        monkeypatch.setattr(cli, "ACTIVE", custom)
        assert cli._build_data_spec(self._ns()) == custom

    def test_flag_overrides_only_that_field(self, monkeypatch):
        custom = DataSpec(symbol="ETHUSDT", interval="60", num_candles=1234)
        monkeypatch.setattr(cli, "ACTIVE", custom)
        built = cli._build_data_spec(self._ns(symbol="SOLUSDT", candles=500))
        assert built.symbol == "SOLUSDT"           # overridden
        assert built.num_candles == 500            # overridden (candles → num_candles)
        assert built.interval == custom.interval   # inherited, not reset to a default

    def test_module_default_active_is_the_real_active(self):
        # The CLI imports the genuine ACTIVE, so an unflagged run uses it.
        assert cli.ACTIVE is ACTIVE


# ── Gap 4: exit-policy catalog is validated at import ───────────────────────────


class TestExitCatalog:
    def test_real_catalog_passes(self):
        sc._validate_exit_catalog()  # must not raise

    def test_covers_exactly_the_enum(self):
        assert set(PER_STRATEGY_EXIT) == {s.value for s in StrategyName}

    def test_all_values_are_known_presets(self):
        assert all(v in EXIT_PRESETS for v in PER_STRATEGY_EXIT.values())
        assert DEFAULT_EXIT in EXIT_PRESETS

    def test_bad_preset_value_rejected(self, monkeypatch):
        bad = dict(PER_STRATEGY_EXIT, ema="not_a_preset")
        monkeypatch.setattr(sc, "PER_STRATEGY_EXIT", bad)
        with pytest.raises(ValueError):
            sc._validate_exit_catalog()

    def test_missing_strategy_rejected(self, monkeypatch):
        partial = {k: v for k, v in PER_STRATEGY_EXIT.items() if k != "ema"}
        monkeypatch.setattr(sc, "PER_STRATEGY_EXIT", partial)
        with pytest.raises(ValueError):
            sc._validate_exit_catalog()

    def test_unknown_key_rejected(self, monkeypatch):
        extra = dict(PER_STRATEGY_EXIT, not_a_strategy=DEFAULT_EXIT)
        monkeypatch.setattr(sc, "PER_STRATEGY_EXIT", extra)
        with pytest.raises(ValueError):
            sc._validate_exit_catalog()

    def test_unknown_strategy_warns_and_defaults(self, caplog):
        from engine.exits import ChandelierStop
        with caplog.at_level(logging.WARNING, logger="engine.strategy_configurator"):
            policy = exit_policy_for("totally_made_up_prototype")
        assert isinstance(policy, ChandelierStop)          # the DEFAULT_EXIT
        assert any("no PER_STRATEGY_EXIT entry" in r.message for r in caplog.records)

    def test_real_strategy_resolves_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="engine.strategy_configurator"):
            exit_policy_for("ema_touch")
        assert not caplog.records


# ── Gap 5: the no-look-ahead contract is enforced, not just documented ──────────


class TestNoLookahead:
    @pytest.mark.parametrize("name", _RUNNABLE)
    def test_strategy_obeys_causality(self, name):
        df = _ohlcv()
        normal = Backtester(cli._build_strategy(name, StrategyConfig())).run(df, interval="15")
        audited = Backtester(cli._build_strategy(name, StrategyConfig())).run(
            df, interval="15", audit_lookahead=True)
        # audit_lookahead feeds on_bar a view truncated to [0..i]; a strategy
        # that peeked at a future bar would diverge or raise here.
        assert _trade_sig(normal) == _trade_sig(audited)


# ── Gap 6: overrides actually change behaviour, end-to-end ──────────────────────


class TestExitPresetOverride:
    def test_injected_preset_changes_policy_and_trades(self):
        from engine.exits import ChandelierStop

        default = cli._build_strategy("ema_touch", StrategyConfig())
        # ema_touch's assigned preset is fixed_1pct_rr3 (a CompositeExit).
        assert not isinstance(default.exit_policy, ChandelierStop)

        overridden = cli._build_strategy(
            "ema_touch", StrategyConfig(), exit_policy=EXIT_PRESETS["chandelier_2atr"]())
        assert isinstance(overridden.exit_policy, ChandelierStop)

        df = _ohlcv()
        a = Backtester(default).run(df, interval="15")
        b = Backtester(overridden).run(df, interval="15")
        assert _trade_sig(a) != _trade_sig(b)   # the exit policy actually drives trades

    def test_main_wires_exit_preset_flag(self):
        # Mirror what main() does with args.exit_preset, without touching the network.
        exit_policy = EXIT_PRESETS["fixed_2pct_rr3"]()
        strat = cli._build_strategy("supertrend", StrategyConfig(), exit_policy=exit_policy)
        assert strat.exit_policy is exit_policy


class TestSizingModeOverride:
    def test_fixed_vs_risk_changes_equity_not_bps(self):
        # ema_touch supplies an entry stop (fixed_1pct_rr3), so RISK mode actually
        # risk-sizes rather than falling back to fixed-fraction.
        df = _ohlcv()
        fixed = Backtester(
            cli._build_strategy("ema_touch", StrategyConfig()),
            trading_config=TradingConfig(sizing_mode=SizingMode.FIXED),
        ).run(df, interval="15")
        # 50 bps risk against ema_touch's 1% stop sizes each trade to ~half the
        # FIXED 100%-of-equity notional, so the equity curves genuinely diverge
        # (100 bps would coincidentally equal full notional and hide the effect).
        risk = Backtester(
            cli._build_strategy("ema_touch", StrategyConfig()),
            trading_config=TradingConfig(sizing_mode=SizingMode.RISK,
                                         risk_per_trade_bps=50.0),
        ).run(df, interval="15")

        assert fixed.total_trades > 0 and risk.total_trades > 0
        assert risk.risk_sizing_fallbacks == 0          # genuinely risk-sized
        # bps metrics are sizing-invariant …
        assert fixed.total_pnl_bps == pytest.approx(risk.total_pnl_bps)
        # … but the currency equity layer differs.
        assert fixed.final_equity != pytest.approx(risk.final_equity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
