"""Tests for the ML scaffolding (labels, features, splits, strategy)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 12 * np.pi, n)
    mid = 100.0 + 8.0 * np.sin(t)
    close = mid + rng.standard_normal(n) * 0.4
    high = close + np.abs(rng.standard_normal(n)) * 0.5
    low = close - np.abs(rng.standard_normal(n)) * 0.5
    open_ = close + rng.standard_normal(n) * 0.1
    vol = rng.uniform(100, 1000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


class TestOracleLabels:
    def test_label_values(self):
        from engine.ml.labels import (
            LABEL_HOLD, LABEL_LONG, LABEL_SHORT, oracle_swing_labels,
        )

        df = _make_ohlcv(400)
        labels = oracle_swing_labels(df, min_prominence_atr=1.0)
        assert set(labels.unique()).issubset({LABEL_HOLD, LABEL_LONG, LABEL_SHORT})

    def test_pivots_alternate(self):
        from engine.ml.labels import (
            LABEL_HOLD, LABEL_LONG, LABEL_SHORT, oracle_swing_labels,
        )

        df = _make_ohlcv(400)
        labels = oracle_swing_labels(df, min_prominence_atr=1.0)
        pivot_labels = [int(v) for v in labels[labels != LABEL_HOLD].tolist()]
        for a, b in zip(pivot_labels, pivot_labels[1:]):
            assert a != b, "consecutive oracle pivots must alternate long/short"
        assert len(pivot_labels) >= 4

    def test_threshold_calibrates_pivot_count(self):
        from engine.ml.labels import calibrate_threshold, oracle_swing_labels

        df = _make_ohlcv(800)
        target = 20
        thr = calibrate_threshold(df, target_pivots=target)
        labels = oracle_swing_labels(df, min_prominence_atr=thr)
        pivots = int((labels != 0).sum())
        assert abs(pivots - target) <= max(2, target // 4)


class TestFeatures:
    def test_schema_matches_columns_constant(self):
        from engine.ml.features import FEATURE_COLUMNS, build_feature_frame

        df = _make_ohlcv(400)
        feats = build_feature_frame(df)
        assert list(feats.columns) == list(FEATURE_COLUMNS)
        assert len(feats) == len(df)

    def test_warmup_is_nan_then_finite(self):
        from engine.ml.features import build_feature_frame

        df = _make_ohlcv(600)
        feats = build_feature_frame(df)
        # Tail must be fully finite (no leakage of NaN past warmup).
        assert feats.iloc[-50:].notna().all().all()

    def test_no_lookahead(self):
        """Features at bar i must equal features built on df[:i+1].iloc[i]."""
        from engine.ml.features import build_feature_frame

        df = _make_ohlcv(500)
        feats_full = build_feature_frame(df)
        for i in (250, 350, 450):
            feats_partial = build_feature_frame(df.iloc[: i + 1])
            row_full = feats_full.iloc[i]
            row_partial = feats_partial.iloc[i]
            both_finite = row_full.notna() & row_partial.notna()
            assert both_finite.any(), f"bar {i}: no comparable features"
            diff = (row_full[both_finite] - row_partial[both_finite]).abs().max()
            assert diff < 1e-9, f"bar {i}: lookahead leak — max diff {diff}"


class TestPurgedKFold:
    def test_no_train_test_overlap(self):
        from engine.ml.splits import PurgedKFold

        n = 500
        embargo = 10
        splitter = PurgedKFold(n_splits=5, embargo=embargo)
        for train_idx, test_idx in splitter.split(n):
            assert not set(train_idx).intersection(test_idx)
            # Embargo respected: no train index within `embargo` of test bounds.
            test_lo, test_hi = test_idx.min(), test_idx.max()
            for ti in train_idx:
                assert ti < test_lo - embargo or ti > test_hi + embargo

    def test_test_folds_cover_all_samples(self):
        from engine.ml.splits import PurgedKFold

        n = 500
        splitter = PurgedKFold(n_splits=5, embargo=0)
        covered = np.zeros(n, dtype=bool)
        for _, test_idx in splitter.split(n):
            covered[test_idx] = True
        assert covered.all()

    def test_walk_forward_chronological(self):
        from engine.ml.splits import walk_forward_iter

        splits = list(walk_forward_iter(n_samples=1000, train_size=300, test_size=100, embargo=10))
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min()
            gap = test_idx.min() - train_idx.max() - 1
            assert gap >= 10


class TestMLStrategySmoke:
    def _train_dummy_model(self, df: pd.DataFrame) -> Path:
        """Train a tiny LR on oracle labels and return the saved model path."""
        from sklearn.linear_model import LogisticRegression

        from engine.ml.features import FEATURE_COLUMNS, build_feature_frame
        from engine.ml.labels import (
            LABEL_HOLD, LABEL_LONG, LABEL_SHORT, oracle_swing_labels,
        )

        feats = build_feature_frame(df)
        labels = oracle_swing_labels(df, min_prominence_atr=1.0)
        mask = feats.notna().all(axis=1)
        X = feats.loc[mask].to_numpy()
        y = labels.loc[mask].to_numpy()

        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(X, y)

        bundle = {
            "model": clf,
            "classes": clf.classes_.tolist(),
            "features": tuple(FEATURE_COLUMNS),
        }
        tmp = Path(tempfile.mkstemp(suffix=".joblib")[1])
        joblib.dump(bundle, tmp)
        return tmp

    def test_strategy_loads_and_runs_backtest(self):
        from engine.backtester import Backtester
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import MLSwingZigZagStrategy

        df = _make_ohlcv(600)
        model_path = self._train_dummy_model(df)
        try:
            cfg = StrategyConfig(ml_model_path=str(model_path), ml_p_threshold=0.4)
            strat = MLSwingZigZagStrategy(cfg)
            result = Backtester(strat, symbol="BTCUSDT").run(df, interval="15")
            assert result.num_bars == 600
            assert result.strategy_name == "swing_zigzag_ml"
            assert all(t.is_closed for t in result.trades)
        finally:
            model_path.unlink(missing_ok=True)

    def test_missing_model_raises(self):
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import MLSwingZigZagStrategy

        cfg = StrategyConfig(ml_model_path="ml_models/does_not_exist.joblib")
        with pytest.raises(FileNotFoundError):
            MLSwingZigZagStrategy(cfg)


class TestOrderFlow:
    def _make_synthetic_trades(self, n: int = 5000, seed: int = 7) -> pd.DataFrame:
        """Synthetic tick-level trades over one UTC day. Buy/sell mix is balanced
        with a slight buy lean (so OFI imbalance is non-zero)."""
        rng = np.random.default_rng(seed)
        start = pd.Timestamp("2025-06-15", tz="UTC")
        # Random arrival times spread over the day
        offsets = np.sort(rng.uniform(0, 86_400, n))
        ts = start + pd.to_timedelta(offsets, unit="s")
        sides = np.where(rng.random(n) < 0.52, "Buy", "Sell")
        sizes = rng.lognormal(mean=-3.0, sigma=1.5, size=n)
        prices = 60_000.0 + rng.normal(0, 200, n).cumsum() * 0.01
        notional = sizes * prices
        return pd.DataFrame(
            {"side": sides, "size": sizes, "price": prices, "notional_usd": notional},
            index=ts,
        )

    def test_aggregate_to_bars_schema(self):
        from engine.ml.order_flow import OFI_BAR_COLUMNS, aggregate_to_bars

        trades = self._make_synthetic_trades(n=3000)
        bars = aggregate_to_bars(trades, interval="15min")
        assert list(bars.columns) == list(OFI_BAR_COLUMNS)
        assert len(bars) > 0

    def test_aggregate_imbalance_signs(self):
        from engine.ml.order_flow import aggregate_to_bars

        trades = self._make_synthetic_trades(n=5000, seed=1)
        bars = aggregate_to_bars(trades, interval="15min")
        # Buy-leaning synthetic mix → cumulative buy volume > cumulative sell.
        finite = bars["ofi_imbalance"].dropna()
        assert finite.between(-1.0, 1.0).all()
        assert bars["ofi_buy_volume_usd"].sum() > bars["ofi_sell_volume_usd"].sum(), (
            "52% buy-aggressor lean must produce more cumulative buy volume"
        )

    def test_aggregate_volume_conservation(self):
        from engine.ml.order_flow import aggregate_to_bars

        trades = self._make_synthetic_trades(n=4000)
        bars = aggregate_to_bars(trades, interval="15min")
        # Sum of buy + sell ≈ sum of total (modulo zero-side rows).
        s = (bars["ofi_buy_volume_usd"] + bars["ofi_sell_volume_usd"]).sum()
        t = bars["ofi_total_volume_usd"].sum()
        assert abs(s - t) / max(t, 1.0) < 1e-9

    def test_compute_derived_columns(self):
        from engine.ml.order_flow import (
            OFI_FEATURE_COLUMNS, aggregate_to_bars, compute_derived_orderflow,
        )

        trades = self._make_synthetic_trades(n=4000)
        bars = aggregate_to_bars(trades, interval="15min")
        ofi = compute_derived_orderflow(bars)
        assert list(ofi.columns) == list(OFI_FEATURE_COLUMNS)
        # Derived columns finite past the warmup
        tail = ofi.iloc[20:]
        assert tail.notna().all().all()

    def test_merge_alignment_and_gap_fill(self):
        """OFI feature merge: kline bars without trades get neutral fills, not NaN."""
        from engine.ml.order_flow import (
            OFI_FEATURE_COLUMNS, aggregate_to_bars, compute_derived_orderflow,
            merge_orderflow_features,
        )

        trades = self._make_synthetic_trades(n=3000)
        bars = aggregate_to_bars(trades, interval="15min")
        ofi = compute_derived_orderflow(bars)

        # Klines: 24h grid including hours with no trades
        kline_idx = pd.date_range(
            "2025-06-15", periods=96, freq="15min", tz="UTC",
        )
        klines = pd.DataFrame(
            {"open": 60_000.0, "high": 60_100.0, "low": 59_900.0,
             "close": 60_000.0, "volume": 1.0},
            index=kline_idx,
        )
        merged = merge_orderflow_features(klines, ofi)
        assert list(merged.columns) == list(OFI_FEATURE_COLUMNS)
        assert merged.notna().all().all(), "merge must leave no NaNs in OFI columns"

    def test_t3_strategy_loads_and_runs(self):
        """End-to-end: train a tiny model on T3 features, save it, load via
        MLSwingZigZagStrategy, run the Backtester with pre-merged OFI columns."""
        import tempfile
        from pathlib import Path

        import joblib as _joblib
        from sklearn.linear_model import LogisticRegression

        from engine.backtester import Backtester
        from engine.ml.features import FEATURE_COLUMNS_T3, build_feature_frame_t3
        from engine.ml.labels import oracle_swing_labels
        from engine.ml.order_flow import (
            OFI_FEATURE_COLUMNS, aggregate_to_bars, compute_derived_orderflow,
            merge_orderflow_features,
        )
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import MLSwingZigZagStrategy

        # 1) Synthetic klines (4 days, 15m) + synthetic trades over the same window.
        n = 96 * 4
        idx = pd.date_range("2025-06-15", periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(0)
        close = 60_000.0 + rng.standard_normal(n).cumsum() * 30
        klines = pd.DataFrame({
            "open": close, "high": close + 30, "low": close - 30,
            "close": close, "volume": rng.uniform(50, 500, n),
        }, index=idx)
        trades = self._make_synthetic_trades(n=8000)
        # Spread trades over 4 days
        offsets = np.linspace(0, 4 * 86_400, len(trades))
        trades.index = pd.Timestamp("2025-06-15", tz="UTC") + pd.to_timedelta(offsets, unit="s")

        bars = aggregate_to_bars(trades, interval="15min")
        ofi = compute_derived_orderflow(bars)
        ofi_merged = merge_orderflow_features(klines, ofi)
        df_with_ofi = pd.concat([klines, ofi_merged], axis=1)

        feats = build_feature_frame_t3(klines, ofi)
        labels = oracle_swing_labels(klines, min_prominence_atr=1.0)
        mask = feats.notna().all(axis=1)
        X = feats.loc[mask].to_numpy()
        y = labels.loc[mask].to_numpy()

        clf = LogisticRegression(max_iter=400, class_weight="balanced")
        clf.fit(X, y)
        bundle = {
            "model": clf,
            "classes": clf.classes_.tolist(),
            "features": tuple(FEATURE_COLUMNS_T3),
        }
        tmp = Path(tempfile.mkstemp(suffix=".joblib")[1])
        try:
            _joblib.dump(bundle, tmp)
            cfg = StrategyConfig(ml_model_path=str(tmp), ml_p_threshold=0.4)
            strat = MLSwingZigZagStrategy(cfg)
            # df_with_ofi already has the OFI columns pre-merged.
            result = Backtester(strat, symbol="BTCUSDT").run(df_with_ofi, interval="15")
            assert result.strategy_name == "swing_zigzag_ml"
            assert all(t.is_closed for t in result.trades)
        finally:
            tmp.unlink(missing_ok=True)

    def test_t3_strategy_without_ofi_columns_errors(self):
        """Loading a T3 model and running on a df that lacks OFI columns must error."""
        import tempfile
        from pathlib import Path

        import joblib as _joblib
        from sklearn.linear_model import LogisticRegression

        from engine.backtester import Backtester
        from engine.ml.features import FEATURE_COLUMNS_T3
        from engine.strategy_configurator import StrategyConfig
        from engine.strategies import MLSwingZigZagStrategy

        # Use a real (tiny) fitted estimator so joblib can pickle it.
        n_feat = len(FEATURE_COLUMNS_T3)
        rng = np.random.default_rng(0)
        X_dummy = rng.standard_normal((300, n_feat))
        y_dummy = rng.choice([-1, 0, 1], size=300)
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(X_dummy, y_dummy)

        bundle = {
            "model": clf, "classes": clf.classes_.tolist(),
            "features": tuple(FEATURE_COLUMNS_T3),
        }
        tmp = Path(tempfile.mkstemp(suffix=".joblib")[1])
        try:
            _joblib.dump(bundle, tmp)
            cfg = StrategyConfig(ml_model_path=str(tmp), ml_p_threshold=0.4)
            strat = MLSwingZigZagStrategy(cfg)
            df = _make_ohlcv(300)
            with pytest.raises(ValueError, match="order-flow"):
                Backtester(strat, symbol="BTCUSDT").run(df, interval="15")
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
