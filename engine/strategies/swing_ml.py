"""ML swing-pivot strategy — imitation learning on oracle swings.

A trained scikit-learn classifier replaces the geometric ATR-prominence
detector. At each bar the model emits ``P(hold)``, ``P(flip-long)``, and
``P(flip-short)``. The strategy enters / flips when ``max(P_long, P_short)``
clears ``ml_p_threshold``, and exits on:

- An opposite-side prediction above threshold (signal flip), or
- An ATR trailing stop (optional, ``ml_use_stop``).

The model artifact is a dict produced by the training notebook
(``strategy_notebooks/swing_ml.ipynb``) and serialized with joblib:

    {
        "model":   sklearn estimator with .predict_proba,
        "classes": [LABEL_HOLD, LABEL_LONG, LABEL_SHORT] in the model's order,
        "features": tuple of feature column names (must match build_feature_frame),
    }

If the model file is missing, the strategy raises immediately at construction
WHEN ``require_model=True`` (the default) — there is no silent fallback. With
``require_model=False`` the model is never loaded; ``prepare()`` instead runs on
pre-injected ``ml_p_long`` / ``ml_p_short`` / ``ml_valid`` columns (used to
backtest out-of-sample probabilities from the research notebooks), raising a
``ValueError`` only if those columns are absent.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..ml.features import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_T3,
    build_feature_frame,
    build_feature_frame_t3,
)
from ..ml.labels import LABEL_HOLD, LABEL_LONG, LABEL_SHORT
from ..ml.order_flow import OFI_FEATURE_COLUMNS
from ..core import Direction, ExitReason, PositionState
from ..exits import ExitPolicy
from ..strategy_configurator import SwingMlParams
from ..swing_detector import wilder_atr
from .base import BaseStrategy


def _resolve_model_path(path_str: str) -> Path:
    """Resolve model paths relative to the project root, not the CWD.

    Live mode runs from the repo root; notebooks run from
    strategy_notebooks/. Always anchor to the package's grandparent dir.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / p


class SwingMLStrategy(BaseStrategy):
    name = "swing_ml"

    def __init__(
        self,
        config: SwingMlParams,
        exit_policy: ExitPolicy | None = None,
        *,
        require_model: bool = True,
    ):
        super().__init__(config, exit_policy)
        self._model_path = _resolve_model_path(config.ml_model_path)
        self._model = None
        self._classes: list[int] = []
        self._trained_features: tuple[str, ...] = ()
        self._schema: str | None = None
        self._idx_long = self._idx_short = -1
        # require_model=False builds the strategy for backtesting on externally
        # supplied probabilities (e.g. cross-validated out-of-sample probs from a
        # research notebook): prepare() then passes through pre-injected
        # ml_p_long / ml_p_short / ml_valid columns instead of running inference.
        if require_model:
            self._load_model()

    def _load_model(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"ML model not found at {self._model_path}. "
                "Train it with strategy_notebooks/swing_ml.ipynb."
            )
        bundle = joblib.load(self._model_path)
        self._model = bundle["model"]
        self._classes = list(bundle["classes"])
        self._trained_features = tuple(bundle.get("features", FEATURE_COLUMNS))
        if self._trained_features == tuple(FEATURE_COLUMNS):
            self._schema = "t1"
        elif self._trained_features == tuple(FEATURE_COLUMNS_T3):
            self._schema = "t3"
        else:
            raise ValueError(
                f"Unknown trained feature schema with {len(self._trained_features)} columns. "
                "Expected T1 (OHLCV only) or T3 (OHLCV + order flow). "
                "Re-train the model after changing the feature set."
            )
        self._idx_long = self._classes.index(LABEL_LONG)
        self._idx_short = self._classes.index(LABEL_SHORT)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Pre-injected probabilities (e.g. out-of-sample probs supplied by a
        # research notebook) bypass model inference: on_bar reads ml_p_long /
        # ml_p_short / ml_valid the same way whether the model or a notebook
        # produced them. ml_atr (the trailing-stop input) is computed if absent.
        if {"ml_p_long", "ml_p_short", "ml_valid"}.issubset(df.columns):
            if "ml_atr" not in df.columns:
                df["ml_atr"] = wilder_atr(df, self.config.ml_atr_period)
            return df

        if self._model is None:
            raise ValueError(
                "SwingMLStrategy was built with require_model=False, so "
                "prepare() needs pre-injected ml_p_long / ml_p_short / ml_valid "
                "columns (supply out-of-sample probabilities, or rebuild with "
                "require_model=True to run the model)."
            )

        if self._schema == "t3":
            missing = [c for c in OFI_FEATURE_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"T3 strategy needs order-flow columns pre-merged into df; "
                    f"missing: {missing[:3]}... ({len(missing)} cols). "
                    "Build them via engine.ml.order_flow.build_orderflow_features() "
                    "and merge into df before passing to Backtester."
                )
            ofi_df = df[list(OFI_FEATURE_COLUMNS)]
            feats = build_feature_frame_t3(df.drop(columns=list(OFI_FEATURE_COLUMNS)), ofi_df)
        else:
            feats = build_feature_frame(df)

        df["ml_atr"] = wilder_atr(df, self.config.ml_atr_period)

        valid_mask = feats.notna().all(axis=1)
        p_long = np.zeros(len(df), dtype=np.float64)
        p_short = np.zeros(len(df), dtype=np.float64)

        if valid_mask.any():
            X = feats.loc[valid_mask].to_numpy()
            proba = self._model.predict_proba(X)
            p_long[valid_mask.to_numpy()] = proba[:, self._idx_long]
            p_short[valid_mask.to_numpy()] = proba[:, self._idx_short]

        df["ml_p_long"] = p_long
        df["ml_p_short"] = p_short
        df["ml_valid"] = valid_mask.astype(np.int8)
        return df

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if not df["ml_valid"].iloc[i]:
            return

        p_long = float(df["ml_p_long"].iloc[i])
        p_short = float(df["ml_p_short"].iloc[i])
        threshold = self.config.ml_p_threshold

        close_i = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]
        atr_val = df["ml_atr"].iloc[i]
        ts = df.index[i]

        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Exit logic ────────────────────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade

            # Trailing stop (optional) delegated to the exit policy; gated by
            # ml_use_stop + a valid ATR, no return (flip-through entry may follow).
            if (
                self.config.ml_use_stop
                and np.isfinite(atr_val)
                and atr_val > 0
            ):
                decision = self.exit_policy.evaluate(self._exit_ctx(i, df, trade, atr_val))
                if decision is not None:
                    state.exit(ts, decision.price, decision.reason)

            # Opposite-side signal above threshold → flip (exit, then re-enter).
            if state.current_trade is not None:
                trade = state.current_trade
                flip_long = trade.direction == Direction.LONG and p_short >= threshold
                flip_short = trade.direction == Direction.SHORT and p_long >= threshold
                if flip_long or flip_short:
                    state.exit(ts, close_i, ExitReason.SIGNAL_FLIP)

        # ── Entry logic ──────────────────────────────────────────────────────
        if state.current_trade is None:
            if p_long >= threshold and p_long > p_short:
                state.enter(Direction.LONG, ts, close_i)
            elif p_short >= threshold and p_short > p_long:
                state.enter(Direction.SHORT, ts, close_i)
