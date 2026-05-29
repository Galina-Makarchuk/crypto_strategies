"""Causal feature transformers for the swing-pivot classifier.

Every column in the returned frame is read-safe at bar i — only data from
bars 0..i is used. The causality test in ``tests/test_ml.py`` verifies this
by comparing features built on df[:i+1] to features built on the full df.

Design choices:

- All features are vectorized (numpy / pandas) — no Python loops over bars.
- NaNs are emitted for the warmup window (the first few hundred bars). The
  trainer drops them or imputes; the strategy at runtime skips bars where
  any feature is NaN.
- Higher-TF features (1h trend / slope) use only the LAST CLOSED 1h bar at
  each 15m timestamp — never the in-progress bar. This is the single most
  common lookahead bug; the test explicitly guards against it.
- The complete feature list is exposed as ``FEATURE_COLUMNS`` so the strategy
  and trainer agree on the column order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import rsi
from ..swings import wilder_atr
from .order_flow import OFI_FEATURE_COLUMNS, merge_orderflow_features

# ── Feature configuration ─────────────────────────────────────────────────────

RETURN_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20, 50, 100)
VOL_WINDOWS: tuple[int, ...] = (10, 50, 200)
VOLUME_WINDOW: int = 50
ATR_PERIOD: int = 14
RSI_PERIOD: int = 14
ADX_PERIOD: int = 14
HTF_RESAMPLE: str = "1h"
HTF_EMA: int = 20
HTF_SLOPE_LOOKBACK: int = 3

FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"ret_{k}" for k in RETURN_HORIZONS),
    *(f"ret_{k}_atr" for k in RETURN_HORIZONS),
    *(f"vol_{w}" for w in VOL_WINDOWS),
    "volume_z",
    "volume_log_diff",
    "rsi",
    "rsi_slope",
    "adx",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_expansion",
    "atr_z",
    "htf_ema_dist",
    "htf_ema_slope",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

# Tier 3 schema = Tier 1 OHLCV features + order-flow features from Bybit
# bulk trade dumps. See ml/order_flow.py for how OFI columns are built.
FEATURE_COLUMNS_T3: tuple[str, ...] = FEATURE_COLUMNS + OFI_FEATURE_COLUMNS


def _rolling_z(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=max(2, window // 4)).mean()
    std = s.rolling(window, min_periods=max(2, window // 4)).std(ddof=0)
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.clip(-6.0, 6.0)


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder-smoothed ADX (range-bound on 0..100)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift()

    plus_dm = (high.diff()).clip(lower=0.0)
    minus_dm = (-low.diff()).clip(lower=0.0)
    # When down-move > up-move, +DM = 0 (and vice versa).
    mask_plus = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask_plus, 0.0)
    minus_dm = minus_dm.where(~mask_plus, 0.0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1.0 / period
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_s.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_s.replace(0.0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan) * 100.0
    return dx.ewm(alpha=alpha, adjust=False).mean()


def _htf_trend(df: pd.DataFrame, rule: str, ema: int, slope_lookback: int) -> pd.DataFrame:
    """Resample to a higher TF, compute EMA + slope, then merge BACK as-of
    onto the original index using only CLOSED higher-TF bars.

    The "only closed" guarantee comes from shifting the resampled series by
    one bucket before merging — at any 15m bar inside the current 1h bucket,
    we read the previous 1h bucket's value, never the still-forming one.
    """
    htf = df["close"].resample(rule, label="right", closed="right").last().dropna()
    htf_ema = htf.ewm(span=ema, adjust=False).mean()
    htf_slope = (htf_ema - htf_ema.shift(slope_lookback)) / htf_ema.shift(slope_lookback)
    # Shift by one bucket → use the most recently CLOSED HTF bar.
    htf_ema = htf_ema.shift(1)
    htf_slope = htf_slope.shift(1)

    out = pd.DataFrame({"htf_ema": htf_ema, "htf_slope": htf_slope})
    return out.reindex(df.index, method="ffill")


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the causal feature frame. Returns a frame with `FEATURE_COLUMNS`
    in order, aligned to ``df.index``.

    The first ~200 rows contain NaNs (warmup). The trainer drops them; the
    live strategy skips any bar where any feature is NaN.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("build_feature_frame requires a DatetimeIndex")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"df is missing '{col}' column")

    feats = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    vol = df["volume"]

    # ── Returns + ATR-normalized returns ─────────────────────────────────────
    atr = pd.Series(wilder_atr(df, ATR_PERIOD), index=df.index, name="atr")
    for k in RETURN_HORIZONS:
        ret = np.log(close).diff(k)
        feats[f"ret_{k}"] = ret
        feats[f"ret_{k}_atr"] = ret * close / atr.replace(0.0, np.nan)

    # ── Realized vol ─────────────────────────────────────────────────────────
    log_ret = np.log(close).diff()
    for w in VOL_WINDOWS:
        feats[f"vol_{w}"] = log_ret.rolling(w, min_periods=max(2, w // 4)).std(ddof=0)

    # ── Volume features ──────────────────────────────────────────────────────
    feats["volume_z"] = _rolling_z(vol, VOLUME_WINDOW)
    log_vol = np.log(vol.replace(0.0, np.nan))
    feats["volume_log_diff"] = log_vol.diff()

    # ── Momentum & range shape ───────────────────────────────────────────────
    rsi_s = rsi(close, period=RSI_PERIOD)
    feats["rsi"] = rsi_s
    feats["rsi_slope"] = rsi_s.diff(3)
    feats["adx"] = _adx(df, ADX_PERIOD)

    bar_range = (high - low).replace(0.0, np.nan)
    feats["body_ratio"] = (close - open_) / bar_range
    feats["upper_wick_ratio"] = (high - close.combine(open_, max)) / bar_range
    feats["lower_wick_ratio"] = (close.combine(open_, min) - low) / bar_range

    avg_range = bar_range.rolling(20, min_periods=5).mean()
    feats["range_expansion"] = bar_range / avg_range.replace(0.0, np.nan)
    feats["atr_z"] = _rolling_z(atr, 100)

    # ── Higher-TF bias (closed bars only) ────────────────────────────────────
    htf = _htf_trend(df, HTF_RESAMPLE, HTF_EMA, HTF_SLOPE_LOOKBACK)
    feats["htf_ema_dist"] = (close - htf["htf_ema"]) / htf["htf_ema"]
    feats["htf_ema_slope"] = htf["htf_slope"]

    # ── Cyclic time encodings ────────────────────────────────────────────────
    idx = df.index
    feats["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    return feats[list(FEATURE_COLUMNS)]


def build_feature_frame_t3(
    df: pd.DataFrame,
    ofi: pd.DataFrame,
) -> pd.DataFrame:
    """Tier 3: Tier 1 OHLCV features + order-flow features from bulk trades.

    Args:
        df: OHLCV frame (same contract as :func:`build_feature_frame`).
        ofi: order-flow feature frame from
            :func:`engine.ml.order_flow.build_orderflow_features`,
            indexed on the same bar grid as ``df``.

    Returns:
        DataFrame with all columns in :data:`FEATURE_COLUMNS_T3`, aligned to
        ``df.index``. OFI columns are reindexed and gap-filled
        (no-flow bars get neutral zeros — see ``merge_orderflow_features``).
    """
    base = build_feature_frame(df)
    aligned_ofi = merge_orderflow_features(df, ofi)
    out = pd.concat([base, aligned_ofi[list(OFI_FEATURE_COLUMNS)]], axis=1)
    return out[list(FEATURE_COLUMNS_T3)]
