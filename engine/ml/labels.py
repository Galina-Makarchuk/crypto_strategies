"""Oracle swing labels for supervised learning.

The labels come from running the swing detector *with hindsight* — on the
full df, with no causal constraint. Each bar gets one of three labels:

    LABEL_HOLD  =  0    (no swing pivot here)
    LABEL_LONG  = +1    (this bar is a swing low — fade the dip = go long)
    LABEL_SHORT = -1    (this bar is a swing high — fade the bounce = go short)

The classifier learns to predict these from *causal* features. At inference
time the classifier is causal — labels are oracle-only and never reach the
strategy at runtime.

Why use the hindsight detector instead of a true count-constrained DP?

- The detector already encodes the "alternating ATR-prominent pivot" structure
  we want the classifier to imitate. Running it without causality gives an
  unbiased estimator of where the true pivots are.
- The DP at n = 70K is O(n²) ≈ 5×10⁹ ops — slow when recomputed per CV fold.
- Empirically, in-sample swings from a tight-threshold detector match what
  the count-constrained DP picks for K up to a few hundred pivots/year.

To match a target trade frequency K, sweep ``min_prominence_atr`` until the
detector emits the right number of pivots. Helper ``calibrate_threshold``
does this with binary search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..swings import detect_swings

LABEL_HOLD: int = 0
LABEL_LONG: int = 1
LABEL_SHORT: int = -1


def oracle_swing_labels(
    df: pd.DataFrame,
    min_prominence_atr: float = 1.5,
    atr_period: int = 14,
    min_bars_between: int = 3,
) -> pd.Series:
    """Generate oracle swing labels for every bar in ``df``.

    Important: this function uses the entire ``df`` to identify pivots — it is
    intentionally NOT causal, since we want the *target* (not the features) to
    encode the truth the model is trying to imitate. Always partition df into
    train / test BEFORE calling this; do not generate labels on the full
    history and then split, or you leak test-period structure into the train
    labels via the detector's stateful confirmation logic.
    """
    swings = detect_swings(
        df,
        atr_period=atr_period,
        min_prominence_atr=min_prominence_atr,
        min_bars_between=min_bars_between,
        return_provisional=False,
    )
    labels = np.full(len(df), LABEL_HOLD, dtype=np.int8)
    for s in swings:
        labels[s.idx] = LABEL_LONG if s.side == "low" else LABEL_SHORT
    return pd.Series(labels, index=df.index, name="label")


def calibrate_threshold(
    df: pd.DataFrame,
    target_pivots: int,
    atr_period: int = 14,
    min_bars_between: int = 3,
    low: float = 0.3,
    high: float = 6.0,
    max_iter: int = 30,
    tolerance: float = 0.05,
) -> float:
    """Binary-search ``min_prominence_atr`` so the detector emits ~target_pivots.

    Pivot count is monotone-decreasing in the threshold, so bisection
    converges fast. Returns a threshold whose pivot count is within
    ``tolerance`` (fraction) of the target, or the best found after
    ``max_iter`` iterations.
    """
    target_pivots = max(2, int(target_pivots))
    best_thr = (low + high) / 2.0
    best_diff = float("inf")
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        swings = detect_swings(
            df,
            atr_period=atr_period,
            min_prominence_atr=mid,
            min_bars_between=min_bars_between,
            return_provisional=False,
        )
        count = len(swings)
        diff = abs(count - target_pivots) / target_pivots
        if diff < best_diff:
            best_diff = diff
            best_thr = mid
        if diff <= tolerance:
            return mid
        # More pivots than wanted → raise the bar (suppress small swings).
        if count > target_pivots:
            low = mid
        else:
            high = mid
    return best_thr


def label_distribution(labels: pd.Series) -> dict[str, int]:
    """Sanity-check counter for the 3-class label distribution."""
    counts = labels.value_counts().to_dict()
    return {
        "hold": int(counts.get(LABEL_HOLD, 0)),
        "long": int(counts.get(LABEL_LONG, 0)),
        "short": int(counts.get(LABEL_SHORT, 0)),
    }
