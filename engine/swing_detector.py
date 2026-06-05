"""ATR-prominence ZigZag swing detection.

Confirms swing highs and lows as the running max/min between confirmations.
A candidate is committed when its retracement (measured in ATR units at the
candidate's bar) exceeds ``min_prominence_atr`` AND the candidate's pivot is
at least ``min_bars_between`` bars after the previously confirmed pivot.

Key properties vs strict N-bar pivots:

- Volatility-normalized — threshold is in ATR units, not points or percent,
  so the same parameters work across symbols and regimes.
- Graded — each confirmed swing carries a continuous ``prominence_atr`` plus
  volume and range context, combined into a ``score``.
- Adaptive lag — a decisive reversal confirms in 1 bar; a slow rollover takes
  as many bars as it needs.
- Direction-alternating — emits a strict high → low → high → low sequence
  (except a possible trailing provisional swing at end-of-data).
- Multi-scale via ``detect_swings_tiered`` — same algorithm at multiple
  prominence thresholds, returned as independent series.

ATR convention here is Wilder's EWMA (``tr.ewm(alpha=1/period,
adjust=False).mean()``), which differs from the SMA-based ``indicators.atr``.
Keeping a local implementation preserves this module's original semantics so
swing positions don't shift when the project's default ATR formula changes.

Causality: at bar i the detector only reads bars 0..i, so the strategy that
consumes it (via swing.confirmation_idx) inherits look-ahead-free behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Swing:
    idx: int                              # bar index of the pivot
    side: Literal["high", "low"]
    price: float
    confirmation_idx: int                 # bar where this swing was confirmed
    prominence_atr: float                 # retrace at confirmation / ATR at pivot
    bars_to_confirm: int                  # confirmation_idx - idx
    volume_z: float                       # rolling z-score of volume at pivot
    range_z: float                        # (high-low at pivot)/ATR(pivot) - 1
    score: float                          # composite quality score
    provisional: bool = False             # True for an unconfirmed candidate at EOD


# ── ATR + scoring helpers ─────────────────────────────────────────────────────


def wilder_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder's ATR via EWMA of TR (alpha = 1/period, adjust=False).

    Returns a numpy array of length len(df); the very first entry can be NaN
    since TR is undefined on bar 0 (no prior close).
    """
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    if len(df) == 0:
        return np.array([], dtype=float)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift()
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()


def _zscore(values: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(values)
    mean = s.rolling(window, min_periods=1).mean()
    std = s.rolling(window, min_periods=1).std(ddof=0).replace(0.0, np.nan)
    z = (s - mean) / std
    return z.fillna(0.0).clip(-5.0, 5.0).to_numpy()


def _composite_score(prominence_atr: float, volume_z: float, range_z: float) -> float:
    # Asymmetric clipping: negative z-scores don't penalize, positive z-scores
    # boost up to a saturation cap.
    v_boost = 1.0 + 0.3 * max(0.0, min(volume_z, 3.0))
    r_boost = 1.0 + 0.2 * max(0.0, min(range_z, 3.0))
    return prominence_atr * v_boost * r_boost


# ── Core detector ─────────────────────────────────────────────────────────────


def detect_swings(
    df: pd.DataFrame,
    atr_period: int = 14,
    min_prominence_atr: float = 1.5,
    min_bars_between: int = 3,
    vol_lookback: int = 50,
    return_provisional: bool = True,
) -> list[Swing]:
    """ATR-prominence ZigZag swing detection.

    Args:
        df: DataFrame with columns ``high``, ``low``, ``close``, ``volume``,
            sorted ascending by time.
        atr_period: Wilder ATR period.
        min_prominence_atr: required retrace (in ATR units, measured at the
            pivot bar) for a candidate to be confirmed. Higher = fewer, larger
            swings.
        min_bars_between: minimum bars between consecutive *confirmation*
            events. Suppresses bar-by-bar oscillation while still allowing
            close pivots when justified. (Gating on pivot distance instead
            would lock out the algorithm if the only candidate sits inside
            the forbidden window — the cooldown formulation avoids that.)
        vol_lookback: rolling window for the volume z-score input to the
            composite score.
        return_provisional: if True and there is an open candidate at end of
            data, emit it with ``provisional=True``.

    Returns:
        List of Swings sorted ascending by ``idx``. Confirmed swings strictly
        alternate side. A trailing provisional swing (if any) is opposite to
        the last confirmed swing.
    """
    if df.empty:
        return []
    for col in ("high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"df must contain '{col}' column")
    if min_prominence_atr <= 0:
        raise ValueError(f"min_prominence_atr must be > 0; got {min_prominence_atr}")
    if min_bars_between < 1:
        raise ValueError(f"min_bars_between must be >= 1; got {min_bars_between}")

    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    atr = wilder_atr(df, atr_period)
    volume_z = _zscore(df["volume"].to_numpy(dtype=float), vol_lookback)

    start = atr_period
    if n <= start + 1:
        return []

    cand_high_idx = start
    cand_high_price = high[start]
    cand_low_idx = start
    cand_low_price = low[start]

    last_confirm_bar = -10**9
    expect: Literal["either", "high", "low"] = "either"

    swings: list[Swing] = []

    def _emit(side: str, p_idx: int, p_price: float, conf_idx: int, retrace: float) -> Swing:
        atr_p = atr[p_idx]
        if not np.isfinite(atr_p) or atr_p <= 0:
            prom = 0.0
            range_z = 0.0
        else:
            prom = retrace / atr_p
            range_z = (high[p_idx] - low[p_idx]) / atr_p - 1.0
        v_z = float(volume_z[p_idx])
        return Swing(
            idx=int(p_idx),
            side=side,                      # type: ignore[arg-type]
            price=float(p_price),
            confirmation_idx=int(conf_idx),
            prominence_atr=float(prom),
            bars_to_confirm=int(conf_idx - p_idx),
            volume_z=v_z,
            range_z=float(range_z),
            score=float(_composite_score(prom, v_z, range_z)),
            provisional=False,
        )

    for i in range(start + 1, n):
        # Step 1: update running candidates with the new bar
        if high[i] > cand_high_price:
            cand_high_price = high[i]
            cand_high_idx = i
        if low[i] < cand_low_price:
            cand_low_price = low[i]
            cand_low_idx = i

        # Step 2: confirmation cooldown
        if i - last_confirm_bar < min_bars_between:
            continue

        # Step 3: check for confirmations against the current bar's opposing wick
        confirm_high = None
        if expect in ("either", "high"):
            atr_p = atr[cand_high_idx]
            if np.isfinite(atr_p) and atr_p > 0:
                retrace = cand_high_price - low[i]
                if retrace >= min_prominence_atr * atr_p:
                    confirm_high = (retrace, cand_high_idx, cand_high_price)

        confirm_low = None
        if expect in ("either", "low"):
            atr_p = atr[cand_low_idx]
            if np.isfinite(atr_p) and atr_p > 0:
                retrace = high[i] - cand_low_price
                if retrace >= min_prominence_atr * atr_p:
                    confirm_low = (retrace, cand_low_idx, cand_low_price)

        # Bootstrap tie: commit the earlier-in-time pivot first.
        if confirm_high is not None and confirm_low is not None:
            if confirm_high[1] <= confirm_low[1]:
                confirm_low = None
            else:
                confirm_high = None

        if confirm_high is not None:
            retrace, p_idx, p_price = confirm_high
            swings.append(_emit("high", p_idx, p_price, i, retrace))
            last_confirm_bar = i
            # Reseed cand_low to the min over (p_idx, i].
            if i > p_idx:
                slc = low[p_idx + 1 : i + 1]
                rel = int(np.argmin(slc))
                cand_low_idx = p_idx + 1 + rel
                cand_low_price = float(slc[rel])
            else:
                cand_low_idx = i
                cand_low_price = float("inf")
            cand_high_idx = i
            cand_high_price = float("-inf")
            expect = "low"

        elif confirm_low is not None:
            retrace, p_idx, p_price = confirm_low
            swings.append(_emit("low", p_idx, p_price, i, retrace))
            last_confirm_bar = i
            if i > p_idx:
                slc = high[p_idx + 1 : i + 1]
                rel = int(np.argmax(slc))
                cand_high_idx = p_idx + 1 + rel
                cand_high_price = float(slc[rel])
            else:
                cand_high_idx = i
                cand_high_price = float("-inf")
            cand_low_idx = i
            cand_low_price = float("inf")
            expect = "high"

    # End-of-data: emit a provisional swing for the open opposite-side candidate.
    if return_provisional and swings:
        last = swings[-1]
        last_bar = n - 1
        if last.side == "high" and cand_low_idx > last.idx:
            atr_p = atr[cand_low_idx]
            if np.isfinite(atr_p) and atr_p > 0:
                retrace = high[last_bar] - cand_low_price
                prov = _emit("low", cand_low_idx, cand_low_price, last_bar, retrace)
                prov.provisional = True
                swings.append(prov)
        elif last.side == "low" and cand_high_idx > last.idx:
            atr_p = atr[cand_high_idx]
            if np.isfinite(atr_p) and atr_p > 0:
                retrace = cand_high_price - low[last_bar]
                prov = _emit("high", cand_high_idx, cand_high_price, last_bar, retrace)
                prov.provisional = True
                swings.append(prov)

    return swings


def detect_swings_tiered(
    df: pd.DataFrame,
    tiers: Union[list[float], tuple[float, ...]] = (0.8, 1.5, 2.5),
    atr_period: int = 14,
    min_bars_between: int = 3,
    vol_lookback: int = 50,
    return_provisional: bool = True,
) -> dict[float, list[Swing]]:
    """Run :func:`detect_swings` at multiple prominence thresholds.

    Each tier is an independent run — the state machine takes different paths
    at different thresholds, so a swing in one tier may not appear in another.
    Returned dict is keyed by threshold (float) and ordered as given.
    """
    if not tiers:
        raise ValueError("tiers must be non-empty")
    return {
        float(t): detect_swings(
            df,
            atr_period=atr_period,
            min_prominence_atr=float(t),
            min_bars_between=min_bars_between,
            vol_lookback=vol_lookback,
            return_provisional=return_provisional,
        )
        for t in tiers
    }
