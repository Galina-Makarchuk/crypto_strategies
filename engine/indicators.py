"""Pure indicator functions.

Every function here is (DataFrame | Series, params) → Series.
No side-effects, no DataFrame mutation, no strategy logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── ATR ────────────────────────────────────────────────────────────────────────


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, Wilder's smoothing (the RMA = an EWMA of True Range)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing: an EWMA with alpha = 1/period (com = period - 1 is the
    # equivalent form). adjust=False makes it the classic recursive RMA used
    # across trading platforms, seeded from the first TR value — matching how
    # rsi() smooths above. Strictly causal: bar i depends only on bars 0..i.
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()
    # Plain SMA of True Range (if needed instead of EWMA):
    # return tr.rolling(period, min_periods=1).mean()


# ── VWAP Stdev Bands ───────────────────────────────────────────────────────────


def vwap_stdev_bands(
    df: pd.DataFrame,
    devs: tuple[float, ...] = (1.28, 2.01, 2.51, 3.09, 4.01),
    session: str = "D",
) -> tuple[pd.Series, list[tuple[pd.Series, pd.Series]]]:
    """Session-anchored VWAP with volume-weighted standard-deviation bands.

    Port of the TradingView "VWAP Stdev Bands v2 Mod" indicator. The VWAP and
    its stdev reset at each session boundary (default: UTC day). Both are
    causal — at bar i they only depend on bars 0..i within the current session
    — so look-ahead is structurally impossible.

    Args:
        df: OHLCV frame with a tz-aware DatetimeIndex.
        devs: stdev multipliers, inner-to-outer (e.g. the last entry is the
            furthest band).
        session: pandas floor alias for the session anchor (e.g. "D", "h", "W").

    Returns:
        (vwap, bands) where bands[k] = (upper_k, lower_k) for devs[k].
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("vwap_stdev_bands requires a DatetimeIndex")

    hl2 = (df["high"] + df["low"]) / 2.0
    vol = df["volume"]

    session_id = df.index.floor(session)
    pv_cum = (hl2 * vol).groupby(session_id).cumsum()
    v_cum = vol.groupby(session_id).cumsum()
    pv2_cum = (hl2 * hl2 * vol).groupby(session_id).cumsum()

    safe_v = v_cum.where(v_cum > 0, np.nan)
    vwap = (pv_cum / safe_v).rename("vwap")
    variance = (pv2_cum / safe_v) - vwap * vwap
    stdev = np.sqrt(variance.clip(lower=0).fillna(0.0))

    bands: list[tuple[pd.Series, pd.Series]] = []
    for k, mult in enumerate(devs):
        upper = (vwap + mult * stdev).rename(f"vwap_upper_{k}")
        lower = (vwap - mult * stdev).rename(f"vwap_lower_{k}")
        bands.append((upper, lower))
    return vwap, bands


# ── EMA ────────────────────────────────────────────────────────────────────────

# adjust=False gives the classic recursive EMA used in technical analysis across almost all trading platforms; better for real intraday trading
# adjust=True gives the statistical weighted version which uses normalized weights
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ── RSI (Wilder smoothing) ─────────────────────────────────────────────────────


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # loss == 0 makes rs NaN. That's genuine overbought (RSI 100) only when there
    # were gains; a truly flat window (no gains AND no losses) carries no
    # information and is neutral (50), not maximally overbought.
    out = out.where(~((loss == 0) & (gain > 0)), 100.0)
    return out.fillna(50.0)


# ── SuperTrend ─────────────────────────────────────────────────────────────────


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Canonical SuperTrend with proper band ratcheting.

    Returns:
        (supertrend_line, trend_direction)
        trend_direction: +1 = uptrend, -1 = downtrend
    """
    atr_vals = atr(df, period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()

    upper_basic = hl2 + multiplier * atr_vals
    lower_basic = hl2 - multiplier * atr_vals

    n = len(df)
    final_upper = np.empty(n)
    final_lower = np.empty(n)
    trend = np.ones(n, dtype=np.int8)  # +1 = up, -1 = down
    st_line = np.empty(n)

    final_upper[0] = upper_basic[0]
    final_lower[0] = lower_basic[0]

    first_signal = -1
    for i in range(1, n):
        # Ratchet upper band DOWN (tighter) in downtrend
        if upper_basic[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_basic[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Ratchet lower band UP (tighter) in uptrend
        if lower_basic[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_basic[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Determine trend
        if close[i] > final_upper[i - 1]:
            trend[i] = 1
            if first_signal < 0:
                first_signal = i
        elif close[i] < final_lower[i - 1]:
            trend[i] = -1
            if first_signal < 0:
                first_signal = i
        else:
            trend[i] = trend[i - 1]

    # Before the first band break the trend is genuinely undetermined; the
    # unconditional +1 seed (trend = np.ones) makes that first break read as a
    # +1→-1 flip when the series opens in a downtrend, which the supertrend
    # strategies trade as a phantom signal (audit L4). Backfill the pre-signal
    # bars to the first break's OWN direction so the break establishes the trend
    # instead of flipping it — symmetric with the uptrend-open case, and causal.
    if first_signal > 0:
        trend[:first_signal] = trend[first_signal]

    # Build the SuperTrend line
    for i in range(n):
        st_line[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    idx = df.index
    return pd.Series(st_line, index=idx, name="supertrend"), pd.Series(
        trend, index=idx, name="trend_direction"
    )


# ── Trend strength (ADX) ─────────────────────────────────────────────────────


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index.

    Measures trend strength on a 0–100 scale:
      - Below ~20–25: weak trend (range / chop)
      - Above ~25: strong trend (directional move)
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    # ADX needs 2*period bars of warmup; with n <= period the Wilder seed write
    # (smoothed_tr[period] = …) would index past the array. Such a short frame is
    # entirely warmup, so return all-NaN instead of raising IndexError.
    if n <= period:
        return pd.Series(np.nan, index=df.index, name="adx")

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Wilder smoothing (EMA with alpha = 1/period)
    smoothed_tr = np.zeros(n)
    smoothed_plus = np.zeros(n)
    smoothed_minus = np.zeros(n)

    # Seed with SMA over first `period` bars
    smoothed_tr[period] = np.sum(tr[1 : period + 1])
    smoothed_plus[period] = np.sum(plus_dm[1 : period + 1])
    smoothed_minus[period] = np.sum(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        smoothed_tr[i] = smoothed_tr[i - 1] - smoothed_tr[i - 1] / period + tr[i]
        smoothed_plus[i] = smoothed_plus[i - 1] - smoothed_plus[i - 1] / period + plus_dm[i]
        smoothed_minus[i] = smoothed_minus[i - 1] - smoothed_minus[i - 1] / period + minus_dm[i]

    # +DI / -DI (np.where guards against zero but numpy still warns)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(smoothed_tr > 0, 100 * smoothed_plus / smoothed_tr, 0.0)
        minus_di = np.where(smoothed_tr > 0, 100 * smoothed_minus / smoothed_tr, 0.0)

        # DX
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    # ADX = Wilder-smoothed DX
    adx_arr = np.zeros(n)
    # Seed ADX with SMA of first `period` valid DX values
    start = 2 * period
    if start < n:
        adx_arr[start] = np.mean(dx[period + 1 : start + 1])
        for i in range(start + 1, n):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

    result = pd.Series(adx_arr, index=df.index, name="adx")
    # Zero-fill warmup period → NaN for clarity
    result.iloc[: 2 * period] = np.nan
    return result


# ── Swing-point detection ──────────────────────────────────────────────────────


def detect_swing_highs(
    highs: pd.Series, left: int = 5, right: int = 5
) -> list[tuple[int, float]]:
    """Return list of (bar_index, price) for confirmed swing highs."""
    arr = highs.to_numpy()
    n = len(arr)
    results: list[tuple[int, float]] = []
    for i in range(left, n - right):
        is_pivot = True
        for j in range(1, left + 1):
            if arr[i] <= arr[i - j]:
                is_pivot = False
                break
        if is_pivot:
            for j in range(1, right + 1):
                if arr[i] <= arr[i + j]:
                    is_pivot = False
                    break
        if is_pivot:
            results.append((i, float(arr[i])))
    return results


def detect_swing_lows(
    lows: pd.Series, left: int = 5, right: int = 5
) -> list[tuple[int, float]]:
    """Return list of (bar_index, price) for confirmed swing lows."""
    arr = lows.to_numpy()
    n = len(arr)
    results: list[tuple[int, float]] = []
    for i in range(left, n - right):
        is_pivot = True
        for j in range(1, left + 1):
            if arr[i] >= arr[i - j]:
                is_pivot = False
                break
        if is_pivot:
            for j in range(1, right + 1):
                if arr[i] >= arr[i + j]:
                    is_pivot = False
                    break
        if is_pivot:
            results.append((i, float(arr[i])))
    return results


def merge_price_levels(prices: list[float], tolerance: float = 0.0015) -> list[float]:
    """Merge prices within `tolerance` (relative) into averaged levels."""
    if not prices:
        return []
    sorted_p = sorted(set(prices))
    merged = [sorted_p[0]]
    for p in sorted_p[1:]:
        if abs(p - merged[-1]) / merged[-1] <= tolerance:
            merged[-1] = (merged[-1] + p) / 2
        else:
            merged.append(p)
    return merged
