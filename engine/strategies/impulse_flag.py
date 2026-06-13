"""Impulse + Consolidation ("flag") breakout strategy.

PATTERN
    One directional "impulse" candle (large body + above-average volume,
    closing in the top/bottom third of its range) followed by 2-4 smaller
    "consolidation" candles whose aggregate range sits inside a fraction
    of the impulse range and holds above/below the 50% retrace.

TRACKS
    Track A — Continuation WITH the HTF trend.  Stop-order entry at the
              impulse extreme.  Skipped if price is within
              `flag_level_proximity_pct` of the rolling 24H hi/lo.
    Track B — Failure fade AGAINST a counter-trend impulse.  Stop-order
              entry at the far side of the consolidation.

RISK & EXITS
    ATR-scaled stop buffer (`flag_stop_atr_mult` * ATR), min 1.5:1 R:R
    gate on the averaged target.  T1 at 1R, T2 at max(2R, impulse-range
    measured move).  Since `PositionState` cannot scale out, T1 is used
    only to shift the stop to break-even — full exit occurs on stop
    (possibly BE), on T2, or when the pending order expires unfilled
    after `flag_breakout_window` bars.

NO LOOK-AHEAD
    `prepare()` computes the HTF bias on a resample with an explicit
    `.shift(1)` so only the last CLOSED HTF bar is visible.  Pattern
    detection in `on_bar(i, …)` searches bars ending at `i-1`, and the
    triggered entry fill compares against `df.iloc[i]` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..indicators import atr as calc_atr
from ..indicators import ema as calc_ema
from ..core import Direction, ExitReason, PositionState
from ..strategy_configurator import ImpulseFlagParams
from .base import BaseStrategy


@dataclass
class _Pattern:
    impulse_idx: int
    cluster_end_idx: int
    direction: int  # +1 bullish impulse, -1 bearish
    impulse_high: float
    impulse_low: float
    impulse_range: float
    cluster_high: float
    cluster_low: float


@dataclass
class _Setup:
    track: str       # 'A' or 'B'
    side: int        # +1 long trade, -1 short trade
    trigger: float
    stop: float
    t1: float
    t2: float
    expiry_bar: int  # inclusive last bar eligible to fill


class ImpulseFlagStrategy(BaseStrategy):
    """Impulse + consolidation breakout with HTF bias and two tracks."""

    name = "impulse_flag"

    def __init__(self, config: ImpulseFlagParams, exit_policy=None):
        super().__init__(config, exit_policy)
        self._pending: Optional[_Setup] = None
        self._active_stop: float = 0.0
        self._t1: float = 0.0
        self._t2: float = 0.0
        self._t1_hit: bool = False

    # ── prepare ────────────────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config

        # Interval-aware knobs: the rolling-level window is expressed in hours
        # (TF-independent) and the HTF bucket is clamped to at least 4× LTF so
        # the bias isn't a same-timeframe no-op on coarser intervals.
        ltf_minutes = _infer_ltf_minutes(df, default=5)
        lookback_bars = max(
            1, int(round(60.0 * cfg.flag_level_lookback_hours / ltf_minutes))
        )
        htf_minutes = cfg.flag_htf_minutes
        if htf_minutes <= ltf_minutes:
            htf_minutes = max(60, 4 * ltf_minutes)

        body = (df["close"] - df["open"]).abs()
        df["flag_body"] = body
        df["flag_body_sma"] = body.rolling(cfg.flag_body_sma).mean()
        df["flag_vol_sma"] = df["volume"].rolling(cfg.flag_vol_sma).mean()
        df["flag_ema"] = calc_ema(df["close"], cfg.flag_ema_fast)
        df["atr"] = calc_atr(df, cfg.atr_period)
        df["flag_hi_window"] = df["high"].rolling(lookback_bars, min_periods=1).max()
        df["flag_lo_window"] = df["low"].rolling(lookback_bars, min_periods=1).min()

        # HTF bias — resample, compute EMA + slope, shift(1) so only the last
        # CLOSED HTF bar is ever visible, reindex back with forward-fill.
        rule = f"{htf_minutes}min"
        htf = (
            df[["open", "high", "low", "close", "volume"]]
            .resample(rule, label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        if len(htf) > 0:
            htf_ema = calc_ema(htf["close"], cfg.flag_htf_ema)
            htf_slope = htf_ema.diff(cfg.flag_htf_slope_lookback)
            bias = pd.Series(0, index=htf.index, dtype=int)
            bias[(htf["close"] > htf_ema) & (htf_slope > 0)] = 1
            bias[(htf["close"] < htf_ema) & (htf_slope < 0)] = -1
            bias = bias.shift(1)
            df["flag_htf_bias"] = (
                bias.reindex(df.index, method="ffill").fillna(0).astype(int)
            )
        else:
            df["flag_htf_bias"] = 0

        # Reset runtime state (live engine calls prepare() every tick).
        self._pending = None
        self._reset_trade_state()
        return df

    # ── helpers ────────────────────────────────────────────────────────────

    def _reset_trade_state(self) -> None:
        self._active_stop = 0.0
        self._t1 = 0.0
        self._t2 = 0.0
        self._t1_hit = False

    def _detect_pattern(self, i: int, df: pd.DataFrame) -> Optional[_Pattern]:
        """Search for a valid impulse+cluster pattern ending at bar `i-1`."""
        cfg = self.config
        if i - 1 - cfg.flag_max_cluster < cfg.flag_body_sma:
            return None

        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        v = df["volume"].to_numpy()
        body_sma = df["flag_body_sma"].to_numpy()
        vol_sma = df["flag_vol_sma"].to_numpy()

        # Cluster has `k` bars at (i-k)..(i-1); impulse sits at (i-1-k).
        for k in range(cfg.flag_min_cluster, cfg.flag_max_cluster + 1):
            imp = i - 1 - k
            if imp < cfg.flag_body_sma:
                continue

            body_i = abs(c[imp] - o[imp])
            rng_i = h[imp] - l[imp]
            if rng_i <= 0 or body_i <= 0:
                continue
            bsma = body_sma[imp]
            vsma = vol_sma[imp]
            if not (np.isfinite(bsma) and bsma > 0):
                continue
            if not (np.isfinite(vsma) and vsma > 0):
                continue
            if body_i < cfg.flag_body_mult * bsma:
                continue
            if v[imp] < cfg.flag_vol_mult * vsma:
                continue

            if c[imp] > o[imp]:
                if (c[imp] - l[imp]) / rng_i < cfg.flag_close_pos_min:
                    continue
                direction = 1
            elif c[imp] < o[imp]:
                if (h[imp] - c[imp]) / rng_i < cfg.flag_close_pos_min:
                    continue
                direction = -1
            else:
                continue

            j0, j1 = imp + 1, imp + k  # inclusive cluster indices
            c_h = float(h[j0 : j1 + 1].max())
            c_l = float(l[j0 : j1 + 1].min())
            if (c_h - c_l) > cfg.flag_cluster_range_ratio * rng_i:
                continue
            c_bodies = np.abs(c[j0 : j1 + 1] - o[j0 : j1 + 1])
            if (c_bodies > cfg.flag_cluster_body_ratio * body_i).any():
                continue

            if direction == 1:
                mid = l[imp] + cfg.flag_retrace_limit * rng_i
                if c_l < mid:
                    continue
                if (c[j0 : j1 + 1] < l[imp]).any():
                    continue
            else:
                mid = h[imp] - cfg.flag_retrace_limit * rng_i
                if c_h > mid:
                    continue
                if (c[j0 : j1 + 1] > h[imp]).any():
                    continue

            return _Pattern(
                impulse_idx=imp,
                cluster_end_idx=j1,
                direction=direction,
                impulse_high=float(h[imp]),
                impulse_low=float(l[imp]),
                impulse_range=float(rng_i),
                cluster_high=c_h,
                cluster_low=c_l,
            )
        return None

    def _build_setup(
        self, i: int, df: pd.DataFrame, p: _Pattern
    ) -> Optional[_Setup]:
        cfg = self.config
        ctx = p.cluster_end_idx  # context evaluated at the cluster end bar

        e = df["flag_ema"].iloc[ctx]
        a = df["atr"].iloc[ctx]
        if not (np.isfinite(e) and np.isfinite(a) and a > 0):
            return None

        price_close = float(df["close"].iloc[ctx])
        slope_anchor = max(0, ctx - cfg.flag_ema_slope_lookback)
        ema_slope = float(df["flag_ema"].iloc[ctx] - df["flag_ema"].iloc[slope_anchor])
        htf_b = int(df["flag_htf_bias"].iloc[ctx])

        hi_window = float(df["flag_hi_window"].iloc[ctx])
        lo_window = float(df["flag_lo_window"].iloc[ctx])
        near_hi = (hi_window - price_close) / price_close < cfg.flag_level_proximity_pct
        near_lo = (price_close - lo_window) / price_close < cfg.flag_level_proximity_pct

        if p.direction == 1:
            with_trend = (price_close > e) and (ema_slope > 0) and (htf_b > 0)
            counter_trend = (htf_b < 0) or (price_close < e and ema_slope < 0)
            near_level = bool(near_hi)
        else:
            with_trend = (price_close < e) and (ema_slope < 0) and (htf_b < 0)
            counter_trend = (htf_b > 0) or (price_close > e and ema_slope > 0)
            near_level = bool(near_lo)

        track: Optional[str] = None
        side: Optional[int] = None
        if cfg.flag_enable_track_a and with_trend and not near_level:
            track, side = "A", p.direction
        elif cfg.flag_enable_track_b and counter_trend:
            track, side = "B", -p.direction
        else:
            return None

        buf = max(cfg.flag_stop_atr_mult * float(a), 0.0)
        tick = cfg.flag_bar_tick
        if track == "A" and side == 1:
            trigger = p.impulse_high + tick
            stop = p.cluster_low - buf
        elif track == "A" and side == -1:
            trigger = p.impulse_low - tick
            stop = p.cluster_high + buf
        elif track == "B" and side == 1:
            trigger = p.cluster_high + tick
            stop = p.impulse_low - buf
        else:  # track B short
            trigger = p.cluster_low - tick
            stop = p.impulse_high + buf

        if side == 1 and stop >= trigger:
            return None
        if side == -1 and stop <= trigger:
            return None
        risk = abs(trigger - stop)
        if risk <= 0:
            return None

        if side == 1:
            t1 = trigger + cfg.flag_t1_r * risk
            t2_rr = trigger + cfg.flag_t2_r * risk
            t2_mm = trigger + p.impulse_range
            t2 = max(t2_rr, t2_mm) if cfg.flag_use_measured_move else t2_rr
        else:
            t1 = trigger - cfg.flag_t1_r * risk
            t2_rr = trigger - cfg.flag_t2_r * risk
            t2_mm = trigger - p.impulse_range
            t2 = min(t2_rr, t2_mm) if cfg.flag_use_measured_move else t2_rr

        rr_avg = abs(0.5 * (t1 + t2) - trigger) / risk
        if rr_avg < cfg.flag_min_rr:
            return None

        return _Setup(
            track=track,
            side=side,
            trigger=float(trigger),
            stop=float(stop),
            t1=float(t1),
            t2=float(t2),
            expiry_bar=i + cfg.flag_breakout_window - 1,
        )

    # ── on_bar ─────────────────────────────────────────────────────────────

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        open_i = float(df["open"].iloc[i])
        high_i = float(df["high"].iloc[i])
        low_i = float(df["low"].iloc[i])
        ts = df.index[i]

        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Manage open position ──────────────────────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade

            # Stop (dynamic — may be BE-shifted) + T2 target, delegated to the
            # exit policy. Stop-first (pessimistic): ref_stop tracks the live
            # active stop, ref_target is T2; both fill intrabar at their level.
            decision = self.exit_policy.evaluate(
                self._exit_ctx(i, df, trade, 0.0,
                               ref_stop=self._active_stop, ref_target=self._t2)
            )
            if decision is not None:
                state.exit(ts, decision.price, decision.reason)
                self._reset_trade_state()
                return

            # T1 breakeven shift (native — modifies the stop for later bars, no exit).
            if trade.direction == Direction.LONG:
                t1_hit = (not self._t1_hit) and high_i >= self._t1
            else:
                t1_hit = (not self._t1_hit) and low_i <= self._t1
            if t1_hit:
                self._t1_hit = True
                if self.config.flag_be_shift_after_t1:
                    self._active_stop = trade.entry_price
            return

        # ── Flat: manage pending stop order ──────────────────────────────
        if self._pending is not None and i > self._pending.expiry_bar:
            self._pending = None

        if self._pending is None:
            p = self._detect_pattern(i, df)
            if p is not None:
                self._pending = self._build_setup(i, df, p)

        if self._pending is None:
            return

        setup = self._pending
        if setup.side == 1 and high_i >= setup.trigger:
            fill = max(open_i, setup.trigger)
            if state.enter(Direction.LONG, ts, fill, stop_price=setup.stop) is not None:
                self._arm_exits(setup)
            self._pending = None
        elif setup.side == -1 and low_i <= setup.trigger:
            fill = min(open_i, setup.trigger)
            if state.enter(Direction.SHORT, ts, fill, stop_price=setup.stop) is not None:
                self._arm_exits(setup)
            self._pending = None

    def _arm_exits(self, setup: _Setup) -> None:
        self._active_stop = setup.stop
        self._t1 = setup.t1
        self._t2 = setup.t2
        self._t1_hit = False


def _infer_ltf_minutes(df: pd.DataFrame, default: int) -> int:
    """Median gap between consecutive timestamps, rounded to minutes."""
    if len(df) < 2:
        return default
    deltas = df.index.to_series().diff().dropna()
    if deltas.empty:
        return default
    seconds = deltas.median().total_seconds()
    if not np.isfinite(seconds) or seconds <= 0:
        return default
    return max(1, int(round(seconds / 60.0)))
