"""Order Block reaction strategy.

PATTERN
    Mark an "Order Block" (OB): the last opposing candle immediately
    before an impulsive move.  Wait for price to revisit the OB zone and
    print a same-direction confirmation candle (engulfs the zone AND
    either EMA-fast crosses EMA-slow OR closes past the OB extreme).
    Only take the trade when the higher-timeframe bias agrees.

IMPULSE
    Either a single candle with body > `ob_body_mult` * ATR, OR a run of
    >= `ob_consecutive_min` same-direction candles whose combined body
    exceeds `ob_consecutive_range_pct` of the latest close.

HTF BIAS
    Resample to `ob_htf_minutes` bars, EMA(`ob_htf_ema`) on the resample,
    `.shift(1)` so only the last CLOSED HTF bar is visible, reindex back
    with forward-fill.  +1 if HTF close >= EMA else -1.

EXITS (fixed)
    Stop  = OB extreme ± (`ob_stop_buffer_pct` * extreme).
    Target = entry ± `ob_rr` * |entry − stop|.

NO LOOK-AHEAD
    HTF bias is shifted by one HTF bar in `prepare()`.  Impulse / OB
    detection at bar `i` only reads `df.iloc[..i]`; the OB itself is
    always at an index < `i`.  Confirmation reads `df.iloc[i-1..i]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ..indicators import atr as calc_atr
from ..indicators import ema as calc_ema
from ..models import Direction, ExitReason, PositionState, StrategyConfig
from .base import BaseStrategy


@dataclass
class _OrderBlock:
    kind: str  # 'bullish' or 'bearish'
    high: float
    low: float
    time: pd.Timestamp


class OrderBlockStrategy(BaseStrategy):
    """Order Block reaction with HTF bias and fixed R:R exits."""

    name = "order_block"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._obs: List[_OrderBlock] = []
        self._active_stop: float = 0.0
        self._active_target: float = 0.0

    # ── prepare ────────────────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config

        df["ob_ema_fast"] = calc_ema(df["close"], cfg.ob_ema_fast)
        df["ob_ema_slow"] = calc_ema(df["close"], cfg.ob_ema_slow)
        df["atr"] = calc_atr(df, cfg.atr_period)

        ltf_minutes = _infer_ltf_minutes(df, default=5)
        htf_minutes = cfg.ob_htf_minutes
        if htf_minutes <= ltf_minutes:
            htf_minutes = max(60, 3 * ltf_minutes)

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
            htf_ema = calc_ema(htf["close"], cfg.ob_htf_ema)
            bias = pd.Series(0, index=htf.index, dtype=int)
            bias[htf["close"] >= htf_ema] = 1
            bias[htf["close"] < htf_ema] = -1
            bias = bias.shift(1)
            df["ob_htf_bias"] = (
                bias.reindex(df.index, method="ffill").fillna(0).astype(int)
            )
        else:
            df["ob_htf_bias"] = 0

        # Live engine calls prepare() every tick — reset runtime state.
        self._obs = []
        self._reset_trade_state()
        return df

    # ── helpers ────────────────────────────────────────────────────────────

    def _reset_trade_state(self) -> None:
        self._active_stop = 0.0
        self._active_target = 0.0

    def _is_impulse(self, i: int, df: pd.DataFrame) -> bool:
        cfg = self.config
        if i < cfg.atr_period + 1:
            return False
        atr_v = float(df["atr"].iloc[i])
        if not np.isfinite(atr_v) or atr_v <= 0:
            return False

        o = float(df["open"].iloc[i])
        c = float(df["close"].iloc[i])
        body = abs(c - o)

        if body > cfg.ob_body_mult * atr_v:
            return True

        direction = np.sign(c - o)
        if direction == 0:
            return False

        consec = 1
        total_body = body
        for k in range(1, cfg.ob_consecutive_min + 2):
            j = i - k
            if j < 0:
                break
            prev_o = float(df["open"].iloc[j])
            prev_c = float(df["close"].iloc[j])
            if np.sign(prev_c - prev_o) != direction:
                break
            consec += 1
            total_body += abs(prev_c - prev_o)

        if consec < cfg.ob_consecutive_min:
            return False
        close_ref = c if c != 0 else 1.0
        return (total_body / close_ref) > cfg.ob_consecutive_range_pct

    def _detect_ob(self, i: int, df: pd.DataFrame) -> Optional[_OrderBlock]:
        if not self._is_impulse(i, df):
            return None

        direction = np.sign(
            float(df["close"].iloc[i]) - float(df["open"].iloc[i])
        )
        start = i
        for k in range(1, 10):
            j = i - k
            if j < 0:
                break
            prev_dir = np.sign(
                float(df["close"].iloc[j]) - float(df["open"].iloc[j])
            )
            if prev_dir != direction:
                break
            start = j

        ob_idx = start - 1
        if ob_idx < 0:
            return None

        ob_o = float(df["open"].iloc[ob_idx])
        ob_c = float(df["close"].iloc[ob_idx])
        kind = "bullish" if ob_c < ob_o else "bearish"
        return _OrderBlock(
            kind=kind,
            high=float(df["high"].iloc[ob_idx]),
            low=float(df["low"].iloc[ob_idx]),
            time=df.index[ob_idx],
        )

    def _confirmed(self, i: int, df: pd.DataFrame, ob: _OrderBlock) -> bool:
        if i < 1:
            return False
        o = float(df["open"].iloc[i])
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])
        ef_now = float(df["ob_ema_fast"].iloc[i])
        es_now = float(df["ob_ema_slow"].iloc[i])
        ef_prev = float(df["ob_ema_fast"].iloc[i - 1])
        es_prev = float(df["ob_ema_slow"].iloc[i - 1])
        if not (np.isfinite(ef_now) and np.isfinite(es_now)):
            return False

        touches = (l <= ob.high) and (h >= ob.low)
        if not touches:
            return False

        if ob.kind == "bullish":
            if c <= o:
                return False
            cross = (ef_prev <= es_prev) and (ef_now > es_now)
            return cross or (c > ob.low)
        else:
            if c >= o:
                return False
            cross = (ef_prev >= es_prev) and (ef_now < es_now)
            return cross or (c < ob.high)

    # ── on_bar ─────────────────────────────────────────────────────────────

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        cfg = self.config
        ts = df.index[i]
        high_i = float(df["high"].iloc[i])
        low_i = float(df["low"].iloc[i])
        close_i = float(df["close"].iloc[i])

        if state.current_trade is not None:
            state.update_peak(high_i, low_i)

        # ── Manage open position: fixed SL / TP ──────────────────────────
        if state.current_trade is not None:
            trade = state.current_trade
            if trade.direction == Direction.LONG:
                if low_i <= self._active_stop:
                    state.exit(ts, self._active_stop, ExitReason.STOP_LOSS)
                    self._reset_trade_state()
                    return
                if high_i >= self._active_target:
                    state.exit(ts, self._active_target, ExitReason.TAKE_PROFIT)
                    self._reset_trade_state()
                    return
            else:
                if high_i >= self._active_stop:
                    state.exit(ts, self._active_stop, ExitReason.STOP_LOSS)
                    self._reset_trade_state()
                    return
                if low_i <= self._active_target:
                    state.exit(ts, self._active_target, ExitReason.TAKE_PROFIT)
                    self._reset_trade_state()
                    return
            return

        # ── Register new OB when an impulse completes on this bar ────────
        new_ob = self._detect_ob(i, df)
        if new_ob is not None:
            self._obs.append(new_ob)

        # ── Age out stale OBs ────────────────────────────────────────────
        cutoff = ts - pd.Timedelta(hours=cfg.ob_max_age_hours)
        self._obs = [ob for ob in self._obs if ob.time >= cutoff]
        if not self._obs:
            return

        htf_bias = int(df["ob_htf_bias"].iloc[i])
        for ob in list(self._obs):
            if not self._confirmed(i, df, ob):
                continue
            if ob.kind == "bullish" and htf_bias <= 0:
                continue
            if ob.kind == "bearish" and htf_bias >= 0:
                continue

            if ob.kind == "bullish":
                stop = ob.low * (1.0 - cfg.ob_stop_buffer_pct)
                risk = close_i - stop
                if risk <= 0:
                    continue
                target = close_i + cfg.ob_rr * risk
                direction = Direction.LONG
            else:
                stop = ob.high * (1.0 + cfg.ob_stop_buffer_pct)
                risk = stop - close_i
                if risk <= 0:
                    continue
                target = close_i - cfg.ob_rr * risk
                direction = Direction.SHORT

            if state.enter(direction, ts, close_i, stop_price=stop) is not None:
                self._active_stop = float(stop)
                self._active_target = float(target)
                self._obs.remove(ob)
                return


def _infer_ltf_minutes(df: pd.DataFrame, default: int) -> int:
    if len(df) < 2:
        return default
    deltas = df.index.to_series().diff().dropna()
    if deltas.empty:
        return default
    seconds = deltas.median().total_seconds()
    if not np.isfinite(seconds) or seconds <= 0:
        return default
    return max(1, int(round(seconds / 60.0)))
