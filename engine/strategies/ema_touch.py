"""EMA touch-and-rejection strategy.

Ported from the `ema` project's ``backtest.py``. The signal: a bar's wick
*touches* the entry EMA within a ``delta`` tolerance **and** the bar *closes back
on the rejection side* — long when the low tested the EMA and ``close >= EMA``,
short when the high tested it and ``close <= EMA`` — optionally gated by a slower
regime EMA (longs only above it, shorts only below). Entry fills at that bar's
close.

The price stop / take-profit is delegated to the injected exit policy (default
preset ``fixed_1pct_rr3`` = a 1% fixed stop + a 3R target), which runs the stop
first on an ambiguous bar — matching the source's conservative intrabar
resolution. Strategy-level knobs live on ``EmaTouchParams`` (``ema_touch_*``);
cost / sizing / the long-short-both gate are the framework's job (TradingConfig).
The strategy emits both sides (long tried first on an ambiguous bar) and the
direction gate vetoes disallowed sides at ``state.enter``.

Faithfulness notes: the entry EMA is ``ewm(span, adjust=False)`` (= our
``indicators.ema``); the touch test uses ``abs(low - ema) <= tol`` so a wick that
overshoots the EMA by <= tol still counts; an entry whose computed stop lands on
the wrong side of the fill is rejected. Slippage/fees are *not* baked into the
fill here (the runner applies round-trip ``cost_bps``), so trade timing/levels
match the source while P&L follows this engine's cost model.
"""

from __future__ import annotations

import pandas as pd

from ..indicators import ema
from ..swing_detector import wilder_atr
from ..core import Direction, PositionState
from .base import BaseStrategy


class EmaTouchStrategy(BaseStrategy):
    name = "ema_touch"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config

        long_period = cfg.ema_touch_period_long if cfg.ema_touch_period_long is not None else cfg.ema_touch_period
        short_period = cfg.ema_touch_period_short if cfg.ema_touch_period_short is not None else cfg.ema_touch_period
        self._symmetric = short_period == long_period

        df["ema_touch_long"] = ema(df["close"], long_period)
        df["ema_touch_short"] = df["ema_touch_long"] if self._symmetric else ema(df["close"], short_period)

        # Wilder ATR (matches the source's _compute_atr) — used by the 'atr' delta
        # mode and exposed as ctx.atr for any ATR-based exit preset.
        df["ema_touch_atr"] = wilder_atr(df, cfg.ema_touch_atr_period)

        df["ema_touch_tol_long"] = self._tol(df["ema_touch_long"], df["ema_touch_atr"])
        df["ema_touch_tol_short"] = (
            df["ema_touch_tol_long"] if self._symmetric
            else self._tol(df["ema_touch_short"], df["ema_touch_atr"])
        )

        long_regime = cfg.ema_touch_regime_filter_long if cfg.ema_touch_regime_filter_long is not None else cfg.ema_touch_regime_filter
        short_regime = cfg.ema_touch_regime_filter_short if cfg.ema_touch_regime_filter_short is not None else cfg.ema_touch_regime_filter
        self._has_regime_long = long_regime is not None
        self._has_regime_short = short_regime is not None
        df["ema_touch_regime_long"] = ema(df["close"], long_regime) if self._has_regime_long else float("nan")
        df["ema_touch_regime_short"] = ema(df["close"], short_regime) if self._has_regime_short else float("nan")

        # Skip the unsettled head of the EMAs (matches the source's `warmup`).
        warmup = max(long_period, short_period)
        if cfg.ema_touch_delta_mode == "atr":
            warmup = max(warmup, cfg.ema_touch_atr_period)
        if self._has_regime_long:
            warmup = max(warmup, long_regime)
        if self._has_regime_short:
            warmup = max(warmup, short_regime)
        self._warmup = warmup
        return df

    def _tol(self, ema_series: pd.Series, atr_series: pd.Series) -> pd.Series:
        """The per-bar touch-tolerance band. ``absolute`` = fixed quote points;
        ``percent`` = delta% of the EMA level; ``atr`` = delta × ATR (cross-symbol)."""
        mode = self.config.ema_touch_delta_mode
        delta = self.config.ema_touch_delta
        if mode == "percent":
            return ema_series.abs() * (delta / 100.0)
        if mode == "atr":
            return atr_series * delta
        if mode == "absolute":
            return pd.Series(delta, index=ema_series.index, dtype=float)
        raise ValueError(f"ema_touch_delta_mode must be 'absolute', 'percent' or 'atr'; got {mode!r}")

    def on_bar(self, i: int, df: pd.DataFrame, state: PositionState) -> None:
        if i < self._warmup:
            return

        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]

        # ── update peak → exit (only when already open; never on the entry bar) ──
        if state.current_trade is not None:
            trade = state.current_trade
            state.update_peak(high_i, low_i)
            decision = self.exit_policy.evaluate(
                self._exit_ctx(i, df, trade, df["ema_touch_atr"].iloc[i])
            )
            if decision is not None:
                state.exit(df.index[i], decision.price, decision.reason)
            return

        # ── entry: long tried first; if it doesn't open, try short ──
        if self._enter_side(Direction.LONG, i, df, state):
            return
        self._enter_side(Direction.SHORT, i, df, state)

    def _enter_side(self, direction: Direction, i: int, df: pd.DataFrame, state: PositionState) -> bool:
        close = df["close"].iloc[i]
        high_i = df["high"].iloc[i]
        low_i = df["low"].iloc[i]

        if direction is Direction.LONG:
            ema_i = df["ema_touch_long"].iloc[i]
            tol_i = df["ema_touch_tol_long"].iloc[i]
            touched = abs(low_i - ema_i) <= tol_i
            rejected = close >= ema_i
            macro_ok = (not self._has_regime_long) or (close > df["ema_touch_regime_long"].iloc[i])
        else:
            ema_i = df["ema_touch_short"].iloc[i]
            tol_i = df["ema_touch_tol_short"].iloc[i]
            touched = abs(high_i - ema_i) <= tol_i
            rejected = close <= ema_i
            macro_ok = (not self._has_regime_short) or (close < df["ema_touch_regime_short"].iloc[i])

        if not (touched and rejected and macro_ok):
            return False

        stop = self._entry_stop(direction, close, df["ema_touch_atr"].iloc[i])
        # Reject entries whose stop lands on the wrong side of the fill (source parity).
        if stop is not None:
            if direction is Direction.LONG and stop >= close:
                return False
            if direction is Direction.SHORT and stop <= close:
                return False

        return state.enter(direction, df.index[i], close, stop_price=stop) is not None
