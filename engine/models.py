"""Typed domain models for the trading system.

All shared types live here so every module imports from one place.
No business logic — just data definitions and validation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Optional


# ── Enums ──────────────────────────────────────────────────────────────────────


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


class SignalAction(Enum):
    ENTRY = "entry"
    EXIT = "exit"


class PositionStatus(Enum):
    FLAT = auto()
    OPEN = auto()


class StrategyName(Enum):
    SWING = "swing"
    SWING_INV = "swing_inv"
    EMA_CROSS = "ema"
    EMA_CROSS_INV = "ema_inv"
    SUPERTREND = "supertrend"
    SUPERTREND_INV = "supertrend_inv"
    SUPERTREND_ADAPTIVE = "supertrend_adaptive"
    EXHAUSTION_REVERSAL = "exhaustion_reversal"
    IMPULSE_FLAG = "impulse_flag"
    ORDER_BLOCK = "order_block"
    ORDER_BLOCK_INV = "order_block_inv"
    VWAP_BANDS = "vwap_bands"
    SWING_ZIGZAG = "swing_zigzag"
    SWING_ZIGZAG_ML = "swing_zigzag_ml"


class RunMode(Enum):
    HISTORICAL = "historical"
    LIVE = "live"


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    SIGNAL_FLIP = "signal_flip"
    TIME_STOP = "time_stop"
    INVALIDATION = "invalidation"
    FORCE_CLOSE = "force_close"


# ── Validated intervals ────────────────────────────────────────────────────────

VALID_INTERVALS = frozenset(
    {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}
)


def validate_interval(interval: str) -> str:
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Must be one of: {sorted(VALID_INTERVALS)}"
        )
    return interval


# ── Validated product categories ────────────────────────────────────────────────

VALID_CATEGORIES = frozenset({"linear", "inverse"})


def validate_category(category: str) -> str:
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {sorted(VALID_CATEGORIES)}"
        )
    return category


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategyConfig:
    """Immutable strategy parameters.  All magic numbers live here."""

    # Swing detection
    left: int = 5
    right: int = 5
    merge_tolerance: float = 0.0015  # 0.15 %

    # EMA crossover
    ema_fast: int = 9
    ema_slow: int = 21

    # RSI
    rsi_period: int = 14

    # SuperTrend
    supertrend_period: int = 10
    supertrend_mult: float = 3.0

    # ATR / risk
    atr_period: int = 14
    atr_trail_mult: float = 2.0

    # Exhaustion-reversal strategy
    exhaustion_push_min_len: int = 2        # min |length| of the push-leg streak
    exhaustion_stall_min_count: int = 2     # min number of stall streaks after the push
    exhaustion_stall_max_len: int = 1       # max |length| of a stall streak
    exhaustion_trigger_min_len: int = 2     # min |length| of the in-progress trigger streak
    exhaustion_volume_factor: float = 1.0   # trigger per-candle vol / push per-candle vol
    exhaustion_stop_atr_mult: float = 0.25  # stop buffer above/below cluster, in ATRs
    exhaustion_target_rr: float = 2.0       # fixed reward:risk target multiple
    exhaustion_time_stop_bars: int = 12     # force-exit after this many bars in trade
    exhaustion_invalidation_len: int = 3    # structural invalidation streak length

    # Impulse + consolidation ("flag") strategy
    flag_body_mult: float = 1.5             # impulse body >= mult * SMA(body)
    flag_vol_mult: float = 1.2              # impulse volume >= mult * SMA(vol)
    flag_close_pos_min: float = 2.0 / 3.0   # impulse close must sit in top/bottom third
    flag_min_cluster: int = 2
    flag_max_cluster: int = 4
    flag_cluster_body_ratio: float = 0.5    # cluster bar body <= ratio * impulse body
    flag_cluster_range_ratio: float = 0.7   # cluster H-L <= ratio * impulse H-L
    flag_retrace_limit: float = 0.5         # cluster must hold above/below 50% retrace
    flag_breakout_window: int = 3           # bars after cluster to fill the entry
    flag_ema_fast: int = 20
    flag_vol_sma: int = 9
    flag_body_sma: int = 20
    flag_ema_slope_lookback: int = 5
    flag_htf_minutes: int = 60
    flag_htf_ema: int = 20
    flag_htf_slope_lookback: int = 3
    flag_level_proximity_pct: float = 0.003  # Track A skip if within 0.3% of 24H hi/lo
    flag_level_lookback_hours: float = 24.0  # rolling-level window, converted to bars at runtime
    flag_stop_atr_mult: float = 0.5          # stop buffer = mult * ATR
    flag_min_rr: float = 1.5                 # gate on averaged target
    flag_enable_track_a: bool = True
    flag_enable_track_b: bool = True
    flag_t1_r: float = 1.0                   # T1 distance in R-multiples
    flag_t2_r: float = 2.0                   # T2 distance in R-multiples
    flag_use_measured_move: bool = True      # T2 = max(2R, impulse range projection)
    flag_be_shift_after_t1: bool = True      # move stop to BE once T1 is touched
    flag_bar_tick: float = 0.1               # trigger offset past the level

    # Order-block strategy
    ob_body_mult: float = 1.8                # impulse body >= mult * ATR
    ob_consecutive_min: int = 3              # min same-direction candle run
    ob_consecutive_range_pct: float = 0.006  # combined body / close threshold
    ob_max_age_hours: float = 3.0            # OBs older than this are discarded
    ob_htf_minutes: int = 60                 # HTF resample for bias filter
    ob_htf_ema: int = 20                     # EMA period on HTF bias
    ob_ema_fast: int = 9                     # LTF fast EMA (confirmation cross)
    ob_ema_slow: int = 20                    # LTF slow EMA (confirmation cross)
    ob_stop_buffer_pct: float = 0.005        # stop buffer beyond OB extreme
    ob_rr: float = 2.5                       # fixed reward:risk on target

    # VWAP Stdev Bands strategy (defaults mirror the TradingView "v2 Mod" preset)
    vwap_band_devs: tuple[float, ...] = (1.28, 2.01, 2.51, 3.09, 4.01)
    vwap_session: str = "D"                  # session anchor for VWAP reset
    vwap_entry_band: int = 4                 # 0-indexed; default = furthest band

    # ATR-prominence ZigZag swing strategy
    # See engine/swings.py for the detector contract.
    swing_zz_atr_period: int = 14
    swing_zz_min_prominence_atr: float = 1.5
    swing_zz_min_bars_between: int = 3
    swing_zz_vol_lookback: int = 50
    swing_zz_min_score: float = 0.0
    swing_zz_use_stop: bool = True
    swing_zz_stop_atr_mult: float = 3.0

    # ML swing-pivot classifier strategy.
    # ml_model_path is resolved relative to the repo root if not absolute.
    ml_model_path: str = "ml_models/swing_zz_ml.joblib"
    ml_p_threshold: float = 0.55
    ml_use_stop: bool = True
    ml_stop_atr_mult: float = 3.0

    # Costs (basis points)
    # Bybit's taker fee is 0.04% per trade. 1 basis point = 0.01%, so 4 bps = 0.04%
    # Estimated 0.02% price slip per trade from the difference between the price you see and the price you actually fill at (spread, latency, orderbook depth)
    fee_bps: float = 4.0  # taker fee per side
    slippage_bps: float = 2.0

    # Risk management
    risk_per_trade_pct: float = 1.0  # % of equity risked per trade, risks 1% of the account equity per trade
    max_open_trades: int = 1

    def total_cost_bps(self) -> float:
        """Round-trip cost (entry + exit)."""
        return 2 * (self.fee_bps + self.slippage_bps)


# ── Signal / Trade models ──────────────────────────────────────────────────────


@dataclass
class Signal:
    """Emitted by a strategy; consumed by the backtester / live engine."""

    timestamp: datetime
    action: SignalAction
    direction: Direction
    price: float
    label: str = ""


@dataclass
class Trade:
    """A completed round-trip trade with P&L."""

    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    direction: Direction = Direction.LONG
    entry_ts: Optional[datetime] = None
    entry_price: float = 0.0
    exit_ts: Optional[datetime] = None
    exit_price: float = 0.0
    pnl_bps: float = 0.0  # net of fees
    peak_price: float = 0.0  # high-water mark (longs) / low-water (shorts)
    exit_reason: Optional[ExitReason] = None

    @property
    def is_closed(self) -> bool:
        return self.exit_ts is not None

    @property
    def duration(self) -> Optional[timedelta]:
        if self.entry_ts is None or self.exit_ts is None:
            return None
        return self.exit_ts - self.entry_ts


@dataclass
class PositionState:
    """Tracks the current open position, if any."""

    status: PositionStatus = PositionStatus.FLAT
    current_trade: Optional[Trade] = None
    closed_trades: list[Trade] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    _trade_counter: int = 0

    # ── state transitions ──────────────────────────────────────────────────

    def enter(self, direction: Direction, ts: datetime, price: float) -> Signal | None:
        if self.status != PositionStatus.FLAT:
            return None
        self._trade_counter += 1
        self.current_trade = Trade(
            direction=direction,
            entry_ts=ts,
            entry_price=price,
            peak_price=price,
        )
        self.status = PositionStatus.OPEN
        sig = Signal(
            timestamp=ts,
            action=SignalAction.ENTRY,
            direction=direction,
            price=price,
            label=f"{direction.value.capitalize()} Entry #{self._trade_counter}",
        )
        self.signals.append(sig)
        return sig

    def update_peak(self, high: float, low: float) -> None:
        """Update trailing-stop high-water / low-water mark."""
        if self.current_trade is None:
            return
        if self.current_trade.direction == Direction.LONG:
            self.current_trade.peak_price = max(self.current_trade.peak_price, high)
        else:
            if self.current_trade.peak_price == 0:
                self.current_trade.peak_price = low
            self.current_trade.peak_price = min(self.current_trade.peak_price, low)

    def exit(
        self,
        ts: datetime,
        price: float,
        cost_bps: float = 0.0,
        reason: ExitReason = ExitReason.FORCE_CLOSE,
    ) -> Signal | None:
        if self.status != PositionStatus.OPEN or self.current_trade is None:
            return None
        trade = self.current_trade
        trade.exit_ts = ts
        trade.exit_price = price
        trade.exit_reason = reason

        # P&L in basis points
        if trade.direction == Direction.LONG:
            raw_bps = (price - trade.entry_price) / trade.entry_price * 10_000
        else:
            raw_bps = (trade.entry_price - price) / trade.entry_price * 10_000
        trade.pnl_bps = raw_bps - cost_bps

        self.closed_trades.append(trade)
        self.status = PositionStatus.FLAT
        dir_label = trade.direction.value.capitalize()
        sig = Signal(
            timestamp=ts,
            action=SignalAction.EXIT,
            direction=trade.direction,
            price=price,
            label=f"{dir_label} Exit #{self._trade_counter}",
        )
        self.signals.append(sig)
        self.current_trade = None
        return sig
