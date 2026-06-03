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
# StrategyConfig (strategy parameters) moved to engine/strategy_configurator.py.
# This module holds only domain types + the PositionState state machine; the
# config layers live in strategy_configurator.py / trade_configurator.py.


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

    # Initial stop price at entry, when the strategy exposes one (used by
    # RISK-mode position sizing). None for trailing/flip-exit strategies.
    stop_price: Optional[float] = None

    # Equity layer (additive; filled by the backtester's post-run sizing pass —
    # PositionState itself stays equity-agnostic). pnl_bps above is unaffected.
    notional: float = 0.0       # position size in quote ccy at entry
    pnl_currency: float = 0.0   # notional * pnl_bps / 10_000
    equity_after: float = 0.0   # running account equity after this trade closed

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
    """Tracks the current open position, if any.

    Trade-level policy is *seeded by the runner* (backtester / live engine) from
    a TradingConfig — PositionState never imports trade_configurator, it just
    holds primitives. ``cost_bps`` is applied at exit; ``allow_long`` /
    ``allow_short`` and ``max_daily_loss_bps`` gate entries. Defaults reproduce
    the pre-trade-config behaviour (free, both directions, no daily cap).
    """

    status: PositionStatus = PositionStatus.FLAT
    current_trade: Optional[Trade] = None
    closed_trades: list[Trade] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    _trade_counter: int = 0

    # Runner-seeded trade-level policy (see TradingConfig).
    cost_bps: float = 0.0
    allow_long: bool = True
    allow_short: bool = True
    max_daily_loss_bps: Optional[float] = None

    # Observability: entries a strategy attempted but trade-level policy blocked
    # (direction gate or daily-loss halt). Surfaced by the runner so a gate that
    # silently drops signals is visible rather than looking like "fewer trades".
    suppressed_entries: int = 0

    # ── state transitions ──────────────────────────────────────────────────

    def _daily_loss_halted(self, ts: datetime) -> bool:
        """True if the day's realized P&L has hit the configured loss cap, so no
        new entries are allowed for the rest of that UTC day."""
        if self.max_daily_loss_bps is None:
            return False
        day = ts.date()
        realized = sum(
            t.pnl_bps
            for t in self.closed_trades
            if t.exit_ts is not None and t.exit_ts.date() == day
        )
        return realized <= -self.max_daily_loss_bps

    def enter(
        self,
        direction: Direction,
        ts: datetime,
        price: float,
        stop_price: float | None = None,
    ) -> Signal | None:
        if self.status != PositionStatus.FLAT:
            return None  # already in a position — normal, not a policy suppression
        # Direction gate (trade-level): reject a side the run isn't allowed to take.
        if direction == Direction.LONG and not self.allow_long:
            self.suppressed_entries += 1
            return None
        if direction == Direction.SHORT and not self.allow_short:
            self.suppressed_entries += 1
            return None
        # Daily-loss overlay: block entries once today's realized loss cap is hit.
        if self._daily_loss_halted(ts):
            self.suppressed_entries += 1
            return None
        self._trade_counter += 1
        self.current_trade = Trade(
            direction=direction,
            entry_ts=ts,
            entry_price=price,
            peak_price=price,
            stop_price=stop_price,
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
        reason: ExitReason = ExitReason.FORCE_CLOSE,
    ) -> Signal | None:
        if self.status != PositionStatus.OPEN or self.current_trade is None:
            return None
        trade = self.current_trade
        trade.exit_ts = ts
        trade.exit_price = price
        trade.exit_reason = reason

        # P&L in basis points, net of the runner-seeded round-trip cost.
        if trade.direction == Direction.LONG:
            raw_bps = (price - trade.entry_price) / trade.entry_price * 10_000
        else:
            raw_bps = (trade.entry_price - price) / trade.entry_price * 10_000
        trade.pnl_bps = raw_bps - self.cost_bps

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
