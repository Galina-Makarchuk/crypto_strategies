"""Single source of truth for *trade-level* parameters.

How you trade ANY signal — costs, sizing, direction, leverage, risk overlays —
independent of which strategy generated it. Edit the ACTIVE_TRADE block once:
notebooks/scripts pass it straight to ``Backtester(trading_config=ACTIVE_TRADE)``,
and CLI runs (``python -m engine``) seed from it too — an individual ``--flag``
overrides only that one field, everything else inherits ACTIVE_TRADE. So it is
genuinely the project-wide default, mirroring the ACTIVE dataset block in
:mod:`engine.data_configurator`.

Strategy-specific parameters (indicator periods, ATR/RR stops, etc.) stay on
the per-family ``*Params`` classes in :mod:`engine.strategy_configurator` — they
describe *how signals are generated* and are not configured here.

Units: everything bps-denominated follows the engine convention — 1 bp = 0.01%,
100 bps = 1%, 10_000 bps = 100%. Bybit quotes fees in %, so convert once on the
way in (taker 0.04% → ``fee_bps=4.0``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from . import config_validation as cv

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Which sides a run may take — a gate applied to every signal a strategy
    emits. Distinct from :class:`engine.core.Direction` (the signed side of an
    individual trade): a strategy still decides long/short per setup; this only
    filters which of those are allowed to open."""

    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class SizingMode(Enum):
    """How position notional is chosen.

    FIXED — a flat fraction of equity (``position_size_bps`` × ``leverage``).
    RISK  — sized so that hitting the trade's stop loses ``risk_per_trade_bps``
            of equity. Requires a stop price at entry; strategies that expose no
            fixed stop (trailing/flip exits) fall back to FIXED, which the runner
            counts and logs so the fallback is never silent."""

    FIXED = "fixed"
    RISK = "risk"


@dataclass(frozen=True)
class TradingConfig:
    """Immutable trade-level parameters, consumed by the backtester / live engine."""

    # ── Equity & position sizing ─────────────────────────────────────────────
    initial_equity: float = 10_000.0      # starting account equity (quote ccy)
    position_size_bps: float = 10_000.0   # notional per trade as bps of equity (10_000 = 100%)
    leverage: float = 1.0                 # multiplies the notional (FIXED mode)
    sizing_mode: SizingMode = SizingMode.FIXED  # FIXED = position_size_bps; RISK = risk_per_trade_bps
    risk_per_trade_bps: float = 100.0     # equity risked per trade in RISK mode (100 bps = 1%)

    # ── Costs (per side, basis points) ───────────────────────────────────────
    fee_bps: float = 4.0                  # Bybit taker 0.04% = 4 bps
    slippage_bps: float = 2.0             # estimated 0.02% slip = 2 bps

    # ── Risk overlays (off by default → preserves today's behaviour) ──────────
    max_daily_loss_bps: float | None = None  # halt new entries once a UTC day's realized loss reaches this; None = off
    max_holding_bars: int | None = None       # force-close a trade after this many bars; None = off

    # ── Direction gate ───────────────────────────────────────────────────────
    direction: TradeDirection = TradeDirection.BOTH  # default = current behaviour

    def __post_init__(self) -> None:
        # Validation rules delegated to config_validation (shared with the
        # per-strategy *Params classes); the hook stays here so construction and
        # dataclasses.replace both re-validate.
        o = "TradingConfig"
        cv.positive_number(o, "initial_equity", self.initial_equity)
        cv.positive_number(o, "position_size_bps", self.position_size_bps)
        cv.positive_number(o, "leverage", self.leverage)
        cv.positive_number(o, "risk_per_trade_bps", self.risk_per_trade_bps)
        cv.non_negative_number(o, "fee_bps", self.fee_bps)
        cv.non_negative_number(o, "slippage_bps", self.slippage_bps)
        cv.optional_positive_number(o, "max_daily_loss_bps", self.max_daily_loss_bps)
        cv.optional_positive_int(o, "max_holding_bars", self.max_holding_bars)
        # Enum fields: a raw-string override (e.g. direction='long' instead of
        # TradeDirection.LONG) would otherwise be accepted silently and gate out
        # every trade / revert sizing to FIXED with no error. Reject it here.
        cv.enum_member(o, "sizing_mode", self.sizing_mode, SizingMode)
        cv.enum_member(o, "direction", self.direction, TradeDirection)

    # ── Derived ──────────────────────────────────────────────────────────────

    def total_cost_bps(self) -> float:
        """Round-trip cost (entry + exit), in bps. Derived — never stored, so it
        can't drift from ``fee_bps`` / ``slippage_bps``."""
        return 2 * (self.fee_bps + self.slippage_bps)

    def allows_long(self) -> bool:
        return self.direction in (TradeDirection.LONG, TradeDirection.BOTH)

    def allows_short(self) -> bool:
        return self.direction in (TradeDirection.SHORT, TradeDirection.BOTH)

    def is_asymmetric(self) -> bool:
        """True when the gate blocks one side (LONG-only or SHORT-only). Layering
        this on an ``*_inv`` strategy silently disengages one inversion leg, so
        the runners warn on that combination."""
        return not (self.allows_long() and self.allows_short())

    def notional(self, equity: float) -> float:
        """Fixed-fraction notional for a given account equity (FIXED mode)."""
        return equity * (self.position_size_bps / 10_000.0) * self.leverage

    def risk_notional(
        self, equity: float, entry_price: float, stop_price: float | None
    ) -> float | None:
        """RISK-mode notional: sized so a stop-out loses ~``risk_per_trade_bps``
        of equity. Returns None when there's no usable stop distance (the caller
        then falls back to fixed-fraction). Leverage is intentionally NOT applied
        — risk-based sizing sets the size directly from risk budget ÷ stop
        distance, so applying leverage would multiply the risk taken."""
        if stop_price is None or entry_price <= 0:
            return None
        stop_frac = abs(entry_price - stop_price) / entry_price
        if stop_frac <= 0:
            return None
        return equity * (self.risk_per_trade_bps / 10_000.0) / stop_frac

    def size_notional(
        self, equity: float, entry_price: float, stop_price: float | None
    ) -> tuple[float, bool]:
        """Notional for one trade and whether it fell back to fixed-fraction.
        RISK mode sizes from the entry stop; with no usable stop it falls back to
        fixed-fraction (second element ``True``). FIXED mode never falls back.
        Shared by the backtester's post-run equity pass and the live engine's
        incremental one, so both modes size a given trade identically."""
        if self.sizing_mode == SizingMode.RISK:
            risk_n = self.risk_notional(equity, entry_price, stop_price)
            if risk_n is not None:
                return risk_n, False
            return self.notional(equity), True   # no usable stop → fixed-fraction
        return self.notional(equity), False


def warn_if_inverse_gated(strategy_name: str, config: TradingConfig) -> None:
    """Warn when an asymmetric direction gate is layered on an ``*_inv`` strategy:
    it silently disengages one leg of the inversion (e.g. ``supertrend_inv`` with
    ``direction=SHORT`` drops the dip-buy leg, often leaving very few trades).
    Called by the runners; no-op for symmetric gates or base strategies."""
    if strategy_name.endswith("_inv") and config.is_asymmetric():
        logger.warning(
            "Direction gate '%s' on inverse strategy '%s' disables one inversion "
            "leg — only %s entries can open. Use direction=both for full "
            "mean-reversion, or run the base (non-_inv) strategy.",
            config.direction.value,
            strategy_name,
            "long" if config.allows_long() else "short",
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            ✏️   EDIT HERE   ✏️                              ║
# ║  The ONE place to configure project-wide trade parameters. Change a value, ║
# ║  save, and re-run any notebook/script — they all read ACTIVE_TRADE.        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
ACTIVE_TRADE = TradingConfig(
    initial_equity     = 10_000.0,
    position_size_bps  = 10_000.0,            # 100% of equity deployed as notional (FIXED mode)
    leverage           = 1.0,
    sizing_mode        = SizingMode.FIXED,    # FIXED → position_size_bps; RISK → risk_per_trade_bps
    risk_per_trade_bps = 100.0,               # 1% of equity risked per trade (used in RISK mode)
    fee_bps            = 4.0,                 # Bybit taker 0.04%
    slippage_bps       = 2.0,                 # estimated 0.02% slip
    max_daily_loss_bps = None,                # e.g. 300 → halt entries after −3% realized in a UTC day
    max_holding_bars   = None,                # e.g. 48 → force-close after 48 bars
    direction          = TradeDirection.BOTH,
)
# ── end edit block ─────────────────────────────────────────────────────────
