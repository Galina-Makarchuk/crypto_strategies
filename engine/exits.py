"""Exit policies — selectable, composable stop-loss and take-profit mechanisms.

An exit is an exit: a stop-loss and a take-profit both decide, on a given bar,
whether to close the open position and at what price/reason. So both live behind
one interface (:class:`ExitPolicy`) rather than separate hierarchies.

Policies are pure *mechanisms*; they're configured/selected in
``strategy_configurator.py`` and (later) injected into strategies. This module
depends only on ``models`` (Direction/ExitReason) — never on strategies — so it
sits cleanly below them in the import graph.

Two fill conventions, chosen to mirror the strategies they'll replace:
  * **Trailing stop** (chandelier) triggers on the *close* and fills at the close.
  * **Fixed stop / target** triggers *intrabar* (bar low/high pierces the level)
    and fills at the level.

Stop-vs-target precedence on an ambiguous bar (both could have filled): the
backtester can't see intrabar order, so :class:`CompositeExit` is **stop-first** —
list the stop before the target and the stop wins. Conservative by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .models import Direction, ExitReason


@dataclass(frozen=True)
class ExitContext:
    """Everything a policy needs to judge one bar of an open trade.

    ``stop_price`` is the trade's stop fixed at entry (what ``initial_stop``
    returned, recorded on the Trade and fed back each bar) — fixed stops and
    R-multiple targets read it. ``ref_stop`` / ``ref_target`` are strategy-supplied
    structural levels (e.g. an order-block extreme) for policies that need them.
    """

    direction: Direction
    entry_price: float
    peak_price: float          # high-water (long) / low-water (short)
    high: float
    low: float
    close: float
    atr: float                 # ATR at this bar (the strategy provides it)
    stop_price: Optional[float] = None
    ref_stop: Optional[float] = None
    ref_target: Optional[float] = None


@dataclass(frozen=True)
class ExitDecision:
    """A policy's verdict for a bar: close at ``price`` for ``reason``."""

    price: float
    reason: ExitReason


class ExitPolicy(ABC):
    """Decides whether/where to close an open position."""

    def initial_stop(self, ctx: ExitContext) -> Optional[float]:
        """Stop price at entry, used to seed ``state.enter(stop_price=…)`` so
        risk-based sizing can use it. ``None`` when the policy has no fixed entry
        stop (e.g. a pure take-profit)."""
        return None

    @abstractmethod
    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        """Return an exit for this bar, or ``None`` to stay in the trade."""


# ── Stops ────────────────────────────────────────────────────────────────────


class ChandelierStop(ExitPolicy):
    """ATR trailing stop from the high/low-water mark, close-triggered.

    This is the 'chandelier': stop = peak − atr_mult·ATR (long) and ratchets up
    with the peak. Matches the trend strategies' existing trailing-stop logic
    (``close`` vs ``peak ∓ atr_mult·ATR``)."""

    def __init__(self, atr_mult: float = 3.0):
        self.atr_mult = atr_mult

    def initial_stop(self, ctx: ExitContext) -> Optional[float]:
        off = self.atr_mult * ctx.atr
        return ctx.entry_price - off if ctx.direction is Direction.LONG else ctx.entry_price + off

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        off = self.atr_mult * ctx.atr
        if ctx.direction is Direction.LONG:
            if ctx.close < ctx.peak_price - off:
                return ExitDecision(ctx.close, ExitReason.TRAILING_STOP)
        else:
            if ctx.close > ctx.peak_price + off:
                return ExitDecision(ctx.close, ExitReason.TRAILING_STOP)
        return None


class _FixedStop(ExitPolicy):
    """Base for stops whose level is fixed at entry. The level is recorded on the
    trade (``ctx.stop_price``) and checked intrabar, filling at the stop. Subclasses
    supply ``initial_stop``."""

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        lvl = ctx.stop_price
        if lvl is None:
            return None
        if ctx.direction is Direction.LONG:
            if ctx.low <= lvl:
                return ExitDecision(lvl, ExitReason.STOP_LOSS)
        else:
            if ctx.high >= lvl:
                return ExitDecision(lvl, ExitReason.STOP_LOSS)
        return None


class AtrStop(_FixedStop):
    """Fixed stop at entry ∓ atr_mult·ATR (set once at entry, not trailing)."""

    def __init__(self, atr_mult: float = 1.5):
        self.atr_mult = atr_mult

    def initial_stop(self, ctx: ExitContext) -> Optional[float]:
        off = self.atr_mult * ctx.atr
        return ctx.entry_price - off if ctx.direction is Direction.LONG else ctx.entry_price + off


class FixedPctStop(_FixedStop):
    """Fixed stop a fixed percent away from entry (e.g. 2%)."""

    def __init__(self, pct: float):
        self.frac = pct / 100.0

    def initial_stop(self, ctx: ExitContext) -> Optional[float]:
        if ctx.direction is Direction.LONG:
            return ctx.entry_price * (1.0 - self.frac)
        return ctx.entry_price * (1.0 + self.frac)


class StructuralStop(ExitPolicy):
    """Stop at a strategy-supplied level — e.g. just beyond an order-block extreme,
    or a flag stop that the strategy moves to breakeven. Uses ``ctx.ref_stop`` when
    supplied (so the strategy can move the stop per bar), else the trade's entry
    ``stop_price``. Intrabar trigger, fills at the level."""

    def initial_stop(self, ctx: ExitContext) -> Optional[float]:
        return ctx.ref_stop

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        lvl = ctx.ref_stop if ctx.ref_stop is not None else ctx.stop_price
        if lvl is None:
            return None
        if ctx.direction is Direction.LONG:
            if ctx.low <= lvl:
                return ExitDecision(lvl, ExitReason.STOP_LOSS)
        else:
            if ctx.high >= lvl:
                return ExitDecision(lvl, ExitReason.STOP_LOSS)
        return None


# ── Targets (take-profits) ─────────────────────────────────────────────────────


class _Target(ExitPolicy):
    """Base for take-profits: a fixed target level checked intrabar, filling at
    the target. Subclasses supply ``target``. No ``initial_stop`` (a target isn't
    a stop)."""

    def target(self, ctx: ExitContext) -> Optional[float]:
        raise NotImplementedError

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        tgt = self.target(ctx)
        if tgt is None:
            return None
        if ctx.direction is Direction.LONG:
            if ctx.high >= tgt:
                return ExitDecision(tgt, ExitReason.TAKE_PROFIT)
        else:
            if ctx.low <= tgt:
                return ExitDecision(tgt, ExitReason.TAKE_PROFIT)
        return None


class FixedPctTarget(_Target):
    """Take-profit a fixed percent from entry."""

    def __init__(self, pct: float):
        self.frac = pct / 100.0

    def target(self, ctx: ExitContext) -> Optional[float]:
        if ctx.direction is Direction.LONG:
            return ctx.entry_price * (1.0 + self.frac)
        return ctx.entry_price * (1.0 - self.frac)


class StructuralTarget(_Target):
    """Take-profit at a strategy-supplied level (``ctx.ref_target``) — e.g. an
    order-block's R:R target or a measured move. The strategy computes the level
    and passes it each bar; this policy provides the fill mechanism."""

    def target(self, ctx: ExitContext) -> Optional[float]:
        return ctx.ref_target


class CloseCrossTarget(ExitPolicy):
    """Take-profit when the *close* reverts to/through a strategy-supplied level
    (``ctx.ref_target``), filling at the close — e.g. a VWAP mean-reversion exit
    where price closes back across the VWAP line. Close-triggered and close-filled,
    unlike the intrabar, fill-at-level :class:`StructuralTarget`."""

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        if ctx.ref_target is None:
            return None
        if ctx.direction is Direction.LONG and ctx.close >= ctx.ref_target:
            return ExitDecision(ctx.close, ExitReason.TAKE_PROFIT)
        if ctx.direction is Direction.SHORT and ctx.close <= ctx.ref_target:
            return ExitDecision(ctx.close, ExitReason.TAKE_PROFIT)
        return None


class RrTarget(_Target):
    """Reward:risk take-profit at entry ± rr·risk, where risk = |entry − stop|.
    Needs a stop to define risk; returns no target when ``stop_price`` is unset."""

    def __init__(self, rr: float = 2.0):
        self.rr = rr

    def target(self, ctx: ExitContext) -> Optional[float]:
        if ctx.stop_price is None:
            return None
        risk = abs(ctx.entry_price - ctx.stop_price)
        if risk <= 0:
            return None
        if ctx.direction is Direction.LONG:
            return ctx.entry_price + self.rr * risk
        return ctx.entry_price - self.rr * risk


# ── Composition ────────────────────────────────────────────────────────────────


class CompositeExit(ExitPolicy):
    """Run several policies in priority order; the first to fire wins. List the
    stop before any target for **stop-first** precedence on ambiguous bars.
    ``initial_stop`` returns the first policy that defines one (so the stop's
    level is what seeds risk sizing)."""

    def __init__(self, *policies: ExitPolicy):
        self.policies = policies

    def initial_stop(self, ctx: ExitContext) -> Optional[float]:
        for p in self.policies:
            lvl = p.initial_stop(ctx)
            if lvl is not None:
                return lvl
        return None

    def evaluate(self, ctx: ExitContext) -> Optional[ExitDecision]:
        for p in self.policies:
            decision = p.evaluate(ctx)
            if decision is not None:
                return decision
        return None
