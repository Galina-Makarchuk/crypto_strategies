"""Event-driven backtester.

Feeds bars to the strategy one at a time.  The strategy's `on_bar(i, df, state)`
is the only hook — and `i` is the latest known bar.  Look-ahead is prevented by
the on_bar contract (see BaseStrategy.on_bar): strategies may only read
df.iloc[0..i].  The contract is documented, not enforced by the runtime.

After the run, produces a typed summary with P&L stats.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import ExitReason, PositionState, Trade
from .strategies.base import BaseStrategy
from .trade_configurator import ACTIVE_TRADE, TradingConfig, warn_if_inverse_gated

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Summary statistics for a backtest run."""

    strategy_name: str = ""
    symbol: str = ""
    interval: str = ""
    num_bars: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    win_rate: float = 0.0
    total_pnl_bps: float = 0.0
    avg_pnl_bps: float = 0.0
    max_win_bps: float = 0.0
    max_loss_bps: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_bps: float = 0.0
    sharpe_approx: float = 0.0
    # Entries a strategy attempted but the direction/daily-loss gate blocked.
    suppressed_entries: int = 0
    # RISK-mode trades that fell back to fixed-fraction sizing (no entry stop).
    risk_sizing_fallbacks: int = 0
    # Equity layer (additive — driven by TradingConfig sizing). The bps
    # metrics above are unaffected by it.
    initial_equity: float = 0.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    trades: list[Trade] = field(default_factory=list)

    def summary(self) -> str:
        durations = [
            t.duration.total_seconds() / 60
            for t in self.trades
            if t.duration is not None
        ]
        avg_duration_min = sum(durations) / len(durations) if durations else 0.0
        net_profit = self.final_equity - self.initial_equity
        lines = [
            f"{'═' * 60}",
            f"  Backtest Summary: {self.strategy_name}",
            f"  {self.symbol} | {self.interval} | {self.num_bars} bars",
            f"{'═' * 60}",
            f"  Total trades      : {self.total_trades}",
            f"  Avg duration min  : {avg_duration_min:.1f}",
            f"  Suppressed entries : {self.suppressed_entries}  (blocked by direction/daily-loss gate)",
            f"  Win / Loss / BE    : {self.winning_trades} / {self.losing_trades} / {self.break_even_trades}",
            f"  Win rate           : {self.win_rate:.1%}",
            f"  Total P&L (bps)    : {self.total_pnl_bps:+.1f}",
            f"  Avg P&L (bps)      : {self.avg_pnl_bps:+.1f}",
            f"  Max win (bps)      : {self.max_win_bps:+.1f}",
            f"  Max loss (bps)     : {self.max_loss_bps:+.1f}",
            f"  Profit factor      : {self.profit_factor:.2f}",
            f"  Max drawdown (bps) : {self.max_drawdown_bps:.1f}",
            f"  Sharpe (approx)    : {self.sharpe_approx:.2f}",
            f"  {'─' * 40}",
            f"  Initial balance    : ${self.initial_equity:,.2f}",
            f"  Final balance      : ${self.final_equity:,.2f}",
            f"  Net profit         : ${net_profit:+,.2f}",
            f"  Net return         : {self.total_return_pct:+.2f}%",
            f"  Max drawdown       : {self.max_drawdown_pct:.2f}%",
        ]
        if self.risk_sizing_fallbacks:
            lines.append(
                f"  Risk-sizing fallbacks: {self.risk_sizing_fallbacks}  "
                "(no entry stop → fixed-fraction)"
            )
        reasons = Counter(
            t.exit_reason.value for t in self.trades if t.exit_reason is not None
        )
        if reasons:
            lines.append(f"  {'─' * 40}")
            lines.append("  Exits by reason:")
            for reason, count in reasons.most_common():
                lines.append(f"    {reason:<17}: {count}")
        lines.append(f"{'═' * 60}")
        return "\n".join(lines)


class Backtester:
    """Bar-by-bar event-driven backtester."""

    def __init__(
        self,
        strategy: BaseStrategy,
        symbol: str = "BTCUSDT",
        trading_config: TradingConfig | None = None,
    ):
        self.strategy = strategy
        self.symbol = symbol
        # Trade-level policy (costs, sizing, direction, risk overlays). Omitting it
        # falls back to the project-wide ACTIVE_TRADE block (not a bare
        # TradingConfig()), so a caller that doesn't pass one still inherits the
        # user's edited trade defaults instead of silently diverging from them.
        self.trading_config = trading_config or ACTIVE_TRADE

    def _new_state(self) -> PositionState:
        """A PositionState seeded with the run's trade-level policy."""
        tc = self.trading_config
        return PositionState(
            cost_bps=tc.total_cost_bps(),
            allow_long=tc.allows_long(),
            allow_short=tc.allows_short(),
            max_daily_loss_bps=tc.max_daily_loss_bps,
        )

    def run(
        self,
        df: pd.DataFrame,
        interval: str = "15",
        audit_lookahead: bool = False,
    ) -> BacktestResult:
        logger.info("Running backtest: %s on %d bars …", self.strategy.name, len(df))
        warn_if_inverse_gated(self.strategy.name, self.trading_config)

        # Strategies are contractually required to return a fresh DataFrame
        # from prepare() (see BaseStrategy.prepare); no defensive copy here.
        prepared = self.strategy.prepare(df)
        state = self._new_state()
        max_hold = self.trading_config.max_holding_bars

        # The on_bar contract is "may only read df.iloc[0..i]". audit_lookahead
        # *enforces* it: each bar sees a view truncated to [0..i], so reading a
        # future row raises (or changes results) instead of silently peeking.
        # Off by default (slicing per bar has a cost); the no-look-ahead test
        # turns it on and asserts the trades are identical to a normal run.

        # Feed bars one at a time. open_bar tracks the index where the current
        # trade opened, so the max-holding overlay can force a time-stop.
        open_bar: int | None = None
        prev_counter = state._trade_counter
        for i in range(len(prepared)):
            view = prepared.iloc[: i + 1] if audit_lookahead else prepared
            self.strategy.on_bar(i, view, state)

            if state.current_trade is not None:
                if state._trade_counter != prev_counter:  # a new trade just opened
                    open_bar = i
                    prev_counter = state._trade_counter
                if (
                    max_hold is not None
                    and open_bar is not None
                    and (i - open_bar) >= max_hold
                ):
                    state.exit(
                        prepared.index[i],
                        float(prepared["close"].iloc[i]),
                        ExitReason.TIME_STOP,
                    )
                    open_bar = None
            else:
                open_bar = None

        # Force-close any dangling position at last bar
        if state.current_trade is not None:
            state.exit(prepared.index[-1], prepared["close"].iloc[-1], ExitReason.FORCE_CLOSE)
            logger.info("Force-closed open position at end of data.")

        result = self._compute_stats(state, prepared, interval)
        logger.info(
            "Backtest complete: %d trades, total P&L %+.1f bps",
            result.total_trades,
            result.total_pnl_bps,
        )
        if result.suppressed_entries:
            logger.info(
                "%d entries suppressed by the direction/daily-loss gate "
                "(direction=%s) — they are not in the trade count.",
                result.suppressed_entries,
                self.trading_config.direction.value,
            )
        if result.risk_sizing_fallbacks:
            logger.info(
                "%d of %d trades fell back to fixed-fraction sizing (no entry "
                "stop available for RISK mode).",
                result.risk_sizing_fallbacks,
                result.total_trades,
            )
        return result

    def _compute_stats(
        self, state: PositionState, df: pd.DataFrame, interval: str
    ) -> BacktestResult:
        trades = state.closed_trades
        result = BacktestResult(
            strategy_name=self.strategy.name,
            symbol=self.symbol,
            interval=interval,
            num_bars=len(df),
            trades=trades,
            total_trades=len(trades),
            suppressed_entries=state.suppressed_entries,
            initial_equity=self.trading_config.initial_equity,
            final_equity=self.trading_config.initial_equity,
        )

        if not trades:
            return result

        pnls = np.array([t.pnl_bps for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        break_evens = pnls[pnls == 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.break_even_trades = len(break_evens)
        result.win_rate = len(wins) / len(trades)
        result.total_pnl_bps = float(pnls.sum())
        result.avg_pnl_bps = float(pnls.mean())
        result.max_win_bps = float(wins.max()) if len(wins) else 0.0
        result.max_loss_bps = float(losses.min()) if len(losses) else 0.0

        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            result.profit_factor = float("inf")
        # else (all break-evens) leave at 0.0

        # Max drawdown of the cumulative trade-PnL curve, sampled at
        # trade-close points only — intra-trade equity dips are not captured.
        cum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum)
        drawdowns = running_max - cum
        result.max_drawdown_bps = float(drawdowns.max()) if len(drawdowns) else 0.0

        # Approximate Sharpe (trade-level, sample stdev, not time-scaled).
        # When stdev is zero (identical or single trade), fall back to the
        # sign of the mean rather than silently reporting 0.
        if len(pnls) >= 2:
            std = float(pnls.std(ddof=1))
            mean = result.avg_pnl_bps
            if std > 0:
                result.sharpe_approx = mean / std
            elif mean > 0:
                result.sharpe_approx = float("inf")
            elif mean < 0:
                result.sharpe_approx = float("-inf")

        self._apply_equity_layer(trades, result)
        return result

    def _apply_equity_layer(
        self, trades: list[Trade], result: BacktestResult
    ) -> None:
        """Walk closed trades in order, sizing each (fixed-fraction or risk-based)
        and compounding equity. Fills each Trade's currency fields and the
        result's equity metrics. Purely additive — pnl_bps is untouched."""
        tc = self.trading_config
        equity = tc.initial_equity
        peak = equity
        max_dd_pct = 0.0
        fallbacks = 0
        for t in trades:  # single-position ⇒ close order == entry order
            t.notional, fell_back = self._notional_for(t, equity)
            fallbacks += fell_back
            t.pnl_currency = t.notional * (t.pnl_bps / 10_000.0)
            equity += t.pnl_currency
            # Floor a blown account at zero: a loss can't take equity negative
            # (you can't owe more than the account). Without this, a sub-zero
            # equity makes the next trade's notional negative, which flips the
            # sign of subsequent P&L and corrupts the whole curve.
            if equity < 0.0:
                equity = 0.0
            t.equity_after = equity
            peak = max(peak, equity)
            if peak > 0:
                max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100.0)

        result.final_equity = equity
        result.max_drawdown_pct = max_dd_pct
        result.risk_sizing_fallbacks = fallbacks
        if tc.initial_equity > 0:
            result.total_return_pct = (equity / tc.initial_equity - 1.0) * 100.0

    def _notional_for(self, trade: Trade, equity: float) -> tuple[float, bool]:
        """Notional for one trade and whether it fell back to fixed-fraction.
        Delegates to TradingConfig.size_notional so the backtest equity layer and
        the live engine size every trade through the same code path."""
        return self.trading_config.size_notional(
            equity, trade.entry_price, trade.stop_price
        )
