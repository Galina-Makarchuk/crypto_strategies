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

from .models import ExitReason, PositionState, Trade
from .strategies.base import BaseStrategy

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
    trades: list[Trade] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'═' * 60}",
            f"  Backtest Summary: {self.strategy_name}",
            f"  {self.symbol} | {self.interval} | {self.num_bars} bars",
            f"{'═' * 60}",
            f"  Total trades      : {self.total_trades}",
            f"  Win / Loss / BE    : {self.winning_trades} / {self.losing_trades} / {self.break_even_trades}",
            f"  Win rate           : {self.win_rate:.1%}",
            f"  Total P&L (bps)    : {self.total_pnl_bps:+.1f}",
            f"  Avg P&L (bps)      : {self.avg_pnl_bps:+.1f}",
            f"  Max win (bps)      : {self.max_win_bps:+.1f}",
            f"  Max loss (bps)     : {self.max_loss_bps:+.1f}",
            f"  Profit factor      : {self.profit_factor:.2f}",
            f"  Max drawdown (bps) : {self.max_drawdown_bps:.1f}",
            f"  Sharpe (approx)    : {self.sharpe_approx:.2f}",
        ]
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

    def __init__(self, strategy: BaseStrategy, symbol: str = "BTCUSDT"):
        self.strategy = strategy
        self.symbol = symbol

    def run(self, df: pd.DataFrame, interval: str = "15") -> BacktestResult:
        logger.info("Running backtest: %s on %d bars …", self.strategy.name, len(df))

        # Strategies are contractually required to return a fresh DataFrame
        # from prepare() (see BaseStrategy.prepare); no defensive copy here.
        prepared = self.strategy.prepare(df)
        state = PositionState()

        # Feed bars one at a time
        for i in range(len(prepared)):
            self.strategy.on_bar(i, prepared, state)

        # Force-close any dangling position at last bar
        if state.current_trade is not None:
            cost = self.strategy.config.total_cost_bps()
            state.exit(prepared.index[-1], prepared["close"].iloc[-1], cost, ExitReason.FORCE_CLOSE)
            logger.info("Force-closed open position at end of data.")

        result = self._compute_stats(state, prepared, interval)
        logger.info(
            "Backtest complete: %d trades, total P&L %+.1f bps",
            result.total_trades,
            result.total_pnl_bps,
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

        return result
