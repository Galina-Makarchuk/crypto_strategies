"""Live trading loop.

Production features:
- Graceful shutdown on SIGTERM / SIGINT
- Circuit breaker: stops after N consecutive fetch failures
- Crash recovery: position + trade history persist across restarts
- Full candle refetch each tick (the still-forming last bar is dropped); no
  incremental merge — simpler and safe against gaps
- Writes chart to a single file (no browser-tab spam)
- Bounded signal list (rolling window)
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


from .fetcher import BybitFetcher
from .core import ExitReason, PositionState, Trade, validate_category, validate_interval
from .live_records import LiveRecords
from .strategies.base import BaseStrategy
from .trade_configurator import ACTIVE_TRADE, TradingConfig, warn_if_inverse_gated
from .visualization import build_chart

logger = logging.getLogger(__name__)

_MAX_SIGNALS_KEPT = 500  # rolling window size
_CIRCUIT_BREAKER_LIMIT = 10  # consecutive failures before stopping


def _in_jupyter() -> bool:
    """True only inside a Jupyter/VS Code kernel (ZMQInteractiveShell), where a
    rich HTML link can be rendered. Terminal IPython and plain scripts → False."""
    try:
        from IPython import get_ipython  # noqa: PLC0415 (optional dep, lazy)
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False

# Bar length in minutes, used to convert a max-holding-bars overlay into a time
# delta in live mode (mirrors data_configurator._INTERVAL_MINUTES).
_INTERVAL_MINUTES: dict[str, int] = {
    "1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60,
    "120": 120, "240": 240, "360": 360, "720": 720,
    "D": 1440, "W": 10080, "M": 43200,
}


class LiveEngine:
    """Runs a strategy in a poll loop against real-time Bybit data."""

    def __init__(
        self,
        strategy: BaseStrategy,
        symbol: str = "BTCUSDT",
        interval: str = "15",
        category: str = "linear",
        num_candles: int = 500,
        poll_seconds: int = 30,
        chart_path: str = "live_chart.html",
        db_path: str = "trading_state.db",
        trading_config: TradingConfig | None = None,
    ):
        validate_interval(interval)
        validate_category(category)
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.category = category
        self.num_candles = num_candles
        self.poll_seconds = poll_seconds
        self.chart_path = chart_path
        # Omitting trading_config inherits the project-wide ACTIVE_TRADE block (not
        # a bare TradingConfig()), so live runs use the same trade defaults as the
        # CLI and notebooks instead of silently diverging from them.
        self.trading_config = trading_config or ACTIVE_TRADE

        self._fetcher = BybitFetcher()
        self._store = LiveRecords(db_path)
        self._state: PositionState = self._store.load_state()
        self._seed_state_policy(self._state)
        # Paper-equity layer: a continuous, restart-safe simulated equity curve
        # (the live engine places no real orders — this forward-tests the same
        # sizing the backtester uses). Seeded from initial_equity on first run.
        self._equity: float = self._store.load_equity(
            default=self.trading_config.initial_equity
        )
        self._sized_trade_ids: set[str] = self._store.known_trade_ids()
        self._running = True
        self._consecutive_failures = 0

    def _seed_state_policy(self, state: PositionState) -> None:
        """Apply trade-level policy (cost, direction gate, daily-loss cap) onto a
        loaded/fresh PositionState. The store persists only position data, not
        config, so this re-seeds it from trading_config every run."""
        tc = self.trading_config
        state.cost_bps = tc.total_cost_bps()
        state.allow_long = tc.allows_long()
        state.allow_short = tc.allows_short()
        state.max_daily_loss_bps = tc.max_daily_loss_bps

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down gracefully …", sig_name)
        self._running = False

    def run(self) -> None:
        logger.info(
            "LIVE MODE started | %s %s %s | %s | poll=%ds",
            self.category,
            self.symbol,
            self.interval,
            self.strategy.name,
            self.poll_seconds,
        )
        warn_if_inverse_gated(self.strategy.name, self.trading_config)
        self._announce_chart()

        while self._running:
            try:
                self._tick()
                self._consecutive_failures = 0
            except KeyboardInterrupt:
                break
            except Exception:
                self._consecutive_failures += 1
                logger.exception(
                    "Tick failed (%d/%d consecutive failures)",
                    self._consecutive_failures,
                    _CIRCUIT_BREAKER_LIMIT,
                )
                if self._consecutive_failures >= _CIRCUIT_BREAKER_LIMIT:
                    logger.critical(
                        "Circuit breaker tripped after %d consecutive failures — stopping.",
                        _CIRCUIT_BREAKER_LIMIT,
                    )
                    break
                # Exponential backoff: 2^failures seconds, capped at 5 min
                backoff = min(2 ** self._consecutive_failures, 300)
                logger.info("Backing off for %ds …", backoff)
                time.sleep(backoff)
                continue

            # Normal sleep between ticks
            time.sleep(self.poll_seconds)

        self._cleanup()

    def _announce_chart(self) -> None:
        """Surface where the live chart lives so it's easy to find: a clickable
        link inside a notebook, otherwise a ``file://`` URL in the logs (CLI).
        The chart auto-refreshes every ``poll_seconds`` once the first tick
        writes it."""
        uri = Path(self.chart_path).resolve().as_uri()
        label = f"{self.symbol} {self.interval}m {self.strategy.name}"
        if _in_jupyter():
            from IPython.display import display, HTML  # noqa: PLC0415 (optional dep, lazy)
            display(HTML(
                f'<a href="{uri}" target="_blank" rel="noopener">'
                f'Open live chart — {label} ↗</a>'
            ))
        else:
            logger.info("Live chart → %s", uri)

    def _tick(self) -> None:
        """Single iteration of the live loop."""
        df = self._fetcher.fetch_klines(
            symbol=self.symbol,
            interval=self.interval,
            num_candles=self.num_candles,
            category=self.category,
        )

        # Bybit returns the still-forming current candle as the most recent row.
        # Acting on it repaints — its high/low/close keep changing intra-bar, so
        # an entry/exit can fire on a wick the bar later erases. Drop it and
        # evaluate only the last *closed* bar, the same data a backtest sees.
        if len(df) > 0:
            df = df.iloc[:-1]
        if len(df) == 0:
            logger.warning("No closed candles this tick (only a forming bar) — skipping.")
            return

        # Prepare indicators
        prepared = self.strategy.prepare(df)

        # Record state before
        prev_trade_count = self._state._trade_counter
        prev_signals_count = len(self._state.signals)
        prev_suppressed = self._state.suppressed_entries

        # Evaluate only the LAST bar (live mode)
        last_bar = len(prepared) - 1
        if self._state.current_trade is not None:
            self._state.update_peak(
                prepared["high"].iloc[last_bar],
                prepared["low"].iloc[last_bar],
            )
        self.strategy.on_bar(last_bar, prepared, self._state)

        # max-holding overlay: if the strategy didn't exit, force a time-stop
        # once the position has been held for the configured number of bars.
        self._enforce_max_holding(prepared.index[last_bar], float(prepared["close"].iloc[last_bar]))

        # Size any trade that just closed and compound the paper-equity curve.
        self._apply_equity_to_new_closures()

        # Make a gated entry observable (otherwise a blocked signal looks like
        # the strategy simply did nothing this bar).
        if self._state.suppressed_entries != prev_suppressed:
            logger.info(
                "Entry suppressed by trade policy (direction=%s) — not opened.",
                self.trading_config.direction.value,
            )

        # Trim signals to bounded window
        if len(self._state.signals) > _MAX_SIGNALS_KEPT:
            self._state.signals = self._state.signals[-_MAX_SIGNALS_KEPT:]

        # Persist if anything changed
        if (
            self._state._trade_counter != prev_trade_count
            or len(self._state.signals) != prev_signals_count
        ):
            self._store.save_state(self._state)
            # Save completed trades
            for trade in self._state.closed_trades:
                if trade.is_closed:
                    self._store.save_trade(trade)
            logger.info(
                "State updated | pos=%s | trades=%d | last_close=%.2f",
                self._state.status.name,
                self._state._trade_counter,
                prepared["close"].iloc[-1],
            )

        # Update chart (single file, overwritten each tick). Use the rich trade
        # view: closed trades coloured by exit reason with bps + $ P&L on hover and
        # entry→exit path lines, plus the open position's entry marker.
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        build_chart(
            prepared,
            trades=self._chart_trades(prepared.index[0]),
            title=f"LIVE {self.symbol} {self.interval} | {self.strategy.name} | {now_str}",
            save_path=self.chart_path,
            auto_refresh=self.poll_seconds,
        )

    def _chart_trades(self, window_start) -> list[Trade]:
        """Trades to draw on the live chart: every closed trade whose exit falls in
        the visible candle window, plus the currently open trade (if its entry is in
        view). The open trade has no exit_ts, so the trade view renders only its
        entry marker — no exit/path — which is exactly the live 'you are here' cue.
        Filtering to the window keeps the x-axis pinned to the candles instead of
        stretching to an old marker."""
        trades = [
            t for t in self._state.closed_trades
            if t.exit_ts is not None and t.exit_ts >= window_start
        ]
        open_trade = self._state.current_trade
        if (
            open_trade is not None
            and open_trade.entry_ts is not None
            and open_trade.entry_ts >= window_start
        ):
            trades.append(open_trade)
        return trades

    def _apply_equity_to_new_closures(self) -> None:
        """Size each newly closed trade and compound the paper-equity curve.

        Mirrors the backtester's post-run equity layer, applied incrementally
        (live closes one trade at a time). Each closed trade is sized exactly once
        — the known-id set, loaded from the store, makes this idempotent across
        restarts — via the same TradingConfig.size_notional the backtester uses, so
        FIXED/RISK sizing matches. Fills the trade's currency fields and persists
        the running equity; the existing _tick persistence block then writes the
        trade (with those fields) to trade_history."""
        tc = self.trading_config
        for trade in self._state.closed_trades:
            if not trade.is_closed or trade.trade_id in self._sized_trade_ids:
                continue
            notional, fell_back = tc.size_notional(
                self._equity, trade.entry_price, trade.stop_price
            )
            trade.notional = notional
            trade.pnl_currency = notional * (trade.pnl_bps / 10_000.0)
            # Floor a blown account at zero (same as the backtester): equity can't
            # go negative, and a negative notional would flip later P&L signs.
            self._equity = max(0.0, self._equity + trade.pnl_currency)
            trade.equity_after = self._equity
            self._sized_trade_ids.add(trade.trade_id)
            self._store.save_equity(self._equity)
            if fell_back:
                logger.info(
                    "Risk-sizing fallback for trade %s (no entry stop) — "
                    "fixed-fraction notional used.",
                    trade.trade_id,
                )
            logger.info(
                "Trade %s closed | pnl %+.1f bps | notional %.2f | "
                "pnl %+.2f | equity %.2f",
                trade.trade_id,
                trade.pnl_bps,
                notional,
                trade.pnl_currency,
                self._equity,
            )

    def _enforce_max_holding(self, bar_ts, close: float) -> None:
        """Force a time-stop when an open trade has been held for at least
        max_holding_bars. Bars held are derived from the entry timestamp and the
        bar interval (live evaluates only the latest bar, so there's no bar
        counter to lean on)."""
        max_hold = self.trading_config.max_holding_bars
        trade = self._state.current_trade
        if max_hold is None or trade is None or trade.entry_ts is None:
            return
        minutes = _INTERVAL_MINUTES[self.interval]
        # Integer bar count between entry and this (closed) bar — both timestamps
        # are bar-open times aligned to the interval, so round() is exact and
        # mirrors the backtester's `(i - open_bar) >= max_hold`.
        bars_held = round((bar_ts - trade.entry_ts).total_seconds() / (minutes * 60))
        if bars_held >= max_hold:
            self._state.exit(bar_ts, close, ExitReason.TIME_STOP)
            logger.info("Max-holding time-stop after %d bars.", bars_held)

    def _cleanup(self) -> None:
        logger.info("Cleaning up …")
        self._store.save_state(self._state)
        self._store.close()
        self._fetcher.close()
        logger.info("Live engine stopped.")
