"""Live trading loop.

Production features:
- Graceful shutdown on SIGTERM / SIGINT
- Circuit breaker: stops after N consecutive fetch failures
- State persistence: survives restart without losing position
- Incremental candle update (full refetch with dedup for simplicity & safety)
- Writes chart to a single file (no browser-tab spam)
- Bounded signal list (rolling window)
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .fetcher import BybitFetcher
from .models import PositionState, StrategyConfig, validate_interval
from .persistence import StateStore
from .strategies.base import BaseStrategy
from .visualization import build_chart

logger = logging.getLogger(__name__)

_MAX_SIGNALS_KEPT = 500  # rolling window size
_CIRCUIT_BREAKER_LIMIT = 10  # consecutive failures before stopping


class LiveEngine:
    """Runs a strategy in a poll loop against real-time Bybit data."""

    def __init__(
        self,
        strategy: BaseStrategy,
        symbol: str = "BTCUSDT",
        interval: str = "15",
        num_candles: int = 500,
        poll_seconds: int = 30,
        chart_path: str = "live_chart.html",
        db_path: str = "trading_state.db",
    ):
        validate_interval(interval)
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.num_candles = num_candles
        self.poll_seconds = poll_seconds
        self.chart_path = chart_path

        self._fetcher = BybitFetcher()
        self._store = StateStore(db_path)
        self._state: PositionState = self._store.load_state()
        self._running = True
        self._consecutive_failures = 0

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down gracefully …", sig_name)
        self._running = False

    def run(self) -> None:
        logger.info(
            "LIVE MODE started | %s %s | %s | poll=%ds",
            self.symbol,
            self.interval,
            self.strategy.name,
            self.poll_seconds,
        )

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

    def _tick(self) -> None:
        """Single iteration of the live loop."""
        df = self._fetcher.fetch_klines(
            symbol=self.symbol,
            interval=self.interval,
            num_candles=self.num_candles,
        )

        # Prepare indicators
        prepared = self.strategy.prepare(df)

        # Record state before
        prev_trade_count = self._state._trade_counter
        prev_signals_count = len(self._state.signals)

        # Evaluate only the LAST bar (live mode)
        last_bar = len(prepared) - 1
        if self._state.current_trade is not None:
            self._state.update_peak(
                prepared["high"].iloc[last_bar],
                prepared["low"].iloc[last_bar],
            )
        self.strategy.on_bar(last_bar, prepared, self._state)

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

        # Update chart (single file, overwritten each tick)
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        build_chart(
            prepared,
            self._state.signals,
            title=f"LIVE {self.symbol} {self.interval} | {self.strategy.name} | {now_str}",
            save_path=self.chart_path,
            auto_refresh=self.poll_seconds,
        )

    def _cleanup(self) -> None:
        logger.info("Cleaning up …")
        self._store.save_state(self._state)
        self._store.close()
        self._fetcher.close()
        logger.info("Live engine stopped.")
