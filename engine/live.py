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
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path


from .core import ExitReason, PositionState, Trade
from .notify import NotifyEvent, Notifier, in_jupyter, make_notifiers
from .providers import make_provider, resolve_category, validate_spec
from .live_records import LiveRecords
from .strategies.base import BaseStrategy
from .trade_configurator import ACTIVE_TRADE, TradingConfig, warn_if_inverse_gated
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
        category: str | None = "linear",
        provider: str = "bybit",
        num_candles: int = 500,
        poll_seconds: int = 30,
        chart_path: str = "live_chart.html",
        db_path: str = "trading_state.db",
        trading_config: TradingConfig | None = None,
        notifiers: str | Sequence[Notifier] | None = None,
    ):
        validate_spec(provider, interval, category)
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.category = category
        self.provider = provider
        # The category actually sent to the provider — None for providers with no
        # product taxonomy (e.g. yahoo); "linear"/"inverse" for bybit.
        self._fetch_category = resolve_category(provider, category)
        self.num_candles = num_candles
        self.poll_seconds = poll_seconds
        self.chart_path = chart_path
        # Omitting trading_config inherits the project-wide ACTIVE_TRADE block (not
        # a bare TradingConfig()), so live runs use the same trade defaults as the
        # CLI and notebooks instead of silently diverging from them.
        self.trading_config = trading_config or ACTIVE_TRADE

        self._fetcher = make_provider(provider)
        self._store = LiveRecords(db_path)
        self._state: PositionState = self._store.load_state()
        self._seed_state_policy(self._state)
        self._register_shutdown_handlers()
        # Paper-equity layer: a continuous, restart-safe simulated equity curve
        # (the live engine places no real orders — this forward-tests the same
        # sizing the backtester uses). Seeded from initial_equity on first run.
        self._equity: float = self._store.load_equity(
            default=self.trading_config.initial_equity
        )
        self._sized_trade_ids: set[str] = self._store.known_trade_ids()
        self._running = True
        self._consecutive_failures = 0

        # Signal alerts (live only; places no orders). A string spec is resolved
        # here so notebooks/CLI can pass notifiers="browser,desktop"; a sequence
        # of Notifier objects (tests, advanced use) is taken as-is.
        if isinstance(notifiers, str):
            notifiers = make_notifiers(notifiers)
        self._notifiers: list[Notifier] = list(notifiers) if notifiers else []
        # Dedup across the per-tick replay (which re-derives the whole window):
        # the first tick primes these with everything already in the window, so
        # historical trades and a restart-recovered position never alert. Only
        # signals that fire on later ticks notify.
        self._notified_entry_ids: set[str] = set()
        self._notified_exit_ids: set[str] = set()
        self._notify_primed = False

    def _seed_state_policy(self, state: PositionState) -> None:
        """Apply trade-level policy (cost, direction gate, daily-loss cap) onto a
        loaded/fresh PositionState. The store persists only position data, not
        config, so this re-seeds it from trading_config every run."""
        tc = self.trading_config
        state.cost_bps = tc.total_cost_bps()
        state.allow_long = tc.allows_long()
        state.allow_short = tc.allows_short()
        state.max_daily_loss_bps = tc.max_daily_loss_bps

    def _register_shutdown_handlers(self) -> None:
        """Register SIGINT / SIGTERM handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down gracefully …", sig_name)
        self._running = False

    def _market_open(self) -> bool:
        """Whether the venue is currently tradeable. A provider may expose an
        is_market_open() hook (session/holiday calendar); absent one, assume
        always-open — correct for 24/7 crypto, and a safe default elsewhere."""
        hook = getattr(self._fetcher, "is_market_open", None)
        try:
            return bool(hook()) if callable(hook) else True
        except Exception:  # noqa: BLE001 — a flaky hook must not stall the loop
            return True

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
        if self._notifiers:
            logger.info(
                "Signal alerts: %s",
                ", ".join(type(n).__name__ for n in self._notifiers),
            )
        self._announce_chart()

        while self._running:
            if not self._market_open():
                # Crypto is 24/7; non-crypto venues have sessions/holidays. When a
                # provider reports closed, sleep instead of fetching, so off-hours
                # empty fetches don't trip the circuit breaker.
                logger.info(
                    "Market closed for %s (%s) — sleeping %ds.",
                    self.symbol, self.provider, self.poll_seconds,
                )
                time.sleep(self.poll_seconds)
                continue
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
        if in_jupyter():
            from IPython.display import display, HTML  # noqa: PLC0415 (optional dep, lazy)
            display(HTML(
                f'<a href="{uri}" target="_blank" rel="noopener">'
                f'Open live chart — {label} ↗</a>'
            ))
        else:
            logger.info("Live chart → %s", uri)

    def _tick(self) -> None:
        """Single iteration of the live loop.

        The position is rebuilt by **replaying the strategy over the whole candle
        window from a fresh state** each tick — the same bar-by-bar loop the
        backtester runs (minus the end-of-data force-close) — rather than nudging
        a persisted position by the last bar alone. ``prepare()`` resets the
        strategy's per-trade scratchpad and this loop rebuilds it bar-by-bar, so
        the open position is always managed with correct internal state and
        multi-bar setups accumulate. That makes crash recovery automatic (the
        window deterministically re-derives the open position) and removes the
        restart-amnesia bugs where a restored trade met a wiped scratchpad.
        """
        df = self._fetcher.fetch_klines(
            symbol=self.symbol,
            interval=self.interval,
            num_candles=self.num_candles,
            category=self._fetch_category,
        )

        # Bybit returns the still-forming current candle as the most recent row.
        # Acting on it repaints — its high/low/close keep changing intra-bar, so
        # an entry/exit can fire on a wick the bar later erases. Drop it and
        # evaluate only *closed* bars, the same data a backtest sees.
        if len(df) > 0:
            df = df.iloc[:-1]
        if len(df) == 0:
            logger.warning("No closed candles this tick (only a forming bar) — skipping.")
            return

        prev_open = self._state.current_trade
        prepared = self.strategy.prepare(df)
        state = self._replay(prepared)
        self._state = state

        # A position that was open before but is no longer derivable from the
        # window (its entry scrolled off the num_candles window) can't be rebuilt
        # — surface it rather than silently dropping the trade.
        if (
            prev_open is not None
            and prev_open.entry_ts is not None
            and prev_open.entry_ts < prepared.index[0]
            and (state.current_trade is None
                 or state.current_trade.entry_ts != prev_open.entry_ts)
        ):
            logger.warning(
                "Open %s position entered %s has scrolled off the %d-bar window "
                "and can no longer be reconstructed — increase num_candles to keep "
                "long holds recoverable.",
                prev_open.direction.value, prev_open.entry_ts.isoformat(),
                self.num_candles,
            )

        # Size + persist any trade newly seen as closed (dedup by stable id), and
        # compound the paper-equity curve. Only *new* closures are written, so the
        # per-tick write volume is bounded (no full-history re-save).
        self._apply_equity_to_new_closures()

        # Alert on signals that are new this tick (runs after sizing so exit
        # alerts carry pnl_currency / equity_after). No-op if no notifiers.
        if self._notifiers:
            self._emit_notifications(state)

        if state.suppressed_entries:
            logger.info(
                "Entry suppressed by trade policy (direction=%s) — not opened.",
                self.trading_config.direction.value,
            )

        # Trim signals to bounded window
        if len(state.signals) > _MAX_SIGNALS_KEPT:
            state.signals = state.signals[-_MAX_SIGNALS_KEPT:]

        # Persist the current open-position snapshot (single row) for restart
        # display / reporting; closed trades are persisted in the equity pass.
        self._store.save_state(state)
        logger.info(
            "State updated | pos=%s | closed=%d | last_close=%.2f",
            state.status.name,
            len(state.closed_trades),
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

    def _fresh_state(self) -> PositionState:
        """A new PositionState seeded with the run's trade-level policy (cost,
        direction gate, daily-loss cap) — the live analogue of
        Backtester._new_state, used to replay the window from scratch each tick."""
        tc = self.trading_config
        return PositionState(
            cost_bps=tc.total_cost_bps(),
            allow_long=tc.allows_long(),
            allow_short=tc.allows_short(),
            max_daily_loss_bps=tc.max_daily_loss_bps,
        )

    def _replay(self, prepared) -> PositionState:
        """Rebuild the canonical position by running the strategy over every bar
        of the window from a fresh state — identical to Backtester.run's loop
        (including the max-holding overlay), but WITHOUT the end-of-data
        force-close: the final bar's position is the live OPEN position, not a
        closed trade. Because prepare() reset the scratchpad and this loop walks
        all bars, the strategy's internal state is correctly rebuilt for the open
        position (no restart amnesia) and multi-bar setups accumulate."""
        state = self._fresh_state()
        max_hold = self.trading_config.max_holding_bars
        open_bar: int | None = None
        prev_counter = state._trade_counter
        for i in range(len(prepared)):
            view = prepared.iloc[: i + 1]
            self.strategy.on_bar(i, view, state)

            if state.current_trade is not None:
                if state._trade_counter != prev_counter:   # a new trade just opened
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
        return state

    def _apply_equity_to_new_closures(self) -> None:
        """Size + persist each newly closed trade and compound the paper-equity curve.

        Mirrors the backtester's post-run equity layer, applied incrementally as
        the replay surfaces closures. Each closed trade is processed exactly once
        — the stable-id known set, loaded from the store, makes this idempotent
        across ticks and restarts even though the replay re-derives the full
        window's trades every tick — via the same TradingConfig.size_notional the
        backtester uses, so FIXED/RISK sizing matches. Only *new* closures are
        written to trade_history (one upsert each), so per-tick write volume is
        bounded by how many trades closed since the last tick — not the full
        history (audit M3)."""
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
            self._store.save_trade(trade)   # persist this newly-closed trade once
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

    def _emit_notifications(self, state: PositionState) -> None:
        """Fire entry/exit alerts for signals that are NEW since the last tick.

        The first tick after (re)start is a priming pass: every trade already in
        the candle window — and a position carried across a restart — is recorded
        as already-seen and alerts nothing, so there's never a startup
        notification storm. From the second tick on, a closure or a fresh open
        not seen before alerts exactly once (deduped by the stable trade id, the
        same id the equity layer uses). A signal that fires while the engine is
        down is persisted to the DB/chart but not alerted — it's history by the
        time the loop resumes."""
        closed = [t for t in state.closed_trades if t.is_closed]
        open_trade = state.current_trade

        if not self._notify_primed:
            self._notified_entry_ids.update(t.trade_id for t in closed)
            self._notified_exit_ids.update(t.trade_id for t in closed)
            if open_trade is not None:
                self._notified_entry_ids.add(open_trade.trade_id)
            self._notify_primed = True
            return

        # A trade is normally seen open first (entry alert via the open-trade
        # branch below) and closed on a later tick (exit alert here). But if all
        # of a trade's bars close between two notification ticks — e.g. after a
        # circuit-breaker backoff, or a poll interval >= the bar interval — the
        # engine first sees it already closed: emit its entry alert here too, and
        # before the exit, so the pair still reads entry → exit in order.
        for trade in closed:
            if trade.trade_id not in self._notified_entry_ids:
                self._notified_entry_ids.add(trade.trade_id)
                self._notify(self._entry_event(trade))
            if trade.trade_id not in self._notified_exit_ids:
                self._notified_exit_ids.add(trade.trade_id)
                self._notify(self._exit_event(trade))

        if (
            open_trade is not None
            and open_trade.trade_id not in self._notified_entry_ids
        ):
            self._notified_entry_ids.add(open_trade.trade_id)
            self._notify(self._entry_event(open_trade))

    def _entry_event(self, trade: Trade) -> NotifyEvent:
        return NotifyEvent(
            kind="entry",
            strategy=self.strategy.name,
            symbol=self.symbol,
            interval=self.interval,
            direction=trade.direction.value,
            price=trade.entry_price,
            timestamp=trade.entry_ts,
            label=f"{trade.direction.value.capitalize()} entry",
        )

    def _exit_event(self, trade: Trade) -> NotifyEvent:
        return NotifyEvent(
            kind="exit",
            strategy=self.strategy.name,
            symbol=self.symbol,
            interval=self.interval,
            direction=trade.direction.value,
            price=trade.exit_price,
            timestamp=trade.exit_ts,
            label=f"{trade.direction.value.capitalize()} exit",
            exit_reason=trade.exit_reason.value if trade.exit_reason else None,
            pnl_bps=trade.pnl_bps,
            pnl_currency=trade.pnl_currency,
            equity_after=trade.equity_after,
        )

    def _notify(self, event: NotifyEvent) -> None:
        """Dispatch one event to every notifier. A notifier that raises is logged
        and skipped, so a flaky alert channel can never break the trading loop or
        trip the circuit breaker."""
        for notifier in self._notifiers:
            try:
                notifier.notify(event)
            except Exception:  # noqa: BLE001 — alerts must never break the loop
                logger.exception("Notifier %s failed", type(notifier).__name__)

    def _cleanup(self) -> None:
        logger.info("Cleaning up …")
        self._store.save_state(self._state)
        self._store.close()
        self._fetcher.close()
        logger.info("Live engine stopped.")
