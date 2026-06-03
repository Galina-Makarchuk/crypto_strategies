"""CLI entry point.

Usage:
    python -m engine --strategy supertrend --mode historical --interval 15 --candles 800
    python -m engine --strategy ema --mode live --interval 5 --poll 30
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from .backtester import Backtester
from .data_configurator import DataSpec, load_data, save_result
from .live import LiveEngine
from .models import VALID_CATEGORIES, VALID_INTERVALS, StrategyName
from .strategy_configurator import StrategyConfig
from .trade_configurator import ACTIVE_TRADE, SizingMode, TradeDirection, TradingConfig
from .strategies import AdaptiveSuperTrendStrategy, EMACrossoverStrategy, InverseEMACrossoverStrategy, ExhaustionReversalStrategy, ImpulseFlagStrategy, InverseOrderBlockStrategy, InverseSuperTrendStrategy, MLSwingZigZagStrategy, OrderBlockStrategy, SuperTrendStrategy, SwingBreakoutStrategy, InverseSwingBreakoutStrategy, SwingZigZagStrategy, VWAPBandsStrategy
from .visualization import build_chart

logger = logging.getLogger(__name__)

# Sentinel for trade flags: distinguishes "user didn't pass this flag" from any
# real value, so an omitted flag inherits ACTIVE_TRADE rather than a hardcoded
# default. (None can't serve here — it's a valid value for the optional knobs.)
_UNSET = object()


def _build_strategy(name: str, config: StrategyConfig):
    strategies = {
        StrategyName.SWING.value: SwingBreakoutStrategy,
        StrategyName.SWING_INV.value: InverseSwingBreakoutStrategy,
        StrategyName.EMA_CROSS.value: EMACrossoverStrategy,
        StrategyName.EMA_CROSS_INV.value: InverseEMACrossoverStrategy,
        StrategyName.SUPERTREND.value: SuperTrendStrategy,
        StrategyName.SUPERTREND_INV.value: InverseSuperTrendStrategy,
        StrategyName.SUPERTREND_ADAPTIVE.value: AdaptiveSuperTrendStrategy,
        StrategyName.EXHAUSTION_REVERSAL.value: ExhaustionReversalStrategy,
        StrategyName.IMPULSE_FLAG.value: ImpulseFlagStrategy,
        StrategyName.ORDER_BLOCK.value: OrderBlockStrategy,
        StrategyName.ORDER_BLOCK_INV.value: InverseOrderBlockStrategy,
        StrategyName.VWAP_BANDS.value: VWAPBandsStrategy,
        StrategyName.SWING_ZIGZAG.value: SwingZigZagStrategy,
        StrategyName.SWING_ZIGZAG_ML.value: MLSwingZigZagStrategy,
    }
    cls = strategies.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(strategies.keys())}")
    return cls(config)


def _build_trading_config(args) -> TradingConfig:
    """Start from ACTIVE_TRADE and override only the trade flags the user
    explicitly passed (anything still ``_UNSET`` inherits ACTIVE_TRADE)."""
    overrides = {}
    for field in (
        "initial_equity", "position_size_bps", "leverage", "risk_per_trade_bps",
        "fee_bps", "slippage_bps", "max_daily_loss_bps", "max_holding_bars",
    ):
        val = getattr(args, field)
        if val is not _UNSET:
            overrides[field] = val
    if args.direction is not _UNSET:
        overrides["direction"] = TradeDirection(args.direction)
    if args.sizing_mode is not _UNSET:
        overrides["sizing_mode"] = SizingMode(args.sizing_mode)
    return dataclasses.replace(ACTIVE_TRADE, **overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bybit BTCUSDT Professional Trading Strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in StrategyName],
        default="supertrend",
        help="Strategy to run (default: supertrend)",
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "live"],
        default="historical",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair (default: BTCUSDT)",
    )
    parser.add_argument(
        "--interval",
        default="15",
        choices=sorted(VALID_INTERVALS),
        help="Candle interval (default: 15)",
    )
    parser.add_argument(
        "--candles",
        type=int,
        default=800,
        help="Number of historical candles (default: 800; ignored when --start is set)",
    )
    parser.add_argument(
        "--category",
        default="linear",
        choices=sorted(VALID_CATEGORIES),
        help="Bybit product type (default: linear)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Historical range start, ISO e.g. 2026-03-20 (range mode; --candles ignored)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Historical range end, ISO (defaults to now)",
    )
    parser.add_argument(
        "--save",
        default="chart.html",
        help="Save chart to file (default: chart.html)",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=30,
        help="Live mode poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--db",
        default="trading_state.db",
        help="SQLite database path for live state (default: trading_state.db)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Use structured JSON logging",
    )

    # ── Trade-level parameters (TradingConfig) ─────────────────────────────
    # Defaults are the ACTIVE_TRADE block in trade_configurator.py — a flag only
    # overrides ACTIVE_TRADE when explicitly passed (see _UNSET). This keeps
    # ACTIVE_TRADE the single source of truth across notebooks AND the CLI.
    trade = parser.add_argument_group(
        "trade parameters", "Omitted flags inherit the ACTIVE_TRADE block."
    )
    trade.add_argument(
        "--initial-equity", type=float, default=_UNSET,
        help="Starting account equity in quote ccy",
    )
    trade.add_argument(
        "--position-size-bps", type=float, default=_UNSET,
        help="Notional per trade as bps of equity, 10000=100%%",
    )
    trade.add_argument(
        "--leverage", type=float, default=_UNSET,
        help="Leverage multiplier on notional",
    )
    trade.add_argument(
        "--fee-bps", type=float, default=_UNSET,
        help="Taker fee per side in bps; Bybit 0.04%%=4",
    )
    trade.add_argument(
        "--slippage-bps", type=float, default=_UNSET,
        help="Estimated slippage per side in bps",
    )
    trade.add_argument(
        "--max-daily-loss-bps", type=float, default=_UNSET,
        help="Halt entries after this realized loss (bps) in a UTC day",
    )
    trade.add_argument(
        "--max-holding-bars", type=int, default=_UNSET,
        help="Force-close a trade after this many bars",
    )
    trade.add_argument(
        "--direction", choices=[d.value for d in TradeDirection], default=_UNSET,
        help="Allowed trade sides (long/short/both)",
    )
    trade.add_argument(
        "--sizing-mode", choices=[m.value for m in SizingMode], default=_UNSET,
        help="fixed = position_size_bps; risk = risk_per_trade_bps (stop-where-available)",
    )
    trade.add_argument(
        "--risk-per-trade-bps", type=float, default=_UNSET,
        help="Equity risked per trade in risk mode, bps (100 = 1%%)",
    )

    args = parser.parse_args(argv)

    # ── Logging setup (only at entry point, not module level) ──────────
    if args.log_json:
        import json

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps({
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                })

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.root.handlers = [handler]
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    config = StrategyConfig()
    strategy = _build_strategy(args.strategy, config)
    trading_config = _build_trading_config(args)

    if args.mode == "historical":
        return _run_historical(strategy, trading_config, args)
    else:
        return _run_live(strategy, trading_config, args)


def _run_historical(strategy, trading_config, args) -> int:
    # Pull candles through the shared cache (data/ohlcv/…) — the same source of
    # truth notebooks use, so repeat runs are instant and datasets stay identical.
    spec = DataSpec(
        symbol=args.symbol,
        interval=args.interval,
        category=args.category,
        num_candles=args.candles,
        start=args.start,
        end=args.end,
    )
    df = load_data(spec)

    bt = Backtester(strategy, symbol=args.symbol, trading_config=trading_config)
    result = bt.run(df, interval=args.interval)

    # Print summary
    print(result.summary())

    # Persist results (metrics JSON + trades CSV), grouped by dataset signature.
    json_path = save_result(result, spec)
    print(f"Results saved to {json_path.parent}")

    # Build chart with indicator columns from prepare()
    prepared = strategy.prepare(df)
    signals = [s for t in result.trades for s in _trade_to_signals(t)]
    build_chart(
        prepared,
        signals,
        title=f"{args.symbol} {args.interval} | {strategy.name} | {result.total_trades} trades",
        save_path=args.save,
    )

    return 0


def _run_live(strategy, trading_config, args) -> int:
    engine = LiveEngine(
        strategy=strategy,
        symbol=args.symbol,
        interval=args.interval,
        category=args.category,
        num_candles=args.candles,
        poll_seconds=args.poll,
        chart_path=args.save,
        db_path=args.db,
        trading_config=trading_config,
    )
    engine.run()
    return 0


def _trade_to_signals(trade):
    """Convert a Trade object back to Signal-like dicts for the chart."""
    from .models import Signal, SignalAction

    signals = []
    if trade.entry_ts:
        signals.append(Signal(
            timestamp=trade.entry_ts,
            action=SignalAction.ENTRY,
            direction=trade.direction,
            price=trade.entry_price,
            label=f"{trade.direction.value.capitalize()} Entry",
        ))
    if trade.exit_ts:
        signals.append(Signal(
            timestamp=trade.exit_ts,
            action=SignalAction.EXIT,
            direction=trade.direction,
            price=trade.exit_price,
            label=f"{trade.direction.value.capitalize()} Exit",
        ))
    return signals


if __name__ == "__main__":
    sys.exit(main())
