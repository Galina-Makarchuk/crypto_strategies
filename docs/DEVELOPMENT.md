# engine v2.0

Production-grade crypto trading strategy framework for Bybit perpetuals.

## Quick Start

```bash
pip install -r requirements.txt

# Historical backtest (SuperTrend, 15m, 800 candles)
python -m engine --strategy supertrend --interval 15 --candles 800

# EMA crossover backtest
python -m engine --strategy ema --interval 60 --candles 1000 --save ema_chart.html

# Live mode with persisted live records
python -m engine --strategy supertrend --mode live --interval 5 --poll 30

# Structured JSON logging (for log shippers)
python -m engine --strategy ema --log-json --log-level DEBUG
```

## Architecture

```
engine/
├── core.py                      # Typed enums, validation, Signal/Trade/PositionState state machine
├── indicators.py                # Pure functions: ATR, EMA, RSI, ADX, SuperTrend, VWAP, swing/level pivots
├── swing_detector.py            # ATR-prominence ZigZag swing detector (Swing records + tiers)
├── levels/                      # Horizontal S/R detectors — 3 selectable detectors behind one contract
│   │                            #   (__init__: registry + detect_levels dispatch + LEVEL_SOURCE_NAMES)
│   ├── base.py                  #   Normalized Level record + LevelSource contract + tolerance
│   ├── pivot_level.py           #   Pivot-seeded, invalidation-tracked (resistance/support/pullback)
│   ├── cluster_level.py         #   Merge-and-break, strength-scored
│   └── touch_level.py           #   Significance by historical touch count
├── exits.py                     # Pluggable exit policies: stops + take-profits, CompositeExit (stop-first)
├── strategy_configurator.py     # Per-family *Params (signal knobs) + PARAMS registry + exit-policy catalog (EXIT_PRESETS)
├── trade_configurator.py        # TradingConfig: costs, sizing, leverage, direction gate, risk overlays
├── data_configurator.py         # Single source of truth for market data: load_data() + parquet cache
├── providers/                   # Pluggable data-provider seam — pick via DataSpec.provider
│   │                            #   (__init__: PROVIDERS registry + make_provider + per-provider validation)
│   ├── base.py                  #   DataProvider contract + finalize_ohlcv (canonical UTC OHLCV)
│   ├── bybit.py                 #   BybitFetcher — reference provider (Bybit v5 klines, retry + rate limit)
│   └── yahoo.py                 #   YahooProvider (yfinance) — indices/commodities/futures
├── backtester.py                # Event-driven bar-by-bar backtester + additive equity layer
├── evaluation.py                # Robustness toolkit: metrics, sweep, grid_search, walk_forward, monte_carlo
├── visualization.py             # Plotly chart builder + level overlay (plot_levels)
├── live_records.py              # SQLite-backed live records: open position + trade history
├── live.py                      # Live trading loop with circuit breaker + SIGTERM
├── cli.py                       # Argument parsing + strategy dispatch
├── strategies/                  # 24 strategies, each a BaseStrategy (prepare + on_bar)
│   ├── base.py                  # Abstract base: prepare() + on_bar() + exit-policy injection
│   ├── fractal_breakout.py      # N-bar fractal-pivot S/R breakout (indicators.detect_swing_*)
│   ├── fractal_breakout_inv.py  # Inverse: fade the fractal breakout
│   ├── level_base.py            # Shared base for the level_* family (engine.levels, detector selectable)
│   ├── level_breakout.py        # Breakout of horizontal S/R from engine.levels
│   ├── level_breakout_inv.py    # Inverse: fade the level breakout
│   ├── g_bounce.py              # Level bounce / rejection (engine.levels)
│   ├── g_breakout.py            # Squeeze (compression) breakout
│   ├── g_breakout_false.py      # False-breakout reversal
│   ├── g_range.py               # Range / channel fade
│   ├── ema_cross.py             # EMA crossover + RSI filter
│   ├── ema_cross_inv.py         # Inverse EMA crossover
│   ├── ema_cross_adaptive.py    # Adaptive EMA crossover — RSI follow/fade regime switch
│   ├── ema_touch.py             # EMA touch-and-rejection (ported from the ema project)
│   ├── supertrend.py            # SuperTrend with correct band ratcheting
│   ├── supertrend_inv.py        # Inverse SuperTrend
│   ├── supertrend_adaptive.py   # ADX-regime-gated SuperTrend
│   ├── exhaustion_reversal.py   # Push → stall → volume-backed reversal
│   ├── impulse_flag.py          # Impulse + flag-consolidation breakout
│   ├── order_block.py           # Order-block retest
│   ├── order_block_inv.py       # Inverse order block
│   ├── vwap_bands.py            # VWAP stdev-band mean reversion
│   ├── swing_flip.py            # ATR-prominence ZigZag — flip mode
│   ├── swing_bounce.py          # ATR-prominence ZigZag — bounce mode (ported)
│   ├── swing_breakout.py        # ATR-prominence ZigZag — breakout mode (ported)
│   └── swing_ml.py              # ML swing-pivot classifier (imitation learning)
├── ml/                          # ML pipeline for swing_ml: features, labels, splits, order_flow
└── tests/                       # pytest: test_core, test_golden, test_level_detector, test_levels, test_providers, test_live, test_ml, …
```

## Key Design Decisions

### Event-driven backtester (no look-ahead possible)
Strategies implement two methods:
- `prepare(df)` — pre-compute indicators on a **copy** of the DataFrame
- `on_bar(i, df, state)` — evaluate bar `i`, may only read `df.iloc[:i+1]`

The backtester calls `on_bar()` sequentially, handing it a view truncated to
`df.iloc[:i+1]` each bar. Look-ahead bias is structurally impossible because the
runtime enforces it (`Backtester.run(enforce_causality=True)`, the default).

### Typed signal pipeline
No more stringly-typed `"long_entry"` / `"exit"` — signals use `Signal` dataclass
with `SignalAction` and `Direction` enums. Position transitions are validated by
`PositionState` (double-entry rejected, exit-while-flat rejected).

### Real trailing stop
Tracks `peak_price` (high-water mark for longs, low-water for shorts) since entry.
Exit triggers when price falls `ATR × multiplier` below peak — not below previous
bar's close.

### Costs in backtest
Round-trip fees + slippage deducted from every trade's P&L. Default: 4 bps taker
fee + 2 bps slippage per side = 12 bps round-trip.

### Live mode resilience
- SQLite live records: position + trade history survive process restarts
- Circuit breaker: stops after 10 consecutive fetch failures
- Exponential backoff on transient errors
- SIGTERM handler for graceful Docker/systemd shutdown
- Single HTML file overwritten each tick (no browser tab spam)
- Bounded signal list (rolling 500)

## Bugs Fixed from v1

| # | Bug | Impact |
|---|-----|--------|
| 1 | SuperTrend missing `else` branch | Indicator output was wrong |
| 2 | Swing levels computed globally (look-ahead) | Backtest results invalid |
| 3 | `any(close > lvl)` instead of cross detection | Constant whipsaw entries |
| 4 | "Trailing stop" compared to prev close, not peak | Exit logic mislabeled |
| 5 | Same-bar entry + exit possible | Ghost trades at same price |
| 6 | Live `_live_decision` stubs / broken precedence | Live mode non-functional |
| 7 | `fig.show()` in live loop | 120 browser tabs/hour |
| 8 | Timezone-naive timestamps | Times off by hours |
| 9 | DataFrame mutation across strategies | Cross-contamination |
| 10 | RSI divide-by-zero warning | Noisy logs |

## Running Tests

```bash
pytest engine/tests/test_core.py -v
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | `supertrend` | One of the 24 `StrategyName` values: `level_breakout`(+`_inv`), `fractal_breakout`(+`_inv`), `ema`/`ema_inv`/`ema_adaptive`, `ema_touch`, `supertrend`(+`_inv`/`_adaptive`), `exhaustion_reversal`, `impulse_flag`, `order_block`(+`_inv`), `vwap_bands`, `swing_flip`/`swing_bounce`/`swing_breakout`/`swing_ml`, `g_bounce`, `g_breakout`, `g_breakout_false`, `g_range` |
| `--mode` | `historical` | `historical`, `live` |
| `--provider` | inherit `ACTIVE` | `bybit` (crypto) or `yahoo` (indices/commodities/futures) |
| `--symbol` | inherit `ACTIVE` | Trading pair / ticker |
| `--interval` | inherit `ACTIVE` | `1,3,5,15,30,60,120,240,360,720,D,W,M` |
| `--candles` | inherit `ACTIVE` | Number of historical candles (ignored when `--start` is set) |
| `--category` | inherit `ACTIVE` | Bybit product type: `linear`, `inverse` (ignored by non-Bybit providers) |
| `--start` | inherit `ACTIVE` | Range-mode start, ISO e.g. `2026-03-20` (`--candles` ignored) |
| `--end` | inherit `ACTIVE` | Range-mode end, ISO (defaults to now) |
| `--save` | `None` | Output chart path; default is repo-root anchored — `data/results/<dataset>/<strategy>.html` (historical) or `data/live/<symbol>_<interval>_<strategy>.html` (live) |
| `--poll` | `30` | Live poll interval (seconds) |
| `--db` | `None` | SQLite path for live state; default `data/live/<strategy>.db` |
| `--notify` | `None` | Live-mode signal alerts: comma-separated `browser,desktop,telegram` |
| `--log-level` | `INFO` | `DEBUG,INFO,WARNING,ERROR` |
| `--log-json` | off | Structured JSON logging |

Trade-level parameters (the `TradingConfig` group — an omitted flag inherits the `ACTIVE_TRADE` block in `trade_configurator.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--initial-equity` | inherit `ACTIVE_TRADE` | Starting account equity in quote ccy |
| `--position-size-bps` | inherit `ACTIVE_TRADE` | Notional per trade as bps of equity (10000 = 100%) |
| `--leverage` | inherit `ACTIVE_TRADE` | Leverage multiplier on notional |
| `--fee-bps` | inherit `ACTIVE_TRADE` | Taker fee per side in bps (Bybit 0.04% = 4) |
| `--slippage-bps` | inherit `ACTIVE_TRADE` | Estimated slippage per side in bps |
| `--max-daily-loss-bps` | inherit `ACTIVE_TRADE` | Halt entries after this realized loss (bps) in a UTC day |
| `--max-holding-bars` | inherit `ACTIVE_TRADE` | Force-close a trade after this many bars |
| `--direction` | inherit `ACTIVE_TRADE` | Allowed trade sides: `long`, `short`, `both` |
| `--sizing-mode` | inherit `ACTIVE_TRADE` | `fixed` = position_size_bps; `risk` = risk_per_trade_bps (stop-where-available) |
| `--risk-per-trade-bps` | inherit `ACTIVE_TRADE` | Equity risked per trade in risk mode, bps (100 = 1%) |
| `--exit-preset` | strategy's preset | Override the strategy's exit policy with a named `EXIT_PRESETS` key |

License MIT (C) Galina Makarchuk