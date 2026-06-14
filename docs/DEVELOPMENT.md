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
├── strategies/                  # 20 strategies, each a BaseStrategy (prepare + on_bar)
│   ├── base.py                  # Abstract base: prepare() + on_bar() + exit-policy injection
│   ├── fractal_breakout.py      # N-bar fractal-pivot S/R breakout (indicators.detect_swing_*)
│   ├── fractal_breakout_inv.py  # Inverse: fade the fractal breakout
│   ├── level_base.py            # Shared base for the level_* family (engine.levels, detector selectable)
│   ├── level_breakout.py        # Breakout of horizontal S/R from engine.levels
│   ├── level_breakout_inv.py    # Inverse: fade the level breakout
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
| `--strategy` | `supertrend` | `level_breakout`, `level_breakout_inv`, `fractal_breakout`, `fractal_breakout_inv`, `ema`, `ema_inv`, `ema_adaptive`, `ema_touch`, `supertrend`, `supertrend_inv`, `supertrend_adaptive`, `exhaustion_reversal`, `impulse_flag`, `order_block`, `order_block_inv`, `vwap_bands`, `swing_flip`, `swing_ml`, `swing_bounce`, `swing_breakout` |
| `--mode` | `historical` | `historical`, `live` |
| `--symbol` | `BTCUSDT` | Any Bybit linear perp |
| `--interval` | `15` | `1,3,5,15,30,60,120,240,360,720,D,W,M` |
| `--candles` | `800` | Number of historical candles |
| `--save` | `None` | Output chart path; default is repo-root anchored — `data/results/<dataset>/<strategy>.html` (historical) or `data/live/<symbol>_<interval>_<strategy>.html` (live) |
| `--poll` | `30` | Live poll interval (seconds) |
| `--db` | `None` | SQLite path for live state; default `data/live/<strategy>.db` |
| `--log-level` | `INFO` | `DEBUG,INFO,WARNING,ERROR` |
| `--log-json` | off | Structured JSON logging |

License MIT (C) Galina Makarchuk