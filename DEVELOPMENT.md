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
├── core.py                      # Typed enums, validation, Signal/Trade/PositionState
├── indicators.py                # Pure functions: ATR, EMA, RSI, ADX, SuperTrend, swing detection
├── fetcher.py                   # Bybit v5 API client with retry + rate limiting
├── backtester.py                # Event-driven bar-by-bar backtester
├── visualization.py             # Plotly chart builder (batched traces)
├── live_records.py              # SQLite-backed live records (position + trade history)
├── live.py                      # Live trading loop with circuit breaker + SIGTERM
├── cli.py                       # Argument parsing + dispatch
├── strategies/
│   ├── base.py                  # Abstract base: prepare() + on_bar()
│   ├── swing.py                 # Swing breakout (no look-ahead, cross detection)
│   ├── swing_inv.py             # Inverse swing: fade breakouts / buy the dip
│   ├── ema_cross.py             # EMA crossover + RSI filter
│   ├── supertrend.py            # SuperTrend with correct band ratcheting
│   ├── supertrend_inv.py        # Inverse SuperTrend: fade trend flips
│   ├── supertrend_adaptive.py   # ADX-regime-gated SuperTrend (trend vs range)
│   └── exhaustion_reversal.py   # Push → stall → volume-backed reversal
└── tests/
    └── test_core.py             # 22 tests covering indicators, core types, strategies
```

## Key Design Decisions

### Event-driven backtester (no look-ahead possible)
Strategies implement two methods:
- `prepare(df)` — pre-compute indicators on a **copy** of the DataFrame
- `on_bar(i, df, state)` — evaluate bar `i`, may only read `df.iloc[:i+1]`

The backtester calls `on_bar()` sequentially. Look-ahead bias is structurally
impossible because the strategy interface enforces it.

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
| `--strategy` | `supertrend` | `swing`, `swing_inv`, `ema`, `supertrend`, `supertrend_inv`, `supertrend_adaptive`, `exhaustion_reversal` |
| `--mode` | `historical` | `historical`, `live` |
| `--symbol` | `BTCUSDT` | Any Bybit linear perp |
| `--interval` | `15` | `1,3,5,15,30,60,120,240,360,720,D,W,M` |
| `--candles` | `800` | Number of historical candles |
| `--save` | `chart.html` | Output chart path |
| `--poll` | `30` | Live poll interval (seconds) |
| `--db` | `trading_state.db` | SQLite path for live state |
| `--log-level` | `INFO` | `DEBUG,INFO,WARNING,ERROR` |
| `--log-json` | off | Structured JSON logging |

License MIT (C) Galina Makarchuk