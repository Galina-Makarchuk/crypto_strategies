# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Crypto trading strategy framework for Bybit linear & inverse perpetuals. Runs historical backtests or a live poll loop from a single CLI. Python package in [engine/](engine/); exploratory notebooks in [strategy_notebooks/](strategy_notebooks/).

## Common commands

```bash
# Install deps into the committed venv
pip install -r requirements.txt

# Historical backtest (candles cached to data/ohlcv/; results saved to data/results/<dataset>/)
python -m engine --strategy supertrend --interval 15 --candles 800

# Explicit date range, on the inverse market
python -m engine --strategy supertrend --interval 15 --start 2026-01-01 --end 2026-04-01 --category inverse

# Live mode (SQLite state persistence, poll loop)
python -m engine --strategy supertrend --mode live --interval 5 --poll 30

# Structured JSON logs (for log shippers)
python -m engine --strategy ema --log-json --log-level DEBUG

# Tests
pytest engine/tests/test_core.py -v
pytest engine/tests/test_core.py::TestATR::test_atr_shape -v   # single test
```

Valid `--strategy` values live on [StrategyName](engine/models.py) — 14 strategies incl. `swing`(+`_inv`), `ema`(+`_inv`), `supertrend`(+`_inv`/`_adaptive`), `exhaustion_reversal`, `impulse_flag`, `order_block`(+`_inv`), `vwap_bands`, `swing_zigzag`(+`_ml`). Valid intervals: [VALID_INTERVALS](engine/models.py); valid `--category`: `linear`, `inverse`.

## Architecture — the big picture

The system is organized around one invariant: **look-ahead bias is structurally impossible**. Understanding how that invariant is enforced is the key to being productive here.

### Three layers, one contract

1. **Indicators** ([indicators.py](engine/indicators.py)) — pure `(DataFrame|Series, params) → Series` functions. No state, no mutation, no strategy logic.
2. **Strategies** ([strategies/](engine/strategies/)) — each subclasses [BaseStrategy](engine/strategies/base.py) and implements two methods:
   - `prepare(df) -> df` — compute indicator columns on a **copy** of the DataFrame (never mutate the caller's df).
   - `on_bar(i, df, state)` — evaluate bar `i`. **May only read `df.iloc[0..i]`**. This is the contract that prevents look-ahead; both the backtester and the live engine depend on it.
3. **Runners** — [Backtester.run()](engine/backtester.py#L73) iterates `on_bar` across every bar; [LiveEngine._tick()](engine/live.py#L110) calls `on_bar` only on the last bar. Same strategy code runs in both modes.

### Signals, positions, and P&L flow through [models.py](engine/models.py)

`on_bar` never returns a signal — it calls `state.enter()` / `state.exit()` on a [PositionState](engine/models.py#L149). The state machine rejects invalid transitions (double-entry, exit-while-flat), which means strategy code does not need to track position status itself.

`state.exit(ts, price, cost_bps)` is where P&L is computed, including round-trip fees + slippage from [StrategyConfig.total_cost_bps()](engine/models.py#L112). Strategies must pass `self.config.total_cost_bps()` when calling `exit` — skipping it silently produces gross (pre-cost) P&L.

### Trailing stop = peak-tracked, not bar-tracked

Every `on_bar` implementation must call `state.update_peak(high, low)` before checking exits when a position is open. The trailing stop compares against [Trade.peak_price](engine/models.py#L142) (high-water for longs, low-water for shorts) — **not** the previous bar's close. This was bug #4 in the v1 rewrite; the pattern is load-bearing.

### Config is frozen and central

All tunable numbers (periods, multipliers, fees, risk %, strategy-specific knobs like the exhaustion-reversal streak thresholds) live on [StrategyConfig](engine/models.py#L67), a frozen dataclass. Add new parameters there rather than hard-coding in strategy files.

### Live mode adds resilience, not new semantics

[LiveEngine](engine/live.py#L35) re-fetches a window of candles each tick, runs `prepare()` fresh, and calls `on_bar` on the last bar. Position state is persisted to SQLite via [StateStore](engine/persistence.py#L47) (WAL mode) so a restart recovers open positions. A circuit breaker halts the loop after 10 consecutive fetch failures; SIGTERM/SIGINT trigger graceful shutdown. Signals are kept to a rolling window of 500.

### Fetcher contract

[BybitFetcher.fetch_klines()](engine/fetcher.py#L117) always returns a **timezone-aware UTC** DataFrame indexed by timestamp with columns `open, high, low, close, volume, turnover`. Naive timestamps are bugs — every downstream component assumes UTC. It accepts a `category` argument (`linear` | `inverse`, default `linear`) — the only place the Bybit product type enters the request.

### Data is configured once, cached, and reused

[engine/data_configurator.py](engine/data_configurator.py) is the single source of truth for market data. Edit the `ACTIVE` `DataSpec` block once — symbol, interval, `category` (`linear`/`inverse`), and either `num_candles` or `start`/`end` — and every notebook, script, and CLI run uses it. `load_data()` fetches via `BybitFetcher` and caches parquet under `data/ohlcv/<category>/…` with a JSON provenance sidecar (pinned `[start, end]` ranges are immutable; count/open-ended windows refetch after one bar interval; `refresh=True` forces). `save_result(result, spec)` persists each backtest to `data/results/<dataset_signature>/<strategy>.json` + `<strategy>_trades.csv`. **Go through `load_data()` — do not instantiate `BybitFetcher` directly.** All of `data/` is git-ignored.

## Adding a new strategy

1. Create `engine/strategies/<name>.py` subclassing `BaseStrategy`, set `name = "<name>"`.
2. Implement `prepare` (copy df, add indicator columns) and `on_bar` (update peak → check exit → check entry, in that order).
3. Register the class in [strategies/__init__.py](engine/strategies/__init__.py) and add an enum member to [StrategyName](engine/models.py#L34) plus a dispatch entry in [cli._build_strategy()](engine/cli.py#L24).
4. Add any new config knobs to [StrategyConfig](engine/models.py#L67).

## Notebooks

[strategy_notebooks/](strategy_notebooks/) is where new strategy ideas are prototyped before being ported into [strategies/](engine/strategies/). [analysis.ipynb](analysis.ipynb) at the repo root is for ad-hoc result analysis. Notebooks are not part of the test surface. They load candles via `load_data()` and persist results via `save_result(result, ACTIVE)` (both from [data_configurator.py](engine/data_configurator.py)) rather than hardcoding symbol/interval or instantiating `BybitFetcher`. The two ML notebooks (`swing_zigzag_ml*`) intentionally keep their own pickle cache for their fixed multi-year training window.

## Plans

- [docs/config-split-plan.md](docs/config-split-plan.md) — proposed (not yet implemented) split of the config surface into `strategy_configurator.py` (per-strategy params + sweeps) and `trade_configurator.py` (direction / risk / costs / stops).
