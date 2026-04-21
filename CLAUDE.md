# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Crypto trading strategy framework for Bybit linear perpetuals. Runs historical backtests or a live poll loop from a single CLI. Python package in [entry_exit_points/](entry_exit_points/); exploratory notebooks in [strategy_notebooks/](strategy_notebooks/).

## Common commands

```bash
# Install deps into the committed venv
pip install -r requirements.txt

# Historical backtest
python -m entry_exit_points --strategy supertrend --interval 15 --candles 800

# Live mode (SQLite state persistence, poll loop)
python -m entry_exit_points --strategy supertrend --mode live --interval 5 --poll 30

# Structured JSON logs (for log shippers)
python -m entry_exit_points --strategy ema --log-json --log-level DEBUG

# Tests
pytest entry_exit_points/tests/test_core.py -v
pytest entry_exit_points/tests/test_core.py::TestATR::test_atr_shape -v   # single test
```

Valid `--strategy` values live on [StrategyName](entry_exit_points/models.py#L34) (`swing`, `swing_inv`, `ema`, `supertrend`, `supertrend_inv`, `supertrend_adaptive`, `exhaustion_reversal`). Valid intervals: [VALID_INTERVALS](entry_exit_points/models.py#L51).

## Architecture — the big picture

The system is organized around one invariant: **look-ahead bias is structurally impossible**. Understanding how that invariant is enforced is the key to being productive here.

### Three layers, one contract

1. **Indicators** ([indicators.py](entry_exit_points/indicators.py)) — pure `(DataFrame|Series, params) → Series` functions. No state, no mutation, no strategy logic.
2. **Strategies** ([strategies/](entry_exit_points/strategies/)) — each subclasses [BaseStrategy](entry_exit_points/strategies/base.py) and implements two methods:
   - `prepare(df) -> df` — compute indicator columns on a **copy** of the DataFrame (never mutate the caller's df).
   - `on_bar(i, df, state)` — evaluate bar `i`. **May only read `df.iloc[0..i]`**. This is the contract that prevents look-ahead; both the backtester and the live engine depend on it.
3. **Runners** — [Backtester.run()](entry_exit_points/backtester.py#L73) iterates `on_bar` across every bar; [LiveEngine._tick()](entry_exit_points/live.py#L110) calls `on_bar` only on the last bar. Same strategy code runs in both modes.

### Signals, positions, and P&L flow through [models.py](entry_exit_points/models.py)

`on_bar` never returns a signal — it calls `state.enter()` / `state.exit()` on a [PositionState](entry_exit_points/models.py#L149). The state machine rejects invalid transitions (double-entry, exit-while-flat), which means strategy code does not need to track position status itself.

`state.exit(ts, price, cost_bps)` is where P&L is computed, including round-trip fees + slippage from [StrategyConfig.total_cost_bps()](entry_exit_points/models.py#L112). Strategies must pass `self.config.total_cost_bps()` when calling `exit` — skipping it silently produces gross (pre-cost) P&L.

### Trailing stop = peak-tracked, not bar-tracked

Every `on_bar` implementation must call `state.update_peak(high, low)` before checking exits when a position is open. The trailing stop compares against [Trade.peak_price](entry_exit_points/models.py#L142) (high-water for longs, low-water for shorts) — **not** the previous bar's close. This was bug #4 in the v1 rewrite; the pattern is load-bearing.

### Config is frozen and central

All tunable numbers (periods, multipliers, fees, risk %, strategy-specific knobs like the exhaustion-reversal streak thresholds) live on [StrategyConfig](entry_exit_points/models.py#L67), a frozen dataclass. Add new parameters there rather than hard-coding in strategy files.

### Live mode adds resilience, not new semantics

[LiveEngine](entry_exit_points/live.py#L35) re-fetches a window of candles each tick, runs `prepare()` fresh, and calls `on_bar` on the last bar. Position state is persisted to SQLite via [StateStore](entry_exit_points/persistence.py#L47) (WAL mode) so a restart recovers open positions. A circuit breaker halts the loop after 10 consecutive fetch failures; SIGTERM/SIGINT trigger graceful shutdown. Signals are kept to a rolling window of 500.

### Fetcher contract

[BybitFetcher.fetch_klines()](entry_exit_points/fetcher.py#L117) always returns a **timezone-aware UTC** DataFrame indexed by timestamp with columns `open, high, low, close, volume, turnover`. Naive timestamps are bugs — every downstream component assumes UTC.

## Adding a new strategy

1. Create `entry_exit_points/strategies/<name>.py` subclassing `BaseStrategy`, set `name = "<name>"`.
2. Implement `prepare` (copy df, add indicator columns) and `on_bar` (update peak → check exit → check entry, in that order).
3. Register the class in [strategies/__init__.py](entry_exit_points/strategies/__init__.py) and add an enum member to [StrategyName](entry_exit_points/models.py#L34) plus a dispatch entry in [cli._build_strategy()](entry_exit_points/cli.py#L24).
4. Add any new config knobs to [StrategyConfig](entry_exit_points/models.py#L67).

## Notebooks

[strategy_notebooks/](strategy_notebooks/) is where new strategy ideas are prototyped before being ported into [strategies/](entry_exit_points/strategies/). [analysis.ipynb](analysis.ipynb) at the repo root is for ad-hoc result analysis. Notebooks are not part of the test surface.
