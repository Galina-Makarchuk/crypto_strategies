# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Crypto trading strategy framework, primarily for Bybit linear & inverse perpetuals, with a **pluggable data-provider seam** so the same strategies/backtester also run on non-crypto markets (indices, commodities, currency futures) via Yahoo Finance — see "Providers" below. Runs historical backtests or a live poll loop from a single CLI. Python package in [engine/](engine/); exploratory notebooks in [strategy_notebooks/](strategy_notebooks/).

## Common commands

```bash
# Install deps into the committed venv
pip install -r requirements.txt

# Historical backtest (candles cached to data/ohlcv/; results saved to data/results/<dataset>/)
python -m engine --strategy supertrend --interval 15 --candles 800

# Explicit date range, on the inverse market
python -m engine --strategy supertrend --interval 15 --start 2026-01-01 --end 2026-04-01 --category inverse

# Non-crypto via the Yahoo provider (daily gold futures); --category is ignored for yahoo
python -m engine --strategy supertrend --provider yahoo --symbol GC=F --interval D --candles 500

# Trade-level params (costs / sizing / direction / overlays) — see trade_configurator.py
python -m engine --strategy supertrend --interval 15 \
  --fee-bps 5.5 --slippage-bps 1.5 --leverage 2 --direction long \
  --initial-equity 25000 --position-size-bps 5000 --max-holding-bars 48 --max-daily-loss-bps 300

# Live mode (SQLite-persisted live records, poll loop)
python -m engine --strategy supertrend --mode live --interval 5 --poll 30

# Structured JSON logs (for log shippers)
python -m engine --strategy ema --log-json --log-level DEBUG

# Tests
pytest engine/tests/test_core.py -v
pytest engine/tests/test_core.py::TestATR::test_atr_shape -v   # single test
```

Valid `--strategy` values live on [StrategyName](engine/core.py) — 20 strategies incl. two breakout families that use **different** level detectors: `level_breakout`(+`_inv`) on the dedicated horizontal-S/R [levels package](engine/levels/) — three interchangeable detectors (`pivot_level` default, `cluster_level`, `touch_level`) selected via the `level_detector` knob on `LevelParams` and dispatched by `engine.levels.detect_levels` — and `fractal_breakout`(+`_inv`) on N-bar fractal pivots from [indicators.py](engine/indicators.py) (`detect_swing_*`). Plus `ema`(+`_inv`/`_adaptive` — the adaptive variant uses RSI as a follow/fade regime switch on the cross), `supertrend`(+`_inv`/`_adaptive`), `exhaustion_reversal`, `impulse_flag`, `order_block`(+`_inv`), `vwap_bands`, the ATR-prominence ZigZag swing family `swing_flip` / `swing_bounce` / `swing_breakout` (+ML `swing_ml`), and `ema_touch` (EMA touch-and-rejection — distinct from the `ema` crossover). The `level_*` family has room to grow (`level_bounce`, `level_retest`) on the shared [level_base.py](engine/strategies/level_base.py). Valid intervals: [VALID_INTERVALS](engine/core.py); valid `--category`: `linear`, `inverse`.

## Architecture — the big picture

The system is organized around one invariant: **look-ahead bias is structurally impossible**. Understanding how that invariant is enforced is the key to being productive here.

### Three layers, one contract

1. **Indicators** ([indicators.py](engine/indicators.py)) — pure `(DataFrame|Series, params) → Series` functions. No state, no mutation, no strategy logic.
2. **Strategies** ([strategies/](engine/strategies/)) — each subclasses [BaseStrategy](engine/strategies/base.py) and implements two methods:
   - `prepare(df) -> df` — compute indicator columns on a **copy** of the DataFrame (never mutate the caller's df).
   - `on_bar(i, df, state)` — evaluate bar `i`. **May only read `df.iloc[0..i]`**. This is the contract that prevents look-ahead; both the backtester and the live engine depend on it. It's also *enforced by the runtime*: [`Backtester.run()`](engine/backtester.py) feeds `on_bar` a view truncated to `[0..i]` by default (`enforce_causality=True`), so a future read is structurally impossible. [test_config_propagation.py](engine/tests/test_config_propagation.py) runs every strategy with `enforce_causality=False` and asserts the trades match the enforced run.
3. **Runners** — [Backtester.run()](engine/backtester.py#L138) iterates `on_bar` across every bar; [LiveEngine._tick()](engine/live.py#L176) calls `on_bar` only on the last bar. Same strategy code runs in both modes.

### Signals, positions, and P&L flow through [core.py](engine/core.py)

`on_bar` never returns a signal — it calls `state.enter()` / `state.exit()` on a [PositionState](engine/core.py#L158). The state machine rejects invalid transitions (double-entry, exit-while-flat), which means strategy code does not need to track position status itself.

`state.exit(ts, price, reason)` is where P&L is computed. Round-trip fees + slippage come from `cost_bps`, which the **runner seeds onto `PositionState`** from [TradingConfig.total_cost_bps()](engine/trade_configurator.py) (see "Two configs" below) — strategies no longer pass cost themselves. The direction gate and daily-loss overlay are likewise enforced in `PositionState.enter()` from runner-seeded fields, so `on_bar` stays free of trade-level policy.

### Trailing stop = peak-tracked, not bar-tracked

Every `on_bar` implementation must call `state.update_peak(high, low)` before checking exits when a position is open. The trailing stop compares against [Trade.peak_price](engine/core.py#L133) (high-water for longs, low-water for shorts) — **not** the previous bar's close. This was bug #4 in the v1 rewrite; the pattern is load-bearing.

### Two configs: strategy params vs trade params

Tunable numbers split across two frozen dataclasses by **who they belong to**:

- **Per-strategy `*Params` classes** (in [strategy_configurator.py](engine/strategy_configurator.py)) — *how signals are generated*: one small frozen dataclass per strategy **family** (`EmaParams`, `SupertrendParams`, `EmaTouchParams`, `FractalParams`, `LevelParams`, `ExhaustionParams`, `ImpulseFlagParams`, `OrderBlockParams`, `VwapParams`, `SwingParams`, `SwingMlParams`), each holding exactly the indicator periods/multipliers/levels that family reads **plus** its default exit assignment (the `EXITS` ClassVar). Consumed by strategies via `self.config`. A foreign knob (e.g. `supertrend_mult` on an EMA strategy) raises `TypeError` at `dataclasses.replace` instead of landing silently inert; each family owns its own `atr_period`. Field validation is delegated to [config_validation.py](engine/config_validation.py) via `__post_init__` (the shared rule home for `TradingConfig`/`DataSpec` too), so a bad knob — in this file, a notebook `dataclasses.replace`, or a `sweep` grid — fails loudly at construction. The `PARAMS` registry maps every strategy name → its class: `params_for(name)` returns its defaults (the single source of truth the CLI and notebooks start from), `params_class_for(name)` lets `BaseStrategy` type-check the config it's handed. This file **also** holds the exit-policy catalog — see "Exit policies" below; `PER_STRATEGY_EXIT` is **derived** from each class's `EXITS`, and import-time guards (`_validate_exit_catalog` / `_validate_params_registry`) pin that every `StrategyName` member has a params class and a valid exit.
- **[TradingConfig](engine/trade_configurator.py)** (in [trade_configurator.py](engine/trade_configurator.py)) — *how you trade any signal*, independent of strategy: `initial_equity`, `position_size_bps`, `leverage`, `sizing_mode`, `risk_per_trade_bps`, `fee_bps`, `slippage_bps`, `max_daily_loss_bps`, `max_holding_bars`, and the `direction` gate (long/short/both). Edit the `ACTIVE_TRADE` block once and it's the **project-wide default**: notebooks pass it directly, and CLI runs *seed from it* — each `--flag` overrides only that one field (so the CLI and notebooks can't silently diverge). Mirrors `data_configurator`'s `ACTIVE`. `total_cost_bps()` is **derived** (`2*(fee+slippage)`), never stored.

  **Sizing modes** (`sizing_mode`): `FIXED` deploys `position_size_bps × leverage` of equity per trade; `RISK` sizes each trade so a stop-out loses `risk_per_trade_bps` of equity (`risk$ ÷ stop-distance`, leverage not applied). RISK is **stop-where-available**: a trade is risk-sized when its exit policy supplies an entry stop (`ExitPolicy.initial_stop`, recorded on `Trade.stop_price` via `state.enter(..., stop_price=...)`) — which now covers every strategy except the genuinely stopless ones (`vwap_bands`, `swing_flip`/`swing_ml`), which **fall back to fixed-fraction**, counted in `risk_sizing_fallbacks` and logged so it's never silent.

The runner ([Backtester](engine/backtester.py) / [LiveEngine](engine/live.py)) seeds the relevant `TradingConfig` values onto `PositionState` (cost at exit; direction + daily-loss gates at entry; `max_holding_bars` enforced in the run loop). On top of bps P&L, the backtester runs an **additive equity layer**: it sizes each trade per `sizing_mode` (fixed-fraction or risk-based, see above) and compounds `initial_equity` into per-trade `notional` / `pnl_currency` / `equity_after` and result-level `final_equity` / `total_return_pct` / `max_drawdown_pct`. This is purely additive — `pnl_bps` and all bps metrics are unchanged by sizing (a golden test pins `FIXED` vs `RISK` to identical bps).

### Exit policies (selectable stop-loss / take-profit)

Stops and take-profits are **pluggable mechanisms** in [exits.py](engine/exits.py), behind one `ExitPolicy` interface (an SL and a TP both just decide "close at price X for reason Y"): `ChandelierStop` (ATR trail from peak, close-triggered), `AtrStop`/`FixedPctStop`/`StructuralStop` (fixed, intrabar fill), `FixedPctTarget`/`RrTarget`/`StructuralTarget`/`CloseCrossTarget`, and `CompositeExit` (runs them **stop-first** so an ambiguous bar resolves to the stop). `ExitPolicy.initial_stop` feeds risk-based sizing.

Each strategy is **assigned** a policy in [strategy_configurator.py](engine/strategy_configurator.py): `EXIT_PRESETS` (named factories) + `DEFAULT_EXIT` (the trend/EMA/fractal-breakout group — `level_breakout`/`_inv` instead use `structural_rr2`/`atr_stop_rr2`) + string-keyed `PER_STRATEGY_EXIT` overrides + `exit_policy_for(name)`. `BaseStrategy` injects it (`self.exit_policy`, default `exit_policy_for(self.name)`); a caller may inject a different one (the CLI exposes `--exit-preset`). In `on_bar`, a strategy delegates its price stop/target to `self.exit_policy.evaluate(self._exit_ctx(...))` and keeps **signal-based exits native** (trend/cross flips, exhaustion's invalidation + time-stop, impulse_flag's T1 breakeven-shift + order expiry). Strategy-specific *levels* (order-block extreme, flag stop, R:R target, VWAP mid) are passed to the policy as `ref_stop` / `ref_target`. Every conversion is pinned byte-for-byte by [test_golden.py](engine/tests/test_golden.py).

### Live mode adds resilience, not new semantics

[LiveEngine](engine/live.py#L56) re-fetches a window of candles each tick, runs `prepare()` fresh, and calls `on_bar` on the last bar. Position state is persisted to SQLite via [LiveRecords](engine/live_records.py#L65) (WAL mode) so a restart recovers open positions. A circuit breaker halts the loop after 10 consecutive fetch failures; SIGTERM/SIGINT trigger graceful shutdown. Signals are kept to a rolling window of 500.

### Providers (pluggable data sources)

Market data comes through a **provider seam** ([engine/providers/](engine/providers/)): a `DataProvider` ([base.py](engine/providers/base.py)) is any source that returns the one canonical contract — a **timezone-aware UTC** DataFrame indexed by `timestamp` with float columns `open, high, low, close, volume, turnover`. Naive timestamps are bugs; everything downstream of `load_data()` is provider-blind. Two providers are registered in [providers/__init__.py](engine/providers/__init__.py): `bybit` ([BybitFetcher](engine/fetcher.py), the reference — crypto linear/inverse perps) and `yahoo` ([YahooProvider](engine/providers/yahoo.py) via `yfinance` — indices / commodities / currency futures, e.g. `GC=F`, `CL=F`, `ES=F`, `6E=F`; keyless; `turnover` synthesized as `close*volume`). Pick one with `DataSpec.provider` (default `bybit`). Validation is **per-provider** (`validate_spec` / `resolve_category`): each provider declares its supported intervals and product categories — `category` (`linear`/`inverse`) is Bybit-only and ignored by providers without a product taxonomy. Adding a provider = implement the contract + register it; nothing downstream changes. `swing_ml*` order-flow (Tier-3) stays Bybit-only.

### Data is configured once, cached, and reused

[engine/data_configurator.py](engine/data_configurator.py) is the single source of truth for market data. Edit the `ACTIVE` `DataSpec` block once — `provider`, symbol, interval, `category`, and either `num_candles` or `start`/`end` — and every notebook, script, and CLI run uses it (the CLI mirrors it: `--provider`, `--symbol`, etc., each overriding one field). `load_data()` dispatches to the spec's provider and caches parquet (`bybit` keeps `data/ohlcv/<category>/…`; other providers are namespaced `data/ohlcv/<provider>/<category-or-_>/…`) with a JSON provenance sidecar (pinned `[start, end]` ranges are immutable; count/open-ended windows refetch after one bar interval; `refresh=True` forces). `save_result(result, spec)` persists each backtest to `data/results/<dataset_signature>/…` (bybit signatures unchanged; other providers prefixed). **Go through `load_data()` — do not instantiate a provider directly** (the two `swing_ml*` notebooks are the deliberate exception: they fetch via `BybitFetcher` into their own pickle cache for a fixed multi-year training window). All of `data/` is git-ignored.

## Adding a new strategy

1. Create `engine/strategies/<name>.py` subclassing `BaseStrategy`, set `name = "<name>"`.
2. Implement `prepare` (copy df, add indicator columns) and `on_bar` (update peak → check exit → check entry, in that order). For price stops/targets, delegate to `self.exit_policy.evaluate(self._exit_ctx(...))` and keep signal-based exits native; for sizing, seed `state.enter(..., stop_price=self._entry_stop(...))`.
3. Register the class in [strategies/__init__.py](engine/strategies/__init__.py) and add an enum member to [StrategyName](engine/core.py#L36) plus a dispatch entry in [cli._build_strategy()](engine/cli.py).
4. Create a `<Name>Params` frozen dataclass in [strategy_configurator.py](engine/strategy_configurator.py) holding the strategy's signal knobs (validate them in `__post_init__` via [config_validation.py](engine/config_validation.py)) **and** its `EXITS` map (name → `EXIT_PRESETS` key, else `DEFAULT`); register it in the `PARAMS` dict keyed by the strategy name. (A family shared by several names — e.g. `supertrend`/`_inv`/`_adaptive` — uses one class listed under each name.) The import-time guards then require the new name to have both a params class and an exit. Trade-level knobs go on [TradingConfig](engine/trade_configurator.py) instead.
5. Add a golden snapshot in [test_golden.py](engine/tests/test_golden.py) so its trades are pinned.

## Notebooks

[strategy_notebooks/](strategy_notebooks/) is where new strategy ideas are prototyped before being ported into [strategies/](engine/strategies/). Notebooks are not part of the test surface. They load candles via `load_data()` and persist results via `save_result(result, ACTIVE)` (both from [data_configurator.py](engine/data_configurator.py)), and pass `trading_config=ACTIVE_TRADE` (from [trade_configurator.py](engine/trade_configurator.py)) to `Backtester` so trade params flow from the one editable block — rather than hardcoding symbol/interval, costs, or instantiating `BybitFetcher`. The standard single-family notebooks share one layout, with [ema_rsi.ipynb](strategy_notebooks/ema_rsi.ipynb) as the canonical template: a **Contents** index → a **Configuration** chapter (Setup / Automatic / Manual / Final configuration) → one section per strategy variant, each with **Backtesting** / **Grid search** / **Walk-forward analysis** / **Monte Carlo simulations** / **Live signals** subsections; any bespoke per-notebook analysis (e.g. `vwap_bands`' band-hit / entry-band sweeps, `super_trend`'s multi-strategy live cell) is preserved as extra trailing sections. The two ML notebooks (`swing_ml*`) intentionally keep their own pickle cache for their fixed multi-year training window. Their out-of-sample backtests inject cross-validated probabilities into `ml_p_long`/`ml_p_short`/`ml_valid` columns and run the **real** `Backtester(SwingMLStrategy(cfg, require_model=False))` — `require_model=False` skips model inference and `prepare()` passes the injected columns straight to `on_bar`, so the OOS evaluation uses the same trade logic / exit policy / costs as live (no hand-rolled loop to drift).

## Plans

- [docs/config-split-plan.md](docs/config-split-plan.md) — the original config-split design. The **trade side is implemented** as [trade_configurator.py](engine/trade_configurator.py) (costs, sizing/equity, direction gate, leverage, daily-loss, max-holding) with the equity layer wired through the backtester. The **strategy side is now also implemented**: the monolithic `StrategyConfig` has been split into per-family `*Params` classes in [strategy_configurator.py](engine/strategy_configurator.py), each owning its fields + `EXITS`, with a `PARAMS` name→class registry and `params_for(name)` as the defaults source. (This went further than the doc's earlier "keep one `StrategyConfig` + a `PARAMS` metadata dict" sketch — the full per-family split makes a foreign-knob override impossible by construction.)
