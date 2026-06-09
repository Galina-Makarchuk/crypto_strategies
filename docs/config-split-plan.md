# Plan: split config into `strategy_configurator` + `trade_configurator`

**Status:** proposed (not yet implemented) · **Date:** 2026-05-29 · Approval pending before any code.

Splits the configurable surface into two orthogonal, centralized, extendable files:
- **`strategy_configurator.py`** — *how signals are generated* (per-strategy params), consumed by strategies.
- **`trade_configurator.py`** — *how you trade any signal* (direction, sizing, risk, costs, stop policy), consumed by the backtester/execution layer.

Builds on the existing `data_configurator.py` ergonomic (an editable default block + helpers; results saved via `save_result`; outputs under git-ignored `data/`). Together they form the triad: **data → strategy → trade**, composed at run time.

## Verified facts (grounded in current code)
- `StrategyConfig` (`engine/core.py`) is a single frozen dataclass shared by all 14 strategies (~78 fields, grouped by prefix).
- Every strategy applies costs via `self.config.total_cost_bps()` passed to `state.exit(...)` — **54 `state.exit(` call sites** across the strategies; ~12 files have a per-`on_bar` `cost = self.config.total_cost_bps()` line.
- `Direction` enum has **only `LONG`/`SHORT`** (no `BOTH`); 14 strategy files reference `Direction.SHORT`, i.e. strategies trade both ways today.
- `save_result` lives in **`engine.data_configurator`** (not a separate module).
- `risk_per_trade_pct` / `max_open_trades` are read **nowhere** — the backtester has no equity/position-sizing layer (computes P&L in bps per trade).
- `AdaptiveSuperTrendStrategy`'s `adx_threshold` (default `25.0`) is now a `StrategyConfig` field (`engine/strategy_configurator.py`), read via `self.config.adx_threshold` — no longer a constructor arg, so it sweeps/overrides like any other signal knob.

## Files

**New**
- `engine/strategy_configurator.py` — per-strategy `PARAMS` registry + `build_config()` + `sweep()`
- `engine/trade_configurator.py` — `TradingConfig` (direction, risk, costs, stop policy) + `ACTIVE_TRADE`

**Edited**
- `engine/core.py` — remove `fee_bps`, `slippage_bps`, `risk_per_trade_pct`, `max_open_trades` + `total_cost_bps()` from `StrategyConfig`; give `PositionState` a `cost_bps` field
- `engine/backtester.py` — `Backtester(..., trading_config=None)`; seed `PositionState` with the cost; apply at exit
- `engine/strategies/*.py` (the 14) — drop the per-`on_bar` `cost = self.config.total_cost_bps()` (~12 files) and the cost arg from the 54 `state.exit(...)` calls
- `engine/cli.py` — add `--fee-bps/--slippage-bps/--direction/--risk-per-trade/--max-open-trades`; build a `TradingConfig`
- the 9 migrated notebooks — swap to the new configurators (Phase 2)

## `strategy_configurator.py`
Unlike data (one `ACTIVE` dataset), there are *many* strategies — so the single source is a **per-strategy registry**, and you materialize a config per strategy:

```python
PARAMS = {
  "supertrend": {
    "supertrend_period": {"default": 10,  "sweep": [7, 10, 14]},
    "supertrend_mult":   {"default": 3.0, "sweep": [2.0, 2.5, 3.0]},
  },
  "ema": {"ema_fast": {"default": 9, "sweep": [5, 9, 13]}, ...},
  # ➕ new strategy = one block here
}

def build_config(strategy, **overrides) -> StrategyConfig: ...   # defaults + overrides
def sweep(strategy, df, trading_config=None) -> pd.DataFrame: ... # full grid → DataFrame
```

## `trade_configurator.py`
```python
class TradeDirection(Enum):          # NEW — Direction has only LONG/SHORT today
    LONG = "long"; SHORT = "short"; BOTH = "both"

@dataclass(frozen=True)
class TradingConfig:
    direction: TradeDirection = TradeDirection.BOTH   # default = current behavior
    risk_per_trade_pct: float = 1.0
    max_open_trades: int = 1
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    global_stop_loss_pct: float | None = None         # optional, off by default
    def total_cost_bps(self) -> float: return 2 * (self.fee_bps + self.slippage_bps)

ACTIVE_TRADE = TradingConfig()       # the one editable trading block (mirrors data's ACTIVE)
```

## Boundary decisions (resolved)

**A) Stops** → per-strategy stop knobs (`atr_trail_mult`, `*_stop_atr_mult`, `ob_stop_buffer_pct`, `flag_stop_atr_mult`) **stay** in strategy params (tuned per strategy, must be sweepable). Add an *optional* `global_stop_loss_pct` to `TradingConfig`, **defer wiring**. Cost now: zero refactor.

**B) Costs** → move `fee_bps`/`slippage_bps` to `TradingConfig`; move cost application out of strategies. **Mechanism:** the backtester creates `PositionState(cost_bps=trading_config.total_cost_bps())`; `PositionState.exit()` uses `self.cost_bps`. Strategies drop the cost arg — all 54 `state.exit(ts, price, reason)` calls + ~12 cost-init lines. Pure mechanical change, **no algorithmic difference** → guarded by a golden-output test (P&L byte-identical). Low-risk interim available (keep `total_cost_bps` delegating to `TradingConfig`), but the full move is safe in one pass.

**C) Direction** → keep the inverse strategy classes as-is; add `direction` as a **trade-level gate**, default `BOTH` (preserves today's behavior). **Caveat:** strategies call `state.enter()` themselves inside `on_bar`, so the gate goes in `PositionState.enter()` (reject entries whose direction isn't allowed), **not** "the backtester filters before enter." Least-baked piece → **defer to Phase 3**.

## Single source of truth (resolved)
`PARAMS` is **metadata only** (grouping + sweep grids); `StrategyConfig` stays the **typed defaults source** (keeps IDE/type-checking). A guard test asserts every `PARAMS` key is a real `StrategyConfig` field, so they can't drift. (Refines the earlier "generate from PARAMS" idea — generating a dataclass would kill autocomplete/typing, so we don't.)

## Notebook before/after
```python
# before
config   = StrategyConfig()
strategy = SuperTrendStrategy(config)
result   = Backtester(strategy, symbol=ACTIVE.symbol).run(df, interval=ACTIVE.interval)
save_result(result, ACTIVE)

# after
from engine.strategy_configurator import build_config, sweep
from engine.trade_configurator import ACTIVE_TRADE
strategy = SuperTrendStrategy(build_config("supertrend"))            # or build_config(..., supertrend_mult=2.5)
result   = Backtester(strategy, symbol=ACTIVE.symbol, trading_config=ACTIVE_TRADE).run(df, interval=ACTIVE.interval)
save_result(result, ACTIVE)                                          # save_result stays in engine.data_configurator

# new: full sweep of this strategy's params → DataFrame, results saved non-colliding
sweep_df = sweep("supertrend", df, trading_config=ACTIVE_TRADE)      # → data/sweeps/supertrend_<sig>.csv
```

## CLI before/after
```bash
# after — trade knobs exposed
python -m engine --strategy supertrend --interval 15 \
  --fee-bps 3.5 --slippage-bps 1.5 --direction both --risk-per-trade 1.5 --max-open-trades 1
```

## Phased rollout (each independently shippable)
1. **Phase 1 — config surface, behavior-preserving.** Create both files; `TradingConfig` owns costs/risk; backtester seeds `PositionState` cost; refactor the 54 exit calls; wire CLI. **Golden test: P&L identical.** ← the core.
2. **Phase 2 — sweep ergonomics + notebooks.** `sweep()` + non-colliding `data/sweeps/` outputs; migrate the 9 notebooks to `build_config`/`ACTIVE_TRADE`.
3. **Phase 3 — activate trade features (optional).** Direction gate in `PositionState.enter`; then position sizing/equity consuming `risk_per_trade_pct`/`max_open_trades` (bigger backtester change).

## Tests
- `test_params_keys_valid` — every `PARAMS` key is a real `StrategyConfig` field.
- **Golden-output test** — same strategy+data+params, pre- vs post-cost-move: P&L, trade count, all metrics identical.
- `sweep()` returns expected grid size + columns; writes a non-colliding file.
- `TradingConfig` validation (direction enum, positive costs); CLI builds it correctly.
- Phase 3: direction gate suppresses the disallowed side.

## Risks
- **Cost-move changing numbers** — mitigated by the golden test (pure subtraction, set once on `PositionState`).
- **Removing 4 fields from `StrategyConfig` is breaking** — first grep for any `StrategyConfig(fee_bps=…)` / `.fee_bps` / `.risk_per_trade_pct` usages (notebooks/tests) and migrate them; expected few (notebooks use `StrategyConfig()` + `replace` on strategy fields only).
- **Direction gate** — must live in `PositionState.enter` (caveat above); keep it Phase 3, default `BOTH`, so nothing changes until opted in.

## Scope mapping (user requirements → plan)
1. *Strategy params* → `strategy_configurator.PARAMS`.
2. *Trading params (direction/sizing/risk/stop)* → `trade_configurator.TradingConfig`.
3. *Common + extendable* → adding a strategy = one `PARAMS` block (+ its fields already typed in `StrategyConfig`).
4. *Sweep without CLI, non-colliding results* → `sweep(strategy, df)` in a notebook → DataFrame + `data/sweeps/<sig>.csv`.
