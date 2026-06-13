# Session record — configuration faithfulness audit, fixes, and live-chart upgrade

Date: 2026-06-13

This document is the full record of one working session: the user's questions, the
answers given, and everything that was changed in the codebase as a result. It is
organised in four parts in the order the conversation happened:

1. A configuration-faithfulness audit answering 12 questions.
2. A "fix all" pass that implemented the corrections.
3. A question about whether live mode shows entry/exit points.
4. An upgrade of the live chart to the rich trade view.

Outcome at a glance: every issue the audit surfaced was fixed; the engine test
suite went from 235 to 239 passing tests (4 new tests added, all prior tests stayed
green and byte-identical where behaviour was meant to be unchanged).

---

## Part 1 — Configuration faithfulness audit (12 questions)

### The request

The user asked for very short, simple answers to 12 questions about whether
configuration flows correctly from the three configurator files
(data_configurator.py, trade_configurator.py, strategy_configurator.py) and the
notebooks into every strategy, and whether anything is silently overridden,
hard-coded, or ignored.

### Method

The audit was run as a multi-agent workflow: a foundation pass that mapped the
config/exit/runner plumbing from the core files, then a per-strategy audit across
all 20 strategies (each finding adversarially verified), a per-notebook audit across
all 18 notebooks, and a synthesis pass. 54 agents total. The most consequential
claims were then re-checked by hand against the source before being reported.

### The 12 questions and the answers given

1. Can all strategies be run with custom exit presets, custom strategy params,
   custom data params, and custom trade params?
   - Mostly yes. All 20 accept custom exit presets (via the CLI exit-preset flag or
     an injected exit_policy), custom trade params, and custom data params. The gap:
     strategy signal params have no per-knob CLI flags, so individual signal params
     are changed by editing the params class or constructing them in a notebook, not
     from the CLI.

2. Are all strategies configurable both via the 3 configurators and inside the
   notebook?
   - Mostly yes. All pull defaults from the configurators and can be overridden in
     notebooks via dataclasses.replace. Exceptions: a few signal numbers in
     order_block, order_block_inv, and swing_ml were hard-coded outside their params
     classes, so those specific knobs were not configurable anywhere.

3. Is there anything that overrides or does not forward configurator params into a
   strategy (silent/hard-coded), including inside notebooks?
   - Yes, three spots. order_block and order_block_inv capped the order-block origin
     lookback at a hard-coded 9 bars (range(1,10)); swing_ml hard-coded its
     trailing-stop ATR period to 14 (no ml_atr_period field); order_block silently
     rewrote ob_htf_minutes — but only as a guard when the configured HTF was at or
     below the chart interval (it never fires on defaults).

4. Is there anything that does not forward params set inside each notebook manually?
   - Mostly no — notebook replace() overrides flow into the real Backtester. The
     exceptions: in vwap_bands the bespoke band-sweep cells did not thread the exit
     override (silently ignored), and the bespoke live cells in super_trend
     (multi-strategy loop) and swing_zigzag omitted trading_config, so a notebook
     trade-config override was dropped there.

5. Are there any defaults passed silently into strategies that bypass the user's
   config in the 3 configurators?
   - Yes, a few: swing_ml's ATR-14 trailing stop; order_block's HTF guard (unlogged);
     and framework-wide, any caller that omitted trading_config silently received a
     bare TradingConfig() instead of ACTIVE_TRADE, and live mode ignored all
     sizing/equity fields.

6. When strategies are configured inside their notebooks, does that override the
   settings in the 3 configurator files?
   - Yes — that is the design. Each notebook starts from the configurator defaults
     (ACTIVE, params_for(name), ACTIVE_TRADE) and applies replace() on top, then
     forwards the result into the run. In the standard notebooks the override slots
     ship empty, so the effective config equals the configurator defaults until a
     user fills them.

7. Do all strategies have configurable exits wired?
   - Yes. All 20 delegate their price stop/target to self.exit_policy.evaluate(...)
     and keep only signal-based exits (cross/flip) native.

8. Do all strategies accept their own exit_policy from strategy_configurator.py?
   - Yes. All 20 default to exit_policy_for(name) in BaseStrategy, and import-time
     guards require every strategy to have a valid exit.

9. Do all strategies override the default exit when a specific one is set for them?
   - Only the 12 that are assigned a non-default preset do (for example
     level_breakout to structural_rr2, ema_touch to fixed_1pct_rr3, vwap_bands to
     vwap_mean, swing_flip and swing_ml to chandelier_3atr). The other 8 (ema x3,
     supertrend x3, fractal_breakout x2) are explicitly assigned the default, so
     there is nothing to override.

10. Is exit_policy being passed when running a strategy live? What exit does it use?
    - It is not passed again live — the policy lives on the strategy instance, and
      LiveEngine runs the same object, so live uses the identical exit as backtest
      (each strategy's assigned preset, or a CLI/notebook override).

11. Do all strategies have backtesting, walk-forward, and live signals wired (in
    notebooks)?
    - Mostly. The 11 standard single-family notebooks wire all of them. Exceptions:
      ema_values and ema_values_inv (sweep-only), strategy_comparison (backtest-only),
      levels (detector explorer, no backtest), swing_ml and swing_ml_t3 (no Monte
      Carlo; t3 has no live), swing_zigzag (no walk-forward / Monte Carlo), intro
      (pure docs).

12. Is there anything that is silently ignored?
    - Yes: live mode dropped all sizing/equity fields (no equity layer live);
      order_block's HTF guard was unlogged; vwap_bands' bespoke sweep cells ignored
      exit overrides; and the bespoke live cells in super_trend and swing_zigzag
      omitted trading_config (falling back to TradingConfig() defaults and a
      hard-coded "15" interval).

### Two audit claims corrected after hand-verification

- The audit claimed swing_zigzag had a broken stale StrategyConfig import. This was
  false — there is no such import; swing_zigzag uses SwingParams, load_data(), and
  trading_config=ACTIVE_TRADE correctly. Its only real issue was the live cell
  (omitted trading_config, hard-coded "15").
- The audit framed order_block's HTF rewrite as a routine silent override. In fact
  it is a misconfiguration guard that only fires when ob_htf_minutes is at or below
  the chart interval; on default config (60-minute HTF on a 15m/5m chart) it never
  triggers.

---

## Part 2 — Fix all

### The request

The user asked to fix everything, and to confirm whether the fixes would also solve
problems 2, 3, 4, 5, and 12, with these specific sub-questions:

- 2: which signal numbers in order_block / order_block_inv / swing_ml are hard-coded
  outside their params classes, so those knobs are not configurable anywhere?
- 3: the three hard-coded spots (origin lookback 9 bars; swing_ml ATR 14; the
  order_block HTF guard).
- 4: fix the notebook overrides that are not forwarded.
- 5: how to fix the bare-TradingConfig() default and live mode ignoring sizing/equity.
- 12: how to fix the silently-ignored items.

### Does fixing it solve problems 2, 3, 4, 5, 12? Yes — mapping

| Problem | Fix |
|---|---|
| 2 / 3 — hard-coded signal numbers not on any params class | Added ob_origin_lookback (replaces range(1,10)) and ml_atr_period (replaces wilder_atr(df, 14)). Both default to the old value, so behaviour is unchanged but the knobs are now configurable and overridable. The order_block HTF 60/3x fallback is now a logged warning instead of silent. |
| 3 — order_block silently rewrote ob_htf_minutes | Now logs a warning when it bumps a mis-set HTF (only fires when ob_htf_minutes is at or below the interval; never on defaults). |
| 4 — vwap_bands sweep cells dropped the manual exit override | Cells 38/41/44 now pass exit_policy=EXIT_POLICY. |
| 5 / 12 — omitting trading_config gave a bare TradingConfig() instead of ACTIVE_TRADE | Backtester and LiveEngine now default to ACTIVE_TRADE. |
| 5 / 12 — live dropped all sizing/equity fields | Live now runs the same compounding equity layer via a shared TradingConfig.size_notional; it fills notional / pnl_currency / equity_after per trade and persists a restart-safe paper-equity ledger in live_records.py. |
| 12 — super_trend + swing_zigzag live cells omitted trading_config and hard-coded "15" | Both rewired to the 3 configurators (symbol/interval from the data spec, trading_config, and exit_policy where applicable). |

### The specific hard-coded numbers (answer to Q2/Q3)

- order_block / order_block_inv — order-block origin walk-back was capped at
  range(1, 10), i.e. 9 bars; now ob_origin_lookback. HTF fallback was
  max(60, 3 * ltf); now logged.
- swing_ml — trailing-stop ATR fixed at wilder_atr(df, 14); now ml_atr_period.

### What changed — engine

New configurable knobs in strategy_configurator.py:

```python
# OrderBlockParams
ob_origin_lookback: int = 10   # max bars walked back to find the OB origin candle
# validated with cv.positive_int(o, "ob_origin_lookback", self.ob_origin_lookback)

# SwingMlParams
ml_atr_period: int = 14        # Wilder ATR period for the trailing-stop input (ml_atr)
# validated with cv.positive_int(o, "ml_atr_period", self.ml_atr_period)
```

- order_block.py and order_block_inv.py: use self.config.ob_origin_lookback in the
  origin walk-back loop, and log a warning when the HTF guard bumps a mis-set
  ob_htf_minutes.
- swing_ml.py: both prepare() paths now use self.config.ml_atr_period instead of a
  literal 14.
- trade_configurator.py: added a shared sizing helper so the backtester and live
  engine size trades identically:

```python
def size_notional(self, equity, entry_price, stop_price) -> tuple[float, bool]:
    """Notional for one trade and whether it fell back to fixed-fraction.
    RISK mode sizes from the entry stop; with no usable stop it falls back to
    fixed-fraction (second element True). FIXED mode never falls back."""
    if self.sizing_mode == SizingMode.RISK:
        risk_n = self.risk_notional(equity, entry_price, stop_price)
        if risk_n is not None:
            return risk_n, False
        return self.notional(equity), True
    return self.notional(equity), False
```

- backtester.py: default trading_config is now ACTIVE_TRADE (was a bare
  TradingConfig()); _notional_for delegates to size_notional.
- live.py: default trading_config is now ACTIVE_TRADE; added an incremental,
  restart-safe paper-equity layer (_apply_equity_to_new_closures) that sizes each
  newly closed trade exactly once, compounds equity, and persists it.
- live_records.py: added an account table (single-row equity ledger) plus
  notional / pnl_currency / equity_after columns on trade_history, with
  backward-compatible ALTER statements; added load_equity, save_equity, and
  known_trade_ids; save_trade and load_trade_history carry the currency fields.

Why this is behaviour-preserving where it should be: both new knobs default to the
exact old hard-coded values, and ACTIVE_TRADE's field values currently equal the
bare TradingConfig() defaults, so the golden snapshots and all existing numeric
tests stayed byte-identical.

### What changed — notebooks

All notebook edits were applied as surgical, minimal-diff text edits (json round-trip
with sort_keys=False, ensure_ascii=True), so only the target cells changed — no cell
ids injected and no output churn.

- vwap_bands.ipynb cells 38, 41, 44: the bespoke band-sweep cells now build
  VWAPBandsStrategy(cfg_k, exit_policy=EXIT_POLICY) so a manual exit override in the
  configuration chapter is honoured.
- super_trend.ipynb cell 84 (multi-strategy live): removed the hard-coded SYMBOL,
  INTERVAL, config, and SupertrendParams import; the three Supertrend variants are
  built from STRATEGY_CONFIG with exit_policy=EXIT_POLICY, and each LiveEngine gets
  trading_config=TRADING_CONFIG. Symbol/interval come from the configured values.
- swing_zigzag.ipynb cell 44 (live): symbol/interval now come from ACTIVE
  (data configurator) and trading_config from ACTIVE_TRADE (trade configurator).

### Verification

- Full suite: 237 passed (235 prior + 2 new live-equity tests). Golden and
  config-propagation tests unchanged.
- Confirmed no remaining wilder_atr(df, 14) or range(1, 10) in the strategies, and
  neither runner defaults to a bare TradingConfig() anymore.
- Confirmed the new knobs flow through params_for(name) and remain overridable via
  dataclasses.replace, with the foreign-knob safety intact.

### Out of scope (noted, not changed)

The two ML notebooks (swing_ml, swing_ml_t3) still use a hard-coded
symbol/interval/date window and a direct fetcher — the sanctioned ML exception per
CLAUDE.md. Folding those onto the data configurator was offered as a follow-up.

---

## Part 3 — Does live mode show entry/exit points?

The user asked whether live mode shows entry/exit points. Answer: yes.

- Each state.enter() / state.exit() appends a Signal (entry/exit, direction, price,
  label) in core.py.
- LiveEngine._tick passes those to build_chart(...), which writes an auto-refreshing
  HTML chart at chart_path each poll.
- build_chart renders them as labeled candlestick markers — entry triangles and exit
  markers by direction, tagged like "Long Entry #1" / "Long Exit #1".

Caveats noted at the time: live used the simpler signal view (no per-exit-reason
coloring, no entry-to-exit path lines, no currency P&L on hover), and markers only
appear for the visible window (last ~500 closed bars), with the signal list capped
at 500. Since live now populates per-trade currency fields, an upgrade to the richer
trade view was offered — which became Part 4.

---

## Part 4 — Upgrade the live chart to the rich trade view

The user accepted the offered upgrade.

### What changed

- live.py: _tick now calls build_chart(..., trades=...) instead of the plain signal
  markers. A new helper _chart_trades(window_start) assembles every closed trade
  whose exit falls in the visible candle window, plus the currently open trade if its
  entry is in view. The open trade has no exit_ts, so the trade view draws only its
  entry marker — the live "you are here" cue — with no exit/path. Filtering to the
  window keeps the x-axis pinned to the candles.
- visualization.py: the exit hover now shows both bps and currency, e.g.
  "P&L +988.00 bps / $+988.00". This is additive and also improves backtest charts,
  since the equity layer already populates pnl_currency.
- test_live.py: added two tests for the window/open-entry assembly.

Result: the live chart now shows closed trades coloured by exit reason with bps and
$ P&L on hover and entry-to-exit path lines, plus the open position's entry marker.
This was verified by rendering a real chart through _tick and confirming the exit
markers, the currency hover, the trade-path line, and the open-position entry marker
were all present.

### Verification

- Full suite: 239 passed (the 2 Part-2 live-equity tests plus 2 new chart tests).

### Follow-ups offered (not done)

- Hydrate the live chart's closed trades from the SQLite trade_history so the rich
  view survives restarts (currently closed-trade markers are in-session only, the
  same as the old signal chart).
- Note that the $ figure is the quote currency (USDT) and is forward-test/paper P&L —
  live places no real orders.

---

## Files changed this session

| File | Change |
|---|---|
| engine/strategy_configurator.py | Added ob_origin_lookback and ml_atr_period knobs + validation |
| engine/strategies/order_block.py | Use ob_origin_lookback; log the HTF guard |
| engine/strategies/order_block_inv.py | Use ob_origin_lookback; log the HTF guard |
| engine/strategies/swing_ml.py | Use ml_atr_period instead of literal 14 (both prepare paths) |
| engine/trade_configurator.py | Added shared size_notional helper |
| engine/backtester.py | Default to ACTIVE_TRADE; delegate sizing to size_notional |
| engine/live.py | Default to ACTIVE_TRADE; incremental paper-equity layer; rich trade chart |
| engine/live_records.py | account ledger table + currency columns; load/save equity; known_trade_ids |
| engine/visualization.py | Exit hover shows bps and $ P&L |
| engine/tests/test_live.py | New tests: live equity layer + chart trade assembly |
| strategy_notebooks/vwap_bands.ipynb | Sweep cells 38/41/44 thread exit_policy |
| strategy_notebooks/super_trend.ipynb | Multi-strategy live cell wired to the 3 configurators |
| strategy_notebooks/swing_zigzag.ipynb | Live cell wired to ACTIVE + ACTIVE_TRADE |

## New configurable knobs — quick reference

- ob_origin_lookback (OrderBlockParams, default 10): max bars walked back from an
  impulse to find the order-block origin candle. Used by order_block and
  order_block_inv.
- ml_atr_period (SwingMlParams, default 14): Wilder ATR period for the swing_ml
  trailing-stop input (ml_atr). Ignored when a notebook injects its own ml_atr column.

Both are set the usual way — edit the defaults in strategy_configurator.py, or in a
notebook via dataclasses.replace(params_for(name), ob_origin_lookback=...).

## Behaviour notes worth remembering

- Backtester and LiveEngine now inherit ACTIVE_TRADE when trading_config is omitted,
  so a bare construction no longer diverges from the project-wide trade defaults.
- Live mode runs the same sizing/equity layer as the backtester, applied
  incrementally and persisted (restart-safe), but it places no real orders — the
  equity curve is forward-test/paper P&L.
- The order_block HTF guard only fires on a mis-set HTF (ob_htf_minutes at or below
  the chart interval) and now logs a warning when it does.
