"""Single source of truth for *strategy-side* configuration.

This is the strategy half of the config split (the trade half is
``trade_configurator.py``). It holds, in one place:

  * :class:`StrategyConfig` — *how signals are generated*: indicator periods,
    multipliers, and per-strategy structural knobs. (Moved here from
    ``core.py``; ``core`` now holds only domain types + the state machine.)
  * the **exit/TP policy catalog** — named presets built from the mechanisms in
    ``engine.exits``, plus a string-keyed per-strategy assignment. This is how
    stop-loss / take-profit behaviour is selected and injected into a strategy
    (``BaseStrategy`` sets ``self.exit_policy = exit_policy_for(self.name)``; the
    CLI's ``--exit-preset`` can override it).

Import-graph note: this module may import ``exits`` (and therefore ``core``)
but must NEVER import strategy classes — per-strategy assignment is keyed by the
string ``name``, so the arrow stays ``strategies → strategy_configurator →
exits → core`` with no cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exits import (
    AtrStop,
    ChandelierStop,
    CloseCrossTarget,
    CompositeExit,
    ExitPolicy,
    FixedPctStop,
    RrTarget,
    StructuralStop,
    StructuralTarget,
    FixedPctTarget,
)


# ── Strategy parameters ──────────────────────────────────────────────────────
# This is where you tune how signals are generated (indicator periods, multipliers, structural levels).
#
# NOTE: trade-level parameters (costs, sizing, direction, leverage, risk overlays)
# intentionally do NOT live here — they describe *how you trade* any signal,
# not how signals are generated. They live on TradingConfig in
# engine/trade_configurator.py, are seeded onto PositionState by the runner,
# and applied at exit (cost) / entry (direction + daily-loss gates).

@dataclass(frozen=True)
class StrategyConfig:
    """Immutable strategy parameters.  All magic numbers live here."""

    # Level-breakout strategy: N-bar fractal-pivot S/R detection
    left: int = 5
    right: int = 5
    merge_tolerance: float = 0.0015  # 0.15 %

    # EMA crossover
    ema_fast: int = 9
    ema_slow: int = 21

    # RSI — momentum filter on the EMA-cross entries (ema / ema_inv).
    # rsi_filter is the master on/off; when off, EMA crosses enter unfiltered.
    # The bounds gate by the *resulting entry direction* (not the cross), so the
    # same two knobs apply identically to the base and inverse strategies:
    #   long  entries are skipped when rsi >= rsi_bullish (overbought)
    #   short entries are skipped when rsi <= rsi_bearish (oversold)
    rsi_period: int = 14
    rsi_filter: bool = True
    rsi_bullish: float = 70.0
    rsi_bearish: float = 30.0

    # SuperTrend
    supertrend_period: int = 10
    supertrend_mult: float = 3.0

    # ATR
    atr_period: int = 14
    # NOTE: the trend/EMA/swing trailing-stop *distance* is not a field here — it
    # lives in the exit catalog below (EXIT_PRESETS["chandelier_2atr"] =
    # ChandelierStop(2.0)), the single source of truth for that stop.

    # Exhaustion-reversal strategy
    exhaustion_push_min_len: int = 2        # min |length| of the push-leg streak
    exhaustion_stall_min_count: int = 2     # min number of stall streaks after the push
    exhaustion_stall_max_len: int = 1       # max |length| of a stall streak
    exhaustion_trigger_min_len: int = 2     # min |length| of the in-progress trigger streak
    exhaustion_volume_factor: float = 1.0   # trigger per-candle vol / push per-candle vol
    exhaustion_stop_atr_mult: float = 0.25  # stop buffer above/below cluster, in ATRs
    exhaustion_target_rr: float = 2.0       # fixed reward:risk target multiple
    exhaustion_time_stop_bars: int = 12     # force-exit after this many bars in trade
    exhaustion_invalidation_len: int = 3    # structural invalidation streak length

    # Impulse + consolidation ("flag") strategy
    flag_body_mult: float = 1.5             # impulse body >= mult * SMA(body)
    flag_vol_mult: float = 1.2              # impulse volume >= mult * SMA(vol)
    flag_close_pos_min: float = 2.0 / 3.0   # impulse close must sit in top/bottom third
    flag_min_cluster: int = 2
    flag_max_cluster: int = 4
    flag_cluster_body_ratio: float = 0.5    # cluster bar body <= ratio * impulse body
    flag_cluster_range_ratio: float = 0.7   # cluster H-L <= ratio * impulse H-L
    flag_retrace_limit: float = 0.5         # cluster must hold above/below 50% retrace
    flag_breakout_window: int = 3           # bars after cluster to fill the entry
    flag_ema_fast: int = 20
    flag_vol_sma: int = 9
    flag_body_sma: int = 20
    flag_ema_slope_lookback: int = 5
    flag_htf_minutes: int = 60
    flag_htf_ema: int = 20
    flag_htf_slope_lookback: int = 3
    flag_level_proximity_pct: float = 0.003  # Track A skip if within 0.3% of 24H hi/lo
    flag_level_lookback_hours: float = 24.0  # rolling-level window, converted to bars at runtime
    flag_stop_atr_mult: float = 0.5          # stop buffer = mult * ATR
    flag_min_rr: float = 1.5                 # gate on averaged target
    flag_enable_track_a: bool = True
    flag_enable_track_b: bool = True
    flag_t1_r: float = 1.0                   # T1 distance in R-multiples
    flag_t2_r: float = 2.0                   # T2 distance in R-multiples
    flag_use_measured_move: bool = True      # T2 = max(2R, impulse range projection)
    flag_be_shift_after_t1: bool = True      # move stop to BE once T1 is touched
    flag_bar_tick: float = 0.1               # trigger offset past the level

    # Order-block strategy
    ob_body_mult: float = 1.8                # impulse body >= mult * ATR
    ob_consecutive_min: int = 3              # min same-direction candle run
    ob_consecutive_range_pct: float = 0.006  # combined body / close threshold
    ob_max_age_hours: float = 3.0            # OBs older than this are discarded
    ob_htf_minutes: int = 60                 # HTF resample for bias filter
    ob_htf_ema: int = 20                     # EMA period on HTF bias
    ob_ema_fast: int = 9                     # LTF fast EMA (confirmation cross)
    ob_ema_slow: int = 20                    # LTF slow EMA (confirmation cross)
    ob_stop_buffer_pct: float = 0.005        # stop buffer beyond OB extreme
    ob_rr: float = 2.5                       # fixed reward:risk on target

    # VWAP Stdev Bands strategy (defaults mirror the TradingView "v2 Mod" preset)
    vwap_band_devs: tuple[float, ...] = (1.28, 2.01, 2.51, 3.09, 4.01)
    vwap_session: str = "D"                  # session anchor for VWAP reset
    vwap_entry_band: int = 4                 # 0-indexed; default = furthest band

    # ATR-prominence ZigZag swing family (flip / bounce / breakout): shared detection knobs.
    # See engine/swing_detector.py for the detector contract.
    swing_zz_atr_period: int = 14
    swing_zz_min_prominence_atr: float = 1.5
    swing_zz_min_bars_between: int = 3
    swing_zz_vol_lookback: int = 50
    swing_zz_min_score: float = 0.0
    swing_zz_use_stop: bool = True  # toggle only; trail distance = PER_STRATEGY_EXIT preset (chandelier_3atr)

    # ML swing-pivot classifier strategy.
    # ml_model_path is resolved relative to the repo root if not absolute.
    ml_model_path: str = "ml_models/swing_zz_ml.joblib"
    ml_p_threshold: float = 0.55
    ml_use_stop: bool = True  # toggle only; trail distance = PER_STRATEGY_EXIT preset (chandelier_3atr)

    # EMA touch-and-rejection strategy (ported from the ema project's backtest.py).
    # Entry: a bar's wick touches the entry EMA within `delta` tolerance AND the
    # bar closes back on the rejection side (long: close >= EMA; short: close <=
    # EMA); optional slower-EMA regime gate. Stop/TP come from the exit policy
    # (default preset fixed_1pct_rr3 = 1% stop + 3R target).
    ema_touch_period: int = 50                      # entry EMA span (symmetric fallback)
    ema_touch_period_long: int | None = None        # per-side entry EMA override (None -> ema_touch_period)
    ema_touch_period_short: int | None = None
    ema_touch_delta: float = 40.0                   # touch tolerance magnitude; units set by delta_mode
    ema_touch_delta_mode: str = "absolute"          # "absolute" (quote points) | "percent" (% of EMA) | "atr" (×ATR)
    ema_touch_atr_period: int = 14                  # ATR period (delta_mode='atr' + any ATR-based exit preset)
    ema_touch_regime_filter: int | None = None      # optional slower regime EMA (longs only above it, shorts below)
    ema_touch_regime_filter_long: int | None = None
    ema_touch_regime_filter_short: int | None = None

    # Swing bounce strategy (mean-reversion off confirmed swing pivots; ported
    # from the ema project's swing_strategy.py bounce mode). Detection reuses the
    # ATR-prominence ZigZag knobs above (swing_zz_atr_period / _min_prominence_atr
    # / _min_bars_between / _vol_lookback / _min_score); these are bounce-specific.
    swing_bounce_test_tolerance_atr: float = 0.5        # how close (in ATR) the wick must come to the swing level
    swing_bounce_require_close_rejection: bool = True   # bar must close back across the level
    swing_bounce_stop_atr_mult: float = 1.0             # entry stop = swing price ∓ mult·ATR (swing-anchored)
    swing_bounce_min_bars_between_trades: int = 0       # cooldown bars after an exit before re-entry

    # Swing breakout strategy (continuation through confirmed swing pivots; ported
    # from the ema project's swing_strategy.py breakout mode). Reuses swing_zz_* above.
    swing_breakout_buffer_atr: float = 0.0              # close must clear the swing level by this many ATR
    swing_breakout_stop_atr_mult: float = 1.0           # entry stop = swing level ∓ mult·ATR (swing-anchored)
    swing_breakout_min_bars_between_trades: int = 0     # cooldown bars after an exit before re-entry


# ── Exit / take-profit policy catalog ──────────────────────────────────────────
# This is where SL/TP behaviour is defined and assigned.
#
# Wired into the strategies via exit_policy_for() below: BaseStrategy injects the
# resolved policy as self.exit_policy (overridable by the CLI's --exit-preset).

# Named presets that build a stop+target from the mechanisms in engine/exits.py
# Each value is a preset so a fresh policy is created per use.
# Add a new SL/TP combo here.
# ExitPolicy:
# — either a single mechanism (a trailing ChandelierStop, or a TP-only CloseCrossTarget)
# - or a CompositeExit(stop, target) that runs a stop then a target (stop-first)
EXIT_PRESETS: dict[str, "callable[[], ExitPolicy]"] = {
    
    # SL-only presets: trailing chandeliers + static fixed stops
    "chandelier_2atr": lambda: ChandelierStop(2.0),    # matches today's trend-strategy trail
    "chandelier_3atr": lambda: ChandelierStop(3.0),
    "fixed_2pct": lambda: FixedPctStop(2.0),   # fixed stop only; position otherwise exits on the strategy's native signal
    "atr_stop":   lambda: AtrStop(1.5),        # fixed (non-trailing) ATR stop only

    # fixed SL + a target
    # List the stop before any target for stop-first precedence on ambiguous bars.
    "atr_stop_rr2":    lambda: CompositeExit(AtrStop(1.5), RrTarget(2.0)),
    "structural_rr2":  lambda: CompositeExit(StructuralStop(), RrTarget(2.0)),
    "fixed_2pct_rr3":  lambda: CompositeExit(FixedPctStop(2.0), RrTarget(3.0)),
    "fixed_1pct_rr3":  lambda: CompositeExit(FixedPctStop(1.0), RrTarget(3.0)),  # ema_touch default (1% stop, 3R)
    
    # fixed SL + fixed TP, both at strategy-supplied levels
    # (ref_stop via the entry stop_price, ref_target passed per bar). Stop-first.
    "structural":      lambda: CompositeExit(StructuralStop(), StructuralTarget()),
    "fixed_2pct_3pct": lambda: CompositeExit(FixedPctStop(2.0), FixedPctTarget(3.0)),
    
    # TP-only presets:
    "fixed_3pct_tp":   lambda: FixedPctTarget(3.0),      # take-profit only, no stop
    # When the close reverts to a level (the VWAP mean). No stop.
    "vwap_mean":       lambda: CloseCrossTarget(),
}

# Global default exit: the fallback policy for the trend/EMA/swing group.
DEFAULT_EXIT = "chandelier_2atr"

# Per-strategy overrides
# (keyed by the strategy's name string, never by importing the class — keeps the import graph acyclic)
# Change which exit a strategy uses here.
PER_STRATEGY_EXIT: dict[str, str] = {
    "order_block": "structural",
    "order_block_inv": "structural",
    "exhaustion_reversal": "structural",
    "impulse_flag": "structural",
    "vwap_bands": "vwap_mean",
    "swing_flip": "chandelier_3atr",        # 3·ATR trail (single source of truth for the distance)
    "swing_ml": "chandelier_3atr",          # 3·ATR trail (single source of truth for the distance)
    "ema_touch": "fixed_1pct_rr3",          # 1% fixed stop + 3R target (matches the source default)
    "swing_bounce": "structural_rr2",       # swing-anchored stop (entry stop_price) + 2R target
    "swing_breakout": "structural_rr2",     # breakout through the swing level, swing-anchored stop + 2R
}

# Resolves the override (else default) into an ExitPolicy
# Finds this strategy's exit recipe (its own if it has one, otherwise the default), and makes a fresh one.
# 3 steps:
# PER_STRATEGY_EXIT.get(name, DEFAULT_EXIT) → the preset name for this strategy, or default "chandelier_2atr" if it has no override.
# EXIT_PRESETS[…] → look up that name's factory (the lambda).
# () → call it to build a brand-new ExitPolicy object.
def exit_policy_for(strategy_name: str) -> ExitPolicy:
    """Build the exit policy assigned to a strategy (its override, else the
    global default). Called by BaseStrategy to seed self.exit_policy."""
    return EXIT_PRESETS[PER_STRATEGY_EXIT.get(strategy_name, DEFAULT_EXIT)]()
