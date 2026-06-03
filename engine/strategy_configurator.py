"""Single source of truth for *strategy-side* configuration.

This is the strategy half of the config split (the trade half is
``trade_configurator.py``). It holds, in one place:

  * :class:`StrategyConfig` — *how signals are generated*: indicator periods,
    multipliers, and per-strategy structural knobs. (Moved here from
    ``core.py``; ``core`` now holds only domain types + the state machine.)
  * the **exit/TP policy catalog** — named presets built from the mechanisms in
    ``engine.exits``, plus a string-keyed per-strategy assignment. This is how
    stop-loss / take-profit behaviour is selected and (later) injected into a
    strategy.

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
)


# ── Strategy parameters ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategyConfig:
    """Immutable strategy parameters.  All magic numbers live here."""

    # Swing detection
    left: int = 5
    right: int = 5
    merge_tolerance: float = 0.0015  # 0.15 %

    # EMA crossover
    ema_fast: int = 9
    ema_slow: int = 21

    # RSI
    rsi_period: int = 14

    # SuperTrend
    supertrend_period: int = 10
    supertrend_mult: float = 3.0

    # ATR / risk
    atr_period: int = 14
    atr_trail_mult: float = 2.0

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

    # ATR-prominence ZigZag swing strategy
    # See engine/swings.py for the detector contract.
    swing_zz_atr_period: int = 14
    swing_zz_min_prominence_atr: float = 1.5
    swing_zz_min_bars_between: int = 3
    swing_zz_vol_lookback: int = 50
    swing_zz_min_score: float = 0.0
    swing_zz_use_stop: bool = True
    swing_zz_stop_atr_mult: float = 3.0

    # ML swing-pivot classifier strategy.
    # ml_model_path is resolved relative to the repo root if not absolute.
    ml_model_path: str = "ml_models/swing_zz_ml.joblib"
    ml_p_threshold: float = 0.55
    ml_use_stop: bool = True
    ml_stop_atr_mult: float = 3.0

    # NOTE: trade-level parameters (costs, sizing, direction, leverage, risk
    # overlays) intentionally do NOT live here — they describe *how you trade*
    # any signal, not how signals are generated. They live on TradingConfig in
    # engine/trade_configurator.py, are seeded onto PositionState by the runner,
    # and applied at exit (cost) / entry (direction + daily-loss gates).


# ── Exit / take-profit policy catalog ──────────────────────────────────────────
# Named presets built from engine.exits mechanisms. Each value is a factory so a
# fresh policy is created per use. List the stop before any target for stop-first
# precedence on ambiguous bars.
#
# NOTE: not yet wired into the strategies — that's the per-strategy conversion
# phase. Defined here now so stop/TP selection and strategy params share one file.

EXIT_PRESETS: dict[str, "callable[[], ExitPolicy]"] = {
    "chandelier_2atr": lambda: ChandelierStop(2.0),    # matches today's trend-strategy trail
    "chandelier_3atr": lambda: ChandelierStop(3.0),
    "atr_stop_rr2":    lambda: CompositeExit(AtrStop(1.5), RrTarget(2.0)),
    "structural_rr2":  lambda: CompositeExit(StructuralStop(), RrTarget(2.0)),
    "fixed_2pct_rr3":  lambda: CompositeExit(FixedPctStop(2.0), RrTarget(3.0)),
    # Fixed stop + fixed target, both at strategy-supplied levels (ref_stop via
    # the entry stop_price, ref_target passed per bar). Stop-first.
    "structural":      lambda: CompositeExit(StructuralStop(), StructuralTarget()),
    # Take-profit only, when the close reverts to a level (the VWAP mean). No stop.
    "vwap_mean":       lambda: CloseCrossTarget(),
}

# Global default exit (covers the trend/EMA/swing group), with per-strategy
# overrides (string-keyed by the strategy's `name`, never by importing the class
# — keeps the import graph acyclic).
DEFAULT_EXIT = "chandelier_2atr"
PER_STRATEGY_EXIT: dict[str, str] = {
    "order_block": "structural",
    "order_block_inv": "structural",
    "exhaustion_reversal": "structural",
    "impulse_flag": "structural",
    "vwap_bands": "vwap_mean",
    "swing_zigzag": "chandelier_3atr",      # swing_zz_stop_atr_mult default
    "swing_zigzag_ml": "chandelier_3atr",   # ml_stop_atr_mult default
}


def exit_policy_for(strategy_name: str) -> ExitPolicy:
    """Build the exit policy assigned to a strategy (its override, else the
    global default). Not yet called by the runner — see the rollout plan."""
    return EXIT_PRESETS[PER_STRATEGY_EXIT.get(strategy_name, DEFAULT_EXIT)]()
