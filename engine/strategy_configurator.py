"""Single source of truth for *strategy-side* configuration.

This is the strategy half of the config split (the trade half is
``trade_configurator.py``). It is organised top-to-bottom into four sections:

  * **Section 1 — Per-strategy parameter classes**: one small frozen dataclass
    per strategy *family*, holding EXACTLY the signal knobs that family reads
    (indicator periods, multipliers, structural levels) PLUS its default exit
    assignment (the ``EXITS`` ClassVar). A foreign override (e.g.
    ``supertrend_mult`` on an EMA strategy) therefore raises ``TypeError`` at
    ``dataclasses.replace`` instead of landing silently inert, and each family
    owns its own ``atr_period``. Field validation is delegated to
    ``config_validation`` (the shared rule home).
  * **Section 2 — Exit / take-profit policy catalog**: the reusable preset menu
    (``EXIT_PRESETS``), each entry a factory built from the mechanisms in
    ``exits.py``. A strategy assigns one by key in its class ``EXITS`` (Section 1).
  * **Section 3 — Params registry + exit resolution**: ``PARAMS`` maps every
    strategy name → its config class; ``PER_STRATEGY_EXIT`` is *derived* from each
    class's ``EXITS`` (the central "glance view"); ``exit_policy_for()`` resolves a
    strategy's assigned preset live from its class ``EXITS``. ``BaseStrategy`` sets
    ``self.exit_policy = exit_policy_for(self.name)``; the CLI's ``--exit-preset``
    (and a notebook's ``EXIT_POLICY``) can override it.
  * **Section 4 — Params accessors**: ``params_for(name)`` (default config for a
    strategy) and ``params_class_for(name)`` (the class it expects).

Import-graph note: this module may import ``exits`` (and therefore ``core``)
but must NEVER import strategy classes — per-strategy wiring is keyed by the
string ``name``, so the arrow stays ``strategies → strategy_configurator →
exits → core`` with no cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

from . import config_validation as cv
from .core import StrategyName
from .levels import LEVEL_SOURCE_NAMES
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

logger = logging.getLogger(__name__)

# Each strategy's default exit-preset assignment lives ON its config class (the
# EXITS ClassVar), so a family's signal knobs and its exit sit together.
# DEFAULT_EXIT is the shared fallback those maps lean on — defined up here so the
# classes can reference it. The reusable preset menu (EXIT_PRESETS) stays central
# in Section 2; PER_STRATEGY_EXIT and the resolver (exit_policy_for) live in
# Section 3, with PER_STRATEGY_EXIT derived from the class EXITS maps (Section 1).
DEFAULT_EXIT = "chandelier_2atr"   # global fallback for the trend/EMA/swing group
DEFAULT = DEFAULT_EXIT             # readability alias for the no-override entries


# ════════════════════════════════════════════════════════════════════════════
# Section 1 — Per-strategy parameter classes
# ════════════════════════════════════════════════════════════════════════════
# One frozen dataclass per strategy family. Fields = exactly the knobs that
# family reads. The `atr_period` knob is declared per-family (independent now);
# families that carry their own ATR field (ema_touch / level / swing) do not.
# Validation (Section: config_validation) rejects only genuinely invalid values
# (negatives, zero where positive is required, bad categoricals, out-of-range
# indices, empty required strings) — NO upper caps and NO cross-field ordering,
# so sweeps roam freely (ema_fast >= ema_slow, big/small magnitudes, etc.).


# ──────────────────────────────────────────────────────────────────────────────
# EMA + RSI   ·   ema, ema_inv
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EmaParams:
    """EMA crossover entries, optionally gated by an RSI momentum filter.

    rsi_filter is the master on/off; when off, EMA crosses enter unfiltered. The
    bounds gate by the *resulting entry direction* (not the cross), so the same
    two knobs apply identically to the base and inverse strategies:
      long  entries are skipped when rsi >= rsi_bullish (overbought)
      short entries are skipped when rsi <= rsi_bearish (oversold)
    """

    atr_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_filter: bool = True
    rsi_bullish: float = 70.0
    rsi_bearish: float = 30.0

    EXITS: ClassVar[dict[str, str]] = {
        "ema":          DEFAULT,             # chandelier_2atr
        "ema_inv":      "chandelier_2atr",
        "ema_adaptive": DEFAULT,             # RSI regime switch; same ATR trail
    }

    def __post_init__(self) -> None:
        o = "EmaParams"
        cv.positive_int(o, "atr_period", self.atr_period)
        cv.positive_int(o, "ema_fast", self.ema_fast)
        cv.positive_int(o, "ema_slow", self.ema_slow)
        cv.positive_int(o, "rsi_period", self.rsi_period)
        cv.non_negative_number(o, "rsi_bullish", self.rsi_bullish)
        cv.non_negative_number(o, "rsi_bearish", self.rsi_bearish)


# ──────────────────────────────────────────────────────────────────────────────
# SuperTrend   ·   supertrend, supertrend_inv, supertrend_adaptive
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SupertrendParams:
    """ATR volatility bands + trend direction. The adaptive variant uses an ADX
    regime switch: ADX >= adx_threshold = trending (follow) else ranging (fade).
    Both the ADX lookback (adx_period) and the regime threshold (adx_threshold)
    are configurable; adx_period defaults to 14 (independent of atr_period)."""

    atr_period: int = 14
    supertrend_period: int = 10
    supertrend_mult: float = 3.0
    adx_period: int = 14            # ADX lookback for the adaptive regime switch
    adx_threshold: float = 25.0     # ADX >= this = trending (follow) else ranging (fade)

    EXITS: ClassVar[dict[str, str]] = {
        "supertrend": DEFAULT,
        "supertrend_inv": DEFAULT,
        "supertrend_adaptive": DEFAULT,
    }

    def __post_init__(self) -> None:
        o = "SupertrendParams"
        cv.positive_int(o, "atr_period", self.atr_period)
        cv.positive_int(o, "supertrend_period", self.supertrend_period)
        cv.positive_number(o, "supertrend_mult", self.supertrend_mult)
        cv.positive_int(o, "adx_period", self.adx_period)
        cv.non_negative_number(o, "adx_threshold", self.adx_threshold)


# ──────────────────────────────────────────────────────────────────────────────
# EMA touch-and-rejection   ·   ema_touch
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EmaTouchParams:
    """A bar's wick touches the entry EMA within `delta` tolerance AND the bar
    closes back on the rejection side (long: close >= EMA; short: close <= EMA);
    optional slower-EMA regime gate. Stop/TP come from the exit policy (default
    preset fixed_1pct_rr3 = 1% stop + 3R target). Distinct from the `ema`
    crossover family. Carries its OWN atr period (ema_touch_atr_period)."""

    ema_touch_period: int = 50                      # entry EMA span (symmetric fallback)
    ema_touch_period_long: int | None = None        # per-side entry EMA override (None -> ema_touch_period)
    ema_touch_period_short: int | None = None
    ema_touch_delta: float = 40.0                   # touch tolerance magnitude; units set by delta_mode
    ema_touch_delta_mode: str = "absolute"          # "absolute" (quote points) | "percent" (% of EMA) | "atr" (×ATR)
    ema_touch_atr_period: int = 14                  # ATR period (delta_mode='atr' + any ATR-based exit preset)
    ema_touch_regime_filter: int | None = None      # optional slower regime EMA (longs only above it, shorts below)
    ema_touch_regime_filter_long: int | None = None
    ema_touch_regime_filter_short: int | None = None

    EXITS: ClassVar[dict[str, str]] = {
        "ema_touch": "fixed_1pct_rr3",   # 1% fixed stop + 3R target (source default)
    }

    def __post_init__(self) -> None:
        o = "EmaTouchParams"
        cv.positive_int(o, "ema_touch_period", self.ema_touch_period)
        cv.optional_positive_int(o, "ema_touch_period_long", self.ema_touch_period_long)
        cv.optional_positive_int(o, "ema_touch_period_short", self.ema_touch_period_short)
        cv.positive_number(o, "ema_touch_delta", self.ema_touch_delta)
        cv.one_of(o, "ema_touch_delta_mode", self.ema_touch_delta_mode,
                  ("absolute", "percent", "atr"))
        cv.positive_int(o, "ema_touch_atr_period", self.ema_touch_atr_period)
        cv.optional_positive_int(o, "ema_touch_regime_filter", self.ema_touch_regime_filter)
        cv.optional_positive_int(o, "ema_touch_regime_filter_long", self.ema_touch_regime_filter_long)
        cv.optional_positive_int(o, "ema_touch_regime_filter_short", self.ema_touch_regime_filter_short)


# ──────────────────────────────────────────────────────────────────────────────
# Fractal breakout   ·   fractal_breakout, fractal_breakout_inv
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FractalParams:
    """N-bar fractal-pivot S/R detection via indicators.detect_swing_* +
    merge_price_levels. Distinct from the level_breakout detector."""

    atr_period: int = 14
    left: int = 5
    right: int = 5
    merge_tolerance: float = 0.0015  # 0.15 %

    EXITS: ClassVar[dict[str, str]] = {
        # DEFAULT preserves the byte-for-byte behaviour of the pre-rename
        # level_breakout, which used this chandelier trail.
        "fractal_breakout": DEFAULT,
        "fractal_breakout_inv": DEFAULT,
    }

    def __post_init__(self) -> None:
        o = "FractalParams"
        cv.positive_int(o, "atr_period", self.atr_period)
        cv.positive_int(o, "left", self.left)
        cv.positive_int(o, "right", self.right)
        cv.positive_number(o, "merge_tolerance", self.merge_tolerance)


# ════════════════════════════════════════════════════════════════════════════
# Level strategies   ·   level_breakout, level_breakout_inv, + new strategies
# ════════════════════════════════════════════════════════════════════════════
# All level strategies trade the horizontal S/R levels from the dedicated
# engine.levels package. The DETECTOR CONFIG is shared and central: it lives on
# the LevelParams base below, so the three interchangeable detectors and their
# knobs are defined once. Each concrete strategy then subclasses LevelParams to
# add ONLY its own signal knobs + its EXITS map, and to pick a per-strategy
# detector default. A foreign signal knob therefore raises TypeError at
# dataclasses.replace just like every other family, while the detector seam stays
# DRY. The new g_* strategies port the level-based models (bounce, squeeze
# breakout, false breakout, range) onto these same detectors.
@dataclass(frozen=True)
class LevelParams:
    """Shared, central horizontal-S/R DETECTOR configuration for the level family.

    This is the base every level strategy subclasses; it owns only the detector
    seam (no per-strategy signal knobs). The detector is selectable via
    level_detector (chosen like any other knob, swept like any other field):

      pivot_level   — pivot-seeded, invalidation-tracked (resistance/support/
                      pullback); the base default and the only one with a
                      pullback family.
      cluster_level — merges nearby pivots into one level; dies on a decisive
                      close-through break; reads the cluster_* knobs.
      touch_level   — significance by historical touch count; reads the touch_*
                      knobs.

    All three read the shared level_* knobs (pivot window, ATR period, tolerance);
    the cluster_*/touch_* knobs are read only by their own detector and are inert
    (validated but unused) under the others. Distinct from the fractal_breakout
    detector. Carries its OWN atr period (level_atr_period). Concrete strategies
    (LevelBreakoutParams, GBounceParams, …) inherit this and add their signal
    knobs + EXITS + their preferred detector default."""

    # Detector selection — pivot_level | cluster_level | touch_level.
    level_detector: str = "pivot_level"

    # Shared knobs (every detector maps these).
    level_pivot_window: int = 3             # symmetric pivot window for all detectors
    level_delta: float = 0.5                # tolerance magnitude; units set by level_delta_mode
    level_delta_mode: str = "atr"           # "absolute" (quote pts) | "percent" (% of level) | "atr" (×ATR)
    level_invalidation_candles: int = 3     # pivot_level: bracket count before a level dies
    level_atr_period: int = 14              # ATR period (level_delta_mode='atr' + exit/sizing ATR)
    level_use_pullback: bool = False        # pivot_level: fold the pullback family into the S/R set

    # pivot_level per-family overrides (None -> use the shared knob above). Let the
    # three families carry distinct tolerances / invalidation budgets / pivot
    # windows, the way the standalone detector exposed them. pivot_level-only.
    level_delta_resistance: float | None = None
    level_delta_support: float | None = None
    level_delta_pullback: float | None = None
    level_inval_resistance: int | None = None
    level_inval_support: int | None = None
    level_inval_pullback: int | None = None
    level_pivot_window_resistance: int | None = None
    level_pivot_window_support: int | None = None
    level_pivot_window_pullback: int | None = None

    # cluster_level knobs (ATR-based merge + close-through break).
    cluster_merge_atr_mult: float = 0.5     # merge a pivot into a level within this ×ATR
    cluster_break_atr_mult: float = 0.1     # close must clear the level by this ×ATR to break it
    cluster_max_levels: int = 300           # cap on simultaneously-active levels (oldest retired)

    # touch_level knobs (tolerance units follow level_delta_mode).
    touch_cluster_mult: float = 0.5         # swing-clustering band magnitude
    touch_band_mult: float = 0.75           # how close counts as a touch
    touch_min_touches: int = 3              # touches before a level becomes observable
    touch_recency_bars: int = 0             # 0 = keep all; else drop levels untouched in the last N bars

    def __post_init__(self) -> None:
        o = type(self).__name__
        cv.one_of(o, "level_detector", self.level_detector, LEVEL_SOURCE_NAMES)
        cv.positive_int(o, "level_pivot_window", self.level_pivot_window)
        cv.non_negative_number(o, "level_delta", self.level_delta)
        cv.one_of(o, "level_delta_mode", self.level_delta_mode,
                  ("absolute", "percent", "atr"))
        cv.positive_int(o, "level_invalidation_candles", self.level_invalidation_candles)
        cv.positive_int(o, "level_atr_period", self.level_atr_period)
        for nm in ("level_delta_resistance", "level_delta_support", "level_delta_pullback"):
            cv.optional_non_negative_number(o, nm, getattr(self, nm))
        for nm in ("level_inval_resistance", "level_inval_support", "level_inval_pullback",
                   "level_pivot_window_resistance", "level_pivot_window_support",
                   "level_pivot_window_pullback"):
            cv.optional_positive_int(o, nm, getattr(self, nm))
        cv.non_negative_number(o, "cluster_merge_atr_mult", self.cluster_merge_atr_mult)
        cv.non_negative_number(o, "cluster_break_atr_mult", self.cluster_break_atr_mult)
        cv.positive_int(o, "cluster_max_levels", self.cluster_max_levels)
        cv.non_negative_number(o, "touch_cluster_mult", self.touch_cluster_mult)
        cv.non_negative_number(o, "touch_band_mult", self.touch_band_mult)
        cv.positive_int(o, "touch_min_touches", self.touch_min_touches)
        cv.non_negative_int(o, "touch_recency_bars", self.touch_recency_bars)


@dataclass(frozen=True)
class LevelBreakoutParams(LevelParams):
    """level_breakout — breakout through a horizontal level. Structural stop
    anchored on the broken level (entry stop_price) + 2R target."""

    level_detector: str = "pivot_level"
    level_breakout_buffer_atr: float = 0.0  # close must clear the level by this ×ATR to trigger
    level_stop_atr_mult: float = 1.5        # entry stop = broken level ∓ mult·ATR (structural)

    EXITS: ClassVar[dict[str, str]] = {"level_breakout": "structural_rr2"}

    def __post_init__(self) -> None:
        super().__post_init__()
        o = type(self).__name__
        cv.non_negative_number(o, "level_breakout_buffer_atr", self.level_breakout_buffer_atr)
        cv.non_negative_number(o, "level_stop_atr_mult", self.level_stop_atr_mult)


@dataclass(frozen=True)
class LevelBreakoutInvParams(LevelParams):
    """level_breakout_inv — fade breakouts of a horizontal level. No broken level
    to anchor against on the fade side, so the stop is a plain ATR stop from entry
    (preset atr_stop_rr2)."""

    level_detector: str = "pivot_level"
    level_breakout_buffer_atr: float = 0.0  # close must clear the level by this ×ATR to trigger the fade

    EXITS: ClassVar[dict[str, str]] = {"level_breakout_inv": "atr_stop_rr2"}

    def __post_init__(self) -> None:
        super().__post_init__()
        cv.non_negative_number(type(self).__name__, "level_breakout_buffer_atr",
                               self.level_breakout_buffer_atr)


@dataclass(frozen=True)
class GBounceParams(LevelParams):
    """g_bounce — bounce (rejection) off a horizontal level. Enter on a confirmed
    rejection back into the range (the bar's extreme tests the level within an ATR
    band and the close finishes back on the original side, pressing into the
    level). Structural stop just beyond the level, 3R target. Ported from the
    level-bounce model onto the engine.levels detectors; defaults to touch_level
    (significance by touch count) for stable, multi-touch levels."""

    level_detector: str = "touch_level"
    g_bounce_tol_atr: float = 0.25          # bar extreme must come within this ×ATR of the level
    g_bounce_stop_buffer_atr: float = 0.5   # stop placed beyond the level by this ×ATR

    EXITS: ClassVar[dict[str, str]] = {"g_bounce": "structural_rr3"}

    def __post_init__(self) -> None:
        super().__post_init__()
        o = type(self).__name__
        cv.non_negative_number(o, "g_bounce_tol_atr", self.g_bounce_tol_atr)
        cv.non_negative_number(o, "g_bounce_stop_buffer_atr", self.g_bounce_stop_buffer_atr)


@dataclass(frozen=True)
class GBreakoutParams(LevelParams):
    """g_breakout — squeeze breakout. A breakout through a level that is preceded
    by a tight consolidation (small bars crawling into the level) with oversized
    ("paranormal") bars rejected — the compression pre-break filter the plain
    level_breakout omits. Structural stop beyond the level, 3R target. Ported from
    the level-breakout model; defaults to cluster_level (decisive merged levels)."""

    level_detector: str = "cluster_level"
    g_breakout_buffer_atr: float = 0.05      # close must clear the level by this ×ATR
    g_breakout_stop_atr_mult: float = 0.5    # structural stop = broken level ∓ mult·ATR
    g_breakout_consol_bars: int = 5          # bars of pre-break approach to inspect
    g_breakout_consol_max_atr: float = 0.6   # mean approach-bar range must be <= this ×ATR (tight)
    g_breakout_paranormal_atr: float = 2.0   # reject if any approach bar's range >= this ×ATR

    EXITS: ClassVar[dict[str, str]] = {"g_breakout": "structural_rr3"}

    def __post_init__(self) -> None:
        super().__post_init__()
        o = type(self).__name__
        cv.non_negative_number(o, "g_breakout_buffer_atr", self.g_breakout_buffer_atr)
        cv.non_negative_number(o, "g_breakout_stop_atr_mult", self.g_breakout_stop_atr_mult)
        cv.positive_int(o, "g_breakout_consol_bars", self.g_breakout_consol_bars)
        cv.positive_number(o, "g_breakout_consol_max_atr", self.g_breakout_consol_max_atr)
        cv.positive_number(o, "g_breakout_paranormal_atr", self.g_breakout_paranormal_atr)


@dataclass(frozen=True)
class GBreakoutFalseParams(LevelParams):
    """g_breakout_false — false-breakout reversal. Price pokes through a level but
    fails to hold and closes back inside; trade the reversal. Mode selects how the
    failed break is recognized: single (one-bar poke), two_bar (a bar closes
    beyond then the next reclaims), complex (a multi-bar excursion is reclaimed).
    Structural stop just beyond the false-break extreme, 3R target. Defaults to
    cluster_level, whose decisive close-through break keeps a level alive through a
    failed wick (so the reversal is observable)."""

    level_detector: str = "cluster_level"
    g_breakout_false_mode: str = "single"          # single | two_bar | complex
    g_breakout_false_max_depth_atr: float = 0.5    # max poke depth beyond the level (×ATR)
    g_breakout_false_consol_max_bars: int = 5      # complex mode: max bars on the break side
    g_breakout_false_stop_buffer_atr: float = 0.1  # stop placed beyond the false-break extreme (×ATR)

    EXITS: ClassVar[dict[str, str]] = {"g_breakout_false": "structural_rr3"}

    def __post_init__(self) -> None:
        super().__post_init__()
        o = type(self).__name__
        cv.one_of(o, "g_breakout_false_mode", self.g_breakout_false_mode,
                  ("single", "two_bar", "complex"))
        cv.non_negative_number(o, "g_breakout_false_max_depth_atr", self.g_breakout_false_max_depth_atr)
        cv.positive_int(o, "g_breakout_false_consol_max_bars", self.g_breakout_false_consol_max_bars)
        cv.non_negative_number(o, "g_breakout_false_stop_buffer_atr", self.g_breakout_false_stop_buffer_atr)


@dataclass(frozen=True)
class GRangeParams(LevelParams):
    """g_range — range / channel fade. When two horizontal levels bracket price,
    buy near the lower edge and sell near the upper edge of a wide-enough channel,
    targeting the far edge with a cushion. Structural stop beyond the near edge,
    structural (far-edge) target. Ported from the range-trading model; defaults to
    cluster_level, whose merged pivots give clean, stable channel edges."""

    level_detector: str = "cluster_level"
    g_range_min_width_atr: float = 4.0       # channel must be at least this wide (×ATR) to trade
    g_range_entry_zone: float = 0.30         # act only in the bottom/top this fraction of the channel
    g_range_tol_atr: float = 0.25            # how close to the edge counts as a test (×ATR)
    g_range_stop_buffer_atr: float = 0.5     # stop beyond the near edge (×ATR)
    g_range_target_cushion: float = 0.20     # leave this fraction of the channel before the far edge

    EXITS: ClassVar[dict[str, str]] = {"g_range": "structural"}

    def __post_init__(self) -> None:
        super().__post_init__()
        o = type(self).__name__
        cv.positive_number(o, "g_range_min_width_atr", self.g_range_min_width_atr)
        cv.non_negative_number(o, "g_range_entry_zone", self.g_range_entry_zone)
        cv.non_negative_number(o, "g_range_tol_atr", self.g_range_tol_atr)
        cv.non_negative_number(o, "g_range_stop_buffer_atr", self.g_range_stop_buffer_atr)
        cv.non_negative_number(o, "g_range_target_cushion", self.g_range_target_cushion)


# ──────────────────────────────────────────────────────────────────────────────
# Exhaustion reversal   ·   exhaustion_reversal
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExhaustionParams:
    """Push-leg → stall → trigger exhaustion reversal."""

    atr_period: int = 14
    exhaustion_push_min_len: int = 2        # min |length| of the push-leg streak
    exhaustion_stall_min_count: int = 2     # min number of stall streaks after the push
    exhaustion_stall_max_len: int = 1       # max |length| of a stall streak
    exhaustion_trigger_min_len: int = 2     # min |length| of the in-progress trigger streak
    exhaustion_volume_factor: float = 1.0   # trigger per-candle vol / push per-candle vol
    exhaustion_stop_atr_mult: float = 0.25  # stop buffer above/below cluster, in ATRs
    exhaustion_target_rr: float = 2.0       # fixed reward:risk target multiple
    exhaustion_time_stop_bars: int = 12     # force-exit after this many bars in trade
    exhaustion_invalidation_len: int = 3    # structural invalidation streak length

    EXITS: ClassVar[dict[str, str]] = {
        "exhaustion_reversal": "structural",   # fixed structural stop + structural target
    }

    def __post_init__(self) -> None:
        o = "ExhaustionParams"
        cv.positive_int(o, "atr_period", self.atr_period)
        cv.positive_int(o, "exhaustion_push_min_len", self.exhaustion_push_min_len)
        cv.positive_int(o, "exhaustion_stall_min_count", self.exhaustion_stall_min_count)
        cv.positive_int(o, "exhaustion_stall_max_len", self.exhaustion_stall_max_len)
        cv.positive_int(o, "exhaustion_trigger_min_len", self.exhaustion_trigger_min_len)
        cv.positive_number(o, "exhaustion_volume_factor", self.exhaustion_volume_factor)
        cv.non_negative_number(o, "exhaustion_stop_atr_mult", self.exhaustion_stop_atr_mult)
        cv.positive_number(o, "exhaustion_target_rr", self.exhaustion_target_rr)
        cv.positive_int(o, "exhaustion_time_stop_bars", self.exhaustion_time_stop_bars)
        cv.positive_int(o, "exhaustion_invalidation_len", self.exhaustion_invalidation_len)


# ──────────────────────────────────────────────────────────────────────────────
# Impulse + consolidation ("flag")   ·   impulse_flag
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ImpulseFlagParams:
    """Impulse leg → tight consolidation cluster → breakout, with HTF bias and
    a two-target (T1/T2) trade manager."""

    atr_period: int = 14
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

    EXITS: ClassVar[dict[str, str]] = {
        "impulse_flag": "structural",   # fixed structural stop + structural target
    }

    def __post_init__(self) -> None:
        o = "ImpulseFlagParams"
        for name in ("atr_period", "flag_min_cluster", "flag_max_cluster",
                     "flag_breakout_window", "flag_ema_fast", "flag_vol_sma",
                     "flag_body_sma", "flag_ema_slope_lookback", "flag_htf_minutes",
                     "flag_htf_ema", "flag_htf_slope_lookback"):
            cv.positive_int(o, name, getattr(self, name))
        for name in ("flag_body_mult", "flag_vol_mult", "flag_level_lookback_hours",
                     "flag_min_rr", "flag_t1_r", "flag_t2_r"):
            cv.positive_number(o, name, getattr(self, name))
        for name in ("flag_close_pos_min", "flag_cluster_body_ratio",
                     "flag_cluster_range_ratio", "flag_retrace_limit",
                     "flag_level_proximity_pct", "flag_stop_atr_mult", "flag_bar_tick"):
            cv.non_negative_number(o, name, getattr(self, name))


# ──────────────────────────────────────────────────────────────────────────────
# Order block   ·   order_block, order_block_inv
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OrderBlockParams:
    """Impulse-origin order blocks with an HTF bias filter and an LTF EMA-cross
    confirmation."""

    atr_period: int = 14
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
    ob_origin_lookback: int = 10             # max bars walked back to find the OB origin candle

    EXITS: ClassVar[dict[str, str]] = {
        "order_block": "structural",
        "order_block_inv": "structural",
    }

    def __post_init__(self) -> None:
        o = "OrderBlockParams"
        cv.positive_int(o, "atr_period", self.atr_period)
        cv.positive_number(o, "ob_body_mult", self.ob_body_mult)
        cv.positive_int(o, "ob_consecutive_min", self.ob_consecutive_min)
        cv.positive_number(o, "ob_consecutive_range_pct", self.ob_consecutive_range_pct)
        cv.positive_number(o, "ob_max_age_hours", self.ob_max_age_hours)
        cv.positive_int(o, "ob_htf_minutes", self.ob_htf_minutes)
        cv.positive_int(o, "ob_htf_ema", self.ob_htf_ema)
        cv.positive_int(o, "ob_ema_fast", self.ob_ema_fast)
        cv.positive_int(o, "ob_ema_slow", self.ob_ema_slow)
        cv.non_negative_number(o, "ob_stop_buffer_pct", self.ob_stop_buffer_pct)
        cv.positive_number(o, "ob_rr", self.ob_rr)
        cv.positive_int(o, "ob_origin_lookback", self.ob_origin_lookback)


# ──────────────────────────────────────────────────────────────────────────────
# VWAP stdev bands   ·   vwap_bands
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VwapParams:
    """Session-anchored VWAP with stdev bands. Defaults mirror the TradingView
    'v2 Mod' preset. Stopless: exits when the close reverts to the VWAP mean."""

    vwap_band_devs: tuple[float, ...] = (1.28, 2.01, 2.51, 3.09, 4.01)
    vwap_session: str = "D"                  # session anchor for VWAP reset
    vwap_entry_band: int = 4                 # 0-indexed; default = furthest band

    EXITS: ClassVar[dict[str, str]] = {
        "vwap_bands": "vwap_mean",   # close reverts to the VWAP mean (no stop)
    }

    def __post_init__(self) -> None:
        o = "VwapParams"
        if not self.vwap_band_devs:
            cv.require(o, "vwap_band_devs", self.vwap_band_devs, False,
                       "a non-empty tuple of band multiples")
        if any(not cv._is_num(d) or d <= 0 for d in self.vwap_band_devs):
            cv.require(o, "vwap_band_devs", self.vwap_band_devs, False,
                       "a tuple of positive band multiples")
        cv.require(o, "vwap_entry_band", self.vwap_entry_band,
                   cv._is_int(self.vwap_entry_band)
                   and 0 <= self.vwap_entry_band < len(self.vwap_band_devs),
                   f"an index in [0, {len(self.vwap_band_devs)})")
        cv.non_empty_str(o, "vwap_session", self.vwap_session)


# ──────────────────────────────────────────────────────────────────────────────
# Swing (ATR-prominence ZigZag)   ·   swing_flip, swing_bounce, swing_breakout
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SwingParams:
    """Shared ZigZag detection knobs (swing_zz_*) read by all three swing modes,
    plus the bounce-/breakout-specific knobs. swing_ml does NOT use these — it
    has its own SwingMlParams. See engine/swing_detector.py."""

    swing_zz_atr_period: int = 14
    swing_zz_min_prominence_atr: float = 1.5
    swing_zz_min_bars_between: int = 3
    swing_zz_vol_lookback: int = 50
    swing_zz_min_score: float = 0.0
    swing_zz_use_stop: bool = True                      # swing_flip: stop toggle; trail = chandelier_3atr preset
    # Swing bounce (swing_bounce): mean-reversion off confirmed swing pivots.
    swing_bounce_test_tolerance_atr: float = 0.5        # how close (in ATR) the wick must come to the swing level
    swing_bounce_require_close_rejection: bool = True   # bar must close back across the level
    swing_bounce_stop_atr_mult: float = 1.0             # entry stop = swing price ∓ mult·ATR (swing-anchored)
    swing_bounce_min_bars_between_trades: int = 1       # cooldown after an exit (>=1 blocks same-bar re-entry; 0 allows it)
    # Swing breakout (swing_breakout): continuation through confirmed swing pivots.
    swing_breakout_buffer_atr: float = 0.0              # close must clear the swing level by this many ATR
    swing_breakout_stop_atr_mult: float = 1.0           # entry stop = swing level ∓ mult·ATR (swing-anchored)
    swing_breakout_min_bars_between_trades: int = 1     # cooldown after an exit (>=1 blocks same-bar re-entry; 0 allows it)

    EXITS: ClassVar[dict[str, str]] = {
        "swing_flip": "chandelier_3atr",        # 3·ATR trail
        "swing_bounce": "structural_rr2",       # swing-anchored stop (entry stop_price) + 2R
        "swing_breakout": "structural_rr2",
    }

    def __post_init__(self) -> None:
        o = "SwingParams"
        cv.positive_int(o, "swing_zz_atr_period", self.swing_zz_atr_period)
        cv.positive_number(o, "swing_zz_min_prominence_atr", self.swing_zz_min_prominence_atr)
        cv.positive_int(o, "swing_zz_min_bars_between", self.swing_zz_min_bars_between)
        cv.positive_int(o, "swing_zz_vol_lookback", self.swing_zz_vol_lookback)
        cv.non_negative_number(o, "swing_zz_min_score", self.swing_zz_min_score)
        cv.non_negative_number(o, "swing_bounce_test_tolerance_atr", self.swing_bounce_test_tolerance_atr)
        cv.non_negative_number(o, "swing_bounce_stop_atr_mult", self.swing_bounce_stop_atr_mult)
        cv.non_negative_int(o, "swing_bounce_min_bars_between_trades", self.swing_bounce_min_bars_between_trades)
        cv.non_negative_number(o, "swing_breakout_buffer_atr", self.swing_breakout_buffer_atr)
        cv.non_negative_number(o, "swing_breakout_stop_atr_mult", self.swing_breakout_stop_atr_mult)
        cv.non_negative_int(o, "swing_breakout_min_bars_between_trades", self.swing_breakout_min_bars_between_trades)


# ──────────────────────────────────────────────────────────────────────────────
# Swing ML   ·   swing_ml
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SwingMlParams:
    """ML swing-pivot classifier. Does NOT read the swing_zz_* knobs (it hardcodes
    its own ATR). ml_model_path is resolved relative to the repo root if not
    absolute."""

    ml_model_path: str = "ml_models/swing_zz_ml.joblib"
    ml_p_threshold: float = 0.55
    ml_use_stop: bool = True   # toggle only; trail distance = its EXITS preset (chandelier_3atr)
    ml_atr_period: int = 14    # Wilder ATR period for the trailing-stop input (ml_atr)

    EXITS: ClassVar[dict[str, str]] = {
        "swing_ml": "chandelier_3atr",
    }

    def __post_init__(self) -> None:
        o = "SwingMlParams"
        cv.non_empty_str(o, "ml_model_path", self.ml_model_path)
        cv.in_range(o, "ml_p_threshold", self.ml_p_threshold, 0, 1)
        cv.positive_int(o, "ml_atr_period", self.ml_atr_period)


# ════════════════════════════════════════════════════════════════════════════
# Section 2 — Exit / take-profit policy catalog
# ════════════════════════════════════════════════════════════════════════════
# The reusable preset menu (assigned per strategy via each class's EXITS). Each
# value is a factory so a fresh policy is created per use, built from the
# mechanisms in engine/exits.py:
#   — either a single mechanism (a trailing ChandelierStop, or a TP-only CloseCrossTarget)
#   — or a CompositeExit(stop, target) that runs a stop then a target (stop-first).
# Add a new SL/TP combo here, then reference it by key in a class's EXITS map.
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
    "structural_rr3":  lambda: CompositeExit(StructuralStop(), RrTarget(3.0)),  # g_bounce / g_breakout / g_breakout_false
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


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — Params registry + exit resolution
# ════════════════════════════════════════════════════════════════════════════
# PARAMS maps every strategy name → its config class. params_for(name) is the
# single source of truth for a strategy's default signal knobs; the default exit
# for each name lives on that class's EXITS map (Section 1). PER_STRATEGY_EXIT
# below is *derived* from those maps — the central "glance view" that
# exit_policy_for() and the catalog validator read — so the per-class EXITS stay
# the single source of truth and this can't drift.

PARAMS: dict[str, type] = {
    "ema": EmaParams,
    "ema_inv": EmaParams,
    "ema_adaptive": EmaParams,
    "supertrend": SupertrendParams,
    "supertrend_inv": SupertrendParams,
    "supertrend_adaptive": SupertrendParams,
    "ema_touch": EmaTouchParams,
    "fractal_breakout": FractalParams,
    "fractal_breakout_inv": FractalParams,
    "level_breakout": LevelBreakoutParams,
    "level_breakout_inv": LevelBreakoutInvParams,
    "g_bounce": GBounceParams,
    "g_breakout": GBreakoutParams,
    "g_breakout_false": GBreakoutFalseParams,
    "g_range": GRangeParams,
    "exhaustion_reversal": ExhaustionParams,
    "impulse_flag": ImpulseFlagParams,
    "order_block": OrderBlockParams,
    "order_block_inv": OrderBlockParams,
    "vwap_bands": VwapParams,
    "swing_flip": SwingParams,
    "swing_bounce": SwingParams,
    "swing_breakout": SwingParams,
    "swing_ml": SwingMlParams,
}


def _assigned_exit(name: str, cls: type) -> str:
    """The exit preset a strategy is assigned, read from its config class's
    EXITS map. Fails loudly at import if a class forgot an entry."""
    try:
        return cls.EXITS[name]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"{cls.__name__}.EXITS is missing an entry for strategy {name!r}"
        ) from exc


# Derived name → preset assignment, gathered from each config class's EXITS.
PER_STRATEGY_EXIT: dict[str, str] = {
    name: _assigned_exit(name, cls) for name, cls in PARAMS.items()
}


# Resolves a strategy's assigned preset into a fresh ExitPolicy. Reads the
# assignment **live** from the strategy's config class EXITS map (via PARAMS) —
# so the class stays the single source of truth and an edit there takes effect
# without rebuilding the derived PER_STRATEGY_EXIT view. 3 steps:
#   PARAMS[name].EXITS[name] → the preset key assigned on this strategy's class.
#   EXIT_PRESETS[…]          → look up that key's factory (the lambda).
#   ()                       → call it to build a brand-new ExitPolicy.
def exit_policy_for(strategy_name: str) -> ExitPolicy:
    """Build the exit policy assigned to a strategy. Called by BaseStrategy to
    seed self.exit_policy.

    Every *real* strategy (a StrategyName member) is guaranteed to have an EXITS
    entry on its config class by _validate_exit_catalog() below — so the only way
    to reach the DEFAULT_EXIT fallback here is an ad-hoc / experimental / test
    strategy not in the registry. That fallback is intentional (a prototype
    shouldn't need a catalog entry to run) but is logged at WARNING so it is
    never silent.
    """
    cls = PARAMS.get(strategy_name)
    key = getattr(cls, "EXITS", {}).get(strategy_name) if cls is not None else None
    if key is None:
        logger.warning(
            "Strategy %r has no EXITS entry on its config class; using "
            "DEFAULT_EXIT (%s). Add it to that class's EXITS map.",
            strategy_name, DEFAULT_EXIT,
        )
        key = DEFAULT_EXIT
    return EXIT_PRESETS[key]()


def _validate_exit_catalog() -> None:
    """Fail at import if the exit catalog is internally inconsistent — so a typo
    in a class's EXITS map or a forgotten new strategy is caught here, not at the
    runtime moment a strategy is first instantiated.

    Checks: (1) DEFAULT_EXIT resolves; (2) every assigned preset is a real
    EXIT_PRESETS key; (3) the assignments cover exactly the StrategyName enum.
    """
    if DEFAULT_EXIT not in EXIT_PRESETS:
        raise ValueError(
            f"DEFAULT_EXIT {DEFAULT_EXIT!r} is not a key in EXIT_PRESETS "
            f"{sorted(EXIT_PRESETS)}"
        )
    bad = {name: key for name, key in PER_STRATEGY_EXIT.items()
           if key not in EXIT_PRESETS}
    if bad:
        raise ValueError(
            f"Exit assignments map to unknown EXIT_PRESETS keys: {bad}. "
            f"Known presets: {sorted(EXIT_PRESETS)}"
        )
    names = {s.value for s in StrategyName}
    mapped = set(PER_STRATEGY_EXIT)
    missing, unknown = names - mapped, mapped - names
    if missing or unknown:
        raise ValueError(
            "Exit assignments must cover exactly the StrategyName members. "
            f"Missing (add an EXITS entry for these): {sorted(missing)}. "
            f"Unknown (not a strategy): {sorted(unknown)}."
        )


_validate_exit_catalog()


# ════════════════════════════════════════════════════════════════════════════
# Section 4 — Params accessors
# ════════════════════════════════════════════════════════════════════════════
# Thin lookups over the PARAMS registry (Section 3). params_for(name) is the
# single source of truth for a strategy's *default* signal knobs — the CLI and
# notebooks both start from it, then override on top. params_class_for(name)
# lets BaseStrategy type-check the config it is handed.

def params_class_for(strategy_name: str) -> type | None:
    """The config class a strategy expects, or None for an unregistered (ad-hoc /
    test) strategy. BaseStrategy uses it to reject a wrong-type config early."""
    return PARAMS.get(strategy_name)


def params_for(strategy_name: str):
    """The default param object for a strategy — single source of truth for its
    signal-knob defaults. Mirrors exit_policy_for(); an unregistered name has no
    params class, so callers building ad-hoc strategies must supply their own."""
    cls = PARAMS.get(strategy_name)
    if cls is None:
        raise ValueError(
            f"No params class registered for strategy {strategy_name!r}. "
            f"Known: {sorted(PARAMS)}"
        )
    return cls()


def _validate_params_registry() -> None:
    """Fail at import if PARAMS doesn't cover exactly the StrategyName enum — so a
    new strategy can't be added without giving it a params class (same guard the
    exit catalog has)."""
    names = {s.value for s in StrategyName}
    mapped = set(PARAMS)
    missing, unknown = names - mapped, mapped - names
    if missing or unknown:
        raise ValueError(
            "PARAMS must list exactly the StrategyName members. "
            f"Missing (add a params class for these): {sorted(missing)}. "
            f"Unknown (not a strategy): {sorted(unknown)}."
        )


_validate_params_registry()
