# Plan: pluggable level detectors (pivot_level / cluster_level / touch_level)

Status: proposed (not yet implemented) · Date: 2026-06-14

Make horizontal support/resistance detection pluggable: three interchangeable
detectors behind one contract, selectable per strategy through a single
strategy_configurator field (the same way you pick atr_period), sweepable for
free, and explorable side by side in an EDA notebook. The work builds on the
seam that already exists — both consumers of levels (the level strategies and the
plotting helper) already speak one shape, so the change is mostly "route one call
through a registry" plus porting two outside detectors into the causal framework.

## Goals

- Three detectors living as peers behind one normalized contract.
- Selection via a LevelParams field, validated and swept like any other knob.
- Any level-consuming strategy — existing (level_breakout, level_breakout_inv) or
  future (level_bounce, level_retest) — inherits the choice with no strategy-code
  change, because dispatch lives in the shared level base.
- An EDA notebook (levels.ipynb) with one section per detector, each mirroring the
  current explorer content (Detect / Inspect / Visualize). No backtest / P&L — it
  stays a research notebook.
- Runs causally and cleanly like every other strategy: passes the causality test,
  pinned by golden snapshots, fields validated, registry pinned by an import guard.

## The three detectors and their names

| Key | Source | Mechanism | Causal today |
|-----|--------|-----------|--------------|
| pivot_level (default) | ours, engine/level_detector.py | pivot-seeded; a level lives until invalidated by a touch within tolerance or by being bracketed N candles; three families (resistance / support / pullback) | yes |
| cluster_level | key_levels.py (KeyLevelDetector) | merges nearby pivots into one mean-price level; the level strengthens on touch and dies only on a decisive close-through break; carries a strength score | yes (streaming) |
| touch_level | the Bybit pair (levels_entries_exits ≡ gerchik, byte-identical) | significance by number of historical touches; median-clustered; optional recency drop | no — needs a causal port |

The names are deliberately parallel (X_level) and describe each detector's
defining trait, so they read clearly as a config value and as a notebook section
title. The folder name gerchik is intentionally not reused for ours, since that
folder holds the Bybit copy and would be the most confusing possible label.

## Architecture: the seam

Today engine/strategies/level_base.py hardcodes detect_all_levels in prepare().
The entire feature is: send that one call through a registry keyed by a config
field, and have all three detectors emit the same families dict that level_base
and plot_levels already consume. Nothing downstream changes.

```
                       cfg.level_detector ──► LEVEL_SOURCES[key]
                                                    │
  pivot_level (ours)        ─┐                      ▼
  cluster_level (port)      ─┼──►  dict[str, list[Level]]  ──►  level_base.prepare()  (unchanged logic)
  touch_level (port)        ─┘          (the seam)         └──►  plot_levels()         (unchanged)
```

Because the dispatch sits in LevelStrategyBase.prepare(), every current and
future level strategy gets all three detectors for free.

## The detector layer: package and contract

Promote the single file engine/level_detector.py into a small package so the three
detectors live as peers:

```
engine/levels/
  __init__.py        # re-exports + the LevelSource type alias
  base.py            # the normalized record (extended Level) + LevelSource
  pivot_level.py     # ours, moved verbatim (logic unchanged)
  cluster_level.py   # causal port of KeyLevelDetector + adapter
  touch_level.py     # causal port of the Bybit logic + adapter
```

Contract. Keep the existing families-dict contract, dict[str, list[Level]] with
keys resistance / support / pullback, because level_base and plot_levels already
consume exactly that. Extend the Level dataclass additively:

- confirmed_idx — first bar at which the level is tradeable.
- strength — cluster_level and touch_level fill this; pivot_level defaults it.

pivot_level sets confirmed_idx = start_idx + pivot_window, so its trades stay
byte-for-byte identical and the golden tests keep passing; level_base then reads
lvl.confirmed_idx instead of recomputing start_idx + pivot_window itself (removing
an assumption that is only true for our detector). plot_levels keeps using
start_idx as the visual draw anchor. cluster_level and touch_level set start_idx
to the visual pivot bar and confirmed_idx to the confirmation bar.

A LevelSource is any callable with the signature:

```
LevelSource = Callable[[pd.DataFrame, LevelParams], dict[str, list[Level]]]
```

cluster_level and touch_level fill the resistance and support families; only
pivot_level produces pullback. plot_levels already iterates whatever families are
present, so missing families are harmless.

## Config surface (engine/strategy_configurator.py)

Mirror the exit-policy machinery, which is the codebase's proven selectable-
mechanism pattern, and put the registry here as requested.

A module-level tuple near the top:

```
LEVEL_SOURCE_NAMES = ("pivot_level", "cluster_level", "touch_level")
```

LevelParams gains a selector plus two namespaced knob blocks (prefixes
cluster_ and touch_), validated in __post_init__ via config_validation like every
existing knob:

```
# selector — choose the detector like you choose atr_period
level_detector: str = "pivot_level"        # one_of(LEVEL_SOURCE_NAMES)

# shared knobs (already present) — every adapter maps these
level_pivot_window: int = 3
level_atr_period: int = 14
level_delta: float = 0.5
level_delta_mode: str = "atr"              # absolute | percent | atr
level_invalidation_candles: int = 3        # pivot_level only
level_use_pullback: bool = False           # pivot_level only
# level_breakout_buffer_atr / level_stop_atr_mult stay strategy-side (level_breakout)

# cluster_level knobs (natively ATR-based, like KeyLevelDetector)
cluster_merge_atr_mult: float = 0.5        # merge pivots within this x ATR
cluster_break_atr_mult: float = 0.1        # close must clear the level by this x ATR to break
cluster_max_levels: int = 300              # prune cap

# touch_level knobs (tolerance units follow level_delta_mode)
touch_cluster_mult: float = 0.5            # median-cluster tolerance magnitude
touch_band_mult: float = 0.75              # how close counts as a touch
touch_min_touches: int = 3                 # level activates after this many past touches
touch_recency_bars: int = 0                # 0 = keep all; else drop levels untouched in the last N bars
```

The registry, getter, dispatcher, and import guard mirror EXIT_PRESETS /
exit_policy_for / _validate_exit_catalog:

```
LEVEL_SOURCES = {                          # key -> adapter (analog of EXIT_PRESETS)
    "pivot_level":   pivot_level_source,
    "cluster_level": cluster_level_source,
    "touch_level":   touch_level_source,
}

def level_source_for(name): ...            # analog of exit_policy_for
def detect_levels(df, cfg): ...            # = level_source_for(cfg.level_detector)(df, cfg)

def _validate_level_sources(): ...         # LEVEL_SOURCES keys == LEVEL_SOURCE_NAMES; called at import
```

The registry lives in strategy_configurator and imports the adapters from
engine.levels, which keeps the import direction one-way (no cycle). detect_levels
is the single entry both level_base and the notebook call.

Accepted trade-off. With all knobs on one class, a knob for an unselected detector
is validated but silently inert (for example cluster_break_atr_mult does nothing
when level_detector is touch_level). This is the same harmless pattern as the RSI
knobs being inert when the RSI filter is off, and it is the cost of having every
knob central and sweepable, which is what was chosen.

## Causality and the touch_level port

The engine makes look-ahead structurally impossible: Backtester.run feeds each
on_bar a view truncated to bars 0..i, and test_config_propagation runs every
strategy with enforce_causality=False and asserts the trades match the enforced
run. So a detector that reads the whole frame would fail that test — which is
exactly the safety net that keeps this clean.

- pivot_level and cluster_level are already causal (ours by construction;
  KeyLevelDetector is a streaming detector with a confirmation delay). They drop
  in directly.
- touch_level must be ported to one pass that uses only past data:
  - swing pivots already confirm at i + pivot_window (causal in the original),
  - cluster incrementally as each swing confirms (not over the whole frame),
  - accumulate touch counts only from bars already seen,
  - a level becomes active once it reaches touch_min_touches,
  - the recency drop uses only past bars.
  After the port it passes the causality gate and gets golden snapshots like the
  others. This is the bulk of the implementation effort.

## Plotting

plot_levels in engine/visualization.py needs no change: it already takes the
families dict and draws each level from its start_idx anchor to invalidated_at (or
the chart edge), and it consumes the extended Level transparently. The notebook
and level_base both obtain levels through detect_levels(df, cfg), so there is one
code path for all three detectors and consistent styling.

## Notebook restructure: levels.ipynb (EDA, not backtest)

levels.ipynb stays an explorer — detect, inspect, visualize, no P&L. It adopts the
ema_rsi template partially: a shared Configuration chapter, then three sections,
one per detector, each mirroring the content the current notebook already has.
This is not three runs of one strategy; it is an EDA notebook comparing the three
detectors.

Target structure:

```
# Dynamic levels — three detectors            (short generic title)
Contents                                        (TOC: Configuration / Pivot level / Cluster level / Touch level)

## Configuration
  ### Setup            (path bootstrap)
  ### Automatic        (imports; DATA_CONFIG = ACTIVE; BASE_LEVEL_PARAMS = params_for("level_breakout"))
  ### Manual           (DATA_OVERRIDES; optional shared-knob overrides on BASE_LEVEL_PARAMS)
  ### Final configuration   (df = load_data(DATA_CONFIG); print the config report)

## Pivot level
  (the moved intro markdown — see note below — goes here, under the section title)
  ### Parameters        (cfg = dataclasses.replace(BASE_LEVEL_PARAMS, level_detector="pivot_level", ...pivot knobs...))
  ### Detect            (levels = detect_levels(df, cfg); print per-family counts)
  ### Inspect           (per-level table: seed ts, invalidated ts, lifespan, crosses, active)
  ### Visualize         (plot_levels(df, levels, ...))
  ### Levels with EMAs  (the same overlay plus user-selected EMAs)
  ### Live signals      (per-section live preview loop for this detector)

## Cluster level
  (markdown: merges nearby pivots into one mean-price level; dies on a decisive close break)
  ### Parameters / Detect / Inspect / Visualize / Levels with EMAs / Live signals   (same shape; level_detector="cluster_level" + cluster_ knobs)

## Touch level
  (markdown: significance by number of historical touches; median-clustered)
  ### Parameters / Detect / Inspect / Visualize / Levels with EMAs / Live signals   (same shape; level_detector="touch_level" + touch_ knobs)
```

Each section differs from the others by exactly one line, the per-detector config:

```
cfg = dataclasses.replace(BASE_LEVEL_PARAMS, level_detector="cluster_level",
                          cluster_merge_atr_mult=0.5, cluster_break_atr_mult=0.1)
levels = detect_levels(df, cfg)
```

Markdown move. The current notebook has two markdown cells between the H1 title and
the Configuration chapter (the three-families description and the look-ahead note).
Those describe the pivot_level detector specifically, so they move into the Pivot
level section, directly under its section title. The title keeps only a short,
detector-neutral intro.

Levels with EMAs and Live signals. Both are repeated per section, so each detector
has its own EMA overlay and its own live preview loop, mirroring the current
notebook's content within each section. The live loop reuses that section's cfg, so
it streams the same detector it was configured with.

The Inspect and Visualize cells are detector-agnostic already — they read price,
start_idx, invalidated_at, and (newly) strength off the Level records — so they are
copy-paste identical across the three sections.

## Security and cleanliness (tests and guards)

- Causality: cluster_level and touch_level are exercised by the existing
  enforce_causality True-vs-False assertion in test_config_propagation.py.
- Golden snapshots: pin level_breakout and level_breakout_inv under each of the
  three detector keys in test_golden.py.
- Validation: the new LevelParams fields go through config_validation in
  __post_init__, so a bad value fails at construction in the CLI, a notebook
  replace, or a sweep grid.
- Import guard: _validate_level_sources keeps the selector choices and the registry
  in lockstep, the same way the exit catalog and the params registry are pinned.

## Sweepability

Because level_detector and all the namespaced knobs are plain fields on LevelParams,
they sweep with the existing machinery, no new sweep code:

```
strategy_grid = {
    "level_detector": ["pivot_level", "cluster_level", "touch_level"],
    "cluster_break_atr_mult": [0.05, 0.1, 0.2],
}
```

(A knob only affects rows whose level_detector selects that detector; see the
inert-knob trade-off above.)

## Files

New
- engine/levels/__init__.py, base.py, pivot_level.py, cluster_level.py, touch_level.py

Edited
- engine/strategy_configurator.py — LEVEL_SOURCE_NAMES; LevelParams fields + validators; LEVEL_SOURCES; level_source_for; detect_levels; _validate_level_sources
- engine/strategies/level_base.py — dispatch via detect_levels; consume confirmed_idx
- engine/visualization.py — consumes the extended Level transparently (no behaviour change)
- engine/tests/test_golden.py — snapshots per detector key
- engine/tests/test_config_propagation.py and a config-validation test — cover the new detectors and fields
- strategy_notebooks/levels.ipynb — rebuilt as above
- CLAUDE.md — update the level-family paragraph

Import migration
- engine/level_detector.py moves to engine/levels/pivot_level.py. Migrate the import
  sites outright (level_base, the notebook's detect_all_levels, and any tests); no
  re-export shim is left behind, for a clean tree.

## Phased rollout (each independently shippable)

1. Package move + extended Level contract + pivot_level adapter. Behaviour-
   preserving; golden tests pin it.
2. Registry + selector field + dispatcher + import guard; level_base dispatch.
3. cluster_level causal-port adapter + golden snapshot.
4. touch_level causal port + golden snapshot. The heavy step.
5. Rebuild levels.ipynb on the template with the three sections.
6. Docs: CLAUDE.md level-family paragraph.

## Non-goals and follow-ups

- No change to level_breakout / level_breakout_inv signal logic.
- No new strategies here. cluster_level's survive-the-touch lifecycle is the
  prerequisite for level_bounce / level_retest, which our touch-kills-the-level
  model cannot express; those strategies are a natural follow-up once the detectors
  land.
- Detector-specific knobs are namespaced on LevelParams now (decided), rather than
  a common-subset-only start or named-variant presets.
