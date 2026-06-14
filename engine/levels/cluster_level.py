"""cluster_level — merge-and-break horizontal levels (causal port).

Ported from the standalone ``KeyLevelDetector``. Confirmed pivots are merged into
a single level per price zone (within ``cluster_merge_atr_mult`` × ATR of an
existing active level), and a level dies only on a *decisive close-through* —
``cluster_break_atr_mult`` × ATR beyond it — so a wick poking the level does not
kill it. Each merge reinforces the level (``strength`` / ``cross_count``), which
distinguishes it from :mod:`pivot_level` (where a single touch invalidates).

Causal in the engine's precompute-then-replay model:

* A pivot at ``center`` is confirmed only at ``center + pivot_window`` (its right
  neighbourhood is in) → ``confirmed_idx`` uses no future data.
* A break at bar ``i`` uses only ``close[i]`` and ATR up to ``i`` →
  ``invalidated_at`` uses no future data.
* The level **price is fixed at its seeding pivot** (it does not migrate to the
  running mean as the streaming source did): a running mean would bake later
  pivots into a price that earlier bars can see, a look-ahead the replay model
  cannot have. Merges therefore only reinforce strength and extend life.

``cluster_max_levels`` caps the simultaneously-active levels: when exceeded, the
oldest active level is retired (``invalidated_at`` = current bar) — a causal,
deterministic age-out.

This is one of three peer detectors behind :data:`engine.levels.LevelSource`
(see :mod:`engine.levels.base`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..swing_detector import wilder_atr
from .base import Level

# Relative tolerance used for merging before ATR is available (the first
# ``atr_period`` bars, where Wilder ATR is NaN).
_FALLBACK_TOL_PCT = 0.002


def _is_pivot_high(highs: np.ndarray, c: int, w: int) -> bool:
    """KeyLevelDetector's asymmetric pivot high: strictly above the left
    neighbourhood, at-or-above the right (so back-to-back equal highs survive)."""
    return bool((highs[c - w:c] < highs[c]).all() and (highs[c + 1:c + 1 + w] <= highs[c]).all())


def _is_pivot_low(lows: np.ndarray, c: int, w: int) -> bool:
    return bool((lows[c - w:c] > lows[c]).all() and (lows[c + 1:c + 1 + w] >= lows[c]).all())


def _merge_or_create(
    active: list[Level], family: list[Level],
    price: float, start_idx: int, confirmed_idx: int, tol: float,
) -> None:
    """Reinforce the nearest active level within ``tol``, else create a new one
    (appended to both the active set and the output family). Price never migrates."""
    best: Level | None = None
    best_dist = tol
    for lvl in active:
        d = abs(lvl.price - price)
        if d <= best_dist:
            best, best_dist = lvl, d
    if best is not None:
        best.strength += 1.0
        best.cross_count += 1
        return
    lvl = Level(price=float(price), start_idx=start_idx, confirmed_idx=confirmed_idx)
    active.append(lvl)
    family.append(lvl)


def _enforce_cap(active_res: list[Level], active_sup: list[Level], max_levels: int, i: int) -> None:
    """Retire the oldest active level(s) when the active count exceeds the cap."""
    while len(active_res) + len(active_sup) > max_levels:
        oldest_r = min(active_res, key=lambda l: l.confirmed_idx, default=None)
        oldest_s = min(active_sup, key=lambda l: l.confirmed_idx, default=None)
        if oldest_s is None or (
            oldest_r is not None and oldest_r.confirmed_idx <= oldest_s.confirmed_idx
        ):
            oldest_r.invalidated_at = i
            active_res.remove(oldest_r)
        else:
            oldest_s.invalidated_at = i
            active_sup.remove(oldest_s)


def cluster_level_source(df: pd.DataFrame, cfg) -> dict[str, list[Level]]:
    """The :data:`engine.levels.LevelSource` adapter for ``cluster_level``."""
    out: dict[str, list[Level]] = {"resistance": [], "support": [], "pullback": []}
    n = len(df)
    if n == 0:
        return out

    w = cfg.level_pivot_window
    merge_mult = cfg.cluster_merge_atr_mult
    break_mult = cfg.cluster_break_atr_mult
    max_levels = cfg.cluster_max_levels

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atr = wilder_atr(df, cfg.level_atr_period)

    active_res: list[Level] = []
    active_sup: list[Level] = []

    for i in range(n):
        atr_i = atr[i]
        have_atr = not np.isnan(atr_i)

        # 1. Breaks on the incoming close (levels that existed before this bar).
        if have_atr:
            buf = atr_i * break_mult
            for lvl in active_res:
                if closes[i] > lvl.price + buf:
                    lvl.invalidated_at = i
            for lvl in active_sup:
                if closes[i] < lvl.price - buf:
                    lvl.invalidated_at = i
            active_res = [lvl for lvl in active_res if lvl.invalidated_at is None]
            active_sup = [lvl for lvl in active_sup if lvl.invalidated_at is None]

        # 2. Confirm a pivot at center = i - w (right neighbourhood now in).
        c = i - w
        if c >= w:
            created = False
            if _is_pivot_high(highs, c, w):
                tol = atr_i * merge_mult if (have_atr and atr_i > 0) else highs[c] * _FALLBACK_TOL_PCT
                before = len(active_res)
                _merge_or_create(active_res, out["resistance"], highs[c], c, i, tol)
                created = created or len(active_res) > before
            if _is_pivot_low(lows, c, w):
                tol = atr_i * merge_mult if (have_atr and atr_i > 0) else lows[c] * _FALLBACK_TOL_PCT
                before = len(active_sup)
                _merge_or_create(active_sup, out["support"], lows[c], c, i, tol)
                created = created or len(active_sup) > before
            if created:
                _enforce_cap(active_res, active_sup, max_levels, i)

    return out
