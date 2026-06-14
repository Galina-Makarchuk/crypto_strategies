"""touch_level — significance-by-touch-count horizontal levels (causal port).

Ported from the standalone swing/cluster/touch detector (the "Bybit" detector).
Its defining trait: a price zone becomes a level only once it has been *touched*
enough times — significance by frequency, not by a single pivot. Conventional
seeding (resistance from swing highs, support from swing lows), median-style
clustering of nearby swings into one zone, and an optional recency drop.

Made causal for the engine's precompute-then-replay model (the original computed
touch counts and clustering over the whole frame, which is look-ahead):

* Swings confirm at ``center + pivot_window`` (right neighbourhood in) — no future
  data seeds a candidate.
* Touches are accumulated **only from bars already seen**. A candidate becomes an
  observable level the bar its touch count first reaches ``touch_min_touches``;
  that bar is its ``confirmed_idx``. A candidate that never reaches the threshold
  is never emitted (the significance filter, applied causally).
* ``touch_recency_bars`` (when > 0) retires a level that goes that many bars
  without a touch (``invalidated_at`` = last-touch + recency). With 0, levels live
  to the end (the source's keep-all behaviour).
* The level **price is fixed at its seeding swing** — it does not migrate to the
  running cluster median, which would let later swings move a price earlier bars
  can see.

Tolerances follow ``level_delta_mode``; ``touch_cluster_mult`` sets the
swing-clustering band and ``touch_band_mult`` sets what counts as a touch.

This is one of three peer detectors behind :data:`engine.levels.LevelSource`
(see :mod:`engine.levels.base`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..swing_detector import wilder_atr
from .base import Level, tolerance


class _Cand:
    """A clustered swing zone accumulating touches toward significance."""

    __slots__ = ("price", "start_idx", "touches", "last_touch", "level", "dead")

    def __init__(self, price: float, start_idx: int) -> None:
        self.price = price
        self.start_idx = start_idx
        self.touches = 1
        self.last_touch = start_idx
        self.level: Level | None = None   # set once it crosses the touch threshold
        self.dead = False                 # retired by recency


def _is_swing_high(highs: np.ndarray, c: int, w: int) -> bool:
    """Bybit swing high: at-or-above the left neighbourhood (``>=``), strictly
    above the right (``>``)."""
    return bool((highs[c] >= highs[c - w:c]).all() and (highs[c] > highs[c + 1:c + 1 + w]).all())


def _is_swing_low(lows: np.ndarray, c: int, w: int) -> bool:
    return bool((lows[c] <= lows[c - w:c]).all() and (lows[c] < lows[c + 1:c + 1 + w]).all())


def touch_level_source(df: pd.DataFrame, cfg) -> dict[str, list[Level]]:
    """The :data:`engine.levels.LevelSource` adapter for ``touch_level``."""
    out: dict[str, list[Level]] = {"resistance": [], "support": [], "pullback": []}
    n = len(df)
    if n == 0:
        return out

    w = cfg.level_pivot_window
    mode = cfg.level_delta_mode
    cluster_mult = cfg.touch_cluster_mult
    band_mult = cfg.touch_band_mult
    min_touches = cfg.touch_min_touches
    recency = cfg.touch_recency_bars

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    atr = wilder_atr(df, cfg.level_atr_period)

    cands_res: list[_Cand] = []
    cands_sup: list[_Cand] = []

    def _confirm(cand: _Cand, family: list[Level], i: int) -> None:
        """Promote a candidate to a Level the bar it first reaches the threshold,
        or keep an existing Level's touch metadata fresh."""
        if cand.level is None and cand.touches >= min_touches:
            cand.level = Level(price=cand.price, start_idx=cand.start_idx,
                               confirmed_idx=i, strength=float(cand.touches))
            family.append(cand.level)
        elif cand.level is not None:
            cand.level.strength = float(cand.touches)
            cand.level.cross_count += 1

    for i in range(n):
        atr_i = atr[i]

        # 1. Recency retire (before this bar's touches): a level idle longer than
        #    the window dies at last-touch + recency.
        if recency > 0:
            for cands in (cands_res, cands_sup):
                for cand in cands:
                    if cand.dead or cand.level is None:
                        continue
                    if i - cand.last_touch > recency:
                        cand.level.invalidated_at = cand.last_touch + recency
                        cand.dead = True

        # 2. Accumulate touches from the incoming bar.
        for cands, family in ((cands_res, out["resistance"]), (cands_sup, out["support"])):
            for cand in cands:
                if cand.dead:
                    continue
                band = tolerance(cand.price, band_mult, mode, atr_i)
                if lows[i] - band <= cand.price <= highs[i] + band:
                    cand.touches += 1
                    cand.last_touch = i
                    _confirm(cand, family, i)

        # 3. Confirm a swing at center = i - w → cluster into a candidate.
        c = i - w
        if c >= w:
            if _is_swing_high(highs, c, w):
                _cluster_or_create(cands_res, out["resistance"], highs[c], c, i,
                                   cluster_mult, band_mult, mode, atr_i, min_touches)
            if _is_swing_low(lows, c, w):
                _cluster_or_create(cands_sup, out["support"], lows[c], c, i,
                                   cluster_mult, band_mult, mode, atr_i, min_touches)

    return out


def _cluster_or_create(
    cands: list[_Cand], family: list[Level],
    price: float, start_idx: int, i: int,
    cluster_mult: float, band_mult: float, mode: str, atr_i: float, min_touches: int,
) -> None:
    """Merge a confirmed swing into the nearest live candidate within the cluster
    band (counts as a touch), else open a new candidate. Price never migrates."""
    best: _Cand | None = None
    best_dist = tolerance(price, cluster_mult, mode, atr_i)
    if not np.isnan(best_dist):
        for cand in cands:
            if cand.dead:
                continue
            d = abs(cand.price - price)
            if d <= best_dist:
                best, best_dist = cand, d
    if best is not None:
        best.touches += 1
        best.last_touch = i
        if best.level is None and best.touches >= min_touches:
            best.level = Level(price=best.price, start_idx=best.start_idx,
                               confirmed_idx=i, strength=float(best.touches))
            family.append(best.level)
        elif best.level is not None:
            best.level.strength = float(best.touches)
            best.level.cross_count += 1
        return
    cands.append(_Cand(price=float(price), start_idx=start_idx))
