"""Golden trade fixtures — the safety net for the exit-policy conversion.

Each strategy is run on one fixed, deterministic synthetic dataset and its trade
list is snapshotted to ``golden_trades.json``. As strategies are migrated from
inline exits to injected exit policies, these tests must stay green: a faithful
conversion reproduces byte-identical trades (entry/exit ts, prices, reason,
direction). A drift fails here and must be reconciled deliberately.

Regenerate the fixture (only when an intended behaviour change is accepted)::

    python -m engine.tests.test_golden   # or: python engine/tests/test_golden.py

The one ML strategy (swing_ml) is excluded — it needs a trained model / order-flow
data, which would make the snapshot non-deterministic across environments. It stays
covered by engine/tests/test_ml.py.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from engine import strategies as S
from engine.backtester import Backtester
from engine.strategy_configurator import params_for
from engine.trade_configurator import TradingConfig

GOLDEN = pathlib.Path(__file__).parent / "golden_trades.json"

# Strategy name → class. The ML strategy (swing_ml) is excluded (model/order-flow dependent).
STRATEGIES = {
    "fractal_breakout": S.FractalBreakoutStrategy,
    "fractal_breakout_inv": S.InverseFractalBreakoutStrategy,
    "level_breakout": S.LevelBreakoutStrategy,
    "level_breakout_inv": S.InverseLevelBreakoutStrategy,
    "g_bounce": S.GBounceStrategy,
    "g_breakout": S.GBreakoutStrategy,
    "g_breakout_false": S.GBreakoutFalseStrategy,
    "g_range": S.GRangeStrategy,
    "ema": S.EMACrossoverStrategy,
    "ema_inv": S.InverseEMACrossoverStrategy,
    "ema_adaptive": S.AdaptiveEMACrossoverStrategy,
    "supertrend": S.SuperTrendStrategy,
    "supertrend_inv": S.InverseSuperTrendStrategy,
    "supertrend_adaptive": S.AdaptiveSuperTrendStrategy,
    "exhaustion_reversal": S.ExhaustionReversalStrategy,
    "impulse_flag": S.ImpulseFlagStrategy,
    "order_block": S.OrderBlockStrategy,
    "order_block_inv": S.InverseOrderBlockStrategy,
    "vwap_bands": S.VWAPBandsStrategy,
    "swing_flip": S.SwingFlipStrategy,
    "swing_breakout": S.SwingBreakoutStrategy,
    "ema_touch": S.EmaTouchStrategy,
    "swing_bounce": S.SwingBounceStrategy,
}


def _golden_df(n: int = 1500) -> pd.DataFrame:
    """One deterministic OHLCV frame: drift + cycle + noise, shaped to trigger a
    variety of strategies (trend flips, pullbacks, impulses)."""
    rng = np.random.RandomState(20240601)
    t = np.arange(n)
    drift = np.cumsum(rng.randn(n) * 0.8 + 0.02)
    cycle = 8.0 * np.sin(t / 50.0)
    close = 100.0 + drift + cycle
    high = close + np.abs(rng.randn(n)) * 1.2 + 0.3
    low = close - np.abs(rng.randn(n)) * 1.2 - 0.3
    openp = close + rng.randn(n) * 0.5
    vol = 10.0 + np.abs(rng.randn(n)) * 5.0
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close,
         "volume": vol, "turnover": vol * close},
        index=idx,
    )


IMPULSE_GOLDEN = pathlib.Path(__file__).parent / "golden_impulse_flag.json"


def _impulse_df(n: int = 900) -> pd.DataFrame:
    """Deterministic impulse+consolidation+breakout pattern stream — the golden
    _golden_df yields 0 impulse_flag trades, so this dedicated set exercises its
    stop / T1-breakeven / T2 / expiry paths."""
    rng = np.random.RandomState(0)
    rows = []
    price = 100.0
    while len(rows) < n:
        for _ in range(rng.randint(6, 12)):
            o = price; c = price + rng.randn() * 0.4
            h = max(o, c) + abs(rng.randn()) * 0.2; l = min(o, c) - abs(rng.randn()) * 0.2
            rows.append((o, h, l, c, 10 + abs(rng.randn()) * 3)); price = c
        d = 1 if rng.rand() > 0.5 else -1                      # impulse bar
        o = price; c = price + d * rng.uniform(2.5, 4.0)
        h, l = (c + 0.05, o - 0.05) if d > 0 else (o + 0.05, c - 0.05)
        rows.append((o, h, l, c, 60 + abs(rng.randn()) * 10)); price = c
        for _ in range(3):                                     # consolidation
            o = price; c = price + rng.randn() * 0.25
            h = max(o, c) + 0.1; l = min(o, c) - 0.1
            rows.append((o, h, l, c, 8 + abs(rng.randn()) * 2)); price = c
        o = price; c = price + d * rng.uniform(1.0, 2.0)       # breakout
        h = max(o, c) + 0.1; l = min(o, c) - 0.1
        rows.append((o, h, l, c, 30.0)); price = c
    rows = rows[:n]
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    return df.assign(turnover=df["close"] * df["volume"])


def _trades_of(result) -> list:
    return [
        [
            t.entry_ts.isoformat() if t.entry_ts else None,
            round(t.entry_price, 6),
            t.exit_ts.isoformat() if t.exit_ts else None,
            round(t.exit_price, 6),
            t.exit_reason.value if t.exit_reason else None,
            t.direction.value,
        ]
        for t in result.trades
    ]


def _impulse_signature() -> list:
    strat = S.ImpulseFlagStrategy(params_for("impulse_flag"))
    return _trades_of(Backtester(strat, trading_config=TradingConfig()).run(_impulse_df(), interval="5"))


def _write_impulse_fixture() -> None:
    IMPULSE_GOLDEN.write_text(json.dumps(_impulse_signature(), indent=1))
    print("wrote", IMPULSE_GOLDEN.name, len(_impulse_signature()), "trades")


def test_impulse_flag_crafted_parity():
    assert _impulse_signature() == json.loads(IMPULSE_GOLDEN.read_text())


# ── Crafted golden: g_breakout's compression → breakout gate ───────────────────
# Like impulse_flag, g_breakout yields 0 trades on _golden_df (which has no tight
# compressions to squeeze), so this dedicated set exercises its squeeze + breakout
# + structural-stop paths on an explicit consolidation-into-level pattern.
G_BREAKOUT_GOLDEN = pathlib.Path(__file__).parent / "golden_g_breakout.json"


def _squeeze_df(n: int = 900) -> pd.DataFrame:
    """Repeated up-leg (seeds a resistance pivot) → pullback → tight multi-bar
    squeeze creeping just under the level → breakout bar closing above it."""
    rng = np.random.RandomState(7)
    rows = []
    price = 100.0
    while len(rows) < n:
        for _ in range(5):                                 # up-leg to a peak (normal bars)
            o = price; c = price + rng.uniform(0.6, 1.1)
            h = max(o, c) + abs(rng.randn()) * 0.15; l = min(o, c) - abs(rng.randn()) * 0.15
            rows.append((o, h, l, c, 12)); price = c
        peak = price + 0.4
        rows.append((price, peak, price - 0.2, price - 0.1, 12)); price = price - 0.1  # pivot-high bar
        for _ in range(4):                                 # pullback below the level
            o = price; c = price - rng.uniform(0.5, 0.9)
            h = max(o, c) + abs(rng.randn()) * 0.15; l = min(o, c) - abs(rng.randn()) * 0.15
            rows.append((o, h, l, c, 12)); price = c
        for _ in range(6):                                 # tight squeeze toward the peak
            o = price; c = min(price + (peak - 0.25 - price) * 0.4 + rng.randn() * 0.03, peak - 0.12)
            h = max(o, c) + 0.05; l = min(o, c) - 0.05
            rows.append((o, h, l, c, 6)); price = c
        o = price; c = peak + 0.6                           # breakout bar
        rows.append((o, c + 0.1, o - 0.1, c, 30)); price = c
        for _ in range(3):                                 # drift to separate blocks
            o = price; c = price + rng.randn() * 0.3
            h = max(o, c) + 0.1; l = min(o, c) - 0.1
            rows.append((o, h, l, c, 10)); price = c
    rows = rows[:n]
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    return df.assign(turnover=df["close"] * df["volume"])


def _g_breakout_signature() -> list:
    strat = S.GBreakoutStrategy(params_for("g_breakout"))
    return _trades_of(Backtester(strat, trading_config=TradingConfig()).run(_squeeze_df(), interval="5"))


def _write_g_breakout_fixture() -> None:
    G_BREAKOUT_GOLDEN.write_text(json.dumps(_g_breakout_signature(), indent=1))
    print("wrote", G_BREAKOUT_GOLDEN.name, len(_g_breakout_signature()), "trades")


def test_g_breakout_crafted_parity():
    assert _g_breakout_signature() == json.loads(G_BREAKOUT_GOLDEN.read_text())


def _signature(strat_cls) -> list:
    strat = strat_cls(params_for(strat_cls.name))   # per-strategy params; values unchanged
    result = Backtester(strat, trading_config=TradingConfig()).run(_golden_df(), interval="15")
    return [
        [
            t.entry_ts.isoformat() if t.entry_ts else None,
            round(t.entry_price, 6),
            t.exit_ts.isoformat() if t.exit_ts else None,
            round(t.exit_price, 6),
            t.exit_reason.value if t.exit_reason else None,
            t.direction.value,
        ]
        for t in result.trades
    ]


def _write_fixture() -> None:
    data = {name: _signature(cls) for name, cls in STRATEGIES.items()}
    GOLDEN.write_text(json.dumps(data, indent=1))
    print("wrote", GOLDEN.name, {k: len(v) for k, v in data.items()})


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_golden_parity(name):
    expected = json.loads(GOLDEN.read_text())[name]
    assert _signature(STRATEGIES[name]) == expected


# ── Per-detector golden: the level_* strategies under each non-default detector ─
# pivot_level (the default) is already pinned by the main fixture above; these pin
# the two ported detectors so a drift in the cluster_level / touch_level port
# fails here. Keyed "<strategy>:<detector>".
LEVEL_DETECTOR_GOLDEN = pathlib.Path(__file__).parent / "golden_level_detectors.json"
# Each level strategy's DEFAULT detector is already pinned by the main fixture; these
# pin it under its two NON-default detectors so a drift in any detector port fails
# here. Keyed "<strategy>:<detector>". g_breakout is omitted — it yields 0 trades on
# _golden_df (no compression to squeeze); its crafted squeeze fixture covers it.
_LEVEL_DETECTOR_STRATS = (
    "level_breakout", "level_breakout_inv", "g_bounce", "g_breakout_false", "g_range",
)
_LEVEL_DETECTOR_CASES = [
    (sname, det)
    for sname in _LEVEL_DETECTOR_STRATS
    for det in ("pivot_level", "cluster_level", "touch_level")
    if det != params_for(sname).level_detector
]


def _level_detector_signature(sname: str, det: str) -> list:
    cls = STRATEGIES[sname]
    cfg = dataclasses.replace(params_for(sname), level_detector=det)
    result = Backtester(cls(cfg), trading_config=TradingConfig()).run(_golden_df(), interval="15")
    return _trades_of(result)


def _write_level_detector_fixture() -> None:
    data = {f"{s}:{d}": _level_detector_signature(s, d) for s, d in _LEVEL_DETECTOR_CASES}
    LEVEL_DETECTOR_GOLDEN.write_text(json.dumps(data, indent=1))
    print("wrote", LEVEL_DETECTOR_GOLDEN.name, {k: len(v) for k, v in data.items()})


@pytest.mark.parametrize("sname,det", _LEVEL_DETECTOR_CASES)
def test_golden_level_detectors(sname, det):
    expected = json.loads(LEVEL_DETECTOR_GOLDEN.read_text())[f"{sname}:{det}"]
    assert _level_detector_signature(sname, det) == expected


if __name__ == "__main__":
    _write_fixture()
    _write_impulse_fixture()
    _write_g_breakout_fixture()
    _write_level_detector_fixture()
