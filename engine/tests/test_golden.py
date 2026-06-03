"""Golden trade fixtures — the safety net for the exit-policy conversion.

Each strategy is run on one fixed, deterministic synthetic dataset and its trade
list is snapshotted to ``golden_trades.json``. As strategies are migrated from
inline exits to injected exit policies, these tests must stay green: a faithful
conversion reproduces byte-identical trades (entry/exit ts, prices, reason,
direction). A drift fails here and must be reconciled deliberately.

Regenerate the fixture (only when an intended behaviour change is accepted)::

    python -m engine.tests.test_golden   # or: python engine/tests/test_golden.py

The two ML strategies are excluded — they need a trained model / order-flow data,
which would make the snapshot non-deterministic across environments. They stay
covered by engine/tests/test_ml.py.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from engine import strategies as S
from engine.backtester import Backtester
from engine.strategy_configurator import StrategyConfig
from engine.trade_configurator import TradingConfig

GOLDEN = pathlib.Path(__file__).parent / "golden_trades.json"

# Strategy name → class. ML strategies excluded (model/order-flow dependent).
STRATEGIES = {
    "swing": S.SwingBreakoutStrategy,
    "swing_inv": S.InverseSwingBreakoutStrategy,
    "ema": S.EMACrossoverStrategy,
    "ema_inv": S.InverseEMACrossoverStrategy,
    "supertrend": S.SuperTrendStrategy,
    "supertrend_inv": S.InverseSuperTrendStrategy,
    "supertrend_adaptive": S.AdaptiveSuperTrendStrategy,
    "exhaustion_reversal": S.ExhaustionReversalStrategy,
    "impulse_flag": S.ImpulseFlagStrategy,
    "order_block": S.OrderBlockStrategy,
    "order_block_inv": S.InverseOrderBlockStrategy,
    "vwap_bands": S.VWAPBandsStrategy,
    "swing_zigzag": S.SwingZigZagStrategy,
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


def _signature(strat_cls) -> list:
    strat = strat_cls(StrategyConfig())
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


if __name__ == "__main__":
    _write_fixture()
