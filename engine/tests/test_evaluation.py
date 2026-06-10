"""Tests for the robustness tooling (sweep / walk-forward / Monte Carlo).

The headline guarantee is the *consistency* test: ``metrics_from_trades`` must
reproduce the engine's own bps metrics byte-for-byte, so a stitched OOS trade
list is scored exactly the way ``Backtester`` scores a single run.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine.backtester import Backtester
from engine.strategies import EMACrossoverStrategy
from engine.strategy_configurator import StrategyConfig
from engine.trade_configurator import TradingConfig
from engine.evaluation import (
    sweep, walk_forward, monte_carlo, metrics_from_trades, equity_curve, grid_search,
)


def _df(n: int = 1500) -> pd.DataFrame:
    """Deterministic OHLCV: drift + cycle + noise (mirrors the golden fixture)."""
    rng = np.random.RandomState(20240601)
    t = np.arange(n)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.8 + 0.02) + 8.0 * np.sin(t / 50.0)
    high = close + np.abs(rng.randn(n)) * 1.2 + 0.3
    low = close - np.abs(rng.randn(n)) * 1.2 - 0.3
    openp = close + rng.randn(n) * 0.5
    vol = 10.0 + np.abs(rng.randn(n)) * 5.0
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close,
         "volume": vol, "turnover": vol * close}, index=idx)


# ── metrics consistency (the load-bearing test) ──────────────────────────────


def test_metrics_from_trades_matches_backtester():
    df = _df()
    res = Backtester(EMACrossoverStrategy(StrategyConfig()),
                     trading_config=TradingConfig()).run(df, interval="15")
    assert res.total_trades >= 2          # exercise the Sharpe branch
    m = metrics_from_trades(res.trades)

    assert m["trades"] == res.total_trades
    assert m["total_pnl_bps"] == pytest.approx(res.total_pnl_bps)
    assert m["avg_pnl_bps"] == pytest.approx(res.avg_pnl_bps)
    assert m["win_rate"] == pytest.approx(res.win_rate)
    assert m["max_drawdown_bps"] == pytest.approx(res.max_drawdown_bps)
    assert m["sharpe_approx"] == pytest.approx(res.sharpe_approx)
    if math.isinf(res.profit_factor):
        assert math.isinf(m["profit_factor"])
    else:
        assert m["profit_factor"] == pytest.approx(res.profit_factor)


def test_metrics_from_trades_empty():
    m = metrics_from_trades([])
    assert m["trades"] == 0
    assert m["total_pnl_bps"] == 0.0 and m["sharpe_approx"] == 0.0


def test_equity_curve_compounds():
    df = _df()
    res = Backtester(EMACrossoverStrategy(StrategyConfig())).run(df, interval="15")
    eq = equity_curve(res.trades, initial=10_000.0)
    assert len(eq) == res.total_trades + 1
    assert eq.iloc[0] == 10_000.0
    # full-fraction compounding of pnl_bps reproduces the final point
    expected = 10_000.0
    for t in res.trades:
        expected *= max(0.0, 1 + t.pnl_bps / 10_000.0)
    assert eq.iloc[-1] == pytest.approx(expected)


# ── sweep ────────────────────────────────────────────────────────────────────


def test_sweep_grid_size_and_columns():
    df = _df()
    grid = {"ema_fast": [5, 9, 13], "ema_slow": [20, 30]}
    out = sweep(EMACrossoverStrategy, df, grid, interval="15")
    assert len(out) == 6                                   # full Cartesian product
    for col in ("ema_fast", "ema_slow", "trades", "total_pnl_bps",
                "sharpe_approx", "max_drawdown_bps", "profit_factor"):
        assert col in out.columns
    # each row's metrics equal a direct backtest of that combo
    import dataclasses
    row = out.iloc[0]
    cfg = dataclasses.replace(StrategyConfig(), ema_fast=int(row.ema_fast),
                              ema_slow=int(row.ema_slow))
    direct = Backtester(EMACrossoverStrategy(cfg)).run(df, interval="15")
    assert row.total_pnl_bps == pytest.approx(direct.total_pnl_bps)


def test_sweep_rejects_unknown_field():
    with pytest.raises(ValueError):
        sweep(EMACrossoverStrategy, _df(200), {"not_a_field": [1, 2]})


# ── walk-forward ─────────────────────────────────────────────────────────────


def test_walk_forward_folds_and_oos():
    df = _df()
    grid = {"ema_fast": [5, 9], "ema_slow": [20, 30]}
    wf = walk_forward(EMACrossoverStrategy, df, grid,
                      train_bars=400, test_bars=200, interval="15")
    # non-overlapping test windows from bar 400: 400/600/800/1000/1200 → 5 folds
    assert len(wf.folds) == 5
    # OOS trades are the concatenation of per-fold OOS trades, chronological
    assert wf.metrics["trades"] == sum(f.oos_trades for f in wf.folds)
    ent = [t.entry_ts for t in wf.oos_trades]
    assert ent == sorted(ent)
    # every fold's chosen params are inside the grid
    for f in wf.folds:
        assert f.best_params["ema_fast"] in grid["ema_fast"]
        assert f.best_params["ema_slow"] in grid["ema_slow"]
        assert isinstance(f.best_params["ema_fast"], int)
    # OOS aggregate equals metrics over the stitched trades
    assert wf.metrics == metrics_from_trades(wf.oos_trades)
    assert "Walk-Forward" in wf.summary()
    assert wf.param_stability().shape[0] == 5


def test_walk_forward_deterministic():
    df = _df()
    grid = {"ema_fast": [5, 9], "ema_slow": [20, 30]}
    a = walk_forward(EMACrossoverStrategy, df, grid, train_bars=400, test_bars=200)
    b = walk_forward(EMACrossoverStrategy, df, grid, train_bars=400, test_bars=200)
    assert a.metrics == b.metrics
    assert [f.best_params for f in a.folds] == [f.best_params for f in b.folds]


def test_walk_forward_needs_enough_bars():
    with pytest.raises(ValueError):
        walk_forward(EMACrossoverStrategy, _df(100), {"ema_fast": [5]},
                     train_bars=400, test_bars=200)


# ── grid search (all dimensions) ─────────────────────────────────────────────


def test_grid_search_cross_product_and_columns():
    df = _df()
    loader = lambda spec: df          # ignore spec → same synthetic frame
    out = grid_search(
        EMACrossoverStrategy,
        strategy_grid={"ema_fast": [5, 9], "ema_slow": [20, 30]},
        trade_grid={"leverage": [1.0, 2.0]},
        exit_grid=[None, "fixed_2pct_rr3"],
        loader=loader,
    )
    assert len(out) == 2 * 2 * 2 * 2          # strat(4) × trade(2) × exit(2)
    for col in ("ema_fast", "ema_slow", "leverage", "exit",
                "total_pnl_bps", "total_return_pct", "final_equity", "sharpe_approx"):
        assert col in out.columns


def test_grid_search_pnl_bps_is_sizing_invariant():
    df = _df()
    loader = lambda spec: df
    out = grid_search(
        EMACrossoverStrategy,
        strategy_grid={"ema_fast": [9]},
        trade_grid={"leverage": [1.0, 3.0]},
        loader=loader,
    )
    assert out["trades"].iloc[0] > 0
    # same signals, different leverage: bps P&L unchanged, equity return scales with leverage
    assert out["total_pnl_bps"].nunique() == 1
    assert out["total_return_pct"].nunique() == 2


def test_grid_search_rejects_unknown_field():
    with pytest.raises(ValueError):
        grid_search(EMACrossoverStrategy, trade_grid={"not_a_field": [1]},
                    loader=lambda spec: _df(200))


# ── Monte Carlo ──────────────────────────────────────────────────────────────


def test_monte_carlo_deterministic_and_bounded():
    pnl = [120.0, -80.0, 50.0, -30.0, 200.0, -150.0, 60.0, -40.0, 90.0, -70.0]
    a = monte_carlo(pnl, n_sims=2000, block=3, seed=7)
    b = monte_carlo(pnl, n_sims=2000, block=3, seed=7)
    assert a.terminal_return_pct == b.terminal_return_pct      # seed → reproducible
    assert a.n_trades == 10 and a.block == 3
    assert 0.0 <= a.prob_profit <= 1.0
    tr = a.terminal_return_pct
    assert tr["p5"] <= tr["p50"] <= tr["p95"]                  # ordered percentiles
    dd = a.max_drawdown_pct
    assert dd["p5"] <= dd["p50"] <= dd["p95"]
    assert dd["p5"] >= 0.0                                     # drawdown is non-negative
    assert "Monte Carlo" in a.summary()


def test_monte_carlo_accepts_trades():
    df = _df()
    res = Backtester(EMACrossoverStrategy(StrategyConfig())).run(df, interval="15")
    mc = monte_carlo(res.trades, n_sims=500, block=4, seed=1)
    assert mc.n_trades == res.total_trades


def test_monte_carlo_empty_raises():
    with pytest.raises(ValueError):
        monte_carlo([], n_sims=10)
