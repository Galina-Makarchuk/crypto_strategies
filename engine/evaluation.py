"""Strategy-robustness tooling: parameter sweep → walk-forward → Monte Carlo.

These three answer *different* questions and are meant to be used **in that
order**:

* :func:`sweep` — run a parameter grid through the real :class:`Backtester` and
  return one metrics row per combination (the building block for the others).
* :func:`walk_forward` — rolling (or anchored) out-of-sample validation: on each
  *train* window pick the best params via :func:`sweep`, then run **only** the
  next *test* window with those params and stitch the OOS trades together. This
  is what tells you whether an edge survives re-tuning on causal data — i.e. it
  attacks **overfitting**, the dominant failure mode of a single-window backtest.
* :func:`monte_carlo` — block-bootstrap the *out-of-sample* trade returns into a
  distribution of terminal return / max drawdown. This characterises **luck /
  sequence risk** around a result.

Order matters: Monte Carlo on an in-sample, parameter-optimised backtest just
launders an overfit edge into a confident-looking distribution. Run it on the
walk-forward OOS trades, never on a tuned in-sample run.

Everything is built on the existing :class:`engine.backtester.Backtester` and
``Trade.pnl_bps`` — no look-ahead is introduced because each OOS segment is a
normal causal backtest of the test window (with an indicator-warmup prefix that
is excluded from attribution).

The bps metric formulas in :func:`metrics_from_trades` mirror
``Backtester._compute_stats`` exactly (a test pins the two together), so OOS
aggregates are computed the same way the engine computes a single run.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtester import Backtester
from .core import Trade
from .strategy_configurator import StrategyConfig
from .trade_configurator import TradingConfig

logger = logging.getLogger(__name__)


# ── metrics ──────────────────────────────────────────────────────────────────

_METRIC_COLUMNS = (
    "trades", "win_rate", "total_pnl_bps", "avg_pnl_bps",
    "profit_factor", "max_drawdown_bps", "sharpe_approx",
)


def metrics_from_trades(trades: list[Trade]) -> dict:
    """Aggregate bps metrics for a list of closed trades.

    Replicates ``Backtester._compute_stats`` (the bps half) so a stitched
    out-of-sample trade list is scored identically to a single engine run:
    trade-level Sharpe (sample stdev, not time-scaled), drawdown of the
    cumulative-PnL curve sampled at trade closes, profit factor with the
    all-wins → ``inf`` convention.
    """
    n = len(trades)
    out = {"trades": n, "win_rate": 0.0, "total_pnl_bps": 0.0, "avg_pnl_bps": 0.0,
           "profit_factor": 0.0, "max_drawdown_bps": 0.0, "sharpe_approx": 0.0}
    if n == 0:
        return out

    pnls = np.array([t.pnl_bps for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    out["win_rate"] = len(wins) / n
    out["total_pnl_bps"] = float(pnls.sum())
    out["avg_pnl_bps"] = float(pnls.mean())

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    if gross_loss > 0:
        out["profit_factor"] = gross_profit / gross_loss
    elif gross_profit > 0:
        out["profit_factor"] = float("inf")

    cum = np.cumsum(pnls)
    drawdowns = np.maximum.accumulate(cum) - cum
    out["max_drawdown_bps"] = float(drawdowns.max())

    if n >= 2:
        std = float(pnls.std(ddof=1))
        mean = out["avg_pnl_bps"]
        if std > 0:
            out["sharpe_approx"] = mean / std
        elif mean > 0:
            out["sharpe_approx"] = float("inf")
        elif mean < 0:
            out["sharpe_approx"] = float("-inf")
    return out


def equity_curve(trades: list[Trade], initial: float = 10_000.0) -> pd.Series:
    """Compounded equity after each trade (fixed full-fraction: ``*= 1+pnl_bps/1e4``),
    indexed by exit timestamp, with the starting point prepended. The per-trade
    growth factor is floored at 0 — you cannot lose more than 100% on a trade."""
    eq = initial
    pts, idx = [initial], [trades[0].entry_ts if trades else None]
    for t in trades:
        eq *= max(0.0, 1.0 + t.pnl_bps / 10_000.0)
        pts.append(eq)
        idx.append(t.exit_ts if t.exit_ts is not None else t.entry_ts)
    return pd.Series(pts, index=idx, name="equity")


# ── parameter sweep ──────────────────────────────────────────────────────────


def _instantiate(strategy_cls, config: StrategyConfig, exit_policy):
    return strategy_cls(config, exit_policy=exit_policy)


def sweep(
    strategy_cls,
    df: pd.DataFrame,
    grid: dict[str, list],
    *,
    symbol: str = "BTCUSDT",
    interval: str = "15",
    trading_config: TradingConfig | None = None,
    base_config: StrategyConfig | None = None,
    exit_policy=None,
) -> pd.DataFrame:
    """Run the Cartesian product of ``grid`` through the real backtester.

    ``grid`` maps :class:`StrategyConfig` field names to value lists, e.g.
    ``{"ema_fast": [5, 9, 13], "ema_slow": [21, 34]}``. Each combination is
    applied with :func:`dataclasses.replace` onto ``base_config`` (default
    ``StrategyConfig()``), backtested, and reduced to one metrics row. Returns a
    DataFrame with the param columns plus :data:`_METRIC_COLUMNS` —
    one row per combination (no filtering; callers rank/filter).
    """
    base = base_config or StrategyConfig()
    tc = trading_config or TradingConfig()
    keys = list(grid)
    if not keys:
        raise ValueError("grid must contain at least one parameter")
    for k in keys:
        if not hasattr(base, k):
            raise ValueError(f"{k!r} is not a StrategyConfig field")

    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        cfg = dataclasses.replace(base, **params)
        res = Backtester(_instantiate(strategy_cls, cfg, exit_policy),
                         symbol=symbol, trading_config=tc).run(df, interval=interval)
        rows.append({
            **params,
            "trades": res.total_trades,
            "win_rate": res.win_rate,
            "total_pnl_bps": res.total_pnl_bps,
            "avg_pnl_bps": res.avg_pnl_bps,
            "profit_factor": res.profit_factor,
            "max_drawdown_bps": res.max_drawdown_bps,
            "sharpe_approx": res.sharpe_approx,
        })
    return pd.DataFrame(rows)


# ── full grid search across all four dimensions ──────────────────────────────


def grid_search(
    strategy_cls,
    *,
    strategy_grid: dict[str, list] | None = None,
    trade_grid: dict[str, list] | None = None,
    exit_grid: list | None = None,
    data_grid: dict[str, list] | None = None,
    base_config: StrategyConfig | None = None,
    base_trading: TradingConfig | None = None,
    base_data=None,
    loader=None,
) -> pd.DataFrame:
    """Cartesian product across **all four** config dimensions at once.

    Each grid is optional — omit one and that dimension stays fixed at its base:

    * ``strategy_grid`` — :class:`StrategyConfig` fields → applied to ``base_config``.
    * ``trade_grid``    — :class:`TradingConfig` fields → applied to ``base_trading``.
    * ``data_grid``     — :class:`~engine.data_configurator.DataSpec` fields →
      applied to ``base_data``; each distinct spec is loaded **once** (cached).
    * ``exit_grid``     — a list whose items are an ``EXIT_PRESETS`` name (str), an
      :class:`~engine.exits.ExitPolicy` instance, or ``None`` (strategy default).

    Returns one row per combination: the varied params + metrics, sortable to find
    the most profitable mix. **Note:** ``pnl_bps`` is sizing-invariant, so when you
    sweep *trade* knobs (leverage / sizing) rank by ``total_return_pct`` /
    ``final_equity`` — ``total_pnl_bps`` will not move.
    """
    from .data_configurator import ACTIVE, load_data           # lazy: avoid import cost when unused
    from .strategy_configurator import EXIT_PRESETS

    base_config = base_config or StrategyConfig()
    base_trading = base_trading or TradingConfig()
    base_data = base_data or ACTIVE
    loader = loader or load_data         # callable(DataSpec) -> df; injectable for tests / custom feeds

    def _validate(grid, proto, what):
        for k in (grid or {}):
            if not hasattr(proto, k):
                raise ValueError(f"{k!r} is not a {what} field")

    _validate(strategy_grid, base_config, "StrategyConfig")
    _validate(trade_grid, base_trading, "TradingConfig")
    _validate(data_grid, base_data, "DataSpec")

    def _combos(base, grid):
        if not grid:
            return [({}, base)]
        keys = list(grid)
        return [(dict(zip(keys, vals)), dataclasses.replace(base, **dict(zip(keys, vals))))
                for vals in itertools.product(*(grid[k] for k in keys))]

    def _exit_combos():
        if exit_grid is None:
            return [(None, None)]
        out = []
        for e in exit_grid:
            if e is None:
                out.append(("default", None))
            elif isinstance(e, str):
                out.append((e, EXIT_PRESETS[e]()))
            else:
                out.append((type(e).__name__, e))
        return out

    data_combos, strat_combos = _combos(base_data, data_grid), _combos(base_config, strategy_grid)
    trade_combos, exit_combos = _combos(base_trading, trade_grid), _exit_combos()
    total = len(data_combos) * len(strat_combos) * len(trade_combos) * len(exit_combos)
    logger.info("grid_search: %d combinations (data %d × strategy %d × trade %d × exit %d)",
                total, len(data_combos), len(strat_combos), len(trade_combos), len(exit_combos))

    rows = []
    for d_params, spec in data_combos:
        df = loader(spec)                                      # one load per dataset (cached)
        for s_params, scfg in strat_combos:
            for t_params, tcfg in trade_combos:
                for exit_label, exit_pol in exit_combos:
                    res = Backtester(_instantiate(strategy_cls, scfg, exit_pol),
                                     symbol=spec.symbol, trading_config=tcfg).run(
                        df, interval=spec.interval)
                    rows.append({
                        **d_params, **s_params, **t_params,
                        **({"exit": exit_label} if exit_grid is not None else {}),
                        "trades": res.total_trades,
                        "win_rate": res.win_rate,
                        "total_pnl_bps": res.total_pnl_bps,
                        "profit_factor": res.profit_factor,
                        "max_drawdown_bps": res.max_drawdown_bps,
                        "sharpe_approx": res.sharpe_approx,
                        "total_return_pct": res.total_return_pct,
                        "final_equity": res.final_equity,
                    })
    return pd.DataFrame(rows)


# ── walk-forward ─────────────────────────────────────────────────────────────


@dataclass
class FoldRecord:
    """One train→test fold of a walk-forward run."""
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    is_objective: float          # the chosen objective's value in-sample
    is_total_pnl_bps: float      # the chosen combo's in-sample P&L (for WF efficiency)
    oos_total_pnl_bps: float
    oos_trades: int
    oos_win_rate: float


@dataclass
class WalkForwardResult:
    """Result of a walk-forward run: per-fold records + the stitched OOS trades."""
    folds: list[FoldRecord]
    oos_trades: list[Trade]
    metrics: dict
    objective: str
    grid: dict
    train_bars: int
    test_bars: int
    anchored: bool

    def folds_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "fold": f.fold,
            "test_start": f.test_start, "test_end": f.test_end,
            **f.best_params,
            f"is_{self.objective}": f.is_objective,
            "oos_total_pnl_bps": f.oos_total_pnl_bps,
            "oos_trades": f.oos_trades, "oos_win_rate": f.oos_win_rate,
        } for f in self.folds])

    def param_stability(self) -> pd.DataFrame:
        """Best params chosen per fold — jumpy columns = overfit-prone params."""
        return pd.DataFrame(
            [{"fold": f.fold, **f.best_params} for f in self.folds]
        ).set_index("fold")

    def equity_curve(self, initial: float = 10_000.0) -> pd.Series:
        return equity_curve(self.oos_trades, initial)

    def summary(self) -> str:
        m = self.metrics
        is_pnl = sum(f.is_total_pnl_bps for f in self.folds)
        oos_sum = sum(f.oos_total_pnl_bps for f in self.folds)
        lines = [
            "═" * 60,
            f"  Walk-Forward ({'anchored' if self.anchored else 'rolling'}) "
            f"— {len(self.folds)} folds, objective={self.objective}",
            f"  train={self.train_bars} bars / test={self.test_bars} bars",
            "═" * 60,
            f"  OOS trades         : {m['trades']}",
            f"  OOS total P&L (bps): {m['total_pnl_bps']:+.1f}",
            f"  OOS win rate       : {m['win_rate']:.1%}",
            f"  OOS profit factor  : {m['profit_factor']:.2f}",
            f"  OOS max DD (bps)   : {m['max_drawdown_bps']:.1f}",
            f"  OOS Sharpe (approx): {m['sharpe_approx']:.2f}",
            f"  {'─' * 40}",
            f"  Σ in-sample  P&L (bps): {is_pnl:+.1f}",
            f"  Σ out-of-sample (bps) : {oos_sum:+.1f}",
        ]
        # WF efficiency = OOS P&L / IS P&L, meaningful only when the in-sample
        # optimisation actually found a profitable config (IS P&L > 0).
        if is_pnl > 0:
            lines.append(
                f"  WF efficiency (OOS/IS): {oos_sum / is_pnl:.2f}   "
                "(≈1 robust · «1 overfit · <0 inverted)")
        else:
            lines.append(
                "  WF efficiency (OOS/IS): n/a   "
                "(in-sample best was unprofitable — no edge to carry over)")
        lines.append("═" * 60)
        return "\n".join(lines)


def walk_forward(
    strategy_cls,
    df: pd.DataFrame,
    grid: dict[str, list],
    *,
    train_bars: int,
    test_bars: int,
    interval: str = "15",
    symbol: str = "BTCUSDT",
    trading_config: TradingConfig | None = None,
    base_config: StrategyConfig | None = None,
    exit_policy=None,
    objective: str = "total_pnl_bps",
    maximize: bool = True,
    min_trades: int = 0,
    warmup_bars: int = 100,
    anchored: bool = False,
) -> WalkForwardResult:
    """Rolling (or anchored) walk-forward optimisation + out-of-sample test.

    For each non-overlapping ``test_bars`` window, the preceding ``train_bars``
    window (or all history before it when ``anchored``) is swept over ``grid``;
    the combination maximising (or minimising) ``objective`` — among combos with
    at least ``min_trades`` trades — is run on the test window and the trades
    that **enter** inside it are collected. ``warmup_bars`` of history are
    prepended to each OOS run so indicators are warm, and are excluded from
    attribution.

    Returns a :class:`WalkForwardResult` whose ``metrics`` summarise the stitched
    OOS trades — the honest, look-ahead-free track record.
    """
    base = base_config or StrategyConfig()
    tc = trading_config or TradingConfig()
    n = len(df)
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    if train_bars + test_bars > n:
        raise ValueError(
            f"need at least train_bars+test_bars={train_bars + test_bars} bars, got {n}")

    folds: list[FoldRecord] = []
    oos_trades: list[Trade] = []
    fold_i = 0
    test_lo = train_bars
    while test_lo + test_bars <= n:
        train_lo = 0 if anchored else test_lo - train_bars
        train_hi = test_lo
        test_hi = test_lo + test_bars

        # In-sample: sweep the train window and pick the best params.
        sweep_df = sweep(strategy_cls, df.iloc[train_lo:train_hi], grid,
                         symbol=symbol, interval=interval, trading_config=tc,
                         base_config=base, exit_policy=exit_policy)
        if objective not in sweep_df.columns:
            raise ValueError(f"objective {objective!r} not in sweep metrics")
        eligible = sweep_df[sweep_df["trades"] >= min_trades] if min_trades else sweep_df
        if eligible.empty:                       # nobody cleared the bar → relax it
            eligible = sweep_df
        best = eligible.sort_values(objective, ascending=not maximize).iloc[0]
        best_params = {k: _coerce(best[k], grid[k]) for k in grid}

        # Out-of-sample: run the chosen params on the test window (+warmup prefix),
        # keep only trades that ENTER inside [test_start, test_end].
        oos_lo = max(0, test_lo - warmup_bars)
        cfg = dataclasses.replace(base, **best_params)
        oos_res = Backtester(_instantiate(strategy_cls, cfg, exit_policy),
                             symbol=symbol, trading_config=tc).run(
            df.iloc[oos_lo:test_hi], interval=interval)
        t_start, t_end = df.index[test_lo], df.index[test_hi - 1]
        fold_oos = [t for t in oos_res.trades
                    if t.entry_ts is not None and t_start <= t.entry_ts <= t_end]
        oos_trades.extend(fold_oos)

        fm = metrics_from_trades(fold_oos)
        folds.append(FoldRecord(
            fold=fold_i, train_start=df.index[train_lo], train_end=df.index[train_hi - 1],
            test_start=t_start, test_end=t_end, best_params=best_params,
            is_objective=float(best[objective]), is_total_pnl_bps=float(best["total_pnl_bps"]),
            oos_total_pnl_bps=fm["total_pnl_bps"], oos_trades=fm["trades"],
            oos_win_rate=fm["win_rate"]))
        fold_i += 1
        test_lo += test_bars

    return WalkForwardResult(
        folds=folds, oos_trades=oos_trades, metrics=metrics_from_trades(oos_trades),
        objective=objective, grid=grid, train_bars=train_bars, test_bars=test_bars,
        anchored=anchored)


def _coerce(value, grid_values):
    """Cast a value pulled from a DataFrame back to the python type of the grid
    (numpy.int64 → int, numpy.bool_ → bool) so it lands cleanly on StrategyConfig."""
    return type(grid_values[0])(value)


# ── Monte Carlo (block bootstrap on OOS trades) ──────────────────────────────


@dataclass
class MonteCarloResult:
    """Bootstrap distribution of outcomes for a trade sequence."""
    n_sims: int
    block: int
    n_trades: int
    initial_equity: float
    terminal_return_pct: dict       # mean + p5/p25/p50/p75/p95
    max_drawdown_pct: dict
    prob_profit: float              # fraction of sims ending above the start
    samples: dict = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        tr, dd = self.terminal_return_pct, self.max_drawdown_pct
        return "\n".join([
            "═" * 60,
            f"  Monte Carlo — {self.n_sims} sims, block={self.block}, "
            f"{self.n_trades} trades",
            "═" * 60,
            f"  Terminal return %  : median {tr['p50']:+.1f}   "
            f"[p5 {tr['p5']:+.1f} … p95 {tr['p95']:+.1f}]",
            f"  Max drawdown %     : median {dd['p50']:.1f}    "
            f"[p5 {dd['p5']:.1f} … p95 {dd['p95']:.1f}]",
            f"  P(profitable)      : {self.prob_profit:.1%}",
            "═" * 60,
        ])


def monte_carlo(
    trades_or_pnl,
    *,
    n_sims: int = 10_000,
    block: int = 5,
    seed: int = 0,
    initial_equity: float = 10_000.0,
) -> MonteCarloResult:
    """Block-bootstrap a trade-return sequence into an outcome distribution.

    Accepts a list of :class:`~engine.core.Trade` (uses ``pnl_bps``) or a raw
    sequence of per-trade bps returns. Resamples contiguous **blocks** (circular)
    rather than individual trades, so local serial correlation — loss clusters,
    regime streaks — is preserved; a plain i.i.d. shuffle understates drawdown.
    Each simulation compounds the resampled returns into an equity curve and
    records the terminal return and max drawdown. Deterministic given ``seed``.
    """
    pnl = np.array(
        [t.pnl_bps for t in trades_or_pnl] if _looks_like_trades(trades_or_pnl)
        else list(trades_or_pnl), dtype=float)
    n = len(pnl)
    if n == 0:
        raise ValueError("need at least one trade for Monte Carlo")
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)

    # (n_sims, n) resampled-return matrix from circular blocks.
    starts = rng.integers(0, n, size=(n_sims, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_sims, n_blocks * block)
    idx = idx[:, :n] % n
    seq = pnl[idx]

    growth = np.clip(1.0 + seq / 10_000.0, 0.0, None)     # floor a trade at −100%
    eq = initial_equity * np.cumprod(growth, axis=1)
    eq = np.concatenate([np.full((n_sims, 1), initial_equity), eq], axis=1)

    terminal = (eq[:, -1] / initial_equity - 1.0) * 100.0
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = ((peak - eq) / peak * 100.0).max(axis=1)

    def dist(a: np.ndarray) -> dict:
        p = np.percentile(a, [5, 25, 50, 75, 95])
        return {"mean": float(a.mean()), "p5": float(p[0]), "p25": float(p[1]),
                "p50": float(p[2]), "p75": float(p[3]), "p95": float(p[4])}

    return MonteCarloResult(
        n_sims=n_sims, block=block, n_trades=n, initial_equity=initial_equity,
        terminal_return_pct=dist(terminal), max_drawdown_pct=dist(mdd),
        prob_profit=float((terminal > 0).mean()),
        samples={"terminal_return_pct": terminal, "max_drawdown_pct": mdd})


def _looks_like_trades(seq) -> bool:
    return len(seq) > 0 and hasattr(seq[0], "pnl_bps")
