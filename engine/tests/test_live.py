"""Tests for LiveEngine wiring (offline — fake fetcher, no network).

Run with: pytest engine/tests/test_live.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.live import LiveEngine
from engine.strategy_configurator import EmaParams


class _StubStrategy:
    """Minimal strategy: LiveEngine._tick only needs name/prepare/on_bar.

    No self.config — the engine never reads strategy.config; a strategy applies
    its own params inside prepare()/on_bar() (proven in TestLiveStrategyConfig)."""

    name = "stub"

    def prepare(self, df):
        return df

    def on_bar(self, i, df, state):
        return None


class _FakeFetcher:
    def __init__(self):
        self.calls: list[dict] = []

    def fetch_klines(self, **kwargs):
        self.calls.append(kwargs)
        idx = pd.DatetimeIndex(
            pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC").values,
            tz="UTC", name="timestamp",
        )
        cols = ["open", "high", "low", "close", "volume", "turnover"]
        return pd.DataFrame({c: np.arange(5, dtype=float) for c in cols}, index=idx)

    def close(self):
        pass


def _engine(tmp_path, category="linear"):
    return LiveEngine(
        strategy=_StubStrategy(),
        symbol="BTCUSDT",
        interval="15",
        category=category,
        num_candles=5,
        db_path=str(tmp_path / "live_records.db"),
        chart_path=str(tmp_path / "chart.html"),
    )


class TestLiveCategory:
    def test_category_stored(self, tmp_path):
        assert _engine(tmp_path, category="inverse").category == "inverse"

    def test_bad_category_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="category"):
            _engine(tmp_path, category="spot")

    def test_tick_forwards_category_to_fetcher(self, tmp_path, monkeypatch):
        monkeypatch.setattr("engine.live.build_chart", lambda *a, **k: None)
        engine = _engine(tmp_path, category="inverse")
        fake = _FakeFetcher()
        engine._fetcher = fake

        engine._tick()

        assert fake.calls[0]["category"] == "inverse"


class TestLiveStrategyConfig:
    """The engine runs the user's configured strategy params. It never reads
    strategy.config itself — the strategy applies its own config inside
    prepare()/on_bar(), and the engine calls those. Proven by a strategy that
    records the config it was handed when the engine ticks."""

    def test_tick_runs_strategy_with_its_configured_params(self, tmp_path, monkeypatch):
        monkeypatch.setattr("engine.live.build_chart", lambda *a, **k: None)

        class _RecordingStrategy:
            name = "stub"

            def __init__(self, config):
                self.config = config
                self.prepared_with = None

            def prepare(self, df):
                self.prepared_with = self.config  # engine calls this each tick
                return df

            def on_bar(self, i, df, state):
                return None

        cfg = EmaParams(ema_fast=7, ema_slow=33)   # non-default knobs
        strat = _RecordingStrategy(cfg)
        engine = LiveEngine(
            strategy=strat, symbol="BTCUSDT", interval="15", num_candles=5,
            db_path=str(tmp_path / "cfg.db"), chart_path=str(tmp_path / "cfg.html"),
        )
        engine._fetcher = _FakeFetcher()

        engine._tick()

        # The engine ran THIS strategy's prepare(), and self.config carried the
        # user's non-default knobs unchanged into the live loop — no divergence.
        assert strat.prepared_with is cfg
        assert strat.prepared_with.ema_fast == 7
        assert strat.prepared_with.ema_slow == 33


class TestLiveEquityLayer:
    """Live mode runs the same compounding equity/sizing as the backtester,
    applied incrementally and persisted so the paper-equity curve survives
    restarts. (Previously live silently dropped all sizing/equity fields.)"""

    def _engine(self, tmp_path, db_name="eq.db", chart="c.html", equity=10_000.0):
        from engine.trade_configurator import TradingConfig

        return LiveEngine(
            strategy=_StubStrategy(),
            symbol="BTCUSDT",
            interval="15",
            num_candles=5,
            db_path=str(tmp_path / db_name),
            chart_path=str(tmp_path / chart),
            trading_config=TradingConfig(initial_equity=equity),
        )

    def test_closed_trade_compounds_sizes_and_persists(self, tmp_path):
        from engine.core import Direction, ExitReason

        eng = self._engine(tmp_path)
        start = eng._equity
        assert start == pytest.approx(10_000.0)  # seeded from initial_equity

        ts = pd.Timestamp("2026-01-01", tz="UTC")
        eng._state.enter(Direction.LONG, ts, 100.0)
        eng._state.exit(ts + pd.Timedelta(minutes=15), 110.0, ExitReason.SIGNAL_FLIP)
        eng._apply_equity_to_new_closures()

        trade = eng._state.closed_trades[-1]
        assert trade.notional > 0                 # was silently 0 before
        assert trade.pnl_currency > 0             # +10% move, net of cost
        assert trade.equity_after == pytest.approx(eng._equity)
        assert eng._equity > start                # equity compounded

    def test_sizing_is_idempotent_and_restart_safe(self, tmp_path):
        from engine.core import Direction, ExitReason

        eng = self._engine(tmp_path)
        ts = pd.Timestamp("2026-01-01", tz="UTC")
        eng._state.enter(Direction.LONG, ts, 100.0)
        eng._state.exit(ts + pd.Timedelta(minutes=15), 110.0, ExitReason.SIGNAL_FLIP)
        eng._apply_equity_to_new_closures()
        after_one = eng._equity
        trade = eng._state.closed_trades[-1]

        # A second pass must not double-count the same closed trade.
        eng._apply_equity_to_new_closures()
        assert eng._equity == pytest.approx(after_one)

        # Persist the trade, then a fresh engine on the same DB resumes the curve.
        eng._store.save_trade(trade)
        eng2 = self._engine(tmp_path, chart="c2.html")
        assert eng2._equity == pytest.approx(after_one)        # equity recovered
        assert trade.trade_id in eng2._sized_trade_ids         # not re-sized


class TestLiveReplayRestart:
    """The live tick rebuilds the position by replaying the strategy over the
    whole window from a fresh state, so any tick — including the first after a
    restart — re-derives the SAME position with correctly-rebuilt internal
    state. This is the durable fix for restart amnesia (audit H1/M1/M4/M5): the
    old loop nudged a persisted position by the last bar against a prepare()-wiped
    scratchpad, which (e.g.) force-closed a restored exhaustion_reversal trade."""

    def _df(self, n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        t = np.linspace(0, 9 * np.pi, n)
        mid = 100.0 + 14.0 * np.sin(t) + np.linspace(0.0, 25.0, n)
        close = mid + rng.standard_normal(n) * 0.5
        high = close + np.abs(rng.standard_normal(n)) * 0.4 + 0.1
        low = close - np.abs(rng.standard_normal(n)) * 0.4 - 0.1
        open_ = close + rng.standard_normal(n) * 0.1
        vol = rng.uniform(50, 500, n)
        idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": vol, "turnover": close * vol},
            index=idx,
        )

    def _engine(self, tmp_path, strat, db="r.db", chart="r.html"):
        return LiveEngine(
            strategy=strat, symbol="BTCUSDT", interval="15", num_candles=500,
            db_path=str(tmp_path / db), chart_path=str(tmp_path / chart),
        )

    @staticmethod
    def _sig(t):
        return (t.entry_ts, round(t.entry_price, 6), t.exit_ts,
                round(t.exit_price, 6), t.exit_reason, t.direction)

    def _strat(self):
        from engine.strategies import ExhaustionReversalStrategy
        from engine.strategy_configurator import params_for
        return ExhaustionReversalStrategy(params_for("exhaustion_reversal"))

    def test_replay_holds_open_position_not_force_closed(self, tmp_path):
        """A window that ends mid-trade leaves the live engine HOLDING the open
        position (rebuilt with correct internal state) — the old loop would have
        force-closed/time-stopped it on the first tick."""
        from engine.backtester import Backtester
        from engine.core import ExitReason

        df = self._df()
        bt = Backtester(self._strat()).run(df, interval="15")
        # A real (non-force-close) multi-bar trade to sit inside.
        real = [t for t in bt.trades
                if t.exit_reason != ExitReason.FORCE_CLOSE
                and df.index.get_loc(t.exit_ts) - df.index.get_loc(t.entry_ts) >= 2]
        assert real, "test df must produce at least one multi-bar exhaustion trade"
        tr = real[0]
        entry_i = df.index.get_loc(tr.entry_ts)
        exit_i = df.index.get_loc(tr.exit_ts)
        window = df.iloc[: (entry_i + exit_i) // 2 + 1]    # cut mid-trade

        eng = self._engine(tmp_path, self._strat())
        state = eng._replay(eng.strategy.prepare(window))
        assert state.current_trade is not None                       # still open
        assert state.current_trade.entry_ts == tr.entry_ts           # the same trade
        assert state.current_trade.direction == tr.direction

    def test_restart_reproduces_position_and_equity(self, tmp_path, monkeypatch):
        """A fresh engine on the same DB (= a process restart) re-derives an
        identical position and does not double-count equity."""
        monkeypatch.setattr("engine.live.build_chart", lambda *a, **k: None)

        df = self._df()
        forming = df.iloc[[-1]].copy()
        forming.index = forming.index + pd.Timedelta(minutes=15)
        feed = pd.concat([df, forming])                # _tick drops the forming bar

        class _Fetcher:
            def fetch_klines(self, **kw):
                return feed.copy()
            def close(self):
                pass

        eng1 = self._engine(tmp_path, self._strat())
        eng1._fetcher = _Fetcher()
        eng1._tick()
        eq1, open1 = eng1._equity, eng1._state.current_trade
        closed1 = [self._sig(t) for t in eng1._state.closed_trades]

        eng2 = self._engine(tmp_path, self._strat(), chart="r2.html")   # restart, same DB
        eng2._fetcher = _Fetcher()
        eng2._tick()

        assert eng2._equity == pytest.approx(eq1)                 # no equity double-count
        assert [self._sig(t) for t in eng2._state.closed_trades] == closed1
        if open1 is not None:
            assert eng2._state.current_trade is not None
            assert eng2._state.current_trade.entry_ts == open1.entry_ts
            assert eng2._state.current_trade.direction == open1.direction


class TestLiveChartTrades:
    """The live chart uses the rich trade view: closed trades in the visible
    window plus the open position's entry marker."""

    def _engine(self, tmp_path):
        return LiveEngine(
            strategy=_StubStrategy(), symbol="BTCUSDT", interval="15", num_candles=5,
            db_path=str(tmp_path / "ch.db"), chart_path=str(tmp_path / "ch.html"),
        )

    def test_chart_trades_includes_closed_in_window_and_open_entry(self, tmp_path):
        from engine.core import Direction, ExitReason

        eng = self._engine(tmp_path)
        t0 = pd.Timestamp("2026-01-01", tz="UTC")
        eng._state.enter(Direction.LONG, t0, 100.0)                         # closed
        eng._state.exit(t0 + pd.Timedelta(minutes=15), 110.0, ExitReason.SIGNAL_FLIP)
        eng._state.enter(Direction.SHORT, t0 + pd.Timedelta(minutes=30), 109.0)  # open

        ct = eng._chart_trades(window_start=t0 - pd.Timedelta(minutes=5))
        assert len(ct) == 2
        assert ct[-1] is eng._state.current_trade   # open trade -> entry marker only
        assert ct[-1].exit_ts is None

    def test_chart_trades_drops_out_of_window(self, tmp_path):
        from engine.core import Direction, ExitReason

        eng = self._engine(tmp_path)
        t0 = pd.Timestamp("2026-01-01", tz="UTC")
        eng._state.enter(Direction.LONG, t0, 100.0)
        eng._state.exit(t0 + pd.Timedelta(minutes=15), 110.0, ExitReason.SIGNAL_FLIP)

        # A window that starts after both the exit and (absent) open entry → empty.
        assert eng._chart_trades(window_start=t0 + pd.Timedelta(hours=1)) == []
