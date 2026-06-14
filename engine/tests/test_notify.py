"""Tests for signal notifications (engine/notify.py + LiveEngine wiring).

Offline — no network, no real notifier backends. A capturing fake notifier
stands in for browser/desktop/telegram so the engine's detect/dedup/prime logic
is what's under test.

Run with: pytest engine/tests/test_notify.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.core import Direction, ExitReason
from engine.live import LiveEngine
from engine.notify import (
    BrowserNotifier,
    MacDesktopNotifier,
    NotifyEvent,
    TelegramNotifier,
    _beep_wav,
    make_notifiers,
)


class _Capturing:
    """Records every event instead of surfacing it anywhere."""

    def __init__(self) -> None:
        self.events: list[NotifyEvent] = []

    def notify(self, event: NotifyEvent) -> None:
        self.events.append(event)


class _Boom:
    def notify(self, event: NotifyEvent) -> None:
        raise RuntimeError("notifier exploded")


class _StubStrategy:
    name = "stub"

    def prepare(self, df):
        return df

    def on_bar(self, i, df, state):
        return None


def _engine(tmp_path, notifiers, db="n.db", chart="n.html"):
    return LiveEngine(
        strategy=_StubStrategy(),
        symbol="BTCUSDT",
        interval="15",
        num_candles=5,
        db_path=str(tmp_path / db),
        chart_path=str(tmp_path / chart),
        notifiers=notifiers,
    )


def _open_close(state, direction, entry_price, exit_price, reason=ExitReason.SIGNAL_FLIP):
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    state.enter(direction, ts, entry_price)
    state.exit(ts + pd.Timedelta(minutes=15), exit_price, reason)


# ── NotifyEvent formatting ───────────────────────────────────────────────────


class TestNotifyEvent:
    def test_entry_headline_and_no_pnl(self):
        ev = NotifyEvent(
            kind="entry", strategy="ema", symbol="BTCUSDT", interval="15",
            direction="long", price=100.5, timestamp=pd.Timestamp("2026-01-01", tz="UTC"),
        )
        assert "ENTRY" in ev.headline() and "LONG" in ev.headline()
        assert not ev.is_exit
        assert "bps" not in ev.detail()      # no P&L on an entry

    def test_exit_detail_carries_pnl_and_reason(self):
        ev = NotifyEvent(
            kind="exit", strategy="ema", symbol="BTCUSDT", interval="15",
            direction="short", price=99.0, timestamp=pd.Timestamp("2026-01-01", tz="UTC"),
            exit_reason="take_profit", pnl_bps=42.0, pnl_currency=12.5, equity_after=10012.5,
        )
        d = ev.detail()
        assert ev.is_exit and ev.is_win
        assert "take_profit" in d and "+42.0 bps" in d and "equity 10012.50" in d

    def test_exit_detail_shows_zero_currency_and_blown_equity(self):
        # 0.0 must render, not vanish: a break-even currency P&L and a blown
        # account (equity floored to 0.00) are exactly what you want to see.
        ev = NotifyEvent(
            kind="exit", strategy="ema", symbol="BTCUSDT", interval="15",
            direction="long", price=100.0, timestamp=pd.Timestamp("2026-01-01", tz="UTC"),
            exit_reason="stop_loss", pnl_bps=-9999.0, pnl_currency=0.0, equity_after=0.0,
        )
        d = ev.detail()
        assert "+0.00" in d and "equity 0.00" in d


def test_telegram_redacts_token_from_logged_exception(monkeypatch, caplog):
    """A requests network error stringifies with the token-bearing URL; the
    logged warning must not contain the token (a full-control credential)."""
    import logging

    import requests

    token = "123456:SUPER_SECRET_TOKEN"
    tg = TelegramNotifier(token, "999")

    def _raise(*args, **kwargs):
        # Mirrors what requests raises on a connection failure: the URL (with the
        # token in the path) appears verbatim in the exception message.
        raise requests.exceptions.ConnectionError(
            f"HTTPSConnectionPool(host='api.telegram.org', port=443): "
            f"Max retries exceeded with url: /bot{token}/sendMessage"
        )

    monkeypatch.setattr(requests, "post", _raise)
    ev = NotifyEvent(
        kind="exit", strategy="ema", symbol="BTCUSDT", interval="15",
        direction="long", price=100.0, timestamp=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    with caplog.at_level(logging.WARNING, logger="engine.notify"):
        tg._send(ev)   # synchronous (bypasses the daemon thread)

    assert token not in caplog.text       # the secret never reaches the logs
    assert "<token>" in caplog.text       # …it was redacted, and we still logged


def test_beep_wav_is_valid_riff():
    raw = _beep_wav(880.0, 50)
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    assert len(raw) > 44   # header + some samples


# ── Factory ──────────────────────────────────────────────────────────────────


class TestMakeNotifiers:
    def test_empty_spec_returns_nothing(self):
        assert make_notifiers(None) == []
        assert make_notifiers("") == []

    def test_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="Unknown notify channel"):
            make_notifiers("sms")

    def test_browser_skipped_outside_jupyter(self, monkeypatch):
        monkeypatch.setattr("engine.notify.in_jupyter", lambda: False)
        assert make_notifiers("browser") == []   # warned + skipped, not raised

    def test_browser_built_in_jupyter(self, monkeypatch):
        monkeypatch.setattr("engine.notify.in_jupyter", lambda: True)
        monkeypatch.setattr(BrowserNotifier, "__init__", lambda self: None)  # no IPython
        built = make_notifiers("browser")
        assert len(built) == 1 and isinstance(built[0], BrowserNotifier)

    def test_desktop_skipped_off_macos(self, monkeypatch):
        monkeypatch.setattr("engine.notify.sys.platform", "linux")
        assert make_notifiers("desktop") == []

    def test_desktop_built_on_macos(self, monkeypatch):
        monkeypatch.setattr("engine.notify.sys.platform", "darwin")
        built = make_notifiers("desktop")
        assert len(built) == 1 and isinstance(built[0], MacDesktopNotifier)

    def test_telegram_skipped_without_env(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert make_notifiers("telegram") == []

    def test_telegram_built_with_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
        built = make_notifiers("telegram")
        assert len(built) == 1 and isinstance(built[0], TelegramNotifier)

    def test_duplicates_and_aliases_collapse(self, monkeypatch):
        monkeypatch.setattr("engine.notify.sys.platform", "darwin")
        built = make_notifiers("desktop, mac , macos")   # all the same channel
        assert len(built) == 1


# ── LiveEngine detect / dedup / prime ─────────────────────────────────────────


class TestLiveNotifyDispatch:
    def test_first_tick_primes_and_emits_nothing(self, tmp_path):
        cap = _Capturing()
        eng = _engine(tmp_path, [cap])
        _open_close(eng._state, Direction.LONG, 100.0, 110.0)   # pre-existing history
        eng._state.enter(Direction.SHORT, pd.Timestamp("2026-01-01 01:00", tz="UTC"), 109.0)

        eng._emit_notifications(eng._state)

        assert cap.events == []                 # whole window treated as already-seen
        assert eng._notify_primed

    def test_open_then_close_emits_entry_once_then_exit(self, tmp_path):
        # The normal path: a trade seen OPEN on one tick, CLOSED on the next.
        cap = _Capturing()
        eng = _engine(tmp_path, [cap])
        eng._emit_notifications(eng._state)     # prime on empty state

        ts = pd.Timestamp("2026-01-01 01:00", tz="UTC")
        eng._state.enter(Direction.LONG, ts, 100.0)         # tick A: now open
        eng._emit_notifications(eng._state)
        eng._state.exit(ts + pd.Timedelta(minutes=15), 110.0, ExitReason.SIGNAL_FLIP)
        eng._emit_notifications(eng._state)                 # tick B: now closed

        assert [e.kind for e in cap.events] == ["entry", "exit"]   # entry NOT repeated
        assert cap.events[0].direction == "long" and cap.events[0].pnl_bps is None
        assert cap.events[1].pnl_bps is not None

    def test_trade_first_seen_closed_gets_catchup_entry(self, tmp_path):
        # The edge path (bars all close between two ticks): emit entry THEN exit,
        # in order, even though the engine never observed the trade open.
        cap = _Capturing()
        eng = _engine(tmp_path, [cap])
        eng._emit_notifications(eng._state)     # prime on empty state

        _open_close(eng._state, Direction.LONG, 100.0, 110.0)   # opened+closed unseen
        eng._state.enter(Direction.SHORT, pd.Timestamp("2026-01-01 02:00", tz="UTC"), 108.0)  # fresh open
        eng._emit_notifications(eng._state)

        kinds = [(e.kind, e.direction) for e in cap.events]
        assert kinds == [("entry", "long"), ("exit", "long"), ("entry", "short")]

    def test_exit_notified_once_across_ticks(self, tmp_path):
        cap = _Capturing()
        eng = _engine(tmp_path, [cap])
        eng._emit_notifications(eng._state)     # prime
        _open_close(eng._state, Direction.LONG, 100.0, 110.0)

        eng._emit_notifications(eng._state)     # tick 2: first sight → notify
        eng._emit_notifications(eng._state)     # tick 3: same closed trade → no re-fire
        assert sum(e.kind == "exit" for e in cap.events) == 1

    def test_notifier_exception_does_not_break_loop(self, tmp_path):
        cap = _Capturing()
        eng = _engine(tmp_path, [_Boom(), cap])   # _Boom raises; cap must still receive
        eng._emit_notifications(eng._state)        # prime
        _open_close(eng._state, Direction.LONG, 100.0, 110.0)

        eng._emit_notifications(eng._state)         # must not raise
        assert sum(e.kind == "exit" for e in cap.events) == 1

    def test_string_spec_resolved_in_constructor(self, tmp_path, monkeypatch):
        # A string spec passed to the engine is parsed via make_notifiers.
        monkeypatch.setattr("engine.notify.sys.platform", "darwin")
        eng = _engine(tmp_path, "desktop")
        assert len(eng._notifiers) == 1
        assert isinstance(eng._notifiers[0], MacDesktopNotifier)

    def test_no_notifiers_is_inert(self, tmp_path):
        eng = _engine(tmp_path, None)
        assert eng._notifiers == []
        # _tick guards on self._notifiers, so this never runs in practice; calling
        # it directly still must not emit or crash.
        _open_close(eng._state, Direction.LONG, 100.0, 110.0)
        eng._emit_notifications(eng._state)   # primes, no notifiers to call
        assert eng._notify_primed
