"""Signal notifications for live (paper) trading.

The live engine places no orders — these notifiers are pure, read-side alerts
that say "a signal just happened": an entry opened or an exit closed. They are
fired in LiveEngine._tick AFTER the strategy has run on the closed bar, so they
can never influence signal generation or reintroduce look-ahead.

Three channels ship here, selected via a comma-separated spec
("browser,desktop,telegram"). ``make_notifiers()`` builds the list and silently
skips any channel that isn't usable in the current environment, so a
misconfigured run degrades to fewer alerts instead of crashing the loop:

  browser  — in the Jupyter cell output: a coloured banner + a short beep
             (works in Safari and Chrome), plus a best-effort browser/OS
             notification. Skipped outside a Jupyter kernel.
  desktop  — a native macOS notification banner + sound via ``osascript``.
             Skipped off macOS.
  telegram — a message to a Telegram chat (reaches phone + desktop). Skipped
             unless both env vars below are set.

Secrets are read from the environment, never stored here:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Adding a channel = implement ``Notifier.notify`` + register it in
``_build_channel`` / ``_ALIASES``; nothing in the engine changes.
"""

from __future__ import annotations

import base64
import html as _htmlmod
import io
import json
import logging
import math
import os
import struct
import subprocess
import sys
import threading
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


def in_jupyter() -> bool:
    """True only inside a Jupyter / VS Code kernel (ZMQInteractiveShell), where
    rich HTML / audio output renders. Terminal IPython and plain scripts → False.
    Shared with the live engine so the chart link and the browser notifier agree
    on what 'in a notebook' means."""
    try:
        from IPython import get_ipython  # noqa: PLC0415 (optional dep, lazy)
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# ── Event ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NotifyEvent:
    """One thing worth alerting on: a live entry or exit. Built by LiveEngine
    from the just-updated PositionState and handed to every Notifier. Exit-only
    fields stay None for entries (no P&L exists until a position closes)."""

    kind: str                              # "entry" | "exit"
    strategy: str
    symbol: str
    interval: str
    direction: str                         # "long" | "short"
    price: float
    timestamp: datetime
    label: str = ""
    exit_reason: Optional[str] = None      # exits only
    pnl_bps: Optional[float] = None        # exits only
    pnl_currency: Optional[float] = None   # exits only
    equity_after: Optional[float] = None   # exits only

    @property
    def is_exit(self) -> bool:
        return self.kind == "exit"

    @property
    def is_win(self) -> bool:
        return (self.pnl_bps or 0.0) >= 0.0

    def headline(self) -> str:
        dot = "🟢" if self.direction == "long" else "🔴"
        verb = "EXIT" if self.is_exit else "ENTRY"
        return f"{dot} {verb} {self.direction.upper()} {self.symbol} {self.interval}"

    def detail(self) -> str:
        bits = [self.strategy, f"@ {self.price:g}"]
        if self.is_exit:
            if self.exit_reason:
                bits.append(self.exit_reason)
            if self.pnl_bps is not None:
                bits.append(f"{self.pnl_bps:+.1f} bps")
            # `is not None`, not truthiness: a break-even (0.00) or a blown
            # account (equity floored to 0.00) must still be shown, not hidden.
            if self.pnl_currency is not None:
                bits.append(f"{self.pnl_currency:+.2f}")
            if self.equity_after is not None:
                bits.append(f"equity {self.equity_after:.2f}")
        return " | ".join(bits)

    def text(self) -> str:
        return f"{self.headline()}\n{self.detail()}"


class Notifier(Protocol):
    """Anything that can surface a NotifyEvent. Implementations must be
    non-blocking (offload slow I/O to a thread) and must not raise — the engine
    wraps each call defensively, but a notifier that swallows its own failures
    keeps the logs clean."""

    def notify(self, event: NotifyEvent) -> None: ...


# ── Sound (a self-contained WAV, so the browser needs no external file) ────────


def _beep_wav(freq: float, ms: int, volume: float = 0.4) -> bytes:
    """A short sine-wave WAV (mono, 16-bit, 22.05 kHz) with a linear fade-out so
    it doesn't click. Returned as raw bytes for base64 embedding."""
    rate = 22_050
    n = int(rate * ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            env = volume * (1.0 - i / n)  # fade to zero over the tone
            sample = int(32767 * env * math.sin(2 * math.pi * freq * i / rate))
            frames += struct.pack("<h", sample)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _beep_data_uri(freq: float, ms: int) -> str:
    b64 = base64.b64encode(_beep_wav(freq, ms)).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


# A higher tone for entries, a lower one for exits — built once at import.
_ENTRY_BEEP = _beep_data_uri(880.0, 160)
_EXIT_BEEP = _beep_data_uri(440.0, 220)


def _html_escape(s: str) -> str:
    return _htmlmod.escape(s)


def _js_string(s: str) -> str:
    """A safely-quoted JS string literal (handles quotes, backslashes, unicode)."""
    return json.dumps(s)


def _osa_quote(s: str) -> str:
    """An AppleScript double-quoted string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ── Channels ───────────────────────────────────────────────────────────────


class BrowserNotifier:
    """In-notebook alert: a coloured banner + an autoplaying beep injected into
    the live cell's output, plus a best-effort browser/OS notification.

    Robust across Jupyter frontends (Lab / Notebook / VS Code) and browsers
    (Safari, Chrome): the banner and the ``<audio>`` element are plain rich-HTML
    output — no ``<script>``, which JupyterLab strips — so they always render.
    The OS-level Notification is layered on via the Javascript display mimetype
    and degrades silently if the browser hasn't granted permission. Sound
    autoplay needs the tab to have had a user gesture; clicking Run on the live
    cell counts, which is why permission is also requested up front there."""

    def __init__(self) -> None:
        from IPython.display import HTML, Javascript, display  # noqa: PLC0415
        self._display = display
        self._HTML = HTML
        self._Javascript = Javascript
        # Ask for notification permission once, while the cell is executing (the
        # Run click is the user gesture browsers require). Best-effort.
        try:
            self._display(self._Javascript(
                "if ('Notification' in window && Notification.permission === 'default')"
                " { Notification.requestPermission(); }"
            ))
        except Exception:  # noqa: BLE001 — a frontend without JS support must not break alerts
            pass

    def notify(self, event: NotifyEvent) -> None:
        beep = _EXIT_BEEP if event.is_exit else _ENTRY_BEEP
        if event.is_exit:
            colour = "#1a7f37" if event.is_win else "#cf222e"
        else:
            colour = "#0969da"
        headline = _html_escape(event.headline())
        detail = _html_escape(event.detail())
        ts = _html_escape(event.timestamp.strftime("%Y-%m-%d %H:%M UTC"))
        self._display(self._HTML(
            f'<div style="border-left:4px solid {colour};background:#f6f8fa;'
            f'padding:6px 10px;margin:3px 0;font-family:-apple-system,BlinkMacSystemFont,'
            f'sans-serif;border-radius:3px;">'
            f'<span style="color:{colour};font-weight:600;">{headline}</span>'
            f'<span style="color:#8c959f;font-size:80%;float:right;">{ts}</span>'
            f'<br><span style="color:#57606a;font-size:90%;">{detail}</span>'
            f'<audio autoplay src="{beep}"></audio>'
            f'</div>'
        ))
        # Best-effort OS notification so a backgrounded tab still surfaces it.
        try:
            self._display(self._Javascript(
                "if ('Notification' in window && Notification.permission === 'granted')"
                f" {{ new Notification({_js_string(event.headline())}, "
                f"{{body: {_js_string(event.detail())}}}); }}"
            ))
        except Exception:  # noqa: BLE001
            pass


class MacDesktopNotifier:
    """Native macOS banner + sound via ``osascript``, fired on a daemon thread so
    a slow AppleScript call never stalls the poll loop."""

    def __init__(self, sound: str = "Glass") -> None:
        self._sound = sound

    def notify(self, event: NotifyEvent) -> None:
        threading.Thread(target=self._send, args=(event,), daemon=True).start()

    def _send(self, event: NotifyEvent) -> None:
        script = (
            f"display notification {_osa_quote(event.detail())} "
            f"with title {_osa_quote(event.headline())} "
            f"sound name {_osa_quote(self._sound)}"
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                timeout=10, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001 — alerts are best-effort
            logger.warning("macOS notification failed: %s", exc)


_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Sends a message to a Telegram chat on a daemon thread. Token + chat id are
    injected (resolved from the environment by the factory), never stored in any
    config file."""

    def __init__(self, token: str, chat_id: str) -> None:
        self._token = token
        self._chat_id = chat_id

    def notify(self, event: NotifyEvent) -> None:
        threading.Thread(target=self._send, args=(event,), daemon=True).start()

    def _redact(self, s: str) -> str:
        """Scrub the bot token from any string before it is logged. The token
        sits in the request URL path (the Telegram API requires it there), so a
        requests connection error stringifies to '...url: /bot<TOKEN>/sendMessage
        ...' — logging that verbatim would leak a full-control credential."""
        return s.replace(self._token, "<token>") if self._token else s

    def _send(self, event: NotifyEvent) -> None:
        import requests  # noqa: PLC0415 (already a project dep; lazy keeps import cheap)
        try:
            resp = requests.post(
                _TELEGRAM_API.format(token=self._token),
                json={"chat_id": self._chat_id, "text": event.text()},
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Telegram notify failed: HTTP %s %s",
                    resp.status_code, self._redact(resp.text[:200]),
                )
        except Exception as exc:  # noqa: BLE001 — alerts are best-effort
            logger.warning("Telegram notify failed: %s", self._redact(str(exc)))


# ── Factory ──────────────────────────────────────────────────────────────────

_ALIASES = {
    "browser": "browser", "jupyter": "browser", "notebook": "browser",
    "desktop": "desktop", "macos": "desktop", "mac": "desktop",
    "telegram": "telegram", "tg": "telegram",
}


def make_notifiers(spec: str | Sequence[str] | None) -> list[Notifier]:
    """Build the notifier list from a comma-separated spec (or sequence) like
    ``"browser,desktop,telegram"``.

    An unknown channel name raises (a typo'd CLI flag should fail loudly); a
    known-but-unavailable channel (browser outside Jupyter, desktop off macOS,
    telegram with no token) is skipped with a warning, so the live loop runs
    with whatever alerts the environment can deliver instead of crashing.
    Duplicate channels collapse to one."""
    if not spec:
        return []
    if isinstance(spec, str):
        names = [p.strip().lower() for p in spec.split(",") if p.strip()]
    else:
        names = [str(p).strip().lower() for p in spec if str(p).strip()]

    notifiers: list[Notifier] = []
    seen: set[str] = set()
    for name in names:
        channel = _ALIASES.get(name)
        if channel is None:
            raise ValueError(
                f"Unknown notify channel '{name}'. Known: {sorted(set(_ALIASES))}"
            )
        if channel in seen:
            continue
        seen.add(channel)
        built = _build_channel(channel)
        if built is not None:
            notifiers.append(built)
    return notifiers


def _build_channel(channel: str) -> Optional[Notifier]:
    if channel == "browser":
        if not in_jupyter():
            logger.warning(
                "notify: 'browser' channel requested but not in a Jupyter kernel "
                "— skipping (use 'desktop' or 'telegram' from the CLI)."
            )
            return None
        return BrowserNotifier()
    if channel == "desktop":
        if sys.platform != "darwin":
            logger.warning(
                "notify: 'desktop' channel is macOS-only (sys.platform=%s) — skipping.",
                sys.platform,
            )
            return None
        return MacDesktopNotifier()
    if channel == "telegram":
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning(
                "notify: 'telegram' channel requested but TELEGRAM_BOT_TOKEN / "
                "TELEGRAM_CHAT_ID are not both set — skipping."
            )
            return None
        return TelegramNotifier(token, chat_id)
    return None  # unreachable: channel validated against _ALIASES above
