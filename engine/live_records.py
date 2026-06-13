"""SQLite-backed live records: the open position and trade history.

Continuously saves the position + every completed trade to SQLite, 
and reloads them on startup.

Stores position state and trade history, so the bot can recover
from crashes without losing track of open positions.

"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .core import Direction, ExitReason, PositionState, PositionStatus, Trade

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS position_state (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    status      TEXT NOT NULL DEFAULT 'flat',
    direction   TEXT,
    entry_ts    TEXT,
    entry_price REAL,
    peak_price  REAL,
    stop_price  REAL,
    trade_id    TEXT,
    trade_counter INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_history (
    trade_id     TEXT PRIMARY KEY,
    direction    TEXT NOT NULL,
    entry_ts     TEXT NOT NULL,
    entry_price  REAL NOT NULL,
    exit_ts      TEXT,
    exit_price   REAL,
    pnl_bps      REAL,
    peak_price   REAL,
    exit_reason  TEXT,
    notional     REAL,
    pnl_currency REAL,
    equity_after REAL,
    created_at   TEXT NOT NULL
);

-- Single-row paper-equity ledger: the running account equity for the live
-- (forward-test) run. Persisted so the simulated equity curve is continuous
-- across restarts, mirroring the backtester's compounding equity layer.
CREATE TABLE IF NOT EXISTS account (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    equity     REAL NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class LiveRecords:
    """Persist the live position and trade history to SQLite."""

    def __init__(self, db_path: str | Path = "trading_state.db"):
        self._db_path = str(db_path)
        # Ensure the parent dir exists (e.g. data/live/ on first live run) — sqlite3
        # can't open/create the file if its directory is missing.
        parent = Path(self._db_path).parent
        if parent != Path(""):
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # Backward-compat: add columns to pre-existing tables created by older
        # schemas (CREATE TABLE IF NOT EXISTS won't alter an existing table).
        for table, column in (
            ("trade_history", "exit_reason TEXT"),
            ("position_state", "stop_price REAL"),
            ("trade_history", "notional REAL"),
            ("trade_history", "pnl_currency REAL"),
            ("trade_history", "equity_after REAL"),
        ):
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.commit()
        logger.info("Live records store initialised: %s", self._db_path)

    def close(self) -> None:
        self._conn.close()

    # ── Save / Load position ──────────────────────────────────────────────

    def save_state(self, state: PositionState) -> None:
        now = datetime.now(timezone.utc).isoformat()
        trade = state.current_trade

        self._conn.execute(
            """
            INSERT OR REPLACE INTO position_state
                (id, status, direction, entry_ts, entry_price, peak_price, stop_price, trade_id, trade_counter, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.status.name.lower(),
                trade.direction.value if trade else None,
                trade.entry_ts.isoformat() if trade and trade.entry_ts else None,
                trade.entry_price if trade else None,
                trade.peak_price if trade else None,
                trade.stop_price if trade else None,
                trade.trade_id if trade else None,
                state._trade_counter,
                now,
            ),
        )
        self._conn.commit()

    def load_state(self) -> PositionState:
        row = self._conn.execute(
            "SELECT status, direction, entry_ts, entry_price, peak_price, stop_price, trade_id, trade_counter FROM position_state WHERE id = 1"
        ).fetchone()

        state = PositionState()
        if row is None:
            return state

        status_str, direction_str, entry_ts_str, entry_price, peak_price, stop_price, trade_id, counter = row
        state._trade_counter = counter or 0

        if status_str == "open" and direction_str and entry_ts_str:
            state.status = PositionStatus.OPEN
            state.current_trade = Trade(
                trade_id=trade_id or "",
                direction=Direction(direction_str),
                entry_ts=datetime.fromisoformat(entry_ts_str),
                entry_price=entry_price or 0.0,
                peak_price=peak_price or 0.0,
                stop_price=stop_price,
            )
            logger.info(
                "Restored open %s position from %.2f (peak %.2f)",
                direction_str,
                entry_price,
                peak_price,
            )
        return state

    # ── Trade history ─────────────────────────────────────────────────────

    def save_trade(self, trade: Trade) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO trade_history
                (trade_id, direction, entry_ts, entry_price, exit_ts, exit_price, pnl_bps, peak_price,
                 exit_reason, notional, pnl_currency, equity_after, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.trade_id,
                trade.direction.value,
                trade.entry_ts.isoformat() if trade.entry_ts else "",
                trade.entry_price,
                trade.exit_ts.isoformat() if trade.exit_ts else None,
                trade.exit_price,
                trade.pnl_bps,
                trade.peak_price,
                trade.exit_reason.value if trade.exit_reason else None,
                trade.notional,
                trade.pnl_currency,
                trade.equity_after,
                now,
            ),
        )
        self._conn.commit()

    # ── Paper-equity ledger ───────────────────────────────────────────────

    def load_equity(self, default: float) -> float:
        """Running paper-equity for the live run, or ``default`` on first run."""
        row = self._conn.execute(
            "SELECT equity FROM account WHERE id = 1"
        ).fetchone()
        if row is None or row[0] is None:
            return default
        return float(row[0])

    def save_equity(self, equity: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO account (id, equity, updated_at) VALUES (1, ?, ?)",
            (equity, now),
        )
        self._conn.commit()

    def known_trade_ids(self) -> set[str]:
        """trade_ids already persisted — so the live engine sizes each closed
        trade into the equity curve exactly once, even across restarts."""
        rows = self._conn.execute("SELECT trade_id FROM trade_history").fetchall()
        return {r[0] for r in rows}

    # ── Load trade history ─────────────────────────────────────────────────────
    # A method to read past trades from SQLite.
    # Right now nothing in the project uses this method.
    # Can be wired later into reporting, if needed.
    def load_trade_history(self, limit: int = 200) -> list[Trade]:
        rows = self._conn.execute(
            "SELECT trade_id, direction, entry_ts, entry_price, exit_ts, exit_price, pnl_bps, peak_price, "
            "exit_reason, notional, pnl_currency, equity_after "
            "FROM trade_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        trades = []
        for r in rows:
            trades.append(
                Trade(
                    trade_id=r[0],
                    direction=Direction(r[1]),
                    entry_ts=datetime.fromisoformat(r[2]) if r[2] else None,
                    entry_price=r[3],
                    exit_ts=datetime.fromisoformat(r[4]) if r[4] else None,
                    exit_price=r[5] or 0.0,
                    pnl_bps=r[6] or 0.0,
                    peak_price=r[7] or 0.0,
                    exit_reason=ExitReason(r[8]) if r[8] else None,
                    notional=r[9] or 0.0,
                    pnl_currency=r[10] or 0.0,
                    equity_after=r[11] or 0.0,
                )
            )
        return trades
