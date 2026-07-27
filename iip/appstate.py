"""Tiny generic key-value store for cross-session app state (e.g. "when did the user last open
this app"). SQLite-backed so it survives real time passing between visits, unlike
st.session_state, which resets every browser session and can't answer "since yesterday."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

DB_PATH = "iip.db"

SCHEMA = "CREATE TABLE IF NOT EXISTS app_state (key TEXT PRIMARY KEY, value TEXT);"


def _con(db):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def get(db: str, key: str) -> str | None:
    con = _con(db)
    con.executescript(SCHEMA)
    row = con.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    con.close()
    return row["value"] if row else None


def set(db: str, key: str, value: str) -> None:
    con = _con(db)
    con.executescript(SCHEMA)
    con.execute("INSERT INTO app_state(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    con.commit()
    con.close()


def get_and_bump_last_visit(db: str) -> str | None:
    """Returns the PREVIOUS last-visit timestamp (None on the very first-ever visit), then
    immediately stamps "now" as the new one. Call once per browser SESSION (gate with
    st.session_state), not once per rerun -- otherwise "since last visit" would always read back
    as "a few seconds ago" instead of spanning actual time away from the app."""
    prev = get(db, "last_visit")
    set(db, "last_visit", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return prev
