"""Watchlist -- tickers you're tracking but don't (necessarily) own.

Deliberately thin: just persists which tickers you want to keep an eye on. Everything shown per
ticker (price, day change, next catalyst, Reddit momentum) reuses existing free lookups elsewhere
in this codebase (iip/data.py, iip/reddit_momentum.py) -- no new data source, no paid feed.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

DB_PATH = "iip.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT UNIQUE, added_ts TEXT, note TEXT
);
"""


def _con(db):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def add(db: str, ticker: str, note: str = "") -> bool:
    """Add a ticker (dedup by ticker, case-insensitive). Returns False if already on the list."""
    ticker = ticker.strip().upper()
    if not ticker:
        return False
    con = _con(db)
    con.executescript(SCHEMA)
    try:
        con.execute("INSERT INTO watchlist(ticker, added_ts, note) VALUES(?,?,?)",
                    (ticker, datetime.now(timezone.utc).isoformat(timespec="seconds"), note))
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        con.close()


def remove(db: str, ticker: str) -> None:
    con = _con(db)
    con.executescript(SCHEMA)
    con.execute("DELETE FROM watchlist WHERE ticker=?", (ticker.strip().upper(),))
    con.commit()
    con.close()


def list_tickers(db: str = DB_PATH) -> list[dict]:
    con = _con(db)
    con.executescript(SCHEMA)
    rows = con.execute("SELECT * FROM watchlist ORDER BY added_ts DESC").fetchall()
    con.close()
    return [dict(r) for r in rows]
