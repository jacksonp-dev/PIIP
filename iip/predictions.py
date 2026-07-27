"""Prediction store (SQLite) — the scorecard. Every prediction is a pre-registered experiment.

Each row is logged the moment a forecast is made and resolved later against reality. We store the
deterministic AND llm predictions separately (source column) so the LLM can be judged vs the
baseline. Structured reasoning is stored so failures are diagnosable, not just countable.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

DB_PATH = "iip.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  resolve_due TEXT NOT NULL,
  ticker TEXT NOT NULL,
  horizon_days INTEGER NOT NULL,
  source TEXT NOT NULL,                      -- 'deterministic' | 'llm'
  spot REAL,
  -- stock thesis
  stock_direction TEXT,
  stock_expected_move_pct REAL,
  stock_prob_up REAL,
  confidence REAL,
  -- simulated option leg
  option_type TEXT,
  option_strike REAL,
  option_expiry TEXT,
  option_entry_premium REAL,
  option_iv_pct REAL,
  option_greeks TEXT,                        -- json
  -- reasoning + cost
  reasoning TEXT,                            -- json: bull/bear/uncertainty/sources
  cost_usd REAL DEFAULT 0,
  -- resolution (filled later)
  resolved INTEGER DEFAULT 0,
  resolve_ts TEXT,
  realized_return_pct REAL,
  realized_vol_pct REAL,
  spy_return_pct REAL,
  underlying_return_pct REAL,
  option_exit_premium REAL,
  option_pnl_pct REAL,
  hit INTEGER                                -- 1 if stock_direction was correct
);
"""


# Every real column this table has, excluding the autoincrement PK (never set programmatically).
# log_prediction()/resolve() build their SQL column list from a dict's keys -- both dicts are
# built from hardcoded literal keys today, so this isn't currently exploitable, but trusting
# dict.keys() directly is fragile: if a future caller ever passed a dict built from user input,
# nothing would stop an arbitrary column name from being spliced into the query. Whitelisting
# closes that door regardless of what future callers do.
_COLUMNS = {
    "ts", "resolve_due", "ticker", "horizon_days", "source", "spot",
    "stock_direction", "stock_expected_move_pct", "stock_prob_up", "confidence",
    "option_type", "option_strike", "option_expiry", "option_entry_premium",
    "option_iv_pct", "option_greeks", "reasoning", "cost_usd",
    "resolved", "resolve_ts", "realized_return_pct", "realized_vol_pct",
    "spy_return_pct", "underlying_return_pct", "option_exit_premium",
    "option_pnl_pct", "hit",
}


def _check_columns(keys) -> None:
    bad = set(keys) - _COLUMNS
    if bad:
        raise ValueError(f"Unknown prediction column(s): {sorted(bad)}")


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def init_db(path: str = DB_PATH) -> None:
    con = connect(path)
    con.executescript(SCHEMA)
    con.commit()
    con.close()


def log_prediction(rec: dict, path: str = DB_PATH) -> int:
    """Insert one prediction. `rec` uses the field names above; dict/list fields are JSON-encoded.
    resolve_due = ts + horizon_days (when it becomes scoreable)."""
    ts = rec.get("ts") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    due = (datetime.fromisoformat(ts) + timedelta(days=int(rec["horizon_days"]))).isoformat(timespec="seconds")
    row = {
        "ts": ts, "resolve_due": due, "ticker": rec["ticker"],
        "horizon_days": int(rec["horizon_days"]), "source": rec["source"], "spot": rec.get("spot"),
        "stock_direction": rec.get("stock_direction"),
        "stock_expected_move_pct": rec.get("stock_expected_move_pct"),
        "stock_prob_up": rec.get("stock_prob_up"), "confidence": rec.get("confidence"),
        "option_type": rec.get("option_type"), "option_strike": rec.get("option_strike"),
        "option_expiry": rec.get("option_expiry"), "option_entry_premium": rec.get("option_entry_premium"),
        "option_iv_pct": rec.get("option_iv_pct"),
        "option_greeks": json.dumps(rec.get("option_greeks")) if rec.get("option_greeks") is not None else None,
        "reasoning": json.dumps(rec.get("reasoning")) if rec.get("reasoning") is not None else None,
        "cost_usd": rec.get("cost_usd", 0.0),
    }
    _check_columns(row.keys())
    cols = ", ".join(row.keys())
    ph = ", ".join("?" for _ in row)
    init_db(path)
    con = connect(path)
    cur = con.execute(f"INSERT INTO predictions ({cols}) VALUES ({ph})", list(row.values()))
    con.commit()
    pid = cur.lastrowid
    con.close()
    return pid


def log_if_new(rec: dict, path: str = DB_PATH) -> int | None:
    """Log only if no same ticker+horizon+source prediction exists for today (UTC) — keeps the
    Scorecard from filling with duplicates when you re-scan or re-view the same ticker in a day."""
    init_db(path)
    today = datetime.now(timezone.utc).date().isoformat()
    con = connect(path)
    exists = con.execute(
        "SELECT 1 FROM predictions WHERE ticker=? AND horizon_days=? AND source=? "
        "AND substr(ts,1,10)=? LIMIT 1",
        (rec["ticker"], int(rec["horizon_days"]), rec.get("source", "deterministic"), today)
    ).fetchone()
    con.close()
    if exists:
        return None
    return log_prediction(rec, path)


def get_due(path: str = DB_PATH, asof: datetime | None = None) -> list[sqlite3.Row]:
    """Unresolved predictions whose horizon has elapsed (ready to score)."""
    asof = (asof or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    init_db(path)
    con = connect(path)
    rows = con.execute(
        "SELECT * FROM predictions WHERE resolved=0 AND resolve_due<=? ORDER BY resolve_due", (asof,)
    ).fetchall()
    con.close()
    return rows


def resolve(pid: int, outcome: dict, path: str = DB_PATH) -> None:
    outcome = {**outcome, "resolved": 1,
               "resolve_ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _check_columns(outcome.keys())
    sets = ", ".join(f"{k}=?" for k in outcome)
    init_db(path)
    con = connect(path)
    con.execute(f"UPDATE predictions SET {sets} WHERE id=?", list(outcome.values()) + [pid])
    con.commit()
    con.close()


def all_rows(path: str = DB_PATH, resolved_only: bool = False) -> list[sqlite3.Row]:
    init_db(path)
    con = connect(path)
    q = "SELECT * FROM predictions"
    if resolved_only:
        q += " WHERE resolved=1"
    rows = con.execute(q + " ORDER BY ts").fetchall()
    con.close()
    return rows


def graded_since(since_ts: str, path: str = DB_PATH) -> list[sqlite3.Row]:
    """Predictions resolved (graded) at/after `since_ts` (an ISO timestamp) -- powers the "since
    your last visit" digest without re-deriving grading state, just filtering resolve_ts."""
    init_db(path)
    con = connect(path)
    rows = con.execute(
        "SELECT * FROM predictions WHERE resolved=1 AND resolve_ts >= ? ORDER BY resolve_ts",
        (since_ts,)).fetchall()
    con.close()
    return rows