"""0DTE Signal Calibration Log — PIIP audit 2026-08, Option C.

The gap this closes: every 0DTE score (Trade Confidence, Entry Quality, Momentum, Timeframe
Alignment...) carries an honest "not backtested against real historical win-rate" caveat, but
nothing was actually LOGGING these numbers anywhere for a future backtest to run against. This is
the same move as iip/market_dna.py's log_snapshot() and iip/reddit_momentum.py's log_snapshot()
(see that module's own docstring: "log_snapshot() starts building that history for real, starting
now; backtesting becomes honest once weeks/months of it exist") — pure append-only collection, no
scoring/precision-recall dashboard here. That dashboard is a SEPARATE module once there's enough
logged data to make it meaningful rather than decorative (same explicit deferral MARKET_DNA_SPEC.md
already made for its own "Self-Validation Dashboard").

Follows the existing per-concern-module convention in this codebase (market_dna.py, watchlist.py,
predictions.py, portfolio.py, journal.py all own their own small SQLite table rather than sharing
one generic "logs" table) — same reasoning applies here.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime

DB_PATH = "iip.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS zero_dte_signal_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  bias_raw_signed REAL,
  bias_recommendation TEXT,
  trade_confidence_score REAL,
  entry_quality_score REAL,
  momentum_continuation_score_pct REAL,
  reversal_pressure_score REAL,
  alignment_agree INTEGER,
  alignment_total INTEGER,
  trend_state TEXT,
  extra TEXT              -- json, room for fields added later without a schema migration
);
"""


def _connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def log_signal_snapshot(ticker: str, bias: dict, confidence: dict, entry: dict,
                        momentum: dict | None, reversal: dict, alignment: dict,
                        trend_state: dict, path: str = DB_PATH) -> None:
    """Append-only, one row per refresh — same best-effort call shape as market_dna.log_snapshot()
    (caller wraps this in try/except so a local DB hiccup never breaks the page). Every field here
    is already computed elsewhere on the page; this only persists it."""
    now = datetime.now()
    with _connect(path) as con:
        con.execute(
            "INSERT INTO zero_dte_signal_log (ts, date, ticker, bias_raw_signed, "
            "bias_recommendation, trade_confidence_score, entry_quality_score, "
            "momentum_continuation_score_pct, reversal_pressure_score, alignment_agree, "
            "alignment_total, trend_state, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), date.today().isoformat(), ticker,
             bias.get("raw_signed"), bias.get("recommendation"), confidence.get("score"),
             entry.get("score"),
             momentum.get("continuation_score_pct") if momentum else None,
             reversal.get("reversal_pressure_score"), alignment.get("agree"),
             alignment.get("total"), trend_state.get("state"), json.dumps({})))


def collection_status(ticker: str | None = None, path: str = DB_PATH) -> dict:
    """Read-only status for the UI: how much has been collected so far, and since when. NOT a
    backtest or win-rate calculation — deliberately so, see module docstring. Returns zeroed
    fields (not an exception) if the table doesn't exist yet, e.g. before the first snapshot."""
    try:
        con = _connect(path)
        con.row_factory = sqlite3.Row
        where = "WHERE ticker=?" if ticker else ""
        params = (ticker,) if ticker else ()
        row = con.execute(
            f"SELECT COUNT(*) AS n, MIN(date) AS first_date, MAX(date) AS last_date, "
            f"COUNT(DISTINCT date) AS days FROM zero_dte_signal_log {where}", params).fetchone()
        con.close()
    except Exception:
        return {"rows": 0, "days": 0, "first_date": None, "last_date": None}
    return {"rows": row["n"] or 0, "days": row["days"] or 0,
            "first_date": row["first_date"], "last_date": row["last_date"]}
