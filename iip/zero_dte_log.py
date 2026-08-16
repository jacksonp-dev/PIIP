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

import pandas as pd

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

# PIIP audit 2026-08, Batch 3 (Phase 9): edge-triggered log of actual regime-state CHANGES only
# (not one row per 30s refresh like zero_dte_signal_log above) -- a human-readable "when did the
# read change, and why" timeline. Separate table, same reasoning as every other per-concern table
# in this module: a different access pattern (read chronologically for one ticker/day) than the
# signal log's own use (join against later prices for forward-outcome stats).
TIMELINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS zero_dte_regime_timeline (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  from_state TEXT,
  to_state TEXT NOT NULL,
  reasons TEXT,            -- json list of strings, from day_regime()'s own reasons -- never
                            -- invented after the fact, only what was actually measured then
  trend_age_minutes REAL
);
"""


def _connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    con.execute(TIMELINE_SCHEMA)
    return con


def log_signal_snapshot(ticker: str, bias: dict, confidence: dict, entry: dict,
                        momentum: dict | None, reversal: dict, alignment: dict,
                        trend_state: dict, spot_price: float | None = None,
                        day_regime: str | None = None, path: str = DB_PATH) -> None:
    """Append-only, one row per refresh — same best-effort call shape as market_dna.log_snapshot()
    (caller wraps this in try/except so a local DB hiccup never breaks the page). Every field here
    is already computed elsewhere on the page; this only persists it.

    `spot_price`/`day_regime` (PIIP audit 2026-08, Batch 3): stored in the existing `extra` JSON
    column rather than a schema migration -- spot_price is the missing piece needed to later
    compute forward returns (see compute_forward_outcomes() below); day_regime lets historical
    stats be grouped by regime state, not just bias direction."""
    now = datetime.now()
    extra = {"spot_price": spot_price, "day_regime": day_regime}
    with _connect(path) as con:
        con.execute(
            "INSERT INTO zero_dte_signal_log (ts, date, ticker, bias_raw_signed, "
            "bias_recommendation, trade_confidence_score, entry_quality_score, "
            "momentum_continuation_score_pct, reversal_pressure_score, alignment_agree, "
            "alignment_total, trend_state, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.date().isoformat(), ticker,
             bias.get("raw_signed"), bias.get("recommendation"), confidence.get("score"),
             entry.get("score"),
             momentum.get("continuation_score_pct") if momentum else None,
             reversal.get("reversal_pressure_score"), alignment.get("agree"),
             alignment.get("total"), trend_state.get("state"), json.dumps(extra)))


def log_regime_transition(ticker: str, from_state: str | None, to_state: str, reasons: list[str],
                          trend_age_minutes: float | None, path: str = DB_PATH) -> None:
    """PIIP audit 2026-08, Batch 3 (Phase 9). Caller (app.py) is responsible for edge-triggering
    this -- only call when day_regime()'s state has actually changed since the last read, same
    pattern as the Exit Quality alert's was_alert/is_alert check. Never called on every refresh."""
    now = datetime.now()
    with _connect(path) as con:
        con.execute(
            "INSERT INTO zero_dte_regime_timeline (ts, date, ticker, from_state, to_state, "
            "reasons, trend_age_minutes) VALUES (?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.date().isoformat(), ticker, from_state,
             to_state, json.dumps(reasons), trend_age_minutes))


def regime_timeline(ticker: str, target_date: str | None = None, path: str = DB_PATH) -> list[dict]:
    """Chronological regime-state changes for one ticker/day -- defaults to today. Empty list
    (not an exception) if nothing's logged yet."""
    target_date = target_date or date.today().isoformat()
    try:
        con = _connect(path)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT ts, from_state, to_state, reasons, trend_age_minutes FROM "
            "zero_dte_regime_timeline WHERE ticker=? AND date=? ORDER BY ts ASC",
            (ticker, target_date)).fetchall()
        con.close()
    except Exception:
        return []
    return [{"ts": r["ts"], "from_state": r["from_state"], "to_state": r["to_state"],
             "reasons": json.loads(r["reasons"]) if r["reasons"] else [],
             "trend_age_minutes": r["trend_age_minutes"]} for r in rows]


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


def compute_forward_outcomes(ticker: str, path: str = DB_PATH) -> pd.DataFrame:
    """PIIP audit 2026-08, Batch 3 (Phase 10): the 'missing half' of calibration -- for every
    logged snapshot that has a recorded spot_price, find the NEXT logged snapshot(s) at each
    forward horizon (5/15/30/60 min, same trading day only -- never crosses a day boundary, that
    would silently include an overnight gap as if it were intraday movement) and compute the %
    price change. A horizon is left as None (not 0.0) when there aren't yet enough LATER rows to
    reach it -- e.g. a snapshot logged 10 minutes before today's close has no real fwd_60m yet.
    Pure read, does not run automatically on every 30s refresh -- caller (regime_stats() below, or
    a UI button) decides when this actually needs to run, per this project's own performance rule
    of separating live state from historical validation."""
    try:
        con = _connect(path)
        df = pd.read_sql_query(
            "SELECT ts, date, bias_recommendation, trade_confidence_score, extra FROM "
            "zero_dte_signal_log WHERE ticker=? ORDER BY ts ASC", con, params=(ticker,))
        con.close()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])

    def _extract(col):
        return df["extra"].apply(lambda e: (json.loads(e) if e else {}).get(col))
    df["spot_price"] = _extract("spot_price")
    df["day_regime"] = _extract("day_regime")
    df = df.dropna(subset=["spot_price"]).reset_index(drop=True)
    if df.empty:
        return df

    today_str = date.today().isoformat()
    horizons = (("fwd_5m", 5), ("fwd_15m", 15), ("fwd_30m", 30), ("fwd_60m", 60))
    out_rows = []
    for d, group in df.groupby("date"):
        group = group.sort_values("ts").reset_index(drop=True)
        is_complete_day = d < today_str
        last_price = float(group["spot_price"].iloc[-1])
        for _, row in group.iterrows():
            t0, p0 = row["ts"], float(row["spot_price"])
            result = {"date": d, "ts": row["ts"], "bias_recommendation": row["bias_recommendation"],
                      "trade_confidence_score": row["trade_confidence_score"],
                      "day_regime": row["day_regime"], "spot_price": p0}
            for label, minutes in horizons:
                future = group[group["ts"] >= t0 + pd.Timedelta(minutes=minutes)]
                result[label] = (round((float(future["spot_price"].iloc[0]) / p0 - 1) * 100, 4)
                                 if not future.empty else None)
            result["fwd_eod"] = round((last_price / p0 - 1) * 100, 4) if is_complete_day else None
            out_rows.append(result)
    return pd.DataFrame(out_rows)


def regime_stats(ticker: str, min_sample: int = 30, path: str = DB_PATH) -> dict:
    """PIIP audit 2026-08, Batch 3 (Phases 12-13): 'when PIIP sees this exact Day Regime state,
    what tends to happen next.' Groups compute_forward_outcomes() by the day_regime state that was
    active AT SIGNAL TIME, reporting real sample stats per forward horizon ONLY when the sample
    meets `min_sample` -- below that, returns INSUFFICIENT SAMPLE rather than a number, per this
    project's own standing rule (see zero_dte.py's Option A docstrings) that a heuristic doesn't
    get called a statistic until it's actually been validated. Expect this to show
    INSUFFICIENT SAMPLE almost everywhere for weeks/months -- collection only started 2026-08-15,
    that's correct, not a bug."""
    df = compute_forward_outcomes(ticker, path)
    if df.empty:
        return {"groups": {}, "total_snapshots": 0,
                "note": "No logged snapshots with a recorded price yet."}
    horizons = ["fwd_5m", "fwd_15m", "fwd_30m", "fwd_60m", "fwd_eod"]
    groups = {}
    for state, g in df.groupby("day_regime"):
        if not state:
            continue
        state_stats = {}
        for h in horizons:
            vals = g[h].dropna()
            n = len(vals)
            if n < min_sample:
                state_stats[h] = {"n": n, "status": "INSUFFICIENT SAMPLE", "min_needed": min_sample}
            else:
                state_stats[h] = {"n": n, "status": "OK",
                                  "positive_rate_pct": round((vals > 0).mean() * 100, 1),
                                  "avg_return_pct": round(vals.mean(), 3),
                                  "median_return_pct": round(vals.median(), 3),
                                  "worst_pct": round(vals.min(), 3), "best_pct": round(vals.max(), 3)}
        groups[state] = state_stats
    return {"groups": groups, "total_snapshots": len(df),
           "note": f"Grouped by Day Regime state at signal time. Any horizon below {min_sample} "
                  "samples shows INSUFFICIENT SAMPLE rather than a statistic."}
