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
  trend_age_minutes REAL,
  detail TEXT              -- json, PIIP audit 2026-08 (state-architecture review, Phase 5): the
                            -- point-in-time facts as of THIS transition (per-timeframe direction,
                            -- VWAP side/distance/crossings, trend integrity/efficiency, reversal
                            -- pressure, participation state) -- see zero_dte.explain_transition(),
                            -- which diffs this row's detail against the PRIOR row's detail to
                            -- build the "What Changed?" explanation at render time. Only measured
                            -- facts are stored here, never a generated explanation string -- the
                            -- explanation is always recomputed from facts, so it can't drift from
                            -- what was actually true at each point in time.
);
"""


def _migrate(con: sqlite3.Connection) -> None:
    """Add columns to an EXISTING zero_dte_regime_timeline table that predates them --
    CREATE TABLE IF NOT EXISTS alone only helps brand-new databases; a table created before this
    audit already exists without `detail` and needs an explicit ALTER TABLE. Best-effort: SQLite
    raises if the column already exists, which is the common case after the first run, so that
    specific error is swallowed; anything else re-raises."""
    cols = {row[1] for row in con.execute("PRAGMA table_info(zero_dte_regime_timeline)").fetchall()}
    if "detail" not in cols:
        try:
            con.execute("ALTER TABLE zero_dte_regime_timeline ADD COLUMN detail TEXT")
        except sqlite3.OperationalError:
            pass


def _connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    con.execute(TIMELINE_SCHEMA)
    _migrate(con)
    return con


def log_signal_snapshot(ticker: str, bias: dict, confidence: dict, entry: dict,
                        momentum: dict | None, reversal: dict, alignment: dict,
                        trend_state: dict, spot_price: float | None = None,
                        day_regime: str | None = None, day_type: str | None = None,
                        state_age_minutes: float | None = None, detail: dict | None = None,
                        path: str = DB_PATH) -> None:
    """Append-only, one row per refresh — same best-effort call shape as market_dna.log_snapshot()
    (caller wraps this in try/except so a local DB hiccup never breaks the page). Every field here
    is already computed elsewhere on the page; this only persists it.

    `spot_price`/`day_regime` (PIIP audit 2026-08, Batch 3): stored in the existing `extra` JSON
    column rather than a schema migration -- spot_price is the missing piece needed to later
    compute forward returns (see compute_forward_outcomes() below); day_regime lets historical
    stats be grouped by regime state, not just bias direction.

    `bias_recommendation` column (PIIP audit 2026-08, state-architecture review, Phase 2): NOT
    renamed, to avoid a schema migration for a column that's purely descriptive text either way --
    but the VALUES it stores changed from trade-directive strings ("CALLS ONLY"/"PUTS ONLY"/
    "CALLS FAVORED"/"PUTS FAVORED"/"NO CLEAR EDGE") to direction-only labels ("Strong Bullish"/
    "Strong Bearish"/"Bullish"/"Bearish"/"No Clear Edge"), since market_bias()'s own
    `direction_label` field (formerly `recommendation`) changed. Rows logged before this change
    will still contain the old values in this column -- anything reading this column historically
    (there is no such reader today; compute_forward_outcomes()/regime_stats() group by
    `day_regime`, not this field) should account for both vocabularies if one is ever added.

    `day_type`/`state_age_minutes`/`detail` (PIIP audit 2026-08, state-architecture review,
    follow-up per user request -- research-data design toward a future MFE/MAE-and-baselines
    analysis, per a ChatGPT-drafted spec reviewed against this codebase first): same `extra` JSON
    pattern as spot_price/day_regime above, all optional and keyword-compatible so existing
    callers/tests that don't pass them keep working unchanged. `day_type` is market_dna.classify()'s
    label (Trend Day / Chop / Grind / etc — a DIFFERENT axis than day_regime, see that module's
    docstring). `state_age_minutes` is how long the CURRENT day_regime state has held (already
    computed for day_regime()'s own Developing-vs-Confirmed tiering, just not persisted before).
    `detail` is the SAME build_regime_detail() bundle already stored on regime-transition rows
    (per-timeframe direction, VWAP side/distance/crossings, trend integrity/efficiency, reversal
    pressure, participation state) -- now captured on every routine snapshot too, not just at
    transitions, so a future stratified analysis (e.g. "4/4 alignment + Strong participation" as
    its own bucket) doesn't have to interpolate between sparse transition-only rows."""
    now = datetime.now()
    extra = {"spot_price": spot_price, "day_regime": day_regime, "day_type": day_type,
            "state_age_minutes": state_age_minutes, "detail": detail}
    with _connect(path) as con:
        con.execute(
            "INSERT INTO zero_dte_signal_log (ts, date, ticker, bias_raw_signed, "
            "bias_recommendation, trade_confidence_score, entry_quality_score, "
            "momentum_continuation_score_pct, reversal_pressure_score, alignment_agree, "
            "alignment_total, trend_state, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.date().isoformat(), ticker,
             bias.get("raw_signed"), bias.get("direction_label"), confidence.get("score"),
             entry.get("score"),
             momentum.get("continuation_score_pct") if momentum else None,
             reversal.get("reversal_pressure_score"), alignment.get("agree"),
             alignment.get("total"), trend_state.get("state"), json.dumps(extra)))


def log_regime_transition(ticker: str, from_state: str | None, to_state: str, reasons: list[str],
                          trend_age_minutes: float | None, detail: dict | None = None,
                          path: str = DB_PATH) -> None:
    """PIIP audit 2026-08, Batch 3 (Phase 9). Caller (app.py) is responsible for edge-triggering
    this -- only call when day_regime()'s state has actually changed since the last read, same
    pattern as the Exit Quality alert's was_alert/is_alert check. Never called on every refresh.

    `detail` (optional, PIIP audit 2026-08, state-architecture review, Phase 5): the point-in-time
    facts as of this transition (see TIMELINE_SCHEMA's own comment) -- stored as-is, never
    computed or interpreted here. Optional and keyword-compatible so existing callers/tests that
    don't pass it still work unchanged; `regime_timeline()`/`explain_transition()` degrade to no
    detailed explanation (just the existing from/to/reasons) for rows logged without it."""
    now = datetime.now()
    with _connect(path) as con:
        con.execute(
            "INSERT INTO zero_dte_regime_timeline (ts, date, ticker, from_state, to_state, "
            "reasons, trend_age_minutes, detail) VALUES (?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.date().isoformat(), ticker, from_state,
             to_state, json.dumps(reasons), trend_age_minutes,
             json.dumps(detail) if detail is not None else None))


def regime_timeline(ticker: str, target_date: str | None = None, path: str = DB_PATH) -> list[dict]:
    """Chronological regime-state changes for one ticker/day -- defaults to today. Empty list
    (not an exception) if nothing's logged yet. `detail` is None for rows logged before the
    Phase 5 schema addition -- callers must handle that, not assume it's always present."""
    target_date = target_date or date.today().isoformat()
    try:
        con = _connect(path)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT ts, from_state, to_state, reasons, trend_age_minutes, detail FROM "
            "zero_dte_regime_timeline WHERE ticker=? AND date=? ORDER BY ts ASC",
            (ticker, target_date)).fetchall()
        con.close()
    except Exception:
        return []
    return [{"ts": r["ts"], "from_state": r["from_state"], "to_state": r["to_state"],
             "reasons": json.loads(r["reasons"]) if r["reasons"] else [],
             "trend_age_minutes": r["trend_age_minutes"],
             "detail": json.loads(r["detail"]) if r["detail"] else None} for r in rows]


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
    of separating live state from historical validation.

    MFE/MAE columns (PIIP audit 2026-08, state-architecture review, follow-up per user request --
    research-data design toward a future outcome-quality analysis, spec reviewed against this
    codebase before building, per the user's own instruction not to duplicate existing forward-
    outcome machinery): `mfe_Xm` is the largest UPWARD move (%) reached at ANY point within the
    window (t0, resolving row], `mae_Xm` is the largest DOWNWARD move (a negative number, or 0).
    Deliberately UNSIGNED/not bias-direction-adjusted -- a bullish thesis reads mfe as its own
    favorable excursion and mae as its own adverse excursion; a bearish thesis reads them the
    other way around (mae is favorable, mfe is adverse). Keeping them raw here, rather than
    picking a convention inside this low-level function, keeps one computation reusable for both
    bull and bear stratified analysis later -- direction-aware interpretation belongs in the
    CONSUMING layer (same "raw primitives here, direction-aware logic in the caller" split this
    project already uses for breadth_score() vs confluence_score()). Precision caveat, not
    addressed by the original proposal: this can only be as granular as how often a row was
    actually logged (~30s while the page is open), NOT true tick-level highs/lows -- yfinance's
    free tier only keeps ~8 days of 1-minute bars, so unlike returns (which only need the ENDPOINT
    prices and stay exact forever), a durable MFE/MAE has no other source to fall back on. Real
    gaps in the window (page closed for a while) mean a genuine extreme could be missed, not
    fabricated -- same honesty tradeoff as everything else in this module, just worth naming
    explicitly since it's not obvious from the numbers alone.

    `day_type`/`state_age_minutes` are also pulled through (from the same widened `extra` JSON,
    see log_signal_snapshot()) so a future stratified analysis (by day type, or by how long the
    state had held) doesn't need a second query -- not consumed by regime_stats() yet, collection
    only, per the user's own 'design collection toward it, don't build the display yet' instruction."""
    try:
        con = _connect(path)
        df = pd.read_sql_query(
            "SELECT ts, date, bias_raw_signed, bias_recommendation, trade_confidence_score, extra "
            "FROM zero_dte_signal_log WHERE ticker=? ORDER BY ts ASC", con, params=(ticker,))
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
    df["day_type"] = _extract("day_type")
    df["state_age_minutes"] = _extract("state_age_minutes")
    df = df.dropna(subset=["spot_price"]).reset_index(drop=True)
    if df.empty:
        return df

    today_str = date.today().isoformat()
    horizons = (("fwd_5m", "mfe_5m", "mae_5m", 5), ("fwd_15m", "mfe_15m", "mae_15m", 15),
               ("fwd_30m", "mfe_30m", "mae_30m", 30), ("fwd_60m", "mfe_60m", "mae_60m", 60))
    out_rows = []
    for d, group in df.groupby("date"):
        group = group.sort_values("ts").reset_index(drop=True)
        is_complete_day = d < today_str
        last_price = float(group["spot_price"].iloc[-1])
        for _, row in group.iterrows():
            t0, p0 = row["ts"], float(row["spot_price"])
            result = {"date": d, "ts": row["ts"], "bias_raw_signed": row["bias_raw_signed"],
                      "bias_recommendation": row["bias_recommendation"],
                      "trade_confidence_score": row["trade_confidence_score"],
                      "day_regime": row["day_regime"], "day_type": row["day_type"],
                      "state_age_minutes": row["state_age_minutes"], "spot_price": p0}
            for fwd_label, mfe_label, mae_label, minutes in horizons:
                future = group[group["ts"] >= t0 + pd.Timedelta(minutes=minutes)]
                if future.empty:
                    result[fwd_label] = None
                    result[mfe_label] = None
                    result[mae_label] = None
                    continue
                result[fwd_label] = round((float(future["spot_price"].iloc[0]) / p0 - 1) * 100, 4)
                # Window = every row strictly after t0, up through the row that resolves this
                # horizon (inclusive) -- the same rows a live viewer would actually have seen.
                resolve_ts = future["ts"].iloc[0]
                window = group[(group["ts"] > t0) & (group["ts"] <= resolve_ts)]
                pct_changes = (window["spot_price"].astype(float) / p0 - 1) * 100
                # Floored/capped at 0 -- standard MFE/MAE convention: if price never rose above
                # entry, the best it did is "no gain" (0%), not the smallest actual excursion in
                # the window (which could itself be positive, e.g. a window that only ever saw
                # small gains would otherwise wrongly report a nonzero MAE despite never once
                # dipping below entry). Caught by a test asserting exactly this scenario.
                result[mfe_label] = round(max(0.0, float(pct_changes.max())), 4) if not pct_changes.empty else 0.0
                result[mae_label] = round(min(0.0, float(pct_changes.min())), 4) if not pct_changes.empty else 0.0
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
    that's correct, not a bug.

    `median_mfe_pct`/`median_mae_pct` (PIIP audit 2026-08, state-architecture review follow-up):
    added alongside the existing return stats for every horizon that has one (not fwd_eod, which
    isn't a fixed-window horizon the same way) -- median rather than avg/worst/best to match what
    the user's own reviewed spec asked the headline table to show, and because MFE/MAE are more
    outlier-prone than returns (a single wide-ranging session can dominate a mean). This is data
    only -- no UI currently renders these fields; see compute_forward_outcomes()'s own docstring
    for the unsigned-excursion convention and the ~30s-logging-interval precision caveat."""
    df = compute_forward_outcomes(ticker, path)
    if df.empty:
        return {"groups": {}, "total_snapshots": 0,
                "note": "No logged snapshots with a recorded price yet."}
    horizons = [("fwd_5m", "mfe_5m", "mae_5m"), ("fwd_15m", "mfe_15m", "mae_15m"),
               ("fwd_30m", "mfe_30m", "mae_30m"), ("fwd_60m", "mfe_60m", "mae_60m"),
               ("fwd_eod", None, None)]
    groups = {}
    for state, g in df.groupby("day_regime"):
        if not state:
            continue
        state_stats = {}
        for h, mfe_col, mae_col in horizons:
            vals = g[h].dropna()
            n = len(vals)
            if n < min_sample:
                state_stats[h] = {"n": n, "status": "INSUFFICIENT SAMPLE", "min_needed": min_sample}
            else:
                entry = {"n": n, "status": "OK",
                        "positive_rate_pct": round((vals > 0).mean() * 100, 1),
                        "avg_return_pct": round(vals.mean(), 3),
                        "median_return_pct": round(vals.median(), 3),
                        "worst_pct": round(vals.min(), 3), "best_pct": round(vals.max(), 3)}
                if mfe_col is not None:
                    mfe_vals, mae_vals = g[mfe_col].dropna(), g[mae_col].dropna()
                    if len(mfe_vals):
                        entry["median_mfe_pct"] = round(mfe_vals.median(), 3)
                    if len(mae_vals):
                        entry["median_mae_pct"] = round(mae_vals.median(), 3)
                state_stats[h] = entry
        groups[state] = state_stats
    return {"groups": groups, "total_snapshots": len(df),
           "note": f"Grouped by Day Regime state at signal time. Any horizon below {min_sample} "
                  "samples shows INSUFFICIENT SAMPLE rather than a statistic."}
