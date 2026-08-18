"""Market DNA — classifies today's SPY/QQQ/IWM/DIA session into a day-type label. See
MARKET_DNA_SPEC.md for the full spec and rule rationale.

Pure functions, no network calls — consumes the same intraday/daily frames and per-ticker
snapshot already fetched/computed by zero_dte.py's batched download (never fetch here).
Thresholds are a first-pass guess (spec open question #1), kept in one config dict so they're
easy to recalibrate once real sessions have been logged, rather than tuned/hardcoded blind.

RELATIONSHIP TO zero_dte.day_regime() (PIIP audit 2026-08, state-architecture review):
These are two DELIBERATELY separate engines answering two different questions, not duplicates
to be merged:
  * Market DNA (this module) answers "WHAT KIND OF DAY IS THIS, SHAPE-WISE?" — Gap & Go, Trend
    Day, Opening Reversal, High Volatility Chop, Slow Grind, Range Bound. It looks at the WHOLE
    session so far (gap%, range-vs-ATR, VWAP-side consistency for the full day) and answers a
    question that's stable once set for the day (a "Trend Day" doesn't stop being a Trend Day
    because of one choppy 15-minute stretch).
  * zero_dte.day_regime() answers "WHAT DIRECTIONAL STATE IS THE MARKET IN RIGHT NOW?" — BULL/
    BEAR DEVELOPING/CONFIRMED, NEUTRAL/CHOP, TREND WEAKENING, REGIME TRANSITION. It's a live,
    trend-following read (built from timeframe.update_trend_state()'s hysteresis machine) that
    can and does change several times within a single Market DNA day-type — e.g. a genuine
    "Trend Day" can pass through BULL DEVELOPING -> BULL CONFIRMED -> TREND WEAKENING -> BULL
    CONFIRMED again as the intraday trend strengthens/pulls back/resumes, without the day-type
    label changing at all.
  * Both are shown together in the UI (Trade Confidence hero shows Market DNA; the Day Regime
    banner sits above it) with a caption cross-referencing them (see app.py's "How the trend/
    direction reads relate" caption) — this is the intended final architecture, not an
    intermediate state waiting to be unified into a third classifier.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import deterministic as det

DB_PATH = "iip.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_dna_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  label TEXT NOT NULL,
  metrics TEXT NOT NULL          -- json
);
"""

MIN_SESSION_MINUTES = 30  # below this, always "Insufficient Evidence" — don't guess early

THRESHOLDS = {
    "gap_pct_significant": 0.3,       # % gap considered "significant"
    "trend_net_vs_range": 0.65,
    "trend_vwap_consistency": 0.8,
    "chop_range_vs_atr": 1.3,
    "chop_net_vs_range": 0.35,
    "grind_net_vs_range": 0.55,
    "grind_range_vs_atr": 0.8,
    "gap_go_net_vs_range": 0.6,
}


def _yesterday_close(daily: pd.DataFrame) -> float | None:
    prior = daily[daily.index.date < date.today()]
    if prior.empty:
        return None
    return float(prior["Close"].iloc[-1])


def _rolling_vwap_side_consistency(intraday: pd.DataFrame) -> float | None:
    """Fraction of today's bars whose close sits on the same side of the RUNNING (cumulative,
    as-of-that-bar) VWAP as the final bar's close. A trend day stays on one side all session;
    chop crosses back and forth repeatedly."""
    c = intraday["Close"]
    running_vwap = det.running_vwap(intraday)
    sides = np.sign(c - running_vwap).dropna()
    if sides.empty:
        return None
    final_side = sides.iloc[-1]
    if final_side == 0:
        return None
    return float((sides == final_side).mean())


def _gap_held(intraday: pd.DataFrame, yesterday_close: float, gap_pct: float) -> bool:
    """True if price never traded back through yesterday's close after the first bar (a
    filled/reversed gap crosses back through it)."""
    if len(intraday) < 2:
        return True
    after_open = intraday.iloc[1:]
    if gap_pct > 0:
        return bool((after_open["Low"] > yesterday_close).all())
    return bool((after_open["High"] < yesterday_close).all())


def classify(snap: dict | None, intraday: pd.DataFrame | None, daily: pd.DataFrame | None,
            now: datetime | None = None) -> dict:
    """Returns {label, note, metrics, as_of}. `snap` is zero_dte.ticker_snapshot()'s output
    (already includes `tech` from deterministic.technical_metrics — not recomputed here)."""
    now = now or datetime.now()
    as_of = now.isoformat(timespec="seconds")

    if snap is None or intraday is None or intraday.empty or daily is None or daily.empty:
        return {"label": "Insufficient Evidence", "note": "No intraday/daily data available.",
                "metrics": {}, "as_of": as_of}

    session_minutes = (intraday.index[-1] - intraday.index[0]).total_seconds() / 60.0
    tech = snap["tech"]
    yesterday_close = _yesterday_close(daily)
    atr14 = tech.get("atr14")
    today_range = snap["high"] - snap["low"]

    gap_pct = ((snap["open"] - yesterday_close) / yesterday_close * 100) if yesterday_close else None
    range_vs_atr = (today_range / atr14) if atr14 else None
    net_vs_range = (abs(snap["last"] - snap["open"]) / today_range) if today_range > 0 else None
    vwap_consistency = _rolling_vwap_side_consistency(intraday)
    gap_held = (_gap_held(intraday, yesterday_close, gap_pct)
               if (yesterday_close and gap_pct is not None) else None)

    metrics = {
        "session_minutes": round(session_minutes, 0),
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "gap_held": gap_held,
        "range_vs_atr": round(range_vs_atr, 2) if range_vs_atr is not None else None,
        "net_vs_range": round(net_vs_range, 2) if net_vs_range is not None else None,
        "vwap_side_consistency": round(vwap_consistency, 2) if vwap_consistency is not None else None,
        "day_change_pct": round(snap["day_change_pct"], 2),
    }

    if session_minutes < MIN_SESSION_MINUTES:
        return {"label": "Insufficient Evidence",
                "note": "Still forming — under 30 minutes of session data.",
                "metrics": metrics, "as_of": as_of}

    T = THRESHOLDS
    label = "Range Bound"
    note = "No clear gap, trend, or chop signature — describing as range-bound by default."

    if gap_pct is not None and abs(gap_pct) > T["gap_pct_significant"]:
        if gap_held and net_vs_range is not None and net_vs_range > T["gap_go_net_vs_range"]:
            label = "Gap & Go"
            note = f"Opened with a {gap_pct:+.2f}% gap that has held and continued."
        elif gap_held is False:
            label = "Opening Reversal"
            note = f"Opened with a {gap_pct:+.2f}% gap that has since been filled/reversed."

    if label == "Range Bound" and net_vs_range is not None and vwap_consistency is not None:
        if net_vs_range > T["trend_net_vs_range"] and vwap_consistency > T["trend_vwap_consistency"]:
            label = "Trend Day"
            note = (f"Price has stayed on one side of VWAP {vwap_consistency*100:.0f}% of the "
                    f"session and is near its {'high' if snap['day_change_pct'] > 0 else 'low'}.")
        elif (range_vs_atr is not None and range_vs_atr > T["chop_range_vs_atr"]
              and net_vs_range < T["chop_net_vs_range"]):
            label = "High Volatility Chop"
            note = (f"Range is {range_vs_atr:.1f}x the 14-day ATR with little net progress — "
                    f"wide two-way action.")
        elif net_vs_range > T["grind_net_vs_range"] and (range_vs_atr or 0) < T["grind_range_vs_atr"]:
            direction = "Higher" if snap["day_change_pct"] > 0 else "Lower"
            label = f"Slow Grind {direction}"
            note = "Steady single-direction drift without the range expansion of a full trend day."

    return {"label": label, "note": note, "metrics": metrics, "as_of": as_of}


def _connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def log_snapshot(ticker: str, result: dict, path: str = DB_PATH) -> None:
    """Append-only log — one row per refresh. Cheap (SQLite, local), lets the day's evolving
    read be reviewed later; the eventual Self-Validation Dashboard (separate module, not built
    yet) reads from this table rather than a new one."""
    import json
    with _connect(path) as con:
        con.execute(
            "INSERT INTO market_dna_log (ts, date, ticker, label, metrics) VALUES (?, ?, ?, ?, ?)",
            (result["as_of"], date.today().isoformat(), ticker, result["label"],
             json.dumps(result["metrics"])),
        )
