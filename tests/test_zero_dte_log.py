"""Tests for iip/zero_dte_log.py — forward-outcome computation, same-day-only 0DTE horizons
(the project's central data-leakage guard), the regime-timeline schema migration, and
log_regime_transition()/regime_timeline() round-tripping. PIIP audit 2026-08, state-architecture
review, Phase 8.

All tests use a THROWAWAY sqlite file in the OS temp dir, created fresh and removed at the end of
each test — never the real iip.db, per this project's own established testing discipline.
"""
from __future__ import annotations

import gc
import json
import os
import sqlite3
import sys
import tempfile
import time
import uuid
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from iip import zero_dte_log as zdlog


@pytest.fixture
def tmp_db():
    path = os.path.join(tempfile.gettempdir(), f"test_piip_zdlog_{uuid.uuid4().hex}.db")
    yield path
    # zero_dte_log's `with _connect(path) as con:` pattern (pre-existing throughout this module,
    # not introduced by this test suite) commits/rolls back on exit but does NOT close the
    # connection -- a known Python sqlite3 gotcha. On Windows that can briefly hold a file lock
    # after the test body returns; retry teardown rather than let a GC-timing artifact fail an
    # otherwise-passing test.
    gc.collect()
    for attempt in range(5):
        try:
            if os.path.exists(path):
                os.remove(path)
            break
        except PermissionError:
            time.sleep(0.1)


def _insert_signal_row(path: str, ticker: str, ts: datetime, spot_price: float,
                       day_regime: str = "BULL CONFIRMED") -> None:
    """Insert a zero_dte_signal_log row with a FULLY CONTROLLED timestamp -- the public
    log_signal_snapshot() always uses datetime.now(), which isn't controllable enough to test
    time-boundary logic (5/15/30/60min horizons, same-day-only grouping) precisely. Writes
    directly via the same schema zero_dte_log.py itself defines."""
    con = zdlog._connect(path)
    extra = json.dumps({"spot_price": spot_price, "day_regime": day_regime})
    con.execute(
        "INSERT INTO zero_dte_signal_log (ts, date, ticker, bias_raw_signed, bias_recommendation, "
        "trade_confidence_score, entry_quality_score, momentum_continuation_score_pct, "
        "reversal_pressure_score, alignment_agree, alignment_total, trend_state, extra) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ts.isoformat(timespec="seconds"), ts.date().isoformat(), ticker, 40.0, "Bullish",
         70.0, 60.0, 55.0, 20.0, 3, 4, "Uptrend", extra))
    con.commit()
    con.close()


# ---------- compute_forward_outcomes: correctness (test area 7) ----------

def test_forward_outcome_basic_5m_horizon(tmp_db):
    base = datetime(2020, 1, 2, 10, 0, 0)   # a real past Thursday -- date < today, "complete day"
    _insert_signal_row(tmp_db, "TEST", base, spot_price=100.0)
    _insert_signal_row(tmp_db, "TEST", base + timedelta(minutes=5), spot_price=101.0)
    df = zdlog.compute_forward_outcomes("TEST", path=tmp_db)
    assert len(df) == 2
    first = df.iloc[0]
    assert first["fwd_5m"] == pytest.approx(1.0, abs=0.001)   # (101/100 - 1) * 100


def test_forward_outcome_none_when_horizon_not_yet_reached(tmp_db):
    """A horizon with no later row available yet must be None, never 0.0 or an extrapolated
    guess."""
    base = datetime(2020, 1, 2, 15, 55, 0)   # near the end of the trading day, nothing after it
    _insert_signal_row(tmp_db, "TEST", base, spot_price=100.0)
    df = zdlog.compute_forward_outcomes("TEST", path=tmp_db)
    row = df.iloc[0]
    assert row["fwd_5m"] is None
    assert row["fwd_15m"] is None
    assert row["fwd_30m"] is None
    assert row["fwd_60m"] is None


# ---------- SAME-DAY-ONLY horizons: the central leakage guard (test area 8) ----------

def test_forward_outcome_never_crosses_a_day_boundary(tmp_db):
    """THE core leakage regression test the audit called for: a snapshot near yesterday's close
    must NOT have tomorrow's opening price used as its '5-minute-later' outcome just because the
    raw wall-clock gap happens to satisfy `ts >= t0 + 5min`. Overnight gaps are not intraday
    movement and must never be counted as such."""
    day1_late = datetime(2020, 1, 2, 15, 58, 0)     # last snapshot of day 1
    day2_open = datetime(2020, 1, 3, 9, 30, 0)       # first snapshot of day 2, hours later
    _insert_signal_row(tmp_db, "TEST", day1_late, spot_price=100.0)
    _insert_signal_row(tmp_db, "TEST", day2_open, spot_price=150.0)  # deliberately a huge jump

    df = zdlog.compute_forward_outcomes("TEST", path=tmp_db)
    day1_row = df[df["date"] == "2020-01-02"].iloc[0]
    # Even though day2_open's ts is (in raw minutes) >> 5/15/30/60 minutes after day1_late, it
    # must NEVER be picked up as day1_late's forward outcome -- there is no later SAME-DAY row.
    assert day1_row["fwd_5m"] is None
    assert day1_row["fwd_15m"] is None
    assert day1_row["fwd_30m"] is None
    assert day1_row["fwd_60m"] is None
    # fwd_eod for day1 (a complete/past day) must reflect day 1's OWN last price, not day 2's.
    assert day1_row["fwd_eod"] == pytest.approx(0.0, abs=0.001)   # only one row that day: itself


def test_forward_outcome_fwd_eod_only_for_complete_days(tmp_db):
    """fwd_eod must be None for TODAY (still in progress -- the close hasn't happened yet) even
    if a later snapshot already exists earlier the same session, and a real number for a day
    that's unambiguously in the past."""
    past_base = datetime(2020, 1, 2, 10, 0, 0)
    _insert_signal_row(tmp_db, "TEST", past_base, spot_price=100.0)
    _insert_signal_row(tmp_db, "TEST", past_base + timedelta(hours=5), spot_price=105.0)

    today_base = datetime.combine(date.today(), datetime.min.time()).replace(hour=10)
    _insert_signal_row(tmp_db, "TEST", today_base, spot_price=200.0)
    _insert_signal_row(tmp_db, "TEST", today_base + timedelta(minutes=30), spot_price=202.0)

    df = zdlog.compute_forward_outcomes("TEST", path=tmp_db)
    past_row = df[df["date"] == "2020-01-02"].iloc[0]
    today_row = df[df["date"] == date.today().isoformat()].iloc[0]
    assert past_row["fwd_eod"] is not None
    assert pd.isna(today_row["fwd_eod"]), "today is not a complete day yet -- must not fabricate an EOD return"


def test_forward_outcome_ignores_rows_without_spot_price(tmp_db):
    """A row logged without a spot_price (e.g. stale-data gate skipped the write, or an older row
    predates spot_price collection) must be dropped, never treated as spot_price=0."""
    con = zdlog._connect(tmp_db)
    con.execute(
        "INSERT INTO zero_dte_signal_log (ts, date, ticker, extra) VALUES (?,?,?,?)",
        ("2020-01-02T10:00:00", "2020-01-02", "TEST", json.dumps({"spot_price": None})))
    con.commit()
    con.close()
    df = zdlog.compute_forward_outcomes("TEST", path=tmp_db)
    assert df.empty


# ---------- regime_stats: INSUFFICIENT SAMPLE gating ----------

def test_regime_stats_insufficient_sample_below_threshold(tmp_db):
    base = datetime(2020, 1, 2, 10, 0, 0)
    _insert_signal_row(tmp_db, "TEST", base, spot_price=100.0, day_regime="BULL CONFIRMED")
    _insert_signal_row(tmp_db, "TEST", base + timedelta(minutes=5), spot_price=101.0,
                       day_regime="BULL CONFIRMED")
    stats = zdlog.regime_stats("TEST", min_sample=30, path=tmp_db)
    bull_stats = stats["groups"].get("BULL CONFIRMED", {})
    if bull_stats:
        assert bull_stats["fwd_5m"]["status"] == "INSUFFICIENT SAMPLE"


def test_regime_stats_no_data(tmp_db):
    stats = zdlog.regime_stats("NOTHING_LOGGED", path=tmp_db)
    assert stats["groups"] == {}
    assert stats["total_snapshots"] == 0


# ---------- regime_timeline schema migration + round-trip (test area 10, part 2) ----------

def test_timeline_migration_adds_detail_column_to_old_table(tmp_db):
    """Simulates a database created BEFORE the Phase 5 `detail` column existed -- must migrate
    cleanly via ALTER TABLE, not crash, and old rows must read back with detail=None."""
    con = sqlite3.connect(tmp_db)
    con.execute("""
        CREATE TABLE zero_dte_regime_timeline (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, date TEXT NOT NULL,
          ticker TEXT NOT NULL, from_state TEXT, to_state TEXT NOT NULL, reasons TEXT,
          trend_age_minutes REAL
        )""")
    con.execute("INSERT INTO zero_dte_regime_timeline (ts, date, ticker, from_state, to_state, "
               "reasons, trend_age_minutes) VALUES (?,?,?,?,?,?,?)",
               ("2020-01-02T10:00:00", "2020-01-02", "TEST", "NEUTRAL", "BULL DEVELOPING",
                json.dumps(["test reason"]), 5.0))
    con.commit()
    con.close()

    # Round-trip through the real module -- must not raise, must migrate the column in place.
    rows = zdlog.regime_timeline("TEST", target_date="2020-01-02", path=tmp_db)
    assert len(rows) == 1
    assert rows[0]["detail"] is None
    assert rows[0]["to_state"] == "BULL DEVELOPING"

    # New writes after migration must persist detail correctly.
    zdlog.log_regime_transition("TEST", "BULL DEVELOPING", "BULL CONFIRMED", ["reason"], 10.0,
                                detail={"5m": "Bullish"}, path=tmp_db)
    rows2 = zdlog.regime_timeline("TEST", target_date=date.today().isoformat(), path=tmp_db)
    assert len(rows2) == 1
    assert rows2[0]["detail"] == {"5m": "Bullish"}


def test_log_regime_transition_without_detail_still_works(tmp_db):
    """Optional/keyword-compatible: existing call shape (no `detail`) must still work unchanged."""
    zdlog.log_regime_transition("TEST", None, "BULL DEVELOPING", ["first read"], 0.0, path=tmp_db)
    rows = zdlog.regime_timeline("TEST", target_date=date.today().isoformat(), path=tmp_db)
    assert len(rows) == 1
    assert rows[0]["detail"] is None
    assert rows[0]["reasons"] == ["first read"]


# ---------- log_signal_snapshot: widened extra JSON (per user request, follow-up) ----------

def test_log_signal_snapshot_persists_day_type_state_age_detail(tmp_db):
    """The public API (not the raw-insert test helper) must round-trip the new optional fields
    through the real function, since that's what app.py actually calls."""
    bias = {"raw_signed": 42.0, "direction_label": "Bullish"}
    confidence = {"score": 71.0}
    entry = {"score": 65.0}
    momentum = {"continuation_score_pct": 58.0}
    reversal = {"reversal_pressure_score": 22.0}
    alignment = {"agree": 3, "total": 4}
    trend_state = {"state": "Uptrend"}
    detail = {"5m": "Bullish", "trend_integrity": 72.0, "participation_state": "STRONG"}

    zdlog.log_signal_snapshot("TEST", bias, confidence, entry, momentum, reversal, alignment,
                              trend_state, spot_price=100.0, day_regime="BULL CONFIRMED",
                              day_type="Trend Day", state_age_minutes=27.5, detail=detail,
                              path=tmp_db)

    con = zdlog._connect(tmp_db)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT extra FROM zero_dte_signal_log WHERE ticker='TEST'").fetchone()
    con.close()
    extra = json.loads(row["extra"])
    assert extra["day_type"] == "Trend Day"
    assert extra["state_age_minutes"] == 27.5
    assert extra["detail"] == detail
    assert extra["spot_price"] == 100.0   # existing fields still work unchanged
    assert extra["day_regime"] == "BULL CONFIRMED"


def test_log_signal_snapshot_without_new_fields_still_works(tmp_db):
    """Optional/keyword-compatible: the OLD call shape (no day_type/state_age/detail) must still
    work exactly as before -- this is what every pre-existing call site/test used until now."""
    bias = {"raw_signed": 10.0, "direction_label": "Bullish"}
    zdlog.log_signal_snapshot("TEST", bias, {"score": 50.0}, {"score": 50.0}, None,
                              {"reversal_pressure_score": 10.0}, {"agree": 1, "total": 4},
                              {"state": "Uptrend"}, spot_price=50.0, day_regime="NEUTRAL / CHOP",
                              path=tmp_db)
    con = zdlog._connect(tmp_db)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT extra FROM zero_dte_signal_log WHERE ticker='TEST'").fetchone()
    con.close()
    extra = json.loads(row["extra"])
    assert extra["day_type"] is None
    assert extra["state_age_minutes"] is None
    assert extra["detail"] is None


# ---------- compute_forward_outcomes: MFE/MAE (per user request, follow-up) ----------

def test_mfe_mae_captures_intra_window_extremes_not_just_endpoint(tmp_db):
    """The exact scenario from the user's own reviewed spec: price dips then rallies within the
    window, ending only slightly positive -- MFE must capture the BEST point reached (not just
    the final value), MAE must capture the WORST point reached."""
    base = datetime(2020, 1, 2, 10, 0, 0)
    # t0=100.00, then within the next 5 minutes: dips to 99.80 (-0.20%), rallies to 100.50 (+0.50%),
    # ends at 100.10 (+0.10%) at the row that resolves the 5m horizon.
    _insert_signal_row(tmp_db, "TEST", base, spot_price=100.00)
    _insert_signal_row(tmp_db, "TEST", base + timedelta(minutes=1), spot_price=99.80)
    _insert_signal_row(tmp_db, "TEST", base + timedelta(minutes=2), spot_price=100.50)
    _insert_signal_row(tmp_db, "TEST", base + timedelta(minutes=5), spot_price=100.10)

    df = zdlog.compute_forward_outcomes("TEST", path=tmp_db)
    first_row = df.iloc[0]
    assert first_row["fwd_5m"] == pytest.approx(0.10, abs=0.001)     # final return: small
    assert first_row["mfe_5m"] == pytest.approx(0.50, abs=0.001)     # best point reached: bigger
    assert first_row["mae_5m"] == pytest.approx(-0.20, abs=0.001)    # worst point reached


def test_mfe_mae_none_when_horizon_not_yet_resolved(tmp_db):
    base = datetime(2020, 1, 2, 15, 58, 0)   # near close, nothing after it
    _insert_signal_row(tmp_db, "TEST", base, spot_price=100.0)
    df = zdlog.compute_forward_outcomes("TEST", path=tmp_db)
    row = df.iloc[0]
    assert row["mfe_5m"] is None
    assert row["mae_5m"] is None


def test_mfe_mae_never_crosses_a_day_boundary(tmp_db):
    """Same leakage guard as returns -- a huge overnight jump must never count as an intra-window
    extreme for the PRIOR day's snapshot."""
    day1_late = datetime(2020, 1, 2, 15, 58, 0)
    day2_open = datetime(2020, 1, 3, 9, 30, 0)
    _insert_signal_row(tmp_db, "TEST", day1_late, spot_price=100.0)
    _insert_signal_row(tmp_db, "TEST", day2_open, spot_price=150.0)   # huge jump, next day
    df = zdlog.compute_forward_outcomes("TEST", path=tmp_db)
    day1_row = df[df["date"] == "2020-01-02"].iloc[0]
    assert day1_row["mfe_5m"] is None
    assert day1_row["mae_5m"] is None


def test_mfe_mae_unsigned_not_bias_adjusted(tmp_db):
    """Confirms the documented convention: mfe/mae are raw up/down excursions regardless of the
    row's own bias direction -- a BEARISH-biased row with price moving up must still report that
    as mfe (the upward number), not silently flip it to mae just because the bias was bearish."""
    con = zdlog._connect(tmp_db)
    now = datetime(2020, 1, 2, 10, 0, 0)
    extra1 = json.dumps({"spot_price": 100.0, "day_regime": "BEAR CONFIRMED"})
    extra2 = json.dumps({"spot_price": 101.0, "day_regime": "BEAR CONFIRMED"})  # price ROSE
    for ts, extra in [(now, extra1), (now + timedelta(minutes=5), extra2)]:
        con.execute(
            "INSERT INTO zero_dte_signal_log (ts, date, ticker, bias_raw_signed, extra) "
            "VALUES (?,?,?,?,?)",
            (ts.isoformat(timespec="seconds"), ts.date().isoformat(), "TEST", -40.0, extra))
    con.commit()
    con.close()
    df = zdlog.compute_forward_outcomes("TEST", path=tmp_db)
    row = df.iloc[0]
    assert row["bias_raw_signed"] == -40.0   # bearish-leaning row
    assert row["mfe_5m"] == pytest.approx(1.0, abs=0.001)   # still reports the UP move as mfe
    assert row["mae_5m"] == pytest.approx(0.0, abs=0.001)   # never went below the start


def test_regime_stats_includes_median_mfe_mae_when_sufficient(tmp_db):
    """regime_stats() surfaces median_mfe_pct/median_mae_pct once a horizon clears min_sample --
    data-layer only, no UI reads this yet."""
    base = datetime(2020, 1, 2, 10, 0, 0)
    for i in range(35):
        day = base + timedelta(days=i)
        _insert_signal_row(tmp_db, "TEST", day, spot_price=100.0, day_regime="BULL CONFIRMED")
        _insert_signal_row(tmp_db, "TEST", day + timedelta(minutes=2), spot_price=100.30)
        _insert_signal_row(tmp_db, "TEST", day + timedelta(minutes=5), spot_price=100.10)
    stats = zdlog.regime_stats("TEST", min_sample=30, path=tmp_db)
    bull = stats["groups"]["BULL CONFIRMED"]["fwd_5m"]
    assert bull["status"] == "OK"
    assert "median_mfe_pct" in bull
    assert bull["median_mfe_pct"] == pytest.approx(0.30, abs=0.01)
    assert "median_mae_pct" in bull
    assert bull["median_mae_pct"] == pytest.approx(0.0, abs=0.01)
