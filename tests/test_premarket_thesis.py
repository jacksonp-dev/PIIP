"""Tests for iip/premarket_thesis.py — signal-family blending, direction/confidence/risk bands,
confirmation/invalidation/trade-permission logic, immutable persistence, and checkpoint grading.
PIIP audit 2026-08, Premarket Thesis feature.

All persistence tests use a THROWAWAY sqlite file in the OS temp dir, created fresh and removed at
the end of each test — never the real iip.db, matching this project's established testing
discipline (see tests/test_zero_dte_log.py).
"""
from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
import uuid
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from iip import premarket_thesis as pt


@pytest.fixture
def tmp_db():
    path = os.path.join(tempfile.gettempdir(), f"test_piip_premarket_{uuid.uuid4().hex}.db")
    yield path
    # premarket_thesis.py's `with _connect(path) as con:` commits but does not close the
    # connection (same pre-existing sqlite3 gotcha as zero_dte_log.py) -- retry teardown on
    # Windows rather than let a GC-timing artifact fail an otherwise-passing test.
    gc.collect()
    for _ in range(5):
        try:
            if os.path.exists(path):
                os.remove(path)
            break
        except PermissionError:
            time.sleep(0.1)


# ---------- _family: bucket-then-score, the double-count guard ----------

def test_family_averages_correlated_instruments_not_double_counts():
    """SPY +1% and ES +1% must blend to ONE +1% contribution, not stack as if two independent
    +1% signals had fired -- the exact bug this module's docstring says it's avoiding."""
    name, pts_both, raw = pt._family("Equity", 1.0, 1.0, scale=0.5, max_pts=20)
    _, pts_one, _ = pt._family("Equity", 1.0, scale=0.5, max_pts=20)
    assert pts_both == pts_one
    assert raw == pytest.approx(1.0)


def test_family_ignores_none_values():
    name, pts, raw = pt._family("X", None, 2.0, None, scale=0.5, max_pts=20)
    assert raw == pytest.approx(2.0)


def test_family_all_none_returns_zero():
    name, pts, raw = pt._family("X", None, None, scale=1.0, max_pts=20)
    assert pts == 0.0
    assert raw is None


def test_family_invert_flips_sign():
    _, pts_normal, _ = pt._family("VIX", 5.0, scale=5.0, max_pts=15, invert=False)
    _, pts_inverted, _ = pt._family("VIX", 5.0, scale=5.0, max_pts=15, invert=True)
    assert pts_normal > 0
    assert pts_inverted < 0
    assert pts_normal == pytest.approx(-pts_inverted)


def test_family_caps_at_max_pts():
    _, pts, _ = pt._family("Equity", 1000.0, scale=0.5, max_pts=20)
    assert pts <= 20


# ---------- market_state: direction bands + confidence ----------

def _families(raw_points: dict) -> dict:
    return {name: {"points": p, "raw_pct": p} for name, p in raw_points.items()}


def test_market_state_bullish_above_threshold():
    state = pt.market_state(_families({"A": 15.0, "B": 10.0}))
    assert state["direction"] == "Bullish"
    assert state["raw_signed"] == pytest.approx(25.0)


def test_market_state_bearish_below_threshold():
    state = pt.market_state(_families({"A": -15.0, "B": -10.0}))
    assert state["direction"] == "Bearish"


def test_market_state_neutral_band():
    state = pt.market_state(_families({"A": 5.0, "B": -3.0}))
    assert state["direction"] == "Neutral"


def test_market_state_raw_signed_clipped_to_100():
    state = pt.market_state(_families({"A": 90.0, "B": 90.0}))
    assert state["raw_signed"] == 100.0


def test_market_state_confidence_bounded_30_to_95():
    strong = pt.market_state(_families({"A": 100.0, "B": 100.0}))
    weak = pt.market_state(_families({"A": 0.0}))
    assert 30 <= weak["confidence"] <= 95
    assert 30 <= strong["confidence"] <= 95
    assert strong["confidence"] >= weak["confidence"]


# ---------- risk_environment: separate axis from direction ----------

def test_risk_environment_unknown_without_vix():
    risk = pt.risk_environment(None, None)
    assert risk["level"] == "UNKNOWN"


def test_risk_environment_extreme_on_high_vix_level():
    risk = pt.risk_environment({"last": 35.0, "day_change_pct": 1.0}, None)
    assert risk["level"] == "EXTREME"


def test_risk_environment_extreme_on_big_vix_spike():
    risk = pt.risk_environment({"last": 18.0, "day_change_pct": 25.0}, None)
    assert risk["level"] == "EXTREME"


def test_risk_environment_elevated_on_moderate_vix():
    risk = pt.risk_environment({"last": 24.0, "day_change_pct": 2.0}, None)
    assert risk["level"] == "ELEVATED"


def test_risk_environment_low_on_calm_falling_vix():
    risk = pt.risk_environment({"last": 12.0, "day_change_pct": -15.0}, None)
    assert risk["level"] == "LOW"


def test_risk_environment_normal_otherwise():
    risk = pt.risk_environment({"last": 16.0, "day_change_pct": 1.0}, None)
    assert risk["level"] == "NORMAL"


def test_risk_environment_direction_independent():
    """A quiet bullish morning and a quiet bearish morning with the same VIX must read the SAME
    risk level -- risk_environment must not encode direction at all."""
    calm_bull = pt.risk_environment({"last": 16.0, "day_change_pct": 1.0}, 0.5)
    calm_bear = pt.risk_environment({"last": 16.0, "day_change_pct": 1.0}, -0.5)
    assert calm_bull["level"] == calm_bear["level"] == "NORMAL"


# ---------- confirmation_conditions ----------

def test_confirmation_neutral_is_na():
    result = pt.confirmation_conditions("Neutral", {"pct_from_vwap": 1.0}, None, None)
    assert result["status"] == "N/A"


def test_confirmation_bearish_all_conditions_met():
    snap = {"pct_from_vwap": -0.5}
    or15 = {"status": "Below breakdown"}
    result = pt.confirmation_conditions("Bearish", snap, or15, -0.2)
    assert result["status"] == "CONFIRMED"
    assert result["confirmed_count"] == 3


def test_confirmation_bearish_partial():
    snap = {"pct_from_vwap": -0.5}
    result = pt.confirmation_conditions("Bearish", snap, None, None)
    assert result["status"] == "PARTIAL"
    assert result["confirmed_count"] == 1


def test_confirmation_bullish_none_met():
    snap = {"pct_from_vwap": -0.5}
    result = pt.confirmation_conditions("Bullish", snap, {"status": "Below breakdown"}, -1.0)
    assert result["status"] == "NOT CONFIRMED"
    assert result["confirmed_count"] == 0


# ---------- invalidation_conditions: symmetric to confirmation ----------

def test_invalidation_neutral_is_na():
    result = pt.invalidation_conditions("Neutral", {}, {})
    assert result["status"] == "N/A"


def test_invalidation_bearish_reclaims_both_levels_is_invalidated():
    snap = {"pct_from_vwap": 0.5, "last": 105.0}
    prev_day = {"close": 100.0}
    result = pt.invalidation_conditions("Bearish", snap, prev_day)
    assert result["status"] == "INVALIDATED"


def test_invalidation_bearish_intact_when_still_below():
    snap = {"pct_from_vwap": -0.5, "last": 95.0}
    prev_day = {"close": 100.0}
    result = pt.invalidation_conditions("Bearish", snap, prev_day)
    assert result["status"] == "INTACT"


def test_invalidation_bullish_loses_both_levels_is_invalidated():
    snap = {"pct_from_vwap": -0.5, "last": 95.0}
    prev_day = {"close": 100.0}
    result = pt.invalidation_conditions("Bullish", snap, prev_day)
    assert result["status"] == "INVALIDATED"


# ---------- trade_permission ----------

def test_trade_permission_neutral_is_no_trade():
    perm = pt.trade_permission("Neutral", {"status": "N/A", "confirmed_count": 0, "total": 0})
    assert perm["status"] == "NO_TRADE"


def test_trade_permission_unconfirmed_is_wait():
    perm = pt.trade_permission("Bullish", {"status": "PARTIAL", "confirmed_count": 1, "total": 3})
    assert perm["status"] == "WAIT"
    assert perm["preferred_direction"] == "CALL"


def test_trade_permission_confirmed_bullish_is_calls_favored():
    perm = pt.trade_permission("Bullish", {"status": "CONFIRMED", "confirmed_count": 3, "total": 3})
    assert perm["status"] == "CALLS_FAVORED"


def test_trade_permission_confirmed_bearish_is_puts_favored():
    perm = pt.trade_permission("Bearish", {"status": "CONFIRMED", "confirmed_count": 3, "total": 3})
    assert perm["status"] == "PUTS_FAVORED"


# ---------- is_primary: hierarchy handoff ----------

def test_is_primary_true_before_alignment_and_before_cutoff():
    now = datetime(2026, 8, 18, 9, 45)
    assert pt.is_primary(now, alignment_total=0) is True


def test_is_primary_false_once_alignment_has_data():
    now = datetime(2026, 8, 18, 9, 45)
    assert pt.is_primary(now, alignment_total=2) is False


def test_is_primary_false_past_backstop_cutoff():
    now = datetime(2026, 8, 18, 10, 30)
    assert pt.is_primary(now, alignment_total=0) is False


# ---------- persistence: immutability (schema-enforced, not just app discipline) ----------

def _thesis(direction="Bearish", confidence=72.0, spot=640.0, risk="ELEVATED"):
    return {"spot_price": spot, "market_state": {"direction": direction, "confidence": confidence},
           "risk_environment": {"level": risk}, "families": {}, "confirmation": {},
           "invalidation": {}, "trade_permission": {}}


def test_log_thesis_once_first_write_succeeds(tmp_db):
    assert pt.log_thesis_once("SPY", _thesis(), path=tmp_db) is True


def test_log_thesis_once_second_write_same_day_is_noop(tmp_db):
    pt.log_thesis_once("SPY", _thesis(direction="Bearish", spot=640.0), path=tmp_db)
    wrote_again = pt.log_thesis_once("SPY", _thesis(direction="Bullish", spot=999.0), path=tmp_db)
    assert wrote_again is False
    readback = pt.get_todays_thesis("SPY", path=tmp_db)
    assert readback["market_state"]["direction"] == "Bearish"
    assert readback["spot_price"] == 640.0


def test_get_todays_thesis_none_when_nothing_logged(tmp_db):
    assert pt.get_todays_thesis("SPY", path=tmp_db) is None


def test_thesis_scoped_per_ticker(tmp_db):
    pt.log_thesis_once("SPY", _thesis(direction="Bearish"), path=tmp_db)
    pt.log_thesis_once("QQQ", _thesis(direction="Bullish"), path=tmp_db)
    assert pt.get_todays_thesis("SPY", path=tmp_db)["market_state"]["direction"] == "Bearish"
    assert pt.get_todays_thesis("QQQ", path=tmp_db)["market_state"]["direction"] == "Bullish"


def test_confirmation_event_edge_triggered(tmp_db):
    first = pt.log_confirmation_event_once("SPY", "Bearish", path=tmp_db)
    second = pt.log_confirmation_event_once("SPY", "Bullish", path=tmp_db)
    assert first is True
    assert second is False
    event = pt.get_confirmation_event("SPY", path=tmp_db)
    assert event["direction"] == "Bearish"   # the FIRST trigger wins, never overwritten


def test_confirmation_event_none_when_nothing_logged(tmp_db):
    assert pt.get_confirmation_event("SPY", path=tmp_db) is None


# ---------- compute_checkpoint: directional accuracy, confirmation, tradeability ----------

def test_checkpoint_directional_accuracy_correct_for_bearish_drop():
    thesis = _thesis(direction="Bearish", spot=640.0)
    checkpoint = pt.compute_checkpoint(thesis, current_spot=635.0, current_direction_now="Bearish",
                                       confirmation_event=None, label="10:00 AM")
    assert checkpoint["directional_accuracy"] == "CORRECT"


def test_checkpoint_directional_accuracy_wrong_for_bearish_rally():
    thesis = _thesis(direction="Bearish", spot=640.0)
    checkpoint = pt.compute_checkpoint(thesis, current_spot=645.0, current_direction_now="Bullish",
                                       confirmation_event=None, label="10:00 AM")
    assert checkpoint["directional_accuracy"] == "WRONG"


def test_checkpoint_directional_accuracy_flat_when_unchanged():
    thesis = _thesis(direction="Bearish", spot=640.0)
    checkpoint = pt.compute_checkpoint(thesis, current_spot=640.0, current_direction_now="Neutral",
                                       confirmation_event=None, label="10:00 AM")
    assert checkpoint["directional_accuracy"] == "FLAT"


def test_checkpoint_directional_accuracy_no_call_for_neutral_thesis():
    thesis = _thesis(direction="Neutral", spot=640.0)
    checkpoint = pt.compute_checkpoint(thesis, current_spot=650.0, current_direction_now="Bullish",
                                       confirmation_event=None, label="10:00 AM")
    assert checkpoint["directional_accuracy"] == "NO_CALL"


def test_checkpoint_tradeability_none_when_never_confirmed():
    thesis = _thesis(direction="Bearish", spot=640.0)
    checkpoint = pt.compute_checkpoint(thesis, current_spot=635.0, current_direction_now="Bearish",
                                       confirmation_event=None, label="10:00 AM")
    assert checkpoint["tradeability"] == "NONE"


def test_checkpoint_tradeability_none_when_confirmation_was_opposite_direction():
    """The confirmation engine fired, but for the OPPOSITE direction of the original thesis --
    must not be credited as if the original thesis was tradeable."""
    thesis = _thesis(direction="Bearish", spot=640.0)
    confirmation_event = {"ts": "2026-08-18T10:15:00", "direction": "Bullish"}
    checkpoint = pt.compute_checkpoint(thesis, current_spot=635.0, current_direction_now="Bearish",
                                       confirmation_event=confirmation_event, label="10:00 AM")
    assert checkpoint["tradeability"] == "NONE"


def test_checkpoint_tradeability_excellent_when_confirmed_quickly():
    thesis = _thesis(direction="Bearish", spot=640.0)
    market_open = datetime(2026, 8, 18, 9, 30)
    confirmation_event = {"ts": "2026-08-18T09:45:00", "direction": "Bearish"}   # 15 min after open
    checkpoint = pt.compute_checkpoint(thesis, current_spot=635.0, current_direction_now="Bearish",
                                       confirmation_event=confirmation_event, label="10:00 AM",
                                       market_open_et=market_open)
    assert checkpoint["tradeability"] == "EXCELLENT"


def test_checkpoint_tradeability_poor_when_confirmed_late():
    thesis = _thesis(direction="Bearish", spot=640.0)
    market_open = datetime(2026, 8, 18, 9, 30)
    confirmation_event = {"ts": "2026-08-18T11:00:00", "direction": "Bearish"}   # 90 min after open
    checkpoint = pt.compute_checkpoint(thesis, current_spot=635.0, current_direction_now="Bearish",
                                       confirmation_event=confirmation_event, label="10:00 AM",
                                       market_open_et=market_open)
    assert checkpoint["tradeability"] == "POOR"


def test_log_checkpoint_is_immutable_per_label(tmp_db):
    thesis = _thesis(direction="Bearish", spot=640.0)
    checkpoint = pt.compute_checkpoint(thesis, current_spot=635.0, current_direction_now="Bearish",
                                       confirmation_event=None, label="10:00 AM")
    first = pt.log_checkpoint("SPY", "10:00 AM", checkpoint, path=tmp_db)
    second = pt.log_checkpoint("SPY", "10:00 AM", checkpoint, path=tmp_db)
    assert first is True
    assert second is False
    stored = pt.get_checkpoints("SPY", path=tmp_db)
    assert len(stored) == 1


def test_checkpoints_scoped_by_date(tmp_db):
    """Leakage-style guard: a checkpoint logged for one day must not appear when a DIFFERENT
    target_date is queried -- mirrors zero_dte_log's same-day-only discipline, applied here to the
    checkpoint log."""
    thesis = _thesis(direction="Bearish", spot=640.0)
    checkpoint = pt.compute_checkpoint(thesis, current_spot=635.0, current_direction_now="Bearish",
                                       confirmation_event=None, label="10:00 AM")
    pt.log_checkpoint("SPY", "10:00 AM", checkpoint, path=tmp_db)
    assert len(pt.get_checkpoints("SPY", path=tmp_db)) == 1
    assert pt.get_checkpoints("SPY", target_date="2020-01-01", path=tmp_db) == []
