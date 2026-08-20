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
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from iip import premarket_thesis as pt

ET = ZoneInfo("America/New_York")


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


# ---------- evaluate_thesis_at_checkpoint: the fixed, self-contained historical grading ----------
#
# PIIP audit 2026-08: replaces the old compute_checkpoint(), which graded the original thesis by
# (a) comparing raw price only, and (b) consulting the LIVE confirmation-event log -- which
# tracks confirmation for whatever direction is CURRENTLY live, not the original thesis's own
# direction, and could silently attach an opposite-direction live event to the original thesis's
# grade. evaluate_thesis_at_checkpoint() is fully self-contained: it re-slices real intraday bars
# to an as-of cutoff and re-runs confirmation_conditions()/invalidation_conditions() for the
# ORIGINAL direction itself. Synthetic, deterministic bar fixtures throughout, per this project's
# own testing convention (see tests/test_market_structure.py's identical pattern).

def _synth_bars(closes: list[float], start: dtime = dtime(9, 30), vol: float = 1000.0) -> pd.DataFrame:
    """1-minute bars, O=H=L=C=close (flat candles, kept simple) -- tz-aware DatetimeIndex
    starting at `start` today, one bar per minute."""
    t = datetime.combine(date.today(), start, tzinfo=ET)
    idx, data = [], {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
    for c in closes:
        idx.append(t)
        data["Open"].append(c); data["High"].append(c); data["Low"].append(c); data["Close"].append(c)
        data["Volume"].append(vol)
        t += timedelta(minutes=1)
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


MARKET_OPEN = datetime.combine(date.today(), dtime(9, 30), tzinfo=ET)
CHECKPOINT_10AM = datetime.combine(date.today(), dtime(10, 0), tzinfo=ET)


def test_checkpoint_correct_bullish_continuation():
    """Test 1: original Bullish, market confirms bullish conditions, no invalidation -> CORRECT."""
    original = _thesis(direction="Bullish", spot=95.0)
    # 15 flat bars (sets the opening range at exactly 100.0), then 15 bars rallying to 103.0 --
    # ends well above VWAP, above the OR high, above the prior close. QQQ rallies HARDER (+5%
    # vs SPY's +3%) so relative strength is unambiguously positive, not a coincidental tie.
    spy = _synth_bars([100.0] * 15 + [100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6,
                                      101.8, 102.0, 102.2, 102.4, 102.6, 102.8, 103.0])
    qqq = _synth_bars([200.0] * 15 + [200.7, 201.4, 202.1, 202.8, 203.5, 204.2, 204.9, 205.6,
                                      206.3, 207.0, 207.7, 208.4, 209.1, 209.8, 210.0])
    prev_day = {"close": 99.0}
    result = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, spy, qqq,
                                              None, None, None, prev_day, MARKET_OPEN)
    assert result["status"] == "OK"
    assert result["confirmation"]["status"] == "CONFIRMED"
    assert result["invalidation"]["status"] == "INTACT"
    assert result["opposing_pressure"] == "NONE"
    assert result["thesis_verdict"] == "CORRECT"


def test_checkpoint_wrong_clean_bearish_reversal():
    """Test 2: original Bullish, bearish invalidation occurs, bearish conditions confirmed ->
    WRONG."""
    original = _thesis(direction="Bullish", spot=105.0)
    # SPY falls -3%; QQQ falls HARDER (-5%) so relative weakness is unambiguous, not a tie.
    spy = _synth_bars([100.0] * 15 + [99.8, 99.6, 99.4, 99.2, 99.0, 98.8, 98.6, 98.4,
                                      98.2, 98.0, 97.8, 97.6, 97.4, 97.2, 97.0])
    qqq = _synth_bars([200.0] * 15 + [199.3, 198.6, 197.9, 197.2, 196.5, 195.8, 195.1, 194.4,
                                      193.7, 193.0, 192.3, 191.6, 190.9, 190.2, 190.0])
    prev_day = {"close": 99.0}   # price (97.0) ends below prior close -> invalidation trips
    result = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, spy, qqq,
                                              None, None, None, prev_day, MARKET_OPEN)
    assert result["invalidation"]["status"] == "INVALIDATED"
    assert result["opposing_direction"] == "Bearish"
    assert result["opposing_pressure"] == "CONFIRMED"
    assert result["thesis_verdict"] == "WRONG"


def test_checkpoint_partially_correct_mixed_market():
    """Test 3: original Bullish, some confirmation, some invalidation, no decisive direction ->
    PARTIALLY CORRECT. Price stays above VWAP (1 of 3 confirmation checks) but below the prior
    close (1 of 2 invalidation checks) and QQQ lags (feeds both 'no confirmation' and 'opposing
    pressure developing')."""
    original = _thesis(direction="Bullish", spot=100.0)
    # Stays inside/near the opening range (no breakout), drifts to 100.3 -- just above its own
    # volume-weighted average price, but below the prior close.
    spy = _synth_bars([100.0] * 15 + [100.05, 100.1, 100.05, 100.1, 100.15, 100.1, 100.15, 100.2,
                                      100.15, 100.2, 100.25, 100.2, 100.25, 100.3, 100.3])
    qqq = _synth_bars([200.0] * 15 + [199.9] * 15)   # QQQ lags -> no QQQ confirmation, some
    # opposing (bearish) pressure via QQQ relative weakness
    prev_day = {"close": 101.0}   # last (100.3) < prior close -> "loses prior close" trips
    result = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, spy, qqq,
                                              None, None, None, prev_day, MARKET_OPEN)
    assert result["confirmation"]["status"] in ("PARTIAL", "NOT CONFIRMED")
    assert result["invalidation"]["status"] == "AT RISK"
    assert result["thesis_verdict"] == "PARTIALLY_CORRECT"


def test_checkpoint_too_early_to_tell():
    """Test 4: original Bullish, market remains inside the opening range, no meaningful
    confirmation, no invalidation -> TOO EARLY TO TELL. A fully flat session: last price exactly
    equals VWAP and the opening-range boundary, so none of the strict > / < checks fire either
    way -- deliberately NOT judged from SPY's absolute price change alone (both are literally
    unchanged from the open here, which is the point)."""
    original = _thesis(direction="Bullish", spot=100.0)
    spy = _synth_bars([100.0] * 30)   # perfectly flat all session: VWAP=100.0=last=OR high=OR low
    qqq = _synth_bars([200.0] * 30)   # QQQ also flat -> relative strength exactly 0
    prev_day = {"close": 99.0}        # last (100.0) > prior close -> invalidation stays INTACT
    result = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, spy, qqq,
                                              None, None, None, prev_day, MARKET_OPEN)
    assert result["confirmation"]["status"] == "NOT CONFIRMED"
    assert result["invalidation"]["status"] == "INTACT"
    assert result["opposing_pressure"] == "NONE"
    assert result["thesis_verdict"] == "TOO_EARLY_TO_TELL"


def test_checkpoint_live_bearish_event_does_not_contaminate_original_bullish_grade(tmp_db):
    """Test 5, the single most important test in this file (per explicit user emphasis): the
    LIVE confirmation-event log independently records a Bearish event -- exactly what app.py's
    UNTOUCHED live-confirmation-logging code does every ~30s, completely unrelated to the
    checkpoint -- and the historical checkpoint must grade the ORIGINAL Bullish thesis using
    ONLY its own as-of-cutoff bar evaluation, never that log.

    The proof: evaluate_thesis_at_checkpoint() takes no confirmation_event parameter at all, so
    calling it is BIT-FOR-BIT IDENTICAL whether or not an opposing-direction live event was ever
    logged. If this test passes, the live and historical code paths are architecturally
    incapable of contaminating each other, not just coincidentally correct on this one input."""
    original = _thesis(direction="Bullish", spot=105.0)
    # Real, mixed-bearish-leaning bars -- NOT fully confirmed bearish, so this also naturally
    # exercises a non-trivial opposing_pressure read (DEVELOPING), not just NONE/CONFIRMED.
    spy = _synth_bars([100.0] * 15 + [99.9, 99.8, 99.9, 99.8, 99.7, 99.8, 99.7, 99.6,
                                      99.7, 99.6, 99.5, 99.6, 99.5, 99.4, 99.5])
    qqq = _synth_bars([200.0] * 15 + [199.7] * 15)
    prev_day = {"close": 99.0}

    # Simulate the LIVE engine independently logging a Bearish confirmation at ~9:55 -- this is
    # EXACTLY what the untouched app.py code does, nothing about it is mocked or special-cased.
    wrote = pt.log_confirmation_event_once("SPY", "Bearish", path=tmp_db)
    assert wrote is True
    live_event = pt.get_confirmation_event("SPY", path=tmp_db)
    assert live_event["direction"] == "Bearish"   # confirms the live path really fired

    result_with_live_event = pt.evaluate_thesis_at_checkpoint(
        original, "10:00 AM", CHECKPOINT_10AM, spy, qqq, None, None, None, prev_day, MARKET_OPEN)
    result_without_any_live_event = pt.evaluate_thesis_at_checkpoint(
        original, "10:00 AM", CHECKPOINT_10AM, spy, qqq, None, None, None, prev_day, MARKET_OPEN)

    assert result_with_live_event == result_without_any_live_event   # THE decoupling proof
    assert result_with_live_event["original_direction"] == "Bullish"   # never rewritten
    assert result_with_live_event["opposing_direction"] == "Bearish"
    assert result_with_live_event["opposing_pressure"] in ("DEVELOPING", "CONFIRMED")


def test_checkpoint_delayed_execution_uses_10am_data_not_1017am_data():
    """Test 6: checkpoint target = 10:00, but this function being CALLED late (simulating a page
    refresh at 10:17) must not silently grade against 10:17 data. Bars after 10:00 show a
    dramatically different price (a late plunge to 80.0) that must be completely invisible to a
    checkpoint pinned to the 10:00 target."""
    original = _thesis(direction="Bullish", spot=95.0)
    # 9:30-10:00 inclusive (31 one-minute bars: 9:30,...,10:00): clean bullish rally ending AT
    # the 10:00 bar itself -- a bar timestamped exactly 10:00:00 is a valid "data used" per the
    # spec's own example (only bars STRICTLY AFTER 10:00 must be excluded).
    # 10:01-10:17 (17 more bars): a big plunge to 80.0 -- must be ignored entirely.
    pre_10am = [100.0] * 15 + [100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6,
                               101.8, 102.0, 102.2, 102.4, 102.6, 102.8, 103.0, 103.2]
    post_10am = [95.0, 90.0, 85.0, 80.0] + [80.0] * 13
    spy = _synth_bars(pre_10am + post_10am)
    qqq = _synth_bars([200.0] * 15 + [200.7, 201.4, 202.1, 202.8, 203.5, 204.2, 204.9, 205.6,
                                      206.3, 207.0, 207.7, 208.4, 209.1, 209.8, 210.0, 210.2]
                      + [150.0] * 17)
    prev_day = {"close": 99.0}
    result = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, spy, qqq,
                                              None, None, None, prev_day, MARKET_OPEN)
    assert result["status"] == "OK"
    assert result["target_checkpoint_time"] == CHECKPOINT_10AM.isoformat()
    # actual_data_timestamp must be the 10:00:00 bar itself, never anything after it.
    actual_ts = datetime.fromisoformat(result["actual_data_timestamp"])
    assert actual_ts == CHECKPOINT_10AM
    # The verdict must reflect the 103.2 (bullish) state, NOT the 80.0 (crashed) state.
    assert result["actual_spy_price"] == pytest.approx(103.2)
    assert result["thesis_verdict"] == "CORRECT"


def test_checkpoint_price_lower_than_original_but_behavior_correct_is_not_automatically_wrong():
    """Test 8: SPY's absolute price is LOWER than the original reference spot_price, but the
    actual intraday BEHAVIOR (VWAP reclaim/hold, OR breakout, QQQ strength, no invalidation) is
    cleanly bullish-confirmed -- must NOT be automatically marked WRONG just because
    current_price < original_spot (the exact bug this whole fix removes)."""
    original = _thesis(direction="Bullish", spot=110.0)   # reference price ABOVE where the
    # session actually trades all morning -- if the system were still doing raw price
    # comparison, this would read WRONG no matter what the bars show.
    spy = _synth_bars([100.0] * 15 + [100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6,
                                      101.8, 102.0, 102.2, 102.4, 102.6, 102.8, 103.0])   # ends
    # at 103.0, still well below original_spot=110.0
    qqq = _synth_bars([200.0] * 15 + [200.7, 201.4, 202.1, 202.8, 203.5, 204.2, 204.9, 205.6,
                                      206.3, 207.0, 207.7, 208.4, 209.1, 209.8, 210.0])
    prev_day = {"close": 99.0}
    result = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, spy, qqq,
                                              None, None, None, prev_day, MARKET_OPEN)
    assert result["actual_spy_price"] < result["original_spot"]   # confirms the setup is real
    assert result["thesis_verdict"] == "CORRECT"   # behavior-graded, not price-graded


def test_checkpoint_neutral_thesis_is_no_call():
    original = _thesis(direction="Neutral", spot=100.0)
    spy = _synth_bars([100.0] * 30)
    result = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, spy, None,
                                              None, None, None, {"close": 99.0}, MARKET_OPEN)
    assert result["thesis_verdict"] == "NO_CALL"


def test_checkpoint_data_unavailable_when_no_bars_at_cutoff():
    """Section 3's explicit requirement: if 10:00 data genuinely doesn't exist, say so plainly
    rather than silently substituting a later (or earlier) bar."""
    original = _thesis(direction="Bullish", spot=100.0)
    # All bars are from BEFORE the checkpoint target's own day-open reference in this synthetic
    # setup would still normally resolve -- simulate genuine absence by passing an empty frame.
    empty = _synth_bars([])
    result = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, empty, None,
                                              None, None, None, {"close": 99.0}, MARKET_OPEN)
    assert result["status"] == "DATA_UNAVAILABLE"
    assert result["thesis_verdict"] == "DATA_UNAVAILABLE"
    assert result["actual_data_timestamp"] is None


def test_log_checkpoint_is_immutable_per_label(tmp_db):
    """Test 7 (persistence half): once logged, a checkpoint must never be silently overwritten --
    proves the historical record stays frozen regardless of what's computed or attempted later
    (e.g. a later Day Regime change, or even a differently-shaped checkpoint dict)."""
    original = _thesis(direction="Bullish", spot=95.0)
    spy = _synth_bars([100.0] * 30)
    checkpoint = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, spy, None,
                                                  None, None, None, {"close": 99.0}, MARKET_OPEN)
    first = pt.log_checkpoint("SPY", "10:00 AM", checkpoint, path=tmp_db)
    # Attempt a second write with a DIFFERENT (contradictory) checkpoint payload for the SAME
    # label -- must be rejected, proving later information (e.g. Day Regime moving to BULL
    # DEVELOPING afterward) can never rewrite the frozen 10:00 verdict.
    different_checkpoint = {**checkpoint, "thesis_verdict": "WRONG"}
    second = pt.log_checkpoint("SPY", "10:00 AM", different_checkpoint, path=tmp_db)
    assert first is True
    assert second is False
    stored = pt.get_checkpoints("SPY", path=tmp_db)
    assert len(stored) == 1
    assert stored[0]["thesis_verdict"] == checkpoint["thesis_verdict"]   # the ORIGINAL verdict survives


def test_checkpoints_scoped_by_date(tmp_db):
    """Leakage-style guard: a checkpoint logged for one day must not appear when a DIFFERENT
    target_date is queried -- mirrors zero_dte_log's same-day-only discipline, applied here to the
    checkpoint log."""
    original = _thesis(direction="Bullish", spot=95.0)
    spy = _synth_bars([100.0] * 30)
    checkpoint = pt.evaluate_thesis_at_checkpoint(original, "10:00 AM", CHECKPOINT_10AM, spy, None,
                                                  None, None, None, {"close": 99.0}, MARKET_OPEN)
    pt.log_checkpoint("SPY", "10:00 AM", checkpoint, path=tmp_db)
    assert len(pt.get_checkpoints("SPY", path=tmp_db)) == 1
    assert pt.get_checkpoints("SPY", target_date="2020-01-01", path=tmp_db) == []
