"""Tests for iip/timeframe.py — resampling windows, alignment, and tie handling.
PIIP audit 2026-08, state-architecture review, Phase 8.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from iip import timeframe as tf
from tests.conftest import make_intraday_1m


# ---------- resample_ohlc: window/bar-count correctness (test area 1) ----------

def test_resample_ohlc_bar_count():
    """N minutes of 1m bars resampled to `m`-minute bars should produce ceil(N/m) bars — the
    literal window-size claim audited in the architecture review (Phase 6/7 of the audit)."""
    df = make_intraday_1m(n_bars=150, pattern="flat")
    bars_5m = tf.resample_ohlc(df, 5)
    bars_15m = tf.resample_ohlc(df, 15)
    assert len(bars_5m) == 30      # 150 / 5
    assert len(bars_15m) == 10     # 150 / 15


def test_resample_ohlc_preserves_ohlc_semantics():
    """Each resampled bar's Open must be the FIRST 1m bar's Open in that window, High the MAX,
    Low the MIN, Close the LAST — not some other aggregation that would silently misrepresent
    the window."""
    df = make_intraday_1m(n_bars=10, pattern="uptrend")
    bars = tf.resample_ohlc(df, 5)
    first_window = df.iloc[0:5]
    assert bars["Open"].iloc[0] == first_window["Open"].iloc[0]
    assert bars["High"].iloc[0] == first_window["High"].max()
    assert bars["Low"].iloc[0] == first_window["Low"].min()
    assert bars["Close"].iloc[0] == first_window["Close"].iloc[-1]


def test_resample_ohlc_empty_input():
    assert tf.resample_ohlc(None, 5).empty
    assert tf.resample_ohlc(pd.DataFrame(), 5).empty


# ---------- timeframe_momentum: the "5m/15m/30m window" claim itself ----------

def test_timeframe_momentum_needs_enough_bars():
    """5m timeframe needs 10 resampled 5m bars = 50 minutes of 1m session data -- fewer than that
    must return None (honest "not available yet"), never a guess from a partial window."""
    short_df = make_intraday_1m(n_bars=40, pattern="uptrend")   # only 8 resampled 5m bars
    assert tf.timeframe_momentum(short_df, 5, "5m") is None

    enough_df = make_intraday_1m(n_bars=50, pattern="uptrend")  # exactly 10 resampled 5m bars
    result = tf.timeframe_momentum(enough_df, 5, "5m")
    assert result is not None
    assert result["bars_used"] == 10
    assert result["timeframe"] == "5m"


def test_timeframe_momentum_direction_matches_price_path():
    up_df = make_intraday_1m(n_bars=50, pattern="uptrend")
    down_df = make_intraday_1m(n_bars=50, pattern="downtrend")
    assert tf.timeframe_momentum(up_df, 5, "5m")["direction"] == "Bullish"
    assert tf.timeframe_momentum(down_df, 5, "5m")["direction"] == "Bearish"


def test_daily_reading_reuses_ema_alignment_not_a_new_calc():
    assert tf.daily_reading({"ema_aligned_bull": True, "ema_aligned_bear": False}) == {
        "timeframe": "Daily", "direction": "Bullish",
        "note": ("EMA20/50/200 alignment on daily bars — the session's broader trend context, "
                "not an intraday read."),
    }
    assert tf.daily_reading({"ema_aligned_bull": False, "ema_aligned_bear": True})["direction"] == "Bearish"
    assert tf.daily_reading({"ema_aligned_bull": None, "ema_aligned_bear": None}) is None


# ---------- timeframe_alignment: agreement counting + TIE HANDLING (test areas 2, 3) ----------

def _snapshot(directions: dict) -> dict:
    """Build a minimal multi_timeframe_snapshot()-shaped dict from {label: direction}."""
    return {label: {"available": True, "direction": d} for label, d in directions.items()}


def test_alignment_majority_bullish():
    snap = _snapshot({"5m": "Bullish", "15m": "Bullish", "30m": "Bearish", "Daily": "Bullish"})
    result = tf.timeframe_alignment(snap)
    assert result["aligned_direction"] == "Bullish"
    assert result["agree"] == 3
    assert result["total"] == 4


def test_alignment_true_tie_is_mixed_not_bullish():
    """Regression test for the exact bug the audit found and the code comment documents: an exact
    2-2 tie used to be silently called "Bullish" (`bulls >= bears`). Must be "Mixed"."""
    snap = _snapshot({"5m": "Bullish", "15m": "Bearish", "30m": "Bullish", "Daily": "Bearish"})
    result = tf.timeframe_alignment(snap)
    assert result["aligned_direction"] == "Mixed"


def test_alignment_zero_zero_tie_is_mixed():
    """All-Flat available timeframes: 0 bulls, 0 bears -- must not be miscounted as Bullish."""
    snap = _snapshot({"5m": "Flat", "15m": "Flat"})
    result = tf.timeframe_alignment(snap)
    assert result["aligned_direction"] == "Mixed"


def test_alignment_no_available_timeframes():
    snap = {"5m": {"available": False}, "15m": {"available": False}}
    result = tf.timeframe_alignment(snap)
    assert result["total"] == 0
    assert result["aligned_direction"] == "Unknown"


def test_alignment_skips_unavailable_without_penalizing():
    """A timeframe that isn't available yet must be listed in `skipped`, never counted as a
    disagreement against the available ones."""
    snap = {"5m": {"available": True, "direction": "Bullish"},
           "15m": {"available": True, "direction": "Bullish"},
           "30m": {"available": False}}
    result = tf.timeframe_alignment(snap)
    assert result["total"] == 2
    assert result["agree"] == 2
    assert "30m" in result["skipped"]


# ---------- interpret_timeframe_sequence: pullback vs. pressure vs. regime-transition reads ----------

def test_sequence_all_aligned():
    snap = _snapshot({"Daily": "Bullish", "30m": "Bullish", "15m": "Bullish", "5m": "Bullish"})
    result = tf.interpret_timeframe_sequence(snap)
    assert "ALIGNED ACROSS ALL TIMEFRAMES" in result["interpretation"]


def test_sequence_short_term_pullback():
    """Higher timeframes bullish, only the SHORTEST disagrees -- the user's own worked example."""
    snap = _snapshot({"Daily": "Bullish", "30m": "Bullish", "15m": "Bullish", "5m": "Bearish"})
    result = tf.interpret_timeframe_sequence(snap)
    assert "SHORT-TERM PULLBACK" in result["interpretation"]
    assert "BULLISH" in result["interpretation"]


def test_sequence_trend_under_pressure():
    snap = _snapshot({"Daily": "Bullish", "30m": "Bearish", "15m": "Bearish", "5m": "Bullish"})
    result = tf.interpret_timeframe_sequence(snap)
    assert "UNDER SHORT-TERM PRESSURE" in result["interpretation"]


def test_sequence_potential_regime_transition():
    """Every shorter timeframe disagrees with the longest available one -- the user's other
    worked example (Daily Bull / 30m Bear / 15m Bear / 5m Bear)."""
    snap = _snapshot({"Daily": "Bullish", "30m": "Bearish", "15m": "Bearish", "5m": "Bearish"})
    result = tf.interpret_timeframe_sequence(snap)
    assert result["interpretation"] == "POTENTIAL REGIME TRANSITION"


def test_sequence_insufficient_timeframes():
    snap = _snapshot({"5m": "Bullish"})
    result = tf.interpret_timeframe_sequence(snap)
    assert result["interpretation"] == "INSUFFICIENT TIMEFRAMES"


# ---------- update_trend_state: hysteresis (must not flip on one noisy read) ----------

def test_hysteresis_requires_consecutive_confirms():
    bullish_alignment = {"total": 4, "alignment_pct": 100.0, "aligned_direction": "Bullish"}
    state1 = tf.update_trend_state(None, bullish_alignment, confirm_reads=2)
    assert state1["state"] == "Uptrend"   # first-ever read seeds state directly, not pending

    chop_alignment = {"total": 4, "alignment_pct": 25.0, "aligned_direction": "Mixed"}
    state2 = tf.update_trend_state(state1, chop_alignment, confirm_reads=2)
    assert state2["state"] == "Uptrend", "must NOT flip on a single disagreeing read"
    assert state2["pending"] == "Range / Chop"
    assert state2["changed"] is False

    state3 = tf.update_trend_state(state2, chop_alignment, confirm_reads=2)
    assert state3["state"] == "Range / Chop", "must flip after 2 CONSECUTIVE confirming reads"
    assert state3["changed"] is True
