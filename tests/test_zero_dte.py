"""Tests for iip/zero_dte.py — trend efficiency, VWAP crossings, trend integrity, participation/
breadth de-duplication, data freshness, and the Phase 2 direction-only-language regression.
PIIP audit 2026-08, state-architecture review, Phase 8.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from iip import zero_dte as zd
from tests.conftest import make_intraday_1m


# ---------- trend_efficiency (test area 4) ----------

def test_trend_efficiency_pure_uptrend_is_high():
    """A monotonic uptrend should read close to 100% -- every bar-to-bar move is in the same
    direction, so net move ≈ total path length."""
    df = make_intraday_1m(n_bars=30, pattern="uptrend")
    result = zd.trend_efficiency(df)
    assert result["efficiency_pct"] > 90.0


def test_trend_efficiency_v_reversal_is_low():
    """A V-shaped reversal that ends near where it started has a small NET move but a large total
    PATH length -- efficiency must read low even though 'today's range' looks large, this is
    exactly the whipsaw-sensitivity trend_efficiency() was built to add over market_dna's
    net_vs_range."""
    df = make_intraday_1m(n_bars=60, pattern="v_reversal")
    result = zd.trend_efficiency(df)
    assert result["efficiency_pct"] < 20.0


def test_trend_efficiency_too_few_bars():
    assert zd.trend_efficiency(make_intraday_1m(n_bars=3)) is None
    assert zd.trend_efficiency(None) is None


# ---------- vwap_crossings (test area 5) ----------

def test_vwap_crossings_clean_trend_has_few():
    """A clean uptrend starting right at/above VWAP should cross rarely (ideally 0-1 times)."""
    df = make_intraday_1m(n_bars=40, pattern="uptrend")
    result = zd.vwap_crossings(df)
    assert result is not None
    assert result["count"] <= 1


def test_vwap_crossings_v_reversal_crosses_at_least_once():
    df = make_intraday_1m(n_bars=60, pattern="v_reversal")
    result = zd.vwap_crossings(df)
    assert result is not None
    assert result["count"] >= 1
    assert result["current_side"] in ("Above VWAP", "Below VWAP")


def test_vwap_crossings_too_few_bars():
    assert zd.vwap_crossings(make_intraday_1m(n_bars=2)) is None


# ---------- trend_integrity: synthesis, no new independent judgment ----------

def test_trend_integrity_clean_trend_scores_high():
    alignment = {"alignment_pct": 100.0, "total": 4}
    crossings = {"count": 0}
    efficiency = {"efficiency_pct": 95.0}
    confluence = {"agree": 6, "total": 6}
    result = zd.trend_integrity(alignment, crossings, efficiency, confluence)
    assert result["label"] == "Clean Trend"
    assert result["score"] > 70


def test_trend_integrity_choppy_scores_low():
    alignment = {"alignment_pct": 25.0, "total": 4}
    crossings = {"count": 8}
    efficiency = {"efficiency_pct": 10.0}
    confluence = {"agree": 1, "total": 6}
    result = zd.trend_integrity(alignment, crossings, efficiency, confluence)
    assert result["label"] == "Choppy / Mixed"
    assert result["score"] < 40


# ---------- participation_state: the breadth/sector/mega-cap de-duplication fix (test area 12) ----------

def _breadth(idx_g, idx_r, mega_g, mega_r, sec_g, sec_r):
    return {"indices_green": idx_g, "indices_red": idx_r,
           "mega_caps_green": mega_g, "mega_caps_red": mega_r,
           "sectors_green": sec_g, "sectors_red": sec_r,
           "score_signed": 0.0, "label": "Mixed", "note": "", "green": 0, "red": 0, "total": 0}


def test_participation_all_buckets_agree_strong():
    breadth = _breadth(3, 0, 10, 0, 10, 0)   # every bucket unanimously green
    result = zd.participation_state(breadth)
    assert result["state"] == "STRONG"
    assert result["agree"] is True
    assert result["n_buckets"] == 3


def test_participation_buckets_disagree_is_mixed():
    """Index bucket green, mega-cap bucket red -- must be MIXED, not silently averaged into a
    false STRONG/WEAK reading."""
    breadth = _breadth(3, 0, 0, 10, 5, 5)
    result = zd.participation_state(breadth)
    assert result["state"] == "MIXED"
    assert result["agree"] is False


def test_participation_insufficient_data():
    breadth = _breadth(0, 0, 0, 0, 0, 0)
    result = zd.participation_state(breadth)
    assert result["state"] == "INSUFFICIENT DATA"


def test_participation_does_not_double_count_correlated_buckets():
    """The regression this whole function exists to fix: 3 mega-caps up (counted once in the
    Mega-Cap bucket) must NOT also inflate the Sector bucket beyond what the sector ETFs
    themselves independently report -- verified structurally here by confirming the function
    always reduces to exactly 3 buckets (not len(mega_snaps)+len(sector_snaps)+len(index_snaps)
    individual votes), regardless of how many tickers are in each bucket."""
    small = _breadth(1, 0, 1, 0, 1, 0)
    large = _breadth(4, 0, 10, 0, 10, 0)   # same 3-bucket unanimous-green shape, more tickers
    assert zd.participation_state(small)["n_buckets"] == zd.participation_state(large)["n_buckets"] == 3
    assert zd.participation_state(small)["state"] == zd.participation_state(large)["state"] == "STRONG"


# ---------- confluence_score: consolidated participation check + tod_rel_vol preference ----------

def test_confluence_uses_participation_not_three_separate_breadth_checks():
    bias = {"raw_signed": 40.0}
    participation_strong = {"state": "STRONG", "avg_signed": 80.0, "buckets": {}, "agree": True, "n_buckets": 3, "note": ""}
    momentum = {"velocity_pct": 0.5}
    reversal = {"reversal_pressure_score": 20.0}
    snap = {"pct_from_vwap": 0.2, "rel_volume": 1.5}
    result = zd.confluence_score(bias, participation_strong, momentum, reversal, snap)
    labels = [label for label, _ in result["checks"]]
    assert "Participation agrees" in labels
    assert "Breadth agrees" not in labels
    assert "Sector Health agrees" not in labels
    assert "Mega Cap Health agrees" not in labels
    assert result["total"] == 6   # was 8 before the fix (no alignment passed here)


def test_confluence_prefers_time_of_day_rvol_when_available():
    bias = {"raw_signed": 40.0}
    participation = {"state": "MIXED", "avg_signed": 0.0, "buckets": {}, "agree": False, "n_buckets": 3, "note": ""}
    momentum = None
    reversal = {"reversal_pressure_score": 50.0}
    # snap's crude rel_volume says "weak" (0.5x) but tod_rel_vol says "strong" (1.4x) -- the
    # confluence check must follow tod_rel_vol when it's available, not the crude fallback.
    snap = {"pct_from_vwap": 0.0, "rel_volume": 0.5}
    tod_rel_vol = {"ratio": 1.4}
    result = zd.confluence_score(bias, participation, momentum, reversal, snap, None, tod_rel_vol)
    rvol_check = dict(result["checks"])["Relative Volume confirms conviction"]
    assert rvol_check is True


# ---------- Phase 2 regression: market_bias must never emit a trade directive ----------

def test_market_bias_never_emits_trade_directives():
    """Regression test for the state-architecture review's Phase 2: market_bias()'s direction
    read must describe MARKET DIRECTION only -- 'recommendation' is gone as a key, and none of
    the old CALLS/PUTS-as-instruction strings can appear as a value."""
    snap = {"tech": {"ema_aligned_bull": True, "ema_aligned_bear": False, "rsi14": 65,
                     "sma200": 90.0, "close": 100.0}, "pct_from_vwap": 0.4}
    breadth = {"score_signed": 60.0}
    vix_snap = {"day_change_pct": -1.0}
    bias = zd.market_bias(snap, breadth, vix_snap)
    assert "recommendation" not in bias
    assert "direction_label" in bias
    forbidden = {"CALLS ONLY", "PUTS ONLY", "CALLS FAVORED", "PUTS FAVORED"}
    assert bias["direction_label"] not in forbidden
    assert bias["direction_label"] in {"Strong Bullish", "Bullish", "Bearish", "Strong Bearish",
                                       "No Clear Edge"}


def test_market_bias_direction_label_matches_sign():
    bullish_snap = {"tech": {"ema_aligned_bull": True, "ema_aligned_bear": False, "rsi14": 70,
                             "sma200": 80.0, "close": 100.0}, "pct_from_vwap": 0.5}
    bearish_snap = {"tech": {"ema_aligned_bull": False, "ema_aligned_bear": True, "rsi14": 30,
                             "sma200": 120.0, "close": 100.0}, "pct_from_vwap": -0.5}
    breadth = {"score_signed": 0.0}
    bull_bias = zd.market_bias(bullish_snap, breadth, None)
    bear_bias = zd.market_bias(bearish_snap, breadth, None)
    assert bull_bias["raw_signed"] > 0
    assert "Bullish" in bull_bias["direction_label"]
    assert bear_bias["raw_signed"] < 0
    assert "Bearish" in bear_bias["direction_label"]


# ---------- data_quality_snapshot: freshness handling (test area 11) ----------

def test_data_quality_fresh_intraday():
    now = pd.Timestamp.now(tz="America/New_York")
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=[now - pd.Timedelta(minutes=2), now])
    result = zd.data_quality_snapshot(df, chain={"calls": None}, tf_snapshot={})
    assert result["underlying"] == "Fresh"


def test_data_quality_stale_intraday():
    now = pd.Timestamp.now(tz="America/New_York")
    old = now - pd.Timedelta(hours=20)
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=[old - pd.Timedelta(minutes=1), old])
    result = zd.data_quality_snapshot(df, chain=None, tf_snapshot={})
    assert result["underlying"] == "Stale"
    assert result["options_available"] is False


def test_data_quality_no_intraday():
    result = zd.data_quality_snapshot(None, None, {})
    assert result["underlying"] == "Unknown"


# ---------- explain_transition: What Changed? (test area 10, part 1 — pure function) ----------

def test_explain_transition_matches_user_worked_example():
    prev = {"5m": "Bullish", "15m": "Bullish", "30m": "Bullish", "Daily": "Bullish",
           "vwap_side": "Above", "vwap_distance_pct": 0.31, "vwap_crossings": 1,
           "trend_integrity": 72.0, "trend_efficiency": 68.0, "reversal_pressure": 20.0,
           "participation_state": "STRONG"}
    now = {"5m": "Bearish", "15m": "Bullish", "30m": "Bullish", "Daily": "Bullish",
          "vwap_side": "Above", "vwap_distance_pct": 0.08, "vwap_crossings": 3,
          "trend_integrity": 38.0, "trend_efficiency": 40.0, "reversal_pressure": 58.0,
          "participation_state": "MODERATE"}
    result = zd.explain_transition(prev, now, "BULL CONFIRMED", "TREND WEAKENING")
    assert result["primary"] == "5m timeframe flipped Bullish → Bearish"
    assert any("VWAP distance fell" in s for s in result["supporting"])
    assert any("Reversal Pressure increased" in s for s in result["supporting"])
    assert "pullback" in result["interpretation"].lower()
    assert "bullish" in result["interpretation"].lower()


def test_explain_transition_no_prior_detail():
    result = zd.explain_transition(None, {"5m": "Bullish"}, None, "BULL DEVELOPING")
    assert result["primary"] is None
    assert result["supporting"] == []


def test_explain_transition_only_facts_no_fabrication():
    """Every 'supporting' line must trace back to an actual before/after field that changed --
    if NOTHING changed between two detail snapshots, there must be no primary and no supporting
    lines (never invent a reason when nothing measured actually moved)."""
    same = {"5m": "Bullish", "15m": "Bullish", "30m": "Bullish", "Daily": "Bullish",
           "vwap_side": "Above", "vwap_distance_pct": 0.2, "vwap_crossings": 1,
           "trend_integrity": 70.0, "trend_efficiency": 70.0, "reversal_pressure": 20.0,
           "participation_state": "STRONG"}
    result = zd.explain_transition(same, dict(same), "BULL DEVELOPING", "BULL CONFIRMED")
    assert result["primary"] is None
    assert result["supporting"] == []


# ---------- explain_transition: the live-data-review follow-up fix ----------
# A real SPY session showed transitions with no timeframe flip and no >=5-point integrity/
# reversal move -- driven instead by the hysteresis/alignment machinery, which wasn't in the
# diffed detail bundle at all. These tests cover the 3 new detection tiers added for that, plus
# the guaranteed to_reasons fallback.

def _base_detail(**overrides):
    d = {"5m": "Bearish", "15m": "Bullish", "30m": "Bullish", "Daily": "Bullish",
        "vwap_side": "Above", "vwap_distance_pct": 0.1, "vwap_crossings": 2,
        "trend_integrity": 20.0, "trend_efficiency": 30.0, "reversal_pressure": 25.0,
        "participation_state": "MIXED", "alignment_pct": 50.0,
        "trend_state_label": "Range / Chop", "trend_age_minutes": 3.0}
    d.update(overrides)
    return d


def test_explain_transition_alignment_crosses_hysteresis_threshold():
    """Real case found live: alignment_pct crossing 60% (the exact threshold
    timeframe.update_trend_state() uses) with no timeframe flip and no big integrity/reversal
    move must still produce a primary driver."""
    prev = _base_detail(alignment_pct=55.0)
    now = _base_detail(alignment_pct=62.0, trend_integrity=22.0, reversal_pressure=23.0)  # < 5pt moves
    # no timeframe flip between prev/now (5m/15m/30m/Daily identical to _base_detail defaults)
    result = zd.explain_transition(prev, now, "NEUTRAL / CHOP", "REGIME TRANSITION")
    assert result["primary"] is not None
    assert "60%" in result["primary"]
    assert "55" in result["primary"] and "62" in result["primary"]


def test_explain_transition_trend_state_label_change():
    prev = _base_detail(trend_state_label="Range / Chop", trend_integrity=22.0, reversal_pressure=23.0)
    now = _base_detail(trend_state_label="Uptrend", trend_integrity=24.0, reversal_pressure=22.0)
    result = zd.explain_transition(prev, now, "NEUTRAL / CHOP", "BULL DEVELOPING")
    assert result["primary"] == "Trend State changed from Range / Chop to Uptrend"


def test_explain_transition_trend_age_crosses_confirmation_threshold():
    prev = _base_detail(trend_age_minutes=8.0, trend_integrity=48.0)
    now = _base_detail(trend_age_minutes=11.0, trend_integrity=49.0)   # < 5pt integrity move
    result = zd.explain_transition(prev, now, "BULL DEVELOPING", "BULL CONFIRMED")
    assert result["primary"] is not None
    assert "10-minute" in result["primary"]


def test_explain_transition_guaranteed_fallback_to_day_regime_reasons():
    """The real gap found live: when NOTHING diffable moved enough, explain_transition() must
    still say something grounded in day_regime()'s own real reasons rather than a useless
    generic pointer -- this is the guaranteed fallback, exercised with to_reasons provided."""
    prev = _base_detail()
    now = _base_detail()   # byte-identical -- nothing at all moved
    result = zd.explain_transition(prev, now, "BEAR DEVELOPING", "TREND WEAKENING",
                                   to_reasons=["Trend Integrity 20/100", "Reversal Pressure 25/100"])
    assert result["primary"] == "Trend Integrity 20/100; Reversal Pressure 25/100"
    assert "Trend Integrity 20/100" in result["interpretation"]


def test_explain_transition_degrades_honestly_without_reasons_fallback():
    """Without to_reasons and with nothing diffable, must degrade honestly -- never fabricate."""
    prev = _base_detail()
    now = _base_detail()
    result = zd.explain_transition(prev, now, "BEAR DEVELOPING", "TREND WEAKENING")
    assert result["primary"] is None
    assert "no individual measurement crossed" in result["interpretation"]
