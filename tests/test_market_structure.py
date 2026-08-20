"""Tests for iip/market_structure.py -- the 0DTE Market Structure Map. Synthetic, deterministic
fixtures throughout (per the spec's own explicit 'do not rely only on today's market data for
tests' requirement), covering POC/VAH/VAL, HVN/LVN, clustering (including the exact chaining bug
caught during this build), acceptance/rejection state, conditional paths (including the exact
circular-reference bug caught during this build), and missing/stale-data handling.
PIIP audit 2026-08, 0DTE Market Structure Map.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from iip import market_structure as ms

ET = ZoneInfo("America/New_York")
TODAY = date.today()


def _bars(rows: list[tuple], start: dtime = dtime(9, 30)) -> pd.DataFrame:
    """rows: list of (open, high, low, close, volume) tuples, one per minute starting at `start`
    today. Builds a real OHLCV DataFrame with a proper tz-aware DatetimeIndex, matching the shape
    every function in market_structure.py expects."""
    t = datetime.combine(TODAY, start, tzinfo=ET)
    idx, data = [], {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
    for o, h, l, c, v in rows:
        idx.append(t)
        data["Open"].append(o); data["High"].append(h); data["Low"].append(l)
        data["Close"].append(c); data["Volume"].append(v)
        t += timedelta(minutes=1)
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


def _flat_bars(n: int, price: float = 100.0, vol: float = 1000.0) -> pd.DataFrame:
    return _bars([(price, price, price, price, vol)] * n)


# ---------- _clean_bars: the real live glitch caught during this build ----------

def test_clean_bars_drops_implausible_wick():
    """The exact regression this build's own live verification caught: a post-market bar with a
    Low far below its own Open/Close (a real yfinance glitch, not a hypothetical)."""
    df = _bars([(100.0, 100.0, 100.0, 100.0, 1000.0)] * 10 +
              [(769.09, 769.10, 732.05, 769.09, 0.0)] +   # the real glitch shape
              [(100.0, 100.0, 100.0, 100.0, 1000.0)] * 10)
    cleaned = ms._clean_bars(df)
    assert 732.05 not in cleaned["Low"].values
    assert len(cleaned) == 20


def test_clean_bars_keeps_normal_wicks():
    df = _bars([(100.0, 100.5, 99.7, 100.2, 1000.0)] * 10)
    cleaned = ms._clean_bars(df)
    assert len(cleaned) == 10


# ---------- volume_profile: POC/VAH/VAL ----------

def test_volume_profile_poc_lands_on_concentrated_volume():
    """Most volume concentrated in a narrow band around 105 -- POC must land there, not at the
    session's arbitrary high/low."""
    rows = [(100.0, 100.2, 99.8, 100.0, 100.0) for _ in range(20)]
    rows += [(105.0, 105.2, 104.8, 105.0, 50000.0) for _ in range(20)]   # heavy concentration
    rows += [(110.0, 110.2, 109.8, 110.0, 100.0) for _ in range(20)]
    profile = ms.volume_profile(_bars(rows), n_bins=20)
    assert profile["status"] == "OK"
    assert 104.5 <= profile["poc"] <= 105.5
    assert profile["val"] <= profile["poc"] <= profile["vah"]


def test_volume_profile_insufficient_data_too_few_bars():
    profile = ms.volume_profile(_flat_bars(5))
    assert profile["status"] == "INSUFFICIENT_DATA"


def test_volume_profile_insufficient_data_none_input():
    assert ms.volume_profile(None)["status"] == "INSUFFICIENT_DATA"


def test_volume_profile_value_area_captures_target_pct():
    rows = [(100.0 + i * 0.1, 100.2 + i * 0.1, 99.8 + i * 0.1, 100.0 + i * 0.1, 1000.0)
           for i in range(30)]
    profile = ms.volume_profile(_bars(rows), value_area_pct=0.70)
    assert profile["status"] == "OK"
    assert profile["value_area_pct_actual"] >= 0.68   # close to target, allowing bin-edge slack


# ---------- detect_hvn_lvn ----------

def test_detect_hvn_lvn_finds_peak_and_trough():
    rows = [(100.0, 100.2, 99.8, 100.0, 500.0) for _ in range(15)]     # baseline
    rows += [(103.0, 103.2, 102.8, 103.0, 20000.0) for _ in range(15)]  # HVN
    rows += [(106.0, 106.2, 105.8, 106.0, 10.0) for _ in range(15)]     # LVN
    rows += [(109.0, 109.2, 108.8, 109.0, 500.0) for _ in range(15)]    # baseline
    profile = ms.volume_profile(_bars(rows), n_bins=30)
    hvn_lvn = ms.detect_hvn_lvn(profile)
    assert hvn_lvn["status"] == "OK"
    assert len(hvn_lvn["hvns"]) >= 1
    assert any(102.5 <= h["price"] <= 103.5 for h in hvn_lvn["hvns"])


def test_detect_hvn_lvn_insufficient_when_profile_insufficient():
    result = ms.detect_hvn_lvn({"status": "INSUFFICIENT_DATA"})
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["hvns"] == [] and result["lvns"] == []


# ---------- swing_points ----------

def test_swing_points_detects_clean_high_and_low():
    # 5 flat, a spike up (swing high), 5 flat, a dip down (swing low), 5 flat -- lookback=5
    # default. Wick sizes kept under _clean_bars()'s own MAX_WICK_PCT (a real 1-minute SPY-scale
    # bar's wick this large IS what that filter exists to catch -- an earlier version of this
    # test used a 5% synthetic spike and _clean_bars() correctly dropped it as implausible data,
    # same as it would a real glitch).
    rows = [(100.0, 100.0, 100.0, 100.0, 100.0)] * 5
    rows += [(100.0, 100.3, 100.0, 100.0, 100.0)]
    rows += [(100.0, 100.0, 100.0, 100.0, 100.0)] * 5
    rows += [(100.0, 100.0, 99.7, 100.0, 100.0)]
    rows += [(100.0, 100.0, 100.0, 100.0, 100.0)] * 5
    result = ms.swing_points(_bars(rows), lookback=5)
    assert result["status"] == "OK"
    assert 100.3 in result["swing_highs"]
    assert 99.7 in result["swing_lows"]


def test_swing_points_insufficient_data():
    assert ms.swing_points(_flat_bars(3), lookback=5)["status"] == "INSUFFICIENT_DATA"


# ---------- intraday_ema_sma ----------

def test_intraday_ema_sma_on_constant_series_equals_the_constant():
    ema, sma = ms.intraday_ema_sma(_flat_bars(60, price=250.0), span=50)
    assert ema == pytest.approx(250.0, abs=0.01)
    assert sma == pytest.approx(250.0, abs=0.01)


def test_intraday_ema_sma_none_when_no_data():
    ema, sma = ms.intraday_ema_sma(None)
    assert ema is None and sma is None


# ---------- collect_levels ----------

def test_collect_levels_gathers_all_non_none():
    levels = ms.collect_levels(vwap=100.0, ema50=101.0, sma50=102.0,
                               opening_range={"high": 103.0, "low": 99.0},
                               prev_day={"high": 104.0, "low": 98.0, "close": 100.5},
                               profile=None, hvn_lvn=None, swings=None)
    names = {l["name"] for l in levels}
    assert names == {"VWAP", "EMA50", "SMA50", "Opening Range High", "Opening Range Low",
                     "Prev Day High", "Prev Day Low", "Prev Day Close"}


def test_collect_levels_skips_none_and_nan():
    levels = ms.collect_levels(vwap=None, ema50=float("nan"), sma50=100.0,
                               opening_range=None, prev_day=None, profile=None,
                               hvn_lvn=None, swings=None)
    assert len(levels) == 1
    assert levels[0]["name"] == "SMA50"


# ---------- cluster_levels: the exact chaining bug caught during this build ----------

def test_cluster_levels_merges_close_levels():
    levels = [{"name": "VWAP", "price": 773.72}, {"name": "MA50", "price": 773.79},
             {"name": "HVN", "price": 773.83}]
    zones = ms.cluster_levels(levels, tolerance_pct=0.03)
    assert len(zones) == 1
    assert zones[0]["n_factors"] == 3
    assert set(zones[0]["contributors"]) == {"VWAP", "MA50", "HVN"}


def test_cluster_levels_does_not_chain_beyond_tolerance():
    """The exact regression this build's own live verification caught: A close to B, B close to
    C, but A far from C -- must NOT all merge into one zone via single-linkage chaining (a real
    bug this project shipped and fixed before release: a $1.58-wide, 13-factor blob)."""
    # tolerance_pct=0.1% of ~770 is ~$0.77. Seed=770.0; 770.7 is within tolerance of the seed
    # (0.09%) but 771.4 is NOT (0.18% from the seed) -- must split into two zones, not chain.
    levels = [{"name": "A", "price": 770.0}, {"name": "B", "price": 770.7},
             {"name": "C", "price": 771.4}]
    zones = ms.cluster_levels(levels, tolerance_pct=0.1)
    assert len(zones) == 2
    assert zones[0]["contributors"] == ["A", "B"]
    assert zones[1]["contributors"] == ["C"]


def test_cluster_levels_empty_input():
    assert ms.cluster_levels([]) == []


# ---------- classify_zones ----------

def test_classify_zones_splits_by_spot_and_limits_count():
    zones = [{"low": p - 0.1, "high": p + 0.1, "mid": p, "contributors": ["X"], "n_factors": 1}
            for p in [95, 97, 99, 101, 103, 105]]
    result = ms.classify_zones(zones, spot=100.0, max_each=2)
    assert len(result["resistance"]) == 2
    assert len(result["support"]) == 2
    assert all(z["mid"] > 100.0 for z in result["resistance"])
    assert all(z["mid"] < 100.0 for z in result["support"])
    # nearest-first
    assert result["resistance"][0]["mid"] == 101
    assert result["support"][0]["mid"] == 99


# ---------- zone_state: every state, engineered deterministically ----------

def _zone(low, high):
    return {"low": low, "high": high, "mid": (low + high) / 2, "contributors": ["X"], "n_factors": 1}


def test_zone_state_acceptance_above_requires_two_consecutive_closes():
    zone = _zone(100.0, 101.0)
    # 2 consecutive closes above the zone
    df = _bars([(102.0, 102.0, 102.0, 102.5, 100.0), (102.5, 102.5, 102.5, 103.0, 100.0)])
    result = ms.zone_state(zone, df, "resistance", confirm_bars=2)
    assert result["state"] == "ACCEPTANCE_ABOVE"


def test_zone_state_breakout_single_close_not_yet_accepted():
    zone = _zone(100.0, 101.0)
    df = _bars([(99.0, 99.0, 99.0, 99.5, 100.0), (99.5, 102.5, 99.5, 102.5, 100.0)])
    result = ms.zone_state(zone, df, "resistance", confirm_bars=2)
    assert result["state"] == "BREAKOUT"


def test_zone_state_rejected_after_testing_then_closing_back_below():
    zone = _zone(100.0, 101.0)
    df = _bars([(99.0, 99.0, 99.0, 99.5, 100.0), (99.5, 100.5, 99.5, 100.5, 100.0),
               (100.5, 100.5, 99.0, 99.2, 100.0)])
    result = ms.zone_state(zone, df, "resistance", confirm_bars=2)
    assert result["state"] == "REJECTED"


def test_zone_state_bounce_for_support_side():
    zone = _zone(100.0, 101.0)
    df = _bars([(102.0, 102.0, 102.0, 102.0, 100.0), (102.0, 102.0, 100.5, 100.5, 100.0),
               (100.5, 102.5, 100.5, 102.5, 100.0)])
    result = ms.zone_state(zone, df, "support", confirm_bars=2)
    assert result["state"] == "BOUNCE"


def test_zone_state_testing_when_currently_inside():
    zone = _zone(100.0, 101.0)
    df = _bars([(99.0, 99.0, 99.0, 99.5, 100.0), (99.5, 100.5, 99.5, 100.5, 100.0)])
    result = ms.zone_state(zone, df, "resistance", confirm_bars=2)
    assert result["state"] == "TESTING"


def test_zone_state_approaching_when_close_but_untouched():
    zone = _zone(100.0, 101.0)
    df = _bars([(99.7, 99.7, 99.7, 99.72, 100.0)])   # ~0.28% from the zone's low (100.0)
    result = ms.zone_state(zone, df, "resistance", confirm_bars=2, approach_pct=0.3)
    assert result["state"] == "APPROACHING"


def test_zone_state_neutral_when_far_away():
    zone = _zone(100.0, 101.0)
    df = _bars([(50.0, 50.0, 50.0, 50.0, 100.0)])
    result = ms.zone_state(zone, df, "resistance", confirm_bars=2)
    assert result["state"] == "NEUTRAL"


def test_zone_state_unknown_no_data():
    zone = _zone(100.0, 101.0)
    result = ms.zone_state(zone, None, "resistance")
    assert result["state"] == "UNKNOWN"


# ---------- conditional_path: the exact circular-reference bug caught during this build ----------

def test_conditional_path_finds_correct_neighbors():
    zones = [_zone(90, 91), _zone(95, 96), _zone(105, 106), _zone(110, 111)]
    zone = zones[2]   # 105-106, a resistance zone with spot below it
    path = ms.conditional_path(zone, "resistance", zones, spot=100.0)
    assert path["if_accepted"]["mid"] == 110.5
    assert path["if_rejected"]["mid"] == 95.5


def test_conditional_path_returns_snapshots_not_live_references():
    """The exact regression this build's own live verification caught: attaching state/path to
    BOTH a resistance and a support zone that reference each other as their nearest neighbor
    created a genuine circular reference (ValueError on json.dumps). Snapshots must be inert --
    adding new keys to the ORIGINAL zone dict after the fact must not appear in the snapshot."""
    zones = [_zone(95, 96), _zone(105, 106)]
    resistance_zone, support_zone = zones[1], zones[0]
    r_path = ms.conditional_path(resistance_zone, "resistance", zones, spot=100.0)
    s_path = ms.conditional_path(support_zone, "support", zones, spot=100.0)
    # Mutate the ORIGINAL zone dicts the way build_structure_map() does -- must not leak into
    # the already-returned snapshots.
    resistance_zone["state"] = {"state": "TESTING"}
    resistance_zone["path"] = r_path
    support_zone["state"] = {"state": "TESTING"}
    support_zone["path"] = s_path
    assert "state" not in s_path["if_rejected"]
    assert "path" not in s_path["if_rejected"]
    # Must be fully JSON-serializable (this IS the regression test for the circular-reference bug).
    json.dumps({"r": r_path, "s": s_path})


def test_conditional_path_none_when_no_further_zone():
    zones = [_zone(105, 106)]
    path = ms.conditional_path(zones[0], "resistance", zones, spot=100.0)
    assert path["if_accepted"] is None
    assert path["if_rejected"] is None


# ---------- expected_range ----------

def test_expected_range_prefers_straddle_over_iv():
    om = {"expected_move_straddle_pct": 1.0, "expected_move_iv_pct": 1.5}
    result = ms.expected_range(100.0, om)
    assert result["status"] == "OK"
    assert result["method"] == "straddle"
    assert result["expected_high"] == pytest.approx(101.0)
    assert result["expected_low"] == pytest.approx(99.0)


def test_expected_range_falls_back_to_iv():
    om = {"expected_move_straddle_pct": None, "expected_move_iv_pct": 2.0}
    result = ms.expected_range(100.0, om)
    assert result["method"] == "iv"
    assert result["expected_high"] == pytest.approx(102.0)


def test_expected_range_insufficient_data_no_metrics():
    assert ms.expected_range(100.0, None)["status"] == "INSUFFICIENT_DATA"
    assert ms.expected_range(None, {"expected_move_straddle_pct": 1.0})["status"] == "INSUFFICIENT_DATA"


# ---------- stabilize_expected_range ----------

def test_stabilize_expected_range_ignores_small_change():
    prev = {"status": "OK", "expected_high": 105.0, "expected_low": 95.0}
    new = {"status": "OK", "expected_high": 105.01, "expected_low": 95.01}
    result, updated = ms.stabilize_expected_range(new, prev, min_change_pct=0.05)
    assert updated is False
    assert result == prev


def test_stabilize_expected_range_accepts_big_change():
    prev = {"status": "OK", "expected_high": 105.0, "expected_low": 95.0}
    new = {"status": "OK", "expected_high": 108.0, "expected_low": 92.0}
    result, updated = ms.stabilize_expected_range(new, prev, min_change_pct=0.05)
    assert updated is True
    assert result == new


def test_stabilize_expected_range_first_call_always_updates():
    new = {"status": "OK", "expected_high": 105.0, "expected_low": 95.0}
    result, updated = ms.stabilize_expected_range(new, None)
    assert updated is True
    assert result == new


# ---------- build_structure_map: end-to-end, JSON-serializable ----------

def test_build_structure_map_end_to_end_is_json_serializable():
    rows = [(100.0 + (i % 5) * 0.05, 100.3 + (i % 5) * 0.05, 99.7 + (i % 5) * 0.05,
            100.0 + (i % 5) * 0.05, 1000.0) for i in range(60)]
    df = _bars(rows)
    or15 = {"high": 100.5, "low": 99.5, "last": 100.1, "status": "Inside range"}
    prev_day = {"high": 101.0, "low": 98.0, "close": 99.8}
    om = {"expected_move_straddle_pct": 1.2, "expected_move_iv_pct": 1.4,
         "expiry": TODAY.isoformat(), "dte_days": 1}
    result = ms.build_structure_map(df, spot=100.1, vwap=99.95, opening_range=or15,
                                    prev_day=prev_day, option_metrics=om)
    assert result["spot"] == 100.1
    assert result["expected_range"]["status"] == "OK"
    json.dumps(result, default=str)   # must never raise -- the circular-reference regression


def test_build_structure_map_handles_missing_data_gracefully():
    result = ms.build_structure_map(None, spot=None, vwap=None, opening_range=None,
                                    prev_day=None, option_metrics=None)
    assert result["volume_profile"]["status"] == "INSUFFICIENT_DATA"
    assert result["expected_range"]["status"] == "INSUFFICIENT_DATA"
    assert result["zones"] == {"resistance": [], "support": []}
    json.dumps(result, default=str)
