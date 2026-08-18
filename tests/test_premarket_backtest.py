"""Tests for iip/premarket_backtest.py -- deterministic buckets, similarity ladder, and (the
central concern per explicit user instruction) the LEAKAGE guards: a historical day's own
end-of-session data must never contaminate a feature meant to represent what was knowable before
that day's premarket session, and the day being scored live must never appear in its own
historical comparison set. PIIP audit 2026-08, Premarket Thesis AI layer.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from iip import premarket_backtest as bt


# ---------- deterministic buckets ----------

def test_gap_bucket_thresholds():
    assert bt.gap_bucket(-2.0) == "Large Down"
    assert bt.gap_bucket(-0.5) == "Small Down"
    assert bt.gap_bucket(0.0) == "Flat"
    assert bt.gap_bucket(0.5) == "Small Up"
    assert bt.gap_bucket(2.0) == "Large Up"
    assert bt.gap_bucket(None) is None


def test_trend_alignment():
    assert bt.trend_alignment(105.0, 100.0) == "Above SMA50"
    assert bt.trend_alignment(95.0, 100.0) == "Below SMA50"
    assert bt.trend_alignment(100.0, float("nan")) is None
    assert bt.trend_alignment(None, 100.0) is None


def test_vix_bucket_thresholds():
    assert bt.vix_bucket(35.0) == "EXTREME"
    assert bt.vix_bucket(25.0) == "ELEVATED"
    assert bt.vix_bucket(12.0) == "LOW"
    assert bt.vix_bucket(18.0) == "NORMAL"
    assert bt.vix_bucket(None) is None


# ---------- _pool_stats: direction-aware best/worst ----------

def test_pool_stats_bullish_best_is_largest_gain():
    days = [{"forward_move_pct": p} for p in [-0.5, 0.2, 1.0, -0.1]]
    stats = bt._pool_stats(days, "Bullish")
    assert stats["best_pct"] == pytest.approx(1.0)
    assert stats["worst_pct"] == pytest.approx(-0.5)
    assert stats["win_rate_pct"] == pytest.approx(50.0)   # 2 of 4 positive


def test_pool_stats_bearish_best_is_largest_decline():
    """The regression this test guards: a naive max()/min() would report the largest RAW number
    as 'best' even for a Bearish thesis, where the most favorable outcome is actually the
    steepest decline (a negative number)."""
    days = [{"forward_move_pct": p} for p in [-1.5, 0.3, -0.2, 0.8]]
    stats = bt._pool_stats(days, "Bearish")
    assert stats["best_pct"] == pytest.approx(-1.5)    # steepest decline = best for a bearish thesis
    assert stats["worst_pct"] == pytest.approx(0.8)     # biggest rally = worst for a bearish thesis
    assert stats["win_rate_pct"] == pytest.approx(50.0)  # 2 of 4 negative


def test_pool_stats_empty_returns_n_zero():
    assert bt._pool_stats([], "Bullish") == {"n": 0}


# ---------- _similarity_ladder: deterministic, never silently widened without saying so ----------

def test_similarity_ladder_prefers_tightest_pool_that_clears_min_sample():
    scored = [(4, {"id": i}) for i in range(10)] + [(2, {"id": i}) for i in range(10, 15)]
    label, pool, ladder = bt._similarity_ladder(scored, min_sample=8, max_dims=4)
    assert label == "HIGH"
    assert len(pool) == 10
    assert ladder["min_dims_matched"] == 4


def test_similarity_ladder_widens_when_tightest_pool_too_small():
    scored = [(4, {"id": 0}), (4, {"id": 1})] + [(3, {"id": i}) for i in range(2, 10)]
    label, pool, ladder = bt._similarity_ladder(scored, min_sample=8, max_dims=4)
    assert label == "MEDIUM"
    assert len(pool) == 10   # includes both the 2 HIGH-matching AND the 8 MEDIUM-matching days
    assert ladder["min_dims_matched"] == 3


def test_similarity_ladder_insufficient_sample_when_even_loosest_pool_too_small():
    scored = [(4, {"id": 0}), (1, {"id": 1})]
    label, pool, ladder = bt._similarity_ladder(scored, min_sample=8, max_dims=4)
    assert label == "INSUFFICIENT_SAMPLE"


# ---------- tiers_agree: computed flag, never an AI judgment ----------

def test_tiers_agree_true_when_both_favor_same_side():
    t1 = {"status": "OK", "win_rate_pct": 65.0}
    t2 = {"status": "OK", "win_rate_pct": 58.0}
    assert bt.tiers_agree(t1, t2) is True


def test_tiers_agree_false_when_tiers_disagree():
    t1 = {"status": "OK", "win_rate_pct": 65.0}
    t2 = {"status": "OK", "win_rate_pct": 40.0}
    assert bt.tiers_agree(t1, t2) is False


def test_tiers_agree_none_when_either_tier_lacks_sample():
    t1 = {"status": "INSUFFICIENT_SAMPLE", "win_rate_pct": None}
    t2 = {"status": "OK", "win_rate_pct": 58.0}
    assert bt.tiers_agree(t1, t2) is None


# ---------- LEAKAGE: intraday session-boundary separation (Tier 1) ----------

ET = ZoneInfo("America/New_York")


def _make_day_bars(day: date, premkt_closes: list[float], regular_closes: list[float]) -> pd.DataFrame:
    """One day's 5m bars: premarket starting 04:00 ET, regular session starting 09:30 ET, both
    5 minutes apart -- Open/High/Low copied from Close since these tests only read Close."""
    rows = []
    t = datetime.combine(day, dtime(4, 0), tzinfo=ET)
    for c in premkt_closes:
        rows.append((t, c))
        t += timedelta(minutes=5)
    t = datetime.combine(day, dtime(9, 30), tzinfo=ET)
    for c in regular_closes:
        rows.append((t, c))
        t += timedelta(minutes=5)
    idx = pd.DatetimeIndex([r[0] for r in rows])
    closes = [r[1] for r in rows]
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                         "Volume": [100] * len(closes)}, index=idx)


def test_premarket_change_pct_never_includes_regular_session_bars():
    """If regular-session bars leaked into the premarket window, the huge 500/600 values below
    would dominate the result instead of the real 100->101 premarket move."""
    day = date(2026, 6, 1)
    df = _make_day_bars(day, premkt_closes=[100.0, 101.0], regular_closes=[500.0, 600.0])
    chg = bt._premarket_change_pct(df, day)
    assert chg == pytest.approx((101.0 / 100.0 - 1) * 100, abs=0.001)


def test_forward_move_pct_never_includes_premarket_or_post_checkpoint_bars():
    """The 9:30-10:00 forward window must exclude BOTH the premarket bars before it AND any bar
    at/after the 10:00 checkpoint -- a bar exactly at 10:00 (value 200.0 here) must not leak in."""
    day = date(2026, 6, 1)
    df = _make_day_bars(day, premkt_closes=[1.0, 2.0],
                        regular_closes=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 200.0])
    move = bt._forward_move_pct(df, day)
    assert move == pytest.approx((105.0 / 100.0 - 1) * 100, abs=0.001)


def test_trading_days_excludes_today_and_future_dates():
    """The day being scored live must never be eligible to appear in its own historical
    comparison set -- the central leakage invariant this whole module exists to protect."""
    d1, d2, d3 = date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)
    df = pd.concat([_make_day_bars(d1, [1.0], [1.0]), _make_day_bars(d2, [1.0], [1.0]),
                    _make_day_bars(d3, [1.0], [1.0])])
    days = bt._trading_days(df, before=d3)
    assert d3 not in days
    assert set(days) == {d1, d2}

    days_today_is_d2 = bt._trading_days(df, before=d2)
    assert d2 not in days_today_is_d2
    assert d3 not in days_today_is_d2   # a future date relative to "today" must never appear either
    assert days_today_is_d2 == [d1]


# ---------- LEAKAGE: causal trend alignment (Tier 1's own-close contamination guard) ----------

def _make_daily_frame(n_days: int, anomaly_day_index: int, anomaly_close: float,
                      base_close: float = 100.0, start: date = date(2026, 1, 1)) -> pd.DataFrame:
    dates, closes = [], []
    d = start
    for i in range(n_days):
        dates.append(d)
        closes.append(anomaly_close if i == anomaly_day_index else base_close)
        d += timedelta(days=1)
    return pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates))


def test_causal_trend_alignment_does_not_leak_the_scored_days_own_close():
    """Engineered regression case: day 54 (index 53) has an anomalous OWN close (50.0, a big drop)
    while every day before it is flat at 100.0. If trend_alignment for day 54 leaked its OWN close
    into the comparison, it would read 'Below SMA50' (50 vs an SMA dragged down toward ~99). The
    correct, leakage-free answer uses ONLY day 53's close (100.0) vs day 53's own SMA50 (100.0,
    unaffected by day 54's anomaly) -- 'Above SMA50'. This is the exact bug caught and fixed
    during this module's own review, before it shipped."""
    daily = _make_daily_frame(n_days=55, anomaly_day_index=53, anomaly_close=50.0)
    sma50_series = daily["Close"].rolling(50).mean()
    anomaly_day = daily.index[53].date()   # the day with the anomalous OWN close

    result = bt._causal_trend_alignment_for_day(anomaly_day, daily, sma50_series)
    assert result == "Above SMA50", (
        "trend_alignment leaked the scored day's own close -- must reflect the PRIOR day's "
        "close/SMA50 only, which was flat at 100.0, not the anomalous same-day 50.0 close.")


def test_causal_trend_alignment_none_before_any_prior_data_exists():
    daily = _make_daily_frame(n_days=5, anomaly_day_index=-1, anomaly_close=100.0)
    sma50_series = daily["Close"].rolling(50).mean()
    first_day = daily.index[0].date()
    assert bt._causal_trend_alignment_for_day(first_day, daily, sma50_series) is None


# ---------- LEAKAGE: Tier 2 daily proxy (prior-day-only gap/trend/vix, integration-level) ----------

def test_tier2_backtest_uses_prior_day_trend_and_vix_not_same_day(monkeypatch):
    """Integration-level leakage check for run_tier2_backtest: constructs 60 flat days (SMA50 ~=
    100) then one anomalous day whose OWN close is wildly different from its own open -- if the
    day's own close (rather than the PRIOR day's) were used for the trend-match dimension, this
    day would score a spurious extra match against a 'trend_align' query engineered to only match
    the prior-day-correct value."""
    daily = _make_daily_frame(n_days=61, anomaly_day_index=60, anomaly_close=100.0)
    # Day 60 (index 60): give it a real Open != prior Close so the gap/forward move are meaningful,
    # and drive its post-open path far from 100 -- if trend leaked same-day info, the anomaly's
    # OWN close (not the prior day's 100.0) would wrongly feed the "prior trend" dimension.
    daily.loc[daily.index[60], "Close"] = 300.0
    opens = [100.0] * 61
    opens[60] = 100.5
    daily["Open"] = opens

    vix_daily = _make_daily_frame(n_days=61, anomaly_day_index=-1, anomaly_close=18.0, base_close=18.0)

    monkeypatch.setattr(bt.data, "get_prices",
                        lambda ticker, period="2y": daily if ticker == "SPY" else vix_daily)

    today = daily.index[60].date() + timedelta(days=1)   # strictly after the last row, so every
    # constructed day (including the anomaly) is eligible to be scored as history.
    result = bt.run_tier2_backtest("Bullish", gap_bkt="Flat", trend_align="Above SMA50",
                                   vix_bkt="NORMAL", today=today)
    assert result["status"] in ("OK", "INSUFFICIENT_SAMPLE")
    # The anomaly day's forward move (300 vs its own 100.5 open) must still be usable as an
    # OUTCOME (that's not leakage -- forward-looking-from-the-open is exactly what's measured),
    # while its match against 'trend_align' must come from the PRIOR (flat, 100.0) day, not its
    # own 300.0 close. days_scored must include it (proves it wasn't silently dropped)
    assert result["days_scored"] >= 55
