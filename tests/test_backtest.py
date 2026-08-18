"""Tests for iip/backtest.py — point-in-time slicing (test area 9), with an explicit
future-data-leakage regression test. PIIP audit 2026-08, state-architecture review, Phase 8.

`run_backtest()` normally fetches live data via yfinance -- monkeypatched here (`backtest._fetch`)
to a fully synthetic, deterministic price series so the test is reproducible offline and doesn't
spend a real network call. Uses throwaway sqlite db files in the OS temp dir, never the real
databases.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from iip import backtest
from iip import predictions as pred


def _synthetic_full(n_days: int = 400, base: float = 100.0, perturb_after: int | None = None,
                    perturb_amount: float = 500.0) -> pd.DataFrame:
    """A deterministic daily OHLCV series. If `perturb_after` is set, every close AFTER that row
    index is shifted by `perturb_amount` -- used to build two otherwise-identical series that
    diverge only in the "future" relative to some cutoff, for the leakage regression test below."""
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    closes = [base + i * 0.1 + 2 * math.sin(i / 9) for i in range(n_days)]
    if perturb_after is not None:
        closes = [c + (perturb_amount if i > perturb_after else 0.0) for i, c in enumerate(closes)]
    closes = pd.Series(closes, index=idx)
    opens = closes.shift(1).fillna(closes.iloc[0])
    highs = pd.concat([opens, closes], axis=1).max(axis=1) + 0.2
    lows = pd.concat([opens, closes], axis=1).min(axis=1) - 0.2
    volume = pd.Series([300_000] * n_days, index=idx)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                         "Volume": volume})


@pytest.fixture
def tmp_db_path():
    path = os.path.join(tempfile.gettempdir(), f"test_piip_backtest_{uuid.uuid4().hex}.db")
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            pass


def test_point_in_time_slice_never_includes_future_rows():
    """Direct test of the slicing PATTERN backtest.py's own loop uses (`full.loc[:Dt]`) -- the
    actual leakage-prevention mechanism the module's docstring claims. Any row after the as-of
    date Dt must be absent from the point-in-time slice."""
    full = _synthetic_full(n_days=50)
    Dt = full.index[30]
    prices_pit = full.loc[:Dt]
    assert prices_pit.index.max() == Dt
    assert (prices_pit.index <= Dt).all()
    assert len(prices_pit) == 31   # rows 0..30 inclusive


def test_run_backtest_future_perturbation_does_not_change_past_resolved_outcomes(
        monkeypatch, tmp_db_path):
    """THE flagship leakage regression test: two runs of run_backtest() against price series that
    are IDENTICAL up through day 300 but diverge sharply afterward. For an observation made (and
    resolved) well before day 300, the two runs must produce byte-identical logged spot/realized
    return values -- proving the perturbed future never leaked backward into an earlier,
    already-resolved observation. If run_backtest() ever accidentally used `full` (the whole
    series) instead of `prices_pit` (the point-in-time slice) for ANY part of the forecast or
    resolution math, this test would catch it."""
    series_a = _synthetic_full(n_days=400, perturb_after=None)
    series_b = _synthetic_full(n_days=400, perturb_after=300, perturb_amount=1000.0)

    # Sanity: the two series really are identical up to day 300 and really do diverge after.
    assert (series_a["Close"].iloc[:300] == series_b["Close"].iloc[:300]).all()
    assert not (series_a["Close"].iloc[301:] == series_b["Close"].iloc[301:]).any()

    def make_fetch(series):
        def _fetch(tk, period="3y"):
            return series.copy()
        return _fetch

    db_a = tmp_db_path
    db_b = tmp_db_path + ".b"
    try:
        monkeypatch.setattr(backtest, "_fetch", make_fetch(series_a))
        # Short horizon + early stop so the earliest as-of dates (well before day 300) fully
        # resolve using only data that's identical between series_a and series_b.
        backtest.run_backtest(["TEST"], horizons=(7,), step_days=60, db=db_a)
        rows_a = {r["ts"]: dict(r) for r in pred.all_rows(db_a)}

        monkeypatch.setattr(backtest, "_fetch", make_fetch(series_b))
        backtest.run_backtest(["TEST"], horizons=(7,), step_days=60, db=db_b)
        rows_b = {r["ts"]: dict(r) for r in pred.all_rows(db_b)}

        assert rows_a, "expected at least one logged prediction"
        common_ts = set(rows_a) & set(rows_b)
        assert common_ts, "expected the same as-of timestamps in both runs (same synthetic series shape)"

        # Every prediction whose ENTIRE observation+resolution window falls before the day-300
        # perturbation point must be byte-identical across the two runs.
        checked_any = False
        for ts in common_ts:
            a, b = rows_a[ts], rows_b[ts]
            obs_date = pd.Timestamp(ts).tz_localize(None)
            cutoff_date = series_a.index[295]   # comfortable margin before the perturbation at 300
            if obs_date >= cutoff_date:
                continue
            checked_any = True
            assert a["spot"] == b["spot"], f"spot price leaked at {ts}"
            assert a["realized_return_pct"] == b["realized_return_pct"], f"realized return leaked at {ts}"
            assert a["spy_return_pct"] == b["spy_return_pct"], f"SPY return leaked at {ts}"
        assert checked_any, "test setup produced no comparable pre-cutoff observations -- adjust step_days"
    finally:
        for p in (db_a, db_b):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except PermissionError:
                    pass


def test_run_backtest_llm_excluded(monkeypatch, tmp_db_path):
    """The module's own docstring states the LLM is deliberately excluded from backtesting
    (its training cutoff would itself be look-ahead). Regression-test that every row
    run_backtest() logs has source='deterministic', never 'llm'."""
    series = _synthetic_full(n_days=400)
    monkeypatch.setattr(backtest, "_fetch", lambda tk, period="3y": series.copy())
    backtest.run_backtest(["TEST"], horizons=(7,), step_days=90, db=tmp_db_path)
    rows = pred.all_rows(tmp_db_path)
    assert rows, "expected at least one logged prediction"
    assert all(r["source"] == "deterministic" for r in rows)
