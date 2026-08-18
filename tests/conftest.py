"""Shared fixtures for PIIP's test suite (PIIP audit 2026-08, state-architecture review, Phase 8).

All synthetic data below is DETERMINISTIC (no randomness) -- built from plain arithmetic/sine
patterns, never `numpy.random` -- so every test is exactly reproducible run to run, and so a
failure always means a real behavior change, never a seed/flake issue.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest


def make_intraday_1m(n_bars: int = 60, start="2026-08-17 09:30", tz="America/New_York",
                     base_price: float = 100.0, pattern: str = "flat") -> pd.DataFrame:
    """A synthetic 1-minute OHLCV frame shaped like real intraday bars (DatetimeIndex, tz-aware,
    Open/High/Low/Close/Volume columns). `pattern` controls the price path:
      - "flat": no net movement, tiny alternating noise (for chop/low-efficiency tests)
      - "uptrend": monotonic rise (for clean-trend/high-efficiency tests)
      - "downtrend": monotonic fall
      - "v_reversal": falls then rises back (for VWAP-crossing/reversal tests)
    """
    idx = pd.date_range(start=start, periods=n_bars, freq="1min", tz=tz)
    closes = []
    for i in range(n_bars):
        if pattern == "flat":
            closes.append(base_price + (0.05 if i % 2 == 0 else -0.05))
        elif pattern == "uptrend":
            closes.append(base_price + i * 0.10)
        elif pattern == "downtrend":
            closes.append(base_price - i * 0.10)
        elif pattern == "v_reversal":
            half = n_bars // 2
            closes.append(base_price - i * 0.10 if i <= half else base_price - half * 0.10 + (i - half) * 0.10)
        else:
            raise ValueError(pattern)
    closes = pd.Series(closes, index=idx)
    opens = closes.shift(1).fillna(closes.iloc[0])
    highs = pd.concat([opens, closes], axis=1).max(axis=1) + 0.02
    lows = pd.concat([opens, closes], axis=1).min(axis=1) - 0.02
    volume = pd.Series([1000 + (i % 5) * 10 for i in range(n_bars)], index=idx)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                         "Volume": volume})


def make_daily(n_days: int = 260, start="2025-01-02", base_price: float = 100.0,
              pattern: str = "uptrend") -> pd.DataFrame:
    """A synthetic daily OHLCV frame long enough for SMA200/EMA200-style indicators."""
    idx = pd.date_range(start=start, periods=n_days, freq="B")
    if pattern == "uptrend":
        closes = pd.Series([base_price + i * 0.15 + 3 * math.sin(i / 7) for i in range(n_days)], index=idx)
    elif pattern == "downtrend":
        closes = pd.Series([base_price - i * 0.15 + 3 * math.sin(i / 7) for i in range(n_days)], index=idx)
    elif pattern == "flat":
        closes = pd.Series([base_price + 2 * math.sin(i / 5) for i in range(n_days)], index=idx)
    else:
        raise ValueError(pattern)
    opens = closes.shift(1).fillna(closes.iloc[0])
    highs = pd.concat([opens, closes], axis=1).max(axis=1) + 0.3
    lows = pd.concat([opens, closes], axis=1).min(axis=1) - 0.3
    volume = pd.Series([500_000 + (i % 11) * 1000 for i in range(n_days)], index=idx)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                         "Volume": volume})


@pytest.fixture
def intraday_uptrend():
    return make_intraday_1m(n_bars=60, pattern="uptrend")


@pytest.fixture
def intraday_flat():
    return make_intraday_1m(n_bars=60, pattern="flat")


@pytest.fixture
def intraday_v_reversal():
    return make_intraday_1m(n_bars=60, pattern="v_reversal")


@pytest.fixture
def daily_uptrend():
    return make_daily(n_days=260, pattern="uptrend")
