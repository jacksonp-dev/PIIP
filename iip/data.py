"""Free-tier data layer (yfinance only — no API keys, no paid feeds).

Everything downstream is only as good as this. Keep it honest: if data is missing or stale,
say so loudly rather than letting an agent hallucinate around a gap.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
import yfinance as yf

# yfinance logs a "possibly delisted; no price data found" line straight to the terminal for
# every ticker with no bars yet (e.g. ^VIX pre-market, before its first daily tick) — noise, not
# an error we need surfaced, since every caller here already handles missing/empty data gracefully.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def nearest_expiry(ticker: str, target_days: int) -> str:
    """Pick the listed expiry whose days-to-expiry is closest to `target_days` (the horizon),
    instead of blindly taking the front-month. A 30d thesis wants ~30d contracts, not 1-DTE."""
    exps = list_expiries(ticker)
    if not exps:
        raise ValueError(f"No options listed for {ticker!r}")
    today = date.today()
    return min(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days - target_days))


def get_prices(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV, split/dividend-adjusted. Raises if empty so callers never analyze nothing."""
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No price data for {ticker!r}")
    df.index = pd.to_datetime(df.index)
    return df


def get_spot(ticker: str) -> float:
    """Most recent close as the working spot price."""
    return float(get_prices(ticker, period="5d")["Close"].iloc[-1])


def get_intraday(ticker: str, interval: str = "5m", period: str = "1d") -> pd.DataFrame | None:
    """Intraday bars for active watching (VWAP, today's move). Small lookback per yfinance limits.
    Returns None if empty (e.g. market closed and no session data)."""
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df


def list_expiries(ticker: str) -> list[str]:
    """[] on any failure (network hiccup, Yahoo returning a non-JSON/empty response, no options
    listed) — never raises. A transient upstream failure here must not be indistinguishable from
    a real crash to callers; every caller already treats an empty expiries list as "nothing
    tradeable right now," which is the correct degrade-gracefully behavior for both cases."""
    try:
        return list(yf.Ticker(ticker).options or [])
    except Exception:
        return []


def get_option_chain(ticker: str, expiry: str | None = None) -> dict:
    """One expiry's calls+puts. Defaults to the nearest expiry.

    Returns raw yfinance frames — the deterministic engine derives IV/greeks/expected-move from
    them. Columns include: strike, bid, ask, lastPrice, impliedVolatility, openInterest, volume.
    Raises ValueError (not a raw yfinance/JSON exception) on any failure to fetch the chain —
    same class of transient failure as list_expiries, given a clean, callers-already-expect-it
    error type instead.
    """
    tk = yf.Ticker(ticker)
    exps = list_expiries(ticker)
    if not exps:
        raise ValueError(f"No options listed for {ticker!r}")
    expiry = expiry or exps[0]
    if expiry not in exps:
        raise ValueError(f"Expiry {expiry} not in {ticker!r} chain; available: {list(exps)[:6]}...")
    try:
        ch = tk.option_chain(expiry)
    except Exception as exc:
        raise ValueError(f"Could not fetch option chain for {ticker!r} {expiry}: {exc}") from exc
    return {"expiry": expiry, "expiries": list(exps), "calls": ch.calls, "puts": ch.puts}


def get_earnings_history(ticker: str, limit: int = 40) -> pd.DataFrame:
    """Past earnings dates with EPS estimate, reported EPS, and surprise% -- the real
    surprise-vs-expectation number, not just a beat/miss flag. Empty frame (not an exception) on
    any failure (network hiccup, or the underlying `lxml` parse dependency missing) -- same
    degrade-gracefully contract as list_expiries. Report timestamps indicate before-open vs
    after-close release, needed to align the reaction window to the right trading day."""
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()