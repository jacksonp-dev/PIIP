"""Deterministic engine — CODE computes every number. The LLM never calculates these.

This is the product. It is fully backtestable on history (no look-ahead, no LLM), so it can be
validated fast, unlike the LLM layer. Prompts downstream say: "interpret these, never invent them."
"""
from __future__ import annotations

from datetime import date, datetime
from math import erf, exp, log, pi, sqrt

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---- normal distribution (no scipy dependency) ----
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _npdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, call: bool = True) -> float:
    """Black-Scholes option value — used to estimate what a contract is worth under a hypothetical
    future spot + time (the 'sell early' what-if). Falls back to intrinsic at/near expiry."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if call else max(0.0, K - S)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if call:
        return S * _ncdf(d1) - K * exp(-r * T) * _ncdf(d2)
    return K * exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, call: bool = True) -> dict:
    """Black-Scholes greeks. theta/day, vega per 1 vol point, rho per 1% rate move. Empty dict on
    degenerate inputs."""
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        return {}
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    delta = _ncdf(d1) if call else _ncdf(d1) - 1.0
    gamma = _npdf(d1) / (S * sigma * sqrt(T))
    vega = S * _npdf(d1) * sqrt(T) / 100.0
    if call:
        theta = (-(S * _npdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * _ncdf(d2)) / 365.0
        rho = K * T * exp(-r * T) * _ncdf(d2) / 100.0
    else:
        theta = (-(S * _npdf(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * _ncdf(-d2)) / 365.0
        rho = -K * T * exp(-r * T) * _ncdf(-d2) / 100.0
    return {"delta": round(delta, 4), "gamma": round(gamma, 6),
            "vega": round(vega, 4), "theta_per_day": round(theta, 4), "rho": round(rho, 4)}


def ema(series: pd.Series, span: int) -> float:
    """Exponential moving average, last value. NaN if there isn't enough history."""
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1]) if len(series) >= span else float("nan")


def technical_metrics(prices: pd.DataFrame) -> dict:
    """Trend / momentum / realized-vol snapshot from OHLCV."""
    c = prices["Close"]
    close = float(c.iloc[-1])

    def sma(n):
        return float(c.rolling(n).mean().iloc[-1]) if len(c) >= n else float("nan")

    # RSI(14), Wilder
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])

    # ATR(14)
    h, l, cl = prices["High"], prices["Low"], prices["Close"]
    tr = pd.concat([h - l, (h - cl.shift()).abs(), (l - cl.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # realized (historical) vol, annualized
    ret = np.log(c / c.shift()).dropna()
    hv20 = float(ret.tail(20).std() * sqrt(TRADING_DAYS)) if len(ret) >= 20 else float("nan")
    hv60 = float(ret.tail(60).std() * sqrt(TRADING_DAYS)) if len(ret) >= 60 else float("nan")

    hi52 = float(c.tail(252).max()) if len(c) >= 20 else close
    lo52 = float(c.tail(252).min()) if len(c) >= 20 else close

    sma50, sma200 = sma(50), sma(200)
    ema20v, ema50v, ema200v = ema(c, 20), ema(c, 50), ema(c, 200)
    return {
        "close": round(close, 4),
        "sma20": round(sma(20), 4), "sma50": round(sma50, 4), "sma200": round(sma200, 4),
        "above_sma200": (close >= sma200) if sma200 == sma200 else None,
        "ema20": round(ema20v, 4) if ema20v == ema20v else None,
        "ema50": round(ema50v, 4) if ema50v == ema50v else None,
        "ema200": round(ema200v, 4) if ema200v == ema200v else None,
        "ema_aligned_bull": (ema20v > ema50v > ema200v)
            if ema20v == ema20v and ema50v == ema50v and ema200v == ema200v else None,
        "ema_aligned_bear": (ema20v < ema50v < ema200v)
            if ema20v == ema20v and ema50v == ema50v and ema200v == ema200v else None,
        "rsi14": round(rsi, 1),
        "atr14": round(atr, 4),
        "atr_pct": round(atr / close * 100, 2) if close else None,
        "hv20_annual_pct": round(hv20 * 100, 1) if hv20 == hv20 else None,
        "hv60_annual_pct": round(hv60 * 100, 1) if hv60 == hv60 else None,
        "ret_20d_pct": round((close / float(c.iloc[-21]) - 1) * 100, 2) if len(c) > 21 else None,
        "ret_60d_pct": round((close / float(c.iloc[-61]) - 1) * 100, 2) if len(c) > 61 else None,
        "pct_from_52w_high": round((close / hi52 - 1) * 100, 2),
        "pct_from_52w_low": round((close / lo52 - 1) * 100, 2),
    }


def intraday_snapshot(df: pd.DataFrame) -> dict | None:
    """Today's VWAP/range/day-change from intraday bars (Close, Open, High, Low, Volume). Pure
    function — extracted so it has exactly one implementation instead of being written inline
    wherever a VWAP is needed (was previously duplicated in app.py's Research-tab helper).
    None if there's no session data (e.g. market closed, empty bars)."""
    if df is None or df.empty:
        return None
    c, v = df["Close"], df["Volume"]
    last, day_open = float(c.iloc[-1]), float(df["Open"].iloc[0])
    hi, lo = float(df["High"].max()), float(df["Low"].min())
    vwap = float((c * v).sum() / v.sum()) if v.sum() > 0 else last
    rng = hi - lo
    return {"last": last, "open": day_open, "high": hi, "low": lo, "vwap": vwap,
            "pct_from_vwap": (last / vwap - 1) * 100 if vwap else 0.0,
            "day_change_pct": (last / day_open - 1) * 100 if day_open else 0.0,
            "range_pos": (last - lo) / rng * 100 if rng > 0 else 50.0}


def _mid(row) -> float:
    b, a = row.get("bid", np.nan), row.get("ask", np.nan)
    if b and a and a > 0 and not (pd.isna(b) or pd.isna(a)):
        return (float(b) + float(a)) / 2.0
    lp = row.get("lastPrice", np.nan)
    return float(lp) if not pd.isna(lp) else float("nan")


def _dte_years(expiry: str) -> float:
    exp_d = datetime.strptime(expiry, "%Y-%m-%d").date()
    days = max((exp_d - date.today()).days, 1)
    return days / 365.0, days


def option_metrics(spot: float, chain: dict, r: float = 0.045) -> dict:
    """ATM IV, expected move (straddle + IV-implied), put/call OI, ATM greeks — all computed."""
    calls, puts, expiry = chain["calls"].copy(), chain["puts"].copy(), chain["expiry"]
    T, dte_days = _dte_years(expiry)

    calls["_dist"] = (calls["strike"] - spot).abs()
    puts["_dist"] = (puts["strike"] - spot).abs()
    atm_call = calls.loc[calls["_dist"].idxmin()]
    atm_put = puts.loc[puts["_dist"].idxmin()]

    ivs = [v for v in (atm_call.get("impliedVolatility"), atm_put.get("impliedVolatility"))
           if v and not pd.isna(v)]
    atm_iv = float(np.mean(ivs)) if ivs else float("nan")

    straddle = _mid(atm_call) + _mid(atm_put)          # ~1 std move to expiry, in $
    em_iv = spot * atm_iv * sqrt(T) if atm_iv == atm_iv else float("nan")

    call_oi = float(calls["openInterest"].fillna(0).sum())
    put_oi = float(puts["openInterest"].fillna(0).sum())

    greeks = bs_greeks(spot, float(atm_call["strike"]), T, r, atm_iv, call=True)
    return {
        "expiry": expiry, "dte_days": dte_days,
        "atm_strike": round(float(atm_call["strike"]), 2),
        "atm_iv_pct": round(atm_iv * 100, 1) if atm_iv == atm_iv else None,
        "expected_move_straddle_$": round(straddle, 2) if straddle == straddle else None,
        "expected_move_straddle_pct": round(straddle / spot * 100, 2) if straddle == straddle else None,
        "expected_move_iv_$": round(em_iv, 2) if em_iv == em_iv else None,
        "expected_move_iv_pct": round(em_iv / spot * 100, 2) if em_iv == em_iv else None,
        "put_call_oi_ratio": round(put_oi / call_oi, 2) if call_oi else None,
        "atm_call_greeks": greeks,
    }


def historical_move_distribution(prices: pd.DataFrame, horizon_days: int = 30) -> dict:
    """Empirical forward-return distribution over `horizon_days` — the honesty check on IV's
    'expected move': does the option market's implied move match what this name actually does?"""
    # horizon_days is CALENDAR days (matches resolve_due + option expiry). Convert to TRADING rows
    # (~252/365) so the forward-return window matches how the prediction is actually SCORED — else
    # a "30d" move is measured over 30 trading days (~42 calendar) but resolved over 30 calendar.
    c = prices["Close"]
    rows = max(1, round(horizon_days * 252 / 365))
    fwd = (c.shift(-rows) / c - 1.0).dropna()
    if len(fwd) < 30:
        return {"n": int(len(fwd)), "note": "insufficient history"}
    return {
        "horizon_days": horizon_days, "trading_rows": rows, "n": int(len(fwd)),
        "p10_pct": round(float(fwd.quantile(0.10)) * 100, 2),
        "p50_pct": round(float(fwd.quantile(0.50)) * 100, 2),
        "p90_pct": round(float(fwd.quantile(0.90)) * 100, 2),
        "std_pct": round(float(fwd.std()) * 100, 2),
        "mean_pct": round(float(fwd.mean()) * 100, 2),
        "prob_up": round(float((fwd > 0).mean()), 3),
    }


