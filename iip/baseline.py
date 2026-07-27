"""Deterministic baseline forecaster — the bar the LLM must beat (LOCKED SPEC v1).

A competent quant's non-AI model: EQUAL-WEIGHT VOTE across transparent factors. No weights, no
optimization, no ML, no tuning — that's what makes it a fair, honest baseline. Every number here
is computed from data; nothing is invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import deterministic as det


def _bollinger(prices: pd.DataFrame, n: int = 20, k: float = 2.0):
    c = prices["Close"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    return float(c.iloc[-1]), float((ma + k * sd).iloc[-1]), float((ma - k * sd).iloc[-1])


def deterministic_forecast(ticker: str, prices: pd.DataFrame, horizon_days: int) -> dict:
    """Stock-thesis forecast for one horizon via equal-weight factor vote."""
    tech = det.technical_metrics(prices)
    hist = det.historical_move_distribution(prices, horizon_days)
    c = prices["Close"]

    votes: dict[str, int] = {}
    # 1. long-term trend
    votes["trend_sma200"] = 1 if tech.get("above_sma200") else -1
    # 2. relative strength (~6mo / 126 trading days)
    if len(c) > 127:
        votes["rel_strength_126d"] = 1 if (float(c.iloc[-1]) / float(c.iloc[-127]) - 1) > 0 else -1
    else:
        votes["rel_strength_126d"] = 0
    # 3. RSI mean-reversion (oversold bounce / overbought fade)
    rsi = tech["rsi14"]
    votes["rsi_meanrev"] = 1 if rsi < 30 else (-1 if rsi > 70 else 0)
    # 4. Bollinger mean-reversion
    close, ub, lb = _bollinger(prices)
    votes["bollinger_meanrev"] = 1 if close < lb else (-1 if close > ub else 0)
    # 5. momentum over the horizon (horizon is CALENDAR days -> convert to trading rows)
    mom_rows = max(1, round(horizon_days * 252 / 365))
    if len(c) > mom_rows + 1:
        votes["momentum_H"] = 1 if (float(c.iloc[-1]) / float(c.iloc[-(mom_rows + 1)]) - 1) > 0 else -1
    else:
        votes["momentum_H"] = 0
    # (sector-relative strength: TODO — needs ticker->sector ETF map)

    score = sum(votes.values())
    n_factors = len(votes)
    direction = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")

    # expected move (magnitude) and prob_up — historical, ATR-fallback. Never invented.
    exp_move = hist.get("std_pct")
    if exp_move is None:
        exp_move = tech.get("atr_pct")
    # prob_up comes from the VOTE (direction + conviction), not the unconditional base rate — the
    # base rate isn't a directional forecast (a stock that historically fell 55% of the time still
    # gets a bullish call when the factors say so). Fixed slope = a prior, NOT tuned to outcomes:
    # full factor agreement maps to ~0.85, never certainty. Direction-consistent by construction.
    prob_up = round(0.5 + 0.35 * (score / n_factors), 3)

    reasoning = {
        "bull_factors": [k for k, v in votes.items() if v > 0],
        "bear_factors": [k for k, v in votes.items() if v < 0],
        "biggest_uncertainty": "equal-weight factor vote; no catalyst/regime/news awareness",
        "sources": ["yfinance OHLCV", "historical forward-return distribution"],
        "votes": votes, "score": score,
    }
    return {
        "source": "deterministic",
        "ticker": ticker, "horizon_days": horizon_days,
        "stock_direction": direction,
        "stock_expected_move_pct": exp_move,
        "stock_prob_up": prob_up,
        "confidence": round(abs(score) / n_factors, 3),
        "reasoning": reasoning,
    }


def simulated_option(spot: float, chain: dict, direction: str) -> dict:
    """Standardized ATM contract implied by the thesis: call if bullish, put if bearish
    (skip if neutral). Entry premium = mid, plus IV and greeks at entry — all computed."""
    if direction == "neutral":
        return {}
    opt_type = "call" if direction == "bullish" else "put"
    side = chain["calls"] if opt_type == "call" else chain["puts"]
    side = side.copy()
    side["_dist"] = (side["strike"] - spot).abs()
    atm = side.loc[side["_dist"].idxmin()]

    iv = atm.get("impliedVolatility")
    iv = float(iv) if iv and not pd.isna(iv) else float("nan")
    entry = det._mid(atm)
    T, _ = det._dte_years(chain["expiry"])
    greeks = det.bs_greeks(spot, float(atm["strike"]), T, 0.045, iv, call=(opt_type == "call"))
    return {
        "option_type": opt_type,
        "option_strike": round(float(atm["strike"]), 2),
        "option_expiry": chain["expiry"],
        "option_entry_premium": round(entry, 2) if entry == entry else None,
        "option_iv_pct": round(iv * 100, 1) if iv == iv else None,
        "option_greeks": greeks,
    }