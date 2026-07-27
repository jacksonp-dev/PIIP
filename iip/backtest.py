"""Deterministic backtest — point-in-time historical replay (engineering validation).

Only the DETERMINISTIC layer is backtested. The LLM is NOT (its training cutoff means it may already
'know' historical outcomes — that would be look-ahead). The option leg is skipped: the free tier has
no historical option chains, so we can only score the STOCK thesis here. Options are validated live.

Point-in-time discipline: the forecast as-of date D uses prices[:D] ONLY; the outcome uses D+H.
Caveat: yfinance auto_adjust bakes later splits/divs into past prices (mild, standard for backtests).
"""
from __future__ import annotations

import os

import pandas as pd
import yfinance as yf

from . import baseline
from . import predictions as pred


def _fetch(tk: str, period: str = "3y") -> pd.DataFrame:
    df = yf.Ticker(tk).history(period=period, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def run_backtest(tickers, horizons=(7, 30, 90), step_days: int = 14,
                 db: str = "iip_backtest.db") -> tuple[str, int]:
    if os.path.exists(db):
        os.remove(db)
    pred.init_db(db)
    spy = _fetch("SPY")["Close"]
    logged = 0

    for tk in tickers:
        full = _fetch(tk)
        if len(full) < 300:
            continue
        close = full["Close"]
        last_date = close.index[-1]
        # as-of dates: start after enough history for SMA200 + distribution; stop early enough that
        # even the longest horizon resolves within available data.
        d = close.index[260]
        while d <= last_date - pd.Timedelta(days=max(horizons) + 3):
            pit = close.index[close.index <= d]
            if len(pit) >= 260:
                Dt = pit[-1]                         # nearest trading day at/before the as-of date
                prices_pit = full.loc[:Dt]           # POINT-IN-TIME: nothing after Dt
                spot = float(close.loc[Dt])
                for H in horizons:
                    fc = baseline.deterministic_forecast(tk, prices_pit, H)
                    pid = pred.log_prediction(
                        {**fc, "ts": Dt.tz_localize("UTC").isoformat(timespec="seconds"), "spot": spot}, db)
                    # resolve against real forward data (calendar horizon, matching live)
                    fut = close.index[close.index <= Dt + pd.Timedelta(days=H)]
                    if len(fut) == 0 or fut[-1] <= Dt:
                        continue
                    end_px = float(close.loc[fut[-1]])
                    realized = (end_px / spot - 1) * 100
                    sa = spy.index[spy.index <= Dt]
                    sr = spy.index[spy.index <= Dt + pd.Timedelta(days=H)]
                    spy_ret = ((float(spy.loc[sr[-1]]) / float(spy.loc[sa[-1]]) - 1) * 100
                               if len(sa) and len(sr) else None)
                    if fc["stock_direction"] == "neutral":
                        hit = None
                    else:
                        up = realized > 0
                        hit = int((fc["stock_direction"] == "bullish" and up) or
                                  (fc["stock_direction"] == "bearish" and not up))
                    pred.resolve(pid, {
                        "realized_return_pct": round(realized, 3),
                        "spy_return_pct": round(spy_ret, 3) if spy_ret is not None else None,
                        "underlying_return_pct": round(realized, 3), "hit": hit}, db)
                    logged += 1
            d = d + pd.Timedelta(days=step_days)
    return db, logged