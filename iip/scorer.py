"""Scorer — the centerpiece. Resolves due predictions against reality and reports calibration +
benchmark performance. This is what separates a forecasting system from a note-taking app.

Honest simplifications (documented, not hidden):
- Option is marked to INTRINSIC at its expiry (held to expiry). This bakes in theta/expiry
  brutally and needs no future-IV assumption — the honest, conservative choice for learning.
- realized_return is the underlying's return over the horizon; SPY over the same window is the
  market benchmark; hold-underlying is the option's benchmark.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from . import predictions as pred


def _window(ticker: str, start: datetime, end: datetime) -> pd.Series | None:
    df = yf.Ticker(ticker).history(start=start.date() - timedelta(days=4),
                                   end=end.date() + timedelta(days=4), auto_adjust=True)
    if df is None or df.empty:
        return None
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _close_on(s: pd.Series, dt: datetime) -> float | None:
    prior = s[s.index <= pd.Timestamp(dt.date())]
    return float(prior.iloc[-1]) if len(prior) else None


def resolve_due(db: str = pred.DB_PATH, asof: datetime | None = None) -> int:
    """Resolve every prediction whose horizon has elapsed. Returns count resolved."""
    asof = asof or datetime.now(timezone.utc)
    rows = pred.get_due(db, asof)
    n = 0
    for r in rows:
        entry_dt = datetime.fromisoformat(r["ts"])
        due_dt = datetime.fromisoformat(r["resolve_due"])
        exp_dt = datetime.fromisoformat(r["option_expiry"]) if r["option_expiry"] else None
        # fetch far enough to cover BOTH the horizon resolve date and the option expiry (which can
        # sit a few days past the horizon) — else the option marks at a stale, pre-expiry price.
        # strip tz for the comparison: due_dt is tz-aware (from resolve_due), exp_dt is naive
        # (from a bare date string) — mixing them in max() raises.
        fetch_end = max(d.replace(tzinfo=None) for d in (due_dt, exp_dt) if d is not None)
        ser = _window(r["ticker"], entry_dt, fetch_end)
        if ser is None:
            continue
        entry_px = r["spot"]
        end_px = _close_on(ser, due_dt)
        if not entry_px or not end_px:
            continue
        realized = (end_px / entry_px - 1) * 100

        # realized vol over the window (annualized)
        win = ser[(ser.index >= pd.Timestamp(entry_dt.date())) & (ser.index <= pd.Timestamp(due_dt.date()))]
        rvol = float(np.log(win / win.shift()).dropna().std() * math.sqrt(252) * 100) if len(win) > 2 else None

        spy = _window("SPY", entry_dt, due_dt)
        spy_ret = None
        if spy is not None:
            se, sl = _close_on(spy, entry_dt), _close_on(spy, due_dt)
            if se and sl:
                spy_ret = (sl / se - 1) * 100

        # option: mark to intrinsic at its expiry (held to expiry)
        opt_exit = opt_pnl = None
        if r["option_type"] and r["option_entry_premium"] and exp_dt is not None:
            px_exp = _close_on(ser, exp_dt) or end_px
            if r["option_type"] == "call":
                intrinsic = max(0.0, px_exp - r["option_strike"])
            else:
                intrinsic = max(0.0, r["option_strike"] - px_exp)
            opt_exit = round(intrinsic, 2)
            opt_pnl = round((intrinsic / r["option_entry_premium"] - 1) * 100, 2)

        # a neutral forecast makes no directional bet -> excluded from hit-rate (hit=None),
        # not counted as a miss (which would unfairly drag the score). A perfectly FLAT outcome
        # (realized == 0) gets the same treatment -- found in this session's accounting audit
        # that `not up` (up = realized > 0) silently treated flat as "down", scoring a bearish
        # call as a hit and a bullish call as a miss on the identical flat outcome. Neither
        # direction was actually right, so neither should be credited or blamed.
        if r["stock_direction"] == "neutral" or realized == 0:
            hit = None
        else:
            up = realized > 0
            hit = int((r["stock_direction"] == "bullish" and up) or
                      (r["stock_direction"] == "bearish" and not up))
        pred.resolve(r["id"], {
            "realized_return_pct": round(realized, 3), "realized_vol_pct": round(rvol, 2) if rvol else None,
            "spy_return_pct": round(spy_ret, 3) if spy_ret is not None else None,
            "underlying_return_pct": round(realized, 3),
            "option_exit_premium": opt_exit, "option_pnl_pct": opt_pnl, "hit": hit,
        }, db)
        n += 1
    return n


def _ci(p: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    return (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))   # a probability CI stays in [0,1]


def _stats(rows) -> dict:
    rows = [r for r in rows if r["hit"] is not None]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    hits = [r["hit"] for r in rows]
    hr = sum(hits) / n
    lo, hi = _ci(hr, n)
    brier = np.mean([(r["stock_prob_up"] - (1 if r["realized_return_pct"] > 0 else 0)) ** 2
                     for r in rows if r["stock_prob_up"] is not None])
    stock_ret = float(np.mean([r["realized_return_pct"] for r in rows]))
    # alpha must be a PAIRED difference over rows that actually have a SPY value — not
    # mean(stock over all) minus mean(spy over a subset), which mixes samples.
    paired = [(r["realized_return_pct"], r["spy_return_pct"]) for r in rows if r["spy_return_pct"] is not None]
    spy_ret = float(np.mean([s for _, s in paired])) if paired else None
    alpha = float(np.mean([a - s for a, s in paired])) if paired else None
    opt = [r["option_pnl_pct"] for r in rows if r["option_pnl_pct"] is not None]
    return {
        "n": n,
        "hit_rate": round(hr, 3), "hit_rate_ci": (round(lo, 3), round(hi, 3)),
        "beats_coinflip": lo > 0.5,
        "brier": round(float(brier), 4),
        "mean_stock_ret_pct": round(stock_ret, 2),
        "mean_spy_ret_pct": round(spy_ret, 2) if spy_ret is not None else None,
        "alpha_vs_spy_pct": round(alpha, 2) if alpha is not None else None,
        "option_expectancy_pct": round(float(np.mean(opt)), 1) if opt else None,
        "option_vs_hold_pct": round(float(np.mean(opt) - stock_ret), 1) if opt else None,
    }


def report(db: str = pred.DB_PATH) -> dict:
    rows = pred.all_rows(db, resolved_only=True)
    out = {"overall": _stats(rows),
           "by_source": {}, "by_horizon": {}}
    for src in ("deterministic", "llm", "reddit_momentum"):
        out["by_source"][src] = _stats([r for r in rows if r["source"] == src])
    for h in (7, 30, 90):
        out["by_horizon"][h] = _stats([r for r in rows if r["horizon_days"] == h])
    # PRE-REGISTERED PRIMARY TEST: 30d stock directional hit-rate, LLM vs deterministic
    d30 = _stats([r for r in rows if r["horizon_days"] == 30 and r["source"] == "deterministic"])
    l30 = _stats([r for r in rows if r["horizon_days"] == 30 and r["source"] == "llm"])
    out["primary_test_30d_llm_vs_deterministic"] = {
        "deterministic_hit_rate": d30.get("hit_rate"), "deterministic_n": d30.get("n"),
        "llm_hit_rate": l30.get("hit_rate"), "llm_n": l30.get("n"),
        "llm_beats_deterministic": (l30.get("hit_rate") or 0) > (d30.get("hit_rate") or 0),
    }
    return out


def format_report(db: str = pred.DB_PATH) -> str:
    rep = report(db)
    o = rep["overall"]
    if o["n"] == 0:
        return "No resolved predictions yet. Come back after the first horizons elapse (7d soonest)."
    lines = ["=== PIIP SCORECARD ===",
             f"Resolved predictions: {o['n']}",
             f"Hit rate: {o['hit_rate']:.0%}  CI[{o['hit_rate_ci'][0]:.0%},{o['hit_rate_ci'][1]:.0%}]"
             f"  beats-coinflip={o['beats_coinflip']}",
             f"Brier: {o['brier']}  (lower=better; 0.25=coinflip)",
             f"Stock {o['mean_stock_ret_pct']:+.1f}%  vs SPY {o.get('mean_spy_ret_pct')}%"
             f"  alpha {o.get('alpha_vs_spy_pct')}%",
             f"Option expectancy {o.get('option_expectancy_pct')}%  vs hold-underlying "
             f"{o.get('option_vs_hold_pct')}%",
             "",
             "GATES: Exploration N>=50, Evidence N>=200 (conditioned by horizon). "
             f"Currently N={o['n']} -> {'collect only' if o['n'] < 50 else 'exploration'}.",
             ""]
    pt = rep["primary_test_30d_llm_vs_deterministic"]
    lines.append(f"PRIMARY TEST (30d hit-rate, LLM vs deterministic): "
                 f"det={pt['deterministic_hit_rate']}(n={pt['deterministic_n']}) "
                 f"llm={pt['llm_hit_rate']}(n={pt['llm_n']}) "
                 f"-> LLM ahead: {pt['llm_beats_deterministic']}")
    lines.append("(Note: LLM edge is FORWARD-ONLY evidence; deterministic can also be backtested.)")
    return "\n".join(lines)