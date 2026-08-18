"""Historical backtest for the SPY Premarket Thesis -- two tiers, per the user's explicit "do
both" decision and the ChatGPT-refined spec. Shown SEPARATELY, never blended into one number --
each tagged with its own METHODOLOGY_VERSION so a future algorithm change can't retroactively
change what an already-logged thesis's historical evidence meant.

TIER 1 (full-fidelity, capped at ~60 trading days): reconstructs SPY/QQQ/IWM/DIA premarket
direction/relative-strength + VIX level/change from REAL intraday bars. The ~60-day cap is a hard
yfinance data-retention limit for 5m bars, not a design choice. Does NOT reconstruct futures/
rates/commodities historically -- those either have no reliable long intraday history via this
project's free data source, or aren't in the same batched intraday fetch. Stated explicitly
rather than silently narrowing what "full-fidelity" means.

TIER 2 (daily proxy, ~2 years): a coarser signal derivable from daily-only bars (gap direction +
prior-day SMA50 trend + VIX level bucket) -- a much larger real sample, but NOT the same signal as
Tier 1 or the live thesis's full 8-family read. The caller is expected to show whether the two
tiers actually agree (see `tiers_agree()`), never average them into one blended number.

LEAKAGE DISCIPLINE (explicit, tested in tests/test_premarket_backtest.py): for every historical
day, the FEATURE window (what would have been visible before that day's own 9:30 ET open) and the
OUTCOME window (what happened after) are strictly separated by that day's own market open -- the
same same-day-only, no-future-leakage discipline zero_dte_log.compute_forward_outcomes() already
established for this project. The day currently being scored live is never eligible to appear in
its own historical comparison set (see `lookback_end` in both run_tier*_backtest() signatures).

SIMILARITY is a fixed, deterministic rule, never an LLM-eyeballed label:
  Tier 1 -- 4 dims (direction, risk_environment, gap_bucket, trend_alignment): HIGH if a
  same-day-count pool of >=4/4-matching days clears MIN_SAMPLE, else MEDIUM at >=3/4, else LOW at
  >=2/4, else INSUFFICIENT_SAMPLE.
  Tier 2 -- 3 dims (gap_bucket, trend_direction, vix_bucket): same ladder, thresholds noted below.
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from . import data
from . import premarket_thesis as pt
from . import zero_dte as zd

METHODOLOGY_VERSION = "v1"

TIER1_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "^VIX"]
MARKET_OPEN_ET = dtime(9, 30)
CHECKPOINT_ET = dtime(10, 0)   # matches the live thesis's own 10:00 AM checkpoint

TIER1_MIN_SAMPLE = 8    # first-pass, not calibrated -- same honesty standard as every other
TIER2_MIN_SAMPLE = 20   # unvalidated threshold in this codebase.


# ------------------------------------------------------------------ shared buckets ----------

def gap_bucket(pct: float | None) -> str | None:
    """First-pass thresholds, not calibrated -- same honesty standard as every other unvalidated
    bucket in this codebase (see premarket_thesis.py's own docstring)."""
    if pct is None:
        return None
    if pct <= -1.0:
        return "Large Down"
    if pct <= -0.2:
        return "Small Down"
    if pct < 0.2:
        return "Flat"
    if pct < 1.0:
        return "Small Up"
    return "Large Up"


def trend_alignment(price: float | None, sma50: float | None) -> str | None:
    if price is None or sma50 is None or sma50 != sma50:
        return None
    return "Above SMA50" if price >= sma50 else "Below SMA50"


def vix_bucket(level: float | None) -> str | None:
    """Same thresholds as premarket_thesis.risk_environment()'s vix_level bands, isolated to
    JUST the level (Tier 2 doesn't have an intraday VIX day_change_pct as cleanly as Tier 1's
    live snapshot does, so it buckets on level alone, not the full risk_environment() call)."""
    if level is None:
        return None
    if level > 30:
        return "EXTREME"
    if level > 22:
        return "ELEVATED"
    if level < 15:
        return "LOW"
    return "NORMAL"


# ------------------------------------------------------------------ Tier 1: intraday reconstruction ----------

def fetch_tier1_history(period: str = "60d", interval: str = "5m") -> dict[str, pd.DataFrame]:
    """ONE batched call for SPY/QQQ/IWM/DIA/VIX intraday history -- same batching discipline as
    every other multi-ticker fetch in this project. 60d is yfinance's own practical retention
    limit for 5m bars, not a chosen lookback."""
    try:
        raw = yf.download(tickers=TIER1_TICKERS, period=period, interval=interval, prepost=True,
                          group_by="ticker", progress=False, threads=True, auto_adjust=True)
    except Exception:
        return {}
    return zd._split_multi(raw, TIER1_TICKERS)


def _trading_days(df: pd.DataFrame, before: date) -> list[date]:
    """Distinct calendar dates present in an intraday frame, strictly before `before` -- the
    leakage guard for backtest inputs: the day being scored live can never be one of its own
    historical comparison days."""
    days = sorted({d for d in df.index.date if d < before})
    return days


def _day_bars(df: pd.DataFrame, day: date, start: dtime | None = None, end: dtime | None = None) -> pd.DataFrame:
    sub = df[df.index.date == day]
    if start is not None:
        sub = sub[sub.index.time >= start]
    if end is not None:
        sub = sub[sub.index.time < end]
    return sub


def _premarket_change_pct(df: pd.DataFrame, day: date) -> float | None:
    """Same shape as det.intraday_snapshot()'s day_change_pct -- first-bar-open to last-bar-close
    of that day's OWN premarket window (bars before 9:30 ET), never touching anything from or
    after that day's own open."""
    bars = _day_bars(df, day, end=MARKET_OPEN_ET)
    if bars.empty or len(bars) < 2:
        return None
    first, last = float(bars["Close"].iloc[0]), float(bars["Close"].iloc[-1])
    return round((last / first - 1) * 100, 3) if first else None


def _forward_move_pct(df: pd.DataFrame, day: date) -> float | None:
    """That day's OWN 9:30-10:00 ET move -- the leakage boundary: this window starts exactly
    where _premarket_change_pct's window ends (that day's own market open), so a historical day's
    feature computation and its outcome computation can never share a bar."""
    bars = _day_bars(df, day, start=MARKET_OPEN_ET, end=CHECKPOINT_ET)
    if bars.empty or len(bars) < 2:
        return None
    first, last = float(bars["Close"].iloc[0]), float(bars["Close"].iloc[-1])
    return round((last / first - 1) * 100, 3) if first else None


def _reconstruct_day(intraday: dict[str, pd.DataFrame], day: date) -> dict | None:
    """One historical day's premarket-only feature set, reconstructed ONLY from bars strictly
    before that day's own 9:30 ET open -- direction/gap/trend from SPY+QQQ+IWM+DIA, risk from
    VIX level+premarket change. Returns None if SPY's own premarket window that day doesn't have
    enough bars to compute anything (e.g. a half day, a data gap)."""
    spy_df = intraday.get("SPY")
    if spy_df is None:
        return None
    spy_chg = _premarket_change_pct(spy_df, day)
    if spy_chg is None:
        return None
    qqq_chg = _premarket_change_pct(intraday.get("QQQ"), day) if intraday.get("QQQ") is not None else None
    iwm_chg = _premarket_change_pct(intraday.get("IWM"), day) if intraday.get("IWM") is not None else None
    dia_chg = _premarket_change_pct(intraday.get("DIA"), day) if intraday.get("DIA") is not None else None
    vix_df = intraday.get("^VIX")
    vix_level, vix_chg = None, None
    if vix_df is not None:
        vix_bars = _day_bars(vix_df, day, end=MARKET_OPEN_ET)
        if not vix_bars.empty:
            vix_level = float(vix_bars["Close"].iloc[-1])
        vix_chg = _premarket_change_pct(vix_df, day)

    qqq_spread = (qqq_chg - spy_chg) if qqq_chg is not None else None
    iwm_spread = (iwm_chg - spy_chg) if iwm_chg is not None else None
    dia_spread = (dia_chg - spy_chg) if dia_chg is not None else None

    # Same bucket-then-score families as the live thesis, restricted to the subset feasible from
    # real intraday history (equity/growth/small-cap/defensive/volatility -- no futures/rates/
    # commodities/breadth reconstruction, see module docstring).
    families = {
        "Equity direction (SPY)": pt._family("Equity direction (SPY)", spy_chg, scale=0.5, max_pts=20),
        "Growth relative strength (QQQ)": pt._family("Growth", qqq_spread, scale=0.5, max_pts=15),
        "Small-cap confirmation (IWM)": pt._family("Small-cap", iwm_spread, scale=0.5, max_pts=10),
        "Defensive/cyclical (DIA)": pt._family("Defensive", dia_spread, scale=0.4, max_pts=10),
        "Volatility (VIX)": pt._family("Volatility", vix_chg, scale=5.0, max_pts=15, invert=True),
    }
    fam_dict = {name: {"points": pts, "raw_pct": raw} for name, (_, pts, raw) in families.items()}
    state = pt.market_state(fam_dict)
    risk = pt.risk_environment({"last": vix_level, "day_change_pct": vix_chg}, None)

    forward = _forward_move_pct(spy_df, day)
    return {"day": day, "direction": state["direction"], "risk_level": risk["level"],
           "gap_bucket": gap_bucket(spy_chg), "trend_alignment": None,  # daily SMA50 filled by caller
           "forward_move_pct": forward}


def _pool_stats(days: list[dict], direction: str) -> dict:
    """best_pct/worst_pct are FROM THE THESIS'S OWN PERSPECTIVE: for a Bearish thesis, the most
    favorable historical outcome is the largest DECLINE (best_pct is negative), not the largest
    raw number -- a naive max()/min() would silently flip this for Bearish setups."""
    moves = [d["forward_move_pct"] for d in days if d["forward_move_pct"] is not None]
    n = len(moves)
    if n == 0:
        return {"n": 0}
    arr = np.array(moves)
    if direction == "Bullish":
        wins, best, worst = int((arr > 0).sum()), float(arr.max()), float(arr.min())
    elif direction == "Bearish":
        wins, best, worst = int((arr < 0).sum()), float(arr.min()), float(arr.max())
    else:
        wins, best, worst = None, float(arr.max()), float(arr.min())
    return {"n": n, "median_move_pct": round(float(np.median(arr)), 3),
           "avg_move_pct": round(float(np.mean(arr)), 3),
           "best_pct": round(best, 3), "worst_pct": round(worst, 3),
           "win_rate_pct": round(wins / n * 100, 1) if wins is not None else None}


def _similarity_ladder(scored: list[tuple[int, dict]], min_sample: int, max_dims: int) -> tuple[str, list[dict], dict]:
    """Deterministic HIGH/MEDIUM/LOW/INSUFFICIENT_SAMPLE assignment -- tries the tightest-matching
    pool first, widens ONLY if that pool doesn't clear min_sample, and always reports which
    threshold was actually used (never silently widens without saying so)."""
    labels = ["HIGH", "MEDIUM", "LOW"] if max_dims >= 3 else ["HIGH", "LOW"]
    thresholds = list(range(max_dims, max_dims - len(labels), -1))
    for label, min_match in zip(labels, thresholds):
        pool = [d for match, d in scored if match >= min_match]
        if len(pool) >= min_sample:
            return label, pool, {"min_dims_matched": min_match, "of_dims": max_dims}
    # Loosest pool still short of min_sample -- report it honestly rather than hide the shortfall.
    loosest_min = thresholds[-1]
    pool = [d for match, d in scored if match >= loosest_min]
    return "INSUFFICIENT_SAMPLE", pool, {"min_dims_matched": loosest_min, "of_dims": max_dims}


def _causal_trend_alignment_for_day(day: date, daily: pd.DataFrame, sma50_series: pd.Series) -> str | None:
    """Trend alignment for `day`, using ONLY the PRIOR trading day's close/SMA50 (strictly BEFORE
    `day`) -- isolated into its own pure function so the leakage guard is directly unit-testable
    without needing a full synthetic intraday dataset. Using `day`'s OWN close here would leak
    that day's own end-of-session price into a feature meant to represent what was knowable
    BEFORE that day's premarket session -- same leakage class zero_dte_log.py's own tests guard
    against, just caught here before it shipped (see tests/test_premarket_backtest.py)."""
    prior_daily = daily[daily.index.date < day]
    if prior_daily.empty:
        return None
    px = float(prior_daily["Close"].iloc[-1])
    sma = float(sma50_series.reindex(prior_daily.index).iloc[-1])
    return trend_alignment(px, sma)


def run_tier1_backtest(direction: str, risk_level: str, gap_bkt: str | None, trend_align: str | None,
                       today: date | None = None) -> dict:
    """The full-fidelity intraday backtest. `today` is the date being scored live -- excluded from
    its own comparison set by construction (_trading_days only returns dates strictly before it)."""
    today = today or date.today()
    intraday = fetch_tier1_history()
    spy_df = intraday.get("SPY")
    if spy_df is None or spy_df.empty:
        return {"status": "NO_DATA", "methodology_version": METHODOLOGY_VERSION, "tier": 1}

    daily = data.get_prices("SPY", period="1y")
    sma50_series = daily["Close"].rolling(50).mean()   # causal by construction -- each row uses
    # only its own and earlier rows, never a future close.

    days = _trading_days(spy_df, before=today)
    scored: list[tuple[int, dict]] = []
    for day in days:
        rec = _reconstruct_day(intraday, day)
        if rec is None:
            continue
        rec["trend_alignment"] = _causal_trend_alignment_for_day(day, daily, sma50_series)
        if rec["forward_move_pct"] is None:
            continue
        match = sum([rec["direction"] == direction, rec["risk_level"] == risk_level,
                    rec["gap_bucket"] == gap_bkt, rec["trend_alignment"] == trend_align])
        scored.append((match, rec))

    if not scored:
        return {"status": "NO_DATA", "methodology_version": METHODOLOGY_VERSION, "tier": 1}

    label, pool, ladder = _similarity_ladder(scored, TIER1_MIN_SAMPLE, max_dims=4)
    stats = _pool_stats(pool, direction)
    return {"status": "OK" if label != "INSUFFICIENT_SAMPLE" else "INSUFFICIENT_SAMPLE",
           "tier": 1, "methodology_version": METHODOLOGY_VERSION, "similarity_label": label,
           "similarity_ingredients": {"direction": direction, "risk_environment": risk_level,
                                      "gap_bucket": gap_bkt, "trend_alignment": trend_align,
                                      **ladder},
           "lookback_days_available": len(days), "days_scored": len(scored), **stats}


# ------------------------------------------------------------------ Tier 2: daily proxy ----------

def run_tier2_backtest(direction: str, gap_bkt: str | None, trend_align: str | None, vix_bkt: str | None,
                       today: date | None = None) -> dict:
    """The coarse daily-only proxy backtest -- ~2 years of real daily bars, no intraday history
    needed. Forward outcome is that day's OWN open-to-close move (strictly after the feature
    window, which only ever looks at that day's gap + PRIOR days' trend/VIX)."""
    today = today or date.today()
    daily = data.get_prices("SPY", period="2y")
    vix_daily = data.get_prices("^VIX", period="2y")
    sma50_series = daily["Close"].rolling(50).mean()

    vix_by_date = {d: float(v) for d, v in zip(vix_daily.index.date, vix_daily["Close"]) if v == v}

    scored: list[tuple[int, dict]] = []
    idx = daily.index
    for i in range(1, len(idx)):
        day = idx[i].date()
        if day >= today:
            continue
        prior_close = float(daily["Close"].iloc[i - 1])
        day_open = float(daily["Open"].iloc[i])
        day_close = float(daily["Close"].iloc[i])
        if prior_close == 0 or day_open != day_open:
            continue
        gap = round((day_open / prior_close - 1) * 100, 3)
        sma = float(sma50_series.iloc[i - 1]) if i - 1 < len(sma50_series) else float("nan")  # PRIOR day's
        # SMA50 only -- never today's own (would leak today's own close into "prior trend").
        this_trend = trend_alignment(prior_close, sma)
        this_vix = vix_bucket(vix_by_date.get(idx[i - 1].date()))   # prior day's VIX level only
        this_gap = gap_bucket(gap)
        forward = round((day_close / day_open - 1) * 100, 3) if day_open else None
        if forward is None:
            continue
        match = sum([this_gap == gap_bkt, this_trend == trend_align, this_vix == vix_bkt])
        scored.append((match, {"day": day, "forward_move_pct": forward}))

    if not scored:
        return {"status": "NO_DATA", "methodology_version": METHODOLOGY_VERSION, "tier": 2}

    label, pool, ladder = _similarity_ladder(scored, TIER2_MIN_SAMPLE, max_dims=3)
    stats = _pool_stats(pool, direction)
    return {"status": "OK" if label != "INSUFFICIENT_SAMPLE" else "INSUFFICIENT_SAMPLE",
           "tier": 2, "methodology_version": METHODOLOGY_VERSION, "similarity_label": label,
           "similarity_ingredients": {"gap_bucket": gap_bkt, "trend_alignment": trend_align,
                                      "vix_bucket": vix_bkt, **ladder},
           "lookback_days_available": len(idx), "days_scored": len(scored), **stats}


def tiers_agree(tier1: dict, tier2: dict) -> bool | None:
    """A computed flag, not an AI judgment call -- per the explicit instruction that the two tiers
    must never be silently blended, but SHOULD be checked for agreement. True/False only when both
    tiers have a usable win_rate; None if either tier lacks enough sample to have an opinion."""
    r1, r2 = tier1.get("win_rate_pct"), tier2.get("win_rate_pct")
    if tier1.get("status") != "OK" or tier2.get("status") != "OK" or r1 is None or r2 is None:
        return None
    return (r1 >= 50) == (r2 >= 50)
