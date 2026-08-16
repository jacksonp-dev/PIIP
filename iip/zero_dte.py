"""0DTE Index Intelligence — Phase 1, free-tier.

NOT a trading bot. Never places trades, never says "buy now." Helps read the market's current
condition for manual 0DTE decisions on SPY/QQQ/IWM/DIA — the tickers where "0DTE" and index-
relative concepts (mega-cap health, sector confirmation, breadth) actually mean something — plus
NVDA (PIIP audit 2026-08, Batch 3), analyzed short-dated/near-0DTE where available since NVDA
doesn't list same-day expiries the way the index ETFs do (nearest_dated_chain() already falls back
to the closest available expiry). The index-relative rows (Breadth, Sector Health, Mega Cap
Health) stay market-wide background context when NVDA is the selected ticker, not NVDA-specific.

Design rules, enforced not aspirational:
  * Every score is a sum of named, itemized point contributions returned alongside the total —
    no black-box numbers. Same shape as iip/reddit_momentum.py::compute_momentum().
  * "Real-time" here means a 30-60s refresh on free yfinance data, not true few-second streaming
    (yfinance rate-limits/bans aggressive polling — confirmed this session). Every section shows
    its own "as of" timestamp so the UI never implies lower latency than it actually has.
  * NYSE TICK / true Advance-Decline have NO free source anywhere (confirmed this session) — this
    module approximates breadth from a basket (mega-caps + sector ETFs + the 4 indices) and labels
    it explicitly as a proxy, never presented as the real thing.
  * Dealer gamma exposure is an ESTIMATE from options open interest (yfinance OI is end-of-prior-
    day, not intraday-live, and the dealer-sign convention is a standard retail approximation) —
    labeled `is_estimate=True` everywhere it appears, context only, never a guaranteed level.
  * Batches all quote fetches into ONE multi-ticker yf.download() call per refresh cycle — polling
    the ~25 basket tickers individually every 30s would be ~50 req/min, well past yfinance's safe
    ~6/min guidance.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from . import data
from . import deterministic as det

INDEX_TICKERS = ["SPY", "QQQ", "IWM", "DIA"]

# PIIP audit 2026-08, Batch 3: NVDA is selectable on this page too, alongside the 4 index ETFs --
# short-dated/near-0DTE where available (NVDA doesn't list same-day expiries like the index ETFs
# do; nearest_dated_chain() already falls back to the closest available expiry, so this is a
# UI/analysis-scope change, not a new options-fetch behavior). Already fetched as part of
# MEGA_CAPS below, so adding it here is free -- no new network call.
ANALYSIS_TICKERS = INDEX_TICKERS + ["NVDA"]

# Approximate relative influence on SPY — NOT official S&P weights (no free source for live index
# weights). Used only to rank which mega-caps matter most for Mega Cap Health, not for precise
# attribution. Revisit periodically; these drift as market caps change.
MEGA_CAP_WEIGHTS = {"NVDA": 7.5, "MSFT": 6.5, "AAPL": 6.5, "AMZN": 4.0, "META": 3.0,
                    "GOOGL": 2.5, "AVGO": 2.5, "TSLA": 2.0, "BRK-B": 1.7, "JPM": 1.5}
MEGA_CAPS = list(MEGA_CAP_WEIGHTS.keys())

SECTOR_ETFS = {"Technology": "XLK", "Financials": "XLF", "Industrials": "XLI",
              "Healthcare": "XLV", "Energy": "XLE", "Utilities": "XLU",
              "Consumer Discretionary": "XLY", "Communication": "XLC",
              "Real Estate": "XLRE", "Materials": "XLB"}

VIX_TICKER = "^VIX"

# NVDA is a top-weighted holding in all 4 -- SPY and QQQ already come from INDEX_TICKERS, SMH and
# SOXX (semiconductor ETFs) are net-new fetches for the NVDA relative-strength comparison (PIIP
# audit 2026-08, Batch 2 / Phase 16). Still one batched call, not a new polling source.
NVDA_COMPARISON_TICKERS = ["SPY", "QQQ", "SMH", "SOXX"]

ALL_BASKET_TICKERS = sorted(set(INDEX_TICKERS + MEGA_CAPS + list(SECTOR_ETFS.values()) +
                                [VIX_TICKER] + NVDA_COMPARISON_TICKERS))


def _split_multi(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """yf.download's multi-ticker return is one wide frame with a (ticker, field) MultiIndex —
    split it back into the per-ticker OHLCV frames the rest of this codebase expects."""
    out: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            if tk in raw.columns.get_level_values(0):
                sub = raw[tk].dropna(how="all")
                if not sub.empty:
                    out[tk] = sub
    elif len(tickers) == 1 and not raw.empty:
        out[tickers[0]] = raw
    return out


def fetch_intraday_batch(tickers: list[str] = ALL_BASKET_TICKERS,
                         interval: str = "1m", period: str = "1d") -> dict[str, pd.DataFrame]:
    """ONE batched request for every basket ticker's intraday bars — never loop per-ticker calls
    here, that's exactly the polling pattern that gets an IP rate-limited."""
    try:
        raw = yf.download(tickers=tickers, period=period, interval=interval,
                          group_by="ticker", progress=False, threads=True, auto_adjust=True)
    except Exception:
        return {}
    return _split_multi(raw, tickers)


def fetch_daily_batch(tickers: list[str] = ALL_BASKET_TICKERS,
                      period: str = "1y") -> dict[str, pd.DataFrame]:
    """Same batching discipline for the daily bars technical_metrics() needs (SMA/EMA/RSI)."""
    try:
        raw = yf.download(tickers=tickers, period=period, interval="1d",
                          group_by="ticker", progress=False, threads=True, auto_adjust=True)
    except Exception:
        return {}
    return _split_multi(raw, tickers)


def ticker_snapshot(ticker: str, intraday: dict[str, pd.DataFrame],
                    daily: dict[str, pd.DataFrame]) -> dict | None:
    """One ticker's combined intraday + daily-technicals snapshot, from already-fetched batch
    data (no new network calls) — the building block every scoring function below consumes."""
    idf, ddf = intraday.get(ticker), daily.get(ticker)
    if idf is None or idf.empty or ddf is None or ddf.empty:
        return None
    snap = det.intraday_snapshot(idf)
    if snap is None:
        return None
    tech = det.technical_metrics(ddf)
    today_vol = float(idf["Volume"].sum())
    avg20_vol = float(ddf["Volume"].tail(20).mean()) if len(ddf) >= 20 else None
    # vs a FULL prior day's average — reads low before market close since it's a partial day's
    # volume; still directionally useful (a 2x reading at 11am is a real signal), just not a
    # time-of-day-adjusted "true" relative volume.
    rel_volume = (today_vol / avg20_vol) if avg20_vol else None
    return {"ticker": ticker, **snap, "tech": tech, "today_volume": today_vol,
            "avg20_volume": avg20_vol, "rel_volume": rel_volume}


def previous_day_levels(daily_df: pd.DataFrame | None) -> dict | None:
    """Previous trading day's High/Low/Close, from the daily bars already fetched for technicals
    -- no new network call. Filters on the actual calendar date (not just `.iloc[-2]`) since
    whether yfinance's daily frame already includes an in-progress "today" row depends on when
    during the session this runs -- date-filtering is correct whether today's row is there yet
    or not, `.iloc[-2]` would silently be wrong in one of those two cases."""
    if daily_df is None or daily_df.empty:
        return None
    prior = daily_df[daily_df.index.date < date.today()]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    return {"high": float(row["High"]), "low": float(row["Low"]), "close": float(row["Close"]),
            "date": prior.index[-1].date().isoformat()}


def opening_range(intraday_df: pd.DataFrame | None, minutes: int = 15) -> dict | None:
    """First `minutes` of TODAY's session bars -> range high/low, plus where the last price sits
    relative to it (inside / above the breakout / below the breakdown) -- one of the most-used
    mechanical 0DTE setups. From the same intraday bars already fetched, no new network call."""
    if intraday_df is None or intraday_df.empty:
        return None
    todays = intraday_df[intraday_df.index.date == date.today()]
    if todays.empty:
        todays = intraday_df   # best-effort fallback rather than showing nothing
    or_df = todays.head(minutes)
    if or_df.empty:
        return None
    hi, lo, last = float(or_df["High"].max()), float(or_df["Low"].min()), float(todays["Close"].iloc[-1])
    status = "Above breakout" if last > hi else "Below breakdown" if last < lo else "Inside range"
    return {"minutes": minutes, "high": hi, "low": lo, "last": last, "status": status}


def _lean(value: float | None, scale: float, max_pts: float) -> float:
    """Linearly scale a raw signed metric into a +/-max_pts contribution, clipped. `scale` is the
    value at which the contribution saturates at max_pts."""
    if value is None or value != value:
        return 0.0
    return round(max(-max_pts, min(max_pts, (value / scale) * max_pts)), 1)


def momentum_engine(intraday_df: pd.DataFrame) -> dict | None:
    """Velocity (recent rate of change), acceleration (change in velocity), and a continuation
    heuristic — from the same intraday bars already fetched, no new network call.

    `continuation_score_pct` (renamed from `continuation_probability_pct`, PIIP audit 2026-08):
    this is a hand-tuned weighted formula (base 50, +/- coefficients on velocity/direction-
    agreement), NOT a value calibrated against historical outcomes. It was named/described
    elsewhere in the app as a "probability," which claims a statistical property this number
    doesn't have -- it's a heuristic SCORE, same category as every other 0-100 number in this
    module, not a calibrated likelihood. Also computed from the last 5 vs. prior 5 bars of
    whatever interval was fetched (1m bars as of this module's fetch_intraday_batch default),
    so it's specifically a short (~5-minute) lookback, not a session-level read."""
    if intraday_df is None or len(intraday_df) < 10:
        return None
    c = intraday_df["Close"]
    recent, prior = c.tail(5), c.tail(10).head(5)
    velocity_pct = float((recent.iloc[-1] / recent.iloc[0] - 1) * 100) if recent.iloc[0] else 0.0
    prior_velocity_pct = float((prior.iloc[-1] / prior.iloc[0] - 1) * 100) if prior.iloc[0] else 0.0
    acceleration_pct = round(velocity_pct - prior_velocity_pct, 3)
    same_direction = (velocity_pct > 0 and acceleration_pct >= 0) or (velocity_pct < 0 and acceleration_pct <= 0)
    continuation_score = min(90, max(10, 50 + abs(velocity_pct) * 20 + (15 if same_direction else -15)))
    return {"velocity_pct": round(velocity_pct, 3), "acceleration_pct": acceleration_pct,
            "velocity_label": "Increasing" if velocity_pct > 0.02 else "Decreasing" if velocity_pct < -0.02 else "Flat",
            "acceleration_label": "Positive" if acceleration_pct > 0 else "Negative" if acceleration_pct < 0 else "Flat",
            "continuation_score_pct": round(continuation_score, 0)}


def ticker_health(snap: dict) -> dict:
    """0-100 health composite for one ticker (SPY/QQQ/IWM/DIA) — trend, momentum, buying
    pressure, structure, relative volume. Itemized, same shape as every other score here."""
    tech = snap["tech"]
    pts = {}
    pts["Trend (EMA alignment)"] = 25 if tech.get("ema_aligned_bull") else (
        -25 if tech.get("ema_aligned_bear") else _lean(
            (tech.get("ema20") or 0) - (tech.get("ema50") or 0), scale=(tech.get("close") or 1) * 0.01, max_pts=12))
    pts["Momentum (RSI)"] = _lean((tech.get("rsi14") or 50) - 50, scale=20, max_pts=20)
    pts["Buying pressure (vs VWAP)"] = _lean(snap.get("pct_from_vwap"), scale=0.3, max_pts=20)
    pts["Structure (day range position)"] = _lean((snap.get("range_pos") or 50) - 50, scale=50, max_pts=15)
    pts["Relative volume"] = _lean(((snap.get("rel_volume") or 1) - 1) * 100, scale=50, max_pts=20)
    raw = sum(pts.values())
    score = round(max(0, min(100, 50 + raw / 2)), 1)   # raw ~ -100..+100 -> 0..100
    return {"score": score, "reasons": pts,
            "trend_label": "Strong Bull" if tech.get("ema_aligned_bull") else
                          "Strong Bear" if tech.get("ema_aligned_bear") else "Mixed",
            "momentum_label": "Increasing" if pts["Momentum (RSI)"] > 5 else
                              "Decreasing" if pts["Momentum (RSI)"] < -5 else "Flat",
            "volume_label": "Strong" if (snap.get("rel_volume") or 0) > 1.3 else
                            "Light" if (snap.get("rel_volume") or 1) < 0.7 else "Normal"}


def market_bias(ticker_snap: dict, breadth: dict, vix_snap: dict | None) -> dict:
    """Signed -100..+100 CALLS-vs-PUTS lean, aggregating multiple independent signals — never
    one indicator alone. Displayed as a 0-100 CALLS-lean bar: (raw+100)/2."""
    tech = ticker_snap["tech"]
    pts = {}
    pts["VWAP position"] = _lean(ticker_snap.get("pct_from_vwap"), scale=0.3, max_pts=15)
    if tech.get("ema_aligned_bull"):
        pts["EMA alignment"] = 20.0
    elif tech.get("ema_aligned_bear"):
        pts["EMA alignment"] = -20.0
    else:
        pts["EMA alignment"] = _lean((tech.get("ema20") or 0) - (tech.get("ema50") or 0),
                                     scale=(tech.get("close") or 1) * 0.01, max_pts=10)
    pts["Breadth agreement"] = _lean(breadth.get("score_signed") if breadth else None, scale=50, max_pts=20)
    vix_chg = vix_snap.get("day_change_pct") if vix_snap else None
    pts["VIX trend"] = _lean(-vix_chg if vix_chg is not None else None, scale=2.0, max_pts=15)
    pts["Momentum (RSI)"] = _lean((tech.get("rsi14") or 50) - 50, scale=20, max_pts=15)
    sma200 = tech.get("sma200")
    close = tech.get("close")
    pct_from_sma200 = ((close / sma200 - 1) * 100) if sma200 and close else None
    pts["Price structure (vs 200d)"] = _lean(pct_from_sma200, scale=5, max_pts=15)

    raw = round(sum(pts.values()), 1)
    raw = max(-100.0, min(100.0, raw))
    calls_score = round((raw + 100) / 2, 0)
    agree = sum(1 for v in pts.values() if (v > 0) == (raw > 0) and v != 0)
    confidence = round(min(95, max(30, 50 + abs(raw) / 100 * 40 + agree * 2)), 0)
    if abs(raw) >= 50:
        env = "Trending Bull" if raw > 0 else "Trending Bear"
    elif abs(raw) >= 20:
        env = "Leaning Bull" if raw > 0 else "Leaning Bear"
    else:
        env = "Choppy / Neutral"
    rec = "CALLS ONLY" if raw >= 50 else "PUTS ONLY" if raw <= -50 else \
          "CALLS FAVORED" if raw > 15 else "PUTS FAVORED" if raw < -15 else "NO CLEAR EDGE"
    return {"calls_pct": calls_score, "puts_pct": round(100 - calls_score, 0), "raw_signed": raw,
            "confidence_pct": confidence, "environment": env, "recommendation": rec, "reasons": pts}


def nearest_dated_chain(ticker: str) -> dict | None:
    """The nearest-to-today listed expiry's chain — '0DTE' if the market has one today, else the
    closest available (index ETFs list daily expiries Mon-Fri; best-effort on holidays/gaps)."""
    try:
        expiry = data.nearest_expiry(ticker, 0)
        return data.get_option_chain(ticker, expiry)
    except Exception:
        return None


def options_health(chain: dict, spot: float) -> dict | None:
    """Liquidity/execution-quality read on the nearest-dated chain — reuses det.option_metrics
    for IV/expected-move/greeks, adds spread% and a liquidity/execution composite."""
    if chain is None:
        return None
    try:
        om = det.option_metrics(spot, chain)
    except Exception:
        return None
    calls = chain["calls"].copy()
    calls["_dist"] = (calls["strike"] - spot).abs()
    atm = calls.loc[calls["_dist"].idxmin()]
    bid, ask = float(atm.get("bid") or 0), float(atm.get("ask") or 0)
    mid = (bid + ask) / 2 if bid and ask else None
    spread_pct = ((ask - bid) / mid * 100) if mid else None
    oi = float(atm.get("openInterest") or 0)
    vol = float(atm.get("volume") or 0)

    spread_score = 100 - _lean(spread_pct, scale=10, max_pts=100) if spread_pct is not None else 50
    spread_score = max(0, min(100, spread_score))
    liquidity_score = round(min(100, (min(oi, 5000) / 5000 * 50) + (min(vol, 2000) / 2000 * 50)), 0)
    execution_quality = round((spread_score + liquidity_score) / 2, 0)

    greeks = om.get("atm_call_greeks") or {}
    return {"atm_strike": om.get("atm_strike"), "atm_iv_pct": om.get("atm_iv_pct"),
            "expected_move_pct": om.get("expected_move_straddle_pct"),
            "spread_pct": round(spread_pct, 2) if spread_pct is not None else None,
            "spread_label": "Excellent" if (spread_pct or 99) < 3 else "Fair" if (spread_pct or 99) < 8 else "Wide",
            "oi": oi, "volume": vol, "delta": greeks.get("delta"), "gamma": greeks.get("gamma"),
            "theta_per_day": greeks.get("theta_per_day"), "liquidity_score": liquidity_score,
            "execution_quality": execution_quality}


# NVDA is a component of SPY, QQQ, and DIA (joined the Dow Nov 2024) but NOT IWM (Russell 2000 is
# small-caps only) -- the relative-strength read below is still shown for IWM since it's useful
# cross-asset context, but the note makes clear NVDA isn't part of that index's own math.
_NVDA_INDEX_MEMBER = {"SPY": True, "QQQ": True, "DIA": True, "IWM": False,
                      "SMH": True, "SOXX": True}


def nvda_relative_strength(ticker: str, index_snap: dict, nvda_snap: dict | None,
                           nvda_intraday: pd.DataFrame | None) -> dict | None:
    """PIIP audit 2026-08, Option C. Is NVDA — the single largest weight in MEGA_CAP_WEIGHTS and
    the market's dominant AI/semis bellwether — leading, lagging, or in-line with the selected
    index today? Built entirely from data already in the batched fetch (NVDA is already one of
    MEGA_CAPS), no new network call. A single-stock read, not a market-wide signal -- shown as
    context alongside Mega Cap Health, never as its own trade signal."""
    if nvda_snap is None:
        return None
    spread_pct = round(nvda_snap["day_change_pct"] - index_snap["day_change_pct"], 2)
    score = round(max(0, min(100, 50 + _lean(spread_pct, scale=1.5, max_pts=50))), 0)
    label = "Outperforming" if spread_pct > 0.15 else "Underperforming" if spread_pct < -0.15 else "In-line"
    nvda_momentum = momentum_engine(nvda_intraday) if nvda_intraday is not None else None
    is_member = _NVDA_INDEX_MEMBER.get(ticker, False)
    return {"score": score, "label": label, "spread_pct": spread_pct,
            "nvda_day_change_pct": round(nvda_snap["day_change_pct"], 2),
            "index_day_change_pct": round(index_snap["day_change_pct"], 2),
            "nvda_momentum_label": nvda_momentum["velocity_label"] if nvda_momentum else None,
            "is_index_member": is_member,
            "note": (f"NVDA is a component of {ticker} — this partly reflects {ticker} moving "
                    f"itself, not an independent confirmation." if is_member else
                    f"NVDA is NOT a component of {ticker} ({ticker} is small/mid-cap-weighted) — "
                    "this is cross-asset risk-sentiment context only, not index math.")}


def nvda_relative_strength_multi(index_snaps: dict, nvda_snap: dict | None,
                                 nvda_intraday: pd.DataFrame | None) -> dict:
    """PIIP audit 2026-08, Batch 2 (Phase 16, partial): NVDA relative strength vs all 4 comparison
    tickers at once (SPY/QQQ/SMH/SOXX), not just the currently-selected index -- reuses
    nvda_relative_strength() per ticker rather than a second implementation. True 'ex-NVDA index'
    construction (NVDA's OWN move backed out of the index) needs live constituent WEIGHTS for
    QQQ/SMH/SOXX, which have no reliable free source -- deliberately NOT attempted here, per the
    spec's own instruction not to fabricate an approximation when the underlying data isn't
    available; this stays a straightforward (and honestly labeled) relative-strength comparison."""
    return {tk: nvda_relative_strength(tk, index_snaps[tk], nvda_snap, nvda_intraday)
           for tk in NVDA_COMPARISON_TICKERS if index_snaps.get(tk)}


def _day_change_at(df: pd.DataFrame, target_ts) -> float | None:
    """Day-change % as of the bar closest to (but not after) `target_ts`, from bars already
    fetched -- the building block for nvda_relative_strength_acceleration()'s 3 point-in-time
    reads below."""
    day_open = df["Open"].iloc[0]
    sub = df[df.index <= target_ts]
    if sub.empty or not day_open:
        return None
    return float(sub["Close"].iloc[-1] / day_open - 1) * 100


def nvda_relative_strength_acceleration(index_intraday: pd.DataFrame | None,
                                        nvda_intraday: pd.DataFrame | None) -> dict | None:
    """PIIP audit 2026-08, Batch 2 (Phase 17): is NVDA's leadership over the index ACCELERATING,
    STABLE, or FADING -- computed from 3 actual point-in-time reads (30 min ago, 15 min ago, now)
    taken directly from today's already-fetched intraday bars for both series. Deliberately reads
    straight from the bars rather than accumulating readings in session_state across refreshes,
    so it works the same on the very first load of the day as on the 50th refresh."""
    if index_intraday is None or nvda_intraday is None or index_intraday.empty or nvda_intraday.empty:
        return None
    now_ts = index_intraday.index[-1]
    readings = {}
    for label, minutes_ago in (("30m_ago", 30), ("15m_ago", 15), ("now", 0)):
        target = now_ts - pd.Timedelta(minutes=minutes_ago)
        idx_chg = _day_change_at(index_intraday, target)
        nvda_chg = _day_change_at(nvda_intraday, target)
        readings[label] = round(nvda_chg - idx_chg, 2) if idx_chg is not None and nvda_chg is not None else None
    vals = [readings["30m_ago"], readings["15m_ago"], readings["now"]]
    if any(v is None for v in vals):
        return {"readings": readings, "trend": "Insufficient session history"}
    leg1, leg2 = vals[1] - vals[0], vals[2] - vals[1]
    trend = "Accelerating" if leg1 > 0 and leg2 > 0 else "Fading" if leg1 < 0 and leg2 < 0 else "Stable / Mixed"
    return {"readings": readings, "trend": trend}


def available_strikes(chain: dict, direction: str) -> list[float]:
    """Sorted listed strikes for the given side of the already-fetched chain — feeds the
    contract-picker UI, no new network call."""
    if chain is None:
        return []
    df = chain["calls"] if direction == "CALL" else chain["puts"]
    return sorted(float(s) for s in df["strike"].dropna().unique())


def contract_quote(chain: dict, spot: float, strike: float, direction: str) -> dict | None:
    """Liquidity/greeks read for ONE SPECIFIC contract the user picks — not the auto-selected ATM
    strike options_health() always shows. PIIP audit 2026-08, Option C: options_health() only
    ever answered 'is the ATM strike tradeable', but the user's actual position (or the one
    they're considering) is often a different strike. Same formulas as options_health() (spread%,
    liquidity/execution composite, det.bs_greeks), just re-run for the requested contract instead
    of the nearest-to-spot one. Snaps to the nearest LISTED strike if the exact one isn't quoted."""
    if chain is None:
        return None
    df = chain["calls"] if direction == "CALL" else chain["puts"]
    if df.empty:
        return None
    df = df.copy()
    df["_dist"] = (df["strike"] - strike).abs()
    row = df.loc[df["_dist"].idxmin()]
    matched_strike = float(row["strike"])
    bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
    mid = (bid + ask) / 2 if bid and ask else None
    spread_pct = ((ask - bid) / mid * 100) if mid else None
    oi = float(row.get("openInterest") or 0)
    vol = float(row.get("volume") or 0)
    iv = row.get("impliedVolatility")

    spread_score = 100 - _lean(spread_pct, scale=10, max_pts=100) if spread_pct is not None else 50
    spread_score = max(0, min(100, spread_score))
    liquidity_score = round(min(100, (min(oi, 5000) / 5000 * 50) + (min(vol, 2000) / 2000 * 50)), 0)
    execution_quality = round((spread_score + liquidity_score) / 2, 0)

    greeks = {}
    dte_days = None
    try:
        T, dte_days = det._dte_years(chain["expiry"])
        if T <= 0:
            T = 1 / 365
        if iv and iv == iv and iv > 0:
            greeks = det.bs_greeks(spot, matched_strike, T, 0.045, float(iv), call=(direction == "CALL"))
    except Exception:
        pass

    # Moneyness (PIIP audit 2026-08, Batch 1 / Phase 18): signed distance from spot to strike, as
    # a % of strike, with an ITM/OTM label per the contract's own direction.
    dist_pct = round(abs(spot - matched_strike) / matched_strike * 100, 2) if matched_strike else None
    is_itm = (spot > matched_strike) if direction == "CALL" else (spot < matched_strike)
    moneyness_label = f"{dist_pct}% {'ITM' if is_itm else 'OTM'}" if dist_pct is not None else None

    # Last-trade age (PIIP audit 2026-08, Batch 1 / Phase 18): yfinance's `lastTradeDate` is when
    # this SPECIFIC contract last actually traded -- NOT a live bid/ask timestamp (yfinance quotes
    # aren't individually timestamped). Labeled "last traded", never "quote age", so it isn't
    # mistaken for bid/ask freshness -- a contract that hasn't traded in hours despite a live spot
    # price is itself a real liquidity signal, distinct from spread%.
    last_trade_minutes = None
    ltd = row.get("lastTradeDate")
    if ltd is not None and ltd == ltd:
        try:
            ltd_ts = pd.Timestamp(ltd)
            now_ts = pd.Timestamp.now(tz=ltd_ts.tz) if ltd_ts.tzinfo else pd.Timestamp.now()
            last_trade_minutes = round((now_ts - ltd_ts).total_seconds() / 60, 1)
        except Exception:
            pass

    return {"requested_strike": strike, "matched_strike": matched_strike,
            "strike_snapped": matched_strike != strike, "direction": direction,
            "expiry": chain.get("expiry"), "dte_days": dte_days,
            "moneyness_pct": dist_pct, "moneyness_label": moneyness_label,
            "last_trade_minutes": last_trade_minutes,
            "bid": bid, "ask": ask, "mid": mid,
            "spread_pct": round(spread_pct, 2) if spread_pct is not None else None,
            "spread_label": "Excellent" if (spread_pct or 99) < 3 else "Fair" if (spread_pct or 99) < 8 else "Wide",
            "oi": oi, "volume": vol, "iv_pct": round(float(iv) * 100, 1) if iv and iv == iv else None,
            "delta": greeks.get("delta"), "gamma": greeks.get("gamma"),
            "theta_per_day": greeks.get("theta_per_day"),
            "liquidity_score": liquidity_score, "execution_quality": execution_quality}


def dealer_positioning(chain: dict, spot: float) -> dict | None:
    """ESTIMATED dealer gamma exposure (GEX) from open interest — the standard retail-approximation
    formula (dealers assumed net long gamma from calls, net short from puts). NOT verified against
    real dealer books; yfinance OI is end-of-PRIOR-day, not intraday-live. Context only — nearest
    high-OI strikes are framed as possible support/resistance, never a guaranteed level."""
    if chain is None:
        return None
    try:
        calls, puts, expiry = chain["calls"].copy(), chain["puts"].copy(), chain["expiry"]
        T, _ = det._dte_years(expiry)
    except Exception:
        return None
    if T <= 0:
        T = 1 / 365

    def _row_gamma(row, is_call):
        iv = row.get("impliedVolatility")
        if not iv or iv != iv or iv <= 0:
            return 0.0
        g = det.bs_greeks(spot, float(row["strike"]), T, 0.045, float(iv), call=is_call)
        return g.get("gamma", 0.0) or 0.0

    strikes_gex = {}
    net_gex = 0.0
    for _, row in calls.iterrows():
        oi = float(row.get("openInterest") or 0)
        if oi <= 0:
            continue
        g = _row_gamma(row, True) * oi * 100 * spot * spot * 0.01
        net_gex += g
        strikes_gex[float(row["strike"])] = strikes_gex.get(float(row["strike"]), 0.0) + oi
    for _, row in puts.iterrows():
        oi = float(row.get("openInterest") or 0)
        if oi <= 0:
            continue
        g = _row_gamma(row, False) * oi * 100 * spot * spot * 0.01
        net_gex -= g
        strikes_gex[float(row["strike"])] = strikes_gex.get(float(row["strike"]), 0.0) + oi

    walls = sorted(strikes_gex.items(), key=lambda kv: kv[1], reverse=True)[:3]
    walls = sorted([w[0] for w in walls])
    regime = "Long gamma (dealers dampen moves)" if net_gex > 0 else "Short gamma (dealers amplify moves)"
    nearest_wall = min(walls, key=lambda w: abs(w - spot)) if walls else None
    return {"is_estimate": True,
            "note": "Estimated from open interest using the standard retail-approximation GEX "
                    "formula — NOT verified dealer data. OI is end-of-prior-day, not live intraday.",
            "net_gex_millions": round(net_gex / 1e6, 1), "regime": regime,
            "gamma_wall_strikes": walls, "nearest_wall": nearest_wall,
            "distance_to_wall_pct": round((nearest_wall / spot - 1) * 100, 2) if nearest_wall else None}


def breadth_score(index_snaps: dict, mega_snaps: dict, sector_snaps: dict) -> dict:
    """Basket-proxy breadth — NOT the real NYSE Advance/Decline or TICK (no free source for
    those, confirmed this session). Counts how many indices/mega-caps/sectors are green vs red
    right now. `score_signed` (-100..+100) feeds market_bias's Breadth agreement input."""
    def _green_red(snaps: dict) -> tuple[int, int, int]:
        green = sum(1 for s in snaps.values() if s and (s.get("day_change_pct") or 0) > 0)
        red = sum(1 for s in snaps.values() if s and (s.get("day_change_pct") or 0) < 0)
        return green, red, len(snaps)

    ig, ir, it = _green_red({k: v for k, v in index_snaps.items() if k != "SPY"})
    mg, mr, mt = _green_red(mega_snaps)
    sg, sr, st_ = _green_red(sector_snaps)
    tg, tr, tt = ig + mg + sg, ir + mr + sr, it + mt + st_
    score_signed = round(((tg - tr) / tt) * 100, 1) if tt else 0.0
    label = "Strong" if score_signed > 50 else "Positive" if score_signed > 15 else \
             "Weak" if score_signed < -50 else "Negative" if score_signed < -15 else "Mixed"
    return {"is_proxy": True,
            "note": "Approximates market breadth from a basket (mega-caps + sector ETFs + "
                    "QQQ/IWM/DIA) — NOT the real NYSE Advance/Decline or TICK; no free source "
                    "exists for those (confirmed this session).",
            "score_signed": score_signed, "label": label,
            "green": tg, "red": tr, "total": tt,
            "indices_green": ig, "indices_red": ir,
            "mega_caps_green": mg, "mega_caps_red": mr,
            "sectors_green": sg, "sectors_red": sr}


def sector_health(sector_snaps: dict) -> dict:
    """Per-sector-ETF score + overall confirmation count — is the WHOLE market behind the move,
    or just one or two sectors carrying it?"""
    sectors = {}
    for name, ticker in SECTOR_ETFS.items():
        snap = sector_snaps.get(ticker)
        if not snap:
            continue
        chg = snap.get("day_change_pct") or 0.0
        sectors[name] = {"ticker": ticker, "day_change_pct": round(chg, 2),
                         "score": round(max(0, min(100, 50 + _lean(chg, scale=1.5, max_pts=50))), 0)}
    n = len(sectors)
    confirming = sum(1 for s in sectors.values() if s["day_change_pct"] > 0)
    overall = round((confirming / n) * 100, 0) if n else 0.0
    return {"sectors": sectors, "confirming": confirming, "total": n,
            "overall_confirmation_pct": overall}


def mega_cap_health(mega_snaps: dict) -> dict:
    """Weighted composite using MEGA_CAP_WEIGHTS (approximate influence, see module docstring —
    not official index weights, no free source for those)."""
    total_w, weighted_chg = 0.0, 0.0
    names = {}
    for tk, w in MEGA_CAP_WEIGHTS.items():
        snap = mega_snaps.get(tk)
        if not snap:
            continue
        chg = snap.get("day_change_pct") or 0.0
        weighted_chg += chg * w
        total_w += w
        names[tk] = {"day_change_pct": round(chg, 2), "weight": w}
    avg_chg = (weighted_chg / total_w) if total_w else 0.0
    score = round(max(0, min(100, 50 + _lean(avg_chg, scale=1.5, max_pts=50))), 0)
    return {"score": score, "weighted_avg_change_pct": round(avg_chg, 2), "names": names,
            "is_approximate_weighting": True}


def reversal_engine(momentum: dict | None, tech: dict, snap: dict) -> dict:
    """Does NOT predict tops — estimates whether the current move is strengthening or weakening
    from momentum deceleration, VWAP overextension, RSI exhaustion, and day-range extremes.

    `reversal_pressure_score` (renamed from `reversal_risk_pct`, PIIP audit 2026-08): a hand-
    weighted sum (25/25/30/20-point contributions), not a value calibrated against how often a
    reversal actually followed historically. "risk_pct" implied a calibrated statistical
    percentage this number never was -- it's a heuristic pressure gauge, not a probability."""
    reasons = {}
    if momentum and momentum["velocity_pct"] != 0:
        decel = (momentum["velocity_pct"] > 0 and momentum["acceleration_pct"] < 0) or \
                (momentum["velocity_pct"] < 0 and momentum["acceleration_pct"] > 0)
        reasons["Momentum deceleration"] = 25.0 if decel else 0.0
    else:
        reasons["Momentum deceleration"] = 0.0
    pfv = abs(snap.get("pct_from_vwap") or 0.0)
    reasons["Overextension (vs VWAP)"] = round(min(25.0, pfv / 0.5 * 25.0), 1)
    rsi = tech.get("rsi14")
    reasons["RSI exhaustion"] = 30.0 if (rsi is not None and (rsi > 70 or rsi < 30)) else 0.0
    rp = snap.get("range_pos")
    reasons["Day-range extreme"] = round(min(20.0, max(0.0, (abs((rp or 50) - 50) - 30) / 20 * 20)), 1)

    reversal_pressure = round(min(95.0, max(5.0, sum(reasons.values()))), 0)
    strength = round(max(5.0, 100.0 - reversal_pressure), 0)
    exhaustion_label = "High" if reversal_pressure > 65 else "Moderate" if reversal_pressure > 35 else "Low"
    confidence = round(min(90.0, 50 + abs(reversal_pressure - 50) / 50 * 35), 0)
    return {"strength": strength, "reversal_pressure_score": reversal_pressure,
            "exhaustion_label": exhaustion_label, "confidence_pct": confidence, "reasons": reasons}


def vwap_crossings(intraday_df: pd.DataFrame | None) -> dict | None:
    """PIIP audit 2026-08, Batch 1 (Phase 4): how many times has price crossed the RUNNING
    (as-of-that-bar) VWAP today, and how long ago was the most recent one. A clean trend day
    should cross rarely; a chop day crosses back and forth. Deliberately does NOT interpret a
    given count as bullish/bearish/choppy here -- that relationship is a hypothesis to validate
    against zero_dte_log's collected history (see iip/zero_dte_log.py), not something to assume
    up front. Feeds trend_integrity() below as one input among several."""
    if intraday_df is None or len(intraday_df) < 5:
        return None
    running = det.running_vwap(intraday_df)
    side = np.sign(intraday_df["Close"] - running).replace(0, np.nan).ffill().dropna()
    if side.empty:
        return None
    crossed = side.diff().fillna(0) != 0
    count = int(crossed.sum())
    crossing_times = side.index[crossed]
    last_cross = crossing_times[-1] if len(crossing_times) else None
    now_ts = intraday_df.index[-1]
    minutes_since_last = round((now_ts - last_cross).total_seconds() / 60, 0) if last_cross is not None else None
    return {"count": count, "minutes_since_last": minutes_since_last,
            "current_side": "Above VWAP" if side.iloc[-1] > 0 else "Below VWAP"}


def trend_efficiency(intraday_df: pd.DataFrame | None) -> dict | None:
    """PIIP audit 2026-08, Batch 1 (Phase 5): Kaufman-style path efficiency -- |net change| over
    the session divided by the sum of every bar-to-bar move (the actual path length walked, not
    just where price ended up). A different, more whipsaw-sensitive metric than market_dna.py's
    net_vs_range (which uses today's HIGH-LOW range as the denominator, not the real path) --
    distinguishes CLEAN directional movement from TWO-WAY CHOP even when both cover the same net
    distance. 100% = every bar moved the same direction; near 0% = pure back-and-forth noise."""
    if intraday_df is None or len(intraday_df) < 5:
        return None
    c = intraday_df["Close"]
    net = abs(float(c.iloc[-1] - c.iloc[0]))
    path = float(c.diff().abs().sum())
    efficiency_pct = round(min(100.0, (net / path * 100) if path > 0 else 0.0), 1)
    return {"efficiency_pct": efficiency_pct, "net_move": round(net, 4), "path_length": round(path, 4)}


def trend_integrity(alignment: dict, crossings: dict | None, efficiency: dict | None,
                    confluence: dict) -> dict:
    """PIIP audit 2026-08, Batch 1 (Phase 3): 'how clean and persistent is the trend right now' --
    a SYNTHESIS of measurements already computed elsewhere (timeframe alignment, VWAP-crossing
    frequency, path efficiency, structural confirmation), never a new independent judgment. Same
    itemized, no-black-box shape as every other score in this module. A market up 0.8% on 1
    VWAP crossing and 90% path efficiency reads very differently here than one up 0.8% on 9
    crossings and 30% efficiency, even though Market Bias might score them similarly."""
    pts = {}
    pts["Timeframe alignment"] = round((alignment["alignment_pct"] - 50) / 50 * 30, 1) if alignment["total"] else 0.0
    pts["Trend efficiency"] = round((efficiency["efficiency_pct"] - 50) / 50 * 30, 1) if efficiency else 0.0
    # Fewer crossings = higher integrity -- 5+ crossings in a session reads as fully chopped.
    pts["VWAP discipline"] = round(max(-25.0, 25.0 - (crossings["count"] * 5)), 1) if crossings else 0.0
    conf_ratio = (confluence["agree"] / confluence["total"]) if confluence["total"] else 0.5
    pts["Structural confirmation"] = round((conf_ratio - 0.5) * 2 * 15, 1)
    raw = sum(pts.values())
    score = round(max(0.0, min(100.0, 50 + raw)), 0)
    label = "Clean Trend" if score > 70 else "Choppy / Mixed" if score < 40 else "Developing"
    return {"score": score, "label": label, "reasons": pts}


def no_clear_edge_reasons(bias: dict, confluence: dict, crossings: dict | None,
                          alignment: dict, reversal: dict) -> list[str]:
    """PIIP audit 2026-08, Batch 1 (Phase 21): makes 'NO CLEAR EDGE' an EXPLAINED first-class
    outcome instead of a bare label nobody can act on -- itemizes the specific reasons, pulled
    only from measurements already computed elsewhere on the page. Empty list when there IS a
    real edge (caller only renders this when bias['recommendation'] == 'NO CLEAR EDGE')."""
    reasons = []
    if abs(bias["raw_signed"]) <= 15:
        reasons.append(f"Market Bias has no real lean ({bias['raw_signed']:+.0f}, needs beyond "
                       "±15 to count as directional)")
    if confluence["total"] and confluence["agree"] < confluence["total"] / 2:
        reasons.append(f"Only {confluence['agree']}/{confluence['total']} signals agree with "
                       "each other")
    if alignment["total"] and alignment["alignment_pct"] < 60:
        reasons.append(f"Timeframes disagree ({alignment['agree']}/{alignment['total']} aligned)")
    if crossings and crossings["count"] > 4:
        reasons.append(f"{crossings['count']} VWAP crossings today — choppy, not clean directional")
    if reversal["reversal_pressure_score"] > 60:
        reasons.append(f"Reversal pressure is elevated ({reversal['reversal_pressure_score']:.0f}/100)")
    return reasons


def fetch_historical_5m(ticker: str, period: str = "60d") -> pd.DataFrame | None:
    """PIIP audit 2026-08, Batch 2 (Phase 7): historical 5-minute bars for ONE ticker, for the
    time-of-day-adjusted relative-volume baseline below. Confirmed live (2026-08): yfinance hard-
    caps 1m bars at 8 days lookback but allows 5m bars up to 60 days -- 5m is the only granularity
    that gets a real sample (~40 trading days) for a same-time-of-day comparison; 1m would only
    give ~8 noisy days. A separate, slower-moving fetch from fetch_intraday_batch() (today's 1m
    bars) -- cache this one for hours, not seconds, at the app.py layer."""
    try:
        raw = yf.download(tickers=[ticker], period=period, interval="5m", progress=False,
                          auto_adjust=True)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw[ticker] if ticker in raw.columns.get_level_values(0) else raw.droplevel(1, axis=1)
    return raw.dropna(how="all")


def time_of_day_relative_volume(historical_5m: pd.DataFrame | None,
                                today_intraday: pd.DataFrame | None) -> dict | None:
    """PIIP audit 2026-08, Batch 2 (Phase 7): actual volume accumulated by THIS POINT in today's
    session, divided by the historical AVERAGE volume accumulated by the same point across up to
    60 trading days -- the time-of-day-adjusted read zero_dte.py's existing `rel_volume` field
    explicitly documents it is NOT (that one compares today's running total against a full prior
    day's average, which reads artificially low before the close). Shown ALONGSIDE the existing
    proxy, not replacing it -- this one depends on a slower historical fetch succeeding, the
    existing one never fails as long as today's bars are there."""
    if historical_5m is None or historical_5m.empty or today_intraday is None or today_intraday.empty:
        return None
    hist = historical_5m.copy()
    hist["date"] = hist.index.date
    hist = hist[hist["date"] < date.today()]   # never let today's own partial bars leak into the baseline
    if hist.empty:
        return None
    hist = hist.sort_index()
    hist["minutes"] = hist.groupby("date").cumcount() * 5
    hist["cum_vol"] = hist.groupby("date")["Volume"].cumsum()

    today_minutes = (today_intraday.index[-1] - today_intraday.index[0]).total_seconds() / 60
    bucket = int(today_minutes // 5) * 5
    available_buckets = sorted(hist["minutes"].unique())
    candidates = [m for m in available_buckets if m <= bucket]
    if not candidates:
        return None
    use_bucket = max(candidates)
    at_bucket = hist[hist["minutes"] == use_bucket]
    expected_vol = float(at_bucket["cum_vol"].mean())
    sample_days = int(at_bucket["date"].nunique())
    actual_vol = float(today_intraday["Volume"].sum())
    ratio = round(actual_vol / expected_vol, 2) if expected_vol > 0 else None
    return {"actual_volume": actual_vol, "expected_volume": round(expected_vol, 0), "ratio": ratio,
            "sample_days": sample_days, "minutes_into_session": use_bucket,
            "method": "5m bars, up to 60-trading-day lookback"}


def day_regime(trend_state: dict, integrity: dict, alignment: dict, reversal: dict,
               trend_age_minutes: float | None) -> dict:
    """PIIP audit 2026-08, Batch 2 (Phase 1): 'what kind of trading environment is this right
    now' -- a SYNTHESIS of iip/timeframe.py's trend_state (already directional + hysteresis-based)
    and trend_integrity() above, deliberately NOT a third parallel state machine alongside those
    two (and market_dna.py's separate day-SHAPE classifier, which answers a different question).
    States: INSUFFICIENT DATA, NEUTRAL / CHOP, BULL/BEAR DEVELOPING, BULL/BEAR CONFIRMED, TREND
    WEAKENING, REGIME TRANSITION. Thresholds below are a first-pass guess (10 min / 50 integrity /
    40 integrity / 60 reversal pressure), same as market_dna.py's own THRESHOLDS dict -- flagged
    for calibration once real sessions have been logged (see iip/zero_dte_log.py), not presented
    as final."""
    if alignment["total"] == 0:
        return {"state": "INSUFFICIENT DATA", "direction": None, "trend_age_minutes": trend_age_minutes,
                "reasons": ["No timeframe has enough of today's session printed yet."]}

    # A hysteresis flip is actively brewing -- this takes priority over the current confirmed
    # state, since it's the most decision-relevant fact ("something is changing right now").
    if trend_state["pending"] != trend_state["state"]:
        return {"state": "REGIME TRANSITION", "direction": None, "trend_age_minutes": trend_age_minutes,
                "reasons": [f"Hysteresis pending: {trend_state['pending']} "
                           f"({trend_state['pending_count']}/{trend_state['confirm_reads']} confirms) "
                           f"vs current {trend_state['state']}"]}

    if trend_state["state"] == "Range / Chop":
        return {"state": "NEUTRAL / CHOP", "direction": None, "trend_age_minutes": trend_age_minutes,
                "reasons": [f"Trend State is Range/Chop (alignment {alignment['alignment_pct']:.0f}%)"]}

    direction = "BULL" if trend_state["state"] == "Uptrend" else "BEAR"
    weakening = integrity["score"] < 40 or reversal["reversal_pressure_score"] > 60
    if weakening and (trend_age_minutes or 0) > 10:
        return {"state": "TREND WEAKENING", "direction": direction, "trend_age_minutes": trend_age_minutes,
                "reasons": [f"Trend Integrity {integrity['score']:.0f}/100",
                           f"Reversal Pressure {reversal['reversal_pressure_score']:.0f}/100"]}

    tier = "CONFIRMED" if (trend_age_minutes or 0) >= 10 and integrity["score"] >= 50 else "DEVELOPING"
    age_bit = f"{trend_age_minutes:.0f} min old" if trend_age_minutes is not None else "just started"
    return {"state": f"{direction} {tier}", "direction": direction, "trend_age_minutes": trend_age_minutes,
            "reasons": [f"Trend Integrity {integrity['score']:.0f}/100", f"State is {age_bit}"]}


def data_quality_snapshot(intraday_df: pd.DataFrame | None, chain: dict | None,
                          tf_snapshot: dict) -> dict:
    """PIIP audit 2026-08, Batch 1 (Phase 22): consolidates the freshness/proxy caveats already
    scattered across this page's captions into one panel. Underlying freshness is computed from
    the gap between the last intraday BAR's own timestamp and now -- a real signal (e.g. outside
    session hours the last bar is hours old), not a cache-hit check (the 25s fetch cache already
    guarantees data is at most 25s stale from whatever Yahoo last had, which is a different
    question from whether the market itself has fresh bars to give)."""
    underlying, minutes_stale = "Unknown", None
    if intraday_df is not None and not intraday_df.empty:
        last_bar = intraday_df.index[-1]
        now_ts = pd.Timestamp.now(tz=last_bar.tz) if last_bar.tzinfo else pd.Timestamp.now()
        minutes_stale = round((now_ts - last_bar).total_seconds() / 60, 1)
        underlying = "Fresh" if minutes_stale < 15 else "Stale"
    options_note = ("Live bid/ask quotes; Open Interest is end-of-PRIOR-day, not intraday-live"
                    if chain is not None else "No listed chain available right now")
    timeframe_availability = {label: bool(r.get("available", False)) for label, r in tf_snapshot.items()}
    return {"underlying": underlying, "underlying_minutes_stale": minutes_stale,
            "options_note": options_note, "options_available": chain is not None,
            "timeframe_availability": timeframe_availability}


def confluence_score(bias: dict, breadth: dict, sector: dict, mega: dict,
                     momentum: dict | None, reversal: dict, snap: dict,
                     alignment: dict | None = None) -> dict:
    """How many independent signals actually agree with Market Bias's own lean, itemized.

    market_bias() already computes something like this internally (an "agree" count feeding its
    own confidence number) but never surfaces the count itself. Direction-aware: every check is
    scored FOR the bias's own lean, not "is this bullish" in isolation -- so when bias is bearish,
    a confirming DECLINE correctly counts as agreement, not as a red flag. Same itemized,
    no-black-box shape as every other score in this module.

    `alignment` (optional, from iip.timeframe.timeframe_alignment(), PIIP audit 2026-08 Option B):
    adds one more itemized check for whether the multi-timeframe read agrees with bias's lean.
    Optional and keyword-compatible so existing callers/tests that don't pass it still work
    unchanged -- when omitted the check is simply left out, not scored as a fail."""
    lean = 1 if bias["raw_signed"] >= 0 else -1
    sector_confirm = sector["overall_confirmation_pct"] if lean > 0 else (100 - sector["overall_confirmation_pct"])
    mega_confirm = mega["score"] if lean > 0 else (100 - mega["score"])
    checks = [
        ("Market Bias has a real lean (not neutral/choppy)", abs(bias["raw_signed"]) > 15),
        ("Breadth agrees", (breadth.get("score_signed") or 0) * lean > 10),
        ("Sector Health agrees", sector_confirm > 50),
        ("Mega Cap Health agrees", mega_confirm > 50),
        ("Momentum agrees", bool(momentum) and momentum["velocity_pct"] * lean > 0),
        ("Price is on the bias side of VWAP", (snap.get("pct_from_vwap") or 0) * lean > 0),
        ("Reversal Pressure is low", reversal["reversal_pressure_score"] < 40),
        ("Relative Volume confirms conviction", (snap.get("rel_volume") or 0) > 1.0),
    ]
    if alignment and alignment["total"] > 0:
        bias_dir = "Bullish" if lean > 0 else "Bearish"
        checks.append(("Multi-timeframe alignment agrees", alignment["aligned_direction"] == bias_dir))
    agree = sum(1 for _, ok in checks if ok)
    return {"agree": agree, "total": len(checks), "checks": checks,
            "lean_label": "CALLS" if lean > 0 else "PUTS"}


def entry_quality(bias: dict, health: dict | None, momentum: dict | None,
                  options_health: dict | None, dealer: dict | None, breadth: dict) -> dict:
    """Does NOT tell you to enter — scores whether current conditions favor an entry, itemized.

    Deliberately excludes Market Bias's own trend and breadth components: bias["raw_signed"]
    already fully incorporates EMA alignment and breadth agreement (see market_bias()), so
    re-scoring them here would double-count the same signal when trade_confidence() combines
    this score with bias's own confidence. This function measures INCREMENTAL confirmation only
    — momentum, dealer positioning, this specific ticker's own health, and entry liquidity — on
    top of what bias already found, not a re-hash of bias itself."""
    pts = {}
    pts["Momentum"] = round((((momentum["continuation_score_pct"] if momentum else 50) - 50)
                             / 50 * 30), 1)
    pts["Liquidity"] = round((((options_health["execution_quality"] if options_health else 50) - 50)
                              / 50 * 30), 1)
    pts["Dealer positioning"] = 15.0 if (dealer and "Short gamma" in dealer.get("regime", "")) \
        else (-8.0 if dealer else 0.0)
    pts["Ticker health"] = round((((health["score"] if health else 50) - 50) / 50 * 25), 1)
    raw = sum(pts.values())
    score = round(max(0.0, min(100.0, 50 + raw)), 0)
    risk_label = "Low" if score > 70 else "Moderate" if score > 40 else "High"
    return {"score": score, "risk_label": risk_label, "reasons": pts}


def exit_quality(direction: str, bias: dict, momentum: dict | None, breadth: dict, snap: dict) -> dict:
    """Only rendered once the user manually flags they're in a trade. Never recommends exiting —
    gives context: is the trend/momentum/breadth still supporting the direction they're in?"""
    aligned = (direction == "CALL" and bias["raw_signed"] > 0) or \
              (direction == "PUT" and bias["raw_signed"] < 0)
    momentum_state = "strengthening" if (momentum and momentum["acceleration_pct"] > 0) else \
                     "weakening" if momentum else "unknown"
    notes = [f"Trend {'intact' if aligned else 'deteriorating'} — bias is {bias['environment']}",
             f"Momentum {momentum_state}"]
    dir_sign = 1 if direction == "CALL" else -1
    if (breadth.get("score_signed") or 0) * dir_sign < -20:
        notes.append("Breadth turning against your direction")
    pfv = snap.get("pct_from_vwap") or 0.0
    if abs(pfv) > 0.3:
        notes.append(f"Price {'stretched above' if pfv > 0 else 'stretched below'} VWAP — extended move")
    return {"trend_aligned": aligned, "momentum_state": momentum_state, "notes": notes}


def trade_confidence(bias: dict, entry: dict, options_health: dict | None) -> dict:
    """Final directional-confidence roll-up — explainable, never a black box.

    Deliberately NOT a flat average of bias-confidence + entry-quality + liquidity. Two fixes
    versus the original version: (1) entry_quality() no longer re-derives bias's own trend/
    breadth components, so combining it with bias here no longer double-counts the same
    underlying signal — bias is still weighted higher (65/35) since it's the primary read and
    entry is now genuinely incremental confirmation, not a rehash. (2) Liquidity/execution
    quality is surfaced ONLY as its own warning check, never blended into the score — a wide
    spread is an execution-cost problem, not evidence the DIRECTION is less likely to be right,
    and folding it into one number obscured which of those two things you should actually worry
    about. NOTE: none of these weights are backtested against real historical win-rate yet (see
    project research history) — this fixes internal double-counting and conflation, it does not
    by itself make the number a calibrated probability."""
    score = round(bias["confidence_pct"] * 0.65 + entry["score"] * 0.35, 0)
    checks = [("✓" if abs(bias["raw_signed"]) > 15 else "⚠", f"{bias['environment']} bias"),
              ("✓" if entry["score"] > 60 else "⚠", f"Entry quality {entry['score']:.0f}/100")]
    if options_health:
        checks.append(("✓" if options_health["execution_quality"] > 60 else "⚠",
                       f"Liquidity {options_health['spread_label']} (execution risk, not directional confidence)"))
    return {"score": score, "bias_direction": "CALLS" if bias["raw_signed"] >= 0 else "PUTS",
            "checks": checks}
