"""0DTE Market Structure Map -- deterministic-only, per the user's explicit spec. This is a
VISUALIZATION layer over already-computed PIIP primitives, not a new market-bias engine: it does
NOT replace or override market_state(), day_regime(), trade_permission(), or the Premarket
Thesis -- it exists to make the structural logic (volume profile, VWAP/MA confluence, support/
resistance, acceptance/rejection) that experienced 0DTE traders already reason about visually
explicit, using PIIP's own data.

HONESTY NOTES (per the spec's own explicit instructions):
- Volume profile is built from already-fetched 1-MINUTE OHLCV bars, NOT tick-level trade data
  (which doesn't exist on this project's free tier). Each bar's volume is distributed evenly
  across the price bins its [Low, High] range overlaps -- the standard approximation retail
  charting tools use without tick data. Stated explicitly, never presented as trade-level
  precision.
- Expected High/Low REUSES det.option_metrics()'s existing options-implied expected move (the
  same 0DTE chain already fetched on this page) rather than inventing a new statistical range
  model, per the spec's own explicit "reuse existing methodology" instruction. FROZEN ONCE PER
  (ticker, day) -- per direct user feedback: recomputing this every 30s re-centers the band on
  whatever the CURRENT price happens to be, which defeats its entire purpose as a stable
  reference range that the day's actual price action is compared against. The first successful
  computation each day is persisted (log_expected_range_once()) and reused for the rest of the
  session, the same immutable-once-per-day pattern premarket_thesis.py already uses for the
  morning thesis itself. Honest limitation: "beginning of the day" means the first time this
  code actually runs that day, not literally 9:30 ET if the app wasn't running yet.
- Every threshold below (clustering tolerance, HVN/LVN peak detection, acceptance confirm-bar
  count, approaching distance) is a first-pass, uncalibrated guess -- same honesty standard as
  every other unvalidated threshold in this codebase (premarket_thesis.py, market_dna.py, etc.).
- "Conditional path" / "if accepted" / "if rejected" language is deliberately never a prediction
  -- these are lookups over already-identified structural zones, not new price forecasts.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

DB_PATH = "iip.db"

EXPECTED_RANGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_structure_expected_range (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  expected_high REAL,
  expected_low REAL,
  move_pct REAL,
  method TEXT,
  expiry TEXT,
  dte_days INTEGER,
  spot_at_calc REAL,
  UNIQUE(date, ticker)
);
"""

MIN_BARS_FOR_PROFILE = 15
MIN_BARS_FOR_SWINGS = 11   # 2*lookback(5) + 1


# ------------------------------------------------------------------ volume profile ----------

MAX_WICK_PCT = 1.0   # a >1% wick on a single 1-minute SPY-scale bar is implausible -- catches a
# real live glitch found during this build: a post-market (zero-volume) bar with Low ~4.8% below
# its own Open/Close, which silently corrupted both the volume profile's price range and swing-
# low detection before this filter existed. First-pass threshold, not calibrated.


def _clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drops bars whose Low/High deviate implausibly far from their own Open/Close -- a legitimate
    bar always has Low <= min(Open,Close) and High >= max(Open,Close), but the DEVIATION should
    be small for a 1-minute bar; a multi-percent one-minute wick is a data glitch, not a real
    trade. Filters at the shared bar-loading point so every consumer below benefits."""
    if df.empty:
        return df
    body_lo = df[["Open", "Close"]].min(axis=1)
    body_hi = df[["Open", "Close"]].max(axis=1)
    safe_close = df["Close"].replace(0, np.nan)
    low_wick_pct = ((body_lo - df["Low"]) / safe_close * 100).clip(lower=0)
    high_wick_pct = ((df["High"] - body_hi) / safe_close * 100).clip(lower=0)
    return df[(low_wick_pct <= MAX_WICK_PCT) & (high_wick_pct <= MAX_WICK_PCT)]


def _todays_bars(intraday_df: pd.DataFrame | None, min_bars: int) -> pd.DataFrame | None:
    if intraday_df is None or len(intraday_df) < min_bars:
        return None
    today = intraday_df[intraday_df.index.date == date.today()]
    if len(today) < min_bars:
        today = intraday_df   # best-effort fallback, same pattern as zero_dte.opening_range()
    today = _clean_bars(today)
    return today if len(today) >= min_bars else None


def volume_profile(intraday_df: pd.DataFrame | None, n_bins: int = 40,
                   value_area_pct: float = 0.70) -> dict:
    """Session volume-by-price profile -- POC (point of control), Value Area High/Low. See
    module docstring for the bar-volume-distribution approximation this uses. INSUFFICIENT_DATA
    if fewer than MIN_BARS_FOR_PROFILE bars are available -- an early-session profile from a
    handful of bars isn't a meaningful distribution, shown honestly rather than as false
    precision."""
    today = _todays_bars(intraday_df, MIN_BARS_FOR_PROFILE)
    if today is None:
        n = 0 if intraday_df is None else len(intraday_df)
        return {"status": "INSUFFICIENT_DATA", "n_bars": n}

    lo, hi = float(today["Low"].min()), float(today["High"].max())
    if hi <= lo:
        return {"status": "INSUFFICIENT_DATA", "n_bars": len(today)}

    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_volumes = np.zeros(n_bins)
    for _, row in today.iterrows():
        b_lo, b_hi, vol = float(row["Low"]), float(row["High"]), float(row["Volume"])
        if vol <= 0:
            continue
        if b_hi <= b_lo:
            idx = min(max(int(np.searchsorted(bin_edges, b_lo, side="right")) - 1, 0), n_bins - 1)
            bin_volumes[idx] += vol
            continue
        overlap_lo = np.clip(bin_edges[:-1], b_lo, b_hi)
        overlap_hi = np.clip(bin_edges[1:], b_lo, b_hi)
        overlap = np.maximum(overlap_hi - overlap_lo, 0)
        total_overlap = overlap.sum()
        if total_overlap > 0:
            bin_volumes += vol * overlap / total_overlap

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = float(bin_volumes.sum())
    if total <= 0:
        return {"status": "INSUFFICIENT_DATA", "n_bars": len(today)}

    poc_idx = int(np.argmax(bin_volumes))
    # Value Area: expand outward from POC, adding whichever neighboring bin has more volume,
    # until value_area_pct of total volume is captured -- standard VA construction method.
    lo_i, hi_i = poc_idx, poc_idx
    captured = float(bin_volumes[poc_idx])
    while captured < value_area_pct * total and (lo_i > 0 or hi_i < n_bins - 1):
        left_vol = bin_volumes[lo_i - 1] if lo_i > 0 else -1.0
        right_vol = bin_volumes[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if right_vol >= left_vol:
            hi_i += 1
            captured += bin_volumes[hi_i]
        else:
            lo_i -= 1
            captured += bin_volumes[lo_i]

    return {"status": "OK", "n_bars": len(today), "n_bins": n_bins,
           "bin_edges": bin_edges.tolist(), "bin_centers": bin_centers.tolist(),
           "bin_volumes": bin_volumes.tolist(),
           "poc": round(float(bin_centers[poc_idx]), 2),
           "vah": round(float(bin_edges[hi_i + 1]), 2), "val": round(float(bin_edges[lo_i]), 2),
           "value_area_pct_target": value_area_pct,
           "value_area_pct_actual": round(captured / total, 3)}


def detect_hvn_lvn(profile: dict, max_each: int = 4) -> dict:
    """Local peaks (HVN) / troughs (LVN) in the volume-by-price distribution -- a bin qualifies
    as an HVN if its volume exceeds BOTH neighbors AND the profile's own mean (avoids flagging
    every tiny wiggle in an otherwise-flat region); LVN is the mirror. First-pass, uncalibrated
    thresholds. Keeps only the strongest `max_each` of each, per the explicit 'do not display
    dozens of levels' requirement."""
    if profile.get("status") != "OK":
        return {"status": profile.get("status", "INSUFFICIENT_DATA"), "hvns": [], "lvns": []}
    vols = np.array(profile["bin_volumes"])
    centers = np.array(profile["bin_centers"])
    edges = np.array(profile["bin_edges"])
    if len(vols) < 3:
        return {"status": "INSUFFICIENT_DATA", "hvns": [], "lvns": []}
    mean_vol = float(vols.mean())
    hvns, lvns = [], []
    for i in range(1, len(vols) - 1):
        entry = {"price": round(float(centers[i]), 2),
                 "range": [round(float(edges[i]), 2), round(float(edges[i + 1]), 2)],
                 "volume": round(float(vols[i]), 0)}
        if vols[i] > vols[i - 1] and vols[i] > vols[i + 1] and vols[i] > mean_vol:
            hvns.append(entry)
        elif vols[i] < vols[i - 1] and vols[i] < vols[i + 1] and vols[i] < mean_vol:
            lvns.append(entry)
    hvns.sort(key=lambda h: -h["volume"])
    lvns.sort(key=lambda l: l["volume"])
    return {"status": "OK", "hvns": hvns[:max_each], "lvns": lvns[:max_each]}


def intraday_ema_sma(intraday_df: pd.DataFrame | None, span: int = 50) -> tuple[float | None, float | None]:
    """Intraday EMA/SMA on THIS SESSION's own 1-minute closes -- the SAME formula
    app.py's _render_intraday_candlestick() already plots on the chart, NOT
    deterministic.technical_metrics()'s DAILY ema50/sma50 (a different lookback answering the
    separate 'Daily timeframe' alignment question elsewhere on this page). Using the daily value
    here would be wrong for intraday confluence -- confirmed live during this build (a ~$19 gap
    between the two for the same session, since daily EMA50 averages the last 50 TRADING DAYS,
    not the last 50 minutes)."""
    today = _todays_bars(intraday_df, 1)
    if today is None or today.empty:
        return None, None
    closes = today["Close"]
    ema_val = float(closes.ewm(span=span, adjust=False).mean().iloc[-1])
    sma_val = float(closes.rolling(window=span, min_periods=1).mean().iloc[-1])
    return ema_val, sma_val


# ------------------------------------------------------------------ swing points ----------

def swing_points(intraday_df: pd.DataFrame | None, lookback: int = 5, max_points: int = 4) -> dict:
    """Fractal-style swing high/low detection: a bar qualifies as a swing high if its High is the
    STRICT max within `lookback` bars on both sides (ties excluded, kept simple/unambiguous).
    Real lag by construction -- the most recent `lookback` bars can never yet confirm a swing,
    since confirmation needs bars on both sides. First-pass lookback, uncalibrated."""
    min_bars = lookback * 2 + 1
    today = _todays_bars(intraday_df, min_bars)
    if today is None:
        return {"status": "INSUFFICIENT_DATA", "swing_highs": [], "swing_lows": []}
    highs, lows = today["High"].values, today["Low"].values
    raw_highs, raw_lows = [], []
    for i in range(lookback, len(today) - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        if highs[i] == window_h.max() and (window_h == window_h.max()).sum() == 1:
            raw_highs.append(float(highs[i]))
        window_l = lows[i - lookback:i + lookback + 1]
        if lows[i] == window_l.min() and (window_l == window_l.min()).sum() == 1:
            raw_lows.append(float(lows[i]))

    def _dedup_recent(vals: list[float]) -> list[float]:
        out: list[float] = []
        for v in reversed(vals):
            if v == 0:
                continue
            if not any(abs(v - o) / o < 0.0015 for o in out):
                out.append(v)
            if len(out) >= max_points:
                break
        return [round(v, 2) for v in out]

    return {"status": "OK", "swing_highs": _dedup_recent(raw_highs),
           "swing_lows": _dedup_recent(raw_lows)}


# ------------------------------------------------------------------ level clustering ----------

def collect_levels(vwap: float | None, ema50: float | None, sma50: float | None,
                   opening_range: dict | None, prev_day: dict | None, profile: dict | None,
                   hvn_lvn: dict | None, swings: dict | None) -> list[dict]:
    """Gathers every named structural level into one flat list -- pure aggregation of ALREADY-
    COMPUTED values (nothing here is new math), ready for clustering."""
    levels: list[dict] = []

    def add(name: str, price):
        if price is not None and price == price:   # not NaN
            levels.append({"name": name, "price": float(price)})

    add("VWAP", vwap)
    add("EMA50", ema50)
    add("SMA50", sma50)
    if opening_range:
        add("Opening Range High", opening_range.get("high"))
        add("Opening Range Low", opening_range.get("low"))
    if prev_day:
        add("Prev Day High", prev_day.get("high"))
        add("Prev Day Low", prev_day.get("low"))
        add("Prev Day Close", prev_day.get("close"))
    if profile and profile.get("status") == "OK":
        add("Volume POC", profile.get("poc"))
        add("Value Area High", profile.get("vah"))
        add("Value Area Low", profile.get("val"))
    if hvn_lvn and hvn_lvn.get("status") == "OK":
        for h in hvn_lvn.get("hvns", []):
            add("HVN", h["price"])
        for l in hvn_lvn.get("lvns", []):
            add("LVN", l["price"])
    if swings and swings.get("status") == "OK":
        for sh in swings.get("swing_highs", []):
            add("Swing High", sh)
        for sl in swings.get("swing_lows", []):
            add("Swing Low", sl)
    return levels


def cluster_levels(levels: list[dict], tolerance_pct: float = 0.03) -> list[dict]:
    """Groups levels within tolerance_pct of PRICE of each other into one confluence zone -- the
    same bucket-then-score pattern already established in premarket_thesis.py for signal
    families (never counting correlated inputs as independent), applied here to price levels.

    Compares each candidate to the zone's FIRST (lowest) member, not the most-recently-added one
    -- single-linkage chaining (compare-to-last) let one zone balloon to a $1.58-wide, 13-factor
    blob during this build's own live verification (A close to B, B close to C, but A far from
    C, all merged anyway); anchoring to the zone's own seed caps every zone's width at
    tolerance_pct of that seed, guaranteed. tolerance_pct is a first-pass, uncalibrated guess
    (0.03% is ~$0.23 on a $770 level)."""
    if not levels:
        return []
    sorted_levels = sorted(levels, key=lambda l: l["price"])
    zones = [[sorted_levels[0]]]
    for lvl in sorted_levels[1:]:
        zone_seed = zones[-1][0]["price"]
        if zone_seed and (lvl["price"] - zone_seed) / zone_seed * 100 <= tolerance_pct:
            zones[-1].append(lvl)
        else:
            zones.append([lvl])

    out = []
    for zone in zones:
        prices = [l["price"] for l in zone]
        names = [l["name"] for l in zone]
        out.append({"low": round(min(prices), 2), "high": round(max(prices), 2),
                    "mid": round(sum(prices) / len(prices), 2),
                    "contributors": names, "n_factors": len(names)})
    return out


def classify_zones(zones: list[dict], spot: float, max_each: int = 2) -> dict:
    """Splits clustered zones into resistance (above spot) / support (below spot), keeping only
    the `max_each` nearest per side -- per the explicit 'do not display dozens of levels, pick
    the most meaningful' requirement. Nearest-first, not strongest-first: a weak-but-close zone
    is more immediately decision-relevant than a strong-but-far one."""
    resistance = sorted([z for z in zones if z["mid"] > spot], key=lambda z: z["mid"])[:max_each]
    support = sorted([z for z in zones if z["mid"] < spot], key=lambda z: -z["mid"])[:max_each]
    return {"resistance": resistance, "support": support}


# ------------------------------------------------------------------ acceptance / rejection ----------

def zone_state(zone: dict, intraday_df: pd.DataFrame | None, side: str,
               confirm_bars: int = 2, approach_pct: float = 0.3) -> dict:
    """Deterministic APPROACHING / TESTING / REJECTED (or BOUNCE) / BREAKOUT / ACCEPTANCE_ABOVE
    (or _BELOW) state for one zone, from TODAY's own closes only.

    Definitions (documented per the spec's explicit requirement):
    - ACCEPTANCE: the last `confirm_bars` CONSECUTIVE closes are all beyond the zone on the
      breakout side -- a single close/wick beyond the zone is explicitly NOT enough.
    - BREAKOUT: the most recent close is beyond the zone, but confirm_bars hasn't been met yet.
    - REJECTED/BOUNCE: price has closed INSIDE the zone at some point today, and the most recent
      close is back on the original side.
    - TESTING: the most recent close is inside the zone's [low, high] range.
    - APPROACHING: price hasn't touched the zone yet today, but is within `approach_pct`% of it.
    - NEUTRAL: none of the above (zone isn't currently relevant to price action)."""
    today = _todays_bars(intraday_df, 1)
    if today is None or today.empty:
        return {"state": "UNKNOWN", "detail": "No bars available."}
    closes = today["Close"].values
    lo, hi = zone["low"], zone["high"]

    def _side_of(price: float) -> str:
        if price > hi:
            return "above"
        if price < lo:
            return "below"
        return "inside"

    sides = [_side_of(c) for c in closes]
    last = sides[-1]
    tested = "inside" in sides
    beyond_label = "above" if side == "resistance" else "below"
    original_label = "below" if side == "resistance" else "above"
    accept_state = "ACCEPTANCE_ABOVE" if side == "resistance" else "ACCEPTANCE_BELOW"
    reject_state = "REJECTED" if side == "resistance" else "BOUNCE"

    if len(sides) >= confirm_bars and all(s == beyond_label for s in sides[-confirm_bars:]):
        return {"state": accept_state,
               "detail": f"Last {confirm_bars} closes confirmed {beyond_label} the zone "
                         f"({lo}-{hi})."}
    if last == beyond_label:
        return {"state": "BREAKOUT",
               "detail": f"Most recent close is {beyond_label} the zone, not yet confirmed by "
                         f"{confirm_bars} consecutive closes."}
    if tested and last == original_label:
        return {"state": reject_state,
               "detail": f"Price tested the zone ({lo}-{hi}) and closed back {original_label} it."}
    if tested:
        return {"state": "TESTING", "detail": f"Most recent close is inside the zone ({lo}-{hi})."}
    ref = lo if side == "resistance" else hi
    dist_pct = abs(closes[-1] - ref) / closes[-1] * 100 if closes[-1] else None
    if dist_pct is not None and dist_pct <= approach_pct:
        return {"state": "APPROACHING", "detail": f"Price is {dist_pct:.2f}% from the zone."}
    return {"state": "NEUTRAL", "detail": "Not currently interacting with this zone."}


def _zone_snapshot(zone: dict) -> dict:
    """A decoupled COPY of a zone's display fields only (low/high/mid/contributors/n_factors) --
    never the live mutable dict. conditional_path() below caught a real circular-reference bug
    during this build: a resistance zone's if_rejected can point to a support zone whose own
    if_rejected points back to the first (a real, common case -- they're each other's nearest
    neighbor across spot), and returning live references meant attaching state/path to both
    created an actual reference cycle. A snapshot has nothing further to attach, so it can't."""
    return {"low": zone["low"], "high": zone["high"], "mid": zone["mid"],
           "contributors": list(zone["contributors"]), "n_factors": zone["n_factors"]}


def conditional_path(zone: dict, side: str, other_zones: list[dict], spot: float) -> dict:
    """The next structural zone in each direction from `zone` -- a lookup over already-clustered
    zones, never a new price forecast. 'if_accepted' points further in the breakout direction;
    'if_rejected' points back toward (or past) spot on the original side. Returns snapshots, not
    live zone references -- see _zone_snapshot()."""
    by_mid = sorted(other_zones, key=lambda z: z["mid"])
    if side == "resistance":
        beyond = [z for z in by_mid if z["mid"] > zone["mid"]]
        back = [z for z in by_mid if z["mid"] < spot]
        return {"if_accepted": _zone_snapshot(beyond[0]) if beyond else None,
               "if_rejected": _zone_snapshot(back[-1]) if back else None}
    beyond = [z for z in by_mid if z["mid"] < zone["mid"]]
    back = [z for z in by_mid if z["mid"] > spot]
    return {"if_accepted": _zone_snapshot(beyond[-1]) if beyond else None,
           "if_rejected": _zone_snapshot(back[0]) if back else None}


# ------------------------------------------------------------------ expected range ----------

def expected_range(spot: float | None, option_metrics: dict | None) -> dict:
    """Expected High/Low from det.option_metrics()'s EXISTING options-implied expected move (the
    same 0DTE chain already fetched on this page) -- per the spec's own explicit 'reuse existing
    methodology, do not invent a new one' instruction. Prefers the straddle-based estimate (more
    robust to a single illiquid quote, per option_metrics()'s own docstring), falls back to the
    IV-implied one."""
    if option_metrics is None or spot is None:
        return {"status": "INSUFFICIENT_DATA"}
    move_pct = (option_metrics.get("expected_move_straddle_pct")
               or option_metrics.get("expected_move_iv_pct"))
    if move_pct is None:
        return {"status": "INSUFFICIENT_DATA"}
    method = "straddle" if option_metrics.get("expected_move_straddle_pct") else "iv"
    return {"status": "OK", "expected_high": round(spot * (1 + move_pct / 100), 2),
           "expected_low": round(spot * (1 - move_pct / 100), 2), "move_pct": move_pct,
           "method": method, "expiry": option_metrics.get("expiry"),
           "dte_days": option_metrics.get("dte_days")}


# ------------------------------------------------------------------ expected-range persistence ----------

def _connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(EXPECTED_RANGE_SCHEMA)
    return con


def log_expected_range_once(ticker: str, rng: dict, spot_at_calc: float | None,
                            path: str = DB_PATH) -> bool:
    """Freezes today's Expected High/Low for `ticker` -- write-once (INSERT OR IGNORE,
    UNIQUE(date, ticker)), the same immutable-once-per-day pattern
    premarket_thesis.log_thesis_once() already uses for the morning thesis. No-op (returns
    False) if `rng` isn't a real OK result, or if today's range for this ticker is already
    frozen. See module docstring for why this must be write-once, not recomputed."""
    if rng.get("status") != "OK":
        return False
    now = datetime.now()
    with _connect(path) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO market_structure_expected_range "
            "(ts, date, ticker, expected_high, expected_low, move_pct, method, expiry, "
            "dte_days, spot_at_calc) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.date().isoformat(), ticker,
             rng["expected_high"], rng["expected_low"], rng["move_pct"], rng["method"],
             rng.get("expiry"), rng.get("dte_days"), spot_at_calc))
        return cur.rowcount > 0


def get_todays_expected_range(ticker: str, path: str = DB_PATH) -> dict | None:
    """Today's frozen Expected High/Low for `ticker`, or None if nothing's been computed yet
    today (the caller should then compute fresh via expected_range() and persist it)."""
    today = date.today().isoformat()
    try:
        con = _connect(path)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM market_structure_expected_range WHERE ticker=? AND date=?",
            (ticker, today)).fetchone()
        con.close()
    except Exception:
        return None
    if not row:
        return None
    return {"status": "OK", "expected_high": row["expected_high"], "expected_low": row["expected_low"],
           "move_pct": row["move_pct"], "method": row["method"], "expiry": row["expiry"],
           "dte_days": row["dte_days"], "spot_at_calc": row["spot_at_calc"], "frozen_at": row["ts"]}


# ------------------------------------------------------------------ orchestration ----------

def build_structure_map(intraday_df: pd.DataFrame | None, spot: float | None, vwap: float | None,
                        opening_range: dict | None, prev_day: dict | None,
                        option_metrics: dict | None, ticker: str = "SPY",
                        max_zones_each_side: int = 2, path: str = DB_PATH) -> dict:
    """Assembles the full Market Structure Map from already-fetched/computed PIIP data -- the
    ONE entry point the UI layer calls. Every sub-piece above already does its own work (and is
    independently testable); this sequences them and packages the result. Does NOT compute a new
    market-bias/direction number -- see module docstring.

    Expected High/Low is the one piece of real persistence here: today's frozen value for
    `ticker` is reused if it already exists; otherwise it's computed fresh from `option_metrics`
    and frozen for the rest of the day (see log_expected_range_once())."""
    profile = volume_profile(intraday_df)
    hvn_lvn = detect_hvn_lvn(profile)
    swings = swing_points(intraday_df)
    ema50, sma50 = intraday_ema_sma(intraday_df)
    levels = collect_levels(vwap, ema50, sma50, opening_range, prev_day, profile, hvn_lvn, swings)
    zones = cluster_levels(levels)
    classified = (classify_zones(zones, spot, max_each=max_zones_each_side) if spot
                 else {"resistance": [], "support": []})

    for side, side_zones in classified.items():
        for z in side_zones:
            z["state"] = zone_state(z, intraday_df, side)
            z["path"] = conditional_path(z, side, zones, spot) if spot else None

    exp_range = None
    try:
        exp_range = get_todays_expected_range(ticker, path=path)
    except Exception:
        pass
    if exp_range is None:
        exp_range = expected_range(spot, option_metrics)
        if exp_range.get("status") == "OK":
            # Same fields get_todays_expected_range() would return on a later, persisted read --
            # set here too so the very FIRST render of the day (before anything's persisted yet)
            # shows an identical, consistent tooltip, not a briefly-different one.
            exp_range = {**exp_range, "spot_at_calc": spot,
                        "frozen_at": datetime.now().isoformat(timespec="seconds")}
        try:
            log_expected_range_once(ticker, exp_range, spot, path=path)
        except Exception:
            pass   # best-effort persistence -- still show today's freshly-computed value either way

    return {"spot": spot, "vwap": vwap, "ema50": ema50, "sma50": sma50,
           "volume_profile": profile, "hvn_lvn": hvn_lvn, "swings": swings,
           "zones": classified, "expected_range": exp_range}
