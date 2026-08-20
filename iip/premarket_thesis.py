"""SPY Premarket Thesis — PIIP audit 2026-08, deterministic phase.

Spec origin: a ChatGPT-drafted proposal, reviewed against the real PIIP codebase before building
(see project chat history for the full audit), then refined through a second ChatGPT pass the user
brought back. Both passes are incorporated below; the architecture decisions in this docstring are
the AGREED, modified version, not the original draft.

WHAT THIS ANSWERS, deliberately narrower than "predict SPY's close": before the intraday Day
Regime engine (zero_dte.day_regime()) has enough session data to say anything (it always returns
INSUFFICIENT DATA before ~30-150 min of real intraday timeframe alignment exists, see
timeframe.py), is there a premarket/opening lean worth being aware of, and has price action
actually confirmed it yet.

HIERARCHY (explicit, per user request — the whole point is to prevent "indicator soup" where
Premarket Bias / Market Bias / Day Regime / Market DNA / Catalyst Bias all compete for the
trader's attention):

    PREMARKET THESIS  →  OPENING CONFIRMATION  →  DAY REGIME  →  MARKET DNA / DAY TYPE

This module owns only the first stage. It does NOT run in parallel with day_regime() once real
session data exists — the caller (app.py) is responsible for the handoff: show this as PRIMARY
before the open and in the first ~30 minutes after, then defer to day_regime() once
`alignment["total"] > 0` (the same condition day_regime() itself uses to leave INSUFFICIENT DATA).

SCOPE: SPY only, not parameterized per-ticker like the rest of zero_dte.py. The spec's own worked
examples are all "SPY 0DTE MARKET OUTLOOK" with QQQ/IWM/DIA as CONFIRMATION instruments, never as
alternate subjects — this is a premarket INDEX read, not a per-ticker score. NVDA (also selectable
elsewhere on the 0DTE page) has no futures/small-cap/defensive analogs, so a generic version was
not built; this stays SPY-specific by design, not as an oversight.

DIRECTION vs RISK, kept as two SEPARATE outputs (per explicit user correction to the first draft,
which called this "premarket_risk_score" as if it were one number): `direction`
(Bullish/Bearish/Neutral) + `confidence` (0-100) answer "which way does the evidence lean and how
strongly" — `risk_environment` (Low/Normal/Elevated/Extreme) separately answers "how much is going
on right now," driven by VIX level/change, not by which direction things are leaning. A quiet
bearish morning and a violent bearish morning should read as the same DIRECTION but different RISK.

SIGNAL FAMILIES, not one-ticker-one-vote (per explicit user correction): SPY and ES are
substantially the same information; so are QQQ and NQ, IWM and RTY. Blending each pair into one
family score before combining families prevents the same underlying move from being counted
2-3 times and inflating confidence artificially — this is the exact same bucket-not-flatten
pattern zero_dte.participation_state() already uses to fix an analogous double-count in breadth.

THE MACHINE CALCULATES, THE AI EXPLAINS (per explicit user instruction, encoded here as a hard
rule for whenever an LLM layer is added on top of this module later — not built in this phase):
an LLM MAY explain, summarize, and contextualize the deterministic outputs below. It MUST NOT
modify scores, thresholds, direction, confidence, confirmation state, or invalidation state. Any
future AI integration point for this module must pass it the fields below as read-only context,
never let it return an alternate score.

DELIBERATELY NOT IN THIS PHASE (per user instruction — added only if evidence justifies it later):
gold, real S&P-500-wide breadth (the existing basket proxy is used as-is), the LLM narrative
layer, and the AI Thesis History / accuracy-dashboard page.

LIVE vs HISTORICAL, kept ARCHITECTURALLY SEPARATE (PIIP audit 2026-08, fixing a real contamination
bug found in production — see project chat history for the full incident): this module has two
genuinely different questions in play, and they must never share a code path:

  LIVE: "What does the confirmation engine say about the market RIGHT NOW?" — answered by
  log_confirmation_event_once()/get_confirmation_event() (the premarket_confirmation_log table),
  computed every ~30s in app.py from whatever direction the CURRENT live thesis recompute shows.
  This is legitimate and stays exactly as it was — it is NOT being removed or weakened.

  HISTORICAL: "Was the ORIGINAL immutable morning thesis correct, based on what actually happened
  between the open and a specific checkpoint time?" — answered ONLY by
  evaluate_thesis_at_checkpoint(), which is fully self-contained: it re-slices the SAME already-
  fetched intraday bars to an as-of cutoff and re-runs confirmation_conditions()/
  invalidation_conditions() for the ORIGINAL thesis's OWN direction against that as-of snapshot.
  It NEVER reads spy_snap["last"] (today's live quote), and it NEVER reads the live confirmation-
  event log above — a live 9:55 Bearish confirmation event is real and useful information about
  the CURRENT state, but it is not evidence about whether the ORIGINAL 8:45 Bullish thesis was
  right, and treating it as such was exactly the bug: a later live recompute's confirmation could
  silently attach itself to the original thesis's grade. The checkpoint computes its own
  "opposing pressure" reading (confirmation_conditions() run for the OPPOSITE direction, against
  the SAME as-of snapshot) as a clearly-separate field instead.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from . import deterministic as det
from . import zero_dte as zd

DB_PATH = "iip.db"

# ES=F/NQ=F/RTY=F confirmed live (2026-08) to return real premarket 5m bars via yfinance with
# prepost=True, same as every other intraday fetch in this project. GC=F (gold) deliberately
# excluded from v1 per user instruction -- add later only if it's shown to improve checkpoint
# accuracy, not assumed to help.
FUTURES_TICKERS = {"ES": "ES=F", "NQ": "NQ=F", "RTY": "RTY=F"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS premarket_thesis_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  spot_price REAL,
  direction TEXT,
  confidence REAL,
  risk_environment TEXT,
  thesis_json TEXT NOT NULL,
  UNIQUE(date, ticker)
);

CREATE TABLE IF NOT EXISTS premarket_confirmation_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  direction TEXT,
  UNIQUE(date, ticker)
);

CREATE TABLE IF NOT EXISTS premarket_checkpoint_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  checkpoint_label TEXT NOT NULL,
  detail_json TEXT NOT NULL,
  UNIQUE(date, ticker, checkpoint_label)
);
"""


def _migrate(con: sqlite3.Connection) -> None:
    """Additive, best-effort ALTER TABLEs for columns added after premarket_thesis_log/
    premarket_checkpoint_log first shipped live this session -- same pattern
    zero_dte_log.py._migrate() already established (PRAGMA table_info check, swallow the
    'column already exists' case). Per the explicit user requirement: the AI layer's frozen
    input (snapshot_json), its validated response (ai_output_json), and the historical
    backtest's own methodology_version travel with the thesis row itself, and every checkpoint
    is linked back to its ORIGINAL thesis row via thesis_id -- not just an implicit date+ticker
    join, so a future schema/methodology change can never quietly re-associate a checkpoint with
    the wrong thesis."""
    thesis_cols = {row[1] for row in con.execute("PRAGMA table_info(premarket_thesis_log)").fetchall()}
    for col, coltype in (("snapshot_json", "TEXT"), ("ai_output_json", "TEXT"),
                        ("historical_methodology_version", "TEXT")):
        if col not in thesis_cols:
            try:
                con.execute(f"ALTER TABLE premarket_thesis_log ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass
    checkpoint_cols = {row[1] for row in con.execute("PRAGMA table_info(premarket_checkpoint_log)").fetchall()}
    if "thesis_id" not in checkpoint_cols:
        try:
            con.execute("ALTER TABLE premarket_checkpoint_log ADD COLUMN thesis_id INTEGER")
        except sqlite3.OperationalError:
            pass


def _connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)   # 3 CREATE TABLE statements -- execute() only allows one at a time
    _migrate(con)
    return con


# ------------------------------------------------------------------ data fetch ----------

def fetch_futures_batch(period: str = "1d", interval: str = "5m") -> dict[str, pd.DataFrame]:
    """ONE batched request for ES/NQ/RTY -- same discipline as zero_dte.fetch_intraday_batch()
    (never a per-ticker loop). prepost=True since the entire point is reading these BEFORE the
    cash session opens."""
    tickers = list(FUTURES_TICKERS.values())
    try:
        raw = yf.download(tickers=tickers, period=period, interval=interval, prepost=True,
                          group_by="ticker", progress=False, threads=True, auto_adjust=True)
    except Exception:
        return {}
    return zd._split_multi(raw, tickers)


def _futures_change_pct(df: pd.DataFrame | None) -> float | None:
    """Simple session-open-to-last %change for a futures contract -- same shape as every other
    ticker's day_change_pct in this project (det.intraday_snapshot()), not a perfectly-synced
    cross-instrument comparison to SPY's own prior-close gap. Honest simplification: futures trade
    ~23h/day, so 'today's session' here means the fetched window's own first-to-last move, which
    is close enough for a directional family blend, not claimed to be more precise than that."""
    if df is None or df.empty or len(df) < 2:
        return None
    first, last = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
    return round((last / first - 1) * 100, 3) if first else None


def overnight_headlines(limit: int = 8) -> list[dict]:
    """Real, deduplicated, deterministically-scored overnight headlines from the existing
    Catalyst Terminal (Finnhub free feed) -- windowed to since the prior session's ~4:00pm ET
    close, top-N by composite_score. Empty list (never an exception, and never silently identical
    to 'no news happened') if no Finnhub key is configured or the feed fails -- the AI snapshot
    layer is expected to label this UNAVAILABLE, not treat an empty list as 'quiet overnight'."""
    from . import catalyst_terminal as ct
    raw = ct.fetch_raw()
    if not raw:
        return []
    now_utc = datetime.now(timezone.utc)
    since_et = (datetime.now(ZoneInfo("America/New_York"))
               .replace(hour=16, minute=0, second=0, microsecond=0) - timedelta(days=1))
    since_utc = since_et.astimezone(timezone.utc)
    scored = ct.score_and_dedupe(raw, now=now_utc)
    overnight = [h for h in scored if datetime.fromisoformat(h["published"]) >= since_utc]
    return overnight[:limit]


# ------------------------------------------------------------------ signal families ----------

def _family(name: str, *changes_pct: float | None, scale: float, max_pts: float,
           invert: bool = False) -> tuple[str, float, float | None]:
    """One signal family's contribution: average whatever real (non-None) values were passed
    (e.g. SPY's change AND ES's change for the 'Equity direction' family), then scale to a
    +/-max_pts contribution -- exactly the bucket-then-score pattern zero_dte.participation_state()
    already uses, applied here to avoid SPY+ES (or QQQ+NQ, IWM+RTY) being counted as 2-3
    independent votes for the same underlying move. Returns (name, points, blended_raw_pct) so the
    raw blended % survives for the 'Why?' breakdown, not just the scaled points."""
    vals = [v for v in changes_pct if v is not None]
    if not vals:
        return (name, 0.0, None)
    blended = sum(vals) / len(vals)
    scored = -blended if invert else blended
    return (name, zd._lean(scored, scale=scale, max_pts=max_pts), round(blended, 3))


def score_signal_families(spy_snap: dict | None, qqq_snap: dict | None, iwm_snap: dict | None,
                          dia_snap: dict | None, vix_snap: dict | None,
                          futures: dict[str, pd.DataFrame], yields_10y_chg_pts: float | None,
                          wti_chg_pct: float | None, participation: dict) -> dict:
    """The 8 signal families, each ONE bounded contribution regardless of how many correlated
    tickers feed it. Weights (20/15/10/10/15/10/10/10 = 100 max) mirror the proportions
    market_bias() already uses elsewhere in this project (VWAP/EMA/Breadth as the heaviest
    single-ticker inputs there) -- a first-pass allocation, not calibrated, same honesty standard
    as every other unvalidated weighting in this codebase."""
    spy_chg = spy_snap.get("day_change_pct") if spy_snap else None
    qqq_chg = qqq_snap.get("day_change_pct") if qqq_snap else None
    iwm_chg = iwm_snap.get("day_change_pct") if iwm_snap else None
    dia_chg = dia_snap.get("day_change_pct") if dia_snap else None
    vix_chg = vix_snap.get("day_change_pct") if vix_snap else None
    es_chg = _futures_change_pct(futures.get(FUTURES_TICKERS["ES"]))
    nq_chg = _futures_change_pct(futures.get(FUTURES_TICKERS["NQ"]))
    rty_chg = _futures_change_pct(futures.get(FUTURES_TICKERS["RTY"]))

    qqq_spread = (qqq_chg - spy_chg) if qqq_chg is not None and spy_chg is not None else None
    nq_spread = (nq_chg - es_chg) if nq_chg is not None and es_chg is not None else None
    iwm_spread = (iwm_chg - spy_chg) if iwm_chg is not None and spy_chg is not None else None
    rty_spread = (rty_chg - es_chg) if rty_chg is not None and es_chg is not None else None
    dia_spread = (dia_chg - spy_chg) if dia_chg is not None and spy_chg is not None else None

    families = [
        _family("Equity direction (SPY/ES)", spy_chg, es_chg, scale=0.5, max_pts=20),
        _family("Growth relative strength (QQQ/NQ)", qqq_spread, nq_spread, scale=0.5, max_pts=15),
        _family("Small-cap confirmation (IWM/RTY)", iwm_spread, rty_spread, scale=0.5, max_pts=10),
        _family("Defensive/cyclical (DIA)", dia_spread, scale=0.4, max_pts=10),
        _family("Volatility (VIX)", vix_chg, scale=5.0, max_pts=15, invert=True),
        # scale=0.15 -- a 15bp (0.15) daily move in ^TNX's own quoted units is already a big one;
        # the original scale=3.0 (fixed after a live sanity check showed a real +5.5bp move
        # producing an almost invisible -0.2pt contribution) would have needed a ~300bp move to
        # ever saturate, which doesn't happen in a single day -- this family would have read near-
        # zero essentially always, silently contributing nothing regardless of real yield moves.
        _family("Rates (10Y yield)", yields_10y_chg_pts, scale=0.15, max_pts=10, invert=True),
        _family("Commodities (oil)", wti_chg_pct, scale=2.0, max_pts=10, invert=True),
        _family("Breadth (participation proxy)", participation.get("avg_signed"),
               scale=50.0, max_pts=10),
    ]
    return {name: {"points": pts, "raw_pct": raw} for name, pts, raw in families}


def market_state(families: dict) -> dict:
    """Direction + confidence, kept SEPARATE from risk_environment (see module docstring) --
    direction/confidence answer 'which way and how strongly', built from the SAME agreement-count
    confidence shape zero_dte.market_bias() already uses. Neutral band (+/-20) is a first-pass
    guess, not calibrated."""
    raw = round(sum(f["points"] for f in families.values()), 1)
    raw = max(-100.0, min(100.0, raw))
    agree = sum(1 for f in families.values() if (f["points"] > 0) == (raw > 0) and f["points"] != 0)
    confidence = round(min(95, max(30, 50 + abs(raw) / 100 * 40 + agree * 3)), 0)
    direction = "Bullish" if raw >= 20 else "Bearish" if raw <= -20 else "Neutral"
    return {"direction": direction, "confidence": confidence, "raw_signed": raw,
           "agree": agree, "total_families": len(families)}


def risk_environment(vix_snap: dict | None, wti_chg_pct: float | None) -> dict:
    """Separate axis from direction -- 'how much is going on' (magnitude), not 'which way'
    (sign). Driven primarily by VIX itself (the market's own risk gauge) since that's the single
    most direct free proxy available; oil's magnitude is a secondary input per the spec's own
    inflation/rates/growth chain reasoning. Thresholds are a first-pass guess, not calibrated --
    same honesty standard as every other unvalidated threshold in this codebase."""
    vix_level = vix_snap.get("last") if vix_snap else None
    vix_chg = vix_snap.get("day_change_pct") if vix_snap else None
    oil_mag = abs(wti_chg_pct) if wti_chg_pct is not None else 0.0
    if vix_level is None or vix_chg is None:
        return {"level": "UNKNOWN", "vix_level": vix_level, "vix_chg_pct": vix_chg,
               "note": "VIX data unavailable right now."}
    if vix_level > 30 or vix_chg > 20 or oil_mag > 4:
        level = "EXTREME"
    elif vix_level > 22 or vix_chg > 10 or oil_mag > 2:
        level = "ELEVATED"
    elif vix_chg < -10 and vix_level < 15:
        level = "LOW"
    else:
        level = "NORMAL"
    return {"level": level, "vix_level": round(vix_level, 2), "vix_chg_pct": round(vix_chg, 2),
           "oil_chg_pct": round(wti_chg_pct, 2) if wti_chg_pct is not None else None}


# ------------------------------------------------------------------ confirmation / invalidation / permission ----------

def confirmation_conditions(direction: str, snap: dict | None, or15: dict | None,
                            qqq_rel_strength_pct: float | None) -> dict:
    """Named boolean checks for whether ACTUAL price action after the open has confirmed the
    premarket direction -- built entirely from primitives already computed elsewhere (VWAP side,
    opening-range status, QQQ relative strength), never a new judgment. Deliberately N/A for a
    Neutral thesis -- there's nothing directional to confirm."""
    if direction not in ("Bullish", "Bearish"):
        return {"status": "N/A", "checks": [], "confirmed_count": 0, "total": 0,
               "note": "No directional thesis to confirm — direction was Neutral."}
    pct_from_vwap = snap.get("pct_from_vwap") if snap else None
    or_status = or15.get("status") if or15 else None
    if direction == "Bearish":
        checks = [("Below VWAP", pct_from_vwap is not None and pct_from_vwap < 0),
                 ("Opening-range breakdown", or_status == "Below breakdown"),
                 ("QQQ relative weakness", qqq_rel_strength_pct is not None and qqq_rel_strength_pct < 0)]
    else:
        checks = [("Above VWAP", pct_from_vwap is not None and pct_from_vwap > 0),
                 ("Opening-range breakout", or_status == "Above breakout"),
                 ("QQQ relative strength", qqq_rel_strength_pct is not None and qqq_rel_strength_pct > 0)]
    confirmed = sum(1 for _, ok in checks if ok)
    status = "CONFIRMED" if confirmed == len(checks) else "PARTIAL" if confirmed > 0 else "NOT CONFIRMED"
    return {"status": status, "checks": checks, "confirmed_count": confirmed, "total": len(checks)}


def invalidation_conditions(direction: str, snap: dict | None, prev_day: dict | None) -> dict:
    """Symmetric to confirmation_conditions() -- what would prove the thesis WRONG, not just
    unconfirmed. Uses the prior day's close as the reclaim/loss level for both directions
    (simpler and more symmetric than mixing prior high/low), plus VWAP side."""
    if direction not in ("Bullish", "Bearish"):
        return {"status": "N/A", "checks": [], "invalidated_count": 0, "total": 0}
    pct_from_vwap = snap.get("pct_from_vwap") if snap else None
    last = snap.get("last") if snap else None
    prev_close = prev_day.get("close") if prev_day else None
    if direction == "Bearish":
        checks = [("Reclaims VWAP", pct_from_vwap is not None and pct_from_vwap > 0),
                 ("Reclaims prior day's close",
                  last is not None and prev_close is not None and last > prev_close)]
    else:
        checks = [("Loses VWAP", pct_from_vwap is not None and pct_from_vwap < 0),
                 ("Loses prior day's close",
                  last is not None and prev_close is not None and last < prev_close)]
    invalidated = sum(1 for _, ok in checks if ok)
    status = "INVALIDATED" if invalidated == len(checks) else "AT RISK" if invalidated > 0 else "INTACT"
    return {"status": status, "checks": checks, "invalidated_count": invalidated, "total": len(checks)}


def trade_permission(direction: str, confirmation: dict) -> dict:
    """The explicit thesis-strength vs trade-permission split (per user instruction) -- a
    confident directional thesis is NOT a trade signal until price action has actually confirmed
    it. NO_TRADE for Neutral; WAIT until CONFIRMED; only then CALLS_FAVORED/PUTS_FAVORED. This
    layer is exactly what a future option-eligibility engine (per the user's own section 17
    proposal) would gate on -- deliberately returns a clean, small, machine-readable status
    rather than prose, for that future consumer."""
    if direction not in ("Bullish", "Bearish"):
        return {"status": "NO_TRADE", "preferred_direction": None,
               "reason": "No directional edge — thesis is Neutral."}
    preferred = "CALL" if direction == "Bullish" else "PUT"
    if confirmation["status"] != "CONFIRMED":
        return {"status": "WAIT", "preferred_direction": preferred,
               "reason": (f"{direction} thesis not yet confirmed by price action "
                         f"({confirmation['confirmed_count']}/{confirmation['total']} conditions met).")}
    return {"status": "CALLS_FAVORED" if direction == "Bullish" else "PUTS_FAVORED",
           "preferred_direction": preferred,
           "reason": f"{direction} thesis confirmed by price action."}


# ------------------------------------------------------------------ orchestration ----------

def build_thesis(spy_snap: dict, qqq_snap: dict | None, iwm_snap: dict | None, dia_snap: dict | None,
                 vix_snap: dict | None, futures: dict, yields_10y_chg_pts: float | None,
                 wti_chg_pct: float | None, participation: dict, or15: dict | None,
                 prev_day: dict | None, qqq_rel_strength_pct: float | None) -> dict:
    """Assembles the full structured object -- MARKET_STATE / THESIS / CONFIRMATION /
    INVALIDATION / TRADE_PERMISSION, per the user's own section-17 proposal for a future option-
    engine consumer. Every field is a real computed value; nothing here is LLM-authored (see
    module docstring's 'the machine calculates, the AI explains' rule)."""
    families = score_signal_families(spy_snap, qqq_snap, iwm_snap, dia_snap, vix_snap, futures,
                                     yields_10y_chg_pts, wti_chg_pct, participation)
    state = market_state(families)
    risk = risk_environment(vix_snap, wti_chg_pct)
    confirmation = confirmation_conditions(state["direction"], spy_snap, or15, qqq_rel_strength_pct)
    invalidation = invalidation_conditions(state["direction"], spy_snap, prev_day)
    permission = trade_permission(state["direction"], confirmation)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "spot_price": spy_snap.get("last") if spy_snap else None,
        "market_state": state,
        "risk_environment": risk,
        "families": families,
        "confirmation": confirmation,
        "invalidation": invalidation,
        "trade_permission": permission,
    }


# ------------------------------------------------------------------ handoff ----------

# 10:00 ET backstop cutoff (per user's "thesis expiration" request) -- even if, for some reason,
# day_regime()'s own alignment data hasn't populated yet, the premarket thesis stops claiming to
# be the PRIMARY read past this time. The intraday engine's own condition (alignment total > 0)
# is checked FIRST and is expected to fire well before this in practice; this is only a backstop.
PRIMARY_CUTOFF_ET = dtime(10, 0)


def is_primary(now_et: datetime, alignment_total: int) -> bool:
    """Whether the premarket thesis should still be shown as the PRIMARY read (per the explicit
    hierarchy in the module docstring) -- True before the intraday alignment engine has any real
    data AND before the 10:00 ET backstop, False once day_regime() can actually say something.
    The caller keeps rendering the thesis in a smaller/archived form after this flips (for the
    10am-checkpoint feature to have something to check against), just not as the top-of-page read."""
    return alignment_total == 0 and now_et.time() < PRIMARY_CUTOFF_ET


# ------------------------------------------------------------------ persistence (immutable) ----------

def log_thesis_once(ticker: str, thesis: dict, path: str = DB_PATH) -> bool:
    """Logs today's thesis EXACTLY ONCE per ticker per day -- enforced at the schema level
    (UNIQUE(date, ticker), INSERT OR IGNORE), not just app-logic discipline, per the user's
    explicit 'the original thesis must never be overwritten' requirement. Returns True if this
    call actually wrote the row (i.e. this was the first call today), False if a thesis already
    existed and this call was correctly a no-op -- callers can use this to know whether to also
    fire the one-time 'starting collection' UI note."""
    now = datetime.now()
    with _connect(path) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO premarket_thesis_log (ts, date, ticker, spot_price, direction, "
            "confidence, risk_environment, thesis_json) VALUES (?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.date().isoformat(), ticker,
             thesis.get("spot_price"), thesis["market_state"]["direction"],
             thesis["market_state"]["confidence"], thesis["risk_environment"]["level"],
             json.dumps(thesis)))
        return cur.rowcount > 0


def get_todays_thesis(ticker: str, path: str = DB_PATH) -> dict | None:
    """The immutable thesis logged earlier today, or None if nothing's been logged yet. Also
    surfaces the row's own id (as `_thesis_id`, for linking a checkpoint back to this exact row)
    and the AI layer's own frozen snapshot/response if `log_thesis_ai_output()` has been called
    for this row yet (both None otherwise -- a thesis can exist deterministically before the AI
    layer is ever invoked, e.g. if the user hasn't pressed 'Ask AI')."""
    today = date.today().isoformat()
    try:
        con = _connect(path)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT id, ts, thesis_json, snapshot_json, ai_output_json, "
            "historical_methodology_version FROM premarket_thesis_log WHERE ticker=? AND date=?",
            (ticker, today)).fetchone()
        con.close()
    except Exception:
        return None
    if not row:
        return None
    out = json.loads(row["thesis_json"])
    out["_logged_at"] = row["ts"]
    out["_thesis_id"] = row["id"]
    out["_ai_snapshot"] = json.loads(row["snapshot_json"]) if row["snapshot_json"] else None
    out["_ai_output"] = json.loads(row["ai_output_json"]) if row["ai_output_json"] else None
    out["_historical_methodology_version"] = row["historical_methodology_version"]
    return out


def log_thesis_ai_output(ticker: str, snapshot: dict, ai_output: dict,
                         methodology_version: str | None, path: str = DB_PATH) -> bool:
    """Attaches the AI layer's frozen input snapshot + validated (or explicitly invalid) response
    to TODAY's already-logged thesis row -- write-once, same immutability principle as the thesis
    itself: an UPDATE that only ever fires while ai_output_json is still NULL, so pressing 'Ask
    AI' a second time the same morning can never silently replace the first real answer. Returns
    False (no-op) if no thesis row exists yet for today, or if the AI layer was already logged."""
    now = datetime.now()
    with _connect(path) as con:
        cur = con.execute(
            "UPDATE premarket_thesis_log SET snapshot_json=?, ai_output_json=?, "
            "historical_methodology_version=? WHERE ticker=? AND date=? AND ai_output_json IS NULL",
            (json.dumps(snapshot, default=str), json.dumps(ai_output, default=str),
             methodology_version, ticker, now.date().isoformat()))
        return cur.rowcount > 0


def log_confirmation_event_once(ticker: str, direction: str, path: str = DB_PATH) -> bool:
    """Edge-triggered, exactly like zero_dte_log.log_regime_transition() -- records ONLY the
    first moment today confirmation_conditions() reached CONFIRMED (schema-enforced via
    UNIQUE(date, ticker), same immutability pattern as the thesis itself). This is what makes an
    honest 'was this actually tradeable, and how quickly' checkpoint possible -- without a real
    timestamp of when confirmation first happened, tradeability could only be guessed at checkpoint
    time, not measured."""
    now = datetime.now()
    with _connect(path) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO premarket_confirmation_log (ts, date, ticker, direction) "
            "VALUES (?,?,?,?)", (now.isoformat(timespec="seconds"), now.date().isoformat(),
                                 ticker, direction))
        return cur.rowcount > 0


def get_confirmation_event(ticker: str, path: str = DB_PATH) -> dict | None:
    today = date.today().isoformat()
    try:
        con = _connect(path)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT ts, direction FROM premarket_confirmation_log WHERE ticker=? AND date=?",
            (ticker, today)).fetchone()
        con.close()
    except Exception:
        return None
    return {"ts": row["ts"], "direction": row["direction"]} if row else None


# ------------------------------------------------------------------ checkpoint (grading) ----------

def _bars_at_or_before(df: pd.DataFrame | None, cutoff_et: datetime) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    return df[df.index <= pd.Timestamp(cutoff_et)]


def snapshot_as_of(intraday_df: pd.DataFrame | None, cutoff_et: datetime) -> dict | None:
    """A det.intraday_snapshot()-shaped reconstruction using ONLY bars at/before `cutoff_et` --
    the as-of, historical counterpart to the live snapshot this page uses premarket. None if no
    bars exist at/before the cutoff (data gap, or the session hasn't reached that time yet) --
    callers must show DATA UNAVAILABLE explicitly, never silently substitute a later bar or
    today's live quote. The returned dict carries `_actual_data_timestamp` (the real timestamp
    of the last bar actually used) so a caller can always show target-vs-actual honestly."""
    bars = _bars_at_or_before(intraday_df, cutoff_et)
    if bars is None or bars.empty:
        return None
    snap = det.intraday_snapshot(bars)
    if snap is None:
        return None
    snap["_actual_data_timestamp"] = bars.index[-1].isoformat()
    return snap


def _first_confirmed_bar(direction: str, spy_intraday_df: pd.DataFrame | None,
                         qqq_intraday_df: pd.DataFrame | None, market_open_et: datetime,
                         cutoff_et: datetime) -> datetime | None:
    """Scans SPY's own 1-minute bars from `market_open_et` through `cutoff_et` (inclusive),
    re-evaluating confirmation_conditions() for `direction` at EACH bar using only data available
    as of that bar -- returns the timestamp of the FIRST bar where status reached CONFIRMED, or
    None if it never did by the cutoff. Deliberately independent of the live confirmation-event
    log (premarket_confirmation_log) -- see module docstring's LIVE vs HISTORICAL section for why
    that log is the wrong source of truth for this. Cheap: ~30 in-memory pandas slices for a
    9:30-10:00 window, no network calls."""
    if spy_intraday_df is None or spy_intraday_df.empty:
        return None
    window = spy_intraday_df[(spy_intraday_df.index >= pd.Timestamp(market_open_et)) &
                             (spy_intraday_df.index <= pd.Timestamp(cutoff_et))]
    for ts in window.index:
        snap = snapshot_as_of(spy_intraday_df, ts)
        if snap is None:
            continue
        qqq_snap = snapshot_as_of(qqq_intraday_df, ts) if qqq_intraday_df is not None else None
        qqq_rel = (round(qqq_snap["day_change_pct"] - snap["day_change_pct"], 3)
                  if qqq_snap else None)
        or_status = zd.opening_range(_bars_at_or_before(spy_intraday_df, ts), minutes=15)
        conf = confirmation_conditions(direction, snap, or_status, qqq_rel)
        if conf["status"] == "CONFIRMED":
            return ts.to_pydatetime()
    return None


def evaluate_thesis_at_checkpoint(original_thesis: dict, label: str, target_time_et: datetime,
                                  spy_intraday_df: pd.DataFrame | None,
                                  qqq_intraday_df: pd.DataFrame | None,
                                  iwm_intraday_df: pd.DataFrame | None,
                                  dia_intraday_df: pd.DataFrame | None,
                                  vix_intraday_df: pd.DataFrame | None,
                                  prev_day: dict | None, market_open_et: datetime) -> dict:
    """The IMMUTABLE checkpoint (e.g. '10:00 AM'): answers 'was the ORIGINAL morning thesis
    correct, based on what actually happened between the open and `target_time_et`' -- NEVER 'what
    does the engine currently say.' Fully self-contained from historical intraday bars sliced to
    `target_time_et`: does NOT read today's live quote, and does NOT read the live confirmation-
    event log (see module docstring's LIVE vs HISTORICAL section for why that log is the wrong
    source of truth for grading an immutable thesis -- a later live recompute's confirmation must
    never silently attach to the original thesis's grade).

    Reuses ONLY existing primitives -- confirmation_conditions()/invalidation_conditions() (run
    for the ORIGINAL direction AND, separately, the opposing direction, both against the SAME
    as-of snapshot), zero_dte.opening_range(), det.intraday_snapshot() -- never a new indicator.

    Returns a dict with `status` ("OK" or "DATA_UNAVAILABLE") and, when OK, `thesis_verdict`
    (CORRECT / PARTIALLY_CORRECT / WRONG / TOO_EARLY_TO_TELL / NO_CALL) plus every field needed
    to reconstruct why -- see the return statement for the full shape."""
    orig_direction = original_thesis["market_state"]["direction"]
    orig_confidence = original_thesis["market_state"]["confidence"]
    orig_spot = original_thesis.get("spot_price")

    snap = snapshot_as_of(spy_intraday_df, target_time_et)
    if snap is None:
        return {"label": label, "status": "DATA_UNAVAILABLE",
               "target_checkpoint_time": target_time_et.isoformat(), "actual_data_timestamp": None,
               "original_direction": orig_direction, "original_confidence": orig_confidence,
               "original_spot": orig_spot, "thesis_verdict": "DATA_UNAVAILABLE",
               "tradeability": "NONE",
               "tradeability_reason": "No intraday data available as of the checkpoint time."}

    actual_ts = snap.pop("_actual_data_timestamp")
    qqq_snap = snapshot_as_of(qqq_intraday_df, target_time_et) if qqq_intraday_df is not None else None
    qqq_rel = (round(qqq_snap["day_change_pct"] - snap["day_change_pct"], 3) if qqq_snap else None)
    or15 = zd.opening_range(_bars_at_or_before(spy_intraday_df, target_time_et), minutes=15)
    iwm_snap = snapshot_as_of(iwm_intraday_df, target_time_et) if iwm_intraday_df is not None else None
    dia_snap = snapshot_as_of(dia_intraday_df, target_time_et) if dia_intraday_df is not None else None
    vix_snap = snapshot_as_of(vix_intraday_df, target_time_et) if vix_intraday_df is not None else None

    base = {"label": label, "status": "OK",
           "target_checkpoint_time": target_time_et.isoformat(), "actual_data_timestamp": actual_ts,
           "original_direction": orig_direction, "original_confidence": orig_confidence,
           "original_spot": orig_spot, "actual_spy_price": snap.get("last"),
           "actual_spy_change_from_open": (round((snap["last"] / snap["open"] - 1) * 100, 3)
                                           if snap.get("open") else None),
           "vwap_relationship": {"vwap": (round(snap["vwap"], 2) if snap.get("vwap") else None),
                                 "pct_from_vwap": snap.get("pct_from_vwap")},
           "opening_range_behavior": or15, "qqq_relative_strength_pct": qqq_rel,
           "iwm_change_pct": iwm_snap.get("day_change_pct") if iwm_snap else None,
           "dia_change_pct": dia_snap.get("day_change_pct") if dia_snap else None,
           "vix_change_pct": vix_snap.get("day_change_pct") if vix_snap else None}

    if orig_direction == "Neutral":
        return {**base, "thesis_verdict": "NO_CALL",
               "expected_behavior": "No directional thesis was issued (Neutral) — nothing to grade.",
               "confirmation": None, "invalidation": None, "opposing_direction": None,
               "opposing_pressure": None, "tradeability": "NONE",
               "tradeability_reason": "No directional thesis to trade."}

    confirmation = confirmation_conditions(orig_direction, snap, or15, qqq_rel)
    invalidation = invalidation_conditions(orig_direction, snap, prev_day)
    opposing_direction = "Bearish" if orig_direction == "Bullish" else "Bullish"
    opposing_confirmation = confirmation_conditions(opposing_direction, snap, or15, qqq_rel)
    opposing_pressure = {"NOT CONFIRMED": "NONE", "PARTIAL": "DEVELOPING",
                        "CONFIRMED": "CONFIRMED"}[opposing_confirmation["status"]]

    # Deterministic verdict -- priority-ordered, never AI-determined (per explicit requirement).
    if invalidation["status"] == "INVALIDATED" or opposing_pressure == "CONFIRMED":
        verdict = "WRONG"
    elif (confirmation["status"] == "CONFIRMED" and invalidation["status"] == "INTACT"
          and opposing_pressure == "NONE"):
        verdict = "CORRECT"
    elif (confirmation["status"] == "NOT CONFIRMED" and invalidation["status"] == "INTACT"
          and opposing_pressure == "NONE"):
        verdict = "TOO_EARLY_TO_TELL"
    else:
        verdict = "PARTIALLY_CORRECT"

    first_confirmed_ts = _first_confirmed_bar(orig_direction, spy_intraday_df, qqq_intraday_df,
                                              market_open_et, target_time_et)
    if first_confirmed_ts is None:
        tradeability, tradeability_reason = ("NONE", "The original thesis's own confirmation "
                                             "conditions were not met by the checkpoint time.")
    else:
        minutes_to_confirm = (first_confirmed_ts - market_open_et).total_seconds() / 60
        tradeability = ("EXCELLENT" if minutes_to_confirm <= 20 else
                        "GOOD" if minutes_to_confirm <= 60 else "POOR")
        tradeability_reason = (f"The original thesis's own confirmation conditions were first "
                               f"met {minutes_to_confirm:.0f} min after the open.")

    expected_behavior = (", ".join(name for name, _ in confirmation["checks"])
                         if confirmation["checks"] else "N/A")

    return {**base, "expected_behavior": expected_behavior,
           "confirmation": {"status": confirmation["status"], "checks": confirmation["checks"],
                            "confirmed_count": confirmation["confirmed_count"],
                            "total": confirmation["total"]},
           "invalidation": {"status": invalidation["status"], "checks": invalidation["checks"],
                            "invalidated_count": invalidation["invalidated_count"],
                            "total": invalidation["total"]},
           "opposing_direction": opposing_direction, "opposing_pressure": opposing_pressure,
           "thesis_verdict": verdict, "tradeability": tradeability,
           "tradeability_reason": tradeability_reason}


def log_checkpoint(ticker: str, label: str, checkpoint: dict, thesis_id: int | None = None,
                   path: str = DB_PATH) -> bool:
    """Same UNIQUE(date, ticker, checkpoint_label)-enforced immutability as the thesis itself --
    a given checkpoint (e.g. '10:00 AM') is graded once and never silently overwritten.
    `thesis_id` (from get_todays_thesis()'s `_thesis_id`) links this checkpoint back to the
    EXACT original thesis row it's grading, per the explicit user requirement -- not just an
    implicit date+ticker join that a future schema change could silently misassociate."""
    now = datetime.now()
    with _connect(path) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO premarket_checkpoint_log (ts, date, ticker, checkpoint_label, "
            "detail_json, thesis_id) VALUES (?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.date().isoformat(), ticker, label,
             json.dumps(checkpoint), thesis_id))
        return cur.rowcount > 0


def get_checkpoints(ticker: str, target_date: str | None = None, path: str = DB_PATH) -> list[dict]:
    target_date = target_date or date.today().isoformat()
    try:
        con = _connect(path)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT ts, checkpoint_label, detail_json, thesis_id FROM premarket_checkpoint_log "
            "WHERE ticker=? AND date=? ORDER BY ts ASC", (ticker, target_date)).fetchall()
        con.close()
    except Exception:
        return []
    return [{"ts": r["ts"], "label": r["checkpoint_label"], "thesis_id": r["thesis_id"],
            **json.loads(r["detail_json"])} for r in rows]
