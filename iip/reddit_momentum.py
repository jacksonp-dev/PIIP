"""Reddit Momentum Research Engine — Phase 1.

NOT a stock picker. Detects unusually rapid Reddit-discussion increases (free, no key, via
ApeWisdom across 6 subreddits — see iip/social.py), scores momentum deterministically from real
fields only, and points confirmed candidates at the EXISTING Deep Research / Catalyst Radar
machinery for the "why / is it real" step, instead of rebuilding fundamental/price/options
analysis from scratch.

What this deliberately does NOT do (scoped out of Phase 1, not silently dropped):
  * No AI classifies-every-post quality/catalyst scoring — cost-gated, needs an explicit
    call-count + $ estimate first, same as every other LLM use in this codebase.
  * No historical win-rate/return/Sharpe backtesting vs SPY — no historical Reddit-spike data
    exists anywhere to compute that from. `log_snapshot()` starts building that history for real,
    starting now; backtesting becomes honest once weeks/months of it exist.
  * No institutional (13F/dark pool) or IV-rank confirmation — already-documented "no free
    source" gaps in deep_research.py's own unknowns() list.
  * Only 6 of 9 originally-requested subreddits (see iip/social.py::SUBREDDITS) — the other 3
    (valueinvesting, biotechstocks, dividends) aren't tracked by the free aggregator; full
    coverage would need a real Reddit API (PRAW) integration.

Every prediction this logs is pre-registered (timestamped before the outcome is known) and graded
later by the existing Scorecard — this IS the spec's "Hypothesis Lab" / "Learning System", not a
new system. Logging a 'bullish' direction is the hypothesis under test ("does a Reddit momentum
spike precede a price gain?"), not an assertion that Reddit is right — never assume that.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

from . import assess, data, fundamentals, sec_edgar, social
from . import deep_research as dr
from . import predictions as pred

DB_PATH = "iip.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reddit_momentum_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scanned_date TEXT, scanned_ts TEXT, ticker TEXT, subreddit TEXT,
  rank INTEGER, mentions INTEGER, mentions_24h_ago INTEGER, upvotes INTEGER,
  UNIQUE(scanned_date, ticker, subreddit)
);
"""


def _con(db):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def init_db(db: str = DB_PATH) -> None:
    con = _con(db)
    con.executescript(SCHEMA)
    con.commit()
    con.close()


def log_snapshot(db: str = DB_PATH, multi: dict | None = None) -> dict:
    """Fetch (or reuse a passed-in) multi-subreddit dataset and upsert today's rows — idempotent,
    re-scanning the same day updates rather than duplicates. Returns the raw multi dict so scan()
    doesn't have to fetch twice."""
    init_db(db)
    multi = multi if multi is not None else social.get_multi_sub_data()
    today = date.today().isoformat()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = _con(db)
    for sub, rows in multi.items():
        for tk, d in rows.items():
            con.execute(
                "INSERT INTO reddit_momentum_history"
                "(scanned_date,scanned_ts,ticker,subreddit,rank,mentions,mentions_24h_ago,upvotes) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(scanned_date,ticker,subreddit) DO UPDATE SET "
                "scanned_ts=excluded.scanned_ts, rank=excluded.rank, mentions=excluded.mentions, "
                "mentions_24h_ago=excluded.mentions_24h_ago, upvotes=excluded.upvotes",
                (today, ts, tk, sub, d.get("rank"), d.get("mentions"),
                 d.get("mentions_24h_ago"), d.get("upvotes")))
    con.commit()
    con.close()
    return multi


def _prior_state(db: str, ticker: str) -> tuple[int | None, bool]:
    """(prior_best_rank, is_new_appearance) from history strictly BEFORE today. is_new=True if
    this ticker has never been logged before (no prior day at all)."""
    init_db(db)
    today = date.today().isoformat()
    con = _con(db)
    rows = con.execute(
        "SELECT rank, scanned_date FROM reddit_momentum_history "
        "WHERE ticker=? AND scanned_date<? ORDER BY scanned_date DESC", (ticker, today)).fetchall()
    con.close()
    if not rows:
        return None, True
    most_recent = rows[0]["scanned_date"]
    ranks = [r["rank"] for r in rows if r["scanned_date"] == most_recent and r["rank"]]
    return (min(ranks) if ranks else None), False


def compute_momentum(today_rows: dict, prior_best_rank: int | None, is_new: bool) -> dict:
    """Deterministic 0-100 score from fields ApeWisdom actually gives us — mention change, how
    many of the 6 tracked subreddits mention it right now, rank vs the prior logged day, and
    whether it's a brand-new appearance. No fabricated dimensions (no unique users/posts/comment
    scores — that needs post-level Reddit API access this phase doesn't have)."""
    n_subs = len(today_rows)
    total_mentions = sum(r["mentions"] for r in today_rows.values())
    total_prior = sum(r["mentions_24h_ago"] for r in today_rows.values())
    if total_prior > 0:
        mention_change_pct = (total_mentions - total_prior) / total_prior * 100
    elif total_mentions > 0:
        mention_change_pct = 999.0   # brand-new attention, no prior baseline to compare against
    else:
        mention_change_pct = 0.0
    # best_rank is per-SUBREDDIT (ApeWisdom ranks each sub's own filter independently), not one
    # global position -- many tickers can legitimately be "#1" simultaneously, each in a
    # different subreddit. best_rank_sub records WHICH one so the UI can say "#1 in
    # r/wallstreetbets" instead of a bare "#1" that reads like a bug when several tickers show it.
    ranked = [(sub, r["rank"]) for sub, r in today_rows.items() if r.get("rank")]
    best_rank_sub, best_rank = min(ranked, key=lambda sr: sr[1]) if ranked else (None, None)

    change_capped = max(-50.0, min(mention_change_pct, 300.0))
    change_pts = (change_capped + 50) / 350 * 50                    # [-50%,+300%] -> [0,50]
    sub_pts = min(n_subs, 5) / 5 * 25                                # up to 5 subs -> 25 pts
    rank_pts = 15 if (best_rank or 1e9) <= 10 else 10 if (best_rank or 1e9) <= 25 \
        else 5 if (best_rank or 1e9) <= 50 else 0
    new_pts = 10 if is_new else 0

    rank_change = None
    rank_change_pts = 0.0
    if prior_best_rank is not None and best_rank is not None:
        rank_change = prior_best_rank - best_rank                    # positive = improved
        rank_change_pts = max(0, min(rank_change, 20)) / 20 * 10

    score = round(min(100.0, change_pts + sub_pts + rank_pts + new_pts + rank_change_pts), 1)
    return {"score": score, "n_subs": n_subs, "total_mentions": total_mentions,
            "mention_change_pct": round(mention_change_pct, 1), "best_rank": best_rank,
            "best_rank_sub": best_rank_sub, "rank_change": rank_change,
            "is_new_appearance": is_new, "subs": sorted(today_rows.keys())}


def scan(db: str = DB_PATH, min_score: float = 60.0,
         subs: list[str] = social.SUBREDDITS, pages: int = 2) -> list[dict]:
    """Fetch current multi-subreddit Reddit data, log it (real history, growing over time), score
    every ticker seen, and return candidates crossing `min_score` sorted descending. Deliberately
    NOT filtered to scanner.UNIVERSE — the point is surfacing tickers you wouldn't already be
    scanning."""
    multi = social.get_multi_sub_data(subs=subs, pages=pages)
    log_snapshot(db, multi)

    by_ticker: dict[str, dict] = {}
    for sub, rows in multi.items():
        for tk, d in rows.items():
            by_ticker.setdefault(tk, {})[sub] = d

    candidates = []
    for tk, today_rows in by_ticker.items():
        prior_rank, is_new = _prior_state(db, tk)
        m = compute_momentum(today_rows, prior_rank, is_new)
        if m["score"] >= min_score:
            m["ticker"] = tk
            candidates.append(m)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def catalyst_summary(ticker: str) -> dict:
    """Upcoming-catalyst date + why, plus the most recent 8-K if any — reuses the same free
    sources as Deep Research / Catalyst Radar (fundamentals.next_earnings, ClinicalTrials.gov,
    SEC EDGAR) without building the full dossier, which is too slow to do for every card on every
    scan. This is the module's answer to 'dates when a catalyst is expected, and why.'"""
    out: dict = {"next_catalyst": None, "next_catalyst_why": None}
    earn = fundamentals.next_earnings(ticker)
    if earn and earn.get("days") is not None and -3 <= earn["days"]:
        # only counts as explaining a CURRENT spike if it's upcoming or very recently reported —
        # a stale earnings date from months ago isn't why Reddit is talking about it today.
        d = earn["days"]
        out["next_catalyst"] = (f"Earnings {earn['date']} ({d}d away)" if d >= 0
                                 else f"Earnings {earn['date']} ({-d}d ago)")
        out["next_catalyst_why"] = "Known dated event — inflates IV, then IV-crush after."

    if not out["next_catalyst"]:
        info = fundamentals.company_info(ticker)
        sector, industry = (info.get("sector") or ""), (info.get("industry") or "")
        is_biotech = "Healthcare" in sector or any(
            k in industry.lower() for k in ("biotech", "drug", "pharma"))
        if is_biotech:
            trials = dr.clinical_trials(info.get("name") or ticker, limit=3)
            near = [t for t in trials if t.get("primary_completion")]
            if near:
                soonest = min(near, key=lambda t: t["primary_completion"])
                out["next_catalyst"] = f"Trial completion ~{soonest['primary_completion']} ({soonest['phase']})"
                out["next_catalyst_why"] = ("Primary completion ≈ when data is collected; "
                                            "readout/PR usually follows.")

    filings = sec_edgar.recent_filings(ticker, forms=("8-K",), lookback_days=14)
    out["recent_8k"] = filings[0] if filings else None
    return out


def risk_flag(momentum: dict, catalyst: dict) -> dict:
    """Reuses assess.py's explained-vs-unexplained framing, applied to Reddit context: high
    momentum with no known catalyst is this module's honest version of 'meme/pump risk' — NOT an
    LLM pump-language classifier (that's cost-gated, deferred)."""
    has_catalyst = bool(catalyst.get("next_catalyst")) or bool(catalyst.get("recent_8k"))
    if has_catalyst:
        return {"color": assess.GREEN, "label": "Catalyst-linked",
                "detail": "A known dated event or recent filing lines up with this spike."}
    if momentum["score"] >= 80:
        return {"color": assess.RED, "label": "Unexplained spike",
                "detail": "Very high Reddit momentum with no known earnings/filing/trial catalyst "
                          "found — could be a real early signal or pure hype. Research before acting."}
    return {"color": assess.YELLOW, "label": "Unconfirmed",
            "detail": "Momentum present but no dated catalyst found yet."}


def log_prediction(ticker: str, momentum: dict, db: str = DB_PATH) -> int | None:
    """Pre-registers this momentum reading as a graded call (source='reddit_momentum'), same
    dedup-per-day pattern as scanner.log_card_forecast. Direction is 'bullish' by construction —
    that's the hypothesis under test ('does Reddit momentum precede a price gain?'), not a claim
    that it's true. The Scorecard will show honestly whether it historically means anything."""
    try:
        spot = data.get_spot(ticker)
    except Exception:
        spot = None
    rec = {"ticker": ticker, "horizon_days": 30, "source": "reddit_momentum", "spot": spot,
           "stock_direction": "bullish", "confidence": momentum["score"] / 100,
           "reasoning": {"mode": "reddit_momentum", "momentum_score": momentum["score"],
                         "mention_change_pct": momentum["mention_change_pct"],
                         "n_subs": momentum["n_subs"], "best_rank": momentum["best_rank"],
                         "subs": momentum["subs"]}}
    return pred.log_if_new(rec, db)
