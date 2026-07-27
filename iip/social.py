"""Reddit social pulse — "what's trending on WSB." CONTEXT ONLY, never a signal.

Retail sentiment (especially r/wallstreetbets) is noisy and, at extremes, often a CONTRARIAN
indicator — the crowd stampeding into a name is frequently a top. Read it like the news panel;
don't trade on it. Data via ApeWisdom's free API (aggregates WSB mentions; no key). Reddit's own
API blocks unauthenticated access, so we use the aggregator. Best-effort: returns {} / [] on failure.
"""
from __future__ import annotations

import requests

_UA = {"User-Agent": "PIIP-research/0.1 (personal research tool)"}
_APE = "https://apewisdom.io/api/v1.0/filter/{sub}/page/{page}"

# Confirmed via live API check: these 6 of the user's 9 requested subreddits have real data on
# ApeWisdom's free aggregator. valueinvesting / biotechstocks / dividends return empty (not
# tracked there) — full coverage of those would need a real Reddit API (PRAW) integration.
SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options", "pennystocks", "StockMarket"]


def _fetch_sub(sub: str, pages: int = 2) -> dict:
    """{ticker: {rank, mentions, mentions_24h_ago, upvotes}} for one subreddit filter. Best-effort:
    returns {} on failure (aggregator down, unknown/untracked subreddit, rate-limited)."""
    out: dict = {}
    try:
        for p in range(1, pages + 1):
            r = requests.get(_APE.format(sub=sub, page=p), headers=_UA, timeout=10)
            for row in r.json().get("results", []):
                tk = row.get("ticker")
                if tk:
                    out[tk] = {"rank": row.get("rank"),
                               "mentions": row.get("mentions") or 0,
                               "mentions_24h_ago": row.get("mentions_24h_ago") or 0,
                               "upvotes": row.get("upvotes") or 0}
    except Exception:
        pass
    return out


def get_wsb_data(pages: int = 2) -> dict:
    """{ticker: {rank, mentions, mentions_24h_ago, upvotes}} for the most-discussed WSB tickers."""
    return _fetch_sub("wallstreetbets", pages)


def get_multi_sub_data(subs: list[str] = SUBREDDITS, pages: int = 2) -> dict[str, dict]:
    """{subreddit: {ticker: {rank, mentions, mentions_24h_ago, upvotes}}} across several
    subreddits — same free ApeWisdom source as get_wsb_data, just looped per filter. Powers
    iip/reddit_momentum.py. Best-effort per sub; a down/untracked subreddit just comes back {}."""
    return {sub: _fetch_sub(sub, pages) for sub in subs}


def _pack(tk: str, d: dict) -> dict:
    prev = d["mentions_24h_ago"]
    change = ((d["mentions"] - prev) / prev * 100) if prev else None
    return {"ticker": tk, "rank": d["rank"], "mentions": d["mentions"],
            "change_pct": change, "upvotes": d["upvotes"]}


def reddit_trending(limit: int = 15, data: dict | None = None) -> list[dict]:
    """Most-mentioned WSB tickers, ranked, with 24h mention change (what's surging)."""
    data = data if data is not None else get_wsb_data()
    rows = sorted(data.items(), key=lambda kv: (kv[1]["rank"] or 1e9))[:limit]
    return [_pack(tk, d) for tk, d in rows]


def reddit_trending_multi(limit: int = 25, data: dict[str, dict] | None = None) -> list[dict]:
    """Most-mentioned tickers across ALL tracked subreddits combined (not just WSB) -- total
    mentions summed across subs, best (lowest) rank seen in any one sub, and which/how-many
    subreddits are actually discussing it. Same real ApeWisdom data reddit_momentum.py scores,
    just the simple 'what's trending right now' cut (no historical-change scoring needed here)."""
    data = data if data is not None else get_multi_sub_data()
    merged: dict[str, dict] = {}
    for sub, rows in data.items():
        for tk, d in rows.items():
            m = merged.setdefault(tk, {"ticker": tk, "total_mentions": 0, "best_rank": None,
                                       "best_rank_sub": None, "upvotes": 0, "subs": []})
            m["total_mentions"] += d["mentions"]
            m["upvotes"] += d["upvotes"]
            m["subs"].append(sub)
            if d["rank"] and (m["best_rank"] is None or d["rank"] < m["best_rank"]):
                # best_rank is per-SUBREDDIT (ApeWisdom ranks each sub's filter independently),
                # not one global position -- lots of tickers can legitimately be "#1" at once,
                # each in a different subreddit. best_rank_sub records WHICH one, so the UI can
                # say "#1 in r/wallstreetbets" instead of a bare "#1" that reads like a bug when
                # several tickers show it (confirmed via a live screenshot: MU/SPY/NVDA/DTE all
                # showing "#1 best rank" at once looked broken even though the numbers were right).
                m["best_rank"] = d["rank"]
                m["best_rank_sub"] = sub
    return sorted(merged.values(), key=lambda m: -m["total_mentions"])[:limit]


def buzz_for(ticker: str, data: dict) -> dict | None:
    """A single ticker's WSB buzz (rank, mentions, 24h change) — or None if not being discussed."""
    d = data.get(ticker)
    return _pack(ticker, d) if d else None
