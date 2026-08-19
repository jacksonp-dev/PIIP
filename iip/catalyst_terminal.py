"""Catalyst Terminal — free-tier market news aggregation, deduplication, and deterministic
scoring (importance / freshness / persistence / confidence). No LLM in the scoring path — keeps
this backtestable and free, per the project's two-layer architecture (data -> deterministic ->
LLM-on-trial, see BUILD_PLAN.md). An LLM interpretation layer over these scores is a natural v2,
once this deterministic baseline has earned its keep.

Data source: Finnhub's free /news?category=general endpoint (FINNHUB_API_KEY in .env — see
.env.example). Polled at most once per REFRESH_SECONDS (5 min default, not the 30s the original
design doc wanted for price data — a free-tier news API quota does not support that cadence).

Scope note: this catches high-impact political/policy catalysts (tariffs, Fed action, executive
orders, etc. — including market-moving statements attributed to any public figure, Trump
included) through ordinary news coverage, not by integrating directly with X/Twitter or Truth
Social. Those platforms either have no free API (Truth Social) or a free tier too small to be
useful (X/Twitter, ~1,500 reads/month) — see project chat history for the full reasoning.

Readiness for a future News -> Market Reaction feature (PIIP audit 2026-08, state-architecture
review, Phase 10 -- explicitly NOT built yet, this is a readiness check only): confirmed the
existing architecture doesn't block it. `score_and_dedupe()`'s `published` field (below) is
already a real UTC timestamp per headline, which is the one piece a future before/after price
comparison would need to anchor on -- combined with the existing intraday bars (already
timestamped, already fetched every 30s on the 0DTE page) and the point-in-time discipline already
proven in zero_dte_log.compute_forward_outcomes() (never uses data from before the observation
timestamp), a "price/trend state before this headline vs. after" comparison is buildable later
without restructuring anything here. Nothing in this module needs to change to enable it -- it
would live as a new function reading `published` + the existing intraday frame, same pattern as
the rest of this codebase's calibration logging.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import requests
from dotenv import find_dotenv, load_dotenv

from iip import catalyst_calibration as calib

load_dotenv(find_dotenv(usecwd=True))     # keys from .env, never hard-coded — same as iip/agents.py

REFRESH_SECONDS = 300
FINNHUB_URL = "https://finnhub.io/api/v1/news"

# Keyword -> importance points. Deliberately broad enough to catch political/policy catalysts
# alongside routine macro-data releases.
IMPORTANCE_KEYWORDS = {
    "fomc": 30, "federal reserve": 25, "fed chair": 25, "rate hike": 25, "rate cut": 25,
    "interest rate": 15, "powell": 20,
    "tariff": 25, "trump": 15, "white house": 15, "executive order": 20, "sanctions": 20,
    "war": 20, "invasion": 25, "government shutdown": 20, "debt ceiling": 20,
    "cpi": 20, "inflation": 15, "jobs report": 20, "nonfarm payroll": 20, "unemployment": 15,
    "recession": 20, "gdp": 15, "retail sales": 18,
    "circuit breaker": 25, "halted": 15, "flash crash": 30,
    # company names -- lets the calibration layer attach ticker-specific historical context
    "nvidia": 15, "nvda": 15, "tesla": 15, "tsla": 15, "broadcom": 12, "avgo": 12,
    "google": 12, "alphabet": 12, "googl": 12, "meta": 12, "microsoft": 12, "msft": 12,
    "apple": 12, "aapl": 12, "amazon": 12, "amzn": 12, "berkshire": 10, "jpmorgan": 12, "jpm": 12,
    # catalyst-category phrases -- combined with a company name above to find the specific
    # researched calibration (e.g. "Nvidia" + "export restriction" -> NVDA china-restriction tiers)
    "export restriction": 20, "export control": 20, "chip ban": 22, "china export": 18,
    "antitrust": 18, "monopoly ruling": 20, "custom chip": 15, "custom silicon": 15,
    "tpu deal": 15, "vehicle deliveries": 15, "delivery numbers": 12, "taiwan": 15,
}

SOURCE_RELIABILITY = {
    "Reuters": 1.0, "Bloomberg": 1.0, "Associated Press": 1.0, "CNBC": 0.9,
    "Wall Street Journal": 1.0, "MarketWatch": 0.85, "Yahoo": 0.7,
}
DEFAULT_SOURCE_RELIABILITY = 0.6

_STOPWORDS = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "is", "as", "at", "by",
             "with", "its", "after", "says", "over", "into", "amid"}


def fetch_raw(limit: int = 60) -> list[dict]:
    """Empty list (not an exception) if no API key is configured or the request fails — callers
    show 'Insufficient Evidence' / a setup hint rather than crashing the page."""
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        return []
    try:
        resp = requests.get(FINNHUB_URL, params={"category": "general", "token": key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data[:limit] if isinstance(data, list) else []
    except Exception:
        return []


def _normalize(headline: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", headline.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _cluster(items: list[dict], similarity_threshold: float = 0.5) -> list[list[dict]]:
    """Groups near-duplicate headlines (same story, different outlets) via word-set Jaccard
    similarity — cheap and deterministic, no embedding/LLM call needed for this."""
    tokensets = [(_normalize(it.get("headline", "")), it) for it in items]
    used = [False] * len(tokensets)
    clusters: list[list[dict]] = []
    for i, (tok_i, item_i) in enumerate(tokensets):
        if used[i]:
            continue
        group = [item_i]
        used[i] = True
        for j in range(i + 1, len(tokensets)):
            if used[j]:
                continue
            tok_j, item_j = tokensets[j]
            if not tok_i or not tok_j:
                continue
            jaccard = len(tok_i & tok_j) / len(tok_i | tok_j)
            if jaccard >= similarity_threshold:
                group.append(item_j)
                used[j] = True
        clusters.append(group)
    return clusters


def _importance_score(headline: str, summary: str) -> tuple[float, list[str]]:
    text = f"{headline} {summary}".lower()
    # Word-boundary match with an optional trailing "s" (tariff/tariffs, sanction/sanctions) —
    # a plain substring check flags "war" inside "heartwarming"; a `\w*` wildcard suffix instead
    # over-corrects and flags "war" inside "warm"/"warden". Exact word, optional plain plural,
    # is the safe middle ground for short/common-prefix keywords like "war".
    hits = [kw for kw in IMPORTANCE_KEYWORDS if re.search(rf"\b{re.escape(kw)}s?\b", text)]
    score = min(sum(IMPORTANCE_KEYWORDS[kw] for kw in hits), 100.0)
    return score, hits


def matched_keywords(text: str) -> list[str]:
    """Public wrapper around _importance_score()'s keyword matching, for callers that just want
    the matched IMPORTANCE_KEYWORDS hits (e.g. catalyst_ai.py running the SAME deterministic
    matching against an 8-K's own document text, not just live headlines) without needing the
    importance score itself."""
    return _importance_score("", text)[1]


# Simple keyword-valence lean -- same mechanism and honesty level as IMPORTANCE_KEYWORDS above
# (deterministic word matching, no LLM), NOT a backtested directional signal. This session's own
# research found that even real historical calibration (oil spikes, generic "war" headlines)
# rarely produces a reliable directional edge -- so this is deliberately framed as a rough,
# transparent reading-level heuristic, the same honesty standard as Feed's "Model lean" flag
# ("~a coin flip, ONE input only"), not something to trade on by itself.
_BULLISH_WORDS = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "soar", "soars", "record high",
    "ceasefire", "truce", "de-escalat", "pause", "paused", "deal reached", "buyback", "buybacks",
    "upgrade", "upgrades", "rate cut", "stimulus", "bailout", "wins approval", "approved",
    "resume", "resumes", "resumption",
}
_BEARISH_WORDS = {
    "war", "invasion", "sanctions", "crash", "plunge", "plunges", "recession", "miss", "misses",
    "escalat", "attack", "attacks", "blockade", "shutdown", "default", "downgrade", "downgrades",
    "layoffs", "bankruptcy", "collapse", "selloff", "sell-off", "tumbl", "warning", "threat",
    "threatens", "threatening",
}


def headline_lean(headline: str, summary: str = "") -> dict:
    """Rough bullish/bearish read from headline word-valence, purely deterministic (regex word
    match, same as _importance_score) -- NOT a verified signal. Ties or no matches -> neutral."""
    text = f"{headline} {summary}".lower()
    bull = sum(1 for w in _BULLISH_WORDS if re.search(rf"\b{re.escape(w)}", text))
    bear = sum(1 for w in _BEARISH_WORDS if re.search(rf"\b{re.escape(w)}", text))
    if bull > bear:
        return {"emoji": "🟢", "label": "Bullish lean"}
    if bear > bull:
        return {"emoji": "🔴", "label": "Bearish lean"}
    return {"emoji": "⚪", "label": "Neutral / mixed"}


def _freshness_score(published: datetime, now: datetime) -> float:
    age_minutes = max((now - published).total_seconds() / 60.0, 0.0)
    if age_minutes <= 15:
        return 100.0
    if age_minutes >= 240:
        return 0.0
    return round(100.0 * (1 - (age_minutes - 15) / (240 - 15)), 1)


def score_and_dedupe(raw: list[dict], now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    results = []
    for group in _cluster(raw):
        primary = min(group, key=lambda it: it.get("datetime", 0))
        published = datetime.fromtimestamp(primary["datetime"], tz=timezone.utc)
        importance, hits = _importance_score(primary.get("headline", ""), primary.get("summary", ""))
        freshness = _freshness_score(published, now)
        sources = {it.get("source", "unknown") for it in group}
        persistence = min(len(sources) * 15, 100)
        reliability = max((SOURCE_RELIABILITY.get(s, DEFAULT_SOURCE_RELIABILITY) for s in sources),
                          default=DEFAULT_SOURCE_RELIABILITY)
        confidence = round(persistence * 0.5 + reliability * 100 * 0.5, 1)
        composite = round(importance * 0.4 + freshness * 0.3 + persistence * 0.15 + confidence * 0.15, 1)
        results.append({
            "headline": primary.get("headline", ""), "url": primary.get("url"),
            "source_count": len(sources), "sources": sorted(sources),
            "published": published.isoformat(timespec="minutes"),
            "importance": importance, "importance_hits": hits,
            "freshness": freshness, "persistence": persistence, "confidence": confidence,
            "composite_score": composite,
            "calibration": calib.lookup(hits),  # {} if no researched category matches yet
            "lean": headline_lean(primary.get("headline", ""), primary.get("summary", "")),
        })
    return sorted(results, key=lambda r: r["composite_score"], reverse=True)
