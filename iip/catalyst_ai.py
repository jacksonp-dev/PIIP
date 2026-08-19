"""AI document-read layer for the Catalyst Radar's 8-K filings -- per user request: extract
specific key dates and a likely trend from the actual filing text, with the trend's "confidence
based on history" grounded in REAL historical calibration (iip/catalyst_calibration.py), never an
AI-invented number. Same guardrail shape as iip/premarket_ai.py, applied to a different surface:

1. Frozen input -- analyze_filing() fetches the filing text and computes historical calibration
   ONCE, deterministically, before ever calling the model. The AI receives that text + ticker +
   the calibration lookup as read-only context; it never fetches anything itself and never
   decides which historical category applies to itself -- that match is the SAME deterministic
   keyword mechanism catalyst_terminal.py already uses for live headlines
   (catalyst_terminal.matched_keywords()), just applied to the filing's own text instead.

2. Hard schema validation -- key_dates/trend/summary only. The response is REJECTED outright (no
   silent repair) if it contains a confidence number, a probability, a reliability label, or any
   bare numeric leaf of its own. The ONLY historical confidence a user ever sees comes from
   catalyst_calibration.lookup()'s real reliability tier (high/medium/low/single-case, assigned by
   real sample size), rendered separately by the caller -- never restated or invented by the model.
   If no calibration category matches, that's shown explicitly as "no historical calibration for
   this event type," never silently filled in with an AI guess.
"""
from __future__ import annotations

import json
import re

import requests

from . import catalyst_calibration as catcal
from . import catalyst_terminal as ct
from . import sec_edgar

MAX_DOC_CHARS = 15000   # 8-Ks are usually short -- comfortably covers a full filing's real prose
# after tag-stripping, while keeping the prompt small even for a long/table-heavy exhibit.

REQUIRED_FIELDS = {"key_dates", "trend", "summary"}
# Anywhere in the response -- an AI-invented confidence/probability/reliability number is a
# schema violation, not something to silently strip and keep the rest.
FORBIDDEN_KEYS_ANYWHERE = {"confidence", "probability", "likelihood", "score", "reliability"}
ALLOWED_TRENDS = {"Bullish", "Bearish", "Neutral", "Mixed"}


# ------------------------------------------------------------------ fetch + clean ----------

def _clean_filing_html(raw_html: str, max_chars: int = MAX_DOC_CHARS) -> str:
    """Strips HTML/XBRL tags and numeric entities, collapses whitespace, truncates -- isolated
    into its own pure function so the cleaning logic is directly unit-testable without a network
    call."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"&#\d+;", " ", text)   # SEC inline-XBRL docs use &#160; etc. heavily
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def fetch_filing_text(url: str, max_chars: int = MAX_DOC_CHARS) -> str | None:
    """Real filing text, cleaned. None on any failure (network, non-200, empty) -- never raises.
    Same SEC-compliant User-Agent as sec_edgar.py (SEC's fair-access policy requires one on every
    request)."""
    try:
        r = requests.get(url, headers=sec_edgar._HEADERS, timeout=15)
        r.raise_for_status()
    except Exception:
        return None
    cleaned = _clean_filing_html(r.text, max_chars)
    return cleaned if cleaned else None


# ------------------------------------------------------------------ historical calibration ----------

def historical_calibration(ticker: str, filing_text: str) -> dict:
    """Deterministic historical-calibration lookup for a filing -- the SAME keyword matching
    catalyst_terminal.py already uses for live headlines, applied to the filing's own text (plus
    the ticker, so a company-specific category like NVDA's china_restriction_severity_tiers can
    match even if the filing text itself never spells out the company name). {} if nothing
    matches -- callers must show 'no historical calibration' explicitly, never silently omit the
    distinction or let the AI invent a number instead."""
    hits = ct.matched_keywords(f"{ticker} {filing_text}")
    return catcal.lookup(hits)


# ------------------------------------------------------------------ schema validation ----------

def _no_numeric_leaves(obj) -> bool:
    if isinstance(obj, dict):
        return all(_no_numeric_leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_no_numeric_leaves(v) for v in obj)
    if isinstance(obj, bool):
        return True
    if isinstance(obj, (int, float)):
        return False
    return True


def _has_forbidden_key(obj) -> bool:
    if isinstance(obj, dict):
        if any(k in FORBIDDEN_KEYS_ANYWHERE for k in obj):
            return True
        return any(_has_forbidden_key(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_forbidden_key(v) for v in obj)
    return False


def validate_filing_analysis(raw) -> dict | None:
    """Hard validation, no silent repair -- None on ANY violation. A caller that gets None must
    show an explicit invalid-response state, never fall back to guessing what the model meant."""
    if not isinstance(raw, dict):
        return None
    if set(raw.keys()) != REQUIRED_FIELDS:
        return None
    if _has_forbidden_key(raw) or not _no_numeric_leaves(raw):
        return None
    key_dates = raw["key_dates"]
    if not isinstance(key_dates, list):
        return None
    for kd in key_dates:
        if not isinstance(kd, dict) or set(kd.keys()) != {"date", "description"}:
            return None
        if not isinstance(kd["date"], str) or not kd["date"].strip():
            return None
        if not isinstance(kd["description"], str) or not kd["description"].strip():
            return None
    if raw["trend"] not in ALLOWED_TRENDS:
        return None
    if not isinstance(raw["summary"], str) or not raw["summary"].strip():
        return None
    return raw


# ------------------------------------------------------------------ the AI call ----------

def _build_prompt(ticker: str, filing_text: str, calibration: dict) -> str:
    return (
        "You are reading a real SEC 8-K filing for a research tool. Extract facts ONLY from the "
        "document text below -- never invent dates or figures not present in it. You were not "
        "given internet access or any tool to fetch more data; reason only from exactly what's "
        "provided.\n\n"
        f"TICKER: {ticker}\n\n"
        f"FILING TEXT:\n{filing_text}\n\n"
        "HISTORICAL CALIBRATION (already computed by the platform from real researched data -- "
        "do NOT restate this as your own number, just consider it when describing the likely "
        f"trend):\n{json.dumps(calibration, default=str) if calibration else 'None available for this event type.'}\n\n"
        "Return ONLY this JSON shape:\n"
        '{"key_dates": [{"date": "", "description": ""}], "trend": "", "summary": ""}\n'
        "key_dates: every specific date mentioned in the filing text and what happens on it "
        "(effective dates, deadlines, commencement dates, expiry, etc.) -- empty list if none "
        "beyond the filing date itself.\n"
        'trend: exactly one of "Bullish", "Bearish", "Neutral", "Mixed" -- your qualitative read '
        "of what this filing's own content implies for the stock.\n"
        "summary: 2-4 sentences on what the filing actually discloses.\n"
        "Do NOT include a confidence number, a probability, a reliability label, or any bare "
        "numeric field of your own -- the calibration data above (if any) is the ONLY historical "
        "grounding a user should see, shown separately by the platform, never restated by you.")


def analyze_filing(ticker: str, url: str, client) -> dict:
    """Fetches the real filing text, computes real historical calibration deterministically
    (BEFORE the AI ever sees anything), then asks the AI to extract key dates + a qualitative
    trend read -- hard-validated before ever reaching a caller. Returns a dict with `_valid`
    always present; `_calibration` (the real, deterministic lookup) is attached on success so
    callers render it as its own clearly-separate block, not blended into the AI's own text."""
    text = fetch_filing_text(url)
    if text is None:
        return {"_valid": False, "_reason": "Could not fetch the filing document."}
    calibration = historical_calibration(ticker, text)
    prompt = _build_prompt(ticker, text, calibration)
    out, cost = client.call(prompt)
    if out is None:   # dry run
        return {"_valid": False, "_dry_run": True, "_cost": 0.0,
               "_reason": "dry-run (no LLM call)", "_calibration": calibration}
    if isinstance(out, dict) and out.get("_skipped"):
        return {"_valid": False, "_cost": 0.0, "_reason": out["_skipped"], "_calibration": calibration}
    validated = validate_filing_analysis(out)
    if validated is None:
        return {"_valid": False, "_cost": cost, "_raw": out, "_calibration": calibration,
               "_reason": "AI response failed schema validation"}
    return {"_valid": True, "_cost": cost, "_calibration": calibration, **validated}
