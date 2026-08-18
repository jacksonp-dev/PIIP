"""AI interpretation layer for the SPY Premarket Thesis -- ChatGPT-refined spec (2026-08-18): 7
fixed questions, ONE structured call, deterministic-inputs-only. Three guardrails, all explicit
user requirements, all enforced in code (not just discipline):

1. FROZEN SNAPSHOT (build_snapshot()): one immutable dict of everything the AI will see. The AI
   call function (ask_ai_thesis()) receives ONLY this snapshot and never fetches additional
   market data itself -- a thesis generated at 8:45 is reproducible: given the same stored
   snapshot, the exact same prompt is rebuilt. AI_LAYER_VERSION is stored alongside every
   snapshot/response so a future prompt/schema change can't retroactively change what an
   already-logged thesis's AI read meant.

2. HARD SCHEMA VALIDATION (validate_ai_response()): the model's JSON is never trusted just
   because the prompt asked for it. Checks: exactly the 7 required sections, every section has a
   non-empty "analysis" string, synthesis.verdict is one of the 4 allowed values, NO forbidden
   field anywhere (an AI-generated confidence number, a trade-state/trade-permission/position-
   size/stop-loss field), and NO bare numeric leaf anywhere in the response -- every number this
   tool shows a user must trace back to something PIIP itself computed and put in the snapshot,
   never a number the model introduced. On failure: an explicit invalid marker, never a
   silent-repair attempt.

3. TRADE STATE OWNERSHIP: this module has no opinion on trade_permission and the validator
   actively REJECTS a response that tries to include one -- callers render trade_permission
   straight from the snapshot's own field (originally premarket_thesis.trade_permission()'s
   output), never from anything the AI says.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

from . import premarket_backtest as bt
from . import premarket_catalysts as pc
from . import premarket_thesis as pt

AI_LAYER_VERSION = "v1"

REQUIRED_SECTIONS = ["trend_context", "signal_conviction", "risk_environment", "catalyst_risk",
                     "overnight_news", "historical_evidence", "synthesis"]

# Anywhere in the response, at any section -- an AI-invented number/decision on any of these is a
# schema violation, not something to silently strip and keep the rest.
FORBIDDEN_KEYS_ANYWHERE = {"confidence", "ai_confidence", "trade_state", "trade_permission",
                           "position_size", "stop_loss", "recommendation", "size"}

SYNTHESIS_VERDICTS = {"SUPPORTS", "CONTRADICTS", "MIXED", "INSUFFICIENT"}


# ------------------------------------------------------------------ frozen snapshot ----------

def _trend_relationship(daily_direction: str | None, premarket_direction: str) -> str:
    """Deterministic label for how the premarket move relates to the daily trend -- PIIP computes
    this classification, never the AI (same 'machine calculates, AI explains' boundary as every
    other derived field here)."""
    if premarket_direction == "Neutral":
        return "Premarket move too small to characterize against the daily trend"
    if daily_direction is None:
        return "Daily trend unknown (insufficient daily history)"
    if daily_direction == premarket_direction:
        return f"Continuation ({premarket_direction.lower()} premarket, same as the daily trend)"
    return (f"Opposing ({premarket_direction.lower()} premarket against a {daily_direction.lower()} "
           f"daily trend -- possible pullback/reversal setup, not a clean continuation)")


def build_snapshot(spy_snap: dict, thesis: dict, releases: dict, tier1: dict, tier2: dict,
                   now_et: datetime | None = None) -> dict:
    """Assembles the ONE frozen input the AI will ever see. Pure aggregation of ALREADY-COMPUTED
    values -- no new data fetching happens here, and nothing computed here is re-derived by the
    AI call itself. `thesis` is premarket_thesis.build_thesis()'s own output (already has
    market_state/families/risk_environment/confirmation/invalidation/trade_permission);
    `releases` is macro.economic_releases_snapshot()'s output; `tier1`/`tier2` are
    premarket_backtest.run_tier*_backtest()'s outputs."""
    now_et = now_et or datetime.now(ZoneInfo("America/New_York"))
    tech = (spy_snap or {}).get("tech") or {}
    daily_direction = ("Bullish" if tech.get("above_sma200") is True else
                       "Bearish" if tech.get("above_sma200") is False else None)
    premarket_direction = thesis["market_state"]["direction"]

    headlines = pt.overnight_headlines()
    finnhub_configured = bool(os.getenv("FINNHUB_API_KEY"))
    overnight_news = {
        "status": "OK" if headlines else ("UNAVAILABLE — Finnhub not configured" if not finnhub_configured
                                          else "OK — no material headlines found overnight"),
        "headlines": [{"headline": h["headline"], "published": h["published"],
                      "composite_score": h["composite_score"], "sources": h["sources"],
                      "lean": h["lean"]["label"], "calibration": h.get("calibration") or {}}
                     for h in headlines],
    }

    return {
        "ai_layer_version": AI_LAYER_VERSION,
        "snapshot_timestamp": now_et.isoformat(timespec="seconds"),
        "spot_price": thesis.get("spot_price"),
        "market_state": thesis["market_state"],
        "signal_families": thesis["families"],
        "trend_context": {"daily_trend_direction": daily_direction,
                          "premarket_direction": premarket_direction,
                          "relationship": _trend_relationship(daily_direction, premarket_direction)},
        "risk_environment": thesis["risk_environment"],
        "catalyst_risk": pc.catalyst_risk_today(releases, today=now_et.date()),
        "overnight_news": overnight_news,
        "historical_tier1": tier1,
        "historical_tier2": tier2,
        "tiers_agree": bt.tiers_agree(tier1, tier2),
        "confirmation": thesis["confirmation"],
        "invalidation": thesis["invalidation"],
        "trade_permission": thesis["trade_permission"],
    }


# ------------------------------------------------------------------ schema validation ----------

def _no_numeric_leaves(obj) -> bool:
    """True if `obj` contains no bare int/float anywhere -- every number a user sees here must
    trace back to the snapshot PIIP already built, never a new number the model introduced. Prose
    inside an "analysis" string MAY reference a number in words/text; that's not a JSON leaf."""
    if isinstance(obj, dict):
        return all(_no_numeric_leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_no_numeric_leaves(v) for v in obj)
    if isinstance(obj, bool):
        return True   # bool is a subclass of int in Python -- explicitly allowed, checked first
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


def validate_ai_response(raw) -> dict | None:
    """Hard validation, no silent repair -- None on ANY violation. A caller that gets None must
    show an explicit invalid-response state, never fall back to guessing what the model meant."""
    if not isinstance(raw, dict):
        return None
    if set(raw.keys()) != set(REQUIRED_SECTIONS):
        return None
    if _has_forbidden_key(raw) or not _no_numeric_leaves(raw):
        return None
    for key in REQUIRED_SECTIONS:
        section = raw[key]
        if not isinstance(section, dict):
            return None
        analysis = section.get("analysis")
        if not isinstance(analysis, str) or not analysis.strip():
            return None
    synthesis = raw["synthesis"]
    if synthesis.get("verdict") not in SYNTHESIS_VERDICTS:
        return None
    failure_mode = synthesis.get("failure_mode")
    if not isinstance(failure_mode, str) or not failure_mode.strip():
        return None
    return raw


# ------------------------------------------------------------------ the AI call ----------

def _build_prompt(snapshot: dict) -> str:
    return (
        "You are the interpretation layer for a deterministic SPY premarket trading-research "
        "tool. Every number in the snapshot below was already computed by the platform -- you "
        "NEVER recompute, invent, restate as a new number, or second-guess them. You were not "
        "given internet access or any tool to fetch more data; reason ONLY from exactly what's "
        "provided below, and say so plainly whenever something is UNAVAILABLE or "
        "INSUFFICIENT_SAMPLE rather than filling the gap.\n\n"
        f"SNAPSHOT (frozen at {snapshot['snapshot_timestamp']}):\n"
        f"{json.dumps(snapshot, default=str)}\n\n"
        "Answer these 7 fixed questions as ONE structured JSON object, no prose outside the "
        "JSON:\n"
        "1. trend_context -- is the premarket move consistent with the daily trend, opposing it, "
        "or in a neutral/unclear regime (see trend_context.relationship, already computed)? What "
        "does that imply about continuation vs reversal risk?\n"
        "2. signal_conviction -- how broad and internally consistent is the directional signal "
        "across signal_families? Which families confirm it, which contradict, and do the "
        "disagreements meaningfully weaken the thesis?\n"
        "3. risk_environment -- how does today's risk_environment affect the RELIABILITY and "
        "TRADABILITY of the thesis? Never recommend position size or stop-loss levels.\n"
        "4. catalyst_risk -- given catalyst_risk (FOMC is exact; CPI/Payrolls are an "
        "approximation, both labeled), could a scheduled event invalidate the thesis, and when "
        "relative to a typical morning trade window?\n"
        "5. overnight_news -- if overnight_news.status is UNAVAILABLE, say so explicitly and "
        "stop there for this section. Otherwise: do the provided headlines support or contradict "
        "the technical direction? Distinguish real catalysts from noise/redundant coverage.\n"
        "6. historical_evidence -- historical_tier1 and historical_tier2 are SEPARATE real "
        "backtests (see tiers_agree, already computed) -- do NOT blend them into one number. "
        "State what each tier shows given its own status/similarity_label/n, and whether they "
        "agree. If either tier's status is INSUFFICIENT_SAMPLE or NO_DATA, say so explicitly "
        "rather than treating its numbers as reliable.\n"
        "7. synthesis -- does the total evidence SUPPORT, CONTRADICT, or leave the thesis MIXED "
        "or INSUFFICIENT? Also identify the single most plausible failure mode: what would prove "
        "this thesis wrong, and what observable market behavior would reveal that early?\n\n"
        "Required JSON shape -- EXACTLY these 7 keys, each an object with a non-empty "
        '"analysis" string (2-4 sentences, prose only). synthesis additionally requires '
        '"verdict" (exactly one of SUPPORTS, CONTRADICTS, MIXED, INSUFFICIENT) and '
        '"failure_mode" (a string). Do NOT include a confidence number, a trade-state or '
        "trade-permission field, a position-size or stop-loss recommendation, or ANY bare "
        "numeric field anywhere in your JSON -- reference numbers only in words inside the "
        "analysis text, never as a JSON number of your own:\n"
        '{"trend_context":{"analysis":""}, "signal_conviction":{"analysis":""}, '
        '"risk_environment":{"analysis":""}, "catalyst_risk":{"analysis":""}, '
        '"overnight_news":{"analysis":""}, "historical_evidence":{"analysis":""}, '
        '"synthesis":{"analysis":"", "verdict":"", "failure_mode":""}}')


def ask_ai_thesis(snapshot: dict, client) -> dict:
    """ONE call, structured JSON, hard-validated before ever reaching a caller. `client` is an
    iip.agents.LLMClient (or anything with the same .call(str) -> (dict|None, float) shape) --
    dry_run/budget/cost governance is entirely that object's responsibility, unchanged from every
    other AI feature in this app. Returns a dict with `_valid` always present; callers branch on
    that, never on whether keys "look right" via .get() alone."""
    prompt = _build_prompt(snapshot)
    out, cost = client.call(prompt)
    if out is None:   # dry run
        return {"_valid": False, "_dry_run": True, "_cost": 0.0,
               "_reason": "dry-run (no LLM call)", "ai_layer_version": AI_LAYER_VERSION}
    if isinstance(out, dict) and out.get("_skipped"):
        return {"_valid": False, "_dry_run": False, "_cost": 0.0, "_reason": out["_skipped"],
               "ai_layer_version": AI_LAYER_VERSION}
    validated = validate_ai_response(out)
    if validated is None:
        return {"_valid": False, "_dry_run": False, "_cost": cost, "_raw": out,
               "_reason": "AI response failed schema validation",
               "ai_layer_version": AI_LAYER_VERSION}
    return {"_valid": True, "_dry_run": False, "_cost": cost, "ai_layer_version": AI_LAYER_VERSION,
           **validated}
