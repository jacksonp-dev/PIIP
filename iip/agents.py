"""LLM agents — ON TRIAL against the deterministic baseline.

Agents INTERPRET pre-computed metrics; they never calculate. Cost-governed: 3 specialist calls
(Technical, Options, Skeptic) once per ticker + 1 Executive call producing all-horizon forecasts
= 4 calls/run, matching the budget cap. Dry-run by default: no API call, no cost — real spend
requires explicit opt-in (PIIP_LIVE_LLM=1 or dry_run=False).
"""
from __future__ import annotations

import json
import os
import re

from . import deterministic as det
from .cost import PRICES_PER_MTOK, Budget, est_call_cost

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

_SYS = (
    "You are a financial analysis agent in a research platform. You INTERPRET pre-computed metrics; "
    "you NEVER invent or recompute numbers (use the historical distributions provided for expected "
    "move and prob_up). Output ONLY valid JSON, no prose. Always include confidence (0-1), separate "
    "bull and bear factors, and name the biggest uncertainty. Never express certainty."
)


def _append_user_question(prompt: str, user_question: str | None) -> str:
    """Appends the user's own optional free-text question to a prompt -- used by every AI section
    in the app (Research/Deep Research/Paper/0DTE) so someone can ask something specific without a
    new call shape per page. Explicitly scoped to the SAME data already in the prompt -- this is
    not a general-purpose chat box, it's a lens on the numbers already shown.

    Critically, this ALSO tells the model to add a dedicated "user_question_answer" field to its
    JSON -- a first version just appended the question as a soft aside with no schema slot for the
    answer, and confirmed live: every response schema here is a strict, itemized JSON contract, so
    the model just kept filling its normal fields and never surfaced a direct answer anywhere the
    UI could find it. A named field is a real place for the answer to live, not a hope that it
    surfaces somewhere inside key_evidence/summary/etc."""
    if not user_question:
        return prompt
    return (prompt + f'\n\nThe user also specifically asked: "{user_question}"\n'
            "Answer this DIRECTLY and explicitly -- using ONLY the data already provided above, "
            "never invent information beyond it, and say so plainly if the data can't answer it. "
            'Add one more field to your JSON response: "user_question_answer" containing that '
            "direct answer (2-3 sentences).")


def _parse(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


class LLMClient:
    def __init__(self, budget: Budget | None = None, model: str = DEFAULT_MODEL, dry_run: bool | None = None):
        self.model = model
        self.budget = budget or Budget()
        self.dry_run = (os.getenv("PIIP_LIVE_LLM") != "1") if dry_run is None else dry_run
        self._client = None
        self.budget.start_run()

    def _lazy(self):
        if self._client is None:
            from anthropic import Anthropic
            from dotenv import find_dotenv, load_dotenv
            load_dotenv(find_dotenv(usecwd=True))     # keys from .env, never hard-coded
            self._client = Anthropic()
        return self._client

    def call(self, user: str) -> tuple[dict | None, float]:
        if self.dry_run:
            return None, 0.0
        est = est_call_cost(self.model)
        ok, why = self.budget.can_spend(est)
        if not ok:
            return {"_skipped": why}, 0.0
        # 2000 tokens: specialists finish well under this; the Executive's multi-horizon JSON needs
        # the headroom (it was truncating at 800 -> invalid JSON -> empty forecast).
        r = self._lazy().messages.create(model=self.model, max_tokens=2000, system=_SYS,
                                         messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in r.content)
        cost = (r.usage.input_tokens + r.usage.output_tokens) / 1e6 * PRICES_PER_MTOK.get(self.model, 3.0)
        self.budget.record(cost)
        if os.getenv("PIIP_DEBUG") == "1":
            path = os.getenv("PIIP_DEBUG_FILE", "piip_llm_debug.log")
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n--- RAW LLM ({len(text)} chars, stop={getattr(r, 'stop_reason', '?')}) ---\n"
                        f"{text}\n--- END ---\n")
        return _parse(text), cost


def _specialist(client: LLMClient, name: str, instruction: str, metrics: dict,
                user_question: str | None = None) -> tuple[dict, float]:
    user = (f"{instruction}\nCOMPUTED METRICS (interpret, do not recompute):\n"
            f"{json.dumps(metrics, default=str)}\n"
            'Return JSON: {"agent","confidence","bull_factors":[],"bear_factors":[],'
            '"biggest_uncertainty","summary"}')
    user = _append_user_question(user, user_question)
    out, cost = client.call(user)
    if out is None:  # dry run
        out = {"agent": name, "confidence": 0.5, "bull_factors": ["dry-run"], "bear_factors": [],
               "biggest_uncertainty": "dry-run (no LLM call)", "summary": "dry-run"}
    return out, cost


def anonymized_forecast(metrics: dict, client: LLMClient, horizons=(7, 30, 90)):
    """ONE call: interpret ANONYMIZED point-in-time metrics (no ticker, no dates, no absolute price)
    so the model cannot recognize the stock/era and cheat via hindsight. This is the only look-ahead-
    safe way to backtest the LLM's interpretation. Returns ({h: llm_forecast}, cost)."""
    user = ("You are interpreting COMPUTED metrics for an ANONYMOUS US stock. Ticker and dates are "
            "withheld on purpose to prevent hindsight — reason ONLY from these numbers, do not guess "
            "the identity. Forecast for each horizon (calendar days). Use the historical move "
            "distribution for expected_move_pct and prob_up; never invent numbers.\n"
            f"METRICS:\n{json.dumps(metrics, default=str)}\n"
            'Return ONLY JSON: {"7":{"direction","expected_move_pct","prob_up","confidence"},'
            '"30":{...},"90":{...}}')
    out, cost = client.call(user)
    if out is None:  # dry run
        out = {str(h): {"direction": "neutral", "expected_move_pct": None,
                        "prob_up": 0.5, "confidence": 0.4} for h in horizons}
    fcs = {}
    for h in horizons:
        e = out.get(str(h)) or out.get(h) or {}
        fcs[h] = {"source": "llm", "horizon_days": h,
                  "stock_direction": e.get("direction", "neutral"),
                  "stock_expected_move_pct": e.get("expected_move_pct"),
                  "stock_prob_up": e.get("prob_up"), "confidence": e.get("confidence"),
                  "reasoning": {"mode": "anonymized_backtest"}}
    return fcs, cost


def interpret_dossier(dossier: dict, client: LLMClient, user_question: str | None = None) -> tuple[dict, float]:
    """ONE call: interpret an already-built Deep Research dossier (fundamentals, clinical trials,
    SEC catalysts, confidence-scored fields) for educational research -- never invents numbers,
    only reasons over what's already fetched. Lighter than run_ticker()'s 4-call pipeline since a
    dossier isn't a directional prediction to grade, just a research summary to interpret."""
    user = ("Interpret this already-researched company dossier for EDUCATIONAL research (not a "
            "trade recommendation). Every field already carries its own confidence/source -- "
            "reason ONLY from what's provided, never invent numbers, and call out UNKNOWN fields "
            "as real gaps rather than guessing around them.\n"
            f"DOSSIER:\n{json.dumps(dossier, default=str)}\n"
            'Return ONLY JSON: {"bull_case","bear_case","biggest_uncertainty","confidence",'
            '"summary"}')
    user = _append_user_question(user, user_question)
    out, cost = client.call(user)
    if out is None:  # dry run
        out = {"bull_case": "dry-run", "bear_case": "dry-run",
               "biggest_uncertainty": "dry-run (no LLM call)", "confidence": 0.5, "summary": "dry-run"}
    return out, cost


def interpret_portfolio(account: dict, client: LLMClient, user_question: str | None = None) -> tuple[dict, float]:
    """ONE call: a plain-English read on how the PAPER (practice) account is actually doing, from
    numbers already computed by portfolio.summary() -- coaching commentary, not a trade
    recommendation, and never invents stats not present in the input. `account` is portfolio.
    summary()'s own return dict; its `closed` list is trimmed to the 10 most recent trades so the
    prompt stays small no matter how long the account's been running."""
    trimmed = {**account, "closed": account.get("closed", [])[-10:]}
    user = ("Give a plain-English read on how this PAPER (practice, no real money) options "
            "account is doing, for someone learning to trade options. Reason ONLY from the "
            "numbers provided -- never invent stats. Note real patterns if visible (e.g. "
            "concentration in one ticker or direction, a cluster of near-term expirations) but do "
            "not recommend specific trades.\n"
            f"ACCOUNT:\n{json.dumps(trimmed, default=str)}\n"
            'Return ONLY JSON: {"read","strengths":[],"risks":[],"biggest_uncertainty","summary"}')
    user = _append_user_question(user, user_question)
    out, cost = client.call(user)
    if out is None:  # dry run
        out = {"read": "dry-run", "strengths": [], "risks": [],
               "biggest_uncertainty": "dry-run (no LLM call)", "summary": "dry-run"}
    return out, cost


# 5 named, weighted agents matching the platform's own agent-weighting spec. Weights sum to 1.0
# and are used ONLY to blend each agent's own scores into the final synthesis below -- never sent
# to the model itself (each agent reasons about its own focus area, not its relative importance).
ZERO_DTE_AGENTS = [
    ("News & Catalyst Agent", 0.30,
     "Identify and weigh news, macro events, earnings, SEC filings, and company catalysts relevant "
     "to today's session. Answer: what is the 'why' behind today's move, if any?"),
    ("Technical & Market Structure Agent", 0.25,
     "Combine technical analysis with VWAP, Opening Range, relative volume, and market internals/"
     "breadth. Answer: is the market confirming the thesis?"),
    ("Options & Positioning Agent", 0.20,
     "Evaluate gamma exposure, expected move, implied volatility, dealer positioning, and open "
     "interest. Critical for 0DTE since options positioning can dominate price action."),
    ("Macro & Cross-Asset Agent", 0.15,
     "Watch Treasury yields, the dollar index, oil, and broad-market breadth (equal-weight vs "
     "cap-weight). Answer: does the broader market environment support or fight the thesis?"),
    ("Skeptic / Risk Agent", 0.10,
     "Try to DISPROVE the thesis. Identify missing evidence and what would invalidate a bullish "
     "read. Be honest -- and score accordingly -- when the case for a trade is weak."),
]

# Conclusion-shaped fields -- the deterministic engine's own downstream BLEND of everything else
# (trade_confidence and entry_quality are weighted composites; bias_direction and confluence
# summarize what other signals already agree on). Shown to NO agent, including the Skeptic --
# confirmed live: with these included, every one of the 5 agents just read the pre-computed
# answer off the page and parroted it back with different wording instead of reasoning
# independently from the underlying signals, which defeats the entire point of running 5 calls.
_ZERO_DTE_ANSWER_FIELDS = ("trade_confidence", "bias_direction", "confluence", "entry_quality")


def zero_dte_scoped_signals(signals: dict, name: str) -> dict:
    """Each agent gets ONLY the raw signals relevant to its own named focus area, not the full
    dict -- the second half of the same fix as _ZERO_DTE_ANSWER_FIELDS above. Confirmed live:
    without this, a 'News & Catalyst' read and a 'Technical' read cited the identical breadth/
    momentum numbers, because every agent could see every OTHER agent's raw material too. Skeptic
    is the one deliberate exception -- finding holes anywhere is its whole job, so it keeps the
    full picture (still with the conclusion fields scrubbed above, so it can't just read the
    answer off the page either)."""
    common = {"ticker": signals.get("ticker"), "session_timing": signals.get("session_timing")}
    macro_sig = signals.get("macro") or {}
    if name == "News & Catalyst Agent":
        return {**common, "market_dna": signals.get("market_dna")}
    if name == "Technical & Market Structure Agent":
        return {**common, "market_bias": signals.get("market_bias"), "breadth": signals.get("breadth"),
               "momentum": signals.get("momentum"),
               "reversal_pressure_score": signals.get("reversal_pressure_score"),
               "timeframe_alignment": signals.get("timeframe_alignment"),
               "trend_state": signals.get("trend_state")}
    if name == "Options & Positioning Agent":
        return {**common, "options_health": signals.get("options_health"),
               "dealer_positioning": signals.get("dealer_positioning"),
               "put_call_oi_ratio": macro_sig.get("put_call_oi_ratio")}
    if name == "Macro & Cross-Asset Agent":
        return {**common, "10y_yield": macro_sig.get("10y_yield"), "dxy": macro_sig.get("dxy"),
               "wti": macro_sig.get("wti"), "rsp_vs_spy": macro_sig.get("rsp_vs_spy"),
               "sector_health_pct": signals.get("sector_health_pct"),
               "mega_cap_health": signals.get("mega_cap_health")}
    # Skeptic / Risk Agent: everything except the conclusion-shaped fields
    return {k: v for k, v in signals.items() if k not in _ZERO_DTE_ANSWER_FIELDS}


def _zero_dte_agent_call(name: str, instruction: str, signals: dict, client: LLMClient,
                         user_question: str | None = None) -> tuple[dict, float]:
    user = (f"You are the {name} in a 0DTE options RESEARCH tool (not a trading signal -- this "
            f"page's own backtest found its highest-confidence bucket statistically "
            f"indistinguishable from a coin flip / SPY's base rate). {instruction} Weigh the "
            f"signals' session_timing explicitly: these are same-day-expiration (0DTE) options, so "
            f"how many minutes remain until today's close materially changes trade_impact and "
            f"confidence -- a setup with little time left to develop is weaker than the same setup "
            f"early in the session. Reason ONLY from the signals below -- never invent numbers; if "
            f"you lack real data for your focus area, say so and score with LOW confidence rather "
            f"than guessing.\n"
            f"SIGNALS:\n{json.dumps(signals, default=str)}\n"
            'Return ONLY JSON: {"bullish_score":0-100,"bearish_score":0-100,"confidence":0-100,'
            '"key_evidence":["...","...","..."],"invalidation":["..."],'
            '"trade_impact":"Low|Medium|High","recommended_bias":"Bullish|Bearish|Neutral"}')
    user = _append_user_question(user, user_question)
    out, cost = client.call(user)
    if out is None:  # dry run
        out = {"bullish_score": 50, "bearish_score": 50, "confidence": 40,
               "key_evidence": ["dry-run"], "invalidation": ["dry-run (no LLM call)"],
               "trade_impact": "Low", "recommended_bias": "Neutral"}
    return out, cost


def zero_dte_agent_synthesis(signals: dict, client: LLMClient, user_question: str | None = None) -> tuple[dict, float]:
    """5 weighted agent calls (see ZERO_DTE_AGENTS), each interpreting ONLY its own scoped slice
    of the 0DTE signals (see zero_dte_scoped_signals) -- never fetches new data, never invents
    numbers, and never sees the deterministic engine's own already-computed conclusion fields.
    The final blended score/confidence is CODE-computed (a weighted average of each agent's own
    0-100 scores), never invented by any single call -- same 'deterministic engine computes, AI
    only interprets' rule used everywhere else in this app. Heavier than the other pages' 1-call
    sections (this page already computes far more distinct signal categories than Research/Deep
    Research/Paper, so a genuine per-category breakdown earns the extra calls). Needs a Budget
    with max_calls_per_run >= 5 -- the default cap is 4, sized for run_ticker()'s pipeline."""
    agents_out, total = {}, 0.0
    for name, weight, instruction in ZERO_DTE_AGENTS:
        scoped = zero_dte_scoped_signals(signals, name)
        out, cost = _zero_dte_agent_call(name, instruction, scoped, client, user_question)
        agents_out[name] = {**out, "weight": weight}
        total += cost

    def _net(o):
        return (o.get("bullish_score") or 0) - (o.get("bearish_score") or 0)

    final_score = sum(_net(o) * o["weight"] for o in agents_out.values())
    final_confidence = sum((o.get("confidence") or 0) * o["weight"] for o in agents_out.values())
    return {"agents": agents_out, "final_score": round(final_score, 1),
           "final_confidence": round(final_confidence, 1)}, total


def run_ticker(ticker, prices, spot, chains: dict, det_forecasts: dict, client: LLMClient,
              user_question: str | None = None):
    """3 specialists + 1 executive => per-horizon LLM forecasts. Returns (forecasts, cost, agent_outputs).
    user_question (optional): the user's own free-text question, appended to every call so any
    agent whose focus area is relevant can address it -- never a new data source, just a lens on
    the same metrics."""
    total = 0.0
    tech = det.technical_metrics(prices)
    hists = {h: det.historical_move_distribution(prices, h) for h in det_forecasts}
    opt_metrics = det.option_metrics(spot, chains[30]) if 30 in chains else next(iter(
        (det.option_metrics(spot, c) for c in chains.values())), {})

    outs = {}
    outs["technical"], c = _specialist(client, "technical",
        "Interpret trend, momentum and volatility regime.", tech, user_question); total += c
    outs["options"], c = _specialist(client, "options",
        "Interpret IV, expected move and greeks. Flag theta bleed and IV-crush risk explicitly.",
        opt_metrics or {}, user_question); total += c
    outs["skeptic"], c = _specialist(client, "skeptic",
        "DESTROY the bullish thesis. Assume the trend is a trap. Surface survivorship bias, missing "
        "data, and the strongest bear case.", {"technical": tech, "options": opt_metrics, "hist": hists},
        user_question); total += c

    exec_input = {"specialists": outs,
                  "deterministic_baseline": det_forecasts,
                  "computed": {"technical": tech, "historical_move": hists, "options": opt_metrics}}
    user = ("Synthesize a forecast for horizons 7, 30 and 90 days. Interpret the computed metrics; "
            "NEVER invent numbers (use the historical distributions for expected_move_pct and "
            "prob_up). You may disagree with the deterministic baseline, but justify it from the "
            f"metrics. INPUT:\n{json.dumps(exec_input, default=str)}\n"
            'Return JSON: {"7":{"direction","expected_move_pct","prob_up","confidence"},'
            '"30":{...},"90":{...},"bull_factors":[],"bear_factors":[],"biggest_uncertainty":"","sources":[]}')
    user = _append_user_question(user, user_question)
    ex, c = client.call(user); total += c
    if ex is None:  # dry run -> echo deterministic so the pipeline produces a coherent record
        ex = {str(h): {"direction": det_forecasts[h]["stock_direction"],
                       "expected_move_pct": det_forecasts[h]["stock_expected_move_pct"],
                       "prob_up": det_forecasts[h]["stock_prob_up"], "confidence": 0.5}
              for h in det_forecasts}
        ex.update({"bull_factors": ["dry-run"], "bear_factors": [],
                   "biggest_uncertainty": "dry-run (no LLM call)", "sources": ["dry-run"]})

    forecasts = {}
    for h in det_forecasts:
        e = ex.get(str(h)) or ex.get(h) or {}
        forecasts[h] = {
            "source": "llm", "ticker": ticker, "horizon_days": h,
            "stock_direction": e.get("direction", "neutral"),
            "stock_expected_move_pct": e.get("expected_move_pct"),
            "stock_prob_up": e.get("prob_up"), "confidence": e.get("confidence"),
            "reasoning": {"bull_factors": ex.get("bull_factors"), "bear_factors": ex.get("bear_factors"),
                          "biggest_uncertainty": ex.get("biggest_uncertainty"), "sources": ex.get("sources"),
                          "user_question_answer": ex.get("user_question_answer"),
                          "specialists": {k: v.get("summary") for k, v in outs.items()}},
        }
    return forecasts, total, outs