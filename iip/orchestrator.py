"""Orchestrator — one ticker -> data -> deterministic engine -> baseline prediction (+ optional
LLM prediction on trial) -> logged as pre-registered experiments.

The deterministic path always runs and costs nothing. The LLM path is opt-in and cost-governed.
"""
from __future__ import annotations

from . import agents, baseline, data
from . import predictions as pred


def research(ticker: str, horizons=(7, 30, 90), use_llm: bool = False,
             db: str = pred.DB_PATH, budget=None, dry_run: bool | None = None) -> dict:
    pred.init_db(db)
    prices = data.get_prices(ticker, period="2y")
    spot = float(prices["Close"].iloc[-1])

    chains, det_fcs, logged = {}, {}, []
    for H in horizons:
        det_fc = baseline.deterministic_forecast(ticker, prices, H)
        chain = data.get_option_chain(ticker, data.nearest_expiry(ticker, H))
        chains[H], det_fcs[H] = chain, det_fc
        opt = baseline.simulated_option(spot, chain, det_fc["stock_direction"])
        pid = pred.log_prediction({**det_fc, "spot": spot, **opt}, db)
        logged.append({"source": "deterministic", "H": H, "id": pid, "fc": det_fc, "opt": opt})

    cost, llm_logged = 0.0, []
    if use_llm:
        client = agents.LLMClient(budget=budget, dry_run=dry_run)
        llm_fcs, cost, _outs = agents.run_ticker(ticker, prices, spot, chains, det_fcs, client)
        for H in horizons:
            lfc = llm_fcs[H]
            opt = baseline.simulated_option(spot, chains[H], lfc["stock_direction"])
            lid = pred.log_prediction({**lfc, "spot": spot, **opt}, db)
            llm_logged.append({"source": "llm", "H": H, "id": lid, "fc": lfc, "opt": opt})

    return {"ticker": ticker, "spot": spot, "deterministic": logged, "llm": llm_logged, "cost_usd": cost}