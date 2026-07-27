# IIP — Build Plan (converged spec, 2026-07)

> Supersedes the original early planning docs for build order.
> Those remain as the long-term vision. This is what we actually build, and why.

## Governing principle
**The primary product is CALIBRATED FORECASTING, not research.** Everything supports one
question: *does this system produce predictions with measurable value beyond a dumb baseline?*
If it can't, no amount of architecture matters — so we validate before we build the cathedral.

## Locked decisions
- **Stack:** lean, local-first. Python + SQLite + CLI. No Postgres/Redis/Celery/Docker/Next.js
  until Phase 1 proves value.
- **Budget:** $0–20/mo. Free data (yfinance, later FRED). Cost governor from day one.
- **Focus:** options-centric, single ticker. Learn options; measure honestly.

## The two-layer architecture (order matters)
```
Data (free)  ->  Deterministic Engine (CODE computes)  ->  LLM Interpretation (narrates, ON TRIAL)
                          |                                          |
                   this is the product                    must beat the deterministic
                   (backtestable NOW)                      signal to justify its cost
```
- **Deterministic engine computes everything numeric:** indicators, ATR, realized vol, option
  greeks/IV/expected-move (Black-Scholes on the chain), historical forward-return distribution.
  The LLM NEVER calculates — prompts say "interpret these computed metrics, never invent them."
- **LLM is on trial.** Every run logs TWO predictions: (a) deterministic-only, (b) LLM-adjusted.
  We score the LLM against the deterministic baseline. Prior (from the crypto bot): the AI added
  ZERO. It must earn its place forward.

## Hard lessons baked in (from the crypto bot project)
1. **Benchmark or it's astrology.** Score every prediction vs actual outcome AND vs SPY AND vs
   holding the underlying AND vs a random pick. Calibration (Brier, reliability curve) is the
   homepage, not a footnote.
2. **Statistical-power wall is real.** 30-day forecasts => hundreds of them = years. Early
   numbers are NOISE. Accumulate faster via more tickers + (where valid) shorter horizons.
   Deterministic layer is backtestable on history immediately; LLM is forward-only.
3. **Look-ahead / training-cutoff trap.** The LLM cannot be validated on historical dates whose
   outcomes it already knows. Historical backtests use the DETERMINISTIC layer only.
4. **Correlated opinions aren't an ensemble.** Different DATA per agent helps; shared model still
   biases interpretation. Don't sell "agent agreement" as independent confirmation.
5. **Cost governor is architecture, not polish.** Budget cap, per-run call ceiling, 24h cache.
6. **Data availability picks the agents.** Free tier supports Technical / Options / Quant / Macro.
   News / Sentiment / Fundamental wait until we can feed them real data.
7. **Don't act on narrative.** A polished score is not permission. The tool informs thinking; it
   does not issue BUY signals.

## Prediction schema (options-aware from day 1)
Each run stores: ticker, timestamp, spot, horizon_days, direction, predicted_move_pct,
predicted_move_prob, predicted_vol, confidence, source ("deterministic" | "llm"),
rationale, cost_usd. Later resolved with: realized_return, realized_vol, spy_return,
underlying_return, hit (bool), brier_component.

## Phase 1 — "Calibrated Forecaster" (what we build now)
- `iip/data.py` — free data layer (yfinance: OHLCV, option chains).
- `iip/deterministic.py` — computes all numeric metrics (technical + options + Black-Scholes
  greeks + historical move distribution). **No LLM. Zero cost. Backtestable.**
- `iip/predictions.py` — SQLite prediction log (the scorecard store).
- `iip/scorer.py` — resolves predictions; computes hit rate, Brier, calibration, alpha vs SPY /
  underlying / random. **This is the centerpiece.**
- `iip/agents.py` — LLM agents that INTERPRET deterministic metrics (Options, Technical, Skeptic,
  Executive). Cost-governed. Optional at first — the deterministic engine runs without them.
- `iip/orchestrator.py` — one ticker -> data -> deterministic -> (optional) agents -> 2 predictions.
- `iip/cli.py` — `research TICKER`, `score`, `report`.
- `iip/cost.py` — budget guard (daily cap, per-run ceiling, cache).

### Gate 1 (before any Phase 2)
Deterministic signal beats baselines on a historical backtest, AND (forward) the LLM beats the
deterministic signal. If deterministic has no edge, the LLM narrating it is worthless. If it does,
the LLM must prove it adds value. Only then does Phase 2 (Postgres/FastAPI/Next.js/more agents) exist.

## LOCKED SPEC v1 (2026-07 — converged with ChatGPT, do not silently change)
**Every prediction is a pre-registered experiment.** Purpose: falsifiable, calibratable forecasts —
not recommendations.

- **Two independent targets per run:**
  - *Stock thesis:* direction, expected_move_pct, prob_up, confidence, horizon.
  - *Simulated option:* standardized contract (ATM, expiry nearest horizon), entry premium, greeks
    at entry, IV%(+percentile when available), later exit premium & realized P&L.
- **Horizons:** 7d (catalyst/options), 30d (swing), 90d (durable). Logged for all three.
- **Deterministic baseline (the bar the LLM must beat) — EQUAL-WEIGHT VOTE, no tuning/ML:**
  factors = SMA200 trend, 3–6mo relative strength, RSI mean-reversion, Bollinger mean-reversion,
  horizon momentum, (sector-relative strength = TODO, needs sector map). Each votes +1/0/−1;
  direction = sign(sum); expected_move = historical fwd-return std / ATR-normalized; prob_up =
  historical. confidence = |sum| / n_factors.
- **LLM on trial:** logs a second prediction; judged vs the deterministic one, FORWARD ONLY.
- **PRIMARY decision test (pre-registered):** 30d stock directional hit-rate, LLM vs deterministic.
  All other horizon/target/metric combos = secondary/exploratory, reported but never decisive
  (avoids the multiple-comparisons / data-mining trap).
- **Structured reasoning logged per prediction:** top-3 bull factors, top-3 bear factors, biggest
  uncertainty, evidence sources. (Turns the log into a learning system — know WHY it failed.)
- **Two gates:**
  - *Exploration (N≥50):* no catastrophic underperformance, no negative expectancy, pipeline
    correct → keep collecting.
  - *Evidence (N≥200 total, conditioned by horizon):* LLM statistically beats deterministic +
    positive expectancy + better calibration + benchmark outperformance → may claim value.
- **Benchmarks:** stock vs SPY + hold-underlying + random pick; option vs systematic same-structure
  option + hold-underlying (profit alone ≠ edge — most calls profit in a bull).
- **Honest caveats:** true IV percentile needs IV history we must accrue (approx w/ realized-vol
  percentile meanwhile, labeled); 90d evidence accrues slowly — no early 90d claims.
- **Validation split:** deterministic layer = backtestable on point-in-time history (engineering
  proof). LLM performance = forward-only (training-cutoff look-ahead poisons history).

## Phase 2 — Scale (only if Gate 1 passes)
Postgres, FastAPI, real web UI, more data feeds, more deterministic analysis, more interpretation
agents, opportunity scanner. Earned by evidence, never assumed.