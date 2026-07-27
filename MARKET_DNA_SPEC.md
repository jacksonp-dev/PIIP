# Market DNA — Module Spec (v1)

> Scoped from `PIIP_0DTE_SPY_Intelligence_Design.docx`. This is the first module pulled out of
> that doc for actual implementation — see the chat history for why the rest (Catalyst Terminal,
> Self-Validation Dashboard, "Bloomberg Terminal" ambitions) is staying as a backlog note, not a
> spec, until this one ships and proves itself.

## Scope

**0DTE SPY only, paper-trading decision support, NOT a trading bot.** Market DNA classifies
*today's session* into a day-type label (Trend Day, Range Bound, Gap & Go, etc.) so the user has
better situational awareness going into the two trading windows they actually act in (market open,
market close) — same governing principle as the rest of PIIP (`BUILD_PLAN.md`): evidence over
prediction, explainable, never fabricate data.

**No new API integrations.** Everything below is computable from data already wired up in
`iip/data.py` (yfinance OHLCV) — no Finnhub/Polygon/news APIs needed. This module is a pure
price-action/volatility classifier layered on data PIIP already has.

## Non-goals (explicitly out of scope for v1)

- No news/catalyst ingestion (that's a separate module, not this one).
- No buy/sell signal — Market DNA describes the day, it does not recommend a trade.
- No cross-linking to the separate `TradingBot-0DTE` paper bot's trade log in v1. (Worth doing
  later — "did open-window CALL entries perform differently on Trend Days vs Chop Days?" — but
  that's a second module once both sides have enough logged data to compare. Noted here so it
  isn't forgotten, not built now.)
- No ML/learned weights. Rule-based, same as the rest of `iip/deterministic.py`.

## Inputs (all already available)

From `iip/data.py` + `iip/deterministic.py`, for SPY:
- Today's intraday bars (`fetch_intraday_batch`, already used by `zero_dte.py`)
- Yesterday's close (from the daily batch) — for gap calculation
- ATR(14) (`technical_metrics`) — today's range vs. normal range
- VWAP (`intraday_snapshot`) — already computed per-refresh

## Classification logic (v1 — thresholds are tunable, not final)

Every metric below is a plain, named, computable number — no black box. Thresholds are a
first-pass guess, flagged for calibration once real sessions have been logged (see Validation).

**Core metrics, computed once per refresh:**
| Metric | Formula |
|---|---|
| `gap_pct` | `(today_open - yesterday_close) / yesterday_close * 100` |
| `range_vs_atr` | `(today_high - today_low) / atr14` |
| `net_vs_range` | `abs(last - today_open) / (today_high - today_low)` (directional persistence — a pure trend closes near one extreme; chop closes near the middle) |
| `vwap_side_consistency` | fraction of today's bars closing on the same side of VWAP as the current close |
| `gap_held` | true if price never traded back through yesterday's close after the first 5 minutes |

**Day-type rules (evaluated in this order — first match wins):**

1. **Insufficient Evidence** — fewer than 30 minutes of session data so far. Don't guess early.
2. **Gap & Go** — `abs(gap_pct) > 0.3` AND `gap_held` AND `net_vs_range > 0.6`
3. **Opening Reversal** — `abs(gap_pct) > 0.3` AND NOT `gap_held`
4. **Trend Day** — `net_vs_range > 0.65` AND `vwap_side_consistency > 0.8`
5. **High Volatility Chop** — `range_vs_atr > 1.3` AND `net_vs_range < 0.35`
6. **Slow Grind** (Higher/Lower by sign of net change) — `net_vs_range > 0.55` AND `range_vs_atr < 0.8`
7. **Range Bound** — everything else (default when no other rule clearly fits)

Output: `{label, as_of_time, metrics: {...all the above, itemized}, note: "still forming" if <90min into session}`.

## Display integration

New subsection in the existing `_render_zero_dte` (`app.py`), same 30s-refresh `st.fragment`
pattern already used for the other 12 sections. One line (label + one-sentence plain-English
explanation built from the itemized metrics, e.g. *"Trend Day — price has stayed above VWAP 92%
of the session and is near its high"*) plus an expander showing the raw metric table for anyone
who wants to check the math.

## Validation (lightweight, v1)

Log `{date, label, metrics, final label at end of day}` to a simple table (reuse `iip.db`
SQLite, new table, following the existing migration pattern in `db/store.py`-style code already
in this codebase). No scoring/precision-recall dashboard yet — that's the "Self-Validation
Dashboard" from the original doc, a separate module once there's enough logged days to make a
dashboard meaningful rather than decorative.

## Implementation notes

- New file: `iip/market_dna.py`, following `iip/zero_dte.py`'s existing conventions (itemized
  reasons dict, pure functions taking already-fetched data, no new network calls).
- Reuses `iip/deterministic.py::technical_metrics` and `intraday_snapshot` — do not recompute
  ATR/VWAP separately.
- Thresholds above go in a small config dict at the top of the file (not hardcoded inline),
  matching the "configurable weights" engineering requirement from the original doc.

## Open questions before implementation

1. Are the rule thresholds above (0.3% gap, 0.65 net/range, etc.) reasonable as a starting guess,
   or do you want to eyeball a few real SPY sessions first and adjust before writing code?
2. Confirm the display location — new section in the existing 0DTE tab, at the top (since it
   frames how to read everything below it) or elsewhere?
