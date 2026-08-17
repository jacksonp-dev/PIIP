# Changelog

Dated entries, newest first. PIIP checks this file (and `VERSION`) against your local copy to let
you know when an update's available — see the notice in the sidebar.

## 2026.08.17

**0DTE Intelligence — Trend Integrity & Day Regime**
- New **Day Regime** read at the top of the page: INSUFFICIENT DATA / NEUTRAL-CHOP / BULL·BEAR
  DEVELOPING or CONFIRMED / TREND WEAKENING / REGIME TRANSITION — a synthesis of existing signals,
  not a new independent score.
- New **Trend Integrity**, **Trend Efficiency**, and **VWAP Crossings** — how clean vs. choppy the
  current move actually is.
- New **Timeframe Sequence** reading (e.g. "bullish higher-timeframe trend, short-term pullback")
  instead of just an agree/disagree count.
- Time-of-day-adjusted **Relative Volume**, shown alongside the existing proxy version.
- **NVDA** is now a fully selectable ticker on the page, with relative-strength reads vs
  SPY/QQQ/SMH/SOXX and a leadership-acceleration read.
- **NO CLEAR EDGE** and **Data Quality** are now first-class, explained states instead of buried
  caveats.
- Signal history is now logged locally for future calibration (Signal Calibration Log, Regime
  Timeline, Historical Regime Stats) — collection-only for now; real stats need real history to
  build up first, and the UI says so honestly rather than faking a result.
- The page is now organized into tabs (Overview / Timeframes & Trend / Market Context /
  Options & Contract / Macro & Diagnostics) instead of one long scroll.
- Bid Simulator and contract-specific quotes (DTE, moneyness, last-traded time) added to the
  Exit Quality section.

**Charts**
- The intraday chart (and the Home page portfolio chart) now use a real trading-chart engine
  (TradingView's Lightweight Charts) instead of a general-purpose plotting library — real
  scroll/pinch zoom and drag-to-pan, with price auto-fitting whatever range is visible, a volume
  panel, toggleable VWAP/EMA(50)/SMA(50) overlays, and a crosshair legend showing OHLC + the exact
  date/time under your cursor.
- A banner now appears directly on the chart whenever it's showing the last completed session
  instead of a live one (e.g. before market open) — no more hovering to discover you're looking at
  an old session.

**Everything above only changed the 0DTE page and these two charts.** No other page, no execution
logic, no account/portfolio math was touched in this release.

## Earlier

PIIP's development history before this changelog started isn't itemized entry-by-entry here — see
the git history and `BUILD_PLAN.md` for the fuller story of how the platform got here.
