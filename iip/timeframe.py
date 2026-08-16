"""Multi-timeframe alignment for the 0DTE page — PIIP audit 2026-08, Option B.

The gap this closes: zero_dte.py's momentum_engine() reads the last 5 vs. prior 5 bars of
whatever interval was fetched (1m bars, so an implicit ~5-minute window) and nothing in the code
or UI ever said so. There was no way to tell "this ticker is up on a 5-minute pop" from "this
ticker is up on a genuine 30-minute trend" — those are very different reads for a 0DTE decision.

Design, same discipline as zero_dte.py:
  * Every timeframe reading is explicitly labeled with its own window — no more silent 5-minute
    default presented as if it were "momentum" in general.
  * Built ENTIRELY from data already fetched — resamples the existing batched 1m intraday bars
    into 5m/15m/30m bars (no new network calls, no new rate-limit exposure), and reuses the daily
    EMA alignment already computed by deterministic.technical_metrics() for the "Daily" timeframe.
    A true 60m intraday read isn't attempted: it would need ~10 hours of 1m bars to get 10 resampled
    60m bars for the same velocity/acceleration calc, more than a single trading session contains —
    rather than fake it with a shorter, inconsistent lookback, this module leaves 60m out and says
    so, honesty over completeness.
  * `continuation_score_pct` here is named consistently with zero_dte.py's Option A fix — a
    heuristic score, never called a probability.
  * Alignment and trend-state are itemized/explainable, same shape as every other score in this
    project — no black-box numbers.
"""
from __future__ import annotations

import pandas as pd

TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "30m": 30}
# Bars needed for the resampled "last N vs prior N" velocity/acceleration read.
_BARS_NEEDED = 10


def resample_ohlc(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample already-fetched 1-minute bars into `minutes`-bar OHLCV bars. No new network call."""
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df_1m.resample(f"{minutes}min").agg(agg).dropna(subset=["Close"])
    return out


def timeframe_momentum(df_1m: pd.DataFrame, minutes: int, label: str) -> dict | None:
    """Same velocity/acceleration/continuation-score shape as zero_dte.momentum_engine(), just
    computed on `minutes`-bar resampled bars instead of the raw 1m bars, and labeled with which
    timeframe it is. Returns None (with a reason the caller can surface) if there aren't yet
    enough resampled bars for this window today."""
    bars = resample_ohlc(df_1m, minutes)
    if len(bars) < _BARS_NEEDED:
        return None
    c = bars["Close"]
    recent, prior = c.tail(5), c.tail(10).head(5)
    velocity_pct = float((recent.iloc[-1] / recent.iloc[0] - 1) * 100) if recent.iloc[0] else 0.0
    prior_velocity_pct = float((prior.iloc[-1] / prior.iloc[0] - 1) * 100) if prior.iloc[0] else 0.0
    acceleration_pct = round(velocity_pct - prior_velocity_pct, 3)
    same_direction = (velocity_pct > 0 and acceleration_pct >= 0) or (velocity_pct < 0 and acceleration_pct <= 0)
    continuation_score = min(90, max(10, 50 + abs(velocity_pct) * 20 + (15 if same_direction else -15)))
    direction = "Bullish" if velocity_pct > 0.02 else "Bearish" if velocity_pct < -0.02 else "Flat"
    return {"timeframe": label, "bars_used": len(bars), "velocity_pct": round(velocity_pct, 3),
            "acceleration_pct": acceleration_pct, "direction": direction,
            "continuation_score_pct": round(continuation_score, 0)}


def daily_reading(tech: dict) -> dict | None:
    """The 'Daily' timeframe — not a new calculation, just the existing daily EMA20/50/200
    alignment (already computed by deterministic.technical_metrics() for every ticker snapshot)
    reframed as one more timeframe in the alignment stack, since it was sitting there unused for
    this purpose."""
    if tech.get("ema_aligned_bull") is None and tech.get("ema_aligned_bear") is None:
        return None
    direction = "Bullish" if tech.get("ema_aligned_bull") else \
                "Bearish" if tech.get("ema_aligned_bear") else "Flat"
    return {"timeframe": "Daily", "direction": direction,
            "note": "EMA20/50/200 alignment on daily bars — the session's broader trend context, "
                    "not an intraday read."}


def multi_timeframe_snapshot(df_1m: pd.DataFrame | None, tech_daily: dict) -> dict:
    """Every available timeframe reading, keyed by label. A timeframe missing from the dict (or
    with `available: False`) means there isn't enough of today's session yet for that window —
    surfaced to the user rather than silently omitted."""
    out = {}
    for label, minutes in TIMEFRAME_MINUTES.items():
        reading = timeframe_momentum(df_1m, minutes, label) if df_1m is not None else None
        out[label] = {**reading, "available": True} if reading else \
                     {"timeframe": label, "available": False,
                      "note": f"Needs {minutes * _BARS_NEEDED} minutes of today's session "
                              f"({_BARS_NEEDED} resampled {label} bars) — check back later."}
    daily = daily_reading(tech_daily)
    out["Daily"] = {**daily, "available": True} if daily else \
                   {"timeframe": "Daily", "available": False,
                    "note": "Not enough daily history for EMA200 yet."}
    return out


def timeframe_alignment(snapshot: dict) -> dict:
    """How many of the AVAILABLE timeframes agree on direction — itemized, same 'explainable, no
    black box' shape as every other score in this codebase (see zero_dte.confluence_score()).
    Timeframes not yet available (not enough of the session printed) are listed separately, not
    silently dropped and not counted against alignment."""
    order = ["5m", "15m", "30m", "Daily"]
    available = {k: snapshot[k] for k in order if snapshot.get(k, {}).get("available")}
    skipped = [k for k in order if k not in available]
    bulls = sum(1 for v in available.values() if v["direction"] == "Bullish")
    bears = sum(1 for v in available.values() if v["direction"] == "Bearish")
    total = len(available)
    if total == 0:
        return {"agree": 0, "total": 0, "aligned_direction": "Unknown", "alignment_pct": 0.0,
                "checks": [], "skipped": skipped,
                "note": "No timeframe has enough of today's session printed yet."}
    # PIIP audit 2026-08 (accounting pass): was `bulls >= bears`, which silently called an exact
    # tie "Bullish" instead of "Mixed". A true tie can never push alignment_pct above 50% (total is
    # always >= 2x the tied count), so update_trend_state()'s 60% threshold was never fooled by
    # this -- but confluence_score()'s "Multi-timeframe alignment agrees" check compares against
    # this label directly, so a real tie could still be counted as false agreement there. Strict
    # `>` on both sides falls through to "Mixed" for ties (including 0-0), with no other behavior
    # change for genuine majorities.
    aligned_direction = "Bullish" if bulls > bears else "Bearish" if bears > bulls else "Mixed"
    agree = bulls if aligned_direction == "Bullish" else \
            bears if aligned_direction == "Bearish" else max(bulls, bears)
    checks = [(label, v["direction"], v["direction"] == aligned_direction)
              for label, v in available.items()]
    return {"agree": agree, "total": total, "aligned_direction": aligned_direction,
            "alignment_pct": round(agree / total * 100, 0), "checks": checks, "skipped": skipped,
            "note": (f"{len(skipped)} timeframe(s) not yet available: {', '.join(skipped)}."
                    if skipped else "All timeframes available.")}


def interpret_timeframe_sequence(snapshot: dict) -> dict:
    """PIIP audit 2026-08, Batch 2 (Phase 8): reads the SEQUENCE of available timeframes from
    longest to shortest (Daily -> 30m -> 15m -> 5m) instead of just counting how many agree --
    'higher timeframes bullish, shortest one bearish' is a pullback inside a bigger trend, not the
    same thing as 'everything just turned bearish', even though timeframe_alignment() alone can't
    tell those apart. Built entirely from the same per-timeframe directions timeframe_alignment()
    already reads -- a different lens on the same data, not a new calculation."""
    order = ["Daily", "30m", "15m", "5m"]
    seq = [(label, snapshot[label]["direction"]) for label in order
          if snapshot.get(label, {}).get("available")]
    if len(seq) < 2:
        return {"interpretation": "INSUFFICIENT TIMEFRAMES", "sequence": seq,
                "note": "Needs at least 2 available timeframes to read a sequence."}

    dirs = [d for _, d in seq]
    if all(d == dirs[0] for d in dirs) and dirs[0] != "Flat":
        return {"interpretation": f"{dirs[0].upper()} ALIGNED ACROSS ALL TIMEFRAMES",
                "sequence": seq, "note": "Every available timeframe agrees."}

    # Anchor on the LONGEST available timeframe's direction and see how much of the shorter end
    # disagrees with it -- one disagreeing (the shortest) reads as a pullback inside the trend;
    # several disagreeing reads as real pressure; ALL of them disagreeing reads as a possible
    # regime transition (the anchor itself may be about to flip).
    higher_dir, shorter_dirs = dirs[0], dirs[1:]
    if higher_dir != "Flat" and shorter_dirs:
        disagree_count = sum(1 for d in shorter_dirs if d != higher_dir)
        if 0 < disagree_count < len(shorter_dirs):
            if disagree_count == 1 and shorter_dirs[-1] != higher_dir:
                return {"interpretation": f"{higher_dir.upper()} HIGHER-TIMEFRAME TREND — "
                                          "SHORT-TERM PULLBACK",
                        "sequence": seq, "note": "Only the shortest available timeframe disagrees "
                                                 "— likely noise inside the trend."}
            return {"interpretation": f"{higher_dir.upper()} TREND UNDER SHORT-TERM PRESSURE",
                    "sequence": seq, "note": f"{disagree_count} of {len(shorter_dirs)} shorter "
                                             "timeframes disagree — more than a single-bar pullback."}
        if disagree_count == len(shorter_dirs):
            return {"interpretation": "POTENTIAL REGIME TRANSITION", "sequence": seq,
                    "note": f"Every shorter timeframe disagrees with the longest available one "
                           f"({higher_dir}) — the anchor itself may be about to flip."}

    return {"interpretation": "MIXED — NO CLEAR TIMEFRAME SEQUENCE", "sequence": seq,
            "note": "No clean higher/lower-timeframe pattern in what's available right now."}


def update_trend_state(prev_state: dict | None, alignment: dict, confirm_reads: int = 2) -> dict:
    """Hysteresis state machine so the 'trend state' label doesn't flip on one noisy 30s refresh.
    A candidate direction (from timeframe_alignment, requiring >=60% agreement among available
    timeframes to count as a real lean rather than 'Range/Chop') must show up on `confirm_reads`
    CONSECUTIVE calls before the displayed state actually changes. Pure function — the caller
    (app.py) persists `prev_state`/return value in st.session_state across the page's 30s
    fragment reruns, same pattern already used for the Exit Quality alert's was_alert/is_alert
    edge trigger."""
    if alignment["total"] == 0:
        candidate = "Insufficient Data"
    elif alignment["alignment_pct"] >= 60:
        candidate = "Uptrend" if alignment["aligned_direction"] == "Bullish" else "Downtrend"
    else:
        candidate = "Range / Chop"

    prev_state = prev_state or {"state": candidate, "pending": candidate, "pending_count": 1,
                                "confirm_reads": confirm_reads}
    state, pending, pending_count = prev_state["state"], prev_state["pending"], prev_state["pending_count"]

    if candidate == pending:
        pending_count += 1
    else:
        pending, pending_count = candidate, 1

    changed = False
    if pending != state and pending_count >= confirm_reads:
        state, changed = pending, True

    return {"state": state, "changed": changed, "pending": pending, "pending_count": pending_count,
            "confirm_reads": confirm_reads, "candidate": candidate}
