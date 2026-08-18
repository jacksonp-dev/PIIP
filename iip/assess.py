"""Turn raw data into DESCRIPTIVE, color-coded flags — the stock's STATE, not a buy/sell call.

The model is ~a coin flip, so a green flag means "this condition is present" (e.g. uptrend), NEVER
"buy this." Colors read as intensity/notability, not good-vs-bad advice. Each flag is
{color, label, detail}. Feed cards use the fast flags; the detail modal adds earnings + analyst.
"""
from __future__ import annotations

from . import baseline, data, fundamentals
from . import deterministic as det

GREEN, YELLOW, RED, GRAY = "🟢", "🟡", "🔴", "⚪"


def _flag(color, label, detail):
    return {"color": color, "label": label, "detail": detail}


def trend_flag(tech: dict) -> dict:
    above200 = bool(tech.get("above_sma200"))
    sma50, close = tech.get("sma50"), tech.get("close")
    above50 = sma50 == sma50 and close is not None and close >= sma50
    if above200 and above50:
        return _flag(GREEN, "Uptrend", "Above both the 50- and 200-day averages")
    if above200 or above50:
        return _flag(YELLOW, "Mixed", "Above one moving average, below the other")
    return _flag(RED, "Downtrend", "Below both the 50- and 200-day averages")


def momentum_flag(tech: dict) -> dict:
    r60, rsi = tech.get("ret_60d_pct"), tech.get("rsi14")
    if rsi is not None and rsi > 70:
        return _flag(YELLOW, "Overbought", f"RSI {rsi:.0f} — stretched, prone to a pullback")
    if rsi is not None and rsi < 30:
        return _flag(YELLOW, "Oversold", f"RSI {rsi:.0f} — beaten down, prone to a bounce")
    if r60 is not None and r60 > 0:
        return _flag(GREEN, "Positive", f"+{r60:.0f}% over 60d, RSI {rsi:.0f}")
    return _flag(RED, "Negative", f"{r60:.0f}% over 60d, RSI {rsi:.0f}" if r60 is not None else "weak")


def vol_flag(tech: dict, iv_pct) -> dict:
    hv = tech.get("hv60_annual_pct")
    if iv_pct is None:
        return _flag(GRAY, "No IV", "No options IV available")
    if hv and iv_pct > hv * 1.4:
        return _flag(RED, "High IV",
                     f"IV {iv_pct:.0f}% >> realized {hv:.0f}% — options expensive + IV-crush risk (a caution, not 'bearish')")
    if hv and iv_pct > hv * 1.1:
        return _flag(YELLOW, "Elevated IV", f"IV {iv_pct:.0f}% > realized {hv:.0f}% — a bit pricey to buy")
    return _flag(GREEN, "Normal IV", f"IV {iv_pct:.0f}% ~ realized {hv:.0f}%" if hv else f"IV {iv_pct:.0f}%")


def model_flag(direction, conf) -> dict:
    lab = {"bullish": "Bullish", "bearish": "Bearish", "neutral": "Neutral"}.get(direction, "?")
    return _flag(GRAY, lab, f"Mechanical lean (conf {conf}) — ~a coin flip, ONE input only")


def earnings_flag(earn: dict | None) -> dict:
    if not earn:
        return _flag(GRAY, "Earnings ?", "No earnings date found")
    d = earn["days"]
    if d < 0:
        return _flag(GRAY, f"Reported ~{-d}d ago", f"Last earnings {earn['date']}")
    if d <= 7:
        return _flag(RED, f"Earnings in {d}d", f"{earn['date']} — big move likely + IV crush after; risky for options")
    if d <= 30:
        return _flag(YELLOW, f"Earnings in {d}d", f"{earn['date']} — within a month; inflates option prices")
    return _flag(GREEN, f"Earnings in {d}d", f"{earn['date']} — none imminent")


def analyst_flag(anal: dict | None, spot: float) -> dict:
    if not anal or not anal.get("mean"):
        return _flag(GRAY, "No coverage", "No analyst targets")
    mean = anal["mean"]
    up = (mean / spot - 1) * 100
    rec = (anal.get("rec") or "").replace("_", " ")
    detail = (f"mean target ${mean:.0f} ({up:+.0f}%), {rec}, {anal.get('n')} analysts "
              "— NOTE: analyst targets skew optimistic and are ~12-month horizons")
    if up > 15:
        return _flag(GREEN, f"Analysts +{up:.0f}%", detail)
    if up > 0:
        return _flag(YELLOW, f"Analysts +{up:.0f}%", detail)
    return _flag(RED, "At/above target", detail)


def catalyst_tag(card: dict, earn: dict | None = None, horizon_days: int = 35) -> dict:
    """Honest 'is there a REASON to research this, or is it just expensive noise?' tag.

    NOT a win-rate signal — direction is still a coin flip. This only asks whether the volatility
    the feed ranked on is *explained*:
      - 🟢 catalyst   : a dated event (earnings) lands inside the option's life -> a real reason + clock
      - 🟡 unexplained: IV is rich vs the stock's OWN norm but nothing is scheduled -> ambient/expensive
                        noise (or a catalyst the free data can't see, e.g. a biotech readout)
      - ⚪ quiet      : IV in line with its norm and nothing scheduled -> probably nothing to see
    The feed's 'hide noise' toggle drops the 🟡 'unexplained' cards.
    """
    tech = card.get("tech", {})
    iv = card.get("iv_pct")
    hv = tech.get("hv60_annual_pct")
    days = earn.get("days") if earn else None
    if days is not None and 0 <= days <= horizon_days:
        return {"key": "catalyst", "color": GREEN, "label": f"Catalyst: earnings in {days}d",
                "detail": f"{earn.get('date')} — a dated event inside the option's life (why IV is bid). "
                          "A REASON to research; NOT a direction — earnings gap both ways + IV crush after."}
    rich = hv and iv and iv > hv * 1.4
    if rich:
        why = ("earnings unknown" if days is None else "no earnings scheduled")
        return {"key": "unexplained", "color": YELLOW, "label": "Unexplained IV",
                "detail": f"IV {iv:.0f}% >> its realized {hv:.0f}% but {why} — likely ambient/expensive "
                          "noise, OR a catalyst the free data can't see (e.g. a biotech readout). Only "
                          "worth research if YOU know the catalyst."}
    return {"key": "quiet", "color": GRAY, "label": "No known catalyst",
            "detail": "IV roughly in line with its own norm and nothing scheduled — no obvious reason to move."}


def quick_flags(card: dict) -> dict:
    """Fast flags for a feed card, from data the scan already computed (no extra fetches)."""
    tech = card.get("tech", {})
    return {
        "Trend": trend_flag(tech),
        "Momentum": momentum_flag(tech),
        "Volatility": vol_flag(tech, card.get("iv_pct")),
        "Model lean": model_flag(card.get("direction_lean"), card.get("lean_conf")),
    }


def full(ticker: str) -> dict:
    """Everything for the detail modal: all flags (incl. earnings + analyst) + supporting data."""
    prices = data.get_prices(ticker, "1y")
    spot = data.last_valid_close(prices)
    tech = det.technical_metrics(prices)
    chain = data.get_option_chain(ticker, data.nearest_expiry(ticker, 30))
    om = det.option_metrics(spot, chain)
    hist = det.historical_move_distribution(prices, 30)
    earn = fundamentals.next_earnings(ticker)
    anal = fundamentals.analyst(ticker)
    fc = baseline.deterministic_forecast(ticker, prices, 30)
    flags = {
        "Trend": trend_flag(tech),
        "Momentum": momentum_flag(tech),
        "Volatility": vol_flag(tech, om.get("atm_iv_pct")),
        "Earnings": earnings_flag(earn),
        "Analysts": analyst_flag(anal, spot),
    }
    return {"ticker": ticker, "spot": spot, "tech": tech, "om": om, "hist": hist,
            "earn": earn, "anal": anal, "flags": flags, "chain": chain,
            "direction": fc["stock_direction"], "confidence": fc["confidence"],
            "reasoning": fc["reasoning"]}