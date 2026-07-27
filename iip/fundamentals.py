"""Free 'company forecast / event' data via Yahoo (yfinance): next earnings date + analyst targets.

Honest caveats baked into how we present these downstream:
- Analyst price targets skew OPTIMISTIC and are ~12-month horizons — not a near-term signal.
- Earnings dates are the useful part: a known catalyst that inflates option prices (then IV-crushes).
Best-effort — returns None on failure so the UI never breaks.
"""
from __future__ import annotations

from datetime import datetime, timezone

import yfinance as yf


def next_earnings(ticker: str) -> dict | None:
    """{'date': ISO, 'days': int} for the next earnings, or None. days<0 means it just reported."""
    try:
        info = yf.Ticker(ticker).info or {}
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if ts:
            d = datetime.fromtimestamp(ts, tz=timezone.utc)
            return {"date": d.date().isoformat(),
                    "days": (d - datetime.now(timezone.utc)).days}
    except Exception:
        pass
    return None


def company_info(ticker: str) -> dict:
    """{'sector','industry','name'} from Yahoo — used to tune news toward sector catalysts."""
    try:
        info = yf.Ticker(ticker).info or {}
        return {"sector": info.get("sector"), "industry": info.get("industry"),
                "name": info.get("shortName") or info.get("longName")}
    except Exception:
        return {}


def analyst(ticker: str) -> dict | None:
    """{'mean','high','low','rec','n'} analyst price targets + consensus, or None."""
    try:
        info = yf.Ticker(ticker).info or {}
        mean = info.get("targetMeanPrice")
        if mean:
            return {"mean": mean, "high": info.get("targetHighPrice"),
                    "low": info.get("targetLowPrice"),
                    "rec": info.get("recommendationKey"),
                    "n": info.get("numberOfAnalystOpinions")}
    except Exception:
        pass
    return None