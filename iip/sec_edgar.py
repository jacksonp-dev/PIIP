"""Free SEC EDGAR access — recent filings (8-K, Form 4) with no API key.

Fills two gaps `deep_research.py` already names in its own `unknowns()` list: recent insider
(Form 4) trades and general SEC filings. 8-Ks in particular are the "technically public, rarely
headlined" catalyst — material events (management changes, agreements, going-concern notices)
that are disclosed but usually get no mainstream coverage before the market reacts.

Best-effort, matches the convention in `deep_research.py::clinical_trials()`: any failure returns
[] so the UI never breaks. SEC's fair-access policy requires a descriptive User-Agent with contact
info on every request — see https://www.sec.gov/os/webmaster-faq#developers.
"""
from __future__ import annotations

import os
from datetime import date, datetime

import requests

# Contact comes from YOUR OWN .env (RESEARCH_CONTACT_EMAIL), not a hardcoded address -- see the
# matching note in screener.py for why (SEC's own policy wants each requester to self-identify,
# not for everyone to anonymously borrow one hardcoded contact).
_HEADERS = {"User-Agent": f"InvestIntelAgent research tool (contact: "
                          f"{os.getenv('RESEARCH_CONTACT_EMAIL', 'not-provided@example.com')})"}

_CIK_MAP: dict[str, str] | None = None   # module-level cache — ticker list rarely changes


def _ticker_cik_map() -> dict[str, str]:
    """{TICKER: 10-digit zero-padded CIK}, fetched once from SEC's own ticker list."""
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                          headers=_HEADERS, timeout=10)
        raw = r.json()
        _CIK_MAP = {row["ticker"].upper(): f"{int(row['cik_str']):010d}" for row in raw.values()}
    except Exception:
        _CIK_MAP = {}
    return _CIK_MAP


def recent_filings(ticker: str, forms: tuple[str, ...] = ("8-K", "4"),
                    lookback_days: int = 14) -> list[dict]:
    """Recent filings of the given form types for a ticker, newest first.

    Returns [{form, filed, days_ago, url}]. [] if the ticker isn't found, has no matching
    filings, or the request fails — never raises.
    """
    cik = _ticker_cik_map().get(ticker.upper())
    if not cik:
        return []
    try:
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                          headers=_HEADERS, timeout=10)
        recent = r.json().get("filings", {}).get("recent", {})
    except Exception:
        return []
    today = date.today()
    out = []
    for form, filed, accn, doc in zip(
        recent.get("form", []), recent.get("filingDate", []),
        recent.get("accessionNumber", []), recent.get("primaryDocument", []),
    ):
        if form not in forms:
            continue
        try:
            filed_d = datetime.fromisoformat(filed).date()
        except ValueError:
            continue
        days_ago = (today - filed_d).days
        if days_ago < 0 or days_ago > lookback_days:
            continue
        accn_nodash = accn.replace("-", "")
        out.append({
            "form": form, "filed": filed, "days_ago": days_ago,
            "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nodash}/{doc}",
        })
    out.sort(key=lambda f: f["days_ago"])
    return out
