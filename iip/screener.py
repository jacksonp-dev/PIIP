"""Wide-universe stock/ETF screener -- the real S&P 500 constituent list (free, no API key:
Wikipedia's own maintained table) plus the major index/sector ETFs (SPY, QQQ, IWM, DIA, XLK,
XLF, ...), batched price/day-change/sparkline scan.

Deliberately separate from scanner.UNIVERSE (Feed's curated ~80-stock list of mega-caps/lower-
priced/biotech/penny names) -- this is meant to search much wider on request, not replace Feed.
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests
import yfinance as yf

from . import zero_dte as zd

_WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia returns HTTP 403 for requests with no User-Agent (confirmed live) -- pd.read_html()
# fetching the URL directly sends none, so fetch the HTML ourselves first. Same header convention
# already used for sec_edgar.py's own fair-access requirement.
#
# Contact comes from YOUR OWN .env (RESEARCH_CONTACT_EMAIL), not a hardcoded address -- this used
# to be the original developer's personal email baked into every clone of this repo. Wikipedia/SEC
# rate-limit by IP, not by this string, so that never "shared" a quota -- but every other person
# running an unmodified clone would still have been identifying THEIR traffic as coming from
# someone else's inbox, which is exactly backwards from the point of a fair-access contact header.
_HEADERS = {"User-Agent": f"InvestIntelAgent research tool (contact: "
                          f"{os.getenv('RESEARCH_CONTACT_EMAIL', 'not-provided@example.com')})"}


def sp500_constituents() -> list[dict]:
    """{ticker, name, sector} for every current S&P 500 constituent, from Wikipedia's own
    maintained table. Best-effort: returns [] if the fetch fails -- no fabricated fallback list."""
    try:
        html = requests.get(_WIKI_SP500_URL, headers=_HEADERS, timeout=10).text
        # StringIO, not the raw string -- passing a bare (huge) HTML string directly gets
        # misdetected as a filename/URL by the underlying lxml parser (confirmed live: it tried
        # to etree.parse() the HTML text itself as a "filename"). Wrapping it removes the ambiguity.
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
        out = []
        for _, row in df.iterrows():
            tk = str(row.get("Symbol", "")).strip().replace(".", "-")   # yfinance wants BRK-B not BRK.B
            if tk:
                out.append({"ticker": tk, "name": str(row.get("Security", tk)),
                           "sector": str(row.get("GICS Sector", ""))})
        return out
    except Exception:
        return []


_INDEX_ETF_NAMES = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
    "IWM": "iShares Russell 2000 ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
}


def etf_universe() -> list[dict]:
    """{ticker, name, sector} for the major index + sector ETFs -- reuses zero_dte.py's own
    INDEX_TICKERS/SECTOR_ETFS basket definitions rather than maintaining a second, separately-
    curated ETF list that could quietly drift out of sync with the 0DTE page's own universe."""
    out = [{"ticker": tk, "name": _INDEX_ETF_NAMES.get(tk, tk), "sector": "ETF — Index"}
           for tk in zd.INDEX_TICKERS]
    out += [{"ticker": tk, "name": f"{sector} Sector SPDR", "sector": f"ETF — {sector}"}
           for sector, tk in zd.SECTOR_ETFS.items()]
    return out


def _split_multi(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Same splitting pattern as zero_dte.py's batch fetchers -- yf.download's multi-ticker
    return is one wide (ticker, field) MultiIndex frame, split back into per-ticker OHLCV."""
    out: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            if tk in raw.columns.get_level_values(0):
                sub = raw[tk].dropna(how="all")
                if not sub.empty:
                    out[tk] = sub
    elif len(tickers) == 1 and not raw.empty:
        out[tickers[0]] = raw
    return out


def batch_scan(tickers: list[str], chunk_size: int = 100) -> dict[str, dict]:
    """{ticker: {price, day_change_pct, volume, spark}} for every ticker with usable data.
    Batched in chunks, never one request per ticker -- a several-hundred-ticker universe in one
    giant call risks timeouts, so this trades one huge request for a handful of medium ones."""
    out: dict[str, dict] = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(tickers=chunk, period="1mo", interval="1d", group_by="ticker",
                              progress=False, threads=True, auto_adjust=True)
        except Exception:
            continue
        for tk, df in _split_multi(raw, chunk).items():
            closes = df["Close"].dropna()
            if len(closes) < 1:
                continue
            try:
                price = float(closes.iloc[-1])
                day_chg = (price / float(closes.iloc[-2]) - 1) * 100 if len(closes) >= 2 else None
                vol = df.get("Volume")
                vol_last = float(vol.iloc[-1]) if vol is not None and len(vol) and vol.iloc[-1] == vol.iloc[-1] else None
                out[tk] = {"price": price, "day_change_pct": day_chg, "volume": vol_last,
                          "spark": [float(v) for v in closes.tail(20).tolist()]}
            except Exception:
                continue
    return out
