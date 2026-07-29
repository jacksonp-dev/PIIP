"""Macro data layer -- Treasury yields, dollar index, oil, and Fed system liquidity.

Two free, keyless sources:
  * yfinance for yields/DXY/oil -- same batching discipline as zero_dte.py::fetch_intraday_batch,
    one multi-ticker call, never a per-ticker loop.
  * FRED's public CSV export endpoint (fred.stlouisfed.org/graph/fredgraph.csv) for system
    liquidity series -- no API key needed, unlike FRED's JSON API. These are the real series
    (not an approximation), unlike zero_dte.py's breadth/GEX proxies.

Verified live this session: all yfinance tickers below return real data; DX=F (dollar futures)
does not exist on Yahoo and was dropped in favor of DX-Y.NYB (the ICE dollar index), which works.
"""
from __future__ import annotations

from io import StringIO

import pandas as pd
import requests
import yfinance as yf

YIELD_TICKERS = {"13W": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
DXY_TICKER = "DX-Y.NYB"
OIL_TICKERS = {"WTI": "CL=F", "Brent": "BZ=F"}

MACRO_TICKERS = list(YIELD_TICKERS.values()) + [DXY_TICKER] + list(OIL_TICKERS.values())

FRED_SERIES = {
    "TGA": "WTREGEN",         # Treasury General Account balance
    "RRP": "RRPONTSYD",       # Overnight reverse repo usage
    "bank_reserves": "WRESBAL",
}
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_macro_batch(period: str = "5d", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """ONE batched call for every yield/DXY/oil ticker -- {} on any failure, never raises."""
    try:
        raw = yf.download(MACRO_TICKERS, period=period, interval=interval, group_by="ticker",
                          progress=False, threads=True, auto_adjust=True)
    except Exception:
        return {}
    out: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in MACRO_TICKERS:
            if tk in raw.columns.get_level_values(0):
                sub = raw[tk].dropna(how="all")
                if not sub.empty:
                    out[tk] = sub
    return out


def fetch_fred_series(series_id: str) -> pd.DataFrame | None:
    """One FRED series as a date/value frame, or None on any failure. '.' placeholder rows
    (no observation that day) are dropped."""
    try:
        r = requests.get(FRED_CSV_URL.format(sid=series_id), timeout=15,
                         headers={"User-Agent": "PIIP-research/0.1"})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = ["date", "value"]
        df = df[df["value"] != "."].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = df["value"].astype(float)
        return df.reset_index(drop=True)
    except Exception:
        return None


def liquidity_snapshot() -> dict:
    """Latest TGA / reverse-repo / bank-reserves levels plus their 1-week change. These move
    system liquidity in ways that matter for risk appetite but have no equities-side proxy."""
    out = {}
    for label, sid in FRED_SERIES.items():
        df = fetch_fred_series(sid)
        if df is None or df.empty:
            out[label] = None
            continue
        latest_date = df["date"].iloc[-1]
        latest_val = float(df["value"].iloc[-1])
        prior = df[df["date"] <= latest_date - pd.Timedelta(days=7)]
        chg_1w = latest_val - float(prior["value"].iloc[-1]) if not prior.empty else None
        out[label] = {"latest": latest_val, "as_of": latest_date.strftime("%Y-%m-%d"),
                      "chg_1w": chg_1w}
    return out


def yield_curve_snapshot(batch: dict[str, pd.DataFrame] | None = None) -> dict:
    """Latest yield at each maturity plus the 10Y-2Y-style spread this project can actually
    compute (13W-10Y, since ^FVX/^TNX/^TYX/^IRX are the only free yfinance yield tickers --
    no free 2Y series was found, so the classic 2s10s spread isn't buildable here)."""
    batch = batch if batch is not None else fetch_macro_batch()
    out = {}
    for label, tk in YIELD_TICKERS.items():
        df = batch.get(tk)
        out[label] = float(df["Close"].iloc[-1]) if df is not None and not df.empty else None
    if out.get("13W") is not None and out.get("10Y") is not None:
        out["spread_13w_10y"] = round(out["10Y"] - out["13W"], 3)
    else:
        out["spread_13w_10y"] = None
    return out


# The "big 3" market-moving macro prints -- deliberately NOT the full FRED catalog (Core CPI/PCE/
# GDP etc. are real and could be added the same way later) to keep this addition scoped to what
# actually moves markets on release day, not an exhaustive econ-data dump.
ECON_RELEASE_SERIES = {
    "CPI": "CPIAUCSL",              # headline CPI index level -- quoted as YoY % change
    "Unemployment Rate": "UNRATE",   # already a %, quoted as-is
    "Nonfarm Payrolls": "PAYEMS",    # jobs level in thousands -- quoted as the MoM change (\"jobs added\")
}


def _fred_yoy_pct(df: pd.DataFrame) -> float | None:
    """% change vs the observation ~12 months back -- how CPI/PCE-style price indices are always
    actually quoted in headlines ('CPI rose 3.2%'), not their raw index level."""
    if df is None or df.empty:
        return None
    latest_date, latest_val = df["date"].iloc[-1], float(df["value"].iloc[-1])
    prior = df[df["date"] <= latest_date - pd.DateOffset(months=12)]
    if prior.empty:
        return None
    return round((latest_val / float(prior["value"].iloc[-1]) - 1) * 100, 2)


def economic_releases_snapshot() -> dict:
    """Latest real macro releases from FRED's free CSV export -- same keyless pattern already
    used by liquidity_snapshot() above, just extended to the headline inflation/jobs prints.
    CPI is quoted as YoY % (what the number means in every headline); unemployment as its own
    level; payrolls as the month-over-month change (the actual 'jobs added' headline number, not
    the raw employment level, which is a huge number that means nothing on its own)."""
    out = {}
    for label, sid in ECON_RELEASE_SERIES.items():
        df = fetch_fred_series(sid)
        if df is None or df.empty:
            out[label] = None
            continue
        latest_val = float(df["value"].iloc[-1])
        entry = {"latest": latest_val, "as_of": df["date"].iloc[-1].strftime("%Y-%m-%d")}
        if label == "CPI":
            entry["yoy_pct"] = _fred_yoy_pct(df)
        elif label == "Nonfarm Payrolls":
            entry["mom_change_thousands"] = (round(latest_val - float(df["value"].iloc[-2]), 1)
                                             if len(df) >= 2 else None)
        out[label] = entry
    return out
