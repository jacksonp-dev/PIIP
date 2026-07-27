"""Opportunity scanner — ranks a universe by EXPECTED MOVEMENT (the options market's implied move,
~30d), the honest read of 'about to move big, in either direction.' News headlines are attached as
CONTEXT (the 'why'), never as the ranking signal (our research flagged news-sentiment as ~noise).

log_card_forecast records the tool's OWN lean per card so the Scorecard can grade, honestly,
whether the model's picks beat chance. That tracking loop is the point.
"""
from __future__ import annotations

import yfinance as yf

from . import baseline, data
from . import deterministic as det
from . import predictions as pred

# ~30 liquid mega-caps with active options markets.
MEGA_CAPS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "WMT", "XOM", "UNH", "MA", "JNJ", "PG", "HD", "COST", "ORCL", "NFLX",
    "AMD", "CRM", "BAC", "KO", "PEP", "DIS", "ADBE", "INTC", "QCOM", "PLTR",
]

# Liquid, typically lower-priced names — their contracts often run $30–$300, so a small
# budget has real, tradeable choices. (Any that are thin/delisted just get skipped by the scan.)
LOWER_PRICED = [
    "SOFI", "F", "T", "NIO", "RIVN", "LCID", "HOOD", "UBER", "SNAP", "PINS",
    "PLUG", "RIOT", "MARA", "CCL", "AAL", "VZ", "PFE", "CSCO", "WBD", "GRAB",
]

# Liquid, catalyst-driven biotech / clinical-stage names — where the "big move before a trial
# readout / FDA decision" lottery plays live. IV often spikes hard ahead of catalysts, so these
# float to the top of the expected-move ranking. (Any delisted/illiquid ones are skipped.)
BIOTECH = [
    "SRPT", "VKTX", "CRSP", "BEAM", "NTLA", "IONS", "MRNA", "BNTX", "ARWR",
    "AXSM", "CYTK", "INSM", "APLS", "RXRX", "KRYS",
]

# Low-priced (~$1-6), liquid, Robinhood-listed "penny-ish" names — for playing with small money. All
# exchange-listed (NOT OTC pink-sheet shells, which Robinhood doesn't carry). These are the noisiest,
# most-dilutive corner, so the honest tools matter MOST here (cash-runway = dilution flag; ⚠️ = thin
# options). Speculative EV / AV / crypto-miner / AI names. Thin/delisted ones are skipped by the scan.
PENNY = [
    "CHPT", "RUN", "FUBO", "SOUN", "BBAI", "CIFR", "WULF", "CLSK", "RIG", "LAZR",
    "JOBY", "ACHR", "AUR", "KDK", "OPEN",
]

UNIVERSE = MEGA_CAPS + LOWER_PRICED + BIOTECH + PENNY

# Ticker -> company name, so cards disambiguate (e.g. BEAM = Beam Therapeutics, not the crypto token).
NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet",
    "META": "Meta Platforms", "TSLA": "Tesla", "AVGO": "Broadcom", "JPM": "JPMorgan Chase", "V": "Visa",
    "WMT": "Walmart", "XOM": "Exxon Mobil", "UNH": "UnitedHealth", "MA": "Mastercard",
    "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble", "HD": "Home Depot", "COST": "Costco",
    "ORCL": "Oracle", "NFLX": "Netflix", "AMD": "Advanced Micro Devices", "CRM": "Salesforce",
    "BAC": "Bank of America", "KO": "Coca-Cola", "PEP": "PepsiCo", "DIS": "Walt Disney",
    "ADBE": "Adobe", "INTC": "Intel", "QCOM": "Qualcomm", "PLTR": "Palantir",
    "SOFI": "SoFi Technologies", "F": "Ford Motor", "T": "AT&T", "NIO": "NIO", "RIVN": "Rivian",
    "LCID": "Lucid Group", "HOOD": "Robinhood", "UBER": "Uber", "SNAP": "Snap", "PINS": "Pinterest",
    "PLUG": "Plug Power", "RIOT": "Riot Platforms", "MARA": "MARA Holdings", "CCL": "Carnival",
    "AAL": "American Airlines", "VZ": "Verizon", "PFE": "Pfizer", "CSCO": "Cisco",
    "WBD": "Warner Bros. Discovery", "GRAB": "Grab Holdings",
    "SRPT": "Sarepta Therapeutics", "VKTX": "Viking Therapeutics", "CRSP": "CRISPR Therapeutics",
    "BEAM": "Beam Therapeutics", "NTLA": "Intellia Therapeutics", "IONS": "Ionis Pharmaceuticals",
    "MRNA": "Moderna", "BNTX": "BioNTech", "ARWR": "Arrowhead Pharmaceuticals",
    "AXSM": "Axsome Therapeutics", "CYTK": "Cytokinetics", "INSM": "Insmed",
    "APLS": "Apellis Pharmaceuticals", "RXRX": "Recursion Pharmaceuticals", "KRYS": "Krystal Biotech",
    "CHPT": "ChargePoint", "RUN": "Sunrun", "FUBO": "fuboTV", "SOUN": "SoundHound AI",
    "BBAI": "BigBear.ai", "CIFR": "Cipher Mining", "WULF": "TeraWulf", "CLSK": "CleanSpark",
    "RIG": "Transocean", "LAZR": "Luminar", "JOBY": "Joby Aviation", "ACHR": "Archer Aviation",
    "AUR": "Aurora Innovation", "KDK": "Kodiak AI", "OPEN": "Opendoor",
}


def _news(tk: str, n: int = 5) -> list[dict]:
    """Quick recent headlines from yfinance (best-effort; schema varies, so handle flat + nested)."""
    try:
        items = yf.Ticker(tk).news or []
    except Exception:
        return []
    out = []
    for it in items[:n]:
        content = it.get("content") or {}
        title = it.get("title") or content.get("title")
        pub = it.get("publisher") or (content.get("provider") or {}).get("displayName")
        link = (it.get("link") or (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url") or "")
        if title:
            out.append({"title": title, "publisher": pub or "", "link": link})
    return out


# sector-specific terms that bias the news search toward that sector's catalysts (relevance ranking)
_SECTOR_TERMS = {
    "Healthcare": "FDA trial phase approval data drug",
    "Technology": "product launch chip AI guidance earnings",
    "Energy": "oil production output OPEC",
    "Financial Services": "earnings guidance rates",
    "Consumer Cyclical": "sales guidance earnings",
    "Consumer Defensive": "sales guidance earnings",
    "Communication Services": "subscribers guidance earnings ads",
    "Basic Materials": "production prices demand",
    "Industrials": "orders guidance earnings",
}

# Finer INDUSTRY-level tuning (checked BEFORE sector) — matches keywords in the industry string,
# so e.g. an EV maker gets delivery/production/recall news, not generic "sales/earnings".
_INDUSTRY_TERMS = [
    (("biotech", "drug", "pharma", "diagnostic", "device", "medical", "life sciences"),
     "FDA trial phase approval data drug"),
    (("auto", "vehicle"), "deliveries production EV battery recall guidance"),
    (("semiconductor",), "chip AI data center guidance foundry"),
    (("software", "internet", "cloud", "information technology"),
     "guidance users subscribers cloud growth"),
    (("bank", "insurance", "capital markets", "credit", "financial"),
     "earnings rates deposits guidance"),
    (("oil", "gas", "energy"), "oil production OPEC prices output"),
    (("airline", "airport", "travel", "lodging", "hotel", "cruise", "resort"),
     "bookings capacity demand guidance"),
    (("retail", "apparel", "restaurant", "footwear"), "sales same-store traffic guidance"),
    (("mining", "metal", "gold", "copper"), "production prices demand"),
    (("aerospace", "defense"), "contract orders backlog guidance"),
]


def full_news(ticker: str, sector: str | None = None, industry: str | None = None,
              n: int = 12) -> list[dict]:
    """More articles via Google News RSS (free, no key): headlines + clickable links + source + date.
    The query is biased toward the ticker's INDUSTRY catalysts (FDA/trial for biotech, deliveries for
    EVs, chip/AI for semis, etc.) so the most relevant catalyst news floats up. Best-effort → [] on fail."""
    import xml.etree.ElementTree as ET
    from urllib.parse import quote

    import requests
    ind = (industry or "").lower()
    terms = ""
    for keys, t in _INDUSTRY_TERMS:
        if any(k in ind for k in keys):
            terms = t
            break
    if not terms:
        terms = _SECTOR_TERMS.get(sector or "", "")          # fall back to broad sector terms
    query = f"{ticker} stock {terms}".strip()
    out = []
    try:
        url = (f"https://news.google.com/rss/search?q={quote(query)}"
               "&hl=en-US&gl=US&ceid=US:en")
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
        for item in root.findall(".//item")[:n]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            src_el = item.find("source")
            pub = src_el.text.strip() if (src_el is not None and src_el.text) else ""
            when = (item.findtext("pubDate") or "")[:16]
            if pub and title.endswith(f"- {pub}"):        # strip the " - Publisher" suffix
                title = title[: -(len(pub) + 2)].strip()
            if title:
                out.append({"title": title, "link": link, "publisher": pub, "date": when})
    except Exception:
        pass
    return out


def cheapest_within_move(spot: float, chain: dict, em_abs: float | None) -> dict | None:
    """Least capital for a *sensible* entry, as {cost, strike, type}: the cheapest TRADEABLE,
    within-expected-move strike (call or put). Skips stale below-intrinsic quotes; excludes far-OTM
    lottery strikes (those need a move > 1 expected move). None if nothing qualifies. Returns the exact
    strike so the card can point you at a *findable* contract, not just an unanchored dollar figure."""
    if not em_abs or em_abs <= 0:
        return None
    best = None
    for key in ("calls", "puts"):
        df = chain.get(key)
        if df is None:
            continue
        is_call = key == "calls"
        for _, row in df.iterrows():
            prem = det._mid(row)
            if not prem or prem != prem:
                continue
            strike = float(row["strike"])
            intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
            if prem < intrinsic - 0.005:                 # stale phantom quote (below intrinsic) — skip
                continue
            eff = max(prem, intrinsic)
            be = (strike + eff) if is_call else (strike - eff)
            req = (be - spot) if is_call else (spot - be)
            if req <= em_abs:                            # within ~1 expected move (a reachable 🟢 strike)
                cost = int(round(eff * 100))
                if best is None or cost < best["cost"]:
                    best = {"cost": cost, "strike": strike, "type": "call" if is_call else "put"}
    return best


def scan_ticker(tk: str) -> dict:
    prices = data.get_prices(tk, "1y")
    spot = float(prices["Close"].iloc[-1])
    exp30 = data.nearest_expiry(tk, 30)                  # the ~30d research horizon (card + chain agree on it)
    chain = data.get_option_chain(tk, exp30)
    om = det.option_metrics(spot, chain)
    fc = baseline.deterministic_forecast(tk, prices, 30)
    _em_pct = om.get("expected_move_straddle_pct") or om.get("expected_move_iv_pct")
    _me = cheapest_within_move(spot, chain, spot * (_em_pct / 100) if _em_pct else None)
    if _me:
        _me["expiry"] = exp30                            # anchor the quote to a specific, findable expiry
    return {
        "ticker": tk, "name": NAMES.get(tk, tk), "spot": spot,
        # expected move = the options market's own forecast of magnitude (IV-based, straddle fallback)
        # straddle-first: it's built from real premiums (robust); yfinance's IV field is often garbage
        "expected_move_pct": _em_pct,
        # cheapest sensible (within-move, non-stale) entry AT ~30d — powers the budget filter + card label
        "min_entry": _me,
        "min_entry_cost": _me["cost"] if _me else None,
        "iv_pct": om.get("atm_iv_pct"), "dte": om.get("dte_days"),
        "direction_lean": fc["stock_direction"], "lean_conf": fc["confidence"],
        "prob_up": fc["stock_prob_up"],
        "tech": det.technical_metrics(prices),   # snapshot for fast feed-card flags
        "call": baseline.simulated_option(spot, chain, "bullish"),
        "put": baseline.simulated_option(spot, chain, "bearish"),
        "news": _news(tk),
    }


def scan(universe: list[str] | None = None) -> list[dict]:
    """Scan the universe and return cards ranked by expected move (biggest first)."""
    cards = []
    for tk in (universe or UNIVERSE):
        try:
            c = scan_ticker(tk)
            if c["expected_move_pct"]:
                cards.append(c)
        except Exception:
            pass                          # a flaky ticker never breaks the whole scan
    cards.sort(key=lambda c: c["expected_move_pct"], reverse=True)
    return cards


def log_card_forecast(card: dict, db: str = pred.DB_PATH, horizon: int = 30) -> int | None:
    """Log the MODEL's own 30d forecast for a card (deduped per ticker/day) so the Scorecard can later
    grade whether the tool's leans beat chance. This is the tool's forecast, NOT the user's trade."""
    direction = card["direction_lean"]
    opt = card["call"] if direction == "bullish" else (card["put"] if direction == "bearish" else {})
    rec = {"ticker": card["ticker"], "horizon_days": horizon, "source": "deterministic",
           "spot": card["spot"], "stock_direction": direction,
           "stock_expected_move_pct": card.get("expected_move_pct"),
           "stock_prob_up": card.get("prob_up"), "confidence": card.get("lean_conf"),
           "reasoning": {"mode": "feed_scan"}}
    if opt:
        rec.update(opt)
    return pred.log_if_new(rec, db)


