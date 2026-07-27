"""Deep Research Mode — Phase 1: a DATA-FIRST research dossier that REDUCES uncertainty.

Design rules (enforced, not aspirational):
  * Every field is a `_field(value, conf, source, why)` — it carries its own confidence, source,
    and 'why it matters'. Nothing is invented: if there's no free source, the value is UNKNOWN.
  * NO buy/sell recommendation. The output exposes what we know, what's weak, and what's missing.
  * Phase 1 is DETERMINISTIC ONLY — free data (yfinance, ClinicalTrials.gov, our own computes). The
    LLM-synthesis sections (bull/counter thesis, 'questions you missed') are Phase 2 and are NOT here.

The Research Completeness scorecard is computed from the fields' confidences — it's a data-availability
readout (how much did we actually verify?), NOT a buy score.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import requests
import yfinance as yf

from . import data, fundamentals, scanner, sec_edgar
from . import deterministic as det

HIGH, MED, LOW, UNKNOWN = "High", "Medium", "Low", "UNKNOWN"
TODAY = lambda: date.today().isoformat()


def _field(value, conf=HIGH, source="", why="", read=""):
    """One dossier fact with confidence, source, date, why-it-matters, and a PLAIN-ENGLISH read
    (what the number actually means for a non-expert). value=None -> UNKNOWN."""
    known = value is not None and value == value and value != ""
    return {"value": value if known else "UNKNOWN", "conf": conf if known else UNKNOWN,
            "source": source, "date": TODAY(), "why": why, "read": read if known else ""}


def _lvl(x, lo, hi, labels):
    """Bucket a number into (low, mid, high) plain-English labels. '' if not a number."""
    if not isinstance(x, (int, float)) or x != x:
        return ""
    return labels[0] if x < lo else (labels[2] if x >= hi else labels[1])


def _pct(x):
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) and x == x else None


def _money(x):
    if not isinstance(x, (int, float)) or x != x:
        return None
    a = abs(x)
    for u, d in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= u:
            return f"${x / u:.2f}{d}"
    return f"${x:,.0f}"


# ─────────────────────────── sections (free data only) ───────────────────────────
def executive_summary(info: dict, spot: float) -> dict:
    return {
        "Company": _field(info.get("longName") or info.get("shortName"), HIGH, "Yahoo Finance"),
        "Sector": _field(info.get("sector"), HIGH, "Yahoo Finance"),
        "Industry": _field(info.get("industry"), HIGH, "Yahoo Finance"),
        "Current price": _field(f"${spot:,.2f}", HIGH, "Yahoo Finance (delayed)"),
        "Market cap": _field(_money(info.get("marketCap")), HIGH, "Yahoo Finance"),
        "Employees": _field(info.get("fullTimeEmployees"), HIGH, "Yahoo Finance"),
        "Business summary": _field(info.get("longBusinessSummary"), MED, "Yahoo Finance",
                                   "How the company describes itself — a starting point, not analysis"),
    }


def financial_health(info: dict) -> dict:
    fcf, cash = info.get("freeCashflow"), info.get("totalCash")
    g, pm = info.get("revenueGrowth"), info.get("profitMargins")
    de, cr = info.get("debtToEquity"), info.get("currentRatio")
    runway = runway_read = None
    if isinstance(fcf, (int, float)) and fcf < 0 and isinstance(cash, (int, float)) and cash > 0:
        yrs = cash / abs(fcf)
        runway = f"~{yrs:.1f} yr at current burn"                 # key for cash-burning biotech
        runway_read = (f"loses money, and its cash lasts ~{yrs:.1f} more years at this burn rate — "
                       f"then it likely must raise money (which dilutes existing shareholders)")
    g_read = (f"sales are {'growing' if g > 0 else 'shrinking'} {abs(g) * 100:.0f}% a year"
              + (" — fast" if isinstance(g, float) and g > 0.20 else "")) if isinstance(g, (int, float)) and g == g else ""
    pm_read = ("unprofitable — it loses money overall" if isinstance(pm, (int, float)) and pm < 0
               else (f"keeps {pm * 100:.0f}¢ of profit per $1 of sales" if isinstance(pm, (int, float)) and pm == pm else ""))
    return {
        "Revenue (ttm)": _field(_money(info.get("totalRevenue")), HIGH, "Yahoo Finance",
                                read="total sales over the last 12 months"),
        "Revenue growth (yoy)": _field(_pct(g), HIGH, "Yahoo Finance", read=g_read),
        "Gross margin": _field(_pct(info.get("grossMargins")), HIGH, "Yahoo Finance",
                               read="profit left after the direct cost of the product/service"),
        "Operating margin": _field(_pct(info.get("operatingMargins")), HIGH, "Yahoo Finance"),
        "Profit margin": _field(_pct(pm), HIGH, "Yahoo Finance", read=pm_read),
        "Trailing EPS": _field(info.get("trailingEps"), HIGH, "Yahoo Finance",
                               read="profit per share over the last year (negative = losing money)"),
        "Total cash": _field(_money(cash), HIGH, "Yahoo Finance", read="cash on hand to fund operations"),
        "Total debt": _field(_money(info.get("totalDebt")), HIGH, "Yahoo Finance"),
        "Debt/Equity": _field(de, HIGH, "Yahoo Finance",
                              read=(f"{_lvl(de, 50, 150, ('low', 'moderate', 'high'))} leverage — borrowed money "
                                    f"vs shareholder money" if isinstance(de, (int, float)) and de == de else "")),
        "Current ratio": _field(cr, HIGH, "Yahoo Finance", "<1 can signal short-term liquidity stress",
                                read=(f"${cr:.1f} of short-term assets per $1 of short-term bills — "
                                      f"{_lvl(cr, 1.0, 1.5, ('tight', 'adequate', 'comfortable'))}"
                                      if isinstance(cr, (int, float)) and cr == cr else "")),
        "Free cash flow (ttm)": _field(_money(fcf), HIGH, "Yahoo Finance",
                                       read=("generates spare cash after spending" if isinstance(fcf, (int, float)) and fcf > 0
                                             else "burns cash — spends more than it brings in" if isinstance(fcf, (int, float)) else "")),
        "Operating cash flow (ttm)": _field(_money(info.get("operatingCashflow")), HIGH, "Yahoo Finance"),
        "Cash runway": _field(runway, MED, "Derived (cash / FCF burn)",
                              "How long until a cash-burner must raise money (dilution risk)", read=runway_read),
    }


def valuation(info: dict, spot: float) -> dict:
    pe, fpe = info.get("trailingPE"), info.get("forwardPE")
    ps, peg = info.get("priceToSalesTrailing12Months"), info.get("pegRatio")
    mean = info.get("targetMeanPrice")
    pe_read = (f"you pay ${pe:.0f} for every $1 of yearly profit — "
               f"{_lvl(pe, 15, 25, ('cheaper than', 'about the same as', 'richer than'))} the market's "
               f"typical ~20×" if isinstance(pe, (int, float)) and pe > 0 else
               ("no P/E — the company isn't profitable, so P/E doesn't apply" if pe in (None, 0) else ""))
    ps_read = (f"you pay ${ps:.1f} per $1 of annual sales — the go-to gauge when there's no profit yet"
               if isinstance(ps, (int, float)) and ps == ps else "")
    peg_read = (f"{_lvl(peg, 1.0, 2.0, ('cheap for its growth', 'fairly priced for its growth', 'pricey for its growth'))}"
                if isinstance(peg, (int, float)) and peg == peg else "")
    tgt_read = (f"analysts' average 12-mo target is ${mean:,.0f} ({(mean / spot - 1) * 100:+.0f}% vs today) "
                f"— but targets skew optimistic" if isinstance(mean, (int, float)) and mean and spot else "")
    return {
        "Trailing P/E": _field(pe, HIGH, "Yahoo Finance", read=pe_read),
        "Forward P/E": _field(fpe, HIGH, "Yahoo Finance",
                              read=("based on NEXT year's expected profit" if isinstance(fpe, (int, float)) and fpe == fpe else "")),
        "Price/Sales": _field(ps, HIGH, "Yahoo Finance", read=ps_read),
        "Price/Book": _field(info.get("priceToBook"), HIGH, "Yahoo Finance"),
        "EV/EBITDA": _field(info.get("enterpriseToEbitda"), HIGH, "Yahoo Finance"),
        "PEG ratio": _field(peg, MED, "Yahoo Finance", read=peg_read),
        "Analyst target (mean)": _field(_dollar(mean), MED, "Yahoo Finance",
                                        "Analyst targets skew optimistic + are ~12-month horizons", read=tgt_read),
        "Analyst target (low/high)": _field(
            f"{_dollar(info.get('targetLowPrice'))} / {_dollar(info.get('targetHighPrice'))}"
            if info.get("targetLowPrice") else None, MED, "Yahoo Finance",
            read="the range of where analysts think it goes — the spread shows how uncertain they are"),
        "Recommendation": _field((info.get("recommendationKey") or "").replace("_", " ") or None,
                                 LOW, "Yahoo Finance", "Sell-side consensus — a soft signal",
                                 read="the average analyst rating (buy/hold/sell) — a soft, often-optimistic signal"),
    }


def _dollar(x):
    return f"${x:,.2f}" if isinstance(x, (int, float)) and x == x else None


def ownership(info: dict, ticker: str) -> dict:
    ss, ssp = info.get("sharesShort"), info.get("sharesShortPriorMonth")
    trend = None
    if isinstance(ss, (int, float)) and isinstance(ssp, (int, float)) and ssp:
        trend = "rising" if ss > ssp * 1.05 else ("falling" if ss < ssp * 0.95 else "flat")
    inst, ins = info.get("heldPercentInstitutions"), info.get("heldPercentInsiders")
    shf = info.get("shortPercentOfFloat")
    return {
        "Institutional ownership": _field(_pct(inst), HIGH, "Yahoo Finance",
            read=(f"{inst * 100:.0f}% is held by big funds — "
                  f"{_lvl(inst, 0.4, 0.7, ('retail-heavy', 'moderately institutional', 'heavily institutional'))}"
                  if isinstance(inst, (int, float)) and inst == inst else "")),
        "Insider ownership": _field(_pct(ins), HIGH, "Yahoo Finance",
            read=("high insider ownership means management has skin in the game"
                  if isinstance(ins, (int, float)) and ins > 0.05 else "")),
        "Short % of float": _field(_pct(shf), HIGH, "Yahoo Finance",
            "High short interest = crowded bearish bet / squeeze fuel",
            read=(f"{shf * 100:.0f}% of shares are bet AGAINST the stock — "
                  f"{_lvl(shf, 0.08, 0.15, ('normal', 'elevated', 'heavily shorted (squeeze fuel)'))}"
                  if isinstance(shf, (int, float)) and shf == shf else "")),
        "Short ratio (days-to-cover)": _field(info.get("shortRatio"), HIGH, "Yahoo Finance",
            read="how many days of normal volume it'd take shorts to buy back — higher = more squeeze potential"),
        "Short interest trend": _field(trend, MED, "Derived (vs prior month)",
            read=("bearish bets are building" if trend == "rising" else
                  "bearish bets are unwinding" if trend == "falling" else "")),
        "Recent insider transactions": _insider_field(ticker),
    }


def _insider_field(ticker: str) -> dict:
    filings = sec_edgar.recent_filings(ticker, forms=("4",), lookback_days=45)
    if not filings:
        return _field(None, UNKNOWN, "SEC EDGAR Form 4",
                      "No Form 4 filed in the last 45 days (or fetch failed) — quiet insider activity")
    newest = filings[0]
    val = f"{len(filings)} Form 4 filing(s) in last 45d — most recent {newest['filed']} ({newest['days_ago']}d ago)"
    return _field(val, HIGH, "SEC EDGAR (data.sec.gov)",
                  "Insiders buying/selling their own stock — a cluster of Form 4s is worth reading",
                  read=f"see the filing: {newest['url']}")


def catalysts(ticker: str, trials: list) -> dict:
    earn = fundamentals.next_earnings(ticker)
    out = {}
    if earn and earn.get("days") is not None:
        d = earn["days"]
        when = f"{earn['date']} ({d}d away)" if d >= 0 else f"{earn['date']} ({-d}d ago)"
        out["Next earnings"] = _field(when, HIGH, "Yahoo Finance",
                                      "Known dated catalyst — inflates IV, then IV-crush after")
    else:
        out["Next earnings"] = _field(None, UNKNOWN, "Yahoo Finance")
    # clinical readouts (biotech) come from ClinicalTrials.gov, attached as dated catalysts
    near = [t for t in trials if t.get("primary_completion")]
    if near:
        soonest = min(near, key=lambda t: t["primary_completion"])
        out["Nearest trial completion"] = _field(
            f"{soonest['primary_completion']} — {soonest['phase']}: {soonest['brief'][:60]}",
            MED, "ClinicalTrials.gov",
            "Primary completion ≈ when data is collected; readout/PR usually follows")
    filings = sec_edgar.recent_filings(ticker, forms=("8-K",), lookback_days=14)
    if filings:
        newest = filings[0]
        val = f"{len(filings)} 8-K(s) in last 14d — most recent {newest['filed']} ({newest['days_ago']}d ago)"
        out["Recent SEC filings (8-K)"] = _field(
            val, HIGH, "SEC EDGAR (data.sec.gov)",
            "Material events (mgmt changes, agreements, going-concern) — public but rarely headlined",
            read=f"see the filing: {newest['url']}")
    else:
        out["Recent SEC filings (8-K)"] = _field(None, UNKNOWN, "SEC EDGAR",
                                                  "No 8-K in the last 14 days (or fetch failed)")
    return out


def options_analysis(ticker: str, spot: float) -> dict:
    try:
        exp = data.nearest_expiry(ticker, 30)
        chain = data.get_option_chain(ticker, exp)
        om = det.option_metrics(spot, chain)
    except Exception:
        return {"Options": _field(None, UNKNOWN, "Yahoo Finance", "No listed options / fetch failed")}
    em = om.get("expected_move_straddle_pct") or om.get("expected_move_iv_pct")
    # max pain: strike where total option value to holders is minimized (OI-weighted)
    max_pain = _max_pain(chain, spot)
    pcr = om.get("put_call_oi_ratio")
    return {
        "Expected move (~30d)": _field(f"±{em:.1f}%" if em else None, HIGH, "Computed (straddle)",
            read=(f"the options market expects the stock to swing about ±{em:.0f}% over the next month "
                  f"— in EITHER direction. Bigger = more uncertainty priced in" if em else "")),
        "ATM implied volatility": _field(f"{om['atm_iv_pct']:.0f}%" if om.get("atm_iv_pct") else None,
            HIGH, "Computed", read="how jumpy options expect the stock to be — higher IV = pricier options"),
        "Put/Call OI ratio": _field(pcr, HIGH, "Computed", ">1 = more puts (hedged/bearish); <1 = call-heavy",
            read=(f"{_lvl(pcr, 0.9, 1.1, ('more calls open (bullish tilt)', 'roughly balanced', 'more puts open (hedged/bearish tilt)'))}"
                  if isinstance(pcr, (int, float)) and pcr == pcr else "")),
        "Max pain strike": _field(_dollar(max_pain), MED, "Computed (OI-weighted)",
            "Where the most options expire worthless — a weak magnet, not a forecast",
            read="the price where option buyers lose the most — a weak pull near expiry, NOT a prediction"),
        "IV Rank / Percentile": _field(None, UNKNOWN, "Needs IV history (no clean free source)",
            "Whether IV is high vs its OWN past — we proxy with IV-vs-realized instead"),
    }


def _max_pain(chain, spot):
    try:
        calls, puts = chain["calls"], chain["puts"]
        strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
        strikes = [k for k in strikes if 0.5 * spot <= k <= 1.5 * spot]
        best, best_pain = None, None
        for k in strikes:
            call_pain = ((k - calls["strike"]).clip(lower=0) * calls["openInterest"].fillna(0)).sum()
            put_pain = ((puts["strike"] - k).clip(lower=0) * puts["openInterest"].fillna(0)).sum()
            pain = float(call_pain + put_pain)
            if best_pain is None or pain < best_pain:
                best, best_pain = k, pain
        return best
    except Exception:
        return None


def technicals(prices) -> dict:
    t = det.technical_metrics(prices)
    rsi = t.get("rsi14")
    return {
        "Trend vs 200-day": _field("above" if t.get("above_sma200") else "below", HIGH, "Computed",
            read=("trading ABOVE its 1-year average price — a longer-term uptrend" if t.get("above_sma200")
                  else "trading BELOW its 1-year average price — a longer-term downtrend")),
        "50-day SMA": _field(_dollar(t.get("sma50")), HIGH, "Computed", read="average price over the last ~10 weeks"),
        "200-day SMA": _field(_dollar(t.get("sma200")), HIGH, "Computed", read="average price over the last ~1 year"),
        "RSI (14)": _field(f"{rsi:.0f}" if rsi else None, HIGH, "Computed", ">70 overbought · <30 oversold",
            read=(f"{_lvl(rsi, 30, 70, ('oversold — beaten down, can bounce', 'neutral momentum', 'overbought — stretched, can pull back'))}"
                  if isinstance(rsi, (int, float)) and rsi == rsi else "")),
        "Return 60d": _field(_pct((t.get("ret_60d_pct") or 0) / 100) if t.get("ret_60d_pct") is not None else None,
            HIGH, "Computed", read="price change over the last ~3 months"),
        "Realized vol (60d, annual)": _field(f"{t['hv60_annual_pct']:.0f}%" if t.get("hv60_annual_pct") else None,
            HIGH, "Computed", read="how much the stock ACTUALLY moved lately (compare to IV to see if options are pricey)"),
    }


# ─────────────────────────── ClinicalTrials.gov (free structured biotech catalysts) ─────────────
def clinical_trials(company: str, limit: int = 8) -> list:
    """Live interventional trials for a sponsor via the free ClinicalTrials.gov v2 API. Returns a list
    of {brief, phase, status, conditions, enrollment, primary_completion, completion, nct}. [] on fail."""
    if not company:
        return []
    spons = company.split(",")[0].replace(" Inc.", "").replace(" Inc", "").strip()
    try:
        r = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.spons": spons, "filter.overallStatus": "RECRUITING|ACTIVE_NOT_RECRUITING",
                    "pageSize": limit,
                    "fields": ("protocolSection.identificationModule,protocolSection.statusModule,"
                               "protocolSection.designModule,protocolSection.conditionsModule")},
            timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        studies = r.json().get("studies", [])
    except Exception:
        return []
    out = []
    for s in studies:
        ps = s.get("protocolSection", {})
        idm, stm = ps.get("identificationModule", {}), ps.get("statusModule", {})
        dm, cm = ps.get("designModule", {}), ps.get("conditionsModule", {})
        phases = dm.get("phases") or []
        out.append({
            "brief": idm.get("briefTitle", ""),
            "nct": idm.get("nctId", ""),
            "phase": ", ".join(phases) if phases else "N/A",
            "status": (stm.get("overallStatus") or "").replace("_", " ").title(),
            "conditions": ", ".join(cm.get("conditions", [])[:3]),
            "enrollment": (dm.get("enrollmentInfo") or {}).get("count"),
            "primary_completion": (stm.get("primaryCompletionDateStruct") or {}).get("date"),
            "completion": (stm.get("completionDateStruct") or {}).get("date"),
        })
    # surface latest-phase, soonest-completion trials first
    order = {"PHASE3": 0, "PHASE2": 1, "PHASE1": 2}
    out.sort(key=lambda t: (min((order.get(p, 3) for p in t["phase"].replace(" ", "").split(",")), default=3),
                            t.get("primary_completion") or "9999"))
    return out


# ─────────────────────────── unknowns + completeness scorecard ───────────────────────────
def unknowns() -> list:
    """Honest list of things an investor wants that have NO reliable free source — exposed, not hidden."""
    return [
        ("Dark pool / block trades", "No free source", "Needs a paid flow-data vendor"),
        ("13F institutional changes (granular)", "Partial", "SEC 13F filings exist but need heavy parsing"),
        ("IV Rank / IV Percentile", "No free source", "Needs historical IV; we proxy with IV-vs-realized"),
        ("Historical FDA approval probability", "No free source", "Proprietary datasets only"),
        ("Borrow rate / hard-to-borrow", "No free source", "Broker/vendor data"),
        ("FDA PDUFA / decision dates", "No free calendar", "Scattered in filings & PRs — read them manually"),
        ("Earnings call transcript", "Not auto-fetched", "Read on the IR site / Seeking Alpha"),
    ]


# Research areas: (weight, which section fields feed it). Higher weight = more decision-critical.
def _score_area(fields: dict) -> float:
    """1.0 if mostly High-confidence, 0.5 if partial, 0.0 if nothing verified."""
    if not fields:
        return 0.0
    confs = [f["conf"] for f in fields.values()]
    known = [c for c in confs if c != UNKNOWN]
    if not known:
        return 0.0
    high = sum(1 for c in known if c == HIGH)
    frac_known = len(known) / len(confs)
    frac_high = high / len(known)
    return round(min(1.0, 0.4 + 0.6 * frac_known) * (0.6 + 0.4 * frac_high), 2)


def completeness_scorecard(sections: dict, is_biotech: bool) -> dict:
    """Rate each research AREA strong/weak/missing from field confidence, compute overall %, and pick
    the single highest-leverage next research step (importance × gap). NOT a buy score."""
    weights = {
        "Financial health": 1.0, "Valuation": 0.8, "Technicals": 0.6, "Options": 0.7,
        "Catalysts": 1.0, "Ownership / smart money": 0.7,
        "Clinical trials": (1.2 if is_biotech else 0.0),
    }
    next_action = {
        "Financial health": "Pull the latest 10-Q from SEC EDGAR and check cash runway + debt schedule.",
        "Valuation": "Build a peer comp table vs 2–3 direct competitors.",
        "Technicals": "Confirm support/resistance on the daily chart.",
        "Options": "Check IV vs its own history on your broker (IV rank).",
        "Catalysts": "Find the exact next dated event (earnings PR, conference, PDUFA) in filings.",
        "Ownership / smart money": "Review recent insider Form 4 trades on SEC EDGAR.",
        "Clinical trials": "Read the Phase 3 clinical trial protocol + primary endpoint on ClinicalTrials.gov.",
    }
    area_fields = {
        "Financial health": sections["Financial health"], "Valuation": sections["Valuation"],
        "Technicals": sections["Technicals"], "Options": sections["Options"],
        "Catalysts": sections["Catalysts"], "Ownership / smart money": sections["Ownership / smart money"],
        "Clinical trials": sections.get("_clinical_fields", {}),
    }
    strong, weak, missing, wsum, score_sum = [], [], [], 0.0, 0.0
    gaps = []
    for area, w in weights.items():
        if w == 0:
            continue
        sc = _score_area(area_fields.get(area, {}))
        wsum += w
        score_sum += w * sc
        (strong if sc >= 0.75 else weak if sc >= 0.35 else missing).append(area)
        gaps.append((w * (1.0 - sc), area))
    pct = round(100 * score_sum / wsum) if wsum else 0
    top = max(gaps, key=lambda g: g[0])[1] if gaps else None
    return {"pct": pct, "strong": strong, "weak": weak, "missing": missing,
            "next_step": next_action.get(top, "Read the latest 10-Q."), "next_area": top}


# ─────────────────────────── orchestrator ───────────────────────────
def dossier(ticker: str) -> dict:
    """Build the full Phase-1 (data-only) dossier for a ticker. Best-effort; sections degrade to UNKNOWN."""
    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception:
        info = {}
    prices = data.get_prices(ticker, "1y")
    spot = float(prices["Close"].iloc[-1])
    sector = (info.get("sector") or "")
    industry = (info.get("industry") or "")
    is_biotech = "Healthcare" in sector or any(
        k in industry.lower() for k in ("biotech", "drug", "pharma"))
    company = info.get("longName") or info.get("shortName") or ticker

    trials = clinical_trials(company) if is_biotech else []
    clin_fields = {}
    for i, t in enumerate(trials[:6]):
        label = f"{t['phase']} · {t['brief'][:50]}"
        val = (f"{t['status']} · enroll {t.get('enrollment') or '?'} · "
               f"primary completion {t.get('primary_completion') or 'UNKNOWN'} · {t['nct']}")
        clin_fields[label] = _field(val, MED, "ClinicalTrials.gov",
                                    "Trial phase/completion = the biotech's real catalyst clock")

    sections = {
        "Executive summary": executive_summary(info, spot),
        "Catalysts": catalysts(ticker, trials),
        "Financial health": financial_health(info),
        "Valuation": valuation(info, spot),
        "Technicals": technicals(prices),
        "Options": options_analysis(ticker, spot),
        "Ownership / smart money": ownership(info, ticker),
        "_clinical_fields": clin_fields,
    }
    score = completeness_scorecard(sections, is_biotech)
    return {"ticker": ticker, "company": company, "sector": sector, "industry": industry,
            "spot": spot, "is_biotech": is_biotech, "trials": trials, "sections": sections,
            "clinical_fields": clin_fields, "unknowns": unknowns(), "scorecard": score,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}